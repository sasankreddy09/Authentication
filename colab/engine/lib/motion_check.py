"""Deterministic spatial / temporal checks of a generated clip against its scene-state contract.

Inputs
  expected: {entity: [box or None per sampled frame]}     from layout_control.expected_tracks
  observed: {entity: [(box, score) or None per frame]}    from a tracker (trackers.py)
  camera:   per-frame cumulative camera shift (dx, dy, reliability) from estimate_camera_motion()
Every check returns a CheckResult: PASS, FAIL or UNKNOWN, with evidence. UNKNOWN is never treated as PASS
(acceptance.py decides what happens).

Camera compensation: when the camera pans right, everything static drifts LEFT on screen. The world motion of a
subject is therefore its screen motion PLUS the camera's own motion (both in frame widths). A clip in which the
camera pans while the person stands still must not count as "the person walked".

Pure numpy; unit-tested on synthetic videos in tests/test_motion_check.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np

VERSION = 2                       # part of the validation cache key: bump when a rule or threshold changes
TH = {"region": 0.18,             # start/end position tolerance (frame widths)
      "travel_share": 0.4,        # observed travel must be ≥ 40 % of the planned travel, same direction
      "still": 0.08,              # a stationary subject may drift at most this much
      "jump": 0.22,               # one-sample jump larger than this = teleport
      "near": 0.10,               # "next to" distance for hand-offs
      "visible": 0.6,             # share of a window in which the subject must be detected
      "cam_reliable": 0.08,       # phase-correlation peak ratio below this = camera estimate unreliable
      "stop_speed": 0.012}        # per-sample speed (frame widths) that counts as "stopped"


@dataclass
class CheckResult:
    check: str
    status: str                   # "PASS" | "FAIL" | "UNKNOWN"
    entity: str | None = None
    reason: str = ""
    evidence: dict = field(default_factory=dict)
    critical: bool = True

    def to_dict(self):
        return _plain(asdict(self))


def _plain(x):
    """numpy numbers -> plain Python (sidecars are JSON)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def _c(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


# ------------------------------------------------------------------------------------------- camera motion
ANCHOR_GAP = 8                    # frames between camera-motion anchor frames


def estimate_camera_motion(frames: np.ndarray, masks: list[list[tuple]] | None = None, scale: int = 4,
                           margin: float = 0.03):
    """Cumulative global shift per frame (fractions of width/height) by phase correlation of grey frames.
    Moving subjects are blanked so a walking person doesn't look like a camera pan; for each compared pair the union
    of both frames' subject boxes (+ margin) is blanked in BOTH images. Callers pass detected AND planned subject
    boxes: when the detector misses the subject and nothing is blanked, a walking person alone measures as a ~0.3
    frame-width "camera move" on a static shot (tests/test_speed.py) — the false 'camera moves 0.25 frame widths'.
    Sub-pixel peaks + anchor frames make slow real pans measurable (whole-pixel peaks read them as 0).
    masks[i]: boxes (x0, y0, x1, y1, 0..1) to ignore in frame i. Returns (shift [n,2], reliability [n])."""
    n, h, w = frames.shape[:3]
    g = frames[:, ::scale, ::scale].astype(np.float32).mean(axis=3)
    hh, ww = g.shape[1:]
    win = np.outer(np.hanning(hh), np.hanning(ww)).astype(np.float32)
    shift = np.zeros((n, 2), np.float32)
    rel = np.ones(n, np.float32)

    def blank(img, boxes):
        img = img.copy()
        if boxes:
            keep = np.ones(img.shape, bool)
            for b in boxes:
                x0, y0, x1, y1 = [int(round(v * s)) for v, s in
                                  zip((b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin), (ww, hh, ww, hh))]
                keep[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = False
            fill = img[keep].mean() if keep.any() else img.mean()
            img[~keep] = fill
        return np.fft.fft2((img - img.mean()) * win)

    def peak(r):
        """Correlation peak with sub-pixel precision (parabola through the peak and its two neighbours per axis):
        on a frame shrunk 8x a slow pan moves ~0.2 px per frame, which a whole-pixel peak rounds to 0 every time."""
        py, px = np.unravel_index(np.argmax(r), r.shape)
        def sub(a, b, c):
            den = a - 2 * b + c
            return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        fy = py + sub(r[(py - 1) % hh, px], r[py, px], r[(py + 1) % hh, px])
        fx = px + sub(r[py, (px - 1) % ww], r[py, px], r[py, (px + 1) % ww])
        return (fx - ww if fx > ww / 2 else fx), (fy - hh if fy > hh / 2 else fy)

    anchor = 0
    for i in range(1, n):
        # compare with an anchor frame up to ANCHOR_GAP frames back (not just the previous frame): a slow pan adds up
        # to a measurable shift before it is read, so rounding errors don't accumulate frame after frame
        boxes = []
        if masks:
            for j in (anchor, i):
                if j < len(masks):
                    boxes += list(masks[j] or [])
        A, B = blank(g[anchor], boxes), blank(g[i], boxes)
        R = B * np.conj(A)
        R /= np.abs(R) + 1e-6
        r = np.abs(np.fft.ifft2(R))
        dx, dy = peak(r)
        # content moved by (dx, dy) pixels: the camera moved the opposite way
        shift[i] = shift[anchor] + np.array([-dx / ww, -dy / hh], np.float32)
        rel[i] = float(r.max() / (r.sum() + 1e-6) * r.size / 50.0)
        if i - anchor >= ANCHOR_GAP:
            anchor = i
    return shift, rel


# ------------------------------------------------------------------------------------------- helpers
def _window(seq, a, b):
    n = len(seq)
    return seq[int(a * (n - 1)): max(int(b * (n - 1)) + 1, int(a * (n - 1)) + 1)]


def _centers(obs):
    return [(_c(o[0]) if o else None) for o in obs]


def _visible_share(obs):
    return sum(o is not None for o in obs) / max(len(obs), 1)


def _mean_x(seq):
    xs = [c[0] for c in seq if c]
    return float(np.mean(xs)) if xs else None


# ------------------------------------------------------------------------------------------- checks
def check_presence(name, exp, obs, critical=True, small=False):
    want = [i for i, e in enumerate(exp) if e is not None]
    if not want:
        return CheckResult("presence", "PASS", name, "not expected on screen", critical=critical)
    share = sum(obs[i] is not None for i in want) / len(want)
    ev = {"visible_share": round(share, 2), "frames_expected": len(want)}
    if share >= TH["visible"]:
        return CheckResult("presence", "PASS", name, f"visible in {share:.0%} of the frames", ev, critical)
    if small:                                       # small objects are often missed by detectors: not proof of absence
        return CheckResult("presence", "UNKNOWN", name, f"small object detected in only {share:.0%} of frames", ev, critical)
    return CheckResult("presence", "FAIL", name, f"only visible in {share:.0%} of the frames it should be in", ev, critical)


def check_region(name, which, exp, obs, critical=True):
    """Start or end position: the subject's centre is where the plan puts it (±TH['region'])."""
    a, b = (0.0, 0.15) if which == "start" else (0.85, 1.0)
    ce = _mean_x([_c(x) if x else None for x in _window(exp, a, b)])
    co = _mean_x(_window(_centers(obs), a, b))
    if ce is None:
        return CheckResult(f"{which}_position", "PASS", name, "not on screen then", critical=critical)
    if co is None:
        return CheckResult(f"{which}_position", "UNKNOWN", name, f"not detected at the {which}", critical=critical)
    ev = {"expected_x": round(ce, 2), "observed_x": round(co, 2)}
    ok = abs(ce - co) <= TH["region"]
    side = lambda x: "left" if x < 0.4 else "right" if x > 0.6 else "centre"
    return CheckResult(f"{which}_position", "PASS" if ok else "FAIL", name,
                       f"{which}s on the {side(co)} (planned: {side(ce)})", ev, critical)


def check_direction(name, exp, obs, cam_shift, cam_rel, critical=True):
    """Travel direction and amount, after removing camera motion."""
    e0, e1 = next((x for x in exp if x), None), next((x for x in reversed(exp) if x), None)
    if e0 is None:
        return CheckResult("direction", "PASS", name, "not on screen", critical=critical)
    planned = _c(e1)[0] - _c(e0)[0]
    cs = _centers(obs)
    idx = [i for i, c in enumerate(cs) if c is not None]
    if len(idx) < 2:
        return CheckResult("direction", "UNKNOWN", name, "not tracked in enough frames", critical=critical)
    i0, i1 = idx[0], idx[-1]
    screen = cs[i1][0] - cs[i0][0]
    cam = float(cam_shift[i1][0] - cam_shift[i0][0]) if cam_shift is not None else 0.0
    world = screen + cam
    reliable = cam_rel is None or float(np.median(cam_rel[i0 + 1:i1 + 1])) >= TH["cam_reliable"]
    ev = {"planned_dx": round(planned, 3), "screen_dx": round(screen, 3), "camera_dx": round(cam, 3),
          "world_dx": round(world, 3), "camera_estimate_reliable": reliable}
    if not reliable and abs(cam) > 0.05:
        return CheckResult("direction", "UNKNOWN", name, "camera motion can't be measured reliably", ev, critical)
    if abs(planned) < 0.05:                          # should stay put
        ok = abs(world) <= TH["still"]
        return CheckResult("stationary", "PASS" if ok else "FAIL", name,
                           "stays in place" if ok else f"drifts {world:+.2f} frame widths (should stay still)", ev, critical)
    if np.sign(world) != np.sign(planned) and abs(world) > TH["still"]:
        return CheckResult("direction", "FAIL", name, f"moves {'right' if world > 0 else 'left'} but should move "
                           f"{'right' if planned > 0 else 'left'}", ev, critical)
    if abs(world) < TH["travel_share"] * abs(planned):
        return CheckResult("direction", "FAIL", name, f"travels only {abs(world):.2f} of the planned "
                           f"{abs(planned):.2f} frame widths", ev, critical)
    return CheckResult("direction", "PASS", name, f"moves {'right' if world > 0 else 'left'} as planned", ev, critical)


def check_teleport(name, obs, critical=True):
    cs = [c for c in _centers(obs)]
    jumps = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(cs, cs[1:]) if a and b]
    big = max(jumps) if jumps else 0.0
    ok = big <= TH["jump"]
    return CheckResult("teleport", "PASS" if ok else "FAIL", name,
                       "moves continuously" if ok else f"jumps {big:.2f} frame widths between two frames",
                       {"max_jump": round(big, 3)}, critical)


def check_stop_before(actor, obs, t_action, critical=True):
    """The actor has (nearly) stopped just before a hand action starts."""
    cs = _centers(obs)
    n = len(cs)
    i = int(t_action * (n - 1))
    seg = [c for c in cs[max(i - 2, 0): i + 1] if c]
    if len(seg) < 2:
        return CheckResult("stop_before_action", "UNKNOWN", actor, "not tracked around the action", critical=critical)
    speed = float(np.mean([abs(b[0] - a[0]) for a, b in zip(seg, seg[1:])]))
    ok = speed <= TH["stop_speed"] * 2
    return CheckResult("stop_before_action", "PASS" if ok else "FAIL", actor,
                       "stops before acting" if ok else "is still moving when the action starts",
                       {"speed": round(speed, 4)}, critical)


def _box_dist(c, box):
    """Distance from a point to a box (0 inside)."""
    dx = max(box[0] - c[0], 0, c[0] - box[2]); dy = max(box[1] - c[1], 0, c[1] - box[3])
    return float(np.hypot(dx, dy))


def check_transfer(obj, obs_obj, obs_from, exp_obj, critical=True):
    """Object placed / handed over: it starts at the giver and ENDS WHERE THE PLAN PUTS IT, away from the giver's
    hand. (Being merely 'near' the target is not enough: a person standing at the counter with the case still in
    her hand is near the counter too.)"""
    co = _centers(obs_obj)
    n = len(co)
    k = max(n // 5, 1)
    first = [c for c in co[:k] if c]
    last = [c for c in co[-k:] if c]
    if not first or not last:
        return CheckResult("object_transfer", "UNKNOWN", obj, "object not tracked at the start and the end",
                           critical=critical)
    giver = [b[0] for b in obs_from[:k] if b]
    plan_end = next((b for b in reversed(exp_obj) if b), None)
    if not giver or plan_end is None:
        return CheckResult("object_transfer", "UNKNOWN", obj, "giver not tracked or no planned end position",
                           critical=critical)
    s0 = np.mean(first, axis=0); s1 = np.mean(last, axis=0)
    d_start = min(_box_dist(s0, g) for g in giver)
    d_end = float(np.hypot(s1[0] - _c(plan_end)[0], s1[1] - _c(plan_end)[1]))
    giver_end = [b[0] for b in obs_from[-k:] if b]
    left_hand = not giver_end or min(_box_dist(s1, g) for g in giver_end) > 0.01 or d_end <= TH["near"] / 2
    ev = {"start_distance_to_giver": round(d_start, 3), "end_distance_to_planned_spot": round(d_end, 3),
          "left_the_hand": bool(left_hand)}
    ok = d_start <= TH["near"] and d_end <= TH["near"] and left_hand
    return CheckResult("object_transfer", "PASS" if ok else "FAIL", obj,
                       "goes from the giver to the planned spot" if ok else
                       ("never leaves the giver" if not left_hand or d_end > 3 * TH["near"] else "ends in the wrong place"),
                       ev, critical)


def check_object_support(obj, obs_obj, obs_people: dict, critical=True):
    """Objects don't move by themselves: whenever the object moves, some person must be within reach of it."""
    co = _centers(obs_obj)
    bad, moving = 0, 0
    for i in range(1, len(co)):
        if not co[i] or not co[i - 1]:
            continue
        if abs(co[i][0] - co[i - 1][0]) + abs(co[i][1] - co[i - 1][1]) <= 0.015:
            continue
        moving += 1
        near_someone = any(seq[i] and _box_dist(co[i], seq[i][0]) <= TH["near"] for seq in obs_people.values()
                           if i < len(seq))
        bad += not near_someone
    ok = bad <= 1
    return CheckResult("object_support", "PASS" if ok else "FAIL", obj,
                       "only moves when someone carries it" if ok else
                       f"moves on its own in {bad} of {moving} moving frames (nobody holding it)",
                       {"unsupported_moving_frames": bad, "moving_frames": moving}, critical)


def check_order(times: dict[str, float | None], order: list[str], critical=True):
    """Events observed in the planned order (times = when each event was seen to complete)."""
    seen = [(k, times.get(k)) for k in order]
    if any(t is None for _, t in seen):
        return CheckResult("event_order", "UNKNOWN", None, "not every event could be timed",
                           {"times": {k: t for k, t in seen}}, critical)
    ok = all(a[1] <= b[1] + 1e-6 for a, b in zip(seen, seen[1:]))
    return CheckResult("event_order", "PASS" if ok else "FAIL", None,
                       "events in the planned order" if ok else "events happen in the wrong order",
                       {"times": {k: round(t, 3) for k, t in seen}}, critical)


def arrival_time(obs, target_x, tol=0.08):
    """Fraction of the clip at which a subject first reaches target_x (None if never)."""
    cs = _centers(obs)
    for i, c in enumerate(cs):
        if c and abs(c[0] - target_x) <= tol:
            return i / max(len(cs) - 1, 1)
    return None


def check_completes_before_cut(name, t_done, clip_seconds, used_seconds, stretch, critical=True):
    """Editing keeps the clip up to `used_seconds` of shot time; the action must be complete before that."""
    if t_done is None:
        return CheckResult("completes_before_cut", "UNKNOWN", name, "completion time unknown", critical=critical)
    done_at = t_done * clip_seconds * stretch
    ok = done_at <= used_seconds - 0.2
    return CheckResult("completes_before_cut", "PASS" if ok else "FAIL", name,
                       f"done at {done_at:.1f}s of {used_seconds:.1f}s" + ("" if ok else " (would be cut off)"),
                       {"done_at_s": round(done_at, 2), "kept_s": round(used_seconds, 2)}, critical)


def validate_clip(state, expected, observed, cam_shift, cam_rel, entities, clip_seconds=5.0, used_seconds=5.0,
                  stretch=1.0):
    """All deterministic checks for one clip. Returns a list of CheckResult."""
    from scene_state import LOCOMOTION, MANIPULATION
    res = []
    small = {k for k, e in entities.items() if getattr(e, "small", False)}
    anchors = {k for k, e in entities.items() if getattr(e, "kind", "") == "anchor"}
    for name, exp in expected.items():
        if name in anchors:
            continue
        obs = observed.get(name) or [None] * len(exp)
        res.append(check_presence(name, exp, obs, small=name in small))
        if name in small:
            continue                                         # positions of tiny objects: left to the VLM / review
        res.append(check_region(name, "start", exp, obs))
        res.append(check_region(name, "end", exp, obs))
        res.append(check_direction(name, exp, obs, cam_shift, cam_rel))
        res.append(check_teleport(name, obs))
        if getattr(entities.get(name), "kind", "") == "prop":
            people = {k: observed.get(k) or [] for k, e in entities.items() if getattr(e, "kind", "") == "character"}
            res.append(check_object_support(name, obs, people))
    times, order = {}, []
    for e in state.events:
        if e.type in LOCOMOTION and e.to_box:
            t = arrival_time(observed.get(e.actor) or [], (e.to_box[0] + e.to_box[2]) / 2)
            times[e.id] = t; order.append(e.id)
        if e.type in ("place", "hand_over", "pick_up") and e.critical:
            obs_a = observed.get(e.actor) or []
            res.append(check_stop_before(e.actor, obs_a, e.t[0]))
            if e.obj and e.obj not in small:
                res.append(check_transfer(e.obj, observed.get(e.obj) or [], obs_a, expected.get(e.obj) or []))
            times[e.id] = None if (e.obj in small or not observed.get(e.obj)) else e.t[1]
            order.append(e.id)
    if len(order) >= 2:
        res.append(check_order(times, order))
    for e in state.events:
        if e.critical and e.id in times:
            res.append(check_completes_before_cut(e.id, times[e.id], clip_seconds, used_seconds, stretch))
    if cam_shift is not None and state.camera.move == "static":
        drift = float(np.abs(cam_shift[-1]).max())
        res.append(CheckResult("camera_static", "PASS" if drift <= 0.06 else "FAIL", None,
                               "camera stays still" if drift <= 0.06 else f"camera moves {drift:.2f} frame widths "
                               "(plan: static) — subjects may only seem to move", {"camera_drift": round(drift, 3)}))
    return res


def run_checks(fn, *args, **kw):
    """Run a validator; an exception becomes an explicit UNKNOWN result (never a silent pass, never a crash)."""
    try:
        return fn(*args, **kw)
    except Exception as e:                                   # noqa: BLE001 - report every failure as evidence
        return [CheckResult("validator_error", "UNKNOWN", None, f"{type(e).__name__}: {e}"[:300])]

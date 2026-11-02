"""From a shot's scene-state to (1) the expected track of every entity and (2) a layout control video.

The SAME expected tracks steer generation (VACE layout control: white background, one coloured 2-px box per
subject, the format of VACE's own `layout_bbox` annotator) and are the contract the generated clip is checked
against (motion_check.py). Objects held by a character ride along with the holder's hand; a "place" event moves
the object from the hand to its target box during the event.

Pure numpy + Pillow; unit-tested in tests/test_scene_state.py.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from scene_state import ShotState, Box, LOCOMOTION, cx

PALETTE = [(220, 30, 30), (30, 90, 220), (20, 160, 60), (230, 140, 0), (150, 40, 190), (0, 170, 170),
           (120, 120, 120), (200, 0, 120)]


def _lerp(a: Box, b: Box, u: float) -> Box:
    u = min(max(u, 0.0), 1.0)
    u = u * u * (3 - 2 * u)                                        # ease in/out (people accelerate and stop)
    return tuple(a[i] + (b[i] - a[i]) * u for i in range(4))


def hand_box(owner: Box, facing: str | None, size=0.06) -> Box:
    """Where a held object sits: at hand height, on the side the person faces."""
    hx = owner[2] - size * 0.2 if facing == "right" else owner[0] + size * 0.2 if facing == "left" else cx(owner)
    hy = owner[1] + (owner[3] - owner[1]) * 0.55
    return (hx - size / 2, hy - size / 2, hx + size / 2, hy + size / 2)


def expected_tracks(state: ShotState, n: int) -> dict[str, list[Box | None]]:
    """Box of every entity at each of n frames (None = not on screen)."""
    out = {k: [] for k in state.start}
    persons = {e.actor for e in state.events} | {k for k, v in state.start.items() if v.facing}
    for e in state.events:
        out.setdefault(e.actor, [])
        if e.obj:
            out.setdefault(e.obj, [])
    for i in range(n):
        t = i / max(n - 1, 1)
        boxes = {k: v.box for k, v in state.start.items()}
        facing = {k: v.facing for k, v in state.start.items()}
        holders = dict(state.holders)
        for e in state.events:
            if t < e.t[0]:
                break
            u = (t - e.t[0]) / (e.t[1] - e.t[0])
            if e.type in LOCOMOTION and e.to_box is not None and e.actor in boxes:
                if cx(e.to_box) != cx(boxes[e.actor]):
                    facing[e.actor] = "right" if cx(e.to_box) > cx(boxes[e.actor]) else "left"
                boxes[e.actor] = _lerp(boxes[e.actor], e.to_box, u)
            elif e.type in ("take_out", "pick_up") and e.obj:
                if u >= 1:
                    holders[e.obj] = e.actor
            elif e.type in ("place", "hand_over") and e.obj and e.actor in boxes:
                start = hand_box(boxes[e.actor], facing.get(e.actor))
                if e.type == "place":
                    end = e.to_box or (boxes.get(e.target) and _on_top(boxes[e.target])) or start
                else:
                    end = hand_box(boxes[e.target], facing.get(e.target)) if e.target in boxes else start
                if u >= 1:
                    holders[e.obj] = e.target
                    boxes[e.obj] = end
                else:
                    holders[e.obj] = "__moving__"
                    boxes[e.obj] = _lerp(start, end, u)
        for obj, h in holders.items():                             # held objects follow the holder's hand
            if obj not in out or h == "__moving__" or h not in boxes:
                continue
            if h in persons:
                boxes[obj] = hand_box(boxes[h], facing.get(h))
            elif obj not in boxes:                                 # resting on a counter/table
                boxes[obj] = _on_top(boxes[h])
        for k in out:
            out[k].append(boxes.get(k))
    return out


def _on_top(anchor: Box, size=0.06) -> Box:
    x = cx(anchor)
    return (x - size / 2, anchor[1] - size, x + size / 2, anchor[1])


def control_video(tracks: dict[str, list[Box | None]], w: int, h: int, colors: dict[str, tuple] | None = None,
                  skip: set[str] = frozenset()):
    """VACE layout control frames (PIL images): white background, one coloured 2-px box outline per entity."""
    names = [k for k in tracks if k not in skip]
    # colour by position in the FULL track list, so a subject keeps its colour when others are left out
    colors = colors or {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(tracks)}
    n = len(next(iter(tracks.values()))) if tracks else 0
    frames = []
    for i in range(n):
        im = Image.new("RGB", (w, h), (255, 255, 255))
        d = ImageDraw.Draw(im)
        for k in names:
            b = tracks[k][i]
            if b is None:
                continue
            x0, y0, x1, y1 = [int(round(v * s)) for v, s in zip(b, (w, h, w, h))]
            d.rectangle([x0, y0, x1, y1], outline=colors[k], width=2)
        frames.append(im)
    return frames


def composition_guide(state: ShotState, tracks, w: int, h: int, names: dict[str, str]):
    """Human-readable storyboard frame: first and last positions with arrows (saved next to the review images)."""
    im = Image.new("RGB", (w, h), (245, 246, 250))
    d = ImageDraw.Draw(im)
    for i, (k, seq) in enumerate(tracks.items()):
        col = PALETTE[i % len(PALETTE)]
        first = next((b for b in seq if b), None)
        last = next((b for b in reversed(seq) if b), None)
        if not first:
            continue
        p0 = [int(v * s) for v, s in zip(first, (w, h, w, h))]
        d.rectangle(p0, outline=col, width=2)
        d.text((p0[0] + 4, p0[1] + 4), names.get(k, k), fill=col)
        if last and last != first:
            p1 = [int(v * s) for v, s in zip(last, (w, h, w, h))]
            d.rectangle(p1, outline=col, width=4)
            d.line([((p0[0] + p0[2]) // 2, (p0[1] + p0[3]) // 2), ((p1[0] + p1[2]) // 2, (p1[1] + p1[3]) // 2)],
                   fill=col, width=3)
    d.text((8, h - 18), f"{state.shot_id}  camera {state.camera.setup} ({state.camera.side}, {state.camera.move})",
           fill=(40, 40, 40))
    return im

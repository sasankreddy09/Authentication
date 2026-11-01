"""Storybook scene renderer: rigged characters inside a layered environment pack (drawkit kind "character").

Same spec as charanim (world, shot, subject, chars, changes, screen, label, push, _speech, _t0, _dur, _n) plus
optional storybook fields:
  "environment_variant": a camera preset of the world's pack (else chosen from world + shot + subject)
  "camera": {"movement": "static|slow_push_in|slow_pull_back|pan_left|pan_right"}  (default: static; push → push-in)
  "_visemes": {char: [viseme per frame]}  (mouth shapes from that speaker's audio; else from "_speech" loudness)
Selected by plan["visual_style_version"] == "storybook_v1"; every other plan keeps the legacy charanim drawing.
The plan pins the asset versions (plan["character_assets"], plan["environments"]); a missing or changed asset is
an error — never a silent fallback to simple shapes.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
from PIL import Image

import envfactory
import storybook as SB

STYLE = "storybook_v1"
_PAINTERS = OrderedDict()
_BG = OrderedDict()


def assets(plan):
    root = SB.library_root(plan)
    if plan.get("cinematic_characters"):
        import cinematic_workflow as CW
        CW.validate_plan(plan, root)
    chars = {}
    for cid, a in (plan.get("character_assets") or {}).items():
        strict = a.get("status") == "approved"           # approved art: byte-pinned; provisional may be redrawn
        chars[cid] = SB.load(cid, a["version"], root, expect_sha=a.get("manifest_sha256") if strict else None,
                             expect_content=a.get("content_sha256") if strict else None)
    return root, chars


def env_for(plan, world):
    env = envfactory.WORLD_ENV.get(world)
    if env is None:
        raise ValueError(f"no environment pack for world '{world}' (known: {sorted(envfactory.WORLD_ENV)})")
    ver = (plan.get("environments") or {}).get(env)
    if ver is None:
        raise ValueError(f"plan has no version for environment '{env}' (plan['environments'])")
    return envfactory.load(env, ver, SB.library_root(plan))


def preset_name(E, spec):
    p = spec.get("environment_variant") or E.preset_for(spec.get("world", "studio"), spec.get("shot", "two_shot"),
                                                         spec.get("subject"))
    if p is None or p not in E.man["presets"]:
        raise ValueError(f"environment {E.id}: no camera preset for world={spec.get('world')} shot={spec.get('shot')} "
                         f"subject={spec.get('subject')}")
    return p


def _painter(ch, s):
    key = (str(ch.dir.resolve()), ch.id, ch.version, ch.man.get("content_sha256"), round(s, 2))
    if key not in _PAINTERS:
        _PAINTERS[key] = SB.Painter(ch, round(s, 2))
        if len(_PAINTERS) > 12:
            _PAINTERS.popitem(last=False)
    _PAINTERS.move_to_end(key)
    return _PAINTERS[key]


def _layer_view(E, name, box, out_size):
    """Crop + scale one master layer to the output (cached for repeated identical boxes)."""
    key = (E.id, E.version, name, tuple(round(v, 1) for v in box), out_size)
    if key in _BG:
        _BG.move_to_end(key)
        return _BG[key]
    x, y, w, h = box
    im = E.img(name).resize(out_size, Image.BILINEAR, box=(x, y, x + w, y + h))
    _BG[key] = im
    if len(_BG) > 60:
        _BG.popitem(last=False)
    return im


def _depth_box(base, cam, depth):
    """Parallax: a layer at `depth` moves/zooms `depth` times as much as the camera (1 = with the characters)."""
    bx, by, bw, bh = base
    cx, cy, cw, ch_ = cam
    bcx, bcy = bx + bw / 2, by + bh / 2
    ccx, ccy = cx + cw / 2, cy + ch_ / 2
    zoom = bw / cw
    z = 1 + (zoom - 1) * depth
    w, h = bw / z, bh / z
    mx, my = bcx + (ccx - bcx) * depth, bcy + (ccy - bcy) * depth
    return (mx - w / 2, my - h / 2, w, h)


# ------------------------------------------------------------------------------------------- per-character state
def timeline(spec, cid):
    """[(t, gesture, expr)] for one character from chars + changes (in order)."""
    st = dict(spec.get("chars", {}).get(cid, {}))
    out = [(0.0, st.get("gesture", "idle"), st.get("expr"))]
    for c in sorted(spec.get("changes", []), key=lambda c: c["t"]):
        v = c.get("chars", {}).get(cid)
        if v:
            g = v.get("gesture", out[-1][1])
            e = v.get("expr", out[-1][2])
            out.append((c["t"], g, e))
    return out


def state_at(spec, cid, t):
    """Current static state (wear/hold/expr/gesture) and the previous gesture with the change time."""
    st = dict(spec.get("chars", {}).get(cid, {}))
    since, prev_g = 0.0, st.get("gesture", "idle")
    for c in sorted(spec.get("changes", []), key=lambda c: c["t"]):
        if c["t"] <= t and cid in c.get("chars", {}):
            v = c["chars"][cid]
            if "gesture" in v and v["gesture"] != st.get("gesture", "idle"):
                prev_g, since = st.get("gesture", "idle"), c["t"]
            st.update(v)
    return st, prev_g, since


def resolve_gesture(g, st, other_dx):
    g = SB.GESTURE_ALIAS.get(g, g) or "idle"
    if g in ("idle", "listen", "nod", "explain", "open_hands") and st.get("hold") and g != "explain":
        g = "hold"
    pose = dict(SB.pose_of(g))
    if g == "point" and other_dx < 0:                  # point toward the other person / the target on the left
        pose = {"L": dict(SB.POSES["point"]["R"]), "R": dict(SB.POSES["idle"]["R"]), "lean": -2}
    return g, pose


def mouth_for(expr, speaking, vis):
    base = SB.EXPRESSIONS.get(expr or "neutral", SB.EXPRESSIONS["neutral"])[2]
    if not speaking:
        return base
    if vis in ("rest",):
        return "closed" if base not in ("smile", "laugh") else "smile"
    if expr in SB.SMILING and vis in ("small", "wide"):
        return "smile_talk"
    return vis


# ------------------------------------------------------------------------------------------- the frame
def character_frame(spec, plan, t, w, h, bg=None, th=None):
    import charanim                                     # shared helpers: movie on the screen, label tag
    dur = float(spec.get("_dur", 4.0))
    n = max(int(spec.get("_n", 1)), 1)
    fi = min(int(round(t * (n - 1))), n - 1)
    t_abs = float(spec.get("_t0", 0.0)) + t * dur
    root, chars = assets(plan)
    E = env_for(plan, spec.get("world", "studio"))
    pname = preset_name(E, spec)
    P = E.man["presets"][pname]
    S = E.man["sets"][P["set"]]
    move = (spec.get("camera") or {}).get("movement") or ("slow_push_in" if spec.get("push") else "static")
    base = P["crop"]
    cam = envfactory.camera(P, t, move)
    view = P.get("view", "front")
    screen = spec.get("screen", "plain")
    for c in sorted(spec.get("changes", []), key=lambda c: c["t"]):   # the screen can change during the shot
        if c["t"] <= t and c.get("screen"):
            screen = c["screen"]
    ids = [c for c in spec.get("chars", {}) if c in chars] or list(chars)
    layers = S["layers"]
    k_char = [i for i, (n_, _) in enumerate(layers) if n_ == "@characters"][0]
    near = [i for i, (_, d) in enumerate(layers) if i > k_char and d > 1.0]   # foreground: own parallax
    # 1. the static plate behind the characters (cached per preset + screen + size)
    out = _plate(E, pname, P, S, layers[:k_char], screen, w, h).copy()
    # 2. characters, then the layers between them and the foreground (light, haze) — at the base framing
    _draw_characters(out, spec, plan, E, P, S, tuple(base), chars, ids, view, fi, t, t_abs, w, h)
    for i in range(k_char + 1, len(layers)):
        if i not in near:
            out.alpha_composite(_layer_view(E, layers[i][0], tuple(base), (w, h)))
    # 3. the camera move = one crop of the composed frame; the foreground moves faster (parallax)
    if move != "static":
        bx, by, bw, bh = base
        cx, cy, cw, chh = cam
        kx, ky = w / bw, h / bh
        out = out.resize((w, h), Image.BILINEAR, box=((cx - bx) * kx, (cy - by) * ky, (cx - bx + cw) * kx, (cy - by + chh) * ky))
    for i in near:
        out.alpha_composite(_layer_view(E, layers[i][0], _depth_box(base, cam, layers[i][1]), (w, h)))
    if spec.get("label"):
        charanim._tag(out, spec["label"]) if w == 1920 else None
    return np.asarray(out.convert("RGB"))


_PLATES = OrderedDict()


def _plate(E, pname, P, S, layers, screen, w, h):
    import charanim
    key = (E.id, E.version, pname, screen, w, h)
    if key in _PLATES:
        _PLATES.move_to_end(key)
        return _PLATES[key]
    base = tuple(P["crop"])
    out = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    for lname, depth in layers:
        if lname == "@screen":
            sx0, sy0, sx1, sy1 = S["screen_rect"]
            kx, ky = w / base[2], h / base[3]
            px0, py0 = int((sx0 - base[0]) * kx), int((sy0 - base[1]) * ky)
            pw, ph = int((sx1 - sx0) * kx), int((sy1 - sy0) * ky)
            out.alpha_composite(charanim.movie_frame(max(pw, 2), max(ph, 2), screen).convert("RGBA"), (px0, py0))
        else:
            out.alpha_composite(_layer_view(E, lname, base, (w, h)))
    _PLATES[key] = out
    if len(_PLATES) > 16:
        _PLATES.popitem(last=False)
    return out


def _draw_characters(out, spec, plan, E, P, S, box, chars, ids, view, fi, t, t_abs, w, h):
    kx = w / box[2]
    light = S.get("lighting", {})
    speech = spec.get("_speech", {})
    vis_track = spec.get("_visemes", {})
    slots = P["slots"]
    present = [c for c in ids if c in slots]
    for cid in present:
        ch = chars[cid]
        sx, sy, sc = slots[cid]
        others = [slots[o][0] for o in present if o != cid] or [slots[o][0] for o in slots if o != cid]
        other_dx = (others[0] - sx) if others else 0
        s = sc * kx                                     # output px per design unit
        if not (-600 * s < (sx - box[0]) * kx < w + 600 * s):
            continue                                    # outside the frame (e.g. the other friend in a close-up)
        st, prev_g, since = state_at(spec, cid, t)
        e = (speech.get(cid) or [0.0])[fi] if speech.get(cid) else 0.0
        vlist = vis_track.get(cid)
        vis = vlist[fi] if vlist and fi < len(vlist) else SB.viseme_from_energy(e, "rest")
        speaking = e > 0.06
        g, pose = resolve_gesture(st.get("gesture", "idle"), st, other_dx)
        age = (t - since) * float(spec.get("_dur", 4.0))
        if age < SB.BLEND_S and prev_g != st.get("gesture"):
            _, p0 = resolve_gesture(prev_g, st, other_dx)
            pose = SB.blend_pose(p0, pose, age / SB.BLEND_S)
        else:
            pose = SB.blend_pose(pose, pose, 1.0)
        if g == "wave":                                 # a real wave: the forearm swings
            pose["R"]["b"] += 14 * math.sin(2 * math.pi * 2.2 * t_abs)
        expr = st.get("expr") or ("listening" if not speaking else "neutral")
        if expr not in SB.EXPRESSIONS:
            expr = "neutral"
        eyes, brows, _, iris = SB.EXPRESSIONS[expr]
        if SB.blink_closed(cid, t_abs) and eyes not in ("closed", "happy"):
            eyes = "closed" if eyes != "wink" else "wink"
        tilt, breathe = SB.idle_motion(cid, t_abs, speaking)
        head_dy = breathe * 0.6
        if g == "nod" and age < 1.6:
            head_dy += 7 * abs(math.sin(math.pi * age / 0.4))
        turn = 0.0 if view == "back" else max(-0.45, min(0.45, 0.35 * (1 if other_dx > 0 else -1 if other_dx < 0 else 0)))
        gx = 9 if other_dx > 0 else -9 if other_dx < 0 else 0
        gy = -3 if P.get("seated") and not speaking else 0
        if expr == "thinking":
            gx, gy = 8 * (1 if other_dx >= 0 else -1), -12
        face = {"eyes": eyes, "brows": brows, "mouth": mouth_for(expr, speaking, vis), "gaze": (gx, gy),
                "turn": round(turn, 2), "glasses": "3d_glasses" in st.get("wear", []), "iris": iris,
                "head_tilt": tilt, "head_dy": head_dy}
        pose["head"] = pose.get("head", 0)
        # draw the character on its own canvas at a quantized scale, then light it and place it
        s0 = round(s, 2)
        cw_, chh = int(1000 * s0), int(1800 * s0)
        canvas = Image.new("RGBA", (cw_ + 2, chh + 2), (0, 0, 0, 0))
        hold = st.get("hold") if st.get("hold") in ("popcorn", "3d_glasses") else None
        if is_atlas(ch):                                # cinematic artwork: whole poses + face patches
            _draw_atlas(canvas, ch, s0, st, g, prev_g, age, view, turn, expr, vis, speaking, cid, t_abs, tilt, breathe)
        else:
            pnt = _painter(ch, s0)
            pnt.draw(canvas, (0, 0), pose, face, view="back" if view == "back" else "front", hold=hold,
                     legs=not P.get("seated"))
        tint = light.get("tint")
        if tint and light.get("tint_strength", 0) > 0:      # match the scene's light colour (C-speed, no numpy)
            from PIL import ImageChops
            mul = ImageChops.multiply(canvas, Image.new("RGBA", canvas.size, tuple(tint) + (255,)))
            a_ = canvas.getchannel("A")
            canvas = Image.blend(canvas, mul, light["tint_strength"])
            canvas.putalpha(a_)
        if abs(s / s0 - 1) > 0.002:
            canvas = canvas.resize((max(1, int(canvas.width * s / s0)), max(1, int(canvas.height * s / s0))), Image.BILINEAR)
        ox = (sx - box[0]) * kx - 500 * s
        oy = (sy - box[1]) * kx - 1800 * s + breathe * s * 0.4
        if not P.get("seated") and view != "back":      # contact shadow on the floor
            from PIL import ImageDraw, ImageFilter
            sh = Image.new("RGBA", (int(420 * s), int(80 * s)), (0, 0, 0, 0))
            ImageDraw.Draw(sh).ellipse([0, 0, sh.width, sh.height], fill=(0, 0, 0, 110))
            sh = sh.filter(ImageFilter.GaussianBlur(10 * s))
            out.alpha_composite(sh, (int(ox + 290 * s), int(oy + 1570 * s)))
        out.alpha_composite(canvas, (int(round(ox)), int(round(oy))))


def is_atlas(ch):
    return getattr(ch, "man", {}).get("asset_type") == "atlas_v1"


_ATLAS_P = OrderedDict()


def atlas_view(ch, g, view, turn, wear, hold):
    """Keep conversational shots front-facing: atlas_v1 has face patches only for the front view.
    Auto-selecting a three-quarter still disables mouth/blink animation. Back shots remain explicit.
    Alternate-view facial animation needs a reviewed per-view face atlas before it can be enabled.
    """
    return "back" if view == "back" else "front"


def _draw_atlas(canvas, ch, s0, st, g, prev_g, age, view, turn, expr, vis, speaking, cid, t_abs, tilt, breathe):
    import storybook_atlas as SA
    wear, hold = tuple(sorted(st.get("wear", []))), st.get("hold")
    g = SB.GESTURE_ALIAS.get(g, g)
    vw = atlas_view(ch, g, view, turn, wear, hold)
    pid = ch.atlas_pose(g, vw, wear, hold)
    if pid is None:
        raise ValueError(ch.missing(g, vw, wear, hold))
    prev = None
    if age < SB.BLEND_S and prev_g and prev_g != g:
        pg = SB.GESTURE_ALIAS.get(prev_g, prev_g)
        prev = ch.atlas_pose(pg, atlas_view(ch, pg, view, turn, wear, hold), wear, hold)
    key = (str(ch.dir.resolve()), ch.id, ch.version, ch.man.get("content_sha256"), s0)
    if key not in _ATLAS_P:
        _ATLAS_P[key] = SA.AtlasPainter(ch, s0)
        if len(_ATLAS_P) > 8:
            _ATLAS_P.popitem(last=False)
    face = {"expr": expr, "viseme": vis, "speaking": speaking, "wear": wear,
            "blink": SB.blink_closed(cid, t_abs) and expr not in ("laughing", "wink")}
    _ATLAS_P[key].draw(canvas, (0, 0), pid, face, pid_prev=prev, u=age / SB.BLEND_S if prev else 1.0,
                       motion=(tilt * 0.35, 0.004 * breathe / 2.2))


# ------------------------------------------------------------------------------------------- checks
def check_spec(spec, plan):
    """Clear errors for anything the storybook renderer can't draw faithfully."""
    out = []
    try:
        root, chars = assets(plan)
    except Exception as e:
        return [f"character assets: {e}"]
    try:
        E = env_for(plan, spec.get("world", "studio"))
        preset_name(E, spec)
    except Exception as e:
        out.append(str(e))
    for cid, st in list(spec.get("chars", {}).items()) + [(k, v) for c in spec.get("changes", []) for k, v in c.get("chars", {}).items()]:
        if cid not in chars:
            out.append(f"unknown character '{cid}' (assets: {sorted(chars)})")
            continue
        g = st.get("gesture", "idle")
        if SB.GESTURE_ALIAS.get(g, g) not in SB.POSES:
            out.append(f"{cid}: unsupported gesture '{g}' (rig poses: {sorted(SB.POSES)})")
        elif is_atlas(chars[cid]):                     # the atlas must hold this exact pose / view / wear / hold
            try:
                E_ = env_for(plan, spec.get("world", "studio"))
                back = E_.man["presets"][preset_name(E_, spec)].get("view") == "back"
            except Exception:
                back = False
            full = dict(spec.get("chars", {}).get(cid, {}), **st)
            m = chars[cid].missing(SB.GESTURE_ALIAS.get(g, g), "back" if back else "front",
                                   tuple(sorted(full.get("wear", []))), full.get("hold"))
            if m:
                out.append(m)
        if st.get("expr", "neutral") not in SB.EXPRESSIONS:
            out.append(f"{cid}: unknown expression '{st.get('expr')}'")
    mv = (spec.get("camera") or {}).get("movement")
    if mv and mv not in ("static", "slow_push_in", "slow_pull_back", "pan_left", "pan_right"):
        out.append(f"unknown camera movement '{mv}'")
    return out

"""Lightweight 2D character scenes on the CPU: the two friends of the Character Bible talking in a place.

No AI model and no GPU. The characters are original vector-style drawings made from the Bible's canonical look
(characters.py), so they look the same in every shot of every video. Per frame only small things change:
breathing, blinking, the speaking mouth (from the speaker's real audio level), a nod, an eyebrow, a gesture.

Speed: each body part is drawn once at 2x (smooth edges), shrunk, and cached by its state; a frame is a cached
background plus a few cached sprites pasted at offsets (~20–40 ms at 1080p on one CPU core).

Mouth: "energy" lip movement follows the loudness of the speaker's own voice track, so the mouth is open when sound
is heard and closed in pauses. It is not phoneme lip sync (Rhubarb would be; optional, not bundled). With no audio
level the mouth stays closed — never random flapping.

Spec (drawkit kind "character"), all times are fractions 0..1 of the visual:
  {"world": "living_room|cinema_seats|cinema_screen|lobby|studio",
   "shot": "two_shot|wide|closeup|reaction|ots|establishing", "subject": "<char id>" (closeup/reaction/ots),
   "chars": {"arjun": {"expr": "curious", "gesture": "point", "wear": ["3d_glasses"], "hold": "3d_glasses"}, ...},
   "screen": "double|clean|plain" (cinema worlds), "changes": [{"t": 0.5, "chars": {...}, "screen": "clean"}],
   "label": "Illustration" (optional corner tag), "push": true (slow push-in),
   "_speech": {"arjun": [0..1 per frame]}, "_t0": seconds (start of this visual in the video), "_dur": seconds}
"""
from __future__ import annotations

import hashlib, math
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import characters as charbible

SS = 2                                    # supersampling factor for smooth edges
W0, H0 = 700, 1000                        # local design box of one character (waist up), at scale 1
WORLDS = ("living_room", "cinema_seats", "cinema_screen", "lobby", "studio")
SHOTS = ("two_shot", "wide", "closeup", "reaction", "ots", "establishing")
WEARABLES = ("3d_glasses",)
HOLDABLES = ("3d_glasses", "popcorn")
_BIBLE = None
_cache: "OrderedDict[tuple, Image.Image]" = OrderedDict()
CACHE_MAX = 160


def bible():
    global _BIBLE
    if _BIBLE is None:
        _BIBLE = charbible.load()
    return _BIBLE


def set_bible(b):
    global _BIBLE
    _BIBLE = b
    _cache.clear()


def _cached(key, make):
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    im = make()
    _cache[key] = im
    if len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)
    return im


def rgb(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def shade(c, k):
    """k < 0 darker, k > 0 lighter."""
    c = rgb(c) if isinstance(c, str) else c
    return tuple(int(max(0, min(255, v + (255 - v) * k if k > 0 else v * (1 + k)))) for v in c)


def _canvas(w, h):
    im = Image.new("RGBA", (int(w * SS), int(h * SS)), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def _down(im, scale):
    w, h = im.size
    return im.resize((max(1, round(w / SS * scale)), max(1, round(h / SS * scale))), Image.LANCZOS)


# ------------------------------------------------------------------------------------------- body parts
def _P(*xy):
    return [v * SS for v in xy]


def torso_sprite(cid, scale):
    """Neck + shoulders + top (waist up), drawn once per character and scale."""
    def make():
        c = bible().characters[cid]
        L = c.look
        im, d = _canvas(W0, H0)
        skin, top, det = rgb(L.skin), rgb(L.top), rgb(L.top_detail)
        d.rounded_rectangle(_P(312, 440, 388, 600), radius=30 * SS, fill=shade(skin, -0.12))     # neck
        if L.hair_style == "short_messy":                                                    # hood behind the neck
            d.ellipse(_P(205, 520, 495, 660), fill=shade(top, -0.22))
        d.rounded_rectangle(_P(120, 560, 580, 1100), radius=150 * SS, fill=top)               # shoulders/torso
        d.rounded_rectangle(_P(120, 560, 580, 1100), radius=150 * SS, outline=shade(top, -0.25), width=4 * SS)
        if L.hair_style == "short_messy":                                                    # hoodie: collar + strings
            d.chord(_P(255, 520, 445, 640), 0, 180, fill=shade(top, -0.3))
            for x in (318, 382):
                d.line(_P(x, 600, x - 4, 760), fill=det, width=7 * SS)
                d.ellipse(_P(x - 13, 752, x + 5, 770), fill=det)
            d.rounded_rectangle(_P(250, 860, 450, 960), radius=30 * SS, outline=shade(top, -0.25), width=5 * SS)
        else:                                                                                # jacket over a T-shirt
            d.polygon(_P(285, 575, 415, 575, 350, 760), fill=det)
            d.line(_P(285, 575, 350, 770, 350, 1000), fill=shade(top, -0.3), width=6 * SS)
            d.line(_P(415, 575, 350, 770), fill=shade(top, -0.3), width=6 * SS)
            d.polygon(_P(270, 565, 300, 560, 345, 720, 320, 735), fill=shade(top, 0.12))   # lapels
            d.polygon(_P(430, 565, 400, 560, 355, 720, 380, 735), fill=shade(top, 0.12))
        return _down(im, scale)
    return _cached(("torso", cid, round(scale, 3)), make)


EXPR = {   # eyebrow lift (px), eyebrow tilt (deg, + = inner end up), mouth shape, eye openness, blush
    "neutral":   (0, 0, "flat", 1.0, False),
    "smile":     (2, 0, "smile", 0.95, False),
    "happy":     (6, 0, "grin", 0.85, True),
    "curious":   (10, 8, "small", 1.05, False),
    "surprised": (22, 4, "o", 1.2, False),
    "confused":  (6, -10, "wavy", 1.0, False),
    "thinking":  (4, -6, "side", 0.9, False),
    "listening": (3, 3, "soft", 1.0, False),
    "wink":      (4, 0, "side", 1.0, False),     # one eye closed (e.g. the one-eye test)
}


def head_sprite(cid, scale, expr, mouth, eyes_open, facing, wear):
    """Head with hair, face and worn props. mouth 0..3 (0 closed), facing -1 left / +1 right / 0 front."""
    key = ("head", cid, round(scale, 3), expr, mouth, eyes_open, facing, tuple(wear))

    def make():
        c = bible().characters[cid]
        L = c.look
        im, d = _canvas(W0, 560)
        skin, hair, eye = rgb(L.skin), rgb(L.hair), rgb(L.eyes)
        fx = 16 * facing                                       # features turn a little toward the other person
        lift, tilt, mshape, eo, blush = EXPR.get(expr, EXPR["neutral"])
        # hair behind the head
        if L.hair_style == "long_tied":
            d.ellipse(_P(470 - 60 * (facing > 0), 250, 600 - 60 * (facing > 0), 470), fill=shade(hair, -0.1))  # bun/tail
            d.ellipse(_P(185, 120, 515, 470), fill=hair)
        elif L.hair_style == "bob":
            d.rounded_rectangle(_P(175, 130, 525, 520), radius=140 * SS, fill=hair)
        # ears + head
        for ex in (200, 500):
            d.ellipse(_P(ex - 24, 300, ex + 24, 380), fill=shade(skin, -0.08))
        d.ellipse(_P(200, 150, 500, 505), fill=skin)
        # front hair
        if L.hair_style == "short_messy":
            pts = [(195, 300), (190, 210), (230, 150), (290, 118), (360, 108), (430, 122), (480, 160), (508, 230),
                   (505, 300), (470, 235), (450, 250), (420, 205), (385, 235), (350, 200), (315, 238), (280, 205),
                   (245, 250), (225, 230)]
            d.polygon(_P(*[v for p in pts for v in p]), fill=hair)
        elif L.hair_style in ("long_tied", "short_neat"):
            d.chord(_P(195, 120, 505, 420), 180, 360, fill=hair)
            d.polygon(_P(205, 270, 260, 175, 350, 150, 330, 215, 270, 250), fill=hair)       # side parting sweep
            d.polygon(_P(495, 270, 440, 175, 360, 150, 380, 205, 450, 240), fill=hair)
        # eyebrows
        for side, ex in ((-1, 290 + fx), (1, 410 + fx)):
            y = 288 - lift
            a = math.radians(tilt) * (-side)
            dx, dy = 34 * math.cos(a), 34 * math.sin(a)
            if expr == "confused" and side == 1:
                dy, y = -dy, y - 10
            d.line(_P(ex - dx, y + dy, ex + dx, y - dy), fill=shade(hair, -0.1), width=11 * SS)
        # eyes
        gaze = 7 * facing
        for ex in (290 + fx, 410 + fx):
            if eyes_open and not (expr == "wink" and ex == 290 + fx):
                ry = 30 * eo
                d.ellipse(_P(ex - 27, 345 - ry, ex + 27, 345 + ry), fill=(250, 250, 250))
                d.ellipse(_P(ex - 16 + gaze, 345 - 17, ex + 16 + gaze, 345 + 17), fill=eye)
                d.ellipse(_P(ex - 5 + gaze, 335, ex + 3 + gaze, 343), fill=(255, 255, 255))
                if expr == "happy":                          # smiling eyes: lower lid up
                    d.chord(_P(ex - 30, 345, ex + 30, 395), 180, 360, fill=skin)
            else:
                d.arc(_P(ex - 26, 330, ex + 26, 360), 20, 160, fill=shade(skin, -0.5), width=7 * SS)
        # glasses (part of Meera's identity) or 3D glasses (a prop)
        if "3d_glasses" in wear:                           # tinted, see-through lenses: the eyes stay readable
            lens = Image.new("RGBA", im.size, (0, 0, 0, 0)); ld = ImageDraw.Draw(lens)
            for x0 in (240, 360):
                ld.rounded_rectangle(_P(x0 + fx, 305, x0 + 100 + fx, 385), radius=14 * SS, fill=(40, 44, 52, 120))
            im.alpha_composite(lens)
            d = ImageDraw.Draw(im)
            d.rounded_rectangle(_P(240 + fx, 305, 340 + fx, 385), radius=14 * SS, outline=(15, 15, 18), width=8 * SS)
            d.rounded_rectangle(_P(360 + fx, 305, 460 + fx, 385), radius=14 * SS, outline=(15, 15, 18), width=8 * SS)
            d.line(_P(340 + fx, 330, 360 + fx, 330), fill=(15, 15, 18), width=8 * SS)
            d.line(_P(240 + fx, 330, 200, 320), fill=(15, 15, 18), width=7 * SS)
            d.line(_P(460 + fx, 330, 500, 320), fill=(15, 15, 18), width=7 * SS)
        elif L.glasses:
            for ex in (290 + fx, 410 + fx):
                d.ellipse(_P(ex - 44, 303, ex + 44, 387), outline=(60, 40, 30), width=6 * SS)
            d.line(_P(334 + fx, 335, 366 + fx, 335), fill=(60, 40, 30), width=6 * SS)
        # nose
        d.arc(_P(330 + fx * 1.4, 370, 370 + fx * 1.4, 420), 30, 150, fill=shade(skin, -0.28), width=6 * SS)
        if blush:
            for ex in (265 + fx, 435 + fx):
                d.ellipse(_P(ex - 26, 395, ex + 26, 420), fill=(230, 120, 120, 70))
        # mouth
        mx, my = 350 + fx * 1.2, 450
        lip, inside = shade(skin, -0.45), (120, 40, 50)
        if mouth > 0:
            wdt = 34 + (8 if mshape in ("smile", "grin") else 0)
            hgt = (0, 10, 20, 32)[mouth]
            d.ellipse(_P(mx - wdt, my - hgt / 2, mx + wdt, my + hgt / 2 + 6), fill=inside, outline=lip, width=4 * SS)
            if mouth >= 2:
                d.chord(_P(mx - wdt + 10, my + hgt / 2 - 6, mx + wdt - 10, my + hgt / 2 + 10), 180, 360,
                        fill=(205, 95, 105))
        elif mshape == "o":
            d.ellipse(_P(mx - 16, my - 16, mx + 16, my + 20), fill=inside, outline=lip, width=4 * SS)
        elif mshape == "grin":
            d.chord(_P(mx - 46, my - 22, mx + 46, my + 34), 0, 180, fill=inside, outline=lip, width=4 * SS)
            d.chord(_P(mx - 36, my - 18, mx + 36, my + 4), 0, 180, fill=(250, 250, 250))
        elif mshape in ("smile", "soft"):
            a = 40 if mshape == "smile" else 30
            d.arc(_P(mx - a, my - 30, mx + a, my + 16), 25, 155, fill=lip, width=7 * SS)
        elif mshape == "wavy":
            d.line(_P(mx - 30, my + 4, mx - 10, my - 4, mx + 10, my + 4, mx + 30, my - 4), fill=lip, width=6 * SS)
        elif mshape == "side":
            d.line(_P(mx - 24, my + 2, mx + 26, my - 6), fill=lip, width=7 * SS)
        elif mshape == "small":
            d.ellipse(_P(mx - 10, my - 6, mx + 10, my + 10), fill=inside, outline=lip, width=3 * SS)
        else:
            d.line(_P(mx - 28, my, mx + 28, my), fill=lip, width=7 * SS)
        return _down(im, scale)
    return _cached(key, make)


# arm poses: (shoulder, elbow, hand) in local coords for the arm on the "front" side (+1 = character's right of
# the image) — mirrored for facing; "rest" is the idle pose every gesture starts from
POSES = {
    "idle":       {"R": ((520, 640), (575, 860), (560, 1060)), "L": ((180, 640), (125, 860), (140, 1060))},
    "explain":    {"R": ((520, 640), (585, 860), (470, 760)), "L": ((180, 640), (125, 860), (140, 1060))},
    "point":      {"R": ((520, 640), (640, 650), (760, 600)), "L": ((180, 640), (125, 860), (140, 1060))},
    "think":      {"R": ((520, 640), (520, 820), (395, 520)), "L": ((180, 640), (230, 860), (420, 840))},
    "open_hands": {"R": ((520, 640), (600, 840), (600, 760)), "L": ((180, 640), (100, 840), (100, 760))},
    "nod":        {"R": ((520, 640), (575, 860), (560, 1060)), "L": ((180, 640), (125, 860), (140, 1060))},
    "show":       {"R": ((520, 640), (620, 800), (560, 600)), "L": ((180, 640), (125, 860), (140, 1060))},
    "duck":       {"R": ((520, 640), (640, 520), (470, 330)), "L": ((180, 640), (60, 520), (230, 330))},
}                  # "duck": both forearms up in front of the face, the body drops (character_frame)                  # "show": a held prop lifted next to the face (used automatically in close-ups)


def arms_sprite(cid, scale, gesture, phase, facing, hold):
    """Both arms for a gesture at phase 0..1 (0 = resting), mirrored so gestures point toward the other person."""
    q = round(phase * 8) / 8
    key = ("arms", cid, round(scale, 3), gesture, q, facing, hold)

    def make():
        c = bible().characters[cid]
        top, skin = rgb(c.look.top), rgb(c.look.skin)
        im, d = _canvas(W0 + 260, H0)                       # wider: a pointing hand reaches out of the body box
        ox = 130
        mirror = facing < 0
        for side in ("R", "L"):
            pts = []
            for a, b in zip(POSES["idle"][side], POSES.get(gesture, POSES["idle"])[side]):
                pts.append((a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q))
            if mirror:
                pts = [(W0 - x, y) for x, y in pts]
            pts = [(x + ox, y) for x, y in pts]
            (sx, sy), (ex, ey), (hx, hy) = pts
            col = shade(top, -0.08 if side == "L" else 0)
            d.line(_P(sx, sy, ex, ey), fill=col, width=86 * SS)
            d.line(_P(ex, ey, hx, hy), fill=col, width=76 * SS)
            for x, y, r in ((sx, sy, 43), (ex, ey, 40)):
                d.ellipse(_P(x - r, y - r, x + r, y + r), fill=col)
            d.ellipse(_P(hx - 36, hy - 36, hx + 36, hy + 36), fill=skin)
            if gesture == "point" and side == "R" and q > 0.5:     # index finger
                dirx = -1 if mirror else 1
                d.line(_P(hx, hy - 8, hx + 70 * dirx, hy - 20), fill=skin, width=20 * SS)
            if hold and side == "R" and q > 0.3 and gesture in ("explain", "open_hands", "point", "show"):
                _held(d, hold, hx, hy)
        return _down(im, scale)
    return _cached(key, make)


def _held(d, what, hx, hy):
    if what == "3d_glasses":
        y = hy - 70
        for x0 in (hx - 105, hx + 5):
            d.rounded_rectangle(_P(x0, y, x0 + 100, y + 70), radius=12 * SS, fill=(40, 44, 52), outline=(15, 15, 18),
                                width=8 * SS)
        d.line(_P(hx - 5, y + 25, hx + 5, y + 25), fill=(15, 15, 18), width=8 * SS)
    elif what == "popcorn":
        d.polygon(_P(hx - 60, hy - 150, hx + 60, hy - 150, hx + 45, hy + 10, hx - 45, hy + 10), fill=(230, 60, 60))
        for k in range(-2, 3):
            d.line(_P(hx + 22 * k, hy - 150, hx + 18 * k, hy + 10), fill=(250, 250, 250), width=10 * SS)
        for k in range(7):
            x = hx - 54 + 18 * k
            d.ellipse(_P(x - 18, hy - 185 - 12 * (k % 2), x + 18, hy - 150), fill=(255, 238, 180))


def back_sprite(cid, scale):
    """A character seen from behind (over-the-shoulder foreground, audience in the cinema)."""
    def make():
        c = bible().characters[cid]
        L = c.look
        im, d = _canvas(W0, H0)
        hair, top, skin = rgb(L.hair), rgb(L.top), rgb(L.skin)
        d.rounded_rectangle(_P(120, 560, 580, 1100), radius=150 * SS, fill=shade(top, -0.35))
        d.rounded_rectangle(_P(312, 440, 388, 600), radius=30 * SS, fill=shade(skin, -0.35))
        for ex in (200, 500):
            d.ellipse(_P(ex - 24, 300, ex + 24, 380), fill=shade(skin, -0.3))
        d.ellipse(_P(195, 140, 505, 500), fill=shade(hair, -0.05))
        if L.hair_style == "long_tied":
            d.ellipse(_P(300, 250, 400, 360), fill=shade(hair, 0.06))
        return _down(im, scale)
    return _cached(("back", cid, round(scale, 3)), make)


# ------------------------------------------------------------------------------------------- worlds
def _grad(w, h, top, bot):
    y = np.linspace(0, 1, h)[:, None, None]
    return (np.array(top, float) * (1 - y) + np.array(bot, float) * y) * np.ones((1, w, 1))


def movie_frame(w, h, mode="plain"):
    """What is on the cinema screen: a mountain valley with a bird and a ball flying toward the audience.
    "double": two slightly shifted copies (what you see WITHOUT the glasses); "clean": one sharp image with depth."""
    mode, _, prop = mode.partition("+")                 # "clean+popcorn": a popcorn bucket instead of the ball

    def one(shift=0):
        im = Image.fromarray(_grad(w, h, (110, 170, 230), (250, 205, 150)).astype(np.uint8)).convert("RGBA")
        d = ImageDraw.Draw(im)
        d.polygon([(0, h * .75), (w * .25, h * .35), (w * .45, h * .7), (w * .65, h * .3), (w, h * .72), (w, h), (0, h)],
                  fill=(70, 110, 120))
        d.polygon([(0, h * .9), (w * .3, h * .65), (w * .6, h * .85), (w, h * .7), (w, h), (0, h)], fill=(50, 120, 70))
        d.ellipse([w * .78, h * .1, w * .9, h * .1 + w * .12], fill=(255, 245, 200))
        cx = w * .38 + shift
        if prop == "popcorn":                                                               # bucket + flying kernels
            d.polygon([(cx - h * .15, h * .42), (cx + h * .15, h * .42), (cx + h * .11, h * .74), (cx - h * .11, h * .74)],
                      fill=(230, 60, 60))
            for k in range(-2, 3):
                d.line([(cx + h * .055 * k, h * .42), (cx + h * .04 * k, h * .74)], fill=(250, 250, 250), width=max(3, h // 50))
            for k, (u, v, r) in enumerate([(-.12, .36, .05), (-.04, .33, .055), (.05, .35, .05), (.12, .37, .045),
                                           (-.2, .22, .04), (.0, .16, .045), (.18, .2, .04), (.3, .1, .035)]):
                x, y = cx + h * u * (1.6 if v < .3 else 1) + shift * (0.5 if v < .3 else 0), h * v
                d.ellipse([x - h * r, y - h * r, x + h * r, y + h * r], fill=(255, 238, 180))
        else:
            d.ellipse([cx - h * .17, h * .38, cx + h * .17, h * .72], fill=(235, 70, 60))   # the ball (near: shifts most)
            d.ellipse([cx - h * .09, h * .43, cx - h * .02, h * .5], fill=(255, 170, 160))
        bx = w * .62 + shift * 0.3
        d.line([(bx - 40, h * .22), (bx, h * .25), (bx + 40, h * .21)], fill=(30, 30, 40), width=max(3, h // 60))
        return im
    if mode == "double":
        a, b = one(-h * .06), one(h * .06)
        return Image.blend(a, b, 0.5)
    im = one()
    if mode == "clean":                                   # the ball "comes out": a soft shadow and a brighter rim
        d = ImageDraw.Draw(im)
        cx = w * .38
        d.ellipse([cx - h * .2, h * .74, cx + h * .2, h * .8], fill=(20, 40, 30, 90))
        if prop != "popcorn":                             # the ball's bright rim (the bucket has its own stripes)
            d.arc([cx - h * .17, h * .38, cx + h * .17, h * .72], 200, 340, fill=(255, 230, 220), width=max(3, h // 70))
    return im


def world_image(world, screen="plain", blur=0.0):
    """1920x1080 background of a place (cached). blur > 0: shallow depth of field for close-ups."""
    def make():
        W, H = 1920, 1080
        if world == "living_room":
            im = Image.fromarray(_grad(W, H, (246, 222, 190), (226, 196, 160)).astype(np.uint8)).convert("RGBA")
            d = ImageDraw.Draw(im)
            d.rectangle([0, 760, W, H], fill=(170, 120, 85))                               # floor
            d.rectangle([0, 740, W, 770], fill=(150, 100, 70))
            d.rounded_rectangle([140, 170, 560, 560], radius=12, fill=(150, 200, 235), outline=(255, 255, 255), width=16)
            d.line([350, 170, 350, 560], fill=(255, 255, 255), width=10)                   # window
            d.line([140, 365, 560, 365], fill=(255, 255, 255), width=10)
            d.ellipse([400, 210, 470, 280], fill=(255, 240, 190))
            d.rectangle([1380, 180, 1640, 380], fill=(120, 90, 70)); d.rectangle([1395, 195, 1625, 365], fill=(250, 238, 210))
            d.polygon([(1420, 340), (1490, 250), (1550, 320), (1580, 280), (1610, 340)], fill=(110, 160, 120))
            d.rounded_rectangle([240, 600, 1680, 860], radius=60, fill=(90, 120, 150))       # sofa
            d.rounded_rectangle([240, 520, 1680, 700], radius=60, fill=(105, 138, 170))
            d.rectangle([1760, 300, 1780, 760], fill=(80, 70, 60)); d.polygon([(1700, 300), (1840, 300), (1800, 200), (1740, 200)],
                                                                                fill=(255, 225, 160))
            d.ellipse([60, 640, 200, 780], fill=(70, 130, 80)); d.rectangle([110, 760, 150, 800], fill=(140, 90, 60))
        elif world == "cinema_seats":                      # facing the audience: dark hall, seat rows behind them
            im = Image.fromarray(_grad(W, H, (22, 18, 40), (40, 18, 30)).astype(np.uint8)).convert("RGBA")
            d = ImageDraw.Draw(im)
            for r, (y, s, k) in enumerate([(420, 70, 0.35), (560, 90, 0.5), (720, 115, 0.7)]):
                col = shade((150, 30, 45), -0.6 + 0.3 * k)
                n = int(W / (s * 1.25)) + 2
                for i in range(n):
                    x = i * s * 1.25 - (r % 2) * s * 0.6
                    d.rounded_rectangle([x, y, x + s, y + s * 1.4], radius=int(s * 0.3), fill=col)
            beam = Image.new("RGBA", (W, H), (0, 0, 0, 0)); bd = ImageDraw.Draw(beam)
            bd.polygon([(960, 60), (0, 1080), (1920, 1080)], fill=(200, 210, 255, 26))         # light from the screen
            im.alpha_composite(beam)
            # light from the screen on the audience (they face it); it changes with the film
            glow = Image.new("RGBA", (W, H), (120, 140, 200, 30) if screen.split("+")[0] != "double" else (150, 120, 170, 30))
            im.alpha_composite(glow)
        elif world == "cinema_screen":                     # from behind the seats toward the big screen
            im = Image.fromarray(_grad(W, H, (12, 10, 22), (25, 15, 25)).astype(np.uint8)).convert("RGBA")
            d = ImageDraw.Draw(im)
            sx0, sy0, sx1, sy1 = 260, 80, 1660, 700
            glow = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
            gd.rectangle([sx0 - 60, sy0 - 40, sx1 + 60, sy1 + 60], fill=(180, 190, 255, 60))
            im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(40)))
            im.alpha_composite(movie_frame(sx1 - sx0, sy1 - sy0, screen), (sx0, sy0))
            d.rectangle([sx0, sy0, sx1, sy1], outline=(30, 30, 40), width=6)
            for r, (y, s) in enumerate([(780, 80), (900, 110)]):
                for i in range(int(W / (s * 1.2)) + 2):
                    x = i * s * 1.2 - (r % 2) * s * 0.6
                    d.ellipse([x, y, x + s, y + s * 1.1], fill=(18, 14, 24))                # heads of the audience
        elif world == "lobby":
            im = Image.fromarray(_grad(W, H, (60, 30, 70), (120, 50, 70)).astype(np.uint8)).convert("RGBA")
            d = ImageDraw.Draw(im)
            d.rectangle([0, 780, W, H], fill=(70, 30, 40))
            for i, x in enumerate((160, 560, 1360)):                                        # movie posters
                d.rectangle([x, 160, x + 300, 600], fill=(250, 220, 120), outline=(255, 250, 230), width=10)
                d.rectangle([x + 25, 185, x + 275, 470], fill=[(60, 120, 200), (200, 80, 80), (80, 160, 120)][i])
                d.rectangle([x + 40, 500, x + 260, 520], fill=(90, 60, 40)); d.rectangle([x + 70, 540, x + 230, 556], fill=(90, 60, 40))
            for k in range(14):                                                             # marquee lights
                d.ellipse([60 + k * 135, 70, 84 + k * 135, 94], fill=(255, 230, 150))
            d.rounded_rectangle([1720, 480, 1920, 800], radius=20, fill=(200, 40, 50))       # popcorn stand
        else:                                                                               # studio: soft plain
            im = Image.fromarray(_grad(W, H, (40, 70, 110), (20, 40, 70)).astype(np.uint8)).convert("RGBA")
        if blur > 0:
            im = im.filter(ImageFilter.GaussianBlur(blur))
        return im.convert("RGB")
    return _cached(("world", world, screen, blur), make)


# ------------------------------------------------------------------------------------------- shots
def _layout(shot, ids, subject):
    """[(char id, x centre, baseline y, scale, facing, view)] back to front, for 1920x1080."""
    a, b = ids[0], ids[-1]
    other = b if subject == a else a
    if shot == "wide":
        return [(a, 700, 1130, 0.62, 1, "front"), (b, 1220, 1130, 0.62, -1, "front")]
    if shot == "establishing":
        return [(a, 820, 1220, 0.55, 0, "back"), (b, 1100, 1220, 0.55, 0, "back")]
    if shot in ("closeup", "reaction"):
        left = subject == a
        return [(subject, 860 if left else 1060, 1460, 1.32, 1 if left else -1, "front")]
    if shot == "ots":                                       # over the listener's shoulder at the speaker
        left = subject == a                                 # speaker on the left, listener's back on the right
        return [(subject, 760 if left else 1160, 1250, 0.95, 1 if left else -1, "front"),
                (other, 1600 if left else 320, 1500, 1.45, 0, "back")]
    return [(a, 640, 1180, 0.84, 1, "front"), (b, 1280, 1180, 0.84, -1, "front")]        # two_shot


def _blinks(cid, t_abs):
    """Deterministic blinks: every 2.6–5.2 s (per-character schedule), 0.13 s long."""
    rng = np.random.default_rng(int(hashlib.sha1(cid.encode()).hexdigest()[:8], 16))
    t = rng.uniform(0.5, 2.0)
    while t < t_abs + 1:
        if t <= t_abs < t + 0.13:
            return False
        t += rng.uniform(2.6, 5.2)
    return True


def _state(spec, t):
    """Character states and screen at shot fraction t (applies 'changes' in order)."""
    chars = {k: dict(v) for k, v in spec.get("chars", {}).items()}
    screen, since = spec.get("screen", "plain"), {k: 0.0 for k in chars}
    for ch in sorted(spec.get("changes", []), key=lambda c: c["t"]):
        if ch["t"] <= t:
            for k, v in ch.get("chars", {}).items():
                chars.setdefault(k, {}).update(v)
                if "gesture" in v:
                    since[k] = ch["t"]
            screen = ch.get("screen", screen)
    return chars, screen, since


def mouth_level(e):
    return 0 if e < 0.12 else 1 if e < 0.35 else 2 if e < 0.65 else 3


def character_frame(spec, plan, t, w, h, bg, th):
    """drawkit renderer for kind "character" (w, h are the 1920x1080 design size).
    Plans with visual_style_version "storybook_v1" are drawn by storybook_scene (rigged illustrated characters in
    environment packs); every other plan keeps this drawing exactly as before."""
    if (plan or {}).get("visual_style_version") == "storybook_v1":
        import storybook_scene
        return storybook_scene.character_frame(spec, plan, t, w, h, bg, th)
    dur = float(spec.get("_dur", 4.0))
    n = max(int(spec.get("_n", 1)), 1)
    fi = min(int(round(t * (n - 1))), n - 1)
    t_abs = float(spec.get("_t0", 0.0)) + t * dur
    chars, screen, since = _state(spec, t)
    ids = list(chars) or list(bible().characters)
    shot, subject = spec.get("shot", "two_shot"), spec.get("subject") or ids[0]
    world = spec.get("world", "studio")
    close = shot in ("closeup", "reaction", "ots")
    frame = world_image(world, screen, blur=6.0 if close else (2.0 if shot == "two_shot" else 0.0)).copy().convert("RGBA")
    speech = spec.get("_speech", {})
    for cid, cx, base, sc, facing, view in _layout(shot, ids, subject):
        st = chars.get(cid, {})
        e = (speech.get(cid) or [0.0] * n)[fi] if speech.get(cid) else 0.0
        breathe = 5 * sc * math.sin(2 * math.pi * t_abs / 3.6 + (0.0 if cid == ids[0] else 1.7))
        if view == "back":
            spr = back_sprite(cid, sc)
            y = base - spr.height + breathe
            frame.alpha_composite(spr, (int(cx - spr.width / 2), int(y)))
            continue
        gest = st.get("gesture", "idle")
        if close and st.get("hold") and gest in ("idle", "explain", "open_hands", "nod"):
            gest = "show"                                # otherwise the prop is below the frame
        g_age = (t - since.get(cid, 0.0)) * dur
        phase = min(g_age / 0.5, 1.0) if gest not in ("idle", "nod") else 0.0
        nod = 0.0
        if gest == "nod" and g_age < 1.6:
            nod = 10 * sc * abs(math.sin(math.pi * g_age / 0.4))
        talk_bob = 4 * sc * e * math.sin(2 * math.pi * t_abs * 3.1)
        expr = st.get("expr", "listening" if e < 0.05 else "neutral")
        torso = torso_sprite(cid, sc)
        x0 = int(cx - torso.width / 2)
        y0 = int(base - torso.height + breathe)
        if gest == "duck":                               # dodge: drop ~12% of the body fast, then stay low
            y0 += int(0.12 * torso.height * phase)
        frame.alpha_composite(torso, (x0, y0))
        head = head_sprite(cid, sc, expr, mouth_level(e), _blinks(cid, t_abs), facing, tuple(st.get("wear", [])))
        frame.alpha_composite(head, (x0, int(y0 + nod + talk_bob + 1.5 * breathe)))
        arms = arms_sprite(cid, sc, gest, phase, facing, st.get("hold"))
        frame.alpha_composite(arms, (int(cx - arms.width / 2), y0))
    if spec.get("push"):
        z = 1.0 + 0.05 * t
        cw, ch_ = int(1920 / z), int(1080 / z)
        frame = frame.crop(((1920 - cw) // 2, (1080 - ch_) // 2, (1920 - cw) // 2 + cw, (1080 - ch_) // 2 + ch_)).resize(
            (1920, 1080), Image.BILINEAR)
    if spec.get("label"):
        _tag(frame, spec["label"])
    return np.asarray(frame.convert("RGB"))


def _tag(frame, text):
    import drawkit
    d = ImageDraw.Draw(frame)
    f = drawkit.font(30, True)
    tw = d.textlength(text, font=f)
    d.rounded_rectangle([1920 - tw - 80, 36, 1880, 92], radius=14, fill=(0, 0, 0, 150))
    d.text((1920 - tw - 58, 64), text, font=f, fill=(255, 255, 255), anchor="lm")


def check_spec(spec, plan=None):
    if (plan or {}).get("visual_style_version") == "storybook_v1":
        import storybook_scene
        return storybook_scene.check_spec(spec, plan)
    out = []
    b = bible()
    if spec.get("world", "studio") not in WORLDS:
        out.append(f"unknown world '{spec.get('world')}' (use {', '.join(WORLDS)})")
    if spec.get("shot", "two_shot") not in SHOTS:
        out.append(f"unknown character shot '{spec.get('shot')}' (use {', '.join(SHOTS)})")
    chars = spec.get("chars", {})
    if not chars:
        out.append("character shot needs 'chars'")
    for cid, st in list(chars.items()) + [(k, v) for c in spec.get("changes", []) for k, v in c.get("chars", {}).items()]:
        if cid not in b.characters:
            out.append(f"unknown character '{cid}' (Character Bible: {', '.join(b.characters)})")
            continue
        if st.get("expr", "neutral") not in b.characters[cid].expressions:
            out.append(f"{cid}: unknown expression '{st.get('expr')}'")
        if st.get("gesture", "idle") not in b.characters[cid].gestures:
            out.append(f"{cid}: unknown gesture '{st.get('gesture')}'")
        for x in st.get("wear", []):
            if x not in WEARABLES:
                out.append(f"{cid}: unknown wearable '{x}'")
        if st.get("hold") and st["hold"] not in HOLDABLES:
            out.append(f"{cid}: unknown held prop '{st['hold']}'")
    if spec.get("shot") in ("closeup", "reaction", "ots") and spec.get("subject") not in chars:
        out.append(f"{spec.get('shot')} needs a 'subject' that is one of the chars")
    for c in spec.get("changes", []):
        if not 0 <= c.get("t", -1) <= 1:
            out.append(f"change time t={c.get('t')} must be a fraction 0..1")
    return out


def reference_sheet(path, cid=None):
    """Asset library: one PNG per character with every expression (front) + IDs (stable content hashes)."""
    b = bible()
    ids, rows = [cid] if cid else list(b.characters), []
    for c in ids:
        ch = b.characters[c]
        exprs = ch.expressions
        sheet = Image.new("RGB", (len(exprs) * 260, 330), (235, 238, 242))
        d = ImageDraw.Draw(sheet)
        for i, e in enumerate(exprs):
            tor, hd = torso_sprite(c, 0.36), head_sprite(c, 0.36, e, 0, True, 0, ())
            sheet.paste(tor, (i * 260 + 4, 330 - tor.height + 40), tor)
            sheet.paste(hd, (i * 260 + 4, 330 - tor.height + 40), hd)
            d.text((i * 260 + 130, 12), e, fill=(30, 30, 40), anchor="mt")
            rows.append({"character": c, "expression": e, "gesture": "idle", "asset_id": charbible.asset_id(ch, e, "idle")})
        out = str(path).replace(".png", f"_{c}.png")
        sheet.save(out)
    return rows

"""Physical demonstrations and licensed photos as drawn shots (no AI, no GPU): the "show, don't tell" visuals of
the two-friend stories. They draw real objects (a hand, eyes, a projector, a screen, glasses, light) acting out
the idea — not boxes and arrows. Short on-screen words only where they label what you see ("LEFT EYE").

drawkit kinds:
  "demo":  {"template": "finger_test" | "eye_views" | "projector_glasses", "caption": optional short text}
  "photo": {"file": absolute path, "credit": "title — author, licence", "move": "push_in|pan_left|pan_right"}
All animation is driven by t (0..1 of the visual), so a demo fits whatever length the dialogue gives it.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import charanim
import drawkit

TEMPLATES = ("finger_test", "eye_views", "projector_glasses")
LABELS = {"te": {"LEFT EYE": "ఎడమ కన్ను", "RIGHT EYE": "కుడి కన్ను", "PROJECTOR": "ప్రొజెక్టర్",
                 "SILVER SCREEN": "సిల్వర్ స్క్రీన్", "BRAIN: ONE PICTURE WITH DEPTH": "మెదడు: లోతు ఉన్న ఒకే చిత్రం",
                 "EACH LENS LETS ONLY ITS OWN PICTURE THROUGH": "ప్రతి లెన్స్ తన చిత్రాన్ని మాత్రమే పంపుతుంది"},
          # code-mixed Telugu plans (plan "code_mix"): the words the friends actually say
          "te_mix": {"LEFT EYE": "Left eye", "RIGHT EYE": "Right eye", "PROJECTOR": "Projector",
                     "SILVER SCREEN": "Silver screen", "BRAIN: ONE PICTURE WITH DEPTH": "Brain: రెండు views = ఒక 3D picture",
                     "EACH LENS LETS ONLY ITS OWN PICTURE THROUGH": "ఒక్కో lens తన pictureని మాత్రమే పంపుతుంది"}}
_LANG = ["en"]                                            # set per frame from the plan's language


def tr(text):
    """On-screen words in the video's language (drawn with the script font, so Telugu shapes correctly)."""
    return LABELS.get(_LANG[0], {}).get(text) or LABELS.get(_LANG[0].split("_")[0], {}).get(text, text)
LEFT_C, RIGHT_C = (255, 150, 60), (70, 170, 255)          # the left-eye picture / the right-eye picture


def _ease(x):
    return drawkit.ease(x)


def _label(d, xy, text, col=(255, 255, 255), size=40, bg=(0, 0, 0, 150)):
    text = tr(text)
    f = drawkit.font(size, True)
    tw = d.textlength(text, font=f)
    x, y = xy
    d.rounded_rectangle([x - tw / 2 - 18, y - size * 0.75, x + tw / 2 + 18, y + size * 0.75], radius=14, fill=bg)
    d.text((x, y), text, font=f, fill=col, anchor="mm")


def _eye_icon(d, cx, cy, s, open_, col=(255, 255, 255)):
    if open_:
        d.ellipse([cx - s, cy - s * 0.6, cx + s, cy + s * 0.6], fill=col, outline=(30, 30, 40), width=4)
        d.ellipse([cx - s * 0.35, cy - s * 0.35, cx + s * 0.35, cy + s * 0.35], fill=(40, 60, 90))
    else:
        d.arc([cx - s, cy - s * 0.7, cx + s, cy + s * 0.5], 20, 160, fill=col, width=7)


# ------------------------------------------------------------------ 1. the finger test (try it yourself)
def finger_test(spec, t, w, h):
    """Your thumb in front of a far window: close one eye, then the other — the near thumb jumps, the far window
    hardly moves. Alternates left/right eye 4 times over the shot."""
    phase = int(t * 4) % 2                                    # 0: left eye open, 1: right eye open
    k = (t * 4) % 1
    blend = _ease(min(k / 0.18, 1.0))                         # quick switch at the start of each quarter
    x_near = (1 if phase == 0 else -1) * 130           # seen by the LEFT eye, a near thumb appears to the RIGHT
    x_prev = -x_near
    near_dx = x_prev + (x_near - x_prev) * blend
    far_dx = near_dx * 0.08
    world = spec.get("world", "living_room")
    bg = charanim.world_image(world if world in ("living_room", "lobby") else "living_room").copy()
    bg = bg.crop((int(160 + far_dx), 100, int(160 + far_dx) + 1600, 1000)).resize((w, h), Image.BILINEAR)
    bg = bg.filter(ImageFilter.GaussianBlur(3)).convert("RGBA")
    d = ImageDraw.Draw(bg)
    skin = charanim.rgb(charanim.bible().by_role("curious").look.skin)
    cx = w / 2 + near_dx
    d.rounded_rectangle([cx - 70, 470, cx + 70, h + 40], radius=60, fill=charanim.shade(skin, -0.05))     # thumb
    d.rounded_rectangle([cx - 52, 500, cx + 52, 590], radius=40, fill=charanim.shade(skin, 0.25))        # nail
    d.rounded_rectangle([cx - 190, 760, cx + 120, h + 80], radius=110, fill=skin)                        # fist
    for j in range(3):
        d.arc([cx - 180, 790 + j * 70, cx + 40, 880 + j * 70], 200, 330, fill=charanim.shade(skin, -0.25), width=6)
    over = Image.new("RGBA", (w, h), (0, 0, 0, 0)); od = ImageDraw.Draw(over)
    od.rounded_rectangle([w / 2 - 260, 40, w / 2 + 260, 170], radius=30, fill=(0, 0, 0, 150))
    _eye_icon(od, w / 2 - 110, 105, 55, phase == 0)
    _eye_icon(od, w / 2 + 110, 105, 55, phase == 1)
    bg.alpha_composite(over)
    _label(ImageDraw.Draw(bg), (w / 2, 230), "LEFT EYE" if phase == 0 else "RIGHT EYE", size=42)
    return bg


# ------------------------------------------------------------------ 2. each eye sees its own picture → one 3D view
def _scene_view(w, h, ball_dx, tint=None):
    """A ball in front of a tree, as one eye sees it (the near ball shifts, the far tree barely)."""
    im = Image.new("RGBA", (w, h), (205, 230, 250, 255)); d = ImageDraw.Draw(im)
    d.rectangle([0, h * 0.72, w, h], fill=(120, 185, 110))
    tx = w * 0.62 + ball_dx * 0.1
    d.rectangle([tx - w * 0.03, h * 0.35, tx + w * 0.03, h * 0.75], fill=(120, 80, 50))
    d.ellipse([tx - w * 0.16, h * 0.08, tx + w * 0.16, h * 0.48], fill=(60, 140, 70))
    bx = w * 0.42 + ball_dx
    d.ellipse([bx - w * 0.11, h * 0.45, bx + w * 0.11, h * 0.45 + w * 0.22], fill=(235, 70, 60))
    d.ellipse([bx - w * 0.06, h * 0.49, bx - w * 0.01, h * 0.49 + w * 0.05], fill=(255, 170, 160))
    if tint:
        im.alpha_composite(Image.new("RGBA", (w, h), tint + (40,)))
    return im


def eye_views(spec, t, w, h):
    """Top: a head from above with sight lines from each eye to the ball. Then the two pictures appear side by side
    (ball in a different place in each), then slide together into one picture where the ball stands out in depth."""
    im = Image.new("RGBA", (w, h), (24, 32, 52, 255)); d = ImageDraw.Draw(im)
    a = _ease(min(t / 0.3, 1))                         # sight lines
    b = drawkit.prog(t, 0.25, 0.45)                    # two pictures appear
    c = drawkit.prog(t, 0.65, 0.85)                    # they merge
    hx, hy = 330, 560                                  # head from above
    skin = charanim.rgb(charanim.bible().by_role("curious").look.skin)
    hair = charanim.rgb(charanim.bible().by_role("curious").look.hair)
    d.ellipse([hx - 150, hy - 170, hx + 150, hy + 170], fill=hair)
    d.ellipse([hx + 60, hy - 120, hx + 175, hy + 120], fill=skin)
    eyes = [(hx + 160, hy - 60), (hx + 160, hy + 60)]
    for (ex, ey), col in zip(eyes, (LEFT_C, RIGHT_C)):
        d.ellipse([ex - 20, ey - 20, ex + 20, ey + 20], fill=(255, 255, 255), outline=col, width=6)
    ball = (760, 560)
    d.ellipse([ball[0] - 50, ball[1] - 50, ball[0] + 50, ball[1] + 50], fill=(235, 70, 60))
    d.ellipse([985 - 40, 520, 985 + 40, 600], fill=(60, 140, 70))          # the far tree, further along the view
    if a > 0:
        for (ex, ey), col in zip(eyes, (LEFT_C, RIGHT_C)):
            x1, y1 = ex + (ball[0] - ex) * a, ey + (ball[1] - ey) * a
            d.line([ex, ey, x1, y1], fill=col, width=7)
    pw, ph = 380, 300
    if c < 1:
        # the left eye sees the near ball further RIGHT against the far tree, the right eye further left
        for k, (col, dx, name) in enumerate(((LEFT_C, 50, "LEFT EYE"), (RIGHT_C, -50, "RIGHT EYE"))):
            if b <= 0:
                continue
            x0 = 1080 + k * (pw + 60) - (k * (pw + 60) - (pw + 60) / 2) * c
            y0 = 260 + 40 * (1 - b)
            pic = _scene_view(pw, ph, dx)
            pic.putalpha(int(255 * b * (1 - 0.5 * c)))
            im.alpha_composite(pic, (int(x0), int(y0)))
            fade = tuple(int(bgc + (cc - bgc) * b) for cc, bgc in zip(col, (24, 32, 52)))   # fades in with the picture
            d.rectangle([x0, y0, x0 + pw, y0 + ph], outline=fade, width=8)
            if c <= 0:                                 # names go before the pictures start to merge
                _label(d, (x0 + pw / 2, y0 + ph + 50), name, col=col, size=34)
    if c > 0:                                          # one picture, with depth: the ball floats in front
        x0, y0 = 1080 + (pw + 60) / 2, 260
        pic = _scene_view(pw, ph, 0)
        pd = ImageDraw.Draw(pic)
        pd.ellipse([pw * 0.42 - pw * 0.13, ph * 0.83, pw * 0.42 + pw * 0.13, ph * 0.9], fill=(30, 60, 30, 110))
        big = pic.resize((int(pw * (1 + 0.15 * c)), int(ph * (1 + 0.15 * c))), Image.BILINEAR)
        big.putalpha(int(255 * c))
        im.alpha_composite(big, (int(x0 - (big.width - pw) / 2), int(y0 - (big.height - ph) / 2)))
        _label(d, (x0 + pw / 2, y0 + ph + 70), "BRAIN: ONE PICTURE WITH DEPTH", size=34)
    return im


# ------------------------------------------------------------------ 3. projector → screen → glasses → eyes
def projector_glasses(spec, t, w, h):
    """Seen from above: the projector (behind the audience) throws two pictures onto the same screen at once —
    orange for the left eye, blue for the right eye — each with its own light "twist" (polarisation). The silver
    screen reflects both to the viewer. Each lens lets one twist through and blocks the other, so each eye gets
    only its own picture."""
    im = Image.new("RGBA", (w, h), (16, 14, 28, 255)); d = ImageDraw.Draw(im)
    a = drawkit.prog(t, 0.0, 0.25)                     # beams to the screen
    b = drawkit.prog(t, 0.25, 0.5)                     # reflected back to both lenses
    c = drawkit.prog(t, 0.5, 0.72)                     # filtered at the lenses
    px, py = 170, 190                                  # projector, above and behind the audience
    d.rounded_rectangle([px - 120, py - 75, px + 80, py + 75], radius=18, fill=(70, 72, 84))
    d.ellipse([px + 50, py - 48, px + 146, py + 48], fill=(40, 42, 50), outline=(180, 190, 210), width=6)
    _label(d, (px, py + 125), "PROJECTOR", size=28)
    sx = 1760
    d.rectangle([sx, 110, sx + 36, 980], fill=(220, 225, 235))
    _label(d, (sx - 30, 70), "SILVER SCREEN", size=28)
    vx, vy = 720, 600                                  # viewer's head from above, facing the screen (right)
    look = charanim.bible().by_role("curious").look
    skin, hair = charanim.rgb(look.skin), charanim.rgb(look.hair)
    d.ellipse([vx - 170, vy - 190, vx + 150, vy + 190], fill=hair)
    d.ellipse([vx + 40, vy - 150, vx + 190, vy + 150], fill=skin)
    eyes = [(vx + 175, vy - 70), (vx + 175, vy + 70)]  # upper = LEFT eye (facing right, left is up)
    lens = [(vx + 250, vy - 70), (vx + 250, vy + 70)]
    for (ex, ey), col in zip(eyes, (LEFT_C, RIGHT_C)):
        d.ellipse([ex - 22, ey - 22, ex + 22, ey + 22], fill=(255, 255, 255), outline=col, width=6)
    d.line([lens[0][0], lens[0][1], lens[1][0], lens[1][1]], fill=(15, 15, 18), width=10)
    for (lx, ly), col in zip(lens, (LEFT_C, RIGHT_C)):
        d.rounded_rectangle([lx - 16, ly - 50, lx + 16, ly + 50], radius=10, fill=(40, 44, 52), outline=col, width=7)

    def beam(p, q, col, k, twist, width=10):
        n = 24
        for i in range(int(n * k)):
            u0, u1 = i / n, (i + 1) / n
            x0, y0 = p[0] + (q[0] - p[0]) * u0, p[1] + (q[1] - p[1]) * u0
            x1, y1 = p[0] + (q[0] - p[0]) * u1, p[1] + (q[1] - p[1]) * u1
            d.line([x0, y0, x1, y1], fill=col, width=width)
            if i % 4 == 2:                              # little rotating marks = the light's twist
                ang = twist * (i * 0.6 + t * 12)
                d.line([x0 - 18 * math.cos(ang), y0 - 18 * math.sin(ang), x0 + 18 * math.cos(ang),
                        y0 + 18 * math.sin(ang)], fill=(255, 255, 255), width=4)

    hit = [(sx, 520), (sx, 580)]
    for (hx, hy), col, tw in zip(hit, (LEFT_C, RIGHT_C), (1, -1)):
        beam((px + 146, py + (-10 if tw > 0 else 10)), (hx, hy), col, a, tw)
    if b > 0:
        for (hx, hy), col, tw in zip(hit, (LEFT_C, RIGHT_C), (1, -1)):
            for (lx, ly) in lens:                      # both pictures reach BOTH lenses
                beam((hx, hy), (lx + 18, ly + (-12 if tw > 0 else 12)), col, b, tw, width=7)
    if c > 0:
        for k, ((lx, ly), (ex, ey), col) in enumerate(zip(lens, eyes, (LEFT_C, RIGHT_C))):
            d.line([lx - 16, ly, lx - 16 + (ex + 22 - lx + 16) * c, ly], fill=col, width=12)
            bad = RIGHT_C if k == 0 else LEFT_C        # the other picture is stopped at this lens
            x0 = lx + 40
            d.line([x0, ly - 22, x0 + 44, ly + 22], fill=bad, width=9)
            d.line([x0, ly + 22, x0 + 44, ly - 22], fill=bad, width=9)
        _label(d, (w / 2, 60), "EACH LENS LETS ONLY ITS OWN PICTURE THROUGH", size=38)   # clear of the subtitles
    return im


def demo_frame(spec, plan, t, w, h, bg, th):
    _LANG[0] = (plan or {}).get("language", "en") + ("_mix" if (plan or {}).get("code_mix") else "")
    fn = {"finger_test": finger_test, "eye_views": eye_views, "projector_glasses": projector_glasses}[spec["template"]]
    im = fn(spec, t, w, h)
    if spec.get("caption"):                            # top: the bottom ~20% belongs to the subtitles
        _label(ImageDraw.Draw(im), (w / 2, 300), spec["caption"], size=40)
    return np.asarray(im.convert("RGB"))


# ------------------------------------------------------------------ licensed photo, fitted + slow camera move
_photo = {}


def photo_frame(spec, plan, t, w, h, bg, th):
    key = (spec["file"], w, h)
    if key not in _photo:
        _photo.clear()
        src = Image.open(spec["file"]).convert("RGB")
        big = Image.new("RGB", (int(w * 1.1), int(h * 1.1)))
        back = src.resize(big.size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(30))
        big.paste(Image.eval(back, lambda v: int(v * 0.75)))
        fg = src.copy(); fg.thumbnail(big.size, Image.LANCZOS)
        big.paste(fg, ((big.width - fg.width) // 2, (big.height - fg.height) // 2))
        _photo[key] = big
    big = _photo[key]
    move = spec.get("move", "push_in")
    z = 1.0 + 0.08 * t if move == "push_in" else 1.04
    cw, ch = int(big.width / z / 1.1 * 1.0), int(big.height / z / 1.1)
    x = (big.width - cw) / 2 + {"pan_left": -1, "pan_right": 1}.get(move, 0) * (t - 0.5) * (big.width - cw) * 0.8
    y = (big.height - ch) / 2
    im = big.crop((int(x), int(y), int(x) + cw, int(y) + ch)).resize((w, h), Image.BILINEAR).convert("RGBA")
    d = ImageDraw.Draw(im)
    f = drawkit.font(24)
    text = spec.get("credit", "")
    while d.textlength(text, font=f) > w * 0.6 and len(text) > 20:
        text = text[:-4].rstrip() + "…"
    if text:
        tw = d.textlength(text, font=f)
        d.rounded_rectangle([w - tw - 60, 24, w - 24, 66], radius=8, fill=(0, 0, 0, 150))
        d.text((w - tw - 42, 45), text, font=f, fill=(235, 235, 240), anchor="lm")
    return np.asarray(im.convert("RGB"))


def check_spec(kind, spec, plan=None):
    out = []
    if kind == "demo" and spec.get("template") not in TEMPLATES:
        out.append(f"unknown demo template '{spec.get('template')}' (use {', '.join(TEMPLATES)})")
    if kind == "photo":
        import os
        if not spec.get("file") or not os.path.exists(spec["file"]):
            out.append(f"photo file missing: {spec.get('file')}")
        if not spec.get("credit"):
            out.append("photo needs its credit line (title, author, licence)")
    return out

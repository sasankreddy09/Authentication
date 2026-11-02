"""Provisional "storybook" artwork for Arjun and Meera: layered, rig-ready parts drawn once by code.

Why code: the real artwork is meant to come from the Character Studio notebook (FLUX.2 [klein] 4B on a Colab T4,
cut into layers, approved by a person). Until that exists, this module draws a PROVISIONAL set in the same layer
format, so the asset library, rig, lip sync, environments, QC and whole videos can be built and tested now. It is
marked "provisional" in every manifest and is never reported as approved artwork.

Style: soft storybook shading (gradient fills, blurred shadow and highlight passes clipped to each shape, soft
coloured outlines), big expressive eyes with two highlights, rounded shapes from cubic Bézier paths — no stick limbs,
no circle-and-line bodies. Every part is drawn complete, including the area another part normally hides (shoulder
caps, the top of the forearm, the neck under the collar), so rotating a joint never opens a seam.

Design units: a 1000 x 1800 canvas (front view, full body), stored at `res` pixels per unit (default 1.6).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

ART_VERSION = "storybook-procedural-1"
CANVAS = (1000, 1800)
SS = 2                                    # supersampling while drawing


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


# ------------------------------------------------------------------------------------------- the two designs
DESIGNS = {
    "arjun": {
        "skin": "#B97A4F", "skin_shadow": "#94593A", "blush": "#E3876A", "lip": "#8E4A38",
        "hair": "#2A1B15", "hair_light": "#55392B", "iris": "#5B3620", "iris_light": "#9A6A40", "brow": "#2A1B15",
        "top": "#1E8C8B", "top_shadow": "#126767", "top_light": "#4DB8B2",        # teal hoodie
        "tee": "#F6C343", "tee_shadow": "#D9A126",                               # warm yellow T-shirt
        "pants": "#2F3547", "pants_shadow": "#21263A",
        "shoe": "#E8573C", "shoe_shadow": "#B83C27", "sole": "#F4F1EA",
        "hair_style": "wavy_short_curl", "lashes": 0, "accessory": None,
    },
    "meera": {
        "skin": "#A8683F", "skin_shadow": "#86502F", "blush": "#E07C67", "lip": "#9A4A45",
        "hair": "#23150F", "hair_light": "#4E3124", "iris": "#4A2A18", "iris_light": "#8C5A34", "brow": "#23150F",
        "top": "#F27B6B", "top_shadow": "#CC5A50", "top_light": "#FFA08F",        # coral jacket
        "tee": "#B9A2E6", "tee_shadow": "#977DCC",                               # lavender top
        "pants": "#33405E", "pants_shadow": "#232D47",
        "shoe": "#A98BE0", "shoe_shadow": "#7E62BA", "sole": "#F4F1EA",
        "hair_style": "wavy_half_up", "lashes": 2, "accessory": "star_clip",
    },
}

# where things are (design units); left/right = SCREEN left/right in the front view
HEAD_C = (500, 470)
EYE_C = {"L": (402, 505), "R": (598, 505)}
EYE_R = (60, 70)
MOUTH_C = (500, 648)
SHOULDER = {"L": (360, 815), "R": (640, 815)}
ELBOW = {"L": (338, 1000), "R": (662, 1000)}
WRIST = {"L": (334, 1170), "R": (666, 1170)}


# ------------------------------------------------------------------------------------------- drawing helpers
class Canvas:
    def __init__(self, res):
        self.res, self.k = res, res * SS
        self.w, self.h = int(CANVAS[0] * self.k), int(CANVAS[1] * self.k)

    def P(self, pts):
        return [(x * self.k, y * self.k) for x, y in pts]

    def new(self):
        return Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))

    def mask(self, pts):
        m = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(m).polygon(self.P(pts), fill=255)
        return m

    def ell_mask(self, c, rx, ry, angle=0.0):
        return self.mask(ellipse_pts(c, rx, ry, angle))

    def finish(self, im):
        return im.resize((int(CANVAS[0] * self.res), int(CANVAS[1] * self.res)), Image.LANCZOS)


def bez(p0, p1, p2, p3, n=24):
    out = []
    for i in range(n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


def path(start, *segs, n=24):
    """start point + cubic segments (c1, c2, end) → polygon points."""
    pts, cur = [start], start
    for c1, c2, end in segs:
        pts += bez(cur, c1, c2, end, n)[1:]
        cur = end
    return pts


def ellipse_pts(c, rx, ry, angle=0.0, n=96):
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return [(c[0] + rx * math.cos(2 * math.pi * i / n) * ca - ry * math.sin(2 * math.pi * i / n) * sa,
             c[1] + rx * math.cos(2 * math.pi * i / n) * sa + ry * math.sin(2 * math.pi * i / n) * ca) for i in range(n)]


def capsule(a, b, r0, r1, n=40):
    """A limb: rounded at both ends, radius r0 at a and r1 at b (complete, so it can rotate without seams)."""
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    pts = []
    for i in range(n + 1):
        t = ang + math.pi / 2 + math.pi * i / n
        pts.append((b[0] + r1 * math.cos(t - math.pi), b[1] + r1 * math.sin(t - math.pi)))
    for i in range(n + 1):
        t = ang + math.pi / 2 + math.pi * i / n
        pts.append((a[0] + r0 * math.cos(t), a[1] + r0 * math.sin(t)))
    return pts


def fill(cv, im, mask, color, shade=None, light=None, top=0.0, bot=1.0, outline=None, ow=4.0):
    """Paint `color` inside `mask`; `shade`: a darker colour toward the bottom/right (soft form shadow);
    `light`: a soft highlight toward the top/left; outline: a soft coloured line just inside the edge."""
    col = Image.new("RGBA", im.size, color + (255,))
    if shade is not None:                                     # vertical gradient inside the shape's own box
        bb = mask.getbbox()
        if bb:
            g = np.zeros((im.size[1],), np.float32)
            y0, y1 = bb[1], bb[3]
            ys = np.arange(im.size[1])
            g = np.clip((ys - (y0 + top * (y1 - y0))) / max((bot - top) * (y1 - y0), 1), 0, 1) ** 1.4
            grad = Image.fromarray((g[:, None] * 255).astype(np.uint8).repeat(im.size[0], 1), "L")
            sh = Image.new("RGBA", im.size, shade + (255,))
            col = Image.composite(sh, col, grad)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    layer.paste(col, (0, 0), mask)
    if light is not None:                                     # a blurred rim light from the upper left
        off = ImageChops.offset(mask, int(14 * cv.k), int(18 * cv.k))
        rim = ImageChops.subtract(mask, off).filter(ImageFilter.GaussianBlur(10 * cv.k))
        rim = ImageChops.multiply(rim, mask)
        lt = Image.new("RGBA", im.size, light + (0,))
        lt.putalpha(rim.point(lambda v: int(v * 0.75)))
        layer.alpha_composite(lt)
    if outline is not None:
        er = mask.filter(ImageFilter.MinFilter(max(3, int(ow * cv.k) | 1)))
        edge = ImageChops.subtract(mask, er).filter(ImageFilter.GaussianBlur(0.6 * cv.k))
        ol = Image.new("RGBA", im.size, outline + (0,))
        ol.putalpha(edge)
        layer.alpha_composite(ol)
    im.alpha_composite(layer)


def soft(cv, im, mask, color, alpha=0.5, blur=12, clip=None, offset=(0, 0)):
    """A blurred shadow / glow, optionally clipped to another mask."""
    m = ImageChops.offset(mask, int(offset[0] * cv.k), int(offset[1] * cv.k)) if offset != (0, 0) else mask
    m = m.filter(ImageFilter.GaussianBlur(blur * cv.k)).point(lambda v: int(v * alpha))
    if clip is not None:
        m = ImageChops.multiply(m, clip)
    lay = Image.new("RGBA", im.size, color + (0,))
    lay.putalpha(m)
    im.alpha_composite(lay)


def stroke(cv, im, pts, color, width, alpha=255):
    d = ImageDraw.Draw(im)
    w = max(1, int(width * cv.k))
    d.line(cv.P(pts), fill=color + (alpha,), width=w, joint="curve")
    r = w / 2
    for x, y in (cv.P([pts[0]])[0], cv.P([pts[-1]])[0]):
        d.ellipse([x - r, y - r, x + r, y + r], fill=color + (alpha,))


# ------------------------------------------------------------------------------------------- parts
def head_shape():
    cx, cy = HEAD_C
    return path((cx, cy - 250), ((cx + 150, cy - 250), (cx + 245, cy - 150), (cx + 245, cy - 10)),
                ((cx + 245, cy + 120), (cx + 170, cy + 235), (cx, cy + 250)),
                ((cx - 170, cy + 235), (cx - 245, cy + 120), (cx - 245, cy - 10)),
                ((cx - 245, cy - 150), (cx - 150, cy - 250), (cx, cy - 250)))


def draw_head(cv, d):
    im = cv.new()
    skin, sh, ol = hexrgb(d["skin"]), hexrgb(d["skin_shadow"]), mix(hexrgb(d["skin_shadow"]), (40, 20, 10), 0.35)
    for sx in (-1, 1):                                        # ears (behind the face edge)
        e = cv.ell_mask((500 + sx * 238, 510), 38, 52)
        fill(cv, im, e, skin, shade=sh, outline=ol)
        soft(cv, im, cv.ell_mask((500 + sx * 240, 512), 16, 26), sh, 0.55, 4)
    m = cv.mask(head_shape())
    fill(cv, im, m, skin, shade=sh, top=0.45, bot=1.25, outline=ol, ow=4)
    soft(cv, im, cv.ell_mask((560, 360), 150, 110), (255, 230, 205), 0.22, 40, clip=m)       # forehead light
    soft(cv, im, cv.mask(head_shape()), sh, 0.35, 18, clip=m, offset=(-22, -16))           # form shadow (right)
    return cv.finish(im)


def draw_neck(cv, d):
    im = cv.new()
    m = cv.mask(path((452, 650), ((452, 700), (450, 760), (446, 800)), ((480, 815), (520, 815), (554, 800)),
                     ((550, 760), (548, 700), (548, 650))))
    fill(cv, im, m, hexrgb(d["skin_shadow"]), shade=mix(hexrgb(d["skin_shadow"]), (40, 20, 10), 0.25))
    soft(cv, im, cv.ell_mask((500, 690), 70, 30), (40, 20, 10), 0.35, 10, clip=m)         # chin shadow
    return cv.finish(im)


def draw_cheeks(cv, d):
    im = cv.new()
    for sx in (-1, 1):
        soft(cv, im, cv.ell_mask((500 + sx * 158, 598), 40, 20), hexrgb(d["blush"]), 0.38, 10)
    return cv.finish(im)


def draw_nose(cv, d):
    im = cv.new()
    sh = hexrgb(d["skin_shadow"])
    soft(cv, im, cv.ell_mask((505, 590), 24, 16), sh, 0.65, 4)
    soft(cv, im, cv.ell_mask((494, 580), 10, 7), (255, 235, 215), 0.5, 3)
    return cv.finish(im)


def eye_sclera_pts(side, variant):
    c = EYE_C[side]
    rx, ry = EYE_R
    if variant == "wide":
        rx, ry = rx * 1.1, ry * 1.16
    pts = ellipse_pts(c, rx, ry)
    if variant == "half":                                    # upper lid lowered: cut the top
        pts = [(x, max(y, c[1] - ry * 0.15)) for x, y in pts]
    if variant == "smile":                                   # cheeks push the lower lid up
        pts = [(x, min(y, c[1] + ry * 0.45 + 0.25 * abs(x - c[0]))) for x, y in pts]
    return pts


def draw_eye(cv, d, side, variant, part="all"):
    """One eye. part "white": the sclera only; "lids": lid lines and lashes only (drawn over the iris, which is a
    separate layer clipped to `eyemask_*` while rendering); "all": closed / happy eyes (no iris)."""
    im = cv.new()
    line = mix(hexrgb(d["hair"]), (0, 0, 0), 0.2)
    c = EYE_C[side]
    rx, ry = EYE_R
    if variant in ("closed", "happy"):
        if variant == "closed":
            pts = bez((c[0] - rx, c[1] + 6), (c[0] - rx * 0.4, c[1] + ry * 0.45), (c[0] + rx * 0.4, c[1] + ry * 0.45),
                      (c[0] + rx, c[1] + 6))
        else:                                                # ^ shaped: laughing / very happy
            pts = bez((c[0] - rx, c[1] + 16), (c[0] - rx * 0.45, c[1] - ry * 0.55), (c[0] + rx * 0.45, c[1] - ry * 0.55),
                      (c[0] + rx, c[1] + 16))
        stroke(cv, im, pts, line, 11)
        if d["lashes"]:
            outer = pts[0] if side == "L" else pts[-1]
            sx = -1 if side == "L" else 1
            stroke(cv, im, [outer, (outer[0] + 22 * sx, outer[1] - 14)], line, 8)
        return cv.finish(im)
    pts = eye_sclera_pts(side, variant)
    m = cv.mask(pts)
    if part == "white":
        fill(cv, im, m, (253, 252, 250), shade=(236, 234, 240), top=0.3, bot=1.2)
        return cv.finish(im)
    soft(cv, im, cv.mask([(x, y - 46) for x, y in pts]), (110, 80, 90), 0.28, 8, clip=m)   # lid shadow on the eye
    top = [p for p in pts if p[1] <= c[1] + 8]
    top.sort()
    lid = Image.new("L", (cv.w, cv.h), 0)
    dl = ImageDraw.Draw(lid)
    for i, (x, y) in enumerate(top):                           # thick in the middle, thin at the corners
        u = i / max(len(top) - 1, 1)
        r = (5 + 9 * math.sin(math.pi * min(max(u, 0.05), 0.95))) * cv.k
        dl.ellipse([x * cv.k - r, (y - 2) * cv.k - r, x * cv.k + r, (y - 2) * cv.k + r], fill=255)
    fill(cv, im, lid, line)
    bottom = [p for p in pts if p[1] > c[1] + ry * 0.35]
    bottom.sort()
    if bottom:
        stroke(cv, im, bottom, mix(line, hexrgb(d["skin"]), 0.55), 4)
    if d["lashes"]:
        outer = top[0] if side == "L" else top[-1]
        sx = -1 if side == "L" else 1
        for k in range(d["lashes"]):
            stroke(cv, im, [outer, (outer[0] + (24 - 6 * k) * sx, outer[1] - 16 - 12 * k)], line, 8)
    return cv.finish(im)


def eye_mask(cv, side, variant):
    m = cv.mask(eye_sclera_pts(side, variant))
    m = m.filter(ImageFilter.MinFilter(max(3, int(5 * cv.k) | 1)))
    return m.resize((int(CANVAS[0] * cv.res), int(CANVAS[1] * cv.res)), Image.LANCZOS)


def draw_iris(cv, d, side, small=False):
    im = cv.new()
    c = EYE_C[side]
    r = 50 if not small else 36
    iris = cv.ell_mask(c, r, r * 1.05)
    fill(cv, im, iris, hexrgb(d["iris_light"]), shade=hexrgb(d["iris"]), top=0.0, bot=0.7)
    soft(cv, im, cv.ell_mask(c, r, r * 1.05), (20, 10, 5), 0.6, 6, clip=iris, offset=(0, -10))
    fill(cv, im, cv.ell_mask((c[0], c[1] + 2), r * 0.52, r * 0.54), (16, 10, 8))
    fill(cv, im, cv.ell_mask((c[0] - r * 0.34, c[1] - r * 0.38), r * 0.34, r * 0.3), (255, 255, 255))   # highlights
    fill(cv, im, cv.ell_mask((c[0] + r * 0.35, c[1] + r * 0.38), r * 0.13, r * 0.12), (255, 255, 255))
    return cv.finish(im)


BROWS = {   # (inner dy, mid dy, outer dy): negative = up
    "neutral": (0, -8, 2), "raised": (-26, -34, -18), "furrow": (14, 4, -4), "sad": (-22, -10, 10),
    "curious": None, "high": (-40, -48, -30),
}


def draw_brows(cv, d, variant):
    im = cv.new()
    col = hexrgb(d["brow"])
    for side, sx in (("L", -1), ("R", 1)):
        shape = BROWS[variant]
        if variant == "curious":                             # one brow up, one level: a questioning look
            shape = BROWS["high"] if side == "R" else BROWS["neutral"]
        c = (EYE_C[side][0], 412)
        inner, mid, outer = shape
        pts = bez((c[0] - 48 * sx, c[1] + inner), (c[0] - 15 * sx, c[1] + mid - 6), (c[0] + 20 * sx, c[1] + mid - 6),
                  (c[0] + 52 * sx, c[1] + outer + 4))
        m = Image.new("L", (cv.w, cv.h), 0)
        dd = ImageDraw.Draw(m)
        for i, (x, y) in enumerate(pts):                      # tapered: thick inside, thin outside
            r = (13 - 6 * i / len(pts)) * cv.k
            dd.ellipse([x * cv.k - r, y * cv.k - r, x * cv.k + r, y * cv.k + r], fill=255)
        fill(cv, im, m, col)
    return cv.finish(im)


MOUTHS = ("rest", "closed", "small", "wide", "round", "teeth", "smile_talk", "smile", "frown", "wavy", "pursed",
          "laugh", "o_big")


def draw_mouth(cv, d, variant):
    im = cv.new()
    cx, cy = MOUTH_C
    lip, dark = hexrgb(d["lip"]), (70, 22, 26)
    tongue, teeth = (226, 112, 110), (250, 248, 244)

    def opening(pts, show_teeth=False, show_tongue=True):
        m = cv.mask(pts)
        fill(cv, im, m, dark)
        bb = [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]
        if show_tongue:
            t = cv.ell_mask(((bb[0] + bb[2]) / 2, bb[3] - 4), (bb[2] - bb[0]) * 0.32, (bb[3] - bb[1]) * 0.35)
            fill(cv, im, Image.fromarray(np.minimum(np.asarray(t), np.asarray(m))), tongue)
        if show_teeth:
            tm = cv.mask([(bb[0], bb[1] - 4), (bb[2], bb[1] - 4), (bb[2], bb[1] + (bb[3] - bb[1]) * 0.3),
                          (bb[0], bb[1] + (bb[3] - bb[1]) * 0.3)])
            fill(cv, im, Image.fromarray(np.minimum(np.asarray(tm), np.asarray(m))), teeth)
        stroke(cv, im, pts + [pts[0]], lip, 4)

    if variant in ("rest", "smile"):
        w, h = (46, 12) if variant == "rest" else (62, 22)
        stroke(cv, im, bez((cx - w, cy - 4), (cx - w / 2, cy + h), (cx + w / 2, cy + h), (cx + w, cy - 4)), lip, 7)
    elif variant == "closed":                                 # m / b / p: lips pressed
        stroke(cv, im, bez((cx - 40, cy), (cx - 15, cy + 6), (cx + 15, cy + 6), (cx + 40, cy)), lip, 9)
    elif variant == "frown":
        stroke(cv, im, bez((cx - 40, cy + 10), (cx - 15, cy - 6), (cx + 15, cy - 6), (cx + 40, cy + 10)), lip, 7)
    elif variant == "wavy":
        stroke(cv, im, bez((cx - 44, cy + 4), (cx - 20, cy - 10), (cx + 10, cy + 14), (cx + 42, cy - 2)), lip, 7)
    elif variant == "pursed":
        fill(cv, im, cv.ell_mask((cx + 18, cy), 16, 12), lip)
    elif variant == "small":
        opening(ellipse_pts((cx, cy + 4), 24, 14), show_tongue=False)
    elif variant == "wide":
        opening(path((cx - 48, cy - 6), ((cx - 20, cy - 12), (cx + 20, cy - 12), (cx + 48, cy - 6)),
                     ((cx + 44, cy + 34), (cx - 44, cy + 34), (cx - 48, cy - 6))), show_teeth=True)
    elif variant == "round":
        opening(ellipse_pts((cx, cy + 6), 22, 26))
    elif variant == "o_big":
        opening(ellipse_pts((cx, cy + 10), 30, 36))
    elif variant == "teeth":                                  # ee: wide and flat, teeth visible
        opening(path((cx - 52, cy - 2), ((cx - 20, cy - 8), (cx + 20, cy - 8), (cx + 52, cy - 2)),
                     ((cx + 40, cy + 18), (cx - 40, cy + 18), (cx - 52, cy - 2))), show_teeth=True, show_tongue=False)
    elif variant == "smile_talk":
        opening(path((cx - 54, cy - 8), ((cx - 20, cy - 2), (cx + 20, cy - 2), (cx + 54, cy - 8)),
                     ((cx + 40, cy + 30), (cx - 40, cy + 30), (cx - 54, cy - 8))), show_teeth=True)
    elif variant == "laugh":
        opening(path((cx - 64, cy - 12), ((cx - 25, cy - 4), (cx + 25, cy - 4), (cx + 64, cy - 12)),
                     ((cx + 52, cy + 52), (cx - 52, cy + 52), (cx - 64, cy - 12))), show_teeth=True)
    return cv.finish(im)


# ---- hair
def draw_hair(cv, d, which):
    im = cv.new()
    hair, light = hexrgb(d["hair"]), hexrgb(d["hair_light"])
    ol = mix(hair, (0, 0, 0), 0.4)
    if d["hair_style"] == "wavy_short_curl":
        if which == "back":
            m = cv.mask(path((240, 520), ((210, 330), (330, 190), (500, 188)), ((680, 186), (800, 320), (762, 520)),
                             ((730, 470), (700, 420), (690, 400)), ((560, 420), (440, 420), (310, 400)),
                             ((290, 430), (270, 470), (240, 520))))
            fill(cv, im, m, mix(hair, (0, 0, 0), 0.15), outline=ol)
            return cv.finish(im)
        # front: wavy fringe + sides, and the signature curl at the top
        pts = path((246, 520), ((226, 360), (300, 236), (420, 210)), ((520, 190), (650, 196), (720, 260)),
                   ((770, 310), (770, 420), (756, 520)), ((735, 470), (722, 430), (705, 398)),
                   ((680, 418), (650, 392), (635, 370)), ((600, 400), (560, 392), (545, 366)),
                   ((515, 398), (470, 396), (455, 368)), ((420, 400), (380, 392), (368, 366)),
                   ((330, 396), (300, 390), (292, 386)), ((282, 430), (268, 470), (246, 520)))
        m = cv.mask(pts)
        fill(cv, im, m, hair, shade=mix(hair, (0, 0, 0), 0.3), top=0.2, bot=1.2, outline=ol)
        for a, b2, c2, e in (((330, 300), (380, 250), (450, 240), (500, 250)), ((540, 236), (600, 232), (650, 250), (690, 290)),
                             ((300, 360), (320, 320), (350, 300), (390, 292))):
            stroke(cv, im, bez(a, b2, c2, e), light, 9, 170)
        curl = []                                            # the signature curl: a small spiral lock on top
        for i in range(60):
            t = i / 59
            ang = math.pi * 0.9 + t * math.pi * 2.3
            r = 46 * (1 - 0.7 * t)
            curl.append((520 + 40 * t + r * math.cos(ang), 214 - 30 * t + r * math.sin(ang)))
        stroke(cv, im, curl, hair, 26)
        stroke(cv, im, curl[10:40], light, 7, 190)
        return cv.finish(im)
    # Meera: wavy hair, half-up (a small bun at the back-top), bangs swept to one side, star clip
    if which == "back":
        m = cv.mask(path((232, 470), ((200, 300), (330, 196), (500, 194)), ((680, 192), (800, 300), (768, 470)),
                         ((800, 640), (840, 830), (790, 990)), ((750, 1040), (700, 1000), (690, 960)),
                         ((640, 900), (660, 760), (640, 650)), ((560, 600), (440, 600), (360, 650)),
                         ((340, 760), (360, 900), (310, 960)), ((300, 1000), (250, 1040), (210, 990)),
                         ((160, 830), (200, 640), (232, 470))))
        fill(cv, im, m, mix(hair, (0, 0, 0), 0.12), shade=mix(hair, (0, 0, 0), 0.35), outline=ol)
        for x0 in (250, 300, 700, 750):                         # wave strands
            sx = 1 if x0 < 500 else -1
            stroke(cv, im, bez((x0, 640), (x0 - 30 * sx, 760), (x0 + 30 * sx, 860), (x0 - 10 * sx, 960)), light, 7, 150)
        bun = cv.ell_mask((560, 200), 78, 64, -15)               # the half-up bun
        fill(cv, im, bun, hair, shade=mix(hair, (0, 0, 0), 0.3), outline=ol)
        stroke(cv, im, bez((505, 190), (540, 160), (590, 160), (620, 190)), light, 8, 170)
        return cv.finish(im)
    pts = path((236, 560), ((216, 370), (300, 236), (430, 212)), ((560, 190), (690, 214), (744, 300)),
               ((786, 370), (780, 470), (764, 600)), ((748, 560), (735, 500), (720, 440)),
               ((690, 420), (650, 380), (620, 330)), ((560, 380), (450, 410), (360, 420)),
               ((330, 440), (300, 470), (296, 520)), ((280, 540), (262, 560), (236, 560)))
    m = cv.mask(pts)
    fill(cv, im, m, hair, shade=mix(hair, (0, 0, 0), 0.3), top=0.2, bot=1.2, outline=ol)
    for a, b2, c2, e in (((340, 300), (420, 250), (520, 236), (600, 250)), ((630, 270), (690, 300), (720, 360), (730, 420)),
                         ((300, 380), (330, 340), (380, 320), (440, 320))):
        stroke(cv, im, bez(a, b2, c2, e), light, 9, 170)
    if d["accessory"] == "star_clip":                         # small yellow star hair clip (her signature)
        cx, cy, R, r = 676, 300, 30, 13
        star = [(cx + (R if i % 2 == 0 else r) * math.cos(-math.pi / 2 + i * math.pi / 5),
                 cy + (R if i % 2 == 0 else r) * math.sin(-math.pi / 2 + i * math.pi / 5)) for i in range(10)]
        fill(cv, im, cv.mask(star), (250, 205, 70), shade=(225, 150, 40), outline=(170, 100, 20))
    return cv.finish(im)


# ---- body
def draw_torso(cv, d):
    im = cv.new()
    top, sh, lt = hexrgb(d["top"]), hexrgb(d["top_shadow"]), hexrgb(d["top_light"])
    tee, tee_sh = hexrgb(d["tee"]), hexrgb(d["tee_shadow"])
    ol = mix(sh, (0, 0, 0), 0.35)
    body = path((400, 762), ((450, 744), (550, 744), (600, 762)), ((660, 772), (690, 800), (686, 860)),
                ((684, 960), (664, 1100), (656, 1180)), ((560, 1196), (440, 1196), (344, 1180)),
                ((336, 1100), (316, 960), (314, 860)), ((310, 800), (340, 772), (400, 762)))
    m = cv.mask(body)
    fill(cv, im, m, top, shade=sh, top=0.3, bot=1.3, light=lt, outline=ol)
    v = path((452, 772), ((470, 790), (530, 790), (548, 772)), ((560, 900), (545, 1080), (540, 1192)),
             ((510, 1196), (490, 1196), (460, 1192)), ((455, 1080), (440, 900), (452, 772)))
    vm = cv.mask(v)
    fill(cv, im, vm, tee, shade=tee_sh, top=0.4, bot=1.4)       # the shirt between the open jacket
    soft(cv, im, vm, (0, 0, 0), 0.25, 8, clip=m, offset=(10, 0))
    hood = path((400, 760), ((440, 730), (560, 730), (600, 760)), ((590, 800), (560, 812), (548, 790)),
                ((530, 770), (470, 770), (452, 790)), ((440, 812), (410, 800), (400, 760)))
    fill(cv, im, cv.mask(hood), mix(top, (255, 255, 255), 0.1), shade=sh, outline=ol)   # hood / collar
    if d["hair_style"] == "wavy_short_curl":
        for x in (468, 532):                                    # hoodie drawstrings
            stroke(cv, im, [(x, 800), (x - 4, 905)], (245, 245, 240), 7)
            fill(cv, im, cv.ell_mask((x - 4, 912), 7, 10), (220, 220, 215))
        stroke(cv, im, bez((370, 1070), (430, 1090), (570, 1090), (630, 1070)), ol, 4, 180)   # front pocket
    else:
        for y in (840, 920, 1000):                              # jacket buttons
            fill(cv, im, cv.ell_mask((436, y), 9, 9), (250, 235, 220), outline=ol, ow=2)
    stroke(cv, im, bez((350, 1160), (430, 1178), (570, 1178), (650, 1160)), ol, 5, 200)       # hem
    return cv.finish(im)


def draw_legs(cv, d):
    im = cv.new()
    p, ps = hexrgb(d["pants"]), hexrgb(d["pants_shadow"])
    ol = mix(ps, (0, 0, 0), 0.4)
    for sx in (-1, 1):
        x = 500 + sx * 78
        leg = path((x - 70, 1170), ((x - 70, 1300), (x - 62, 1420), (x - 58, 1540)),
                   ((x - 20, 1550), (x + 20, 1550), (x + 58, 1540)), ((x + 62, 1420), (x + 70, 1300), (x + 70, 1170)))
        fill(cv, im, cv.mask(leg), p, shade=ps, top=0.2, bot=1.2, outline=ol)
    for sx in (-1, 1):                                          # sneakers
        x = 500 + sx * 82
        shoe = path((x - 64, 1600), ((x - 70, 1550), (x - 40, 1525), (x, 1525)), ((x + 50, 1525), (x + 82, 1560), (x + 92, 1600)),
                    ((x + 60, 1616), (x - 40, 1616), (x - 64, 1600)))
        if sx < 0:
            shoe = [(2 * x - px, py) for px, py in shoe]
        fill(cv, im, cv.mask(shoe), hexrgb(d["shoe"]), shade=hexrgb(d["shoe_shadow"]), outline=mix(hexrgb(d["shoe_shadow"]), (0, 0, 0), 0.4))
        sole = [(px, max(py, 1596)) for px, py in shoe]
        fill(cv, im, cv.mask(sole), hexrgb(d["sole"]))
        stroke(cv, im, [(x - 25, 1550), (x + 25, 1550)], hexrgb(d["sole"]), 6, 220)
    return cv.finish(im)


def draw_upper_arm(cv, d, side):
    """The sleeve from the shoulder to the elbow. It is drawn over the torso: its outline fades out around the
    shoulder so the sleeve grows out of the jacket instead of showing a joint circle."""
    im = cv.new()
    a, b = SHOULDER[side], ELBOW[side]
    m = cv.mask(capsule(a, b, 50, 44))
    fill(cv, im, m, hexrgb(d["top"]), shade=hexrgb(d["top_shadow"]), light=hexrgb(d["top_light"]))
    ol = mix(hexrgb(d["top_shadow"]), (0, 0, 0), 0.35)
    er = m.filter(ImageFilter.MinFilter(max(3, int(4 * cv.k) | 1)))
    edge = ImageChops.subtract(m, er).filter(ImageFilter.GaussianBlur(0.6 * cv.k))
    fade = cv.ell_mask(a, 70, 70).filter(ImageFilter.GaussianBlur(28 * cv.k))
    edge = ImageChops.subtract(edge, fade)
    lay = Image.new("RGBA", im.size, ol + (0,)); lay.putalpha(edge)
    im.alpha_composite(lay)
    return cv.finish(im)


def draw_forearm(cv, d, side):
    im = cv.new()
    a, b = ELBOW[side], WRIST[side]
    m = cv.mask(capsule(a, b, 44, 40))
    fill(cv, im, m, hexrgb(d["top"]), shade=hexrgb(d["top_shadow"]), light=hexrgb(d["top_light"]),
         outline=mix(hexrgb(d["top_shadow"]), (0, 0, 0), 0.35))
    cuff = capsule((b[0] + (a[0] - b[0]) * 0.12, b[1] + (a[1] - b[1]) * 0.12), b, 42, 41)
    fill(cv, im, cv.mask(cuff), mix(hexrgb(d["top_shadow"]), (255, 255, 255), 0.08))
    return cv.finish(im)


HANDS = ("open", "point", "wave", "hold")


def draw_hand(cv, d, side, variant):
    """A soft cartoon hand at the wrist (pointing DOWN in the rest pose; the rig rotates it). One silhouette per hand:
    palm, fingers and thumb are merged before painting, so there are no seams inside the hand."""
    im = cv.new()
    skin, sh = hexrgb(d["skin"]), hexrgb(d["skin_shadow"])
    ol = mix(sh, (40, 20, 10), 0.35)
    wx, wy = WRIST[side]
    sx = -1 if side == "L" else 1                               # thumb toward the body's centre
    parts = [ellipse_pts((wx, wy + 42), 36, 40)]                # palm
    parts.append(capsule((wx - 24 * sx, wy + 30), (wx - 46 * sx, wy + 62), 15, 12))   # thumb
    if variant == "point":
        parts.append(capsule((wx + 4 * sx, wy + 70), (wx + 8 * sx, wy + 138), 13, 11))
        parts.append(ellipse_pts((wx + 2 * sx, wy + 76), 30, 22))                      # curled fingers
    elif variant in ("open", "wave"):
        spread = 1.0 if variant == "wave" else 0.45
        for i, ang in enumerate((-18, -6, 6, 18)):
            r = math.radians(ang * spread)
            a = (wx + (i - 1.5) * 16, wy + 66)
            b = (a[0] + 40 * math.sin(r), a[1] + (40 - 6 * abs(i - 1.5)) * math.cos(r))
            parts.append(capsule(a, b, 11.5, 10.5))
    else:                                                       # hold: a loose fist
        parts.append(ellipse_pts((wx, wy + 70), 36, 26))
    m = Image.new("L", (cv.w, cv.h), 0)
    for pts in parts:
        m = ImageChops.lighter(m, cv.mask(pts))
    fill(cv, im, m, skin, shade=sh, top=0.1, bot=1.3, outline=ol, ow=3.5)
    if variant == "hold":
        for i in range(3):
            stroke(cv, im, [(wx - 18 + 18 * i, wy + 74), (wx - 16 + 18 * i, wy + 88)], ol, 3.5, 150)
    return cv.finish(im)


def draw_glasses(cv, d):
    im = cv.new()
    frame = (32, 34, 40)
    for side in ("L", "R"):
        c = EYE_C[side]
        lens = path((c[0] - 78, c[1] - 52), ((c[0] - 20, c[1] - 60), (c[0] + 20, c[1] - 60), (c[0] + 78, c[1] - 52)),
                    ((c[0] + 82, c[1] + 10), (c[0] + 70, c[1] + 60), (c[0], c[1] + 62)),
                    ((c[0] - 70, c[1] + 60), (c[0] - 82, c[1] + 10), (c[0] - 78, c[1] - 52)))
        m = cv.mask(lens)
        lay = Image.new("RGBA", im.size, (60, 70, 95, 0)); lay.putalpha(m.point(lambda v: int(v * 0.26)))
        im.alpha_composite(lay)
        soft(cv, im, cv.ell_mask((c[0] - 30, c[1] - 30), 30, 14, -20), (255, 255, 255), 0.5, 4, clip=m)
        stroke(cv, im, lens + [lens[0]], frame, 13)
    stroke(cv, im, bez((480, 470), (490, 458), (510, 458), (520, 470)), frame, 13)
    for sx in (-1, 1):
        stroke(cv, im, [(500 + sx * 245, 480), (500 + sx * 262, 470)], frame, 12)
    return cv.finish(im)


def draw_prop(cv, which):
    """Held props, drawn at the RIGHT wrist in the rest pose (they ride on the hand)."""
    im = cv.new()
    wx, wy = WRIST["R"]
    if which == "popcorn":
        x, y = wx, wy + 40
        box = [(x - 70, y - 120), (x + 70, y - 120), (x + 52, y + 60), (x - 52, y + 60)]
        m = cv.mask(box)
        fill(cv, im, m, (232, 62, 58), shade=(190, 40, 40))
        for k in range(-2, 3):
            stripe = [(x + 28 * k - 8, y - 120), (x + 28 * k + 8, y - 120), (x + 21 * k + 6, y + 60), (x + 21 * k - 6, y + 60)]
            fill(cv, im, Image.fromarray(np.minimum(np.asarray(cv.mask(stripe)), np.asarray(m))), (250, 248, 240))
        for k in range(9):
            px, py = x - 64 + 16 * k, y - 132 - 16 * (k % 3)
            fill(cv, im, cv.ell_mask((px, py), 22, 19), (255, 240, 190), shade=(236, 200, 120), outline=(210, 170, 90), ow=2)
    elif which == "3d_glasses":
        x, y = wx, wy + 10
        for sx in (-1, 1):
            lens = cv.ell_mask((x + sx * 52, y), 46, 34)
            lay = Image.new("RGBA", im.size, (60, 70, 95, 0)); lay.putalpha(lens.point(lambda v: int(v * 0.6)))
            im.alpha_composite(lay)
            stroke(cv, im, ellipse_pts((x + sx * 52, y), 46, 34), (32, 34, 40), 11)
        stroke(cv, im, [(x - 8, y - 6), (x + 8, y - 6)], (32, 34, 40), 10)
    return cv.finish(im)


def draw_back(cv, d):
    """Seen from behind (audience in the cinema, over-the-shoulder): hair, head and jacket back in one layer."""
    im = cv.new()
    hair = hexrgb(d["hair"])
    fill(cv, im, cv.mask(path((400, 762), ((450, 744), (550, 744), (600, 762)), ((700, 800), (700, 1000), (690, 1196)),
                              ((560, 1206), (440, 1206), (310, 1196)), ((300, 1000), (300, 800), (400, 762)))),
         hexrgb(d["top"]), shade=hexrgb(d["top_shadow"]), outline=mix(hexrgb(d["top_shadow"]), (0, 0, 0), 0.35))
    fill(cv, im, cv.mask(head_shape()), hexrgb(d["skin_shadow"]))
    for sx in (-1, 1):
        fill(cv, im, cv.ell_mask((500 + sx * 238, 510), 36, 50), hexrgb(d["skin_shadow"]))
    if d["hair_style"] == "wavy_half_up":
        back = path((228, 470), ((200, 250), (360, 190), (500, 190)), ((640, 190), (800, 250), (772, 470)),
                    ((810, 700), (800, 900), (740, 1000)), ((600, 1040), (400, 1040), (260, 1000)),
                    ((200, 900), (190, 700), (228, 470)))
    else:
        back = path((236, 560), ((210, 330), (330, 200), (500, 196)), ((670, 200), (790, 330), (764, 560)),
                    ((700, 640), (300, 640), (236, 560)))
    fill(cv, im, cv.mask(back), hair, shade=mix(hair, (0, 0, 0), 0.3), outline=mix(hair, (0, 0, 0), 0.4))
    if d["hair_style"] == "wavy_half_up":
        fill(cv, im, cv.ell_mask((520, 260), 80, 66), mix(hair, (255, 255, 255), 0.05), outline=mix(hair, (0, 0, 0), 0.4))
    return cv.finish(im)


EYE_VARIANTS = ("open", "half", "wide", "smile", "closed", "happy")
BROW_VARIANTS = tuple(BROWS)


def layers(cid, res=1.6):
    """Every layer of one character: {name: RGBA image (full canvas at `res` px per design unit)}."""
    d = DESIGNS[cid]
    cv = Canvas(res)
    out = {"hair_back": draw_hair(cv, d, "back"), "legs": draw_legs(cv, d), "torso": draw_torso(cv, d),
           "neck": draw_neck(cv, d), "head": draw_head(cv, d), "cheeks": draw_cheeks(cv, d), "nose": draw_nose(cv, d),
           "hair_front": draw_hair(cv, d, "front"), "glasses": draw_glasses(cv, d), "back": draw_back(cv, d)}
    for side in ("L", "R"):
        out[f"upper_arm_{side}"] = draw_upper_arm(cv, d, side)
        out[f"forearm_{side}"] = draw_forearm(cv, d, side)
        for h in HANDS:
            out[f"hand_{side}_{h}"] = draw_hand(cv, d, side, h)
        out[f"iris_{side}"] = draw_iris(cv, d, side)
        out[f"iris_{side}_small"] = draw_iris(cv, d, side, small=True)
        for v in EYE_VARIANTS:
            if v in ("open", "half", "wide", "smile"):
                out[f"eye_{side}_{v}"] = draw_eye(cv, d, side, v, "white")
                out[f"lid_{side}_{v}"] = draw_eye(cv, d, side, v, "lids")
            else:
                out[f"eye_{side}_{v}"] = draw_eye(cv, d, side, v)
            if v in ("open", "half", "wide", "smile"):
                m = eye_mask(cv, side, v)
                out[f"eyemask_{side}_{v}"] = Image.merge("RGBA", (m, m, m, m))
    for v in BROW_VARIANTS:
        out[f"brows_{v}"] = draw_brows(cv, d, v)
    for v in MOUTHS:
        out[f"mouth_{v}"] = draw_mouth(cv, d, v)
    for p in ("popcorn", "3d_glasses"):
        out[f"prop_{p}"] = draw_prop(cv, p)
    return out

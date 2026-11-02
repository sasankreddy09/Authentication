"""Drawn scenes (no AI, no GPU): app screens, a "journey" travel metaphor, and title/recap cards.

Why: AI video models garble text and can't show UI or abstract processes; flowcharts bore non-technical viewers
(the backend video was 13/17 box-and-arrow diagrams). These recipes show the same ideas the way people meet
them in real life: a phone screen, a parcel/letter travelling between places, a calendar, a short card.

All timings in a spec are FRACTIONS of the shot (0..1), so a spec works whatever the voice length is.
Pure Pillow + numpy; frames are piped to ffmpeg by encode(). Unit-tested in tests/test_drawkit.py.

    render(kind, spec, plan, n_frames, fps, w, h) -> iterator of RGB frames (numpy uint8, h x w x 3)
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

THEME = {"bg_top": (17, 34, 64), "bg_bottom": (28, 64, 92), "card": (250, 250, 252), "ink": (24, 30, 42),
         "muted": (110, 118, 132), "accent": (25, 195, 166), "accent2": (255, 107, 107), "gold": (255, 196, 61),
         "line": (226, 230, 236), "good": (46, 170, 90), "white": (255, 255, 255), "shadow": (0, 0, 0, 90)}
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans{b}.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans{lb}.ttf"]
_fonts = {}
SCRIPT_FONT = {}        # {"regular": path, "bold": path}: set for non-Latin videos (e.g. Telugu → Hind Guntur)


def set_script_font(regular, bold=None):
    """Draw ALL text (cards, drawn scenes, diagrams, titles) with one font that has the video's script AND Latin
    letters (Telugu narration keeps PNR, WL 12, ₹ ...). Pillow shapes it with raqm (Telugu conjuncts/vowel signs)."""
    SCRIPT_FONT.clear()
    SCRIPT_FONT.update(regular=str(regular), bold=str(bold or regular))
    _fonts.clear()


def font_file(name):
    """Path of a DejaVu font file (e.g. 'DejaVuSans-Bold.ttf'): the system copy, else the copy matplotlib ships
    (matplotlib is preinstalled on Colab), else None. Never fails. With a script font set, that font instead."""
    import os
    if SCRIPT_FONT:
        return SCRIPT_FONT["bold" if "Bold" in name else "regular"]
    cands = [f"/usr/share/fonts/truetype/dejavu/{name}"]
    try:
        import matplotlib
        cands.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", name))
    except Exception:
        pass
    return next((c for c in cands if os.path.exists(c)), None)


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        f = None
        mpl = font_file("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        for p in ([mpl] if SCRIPT_FONT else FONT_PATHS + ([mpl] if mpl else [])):
            try:
                f = ImageFont.truetype(p.format(b="-Bold" if bold else "", lb="-Bold" if bold else "-Regular"), size)
                break
            except OSError:
                continue
        _fonts[k] = f or ImageFont.load_default(size=size)
    return _fonts[k]


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def prog(t, a, b):
    """0 before a, 1 after b, eased in between (all in shot fractions)."""
    return ease((t - a) / max(b - a, 1e-6))


def rgb(c):
    if isinstance(c, str):
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(c)


def theme(plan):
    th = dict(THEME, currency="$")
    for k, v in (plan.get("look") or {}).items():
        th[k] = rgb(v) if isinstance(v, str) and v.startswith("#") else v     # colours, plus e.g. "currency": "₹"
    return th


def background(w, h, th):
    y = np.linspace(0, 1, h)[:, None, None]
    top, bot = np.array(th["bg_top"], float), np.array(th["bg_bottom"], float)
    img = (top * (1 - y) + bot * y) * np.ones((1, w, 1))
    yy, xx = np.mgrid[0:h, 0:w]                                  # soft vignette light in the middle
    glow = np.exp(-(((xx - w * 0.5) / (w * 0.6)) ** 2 + ((yy - h * 0.45) / (h * 0.7)) ** 2))[..., None] * 18
    return Image.fromarray(np.clip(img + glow, 0, 255).astype(np.uint8))


_shadow_cache = {}
def shadowed(base, box, radius, fill, blur=18, offset=(0, 10)):
    """Rounded rectangle with a soft drop shadow, drawn onto base (RGBA). The blur is done on a small patch
    around the box and cached by size (blurring the full frame made drawn shots ~10x slower)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    m = blur * 3
    key = (x1 - x0, y1 - y0, radius, blur)
    if key not in _shadow_cache:
        sh = Image.new("RGBA", (x1 - x0 + 2 * m, y1 - y0 + 2 * m), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle([m, m, m + x1 - x0, m + y1 - y0], radius, fill=(0, 0, 0, 110))
        _shadow_cache[key] = sh.filter(ImageFilter.GaussianBlur(blur))
    pos = (x0 - m + offset[0], y0 - m + offset[1])
    patch = _shadow_cache[key]
    cx0, cy0 = max(pos[0], 0), max(pos[1], 0)                     # clip to the frame
    cx1, cy1 = min(pos[0] + patch.width, base.width), min(pos[1] + patch.height, base.height)
    if cx1 > cx0 and cy1 > cy0:
        base.alpha_composite(patch.crop((cx0 - pos[0], cy0 - pos[1], cx1 - pos[0], cy1 - pos[1])), (cx0, cy0))
    ImageDraw.Draw(base).rounded_rectangle([x0, y0, x1, y1], radius, fill=fill)


def text_fit(d, text, max_w, size, bold=False, min_size=18):
    while size > min_size and d.textlength(text, font=font(size, bold)) > max_w:
        size -= 2
    return font(size, bold)


def wrap(d, text, f, max_w):
    lines, cur = [], ""
    for wd in text.split():
        t = (cur + " " + wd).strip()
        if cur and d.textlength(t, font=f) > max_w:
            lines.append(cur); cur = wd
        else:
            cur = t
    return lines + ([cur] if cur else [])


# ---------------------------------------------------------------- icons (simple, friendly, consistent)
def icon(d, kind, cx, cy, s, th, active=True, on_light=False):
    """Flat icon centred at (cx, cy), size s. kinds: phone, store, warehouse, truck, bank, card, network,
    processor, person, box, money, clock, calendar, check, question."""
    col = th["accent"] if active else (150, 160, 175)
    ink, white = th["ink"], (th["white"] if not on_light else (196, 214, 226))
    r = s / 2
    if kind == "phone":
        d.rounded_rectangle([cx - r * .5, cy - r * .9, cx + r * .5, cy + r * .9], r * .12, fill=ink, outline=col, width=4)
        d.rounded_rectangle([cx - r * .4, cy - r * .72, cx + r * .4, cy + r * .66], r * .06, fill=col)
    elif kind in ("store", "shop"):
        d.rectangle([cx - r * .8, cy - r * .2, cx + r * .8, cy + r * .8], fill=white)
        for i in range(5):                                       # striped awning
            x0 = cx - r * .9 + i * r * .36
            d.rectangle([x0, cy - r * .55, x0 + r * .36, cy - r * .15], fill=col if i % 2 == 0 else white)
        d.rectangle([cx - r * .2, cy + r * .2, cx + r * .2, cy + r * .8], fill=col)
    elif kind in ("warehouse", "box"):
        if kind == "warehouse":
            d.polygon([(cx - r * .9, cy - r * .1), (cx, cy - r * .7), (cx + r * .9, cy - r * .1)], fill=col)
            d.rectangle([cx - r * .8, cy - r * .1, cx + r * .8, cy + r * .8], fill=white)
            d.rectangle([cx - r * .45, cy + r * .2, cx + r * .45, cy + r * .8], fill=(200, 205, 215))
        else:
            d.rectangle([cx - r * .7, cy - r * .5, cx + r * .7, cy + r * .7], fill=(214, 168, 112))
            d.rectangle([cx - r * .12, cy - r * .5, cx + r * .12, cy + r * .7], fill=(236, 205, 160))
    elif kind == "truck":
        d.rectangle([cx - r * .9, cy - r * .5, cx + r * .25, cy + r * .4], fill=white)
        d.polygon([(cx + r * .25, cy - r * .2), (cx + r * .7, cy - r * .2), (cx + r * .9, cy + r * .1),
                   (cx + r * .9, cy + r * .4), (cx + r * .25, cy + r * .4)], fill=col)
        for x in (cx - r * .55, cx + r * .55):
            d.ellipse([x - r * .18, cy + r * .3, x + r * .18, cy + r * .66], fill=ink)
    elif kind == "bank":
        d.polygon([(cx - r * .95, cy - r * .35), (cx, cy - r * .9), (cx + r * .95, cy - r * .35)], fill=col)
        for i in range(4):
            x = cx - r * .65 + i * r * .43
            d.rectangle([x - r * .1, cy - r * .3, x + r * .1, cy + r * .55], fill=white)
        d.rectangle([cx - r * .95, cy + r * .55, cx + r * .95, cy + r * .75], fill=col)
    elif kind == "card":
        d.rounded_rectangle([cx - r * .9, cy - r * .55, cx + r * .9, cy + r * .55], r * .12, fill=col)
        d.rectangle([cx - r * .9, cy - r * .3, cx + r * .9, cy - r * .1], fill=ink)
        d.rounded_rectangle([cx - r * .7, cy + r * .1, cx - r * .25, cy + r * .35], r * .05, fill=th["gold"])
    elif kind == "network":
        d.ellipse([cx - r * .85, cy - r * .85, cx + r * .85, cy + r * .85], outline=col, width=6)
        d.ellipse([cx - r * .4, cy - r * .85, cx + r * .4, cy + r * .85], outline=col, width=4)
        d.line([cx - r * .85, cy, cx + r * .85, cy], fill=col, width=4)
        for y in (cy - r * .45, cy + r * .45):
            d.line([cx - r * .72, y, cx + r * .72, y], fill=col, width=3)
    elif kind == "processor":
        d.rounded_rectangle([cx - r * .65, cy - r * .65, cx + r * .65, cy + r * .65], r * .15, fill=white)
        for i in range(-2, 3):                                   # chip pins
            o = i * r * .25
            for a, b in (((cx + o, cy - r * .65), (cx + o, cy - r * .9)), ((cx + o, cy + r * .65), (cx + o, cy + r * .9)),
                         ((cx - r * .65, cy + o), (cx - r * .9, cy + o)), ((cx + r * .65, cy + o), (cx + r * .9, cy + o))):
                d.line([a, b], fill=col, width=5)
        d.rounded_rectangle([cx - r * .3, cy - r * .3, cx + r * .3, cy + r * .3], r * .08, fill=col)
    elif kind == "person":
        d.ellipse([cx - r * .32, cy - r * .85, cx + r * .32, cy - r * .2], fill=col)
        d.pieslice([cx - r * .7, cy - r * .1, cx + r * .7, cy + r * 1.2], 180, 360, fill=col)
    elif kind == "money":
        d.rounded_rectangle([cx - r * .9, cy - r * .5, cx + r * .9, cy + r * .5], r * .1, fill=th["good"])
        d.ellipse([cx - r * .3, cy - r * .3, cx + r * .3, cy + r * .3], outline=white, width=5)
        f = font(int(r * .5), True)
        d.text((cx, cy), th.get("currency", "$"), font=f, fill=white, anchor="mm")
    elif kind == "clock":
        d.ellipse([cx - r * .85, cy - r * .85, cx + r * .85, cy + r * .85], fill=white, outline=col, width=8)
        d.line([cx, cy, cx, cy - r * .55], fill=ink, width=7)
        d.line([cx, cy, cx + r * .4, cy + r * .1], fill=ink, width=7)
    elif kind == "check":
        d.ellipse([cx - r * .85, cy - r * .85, cx + r * .85, cy + r * .85], fill=th["good"])
        d.line([(cx - r * .4, cy), (cx - r * .1, cy + r * .32), (cx + r * .45, cy - r * .35)], fill=white, width=int(r * .18))
    elif kind == "question":
        d.ellipse([cx - r * .85, cy - r * .85, cx + r * .85, cy + r * .85], fill=th["accent2"])
        d.text((cx, cy + r * .05), "?", font=font(int(r * 1.2), True), fill=white, anchor="mm")
    elif kind == "calendar":
        d.rounded_rectangle([cx - r * .8, cy - r * .7, cx + r * .8, cy + r * .8], r * .12, fill=white)
        d.rectangle([cx - r * .8, cy - r * .7, cx + r * .8, cy - r * .3], fill=th["accent2"])
    else:
        d.ellipse([cx - r * .6, cy - r * .6, cx + r * .6, cy + r * .6], fill=col)


# ---------------------------------------------------------------- UI recipe: a phone screen
PHONE = {"w": 500, "h": 820, "top": 36}   # ends at y=856: the subtitles below (y≈900–1040) never cover it


def _widget_height(d, wdg, inner_w):
    k = wdg["type"]
    if k == "header":
        return 96
    if k == "product":
        return 150
    if k == "button":
        return 92
    if k == "choice":
        return 66 * len(wdg["options"]) + 20
    if k == "status":
        return 74 * len(wdg["steps"]) + 20
    if k == "statement":
        return 64 * len(wdg["rows"]) + 70
    if k == "notice":
        return 24 + 40 * len(wrap(d, wdg["text"], font(26), inner_w - 90))
    if k == "text":
        return 16 + int(wdg.get("size", 28) * 1.35) * len(wrap(d, wdg["text"], font(wdg.get("size", 28), wdg.get("bold", False)), inner_w))
    return 40


def _draw_widget(img, d, wdg, x, y, iw, t, ev, th):
    """Draws one widget at (x, y) width iw; ev = {"show": p, "tap": p, "status": value, "hl": p} at time t."""
    k = wdg["type"]
    ink, mut = th["ink"], th["muted"]
    if k == "header":
        d.rectangle([x - 24, y - 6, x + iw + 24, y + 90], fill=th["white"])
        d.text((x, y + 42), wdg["text"], font=text_fit(d, wdg["text"], iw - 60, 36, True), fill=ink, anchor="lm")
        if wdg.get("back", True):
            d.text((x + iw, y + 42), "⋮", font=font(34, True), fill=mut, anchor="rm")
        d.line([x - 24, y + 90, x + iw + 24, y + 90], fill=th["line"], width=2)
    elif k == "product":
        d.rounded_rectangle([x, y + 10, x + 120, y + 130], 18, fill=(236, 240, 246))
        icon(d, wdg.get("icon", "box"), x + 60, y + 70, 92, th, on_light=True)
        d.text((x + 140, y + 30), wdg["name"], font=text_fit(d, wdg["name"], iw - 150, 30, True), fill=ink)
        if wdg.get("detail"):
            d.text((x + 140, y + 74), wdg["detail"], font=text_fit(d, wdg["detail"], iw - 150, 24), fill=mut)
        if wdg.get("price"):
            d.text((x + 140, y + 108), wdg["price"], font=font(28, True), fill=ink)
    elif k == "button":
        tap = ev.get("tap", 0.0)
        primary = wdg.get("style", "primary") == "primary"
        base = th["accent"] if primary else th["white"]
        pressed = 1 - abs(2 * tap - 1) if 0 < tap < 1 else 0
        fill = tuple(int(c * (1 - 0.18 * pressed)) for c in base)
        d.rounded_rectangle([x, y + 8, x + iw, y + 80], 36, fill=fill, outline=None if primary else th["accent"], width=3)
        d.text((x + iw / 2, y + 44), wdg["text"], font=text_fit(d, wdg["text"], iw - 40, 30, True),
               fill=th["white"] if primary else th["accent"], anchor="mm")
        if 0 < tap < 1:                                            # finger-tap ripple
            rr = 20 + 120 * tap
            ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(ov).ellipse([x + iw / 2 - rr, y + 44 - rr, x + iw / 2 + rr, y + 44 + rr],
                                       fill=(255, 255, 255, int(150 * (1 - tap))))
            img.alpha_composite(ov)
    elif k == "choice":
        sel = wdg.get("selected")
        p = ev.get("hl", 1.0 if sel is not None else 0.0)
        for i, opt in enumerate(wdg["options"]):
            yy = y + 10 + 66 * i
            on = i == sel and p > 0.5
            d.rounded_rectangle([x, yy, x + iw, yy + 56], 14, fill=(232, 250, 246) if on else th["white"],
                                outline=th["accent"] if on else th["line"], width=3)
            d.ellipse([x + 16, yy + 16, x + 40, yy + 40], outline=th["accent"] if on else mut, width=3)
            if on:
                d.ellipse([x + 22, yy + 22, x + 34, yy + 34], fill=th["accent"])
            d.text((x + 56, yy + 28), opt, font=text_fit(d, opt, iw - 70, 26), fill=ink, anchor="lm")
    elif k == "status":
        cur = ev.get("status", wdg.get("current", 0))              # float: 1.5 = halfway from step 1 to 2
        for i, st in enumerate(wdg["steps"]):
            yy = y + 20 + 74 * i
            done = cur >= i
            if i < len(wdg["steps"]) - 1:                          # connector line, fills as status advances
                d.line([x + 22, yy + 22, x + 22, yy + 96], fill=th["line"], width=6)
                fillp = min(max(cur - i, 0), 1)
                if fillp > 0:
                    d.line([x + 22, yy + 22, x + 22, yy + 22 + 74 * fillp], fill=th["good"], width=6)
            d.ellipse([x + 6, yy + 6, x + 38, yy + 38], fill=th["good"] if done else th["white"],
                      outline=th["good"] if done else th["line"], width=4)
            if done:
                d.line([(x + 14, yy + 22), (x + 20, yy + 29), (x + 31, yy + 15)], fill=th["white"], width=4)
            d.text((x + 56, yy + 22), st, font=text_fit(d, st, iw - 60, 27, done and abs(cur - i) < 0.5),
                   fill=ink if done else mut, anchor="lm")
    elif k == "statement":
        d.text((x, y + 24), wdg.get("title", "Recent activity"), font=font(26, True), fill=mut, anchor="lm")
        for i, row in enumerate(wdg["rows"]):
            yy = y + 56 + 64 * i
            show = ev.get(f"row{i}", 1.0)
            if show <= 0:
                continue
            hl = row.get("highlight") and show > 0
            if hl:
                d.rounded_rectangle([x - 10, yy, x + iw + 10, yy + 58], 12, fill=(232, 250, 240))
            d.text((x, yy + 18), row.get("date", ""), font=font(20), fill=mut, anchor="lm")
            d.text((x, yy + 42), row["text"], font=text_fit(d, row["text"], iw - 170, 24, hl), fill=ink, anchor="lm")
            amt = row.get("amount", "")
            d.text((x + iw, yy + 30), amt, font=font(26, True),
                   fill=th["good"] if amt.startswith("+") else ink, anchor="rm")
    elif k == "notice":
        lines = wrap(d, wdg["text"], font(26), iw - 90)
        hgt = 24 + 40 * len(lines)
        d.rounded_rectangle([x, y, x + iw, y + hgt], 16, fill=(240, 247, 255))
        icon(d, wdg.get("icon", "clock"), x + 36, y + hgt / 2, 44, th, on_light=True)
        for i, ln in enumerate(lines):
            d.text((x + 72, y + 32 + 40 * i), ln, font=font(26), fill=ink, anchor="lm")
    elif k == "text":
        f = font(wdg.get("size", 28), wdg.get("bold", False))
        for i, ln in enumerate(wrap(d, wdg["text"], f, iw)):
            d.text((x, y + 8 + i * int(wdg.get("size", 28) * 1.35)), ln, font=f, fill=ink if wdg.get("bold") else mut)


def _ui_events(spec, t):
    """State of every widget at time t (fraction of the shot) from the spec's timeline."""
    st = {}
    for e in spec.get("events", []):
        a = e["t"]
        if e["do"] == "tap":
            st.setdefault(e["target"], {})["tap"] = min(max((t - a) / 0.08, 0), 1) if t >= a else 0
        elif e["do"] == "show":
            st.setdefault(e["target"], {})["show"] = prog(t, a, a + 0.08)
        elif e["do"] == "select":
            st.setdefault(e["target"], {})["hl"] = 1.0 if t >= a else 0.0
        elif e["do"] == "status":
            prev = st.get(e["target"], {}).get("status", e.get("from", 0))
            st.setdefault(e["target"], {})["status"] = prev + (e["to"] - prev) * prog(t, a, a + e.get("dur", 0.15))
        elif e["do"] == "row":
            st.setdefault(e["target"], {})[f"row{e['row']}"] = prog(t, a, a + 0.06)
    return st


def ui_frame(spec, plan, t, w, h, bg, th):
    """One frame of a UI shot: a phone with the current screen, optional notification, optional caption."""
    img = bg.copy().convert("RGBA")
    d = ImageDraw.Draw(img)
    pw, ph, top = PHONE["w"], PHONE["h"], PHONE["top"]
    # a step label sits top-left: move the phone right so they never overlap (the caption then goes on the left)
    off = spec.get("offset_x", 300 if spec.get("_has_label") else 0)
    px = w // 2 - pw // 2 + off
    shadowed(img, [px - 18, top - 18, px + pw + 18, top + ph + 18], 64, (14, 16, 22, 255))
    d = ImageDraw.Draw(img)
    screens = spec["screens"]
    # which screen is showing: "screen" events switch with a short slide
    cur, nxt, sp = 0, None, 0.0
    for e in spec.get("events", []):
        if e["do"] == "screen":
            if t >= e["t"] + 0.07:
                cur = e["to"]
            elif t >= e["t"]:
                nxt, sp = e["to"], prog(t, e["t"], e["t"] + 0.07)
    ev = _ui_events(spec, t)
    scr = Image.new("RGBA", (pw, ph), th["card"] + (255,))
    for idx, slide in ([(cur, 0)] if nxt is None else [(cur, -sp * pw), (nxt, (1 - sp) * pw)]):
        layer = Image.new("RGBA", (pw, ph), th["card"] + (255,))
        ld = ImageDraw.Draw(layer)
        ld.text((32, 22), spec.get("clock", "9:41"), font=font(22, True), fill=th["ink"], anchor="lm")
        ld.text((pw - 32, 22), "●●● ▮", font=font(18), fill=th["ink"], anchor="rm")
        y, iw = 44, pw - 64
        for wdg in screens[idx]:
            e = ev.get(wdg.get("id"), {})
            show = e.get("show", 0.0 if wdg.get("hidden") else 1.0)
            hgt = _widget_height(ld, wdg, iw)
            if show >= 1:
                _draw_widget(layer, ld, wdg, 32, y, iw, t, e, th)
            elif show > 0:
                wl = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
                _draw_widget(wl, ImageDraw.Draw(wl), wdg, 32, y + int(24 * (1 - show)), iw, t, e, th)
                a = np.array(wl); a[..., 3] = (a[..., 3] * show).astype(np.uint8)
                layer.alpha_composite(Image.fromarray(a))
                ld = ImageDraw.Draw(layer)
            y += hgt + 18
        scr.alpha_composite(layer, (int(slide), 0))
    mask = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, pw - 1, ph - 1], 48, fill=255)
    img.paste(scr, (px, top), mask)
    for e in spec.get("events", []):                            # notification banner slides down over the screen
        if e["do"] == "notify":
            p = prog(t, e["t"], e["t"] + 0.06) * (1 - prog(t, e.get("until", 2.0), e.get("until", 2.0) + 0.06))
            if p > 0:
                bh = 150
                by = top + 14 - int((1 - p) * (bh + 30))
                shadowed(img, [px + 16, by, px + pw - 16, by + bh], 26, (255, 255, 255, 255), blur=12, offset=(0, 6))
                d = ImageDraw.Draw(img)
                icon(d, e.get("icon", "bank"), px + 66, by + 52, 58, th, on_light=True)
                d.text((px + 112, by + 36), e["title"], font=text_fit(d, e["title"], pw - 150, 26, True), fill=th["ink"], anchor="lm")
                for i, ln in enumerate(wrap(d, e["body"], font(23), pw - 150)[:3]):
                    d.text((px + 112, by + 74 + 30 * i), ln, font=font(23), fill=th["muted"], anchor="lm")
    if spec.get("caption"):
        _caption(img, spec["caption"], t, w, h, th, side="right" if off <= 0 else "left")
    return np.asarray(img.convert("RGB"))


def _caption(img, cap, t, w, h, th, side="right"):
    """Short callout next to the phone (≤ 6 words): what the viewer should notice."""
    p = prog(t, cap.get("t", 0.1), cap.get("t", 0.1) + 0.08)
    if p <= 0:
        return
    d = ImageDraw.Draw(img)
    f = font(46, True)
    lines = wrap(d, cap["text"], f, 520)
    x = w // 2 + PHONE["w"] // 2 + 80 if side == "right" else w // 2 - PHONE["w"] // 2 - 600
    y = cap.get("y", 380)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for i, ln in enumerate(lines):
        ld.text((x + int(30 * (1 - p)), y + 62 * i), ln, font=f, fill=th["white"] + (int(255 * p),))
    ld.rectangle([x, y - 26, x + 90 * p, y - 18], fill=th["gold"] + (int(255 * p),))
    img.alpha_composite(layer)


# ---------------------------------------------------------------- journey recipe: travel metaphor
def journey_layout(stations, w):
    n = len(stations)
    xs = [int(w * (0.12 + 0.76 * i / max(n - 1, 1))) for i in range(n)]
    return {s["id"]: (x, 500) for s, x in zip(stations, xs)}


_static = {}
def journey_static(spec, plan, w, bg, th):
    """Road, station tiles and labels: identical in every frame of a shot, so drawn once."""
    key = (id(spec), id(bg))
    if key in _static:
        return _static[key]
    J = plan["journey"]
    stations = J["stations"]
    pos = journey_layout(stations, w)
    img = bg.copy().convert("RGBA")
    d = ImageDraw.Draw(img)
    path = spec.get("path", [])
    # road
    xs = [p[0] for p in pos.values()]
    d.rounded_rectangle([min(xs) - 40, 488, max(xs) + 40, 512], 12, fill=(255, 255, 255, 40))
    for x in range(min(xs), max(xs), 46):
        d.line([x, 500, x + 20, 500], fill=(255, 255, 255, 120), width=4)
    # which stations matter in this shot
    focus = set(spec.get("focus", path))
    for s in stations:
        x, y = pos[s["id"]]
        on = s["id"] in focus
        r = 104 if on else 84
        shadowed(img, [x - r, y - r - 120, x + r, y + r - 120], 40, (255, 255, 255, 255) if on else (205, 212, 224, 255),
                 blur=14, offset=(0, 8))
        d = ImageDraw.Draw(img)
        icon(d, s.get("icon", "store"), x, y - 120, r * 1.25, th, active=on, on_light=True)
        lab = s["label"]
        f = text_fit(d, lab, 300, 34 if on else 28, on)
        for i, ln in enumerate(wrap(d, lab, f, 280)):
            d.text((x, y + 70 + 40 * i), ln, font=f, fill=th["white"] if on else (180, 190, 205), anchor="mm")
    _static.clear()
    _static[key] = (img, pos)
    return img, pos


def journey_frame(spec, plan, t, w, h, bg, th):
    """Stations in a row (same for every journey shot = consistent), the money/parcel token travelling along
    spec["path"], optional caption and business-day calendar."""
    J = plan["journey"]
    static, pos = journey_static(spec, plan, w, bg, th)
    img = static.copy()
    d = ImageDraw.Draw(img)
    path = spec.get("path", [])
    t0, t1 = spec.get("move", [0.1, 0.85])
    # the token travelling along the path
    if len(path) >= 2:
        seg = len(path) - 1
        p = prog(t, t0, t1) * seg
        i = min(int(p), seg - 1)
        f = p - i
        (ax, ay), (bx, by) = pos[path[i]], pos[path[i + 1]]
        tx, ty = ax + (bx - ax) * f, ay + (by - ay) * f - 10 - 26 * math.sin(math.pi * f)
    elif path:
        tx, ty = pos[path[0]][0], pos[path[0]][1] - 10
    else:
        tx = ty = None
    if tx is not None:
        tok = spec.get("token", J.get("token", "money"))
        shadowed(img, [tx - 62, ty - 40, tx + 62, ty + 40], 14, (255, 255, 255, 255), blur=10, offset=(0, 6))
        d = ImageDraw.Draw(img)
        icon(d, tok, tx, ty, 108, th, on_light=True)
    # caption
    if spec.get("caption"):
        p = prog(t, 0.03, 0.12)
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        f = text_fit(ld, spec["caption"], w - 400, 54, True)
        ld.text((w // 2, 215 + int(20 * (1 - p))), spec["caption"], font=f, fill=th["white"] + (int(255 * p),), anchor="mm")
        img.alpha_composite(layer)
    # business-day calendar: days tick over; weekends greyed and skipped
    cal = spec.get("calendar")
    if cal:
        _calendar(img, cal, t, w, h, th)
    return np.asarray(img.convert("RGB"))


DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _calendar(img, cal, t, w, h, th):
    d = ImageDraw.Draw(img)
    start = DAYS.index(cal.get("start", "Mon"))
    n = cal["days"]
    x0, y0, cw = w // 2 - 7 * 92 // 2, 700, 92
    shown = int(prog(t, cal.get("t0", 0.15), cal.get("t1", 0.9)) * n + 0.0001)
    biz = 0
    for i in range(min(n, 14)):
        dn = DAYS[(start + i) % 7]
        weekend = dn in ("Sat", "Sun")
        x = x0 + (i % 7) * cw
        y = y0 + (i // 7) * 110
        fill = (70, 82, 104) if weekend else (255, 255, 255)
        if i < shown and not weekend:
            fill = th["gold"]
        d.rounded_rectangle([x, y, x + cw - 12, y + 96], 14, fill=fill)
        d.text((x + (cw - 12) / 2, y + 30), dn, font=font(24, True), fill=th["ink"] if not weekend else (160, 170, 185), anchor="mm")
        if not weekend and i < shown:
            biz += 1
            d.text((x + (cw - 12) / 2, y + 66), str(biz), font=font(30, True), fill=th["ink"], anchor="mm")
        elif weekend:
            d.text((x + (cw - 12) / 2, y + 66), "closed", font=font(18), fill=(160, 170, 185), anchor="mm")
    if cal.get("label"):
        d.text((w // 2, y0 - 34), cal["label"], font=font(30, True), fill=th["white"], anchor="mm")


# ---------------------------------------------------------------- card recipe: hook question / recap
def card_frame(spec, plan, t, w, h, bg, th):
    img = bg.copy().convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    lines = spec.get("lines", [])
    reveal = spec.get("reveal", len(lines) > 1)
    y = h * 0.5 - (len(lines) * 92) / 2 + (40 if spec.get("icon") else 0)
    if spec.get("icon"):
        p = prog(t, 0.02, 0.1)
        icon(d, spec["icon"], w // 2, int(y - 120 - 20 * (1 - p)), int(150 * (0.6 + 0.4 * p)), th)
    for i, ln in enumerate(lines):
        a = (0.06 + 0.8 * i / max(len(lines), 1)) if reveal else 0.03
        p = prog(t, a, a + 0.08)
        if p <= 0:
            continue
        big = i == 0 and spec.get("headline", True)
        f = text_fit(d, ln, w - 300, 74 if big else 52, big)
        col = th["white"] if big else (225, 232, 242)
        d.text((w // 2, y + 92 * i + int(18 * (1 - p))), ln, font=f, fill=col + (int(255 * p),), anchor="mm")
        if spec.get("bullets") and not big:
            tw = d.textlength(ln, font=f)
            d.ellipse([w // 2 - tw / 2 - 40, y + 92 * i - 10, w // 2 - tw / 2 - 20, y + 92 * i + 10],
                      fill=th["gold"] + (int(255 * p),))
    img.alpha_composite(layer)
    return np.asarray(img.convert("RGB"))


RENDERERS = {"ui": ui_frame, "journey": journey_frame, "card": card_frame}


DESIGN_W, DESIGN_H = 1920, 1080                      # every layout is designed at 1080p, then scaled


def renderer(kind):
    """Frame function of a drawn kind. The two-friend kinds live in their own modules (loaded on first use)."""
    if kind not in RENDERERS:
        if kind == "character":
            import charanim
            RENDERERS[kind] = charanim.character_frame
        elif kind in ("demo", "photo"):
            import demos
            RENDERERS.update(demo=demos.demo_frame, photo=demos.photo_frame)
    return RENDERERS[kind]


def render(kind, spec, plan, n_frames, fps, w, h):
    """Yields n_frames RGB frames (h x w x 3, uint8) for a drawn shot.
    kind "story": spec["visuals"] is a list of sub-visuals, each {"recipe": kind, kind: spec, "frames": n}, drawn
    one after the other (the visual timeline of one dialogue scene; the audio is not affected)."""
    if kind == "story":
        done = 0
        for v in spec["visuals"]:
            n = min(int(v["frames"]), n_frames - done)
            if n > 0:
                yield from render(v["recipe"], dict(v[v["recipe"]], _n=n), plan, n, fps, w, h)
                done += n
        if done < n_frames:                              # never short: hold the last visual's final frame
            last = spec["visuals"][-1]
            f = list(render(last["recipe"], dict(last[last["recipe"]], _n=1), plan, 1, fps, w, h))[0]
            for _ in range(n_frames - done):
                yield f
        return
    th = theme(plan)
    bg = background(DESIGN_W, DESIGN_H, th)
    fn = renderer(kind)
    spec = dict(spec, _n=spec.get("_n", n_frames))
    native = kind == "character" and bool((plan or {}).get("visual_style_version"))   # storybook: drawn at w x h
    for i in range(n_frames):
        if native:
            yield fn(spec, plan, i / max(n_frames - 1, 1), w, h, bg, th)
            continue
        f = fn(spec, plan, i / max(n_frames - 1, 1), DESIGN_W, DESIGN_H, bg, th)
        if (w, h) != (DESIGN_W, DESIGN_H):
            f = np.asarray(Image.fromarray(f).resize((w, h), Image.LANCZOS))
        yield f


def check_spec(kind, spec, plan):
    """Problems with a drawn-shot spec (used by the plan validation). Returns a list of messages."""
    out = []
    if kind == "ui":
        if not spec.get("screens"):
            out.append("ui shot needs 'screens' (a list of screens, each a list of widgets)")
        ids = {w.get("id") for sc in spec.get("screens", []) for w in sc}
        for e in spec.get("events", []):
            if e["do"] in ("tap", "show", "select", "status", "row") and e.get("target") not in ids:
                out.append(f"event '{e['do']}' targets unknown widget '{e.get('target')}'")
            if not 0 <= e.get("t", -1) <= 1:
                out.append(f"event time t={e.get('t')} must be a fraction 0..1 of the shot")
            if e["do"] == "screen" and not 0 <= e.get("to", -1) < len(spec.get("screens", [])):
                out.append(f"screen event goes to missing screen {e.get('to')}")
        for sc in spec.get("screens", []):
            for wdg in sc:
                if wdg.get("type") not in ("header", "product", "button", "choice", "status", "statement", "notice", "text"):
                    out.append(f"unknown widget type '{wdg.get('type')}'")
    elif kind == "journey":
        st = {s["id"] for s in plan.get("journey", {}).get("stations", [])}
        if not st:
            out.append("journey shots need a plan-level 'journey': {'stations': [...]}")
        for sid in spec.get("path", []) + spec.get("focus", []):
            if sid not in st:
                out.append(f"journey path uses unknown station '{sid}'")
    elif kind == "character":
        import charanim
        out += charanim.check_spec(spec, plan)
    elif kind in ("demo", "photo"):
        import demos
        out += demos.check_spec(kind, spec, plan)
    elif kind == "story":
        if not spec.get("visuals"):
            out.append("story shot needs 'visuals'")
        for v in spec.get("visuals", []):
            r = v.get("recipe")
            if r not in ("character", "demo", "photo", "card", "ui", "journey"):
                out.append(f"story visual '{v.get('id')}': recipe '{r}' can't be drawn inside a scene")
            else:
                out += [f"visual {v.get('id')}: {m}" for m in check_spec(r, v.get(r, {}), plan)]
    elif kind == "card":
        if not spec.get("lines"):
            out.append("card shot needs 'lines'")
        for ln in spec.get("lines", []):
            if len(ln.split()) > 9:
                out.append(f"card line too long for the screen ({len(ln.split())} words): '{ln}'")
    return out

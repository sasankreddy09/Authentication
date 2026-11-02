"""Cinematic Environment Factory: reusable, versioned, layered places for the storybook characters.

An environment pack (one folder per version, never rewritten):
  <root>/environments/<env>/<version>/manifest.json   layers (file, depth, sha256), camera presets, lighting,
                                                      character slots, subtitle/diagram safe regions, provenance
  <root>/environments/<env>/<version>/layers/*.png    master-size RGBA layers, back to front

Each pack has one or more SETS (a viewpoint of the same place: e.g. the cinema seen from the screen toward the
seats, and from behind the seats toward the screen). A camera preset frames a set: a crop of the master, a
default movement, the character slots (position, scale, view, seated), and the parallax depth of each layer.
The same place is reused for every shot of a video and across videos: different framings, never a different
cinema per shot. Provisional packs are drawn by code here (status "provisional"); the Environment Studio can
replace them with approved generated art in the same format (each layer generated or cut separately — a
flattened image is never split into fake parallax layers).
"""
from __future__ import annotations

import datetime, hashlib, json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ENV_ART_VERSION = "storybook-env-procedural-1"
MASTER = (2400, 1350)                    # master size of every set (16:9; presets crop it)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _grad(w, h, top, bot, power=1.0):
    t = (np.linspace(0, 1, h) ** power)[:, None, None]
    a = np.array(top, np.float32)[None, None, :]
    b = np.array(bot, np.float32)[None, None, :]
    return Image.fromarray(np.repeat((a + (b - a) * t).astype(np.uint8), w, 1), "RGB").convert("RGBA")


def _blur_layer(im, r):
    return im.filter(ImageFilter.GaussianBlur(r))


def _glow(size, box, color, alpha, blur):
    g = Image.new("RGBA", size, color + (0,))
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).ellipse(box, fill=int(255 * alpha))
    g.putalpha(m.filter(ImageFilter.GaussianBlur(blur)))
    return g


def _seat_row(w, y, seat_w, seat_h, color, n_off=0.0, arm=True, back_shine=0.12):
    """One row of cinema seats (soft rounded backs with a light rim from the screen)."""
    im = Image.new("RGBA", (w, int(y + seat_h * 1.6)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pitch = seat_w * 1.14
    x = -pitch * n_off
    dark = tuple(int(c * 0.62) for c in color)
    while x < w + pitch:
        d.rounded_rectangle([x, y, x + seat_w, y + seat_h], radius=int(seat_w * 0.32), fill=color)
        d.rounded_rectangle([x + seat_w * 0.1, y + seat_h * 0.62, x + seat_w * 0.9, y + seat_h * 1.15],
                            radius=int(seat_w * 0.2), fill=dark)
        d.rounded_rectangle([x + seat_w * 0.12, y + seat_h * 0.06, x + seat_w * 0.88, y + seat_h * 0.16],
                            radius=int(seat_w * 0.1), fill=tuple(min(255, int(c * (1 + back_shine * 3))) for c in color))
        x += pitch
    return im


# ------------------------------------------------------------------------------------------- the packs
def draw_cinema():
    W, H = MASTER
    L = {}
    # SET "seats": we look from the screen toward the audience; the friends sit in a middle row
    back = _grad(W, H, (26, 20, 44), (54, 24, 40), 1.2)
    d = ImageDraw.Draw(back)
    for i in range(9):                                     # wall panels with warm sconces
        x = i * 290 - 60
        d.rounded_rectangle([x, 70, x + 250, 520], radius=30, fill=(40, 30, 58))
        back.alpha_composite(_glow((W, H), [x + 95, 120, x + 155, 180], (255, 196, 120), 0.9, 30))
    back.alpha_composite(_glow((W, H), [W * 0.1, -400, W * 0.9, 500], (120, 150, 255), 0.35, 160))    # projector haze
    beam = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(beam).polygon([(W * 0.46, 0), (W * 0.54, 0), (W * 0.95, H), (W * 0.05, H)], fill=(170, 190, 255, 22))
    back.alpha_composite(_blur_layer(beam, 20))
    L["seats_back"] = back
    rows = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for k, (y, sw, sh_, col) in enumerate([(430, 120, 150, (96, 28, 48)), (560, 150, 190, (120, 32, 54))]):
        r = _seat_row(W, y, sw, sh_, col, n_off=0.5 * k)
        rows.alpha_composite(_blur_layer(r, 3 - k), (0, 0))
    for x in range(40, W, 210):                            # a few audience heads in the back rows
        rows.alpha_composite(_glow((W, H), [x, 430, x + 95, 530], (16, 12, 24), 1.0, 2))
    L["seats_rows"] = rows
    own = Image.new("RGBA", (W, H), (0, 0, 0, 0))          # the friends' own row (behind them)
    own.alpha_composite(_seat_row(W, 720, 300, 330, (150, 38, 62), n_off=0.25, back_shine=0.16))
    L["seats_own_row"] = own
    fg = Image.new("RGBA", (W, H), (0, 0, 0, 0))           # the row in front of them: hides legs, gives depth
    fg.alpha_composite(_seat_row(W, 1110, 420, 380, (70, 18, 32), n_off=0.1, back_shine=0.05))
    L["seats_fg"] = _blur_layer(fg, 6)
    light = _glow((W, H), [W * 0.15, 300, W * 0.85, 1500], (140, 170, 255), 0.10, 220)   # screen light on them
    L["seats_light"] = light
    # SET "screen": from behind the friends toward the big screen (the movie is drawn live into `screen_rect`)
    sb = _grad(W, H, (10, 8, 20), (24, 14, 26))
    d = ImageDraw.Draw(sb)
    d.rectangle([0, 0, W, 120], fill=(36, 14, 30))
    for x in (0, W - 260):                                 # side curtains
        for k in range(6):
            d.rectangle([x + k * 44, 0, x + k * 44 + 30, H], fill=(110 - 6 * k, 18, 34))
    sb.alpha_composite(_glow((W, H), [300, 60, W - 300, 900], (170, 185, 255), 0.45, 120))
    L["screen_back"] = sb
    aud = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for r, (y, s) in enumerate([(930, 70), (1010, 92), (1110, 120)]):
        for i in range(int(W / (s * 1.25)) + 2):
            x = i * s * 1.25 - (r % 2) * s * 0.6
            aud.alpha_composite(_glow((W, H), [x, y, x + s, y + s * 1.2], (14, 10, 20), 1.0, 2))
    L["screen_audience"] = aud
    fgs = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fgs.alpha_composite(_seat_row(W, 1240, 460, 300, (40, 12, 22), back_shine=0.03))
    L["screen_fg"] = _blur_layer(fgs, 8)
    return L


def draw_lobby():
    W, H = MASTER
    L = {}
    back = _grad(W, H, (64, 32, 76), (130, 56, 78), 1.1)
    d = ImageDraw.Draw(back)
    d.rectangle([0, 960, W, H], fill=(84, 34, 46))
    for k in range(17):                                    # marquee bulbs
        back.alpha_composite(_glow((W, H), [60 + k * 140, 70, 92 + k * 140, 102], (255, 225, 150), 1.0, 6))
    for i, (x, col) in enumerate(((180, (60, 120, 200)), (640, (200, 80, 90)), (1560, (80, 160, 120)), (2020, (230, 160, 60)))):
        d.rounded_rectangle([x, 180, x + 300, 640], radius=14, fill=(250, 222, 130))
        d.rounded_rectangle([x + 22, 204, x + 278, 520], radius=8, fill=col)
        d.ellipse([x + 90, 280, x + 210, 400], fill=tuple(min(255, c + 60) for c in col))
        d.rectangle([x + 40, 545, x + 260, 562], fill=(110, 70, 50)); d.rectangle([x + 80, 585, x + 220, 598], fill=(110, 70, 50))
    for x in range(0, W, 160):                             # floor tiles (perspective lines)
        d.line([(W / 2 + (x - W / 2) * 0.55, 960), (x - (W / 2 - x) * 0.8, H)], fill=(96, 40, 52), width=3)
    back.alpha_composite(_glow((W, H), [700, 0, 1700, 700], (255, 200, 170), 0.25, 160))
    L["lobby_back"] = back
    fg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(fg)
    d.rounded_rectangle([W - 330, 620, W + 40, 1180], radius=28, fill=(200, 42, 56))     # popcorn stand (foreground)
    d.rounded_rectangle([W - 300, 560, W - 40, 660], radius=20, fill=(250, 220, 120))
    for k in range(9):
        fg.alpha_composite(_glow((W, H), [W - 290 + k * 28, 520 - (k % 3) * 18, W - 250 + k * 28, 566], (255, 240, 190), 1.0, 1))
    L["lobby_fg"] = _blur_layer(fg, 4)
    return L


def draw_studio():
    W, H = MASTER
    back = _grad(W, H, (48, 82, 130), (22, 44, 76))
    d = ImageDraw.Draw(back)
    d.rectangle([0, 980, W, H], fill=(30, 52, 86))
    back.alpha_composite(_glow((W, H), [600, 100, 1800, 900], (150, 200, 255), 0.35, 200))
    for x in range(120, W, 340):
        back.alpha_composite(_glow((W, H), [x, 160, x + 120, 280], (255, 255, 255), 0.18, 30))
    return {"studio_back": back}


def draw_space_observatory():
    W, H = MASTER
    rng = np.random.default_rng(7)
    sky = _grad(W, H, (6, 8, 26), (22, 14, 48))
    d = ImageDraw.Draw(sky)
    for _ in range(900):
        x, y, r = rng.uniform(0, W), rng.uniform(0, H * 0.8), rng.choice([1, 1, 1, 2, 3])
        b = int(rng.uniform(150, 255))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(b, b, min(255, b + 20), 255))
    sky.alpha_composite(_glow((W, H), [1300, 120, 2100, 600], (170, 90, 220), 0.55, 120))   # a distant galaxy
    sky.alpha_composite(_glow((W, H), [1550, 260, 1850, 420], (255, 220, 255), 0.8, 50))
    room = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(room)
    for i in range(5):                                     # the big window frame
        x = 120 + i * 540
        d.rectangle([x - 22, 0, x + 22, 980], fill=(40, 48, 78))
    d.arc([-200, -500, W + 200, 900], 200, 340, fill=(40, 48, 78), width=40)
    d.rectangle([0, 960, W, H], fill=(34, 40, 66))
    for x in range(0, W, 120):
        d.line([(x, 960), (x + (x - W / 2) * 0.6, H)], fill=(44, 52, 84), width=3)
    L = {"obs_sky": sky, "obs_room": room}
    holo = Image.new("RGBA", (W, H), (0, 0, 0, 0))         # holographic display on a pedestal (centre)
    d = ImageDraw.Draw(holo)
    d.rounded_rectangle([1080, 900, 1320, 1010], radius=18, fill=(60, 72, 110))
    holo.alpha_composite(_glow((W, H), [960, 480, 1440, 960], (90, 220, 255), 0.45, 40))
    for k in range(6):
        d.ellipse([1200 - 40 * k, 720 - 14 * k, 1200 + 40 * k, 720 + 14 * k], outline=(140, 240, 255, 170), width=3)
    L["obs_holo"] = holo
    fg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(fg).rounded_rectangle([-100, 1180, 700, 1500], radius=40, fill=(24, 28, 50))   # console edge
    L["obs_fg"] = _blur_layer(fg, 6)
    return L


PACKS = {
    "cinema": {
        "draw": draw_cinema, "topic": "movies, 3D films, light, vision",
        "sets": {
            "seats": {"layers": [("seats_back", 0.15), ("seats_rows", 0.35), ("seats_own_row", 0.6), ("@characters", 0.75),
                                 ("seats_light", 0.75), ("seats_fg", 1.25)],
                      "lighting": {"key": "front (the screen)", "tint": [205, 215, 255], "tint_strength": 0.15,
                                   "rim": [150, 175, 255], "temperature": "cool"}},
            "screen": {"layers": [("screen_back", 0.1), ("@screen", 0.1), ("screen_audience", 0.5), ("@characters", 0.8),
                                  ("screen_fg", 1.2)],
                       "screen_rect": [360, 110, 2040, 900],
                       "lighting": {"key": "back (the screen)", "tint": [150, 160, 220], "tint_strength": 0.4,
                                    "rim": [200, 210, 255], "temperature": "cool"}},
        },
        "presets": {
            "seats_two_shot": {"set": "seats", "crop": [180, 180, 2040, 1148], "move": "slow_push_in",
                               "slots": {"arjun": [820, 1746, 0.85], "meera": [1580, 1746, 0.85]}, "seated": True},
            "seats_wide": {"set": "seats", "crop": [0, 0, 2400, 1350], "move": "static",
                           "slots": {"arjun": [900, 1471, 0.7], "meera": [1500, 1471, 0.7]}, "seated": True},
            "seats_closeup_arjun": {"set": "seats", "crop": [520, 330, 1300, 731], "move": "slow_push_in",
                                    "slots": {"arjun": [1170, 1435, 0.6], "meera": [1990, 1435, 0.6]}, "seated": True},
            "seats_closeup_meera": {"set": "seats", "crop": [960, 330, 1300, 731], "move": "slow_push_in",
                                    "slots": {"meera": [1610, 1435, 0.6], "arjun": [790, 1435, 0.6]}, "seated": True},
            "screen_back": {"set": "screen", "crop": [0, 0, 2400, 1350], "move": "slow_push_in",
                            "slots": {"arjun": [1020, 1820, 0.55], "meera": [1380, 1820, 0.55]}, "view": "back"},
        },
        "map": {"cinema_seats": {"two_shot": "seats_two_shot", "wide": "seats_wide", "establishing": "seats_wide",
                                 "closeup": "seats_closeup_{subject}", "reaction": "seats_closeup_{subject}",
                                 "ots": "seats_closeup_{subject}"},
                "cinema_screen": {"*": "screen_back"}},
    },
    "lobby": {
        "draw": draw_lobby, "topic": "after the movie, everyday places",
        "sets": {"main": {"layers": [("lobby_back", 0.2), ("@characters", 0.8), ("lobby_fg", 1.3)],
                          "lighting": {"key": "warm overhead", "tint": [255, 226, 205], "tint_strength": 0.14,
                                       "rim": [255, 210, 170], "temperature": "warm"}}},
        "presets": {
            "lobby_two_shot": {"set": "main", "crop": [240, 120, 1920, 1080], "move": "slow_push_in",
                               "slots": {"arjun": [900, 1506, 0.75], "meera": [1500, 1506, 0.75]}},
            "lobby_wide": {"set": "main", "crop": [0, 0, 2400, 1350], "move": "static",
                           "slots": {"arjun": [950, 1298, 0.55], "meera": [1450, 1298, 0.55]}},
            "lobby_closeup_arjun": {"set": "main", "crop": [520, 220, 1300, 731], "move": "slow_push_in",
                                    "slots": {"arjun": [1170, 1325, 0.6], "meera": [1990, 1325, 0.6]}},
            "lobby_closeup_meera": {"set": "main", "crop": [960, 220, 1300, 731], "move": "slow_push_in",
                                    "slots": {"meera": [1610, 1325, 0.6], "arjun": [790, 1325, 0.6]}},
        },
        "map": {"lobby": {"two_shot": "lobby_two_shot", "wide": "lobby_wide", "establishing": "lobby_wide",
                          "closeup": "lobby_closeup_{subject}", "reaction": "lobby_closeup_{subject}",
                          "ots": "lobby_closeup_{subject}"}},
    },
    "studio": {
        "draw": draw_studio, "topic": "neutral explainer room",
        "sets": {"main": {"layers": [("studio_back", 0.2), ("@characters", 0.8)],
                          "lighting": {"key": "soft front", "tint": [230, 238, 255], "tint_strength": 0.08,
                                       "rim": [200, 220, 255], "temperature": "neutral"}}},
        "presets": {
            "studio_two_shot": {"set": "main", "crop": [240, 120, 1920, 1080], "move": "static",
                                "slots": {"arjun": [900, 1506, 0.75], "meera": [1500, 1506, 0.75]}},
            "studio_closeup_arjun": {"set": "main", "crop": [520, 220, 1300, 731], "move": "slow_push_in",
                                     "slots": {"arjun": [1170, 1325, 0.6], "meera": [1990, 1325, 0.6]}},
            "studio_closeup_meera": {"set": "main", "crop": [960, 220, 1300, 731], "move": "slow_push_in",
                                     "slots": {"meera": [1610, 1325, 0.6], "arjun": [790, 1325, 0.6]}},
        },
        "map": {"studio": {"*": "studio_two_shot", "closeup": "studio_closeup_{subject}",
                           "reaction": "studio_closeup_{subject}"},
                "living_room": {"*": "studio_two_shot", "closeup": "studio_closeup_{subject}",
                                "reaction": "studio_closeup_{subject}"}},
    },
    "space_observatory": {
        "draw": draw_space_observatory, "topic": "space, black holes, gravity, stars",
        "sets": {"main": {"layers": [("obs_sky", 0.05), ("obs_room", 0.3), ("obs_holo", 0.45), ("@characters", 0.8),
                                     ("obs_fg", 1.3)],
                          "lighting": {"key": "cool from the hologram", "tint": [170, 200, 255], "tint_strength": 0.3,
                                       "rim": [120, 230, 255], "temperature": "cool"}}},
        "presets": {
            "obs_two_shot": {"set": "main", "crop": [120, 80, 2160, 1215], "move": "slow_push_in",
                             "slots": {"arjun": [760, 1540, 0.75], "meera": [1640, 1540, 0.75]}},
            "obs_wide": {"set": "main", "crop": [0, 0, 2400, 1350], "move": "pan_right",
                         "slots": {"arjun": [700, 1380, 0.6], "meera": [1700, 1380, 0.6]}},
        },
        "map": {"space_observatory": {"*": "obs_two_shot", "wide": "obs_wide", "establishing": "obs_wide"}},
    },
}
SAFE = {"subtitles": [0, 0.80, 1.0, 1.0], "diagram": [0.08, 0.06, 0.92, 0.76]}   # fractions of the frame


def build_environment(root, env, version="v1", log=print):
    pack = PACKS[env]
    vdir = Path(root) / "environments" / env / version
    if (vdir / "manifest.json").exists():
        raise FileExistsError(f"{vdir} already exists: environment versions are immutable")
    tmp = vdir.with_name(version + ".partial")
    (tmp / "layers").mkdir(parents=True, exist_ok=True)
    layers = {}
    for name, im in pack["draw"]().items():
        f = tmp / "layers" / f"{name}.png"
        im.save(f, optimize=True)
        layers[name] = {"file": f"layers/{name}.png", "size": list(im.size), "sha256": _sha(f.read_bytes())}
    man = {"environment": env, "version": version, "status": "provisional", "art_version": ENV_ART_VERSION,
           "master": list(MASTER), "topic": pack["topic"], "sets": pack["sets"], "presets": pack["presets"],
           "map": pack["map"], "safe_regions": SAFE, "layers": layers,
           "provenance": {"method": "procedural drawing (colab/engine/lib/envfactory.py)", "model": None,
                          "license": "project-original (code-drawn)",
                          "created": datetime.datetime.now().isoformat(timespec="seconds"), "approved_by": None}}
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1))
    tmp.rename(vdir)
    log(f"environment {env} {version}: {len(layers)} layers written to {vdir}")
    return vdir


def verify(vdir):
    vdir = Path(vdir)
    try:
        man = json.loads((vdir / "manifest.json").read_text())
    except Exception as e:
        return [f"manifest unreadable: {e!r}"]
    bad = []
    for name, L in man["layers"].items():
        f = vdir / L["file"]
        if not f.exists():
            bad.append(f"missing layer {name}")
        elif _sha(f.read_bytes()) != L["sha256"]:
            bad.append(f"layer {name} changed since the manifest was written")
    for s, st in man["sets"].items():
        for lname, _ in st["layers"]:
            if not lname.startswith("@") and lname not in man["layers"]:
                bad.append(f"set {s} uses missing layer {lname}")
    return bad


class Environment:
    def __init__(self, vdir):
        self.dir = Path(vdir)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        self.id, self.version = self.man["environment"], self.man["version"]
        self._img, self._scaled = {}, {}

    def img(self, name):
        if name not in self._img:
            self._img[name] = Image.open(self.dir / self.man["layers"][name]["file"]).convert("RGBA")
        return self._img[name]

    def preset_for(self, world, shot, subject):
        """Which camera preset frames (world, shot) — or None if this pack doesn't cover that world."""
        m = self.man["map"].get(world)
        if m is None:
            return None
        p = m.get(shot) or m.get("*")
        if p is None:
            return None
        p = p.format(subject=subject or "arjun")
        return p if p in self.man["presets"] else None


_ENV = {}


def load(env, version, root):
    key = (str(root), env, version)
    if key in _ENV:
        return _ENV[key]
    vdir = Path(root) / "environments" / env / version
    if not (vdir / "manifest.json").exists():
        raise FileNotFoundError(f"environment {env} {version} not found in {root}")
    probs = verify(vdir)
    if probs:
        raise ValueError(f"environment {env} {version} is damaged: {probs[:3]}")
    _ENV[key] = Environment(vdir)
    return _ENV[key]


def ensure(env, version, root, log=print):
    vdir = Path(root) / "environments" / env / version
    if not (vdir / "manifest.json").exists():
        build_environment(root, env, version, log=log)
    return load(env, version, root)


WORLD_ENV = {"cinema_seats": "cinema", "cinema_screen": "cinema", "lobby": "lobby", "studio": "studio",
             "living_room": "studio", "space_observatory": "space_observatory"}


def camera(preset, u, move=None):
    """Crop box (x, y, w, h) in master pixels at shot fraction u, for a restrained camera move."""
    x, y, w, h = preset["crop"]
    move = move or preset.get("move", "static")
    if move == "slow_push_in":
        z = 1.0 + 0.05 * u
        return (x + w * (1 - 1 / z) / 2, y + h * (1 - 1 / z) / 2, w / z, h / z)
    if move == "slow_pull_back":
        z = 1.05 - 0.05 * u
        return (x + w * (1 - 1 / z) / 2, y + h * (1 - 1 / z) / 2, w / z, h / z)
    if move in ("pan_right", "pan_left"):
        dx = 0.04 * w * (u - 0.5) * (1 if move == "pan_right" else -1)
        return (x + dx, y, w, h)
    return (x, y, w, h)


# ------------------------------------------------------------------------------------------- generated plates (v2+)
PLATE_PROMPTS = {   # one plate per set; the camera presets of the base version are kept (same crops and slots)
    ("cinema", "seats"): ("Cinematic 3D animated-film background: the inside of a cosy movie theatre seen from the "
                          "screen toward rows of plush red velvet seats, warm wall sconces, soft blue light from the "
                          "projector screen on the seats, shallow depth of field, no people, no text. 16:9."),
    ("cinema", "screen"): ("Cinematic 3D animated-film background: a cosy movie theatre seen from behind the seats "
                           "toward a large glowing BLANK white cinema screen with red curtains, silhouettes of seat "
                           "backs in the foreground, no people, no text. 16:9."),
    ("lobby", "main"): ("Cinematic 3D animated-film background: a warm, colourful cinema lobby with movie posters "
                        "(no readable text), marquee light bulbs, a popcorn stand, polished floor, no people. 16:9."),
    ("studio", "main"): ("Cinematic 3D animated-film background: a calm, softly lit modern learning studio with "
                         "warm wood and teal accents, soft window light, no people, no text. 16:9."),
    ("space_observatory", "main"): ("Cinematic 3D animated-film background: a futuristic observatory with a huge "
                                    "window onto stars and a colourful galaxy, a glowing hologram pedestal in the "
                                    "middle, no people, no text. 16:9."),
}


def import_plates(root, env, version, plates, base_version="v1", provenance=None, screen_rects=None, log=print):
    """A NEW environment version whose back layers are generated plates ({set: path to a 16:9 image}), keeping the
    base version's camera presets, character slots, lighting, world map and safe regions — existing plans and
    camera framing keep working. Sets without a plate keep the base layers. The code-drawn foreground of a set
    that gets a plate is dropped (it would not match the painted plate); screen_rects can re-place the live movie."""
    import datetime, shutil
    base = json.loads((Path(root) / "environments" / env / base_version / "manifest.json").read_text())
    vdir = Path(root) / "environments" / env / version
    if (vdir / "manifest.json").exists():
        raise FileExistsError(f"{vdir} exists: environment versions are immutable")
    tmp = vdir.with_name(version + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "layers").mkdir(parents=True)
    layers, sets = {}, json.loads(json.dumps(base["sets"]))
    for set_name, st in sets.items():
        if set_name in plates:
            im = Image.open(plates[set_name]).convert("RGBA").resize(MASTER, Image.LANCZOS)
            name = f"{set_name}_plate"
            st["layers"] = [(name, 0.2)] + ([("@screen", 0.2)] if any(l == "@screen" for l, _ in st["layers"]) else []) \
                + [("@characters", 0.8)]
            if screen_rects and set_name in screen_rects:
                st["screen_rect"] = screen_rects[set_name]
            imgs = {name: im}
        else:
            imgs = {l: Image.open(Path(root) / "environments" / env / base_version / base["layers"][l]["file"])
                    for l, _ in st["layers"] if not l.startswith("@")}
        for n, im in imgs.items():
            f = tmp / "layers" / f"{n}.png"
            im.save(f, optimize=True)
            layers[n] = {"file": f"layers/{n}.png", "size": list(im.size), "sha256": _sha(f.read_bytes())}
    man = dict(base, version=version, status="provisional", art_version="generated-plates", sets=sets, layers=layers,
               provenance=dict(provenance or {}, base_version=base_version,
                               created=datetime.datetime.now().isoformat(timespec="seconds"), approved_by=None))
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1))
    tmp.rename(vdir)
    log(f"environment {env} {version}: plates for {sorted(plates)} on top of {base_version}'s presets")
    return vdir

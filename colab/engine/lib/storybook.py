"""Storybook character system: versioned character assets + a 2D cutout rig renderer (CPU, Pillow).

Asset library (one folder per character version; never rewritten once it exists):
  <root>/characters/<id>/<version>/manifest.json   identity, status (provisional | approved), provenance, palette,
                                                   rig, poses, expressions, visemes, and every layer's file + sha256
  <root>/characters/<id>/<version>/layers/*.png    cropped RGBA parts (offset recorded in the manifest)
  <root>/characters/<id>/<version>/previews/*.png  contact sheets for the human approval step

A video plan pins {id: {"version", "manifest_sha256"}} (plan["character_assets"]); rendering refuses a library whose
manifest hash differs, so new artwork can never silently change an old video. A missing approved version is an
error, never a fallback to simple shapes. Provisional versions (storybook_art.py) can be rebuilt from code.

Rig (front view): root(hips) → torso → {neck, head, upper arms → forearms → hands}; the head carries the face,
hair and glasses as one group. Joints have limits; poses are joint angles; transitions ease over ~0.45 s.
Face: expression → eyes / brows / resting mouth; blinking on a per-character schedule; gaze moves the irises
(clipped to the eye whites); `turn` (-1..1) shifts the features for a three-quarter look toward the other person.
Mouth: a viseme per frame from that speaker's own audio (silence → closed): only the speaker's mouth moves.
"""
from __future__ import annotations

import datetime, hashlib, json, math, os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

RIG_VERSION = 1
DEFAULT_RES = 1.6                     # stored pixels per design unit
BLEND_S = 0.45                        # gesture transition time

# ------------------------------------------------------------------------------------------- rig + library
JOINT_LIMITS = {"abduct": (-35, 170), "flex": (-110, 160), "wrist": (-45, 45), "head": (-14, 14), "lean": (-8, 8)}

# per side: a = lift outward (deg), b = elbow flex toward the body centre (deg), w = wrist, hand = hand drawing
POSES = {
    "idle":       {"L": dict(a=7, b=10, w=0, hand="open"), "R": dict(a=7, b=10, w=0, hand="open")},
    "listen":     {"L": dict(a=6, b=14, w=0, hand="open"), "R": dict(a=6, b=14, w=0, hand="open")},
    "explain":    {"L": dict(a=7, b=10, w=0, hand="open"), "R": dict(a=16, b=-62, w=-20, hand="open"), "lean": 1.5},
    "open_hands": {"L": dict(a=26, b=-48, w=-15, hand="open"), "R": dict(a=26, b=-48, w=-15, hand="open")},
    "point":      {"L": dict(a=7, b=10, w=0, hand="open"), "R": dict(a=82, b=-8, w=-10, hand="point"), "lean": 2},
    "think":      {"L": dict(a=-8, b=100, w=0, hand="hold"), "R": dict(a=-26, b=160, w=30, hand="hold"), "head": -5},
    "wave":       {"L": dict(a=7, b=10, w=0, hand="open"), "R": dict(a=100, b=-80, w=0, hand="wave")},
    "surprise":   {"L": dict(a=38, b=-35, w=-10, hand="open"), "R": dict(a=38, b=-35, w=-10, hand="open"), "lean": -3, "head": -3},
    "celebrate":  {"L": dict(a=150, b=-15, w=0, hand="open"), "R": dict(a=150, b=-15, w=0, hand="open")},
    "duck":       {"L": dict(a=72, b=-100, w=0, hand="open"), "R": dict(a=72, b=-100, w=0, hand="open"), "drop": 0.07, "head": 4},
    "hold":       {"L": dict(a=7, b=10, w=0, hand="open"), "R": dict(a=6, b=95, w=0, hand="hold")},
    "show":       {"L": dict(a=7, b=10, w=0, hand="open"), "R": dict(a=20, b=135, w=0, hand="hold")},
    "nod":        {"L": dict(a=6, b=14, w=0, hand="open"), "R": dict(a=6, b=14, w=0, hand="open")},
}
GESTURE_ALIAS = {"look_object": "listen", "react": "surprise", "step": "idle"}

# expression → eyes, brows, resting mouth, iris size
EXPRESSIONS = {
    "neutral":   ("open", "neutral", "rest", ""), "listening": ("open", "neutral", "rest", ""),
    "smile":     ("smile", "neutral", "smile", ""), "happy": ("smile", "raised", "smile", ""),
    "curious":   ("open", "curious", "rest", ""), "confused": ("open", "furrow", "wavy", ""),
    "surprised": ("wide", "high", "o_big", "small"), "excited": ("wide", "raised", "smile_talk", ""),
    "thinking":  ("half", "furrow", "pursed", ""), "concerned": ("open", "sad", "frown", ""),
    "laughing":  ("happy", "raised", "laugh", ""), "wink": ("wink", "raised", "smile", ""),
}
SMILING = {"smile", "happy", "excited", "laughing", "wink"}
VISEMES = ("rest", "closed", "small", "wide", "round", "teeth", "smile_talk")

LAYER_NAMES = None                    # filled from storybook_art on build


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def manifest_sha(manifest_path):
    return _sha(Path(manifest_path).read_bytes())


def build_character(root, cid, version="v1", res=DEFAULT_RES, log=print):
    """Draw the provisional layers once and write the version folder. Never overwrites an existing version."""
    import storybook_art as art
    vdir = Path(root) / "characters" / cid / version
    if (vdir / "manifest.json").exists():
        raise FileExistsError(f"{vdir} already exists: asset versions are immutable (make a new version)")
    tmp = vdir.with_name(version + ".partial")             # write to a temp folder, rename when complete
    (tmp / "layers").mkdir(parents=True, exist_ok=True)
    layers = {}
    for name, im in art.layers(cid, res=res).items():
        bb = im.getbbox()
        if bb is None:
            continue
        crop = im.crop(bb)
        f = tmp / "layers" / f"{name}.png"
        crop.save(f, optimize=True)
        layers[name] = {"file": f"layers/{name}.png", "offset": [bb[0] / res, bb[1] / res], "size": list(crop.size),
                        "sha256": _sha(f.read_bytes())}
    d = art.DESIGNS[cid]
    man = {
        "character": cid, "version": version, "status": "provisional", "rig_version": RIG_VERSION,
        "art_version": art.ART_VERSION, "res": res, "canvas": list(art.CANVAS),
        "provenance": {"method": "procedural vector drawing (colab/engine/lib/storybook_art.py)", "model": None,
                       "license": "project-original (code-drawn)", "created": datetime.datetime.now().isoformat(timespec="seconds"),
                       "approved_by": None, "approved_at": None,
                       "note": "PROVISIONAL engineering art: replace with approved Character Studio artwork"},
        "palette": {k: v for k, v in d.items() if isinstance(v, str) and v.startswith("#")},
        "identity": {"hair_style": d["hair_style"], "accessory": d["accessory"]},
        "rig": {"joints": {"shoulder": {s: list(art.SHOULDER[s]) for s in "LR"}, "elbow": {s: list(art.ELBOW[s]) for s in "LR"},
                           "wrist": {s: list(art.WRIST[s]) for s in "LR"}, "neck": [500, 712], "hips": [500, 1180]},
                "limits": JOINT_LIMITS, "layer_order": ["hair_back", "legs", "neck", "torso", "upper_arm_L", "upper_arm_R", "head_group",
                                                         "forearm_L", "hand_L", "forearm_R", "hand_R", "prop"],
                "views": ["front", "three_quarter (feature shift)", "back"]},
        "poses": POSES, "expressions": {k: list(v) for k, v in EXPRESSIONS.items()}, "visemes": list(VISEMES),
        "layers": layers,
    }
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1))
    if vdir.exists():
        raise FileExistsError(str(vdir))
    tmp.rename(vdir)
    log(f"character {cid} {version}: {len(layers)} layers written to {vdir}")
    return vdir


def verify(vdir):
    """Every layer present and unchanged (sha256) → [] or a list of problems."""
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
    return bad


def content_fingerprint(man):
    """sha256 of the artwork only (every layer hash + geometry), NOT of status/approval fields: approving a version
    changes its manifest hash but never this fingerprint."""
    if man.get("asset_type") == "atlas_v1":
        import storybook_atlas
        return storybook_atlas.content_sha(man)
    body = {"layers": {k: v["sha256"] for k, v in sorted(man["layers"].items())}, "rig": man.get("rig")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def approval_blockers(vdir):
    """Why this version cannot be approved yet ([] = it can). Assets complete and unchanged, previews made,
    model licence allowed in strict mode; an atlas must also cover blinking, every mouth shape and every pose."""
    vdir = Path(vdir)
    out = list(verify(vdir))
    if out:
        return out
    man = json.loads((vdir / "manifest.json").read_text())
    if man.get("provenance", {}).get("prototype_only"):
        out.append("prototype asset substitutions cannot be approved for production")
    if man.get("asset_type") == "atlas_v1":
        import storybook_atlas
        cov = storybook_atlas.coverage(man)
        out += [f"pose '{p}' is not mapped" for p, v in cov["poses_mapped"].items() if v is None]
        out += [f"mouth shape '{v}' missing" for v, ok in cov["visemes"].items() if not ok]
        out += ["no blink (eyes-closed) patch"] if not cov["blink"] else []
        previews = sorted((vdir / "previews").glob("*")) if (vdir / "previews").exists() else []
        if not previews:
            out.append("no previews in previews/ (render them with colab/storybook_previews.py --atlas)")
        model = (man.get("provenance") or {}).get("model")
        if model:
            import registry
            ok, why = registry.model_ok(model, strict=True)
            if not ok:
                out.append(f"licence: {why}")
    return out


def approve(vdir, who, note=""):
    """Human approval: records who/when in the manifest of a NEW version folder copy is not needed — approval is
    the one allowed manifest edit (status provisional → approved). Layers can never change afterwards."""
    vdir = Path(vdir)
    man = json.loads((vdir / "manifest.json").read_text())
    if not who or not str(who).strip():
        raise ValueError("approval needs the name of the person who looked at the previews")
    blockers = approval_blockers(vdir)
    if blockers:
        raise ValueError(f"cannot approve: {blockers}")
    fp = content_fingerprint(man)
    man["status"] = "approved"
    man["content_sha256"] = fp
    man["provenance"].update(approved_by=who, approved_at=datetime.datetime.now().isoformat(timespec="seconds"),
                             approval_note=note)
    (vdir / "manifest.json").write_text(json.dumps(man, indent=1))
    return manifest_sha(vdir / "manifest.json")


class Character:
    """One character version loaded for rendering (sprites cached per output scale)."""

    def __init__(self, vdir):
        self.dir = Path(vdir)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        self.id, self.version = self.man["character"], self.man["version"]
        self.res = self.man["res"]
        self._img, self._scaled = {}, OrderedDict()
        j = self.man["rig"]["joints"]
        self.sh, self.el, self.wr = j["shoulder"], j["elbow"], j["wrist"]
        self.neck, self.hips = j["neck"], j["hips"]

    def has(self, name):
        return name in self.man["layers"]

    def img(self, name):
        if name not in self._img:
            self._img[name] = Image.open(self.dir / self.man["layers"][name]["file"]).convert("RGBA")
        return self._img[name]

    def sprite(self, name, s):
        """(image at output scale s px/unit, offset in design units)."""
        key = (name, round(s, 3))
        if key in self._scaled:
            self._scaled.move_to_end(key)
            return self._scaled[key]
        L = self.man["layers"][name]
        im = self.img(name)
        f = s / self.res
        w, h = max(1, int(round(im.width * f))), max(1, int(round(im.height * f)))
        out = (im.resize((w, h), Image.LANCZOS), tuple(L["offset"]))
        self._scaled[key] = out
        if len(self._scaled) > 400:
            self._scaled.popitem(last=False)
        return out


_LIB = {}


def library_root(plan=None):
    """Where character/environment versions live: STORYBOOK_ASSETS (env), else the repo's colab/assets."""
    return Path(os.environ.get("STORYBOOK_ASSETS") or Path(__file__).resolve().parents[2] / "assets")


def load(cid, version, root=None, expect_sha=None, expect_content=None):
    root = Path(root or library_root())
    key = (str(root.resolve()), cid, version, expect_sha, expect_content)
    vdir = root / "characters" / cid / version
    if not (vdir / "manifest.json").exists():
        raise FileNotFoundError(f"character asset {cid} {version} not found in {root} (build or download it; "
                                "the renderer never substitutes simple shapes)")
    if expect_sha and manifest_sha(vdir / "manifest.json") != expect_sha:
        raise ValueError(f"character asset {cid} {version}: manifest differs from the one this plan was made with")
    man = json.loads((vdir / "manifest.json").read_text())
    if expect_content and content_fingerprint(man) != expect_content:
        raise ValueError(f"character asset {cid} {version}: artwork differs from the one this plan was made with")
    if man.get("character") != cid or man.get("version") != version:
        raise ValueError("character identity/version differs from its directory")
    def stat_signature(path):
        st = path.stat()
        return (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)
    signature = (manifest_sha(vdir / "manifest.json"),
                 tuple((v["file"], stat_signature(vdir / v["file"])) for v in man["layers"].values()))
    if key in _LIB and getattr(_LIB[key], "_verified_signature", None) == signature:
        return _LIB[key]
    probs = verify(vdir)
    if probs:
        raise ValueError(f"character asset {cid} {version} is damaged: {probs[:3]}")
    if man.get("asset_type") == "atlas_v1":
        import storybook_atlas
        _LIB[key] = storybook_atlas.AtlasCharacter(vdir)
    else:
        _LIB[key] = Character(vdir)
    _LIB[key]._verified_signature = signature
    return _LIB[key]


def install_pack(data, root=None, log=print):
    """Unpack an asset pack (tar.gz bytes, or the same as base64 text) into the library — only folders that are
    not there yet (an existing version is never overwritten); written to a staging folder, then moved into place."""
    if not data or (isinstance(data, str) and data.startswith("__")):
        return []
    import base64, io, shutil, tarfile
    raw = base64.b64decode(data) if isinstance(data, str) else data
    root = Path(root or library_root())
    stage = root / ".incoming"
    shutil.rmtree(stage, ignore_errors=True)
    installed = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        for m in tf.getmembers():
            top = Path(*Path(m.name).parts[:3])
            if not (root / top / "manifest.json").exists():
                tf.extract(m, stage, filter="data")
    for vdir in sorted(stage.glob("*/*/*")):
        dst = root / vdir.relative_to(stage)
        if not (dst / "manifest.json").exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(vdir), str(dst))
            installed.append(str(dst.relative_to(root)))
            log(f"storybook asset {dst.relative_to(root)} installed from the notebook")
    shutil.rmtree(stage, ignore_errors=True)
    return installed


def _build_one(args):
    root, cid, version = args
    return str(build_character(root, cid, version, log=lambda *a: None))


def prepare_assets(plan, pack=None, root=None, log=print):
    """Make every asset version the plan pins available in the library (Drive), in this order:
      1. already in the library → used as is (verified when loaded);
      2. the plan's asset pack file next to the library (pack = {"name", "sha256"}; you upload it once) → installed
         if its sha256 matches — byte-identical to the art the plan was checked against;
      3. PROVISIONAL code-drawn versions → drawn here from the same code (once; characters in parallel; then cached);
      4. an APPROVED version that is missing → error (approved art comes from the Character Studio on Drive).
    Returns {asset: how} for the run log / QC."""
    import hashlib, time
    import envfactory
    if plan.get("cinematic_characters"):
        import cinematic_workflow as CW
        CW.validate_plan(plan, root or library_root())
    root = Path(root or library_root())
    root.mkdir(parents=True, exist_ok=True)
    chars = {c: a for c, a in (plan.get("character_assets") or {}).items()}
    envs = dict(plan.get("environments") or {})
    have = lambda kind, i, v: (root / kind / i / v / "manifest.json").exists()
    report = {}
    missing = [("characters", c, a["version"]) for c, a in chars.items() if not have("characters", c, a["version"])] + \
              [("environments", e, v) for e, v in envs.items() if not have("environments", e, v)]
    if missing and pack and pack.get("name"):
        f = root / pack["name"]
        if f.exists():
            data = f.read_bytes()
            if hashlib.sha256(data).hexdigest() == pack.get("sha256"):
                for p in install_pack(data, root, log):
                    report[p] = "installed from the uploaded pack (byte-identical)"
            else:
                log(f"asset pack {f} does not match this plan (sha256): ignored")
    todo = []
    for c, a in chars.items():
        if have("characters", c, a["version"]):
            report.setdefault(f"characters/{c}/{a['version']}", "already on Drive")
        elif a.get("status") == "approved":
            raise RuntimeError(f"approved character art {c} {a['version']} is not in {root}: copy it from the Character "
                               "Studio output (videos never fall back to other art)")
        else:
            todo.append((str(root), c, a["version"]))
    if todo:
        t0 = time.time()
        log(f"drawing the provisional characters {[t[1] for t in todo]} once (about 6 min on Colab's CPU; then cached "
            f"on Drive). Faster: upload {pack.get('name') if pack else 'the asset pack'} to {root}")
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(2, len(todo))) as ex:
            for (r_, c, v), _ in zip(todo, ex.map(_build_one, todo)):
                report[f"characters/{c}/{v}"] = f"drawn from code ({time.time() - t0:.0f}s)"
    for e, v in envs.items():
        if not have("environments", e, v):
            envfactory.build_environment(root, e, v, log=lambda *a: None)
            report[f"environments/{e}/{v}"] = "drawn from code"
        else:
            report.setdefault(f"environments/{e}/{v}", "already on Drive")
    for c, a in chars.items():                       # same bytes as the plan was checked with? (information)
        m = root / "characters" / c / a["version"] / "manifest.json"
        if a.get("manifest_sha256"):
            report[f"characters/{c}/{a['version']}"] += "; identical to the checked art" if manifest_sha(m) == \
                a["manifest_sha256"] else "; redrawn (same code and version, not byte-identical)"
    return report


def ensure(cid, version, root=None, log=print):
    """Load a version; build it first if it is a missing PROVISIONAL (code-drawn) version."""
    root = Path(root or library_root())
    vdir = root / "characters" / cid / version
    if not (vdir / "manifest.json").exists():
        build_character(root, cid, version, log=log)
    return load(cid, version, root)


# ------------------------------------------------------------------------------------------- geometry
def _rot(p, c, deg):
    a = math.radians(deg)
    x, y = p[0] - c[0], p[1] - c[1]
    return (c[0] + x * math.cos(a) - y * math.sin(a), c[1] + x * math.sin(a) + y * math.cos(a))


def clamp(v, lim):
    return max(lim[0], min(lim[1], v))


def ease(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def blend_pose(p0, p1, u):
    u = ease(u)
    out = {}
    for side in ("L", "R"):
        a, b = p0[side], p1[side]
        out[side] = {k: a[k] + (b[k] - a[k]) * u for k in ("a", "b", "w")}
        out[side]["hand"] = b["hand"] if u >= 0.5 else a["hand"]
    for k in ("lean", "head", "drop"):
        out[k] = p0.get(k, 0) + (p1.get(k, 0) - p0.get(k, 0)) * u
    return out


def pose_of(name):
    return POSES.get(GESTURE_ALIAS.get(name, name), POSES["idle"])


# ------------------------------------------------------------------------------------------- drawing one character
class Painter:
    """Paints a Character at output scale s (px per design unit) onto a frame."""

    def __init__(self, ch, s):
        self.ch, self.s = ch, s
        self._rot = OrderedDict()
        self._head = OrderedDict()

    def _put(self, frame, im, x, y):
        frame.alpha_composite(im, (int(round(x)), int(round(y))))

    def _rotated(self, key, im, deg, pivot_px):
        """im rotated by deg (clockwise on screen) about pivot_px; returns (image, pivot position in the result)."""
        if abs(deg) < 0.05:
            return im, pivot_px
        k = (key, round(deg * 4) / 4)
        if k in self._rot:
            self._rot.move_to_end(k)
            r, p = self._rot[k]
            return r, p
        W, H = im.size
        cx, cy = pivot_px
        R = int(math.ceil(max(math.hypot(cx - x, cy - y) for x in (0, W) for y in (0, H)))) + 2
        pad = Image.new("RGBA", (2 * R, 2 * R), (0, 0, 0, 0))
        pad.alpha_composite(im, (int(R - cx), int(R - cy)))
        r = pad.rotate(-k[1], resample=Image.BICUBIC, center=(R, R))
        bb = r.getbbox() or (0, 0, 1, 1)
        r = r.crop(bb)
        p = (R - bb[0], R - bb[1])
        self._rot[k] = (r, p)
        if len(self._rot) > 600:
            self._rot.popitem(last=False)
        return r, p

    def part(self, frame, name, origin, M, deg, pivot_u):
        """Draw a layer whose design-unit pivot `pivot_u` lands on output pixel M(pivot_u), rotated by deg."""
        im, off = self.ch.sprite(name, self.s)
        piv_px = ((pivot_u[0] - off[0]) * self.s, (pivot_u[1] - off[1]) * self.s)
        r, p = self._rotated((name, round(self.s, 3)), im, deg, piv_px)
        tx, ty = M(pivot_u)
        self._put(frame, r, origin[0] + tx - p[0], origin[1] + ty - p[1])

    def head_group(self, st, layer_set):
        """Face + hair + glasses composed into one sprite (cached per state), rotated as a whole later."""
        key = (round(self.s, 3), layer_set, st["eyes"], st["brows"], st["mouth"], st["gaze"], st["turn"], st["glasses"],
               st["iris"])
        if key in self._head:
            self._head.move_to_end(key)
            return self._head[key]
        ch, s = self.ch, self.s
        x0, y0, x1, y1 = 180, 120, 840, 1180 if layer_set == "back" else 760   # design-unit box of the group
        canvas = Image.new("RGBA", (int((x1 - x0) * s), int((y1 - y0) * s)), (0, 0, 0, 0))

        def put(name, dx=0.0, dy=0.0):
            if not ch.has(name):
                return
            im, off = ch.sprite(name, s)
            canvas.alpha_composite(im, (int(round((off[0] + dx - x0) * s)), int(round((off[1] + dy - y0) * s))))

        if layer_set == "back":
            put("hair_back")
        else:
            fx = st["turn"] * 26                       # three-quarter look: features slide toward the turn
            put("head"); put("cheeks", fx * 0.8); put("nose", fx * 1.1)
            for side in ("L", "R"):
                ev = st["eyes"]
                if ev == "wink":
                    ev = "open" if side == "L" else "happy"
                if ev in ("open", "half", "wide", "smile"):
                    put(f"eye_{side}_{ev}", fx)
                    iris_name = f"iris_{side}" + ("_small" if st["iris"] == "small" else "")
                    im, off = ch.sprite(iris_name, s)
                    mim, moff = ch.sprite(f"eyemask_{side}_{ev}", s)
                    gx, gy = st["gaze"]
                    tmp = Image.new("RGBA", mim.size, (0, 0, 0, 0))
                    tmp.alpha_composite(im, (int(round((off[0] + gx - moff[0]) * s)), int(round((off[1] + gy - moff[1]) * s))))
                    a = np.minimum(np.asarray(tmp.getchannel("A")), np.asarray(mim.getchannel("A")))
                    tmp.putalpha(Image.fromarray(a))
                    canvas.alpha_composite(tmp, (int(round((moff[0] + fx - x0) * s)), int(round((moff[1] - y0) * s))))
                    put(f"lid_{side}_{ev}", fx)        # lids and lashes over the iris
                else:
                    put(f"eye_{side}_{ev}", fx)
            put(f"brows_{st['brows']}", fx)
            put(f"mouth_{st['mouth']}", fx * 0.9)
            put("hair_front", st["turn"] * 8)
            if st["glasses"]:
                put("glasses", fx)
        out = (canvas, (x0, y0))
        self._head[key] = out
        if len(self._head) > 300:
            self._head.popitem(last=False)
        return out

    def draw(self, frame, origin, pose, face, view="front", hold=None, legs=True):
        """origin: output pixel of design point (0, 0) of this character (before the rig moves the root)."""
        ch, s = self.ch, self.s
        if view == "back":
            im, off = ch.sprite("back", s)
            self._put(frame, im, origin[0] + off[0] * s, origin[1] + off[1] * s)
            return
        drop = pose.get("drop", 0) * 1800
        lean = clamp(pose.get("lean", 0), JOINT_LIMITS["lean"])
        hips = ch.hips
        o = (origin[0], origin[1] + drop * s)

        def torsoM(p):                                  # torso space → output (lean about the hips)
            q = _rot(p, hips, lean)
            return (q[0] * s, q[1] * s)
        # head transform: about the neck, plus lean, plus breathing/head offsets
        hdeg = clamp(pose.get("head", 0) + face.get("head_tilt", 0), JOINT_LIMITS["head"])
        head_dy = face.get("head_dy", 0)

        def headM(p):
            q = _rot(_rot(p, ch.neck, hdeg), hips, lean)
            return (q[0] * s, (q[1] + head_dy) * s)
        # arm chains
        chains = {}
        for side in ("L", "R"):
            sgn = 1 if side == "L" else -1                # L lifts clockwise on screen, R counter-clockwise
            a = clamp(pose[side]["a"], JOINT_LIMITS["abduct"])
            b = clamp(pose[side]["b"], JOINT_LIMITS["flex"])
            w = clamp(pose[side]["w"], JOINT_LIMITS["wrist"])
            up_deg = sgn * a + lean
            fo_deg = up_deg - sgn * b
            ha_deg = fo_deg - sgn * w
            shoulder = torsoM(ch.sh[side])
            el_u = _rot(ch.el[side], ch.sh[side], sgn * a)
            el_u = _rot(el_u, hips, lean)
            elbow = (el_u[0] * s, el_u[1] * s)
            wr_u = _rot(_rot(ch.wr[side], ch.el[side], -sgn * b), ch.sh[side], sgn * a)
            wr_u = _rot(wr_u, hips, lean)
            wrist = (wr_u[0] * s, wr_u[1] * s)
            chains[side] = (up_deg, fo_deg, ha_deg, shoulder, elbow, wrist, pose[side]["hand"])

        def arm(side, part):
            up_deg, fo_deg, ha_deg, shoulder, elbow, wrist, hand = chains[side]
            if part == "upper":
                self.part(frame, f"upper_arm_{side}", o, lambda p: shoulder, up_deg, ch.sh[side])
            elif part == "fore":
                self.part(frame, f"forearm_{side}", o, lambda p: elbow, fo_deg, ch.el[side])
            else:
                self.part(frame, f"hand_{side}_{hand}", o, lambda p: wrist, ha_deg, ch.wr[side])
                if side == "R" and hold:
                    self.part(frame, f"prop_{hold}", o, lambda p: wrist, ha_deg if hold != "popcorn" else 0, ch.wr[side])

        hb, hb_org = self.head_group(face, "back")
        piv = ((ch.neck[0] - hb_org[0]) * s, (ch.neck[1] - hb_org[1]) * s)
        r, p = self._rotated(("hb", face["turn"], round(s, 3)), hb, hdeg + lean, piv)
        nk = headM(ch.neck)
        self._put(frame, r, o[0] + nk[0] - p[0], o[1] + nk[1] - p[1])
        if legs:                                        # legs stay planted (a duck bends the upper body down)
            im, off = ch.sprite("legs", s)
            self._put(frame, im, origin[0] + off[0] * s, origin[1] + off[1] * s)
        self.part(frame, "neck", o, torsoM, lean, ch.hips)
        self.part(frame, "torso", o, torsoM, lean, ch.hips)
        arm("L", "upper"); arm("R", "upper")
        hf, hf_org = self.head_group(face, "front")
        piv = ((ch.neck[0] - hf_org[0]) * s, (ch.neck[1] - hf_org[1]) * s)
        key = ("hf", face["eyes"], face["brows"], face["mouth"], face["gaze"], face["turn"], face["glasses"], face["iris"], round(s, 3))
        r, p = self._rotated(key, hf, hdeg + lean, piv)
        self._put(frame, r, o[0] + nk[0] - p[0], o[1] + nk[1] - p[1])
        for side in ("L", "R"):
            arm(side, "fore")
            arm(side, "hand")


# ------------------------------------------------------------------------------------------- timing helpers
def blink_closed(cid, t_abs):
    rng = np.random.default_rng(int(hashlib.sha1(cid.encode()).hexdigest()[:8], 16))
    t = rng.uniform(0.6, 2.0)
    while t < t_abs + 1:
        if t <= t_abs < t + 0.12:
            return True
        t += rng.uniform(2.4, 5.0)
    return False


def idle_motion(cid, t_abs, speaking):
    """Small, slow, non-repeating head motion (two incommensurate sines); a bit livelier while speaking."""
    ph = (int(hashlib.sha1(cid.encode()).hexdigest()[:4], 16) % 628) / 100
    amp = 1.6 if speaking else 1.0
    tilt = amp * (1.1 * math.sin(2 * math.pi * t_abs / 4.7 + ph) + 0.6 * math.sin(2 * math.pi * t_abs / 2.9 + 2 * ph))
    breathe = 2.2 * math.sin(2 * math.pi * t_abs / 3.8 + ph)
    return tilt, breathe


def viseme_from_energy(e, prev):
    """Fallback when no spectral viseme track exists: loudness → mouth opening only."""
    if e < 0.08:
        return "rest"
    if e < 0.3:
        return "small"
    return "wide" if prev != "wide" or e > 0.6 else "small"


def visemes_from_audio(wav, sr, fps, n_frames=None):
    """Mouth shape per video frame from a speaker's own audio (language independent):
    silence → rest; quiet → small; open vowels by spectral balance: low (o/u) → round, high (i/e, s) → teeth,
    otherwise wide; very short dips inside speech → closed (m/b/p-like). Held ≥ 2 frames (no flicker)."""
    w = wav.mean(1) if wav.ndim > 1 else wav
    hop = sr / fps
    n = n_frames if n_frames is not None else int(len(w) / hop)
    rms, cen = np.zeros(n), np.zeros(n)
    win = np.hanning(int(hop * 2) or 2)
    for i in range(n):
        a = int(i * hop)
        x = w[a:a + len(win)]
        if len(x) < len(win):
            x = np.pad(x, (0, len(win) - len(x)))
        rms[i] = math.sqrt(float(np.mean(x ** 2)) + 1e-12)
        sp = np.abs(np.fft.rfft(x * win))
        f = np.fft.rfftfreq(len(x), 1 / sr)
        band = (f > 80) & (f < 6000)
        cen[i] = float((sp[band] * f[band]).sum() / (sp[band].sum() + 1e-9))
    ref = np.percentile(rms[rms > 0], 90) if np.any(rms > 0) else 1.0
    e = np.clip(rms / max(ref, 1e-6), 0, 1.5)
    out, prev, held = [], "rest", 0
    for i in range(n):
        if e[i] < 0.07:
            v = "rest"
        elif e[i] < 0.22:
            v = "closed" if 0 < i < n - 1 and e[i - 1] > 0.3 and e[i + 1] > 0.3 else "small"
        elif cen[i] < 900:
            v = "round"
        elif cen[i] > 2600:
            v = "teeth"
        else:
            v = "wide"
        if v != prev and held < 2 and prev != "rest" and v != "rest":
            v = prev
        held = held + 1 if v == prev else 1
        out.append(v)
        prev = v
    return out, e

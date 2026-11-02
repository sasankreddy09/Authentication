"""Pose-atlas character assets ("atlas_v1"): cinematic AI artwork animated WITHOUT cutting it into joints.

Why (measured in reports/character_poc/approach_A_cutout_arm_test.png): rotating a cut-out arm of painted, 3D-style
artwork leaves a hard cut down the torso, a broken shoulder and fabric shading that no longer matches the angle —
repairing that for every pose is as much work as rendering the pose. So each pose is a COMPLETE image rendered by
the model from the approved master (reference-conditioned), and only small regions change at runtime:
  - face patches (eyes-closed blink, mouth shapes, expressions), feathered, rendered by the same model from the same
    master so the shading matches (no flat pasted cartoon mouth);
  - whole-character motion: breathing (≤0.6 % vertical), a slow sway (≤0.6° about the feet), eased crossfades
    (BLEND_S) between poses, and three-quarter views when the character turns to the other person.

Manifest (in the same versioned, sha256-verified folder format as storybook.build_character):
  "asset_type": "atlas_v1",
  "poses":   {atlas_pose: {"file", "view": "front|three_quarter_left|three_quarter_right|back", "feet": [x, y],
              "top": y, "head": [centre_x, top_y, chin_y], "face": {"box": [x0, y0, x1, y1]}, "wear": [...],
              "hold": null|"popcorn"|...}}   — "head" places the character by its HEAD, so every camera preset
              frames the face the same way whatever the body proportions are (else feet/top are used)
  "pose_map": {storybook pose (13 names) → atlas_pose}   ← explicit, reviewed substitutes, never implicit
  "face":     {"anchor_pose": atlas_pose, "box": [...], "blink": file, "visemes": {viseme: file},
               "expressions": {expression: file}}   (patches are RGBA crops of the anchor face box)
  "layers":   every file + sha256 (verify/approve/load reuse storybook's functions)
A plan that needs something the atlas does not have (a back view, a held prop, 3D glasses) gets a clear error from
check_spec — the renderer never invents it and never falls back to the code-drawn characters.
"""
from __future__ import annotations

import hashlib, json
from collections import OrderedDict
from pathlib import Path

from PIL import Image

import storybook as SB

ASSET_TYPE = "atlas_v1"
DESIGN_HEIGHT = 1426            # head top (190) → shoe soles (1616) in the storybook design units
DESIGN_FEET = (500, 1616)
DESIGN_HEAD = (500, 210, 720)   # head centre x, head top (hair), chin — the camera presets frame THIS head
VIEWS = ("front", "three_quarter_left", "three_quarter_right", "back")
MOUTH_OF_VISEME = {"rest": None, "closed": "closed", "small": "small", "wide": "wide", "round": "round",
                   "teeth": "teeth", "smile_talk": "smile_talk"}


def is_atlas(man):
    return man.get("asset_type") == ASSET_TYPE


class AtlasCharacter:
    """An atlas_v1 version folder (loaded through storybook.load, which verifies every sha256 first)."""

    def __init__(self, vdir):
        self.dir = Path(vdir)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        self.id, self.version = self.man["character"], self.man["version"]
        self._img = {}

    def has(self, name):
        return name in self.man["layers"]

    def img(self, rel):
        if rel not in self._img:
            self._img[rel] = Image.open(self.dir / rel).convert("RGBA")
        return self._img[rel]

    def wear_by_patch(self, w):
        return w in (self.man.get("face") or {}).get("wear", {})

    def atlas_pose(self, storybook_pose, view="front", wear=(), hold=None):
        """The atlas pose image for a storybook pose in a view, or None if the atlas lacks it. Composition rules
        (documented, not guesses): things worn on the face (3D glasses) are face patches on any FRONT pose and are
        not visible from behind; a held prop uses this gesture's own image with the prop if there is one, otherwise
        the prop's hold pose (holding popcorn → the popcorn pose); seen from behind, every gesture uses the back
        image (a gesture is not visible from behind)."""
        poses = self.man["poses"]
        base = self.man["pose_map"].get(storybook_pose)
        if base is None:
            return None
        need_wear = sorted(w for w in wear if not (view == "back" or (view == "front" and self.wear_by_patch(w))))

        def match(k, h):
            p = poses.get(k)
            return p and p.get("view", "front") == view and sorted(p.get("wear", [])) == need_wear and p.get("hold") == h

        cands = [base] + [k for k, p in poses.items() if p.get("base") in (storybook_pose, base)]
        for k in cands:
            if match(k, hold):
                return k
        if hold:                                   # the prop's own hold pose
            for k, p in poses.items():
                if p.get("hold") == hold and p.get("base") in ("hold", "show") and match(k, hold):
                    return k
        if view == "back":                         # from behind the gesture is not visible: the back image
            for k, p in poses.items():
                if p.get("view") == "back" and p.get("base") in ("idle", None) and match(k, hold):
                    return k
        return None

    def missing(self, storybook_pose, view, wear, hold):
        if self.atlas_pose(storybook_pose, view, wear, hold) is None:
            need = [f"pose '{storybook_pose}'", f"view '{view}'"] + [f"wearing {w}" for w in wear] + \
                   ([f"holding {hold}"] if hold else [])
            return (f"{self.id} {self.version} (atlas) has no image for " + ", ".join(need) +
                    " — add it with the Character Studio (a new version), or change the plan")
        return None


class AtlasPainter:
    """Same role as storybook.Painter, for atlas characters: draws one character at output scale s (px/design unit)."""

    def __init__(self, ch, s):
        self.ch, self.s = ch, s
        self._cache = OrderedDict()

    def _scaled(self, rel, k):
        key = (rel, round(k, 4))
        if key not in self._cache:
            im = self.ch.img(rel)
            self._cache[key] = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
            if len(self._cache) > 64:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def _pose_image(self, pid, face):
        """The pose image with the face patches for this frame (blink / expression / mouth) composited in place."""
        P = self.ch.man["poses"][pid]
        F = self.ch.man.get("face", {})
        im = self.ch.img(P["file"])
        patches = []
        # Face edits are sampled from the idle master, not the current pose. Applying them
        # to independently generated pose faces causes double eyes/mouths and wandering overlays.
        # Until per-pose facial registration has been explicitly reviewed, fail closed.
        if (F and pid == F.get("anchor_pose") and P.get("face")
                and P.get("view", "front") == "front"):
            # order: expression (whole face) → blink (eyes) → worn items (eyes) → mouth shape (mouth)
            expr = face.get("expr")
            if expr and expr in F.get("expressions", {}):
                patches.append(F["expressions"][expr])
            wearing = [w for w in face.get("wear", ()) if w in F.get("wear", {})]
            if face.get("blink") and F.get("blink") and not wearing:     # (behind dark glasses a blink is invisible)
                patches.append(F["blink"])
            patches += [F["wear"][w] for w in wearing]
            mouth = MOUTH_OF_VISEME.get(face.get("viseme") or "rest")
            if face.get("speaking") and mouth and mouth in F.get("visemes", {}):
                patches.append(F["visemes"][mouth])
        if not patches:
            return im
        key = ("composed", pid, json.dumps(patches, sort_keys=True))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        out = im.copy()
        ax0, ay0, ax1, ay1 = F["box"]
        bx0, by0, bx1, by1 = P["face"]["box"]
        if ax1 <= ax0 or ay1 <= ay0 or bx1 <= bx0 or by1 <= by0:
            raise ValueError("invalid atlas face box; correct the reviewed atlas spec")
        kx = (bx1 - bx0) / (ax1 - ax0)
        ky = (by1 - by0) / (ay1 - ay0)
        for ent in patches:
            rel, pb = (ent["file"], ent["box"]) if isinstance(ent, dict) else (ent, F["box"])
            p = self.ch.img(rel)
            if abs(kx - 1) > 1e-3 or abs(ky - 1) > 1e-3:
                p = p.resize((max(1, int(p.width * kx)), max(1, int(p.height * ky))), Image.LANCZOS)
            out.alpha_composite(p, (int(round(bx0 + (pb[0] - ax0) * kx)), int(round(by0 + (pb[1] - ay0) * ky))))
        self._cache[key] = out
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return out

    def _place(self, PP):
        """(scale image→output, anchor in the image, the anchor's design point)."""
        if PP.get("head"):
            cx, top, chin = PP["head"]
            return self.s * (DESIGN_HEAD[2] - DESIGN_HEAD[1]) / max(chin - top, 1), (cx, top), DESIGN_HEAD[:2]
        return self.s * DESIGN_HEIGHT / max(PP["feet"][1] - PP["top"], 1), tuple(PP["feet"]), DESIGN_FEET

    def draw(self, frame, origin, pid, face, pid_prev=None, u=1.0, motion=(0.0, 0.0)):
        """origin: output pixel of design point (0, 0); pid/pid_prev: atlas poses, u: transition progress 0..1;
        motion: (sway degrees, breathing fraction), both about the feet (they stay planted)."""
        P = self.ch.man["poses"][pid]
        k = self._place(P)[0]
        ims = [(self._pose_image(pid, face), P, 1.0)]
        if pid_prev and pid_prev != pid and u < 0.5:
            # Two complete AI-rendered poses cannot be alpha-blended without showing
            # two heads/faces. Use a single registered pose per frame instead.
            Pp = self.ch.man["poses"][pid_prev]
            ims = [(self._pose_image(pid_prev, face), Pp, 1.0)]
        sway, breathe = motion
        sway_q, breathe_q = round(sway * 10) / 10, round(breathe * 2000) / 2000      # quantized: frames reuse sprites
        sprites = []
        for im, PP, a in ims:
            kk, anchor, dpt = self._place(PP)
            key = ("sprite", id(im), round(kk, 4), sway_q, breathe_q)
            if key in self._cache:
                self._cache.move_to_end(key)
                sc, feet, anc = self._cache[key]
            else:
                sk = ("scaled", id(im), round(kk, 4))
                if sk not in self._cache:
                    self._cache[sk] = im.resize((max(1, int(im.width * kk)), max(1, int(im.height * kk))), Image.LANCZOS)
                sc = self._cache[sk]
                stretch = 1 + breathe_q                 # breathing: stretch upward from the feet
                if breathe_q:
                    sc = sc.resize((sc.width, max(1, int(sc.height * stretch))), Image.BILINEAR)
                feet = (PP["feet"][0] * kk, PP["feet"][1] * kk * stretch)
                anc = (anchor[0] * kk, anchor[1] * kk * stretch)
                if abs(sway_q) > 0.02:                  # sway about the feet
                    sc = sc.rotate(-sway_q, resample=Image.BICUBIC, center=feet)
                self._cache[key] = (sc, feet, anc)
                while len(self._cache) > 160:
                    self._cache.popitem(last=False)
            ax, ay = origin[0] + dpt[0] * self.s, origin[1] + dpt[1] * self.s   # where the anchor lands
            sprites.append((sc, (int(round(ax - anc[0])), int(round(ay - anc[1]))), a))
        if len(sprites) == 1:
            frame.alpha_composite(sprites[0][0], sprites[0][1]) if _inside(frame, *sprites[0][:2]) else \
                _paste_clipped(frame, *sprites[0][:2])
            return k
        # crossfade between two pose images (only during the ~0.45 s transition)
        x0 = min(p[0] for _, p, _ in sprites); y0 = min(p[1] for _, p, _ in sprites)
        x1 = max(p[0] + im.width for im, p, _ in sprites); y1 = max(p[1] + im.height for im, p, _ in sprites)
        layers = []
        for im, pos, a in sprites:
            c = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
            c.alpha_composite(im, (pos[0] - x0, pos[1] - y0))
            layers.append((c, a))
        mix = Image.blend(layers[0][0], layers[1][0], layers[1][1])
        _paste_clipped(frame, mix, (x0, y0))
        return k


def _inside(frame, im, pos):
    return pos[0] >= 0 and pos[1] >= 0 and pos[0] + im.width <= frame.width and pos[1] + im.height <= frame.height


def _paste_clipped(frame, im, pos):
    """alpha_composite that tolerates sprites partly outside the frame (close-ups)."""
    x, y = pos
    l, t = max(0, -x), max(0, -y)
    r, b = min(im.width, frame.width - x), min(im.height, frame.height - y)
    if r > l and b > t:
        frame.alpha_composite(im.crop((l, t, r, b)), (x + l, y + t))


# ------------------------------------------------------------------------------------------- building a version
def content_sha(man):
    """Fingerprint of the artwork only (every file hash + the pose/face geometry), independent of approval fields."""
    body = {"layers": {k: v["sha256"] for k, v in sorted(man["layers"].items())},
            "poses": man.get("poses"), "pose_map": man.get("pose_map"), "face": man.get("face")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def import_atlas(src, root, cid, version, spec, provenance, log=print):
    """src: a folder with the files named in `spec` (poses/*.png, face/*.png); spec: {"poses", "pose_map", "face",
    "identity", "palette"}. Writes a NEW immutable version (staging folder, then rename). Every storybook pose must be
    mapped explicitly (to itself or to a reviewed substitute); every referenced file must exist and be non-empty."""
    import datetime, shutil
    src, vdir = Path(src), Path(root) / "characters" / cid / version
    if (vdir / "manifest.json").exists():
        raise FileExistsError(f"{vdir} exists: versions are immutable (make a new version)")
    unmapped = [p for p in SB.POSES if p not in spec["pose_map"]]
    if unmapped:
        raise ValueError(f"pose_map must map every storybook pose explicitly; missing: {unmapped}")
    bad_targets = [f"{k}→{v}" for k, v in spec["pose_map"].items() if v not in spec["poses"]]
    if bad_targets:
        raise ValueError(f"pose_map points to poses the atlas does not have: {bad_targets}")
    files = [p["file"] for p in spec["poses"].values()]
    F = spec.get("face") or {}
    fname = lambda e: e["file"] if isinstance(e, dict) else e
    files += [fname(f) for f in [F.get("blink")] + list(F.get("visemes", {}).values()) +
              list(F.get("expressions", {}).values()) + list(F.get("wear", {}).values()) if f]
    tmp = vdir.with_name(version + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    layers = {}
    for rel in files:
        f = src / rel
        if not f.exists():
            raise FileNotFoundError(f"atlas file missing: {rel}")
        im = Image.open(f).convert("RGBA")
        if im.getbbox() is None:
            raise ValueError(f"atlas file is empty (fully transparent): {rel}")
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, optimize=True)
        layers[rel] = {"file": rel, "size": list(im.size), "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}
    man = {"character": cid, "version": version, "status": "provisional", "asset_type": ASSET_TYPE,
           "rig_version": SB.RIG_VERSION, "art_version": f"atlas:{provenance.get('model', '?')}",
           "provenance": dict(provenance, created=datetime.datetime.now().isoformat(timespec="seconds"),
                              approved_by=None, approved_at=None),
           "identity": spec.get("identity", {}), "palette": spec.get("palette", {}),
           "poses": spec["poses"], "pose_map": spec["pose_map"], "face": F, "views": sorted({p.get("view", "front") for p in spec["poses"].values()}),
           "expressions": sorted(F.get("expressions", {})), "visemes": sorted(F.get("visemes", {})), "layers": layers}
    man["content_sha256"] = content_sha(man)
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1))
    tmp.rename(vdir)
    log(f"atlas {cid} {version}: {len(layers)} files, {len(spec['poses'])} poses → {vdir}")
    return vdir


def coverage(man):
    """What the atlas provides vs what storybook plans can ask for (for the approval report)."""
    F = man.get("face", {})
    return {"poses_mapped": {p: man["pose_map"].get(p) for p in SB.POSES},
            "own_pose_images": sorted({v for v in man["pose_map"].values()}),
            "views": sorted({p.get("view", "front") for p in man["poses"].values()}),
            "expressions": {e: e in F.get("expressions", {}) for e in SB.EXPRESSIONS},
            "visemes": {v: (MOUTH_OF_VISEME[v] is None or MOUTH_OF_VISEME[v] in F.get("visemes", {})) for v in SB.VISEMES},
            "blink": bool(F.get("blink"))}

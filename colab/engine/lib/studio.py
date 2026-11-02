"""Character Studio: one-time production of APPROVED character artwork (Colab T4), in the storybook layer format.

Workflow (colab/colab_character_studio.ipynb runs these steps; nothing here runs during a routine video render):
  1. concepts   FLUX.2 [klein] 4B (Apache-2.0, pinned revision) turns the Character Bible design text into N
                candidate model sheets per character (fixed seeds, recorded) → contact sheet.
  2. choose     a person picks one candidate per character (STUDIO_PICK); nothing is locked automatically.
  3. variants   reference-conditioned edits of the chosen sheet: expressions, mouth shapes, hands, back view.
  4. cut        BiRefNet (MIT) removes the background of each image; the rig parts are then separated with
                part masks on the rig template (semi-automatic: masks can be corrected by hand in any editor,
                because AI segmentation is never assumed to be right) and hidden regions are painted in.
  5. import     import_layers() writes a NEW version folder (manifest + layers + provenance: model, revision,
                seeds, prompts) in the same format as storybook.build_character — status "provisional".
  6. preview    rig previews (poses, expressions, a speaking test) → contact sheet for review.
  7. approve    colab/approve_character.py marks the version approved (who / when); plans then pin it by hash.
Steps 1–4 need a GPU; they are written for a free T4 (fp16, text encoder in 4-bit, sequential loading) but were
NOT run in this repository's environment (no GPU): their first real run is a benchmark, not a guarantee.
"""
from __future__ import annotations

import datetime, hashlib, json
from pathlib import Path

from PIL import Image, ImageDraw

FLUX_MODEL = "black-forest-labs/FLUX.2-klein-4B"
SEG_MODEL = "ZhengPeng7/BiRefNet"

STYLE = ("premium 2.5D storybook animation character, soft cinematic lighting, gentle shading, rounded appealing "
         "shapes, large expressive eyes, clean readable silhouette, original design, full body model sheet, front "
         "view, standing in a relaxed pose, arms slightly away from the body, plain light grey background, "
         "no text, no logo")
NEGATIVE = "photorealistic, uncanny, creepy, extra fingers, deformed hands, text, watermark, logo, famous character"

DESIGN_TEXT = {
    "arjun": ("a cute young animated boy with warm medium-brown skin, large expressive dark-brown eyes, soft rounded "
              "face, short wavy dark hair with one signature curl on top, teal hoodie jacket open over a warm yellow "
              "T-shirt, dark trousers, orange-red sneakers with white soles, friendly curious smile"),
    "meera": ("a cute young animated girl with warm brown skin, large expressive eyes, soft rounded features, wavy "
              "dark hair in a half-up style with a small yellow star hair clip, coral jacket over a lavender top, dark "
              "blue trousers, lavender sneakers with white soles, confident warm smile, expressive eyebrows"),
}
EXPRESSION_EDITS = {
    "happy": "same character, same outfit, smiling happily with cheeks raised",
    "curious": "same character, curious look, one eyebrow raised, small smile",
    "confused": "same character, confused, eyebrows pulled together, lips to one side",
    "surprised": "same character, surprised, eyes wide open, mouth in a round O",
    "excited": "same character, excited, big open smile, eyes sparkling",
    "thinking": "same character, thinking, eyes looking up to the side, lips pursed",
    "concerned": "same character, concerned, inner eyebrows raised, small frown",
    "laughing": "same character, laughing with eyes closed in happy arcs, mouth wide open",
    "back_view": "same character seen from behind, same hair and outfit",
}


def prompt(cid):
    return f"{DESIGN_TEXT[cid]}, {STYLE}"


def gpu_report():
    """What this session can run (never assumes bf16 on a T4)."""
    import torch
    rep = {"cuda": torch.cuda.is_available(), "torch": torch.__version__}
    if rep["cuda"]:
        p = torch.cuda.get_device_properties(0)
        rep.update(name=p.name, vram_gb=round(p.total_memory / 2 ** 30, 1), capability=f"{p.major}.{p.minor}",
                   native_bf16=p.major >= 8, cuda=torch.version.cuda)
    try:
        import psutil
        rep["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        pass
    return rep


def load_flux(revision, log=print):
    """FLUX.2 [klein] 4B for a T4: transformer fp16 on the GPU, text encoder 4-bit (bitsandbytes), model offload.
    bf16 only on GPUs that support it natively. Raises with a clear message if this diffusers has no pipeline."""
    import torch
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as e:
        raise RuntimeError("this diffusers version has no Flux2KleinPipeline: install diffusers from source "
                           "(see the model card) in the Studio notebook") from e
    rep = gpu_report()
    if not rep.get("cuda"):
        raise RuntimeError("Character generation requires an allocated GPU; run this stage in Colab.")
    dtype = torch.bfloat16 if rep.get("native_bf16") else torch.float16
    kw = {}
    try:
        from diffusers import PipelineQuantizationConfig
        kw["quantization_config"] = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit",
                                                               quant_kwargs={"load_in_4bit": True,
                                                                             "bnb_4bit_compute_dtype": dtype},
                                                               components_to_quantize=["text_encoder"])
    except Exception as e:
        raise RuntimeError("4-bit text encoder unavailable; fix the Studio dependencies before loading weights") from e
    pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, revision=revision, torch_dtype=dtype, **kw)
    pipe.enable_model_cpu_offload()
    log(f"FLUX.2 klein 4B loaded ({dtype}, {rep})")
    return pipe, dtype


def generate(pipe, text, seed, size=768, steps=4, image=None, height=None):
    import numpy as np, torch
    g = torch.Generator("cpu").manual_seed(seed)
    kw = dict(prompt=text, height=height or size, width=size, guidance_scale=1.0, num_inference_steps=steps, generator=g)
    if image is not None:
        kw["image"] = image
    im = pipe(**kw).images[0]
    a = np.asarray(im.convert("L"), np.float32)
    if a.std() < 3:                                     # fp16 overflow → a flat/black image: report, don't keep
        raise FloatingPointError("flat image (fp16 overflow?): stop and review precision on this GPU; do not enable unsupported bf16")
    return im


def contact_sheet(items, out, cols=4, cell=384):
    """items: [(label, PIL image)] → one PNG for the human choice / approval."""
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 34)), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    for i, (lab, im) in enumerate(items):
        x, y = (i % cols) * cell, (i // cols) * (cell + 34)
        t = im.convert("RGB").copy()
        t.thumbnail((cell, cell))
        sheet.paste(t, (x + (cell - t.width) // 2, y + 30))
        d.text((x + 8, y + 8), str(lab), fill=(20, 20, 20))
    sheet.save(out)
    return out


def remove_background(images, revision, device="cuda"):
    """BiRefNet (MIT, pinned) → RGBA images. The mask is a proposal: the cut-out step lets a person fix it."""
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    m = AutoModelForImageSegmentation.from_pretrained(SEG_MODEL, revision=revision, trust_remote_code=True)
    m.to(device).eval()
    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    with torch.no_grad():
        for im in images:
            x = tf(im.convert("RGB")).unsqueeze(0).to(device)
            pred = m(x)[-1].sigmoid().cpu()[0, 0]
            mask = transforms.ToPILImage()(pred).resize(im.size)
            r = im.convert("RGBA")
            r.putalpha(mask)
            out.append(r)
    del m
    torch.cuda.empty_cache()
    return out


REQUIRED_LAYERS = (["hair_back", "legs", "torso", "neck", "head", "hair_front", "back"]
                   + [f"{p}_{s}" for p in ("upper_arm", "forearm") for s in "LR"]
                   + [f"hand_{s}_{h}" for s in "LR" for h in ("open", "point", "wave", "hold")]
                   + [f"eye_{s}_{v}" for s in "LR" for v in ("open", "half", "wide", "smile", "closed", "happy")]
                   + [f"lid_{s}_{v}" for s in "LR" for v in ("open", "half", "wide", "smile")]
                   + [f"eyemask_{s}_{v}" for s in "LR" for v in ("open", "half", "wide", "smile")]
                   + [f"iris_{s}" for s in "LR"] + [f"iris_{s}_small" for s in "LR"]
                   + [f"mouth_{m}" for m in ("rest", "closed", "small", "wide", "round", "teeth", "smile_talk", "smile",
                                             "frown", "wavy", "pursed", "laugh", "o_big")]
                   + [f"brows_{b}" for b in ("neutral", "raised", "furrow", "sad", "curious", "high")])


def import_layers(src_dir, root, cid, version, provenance, res=1.6, rig=None):
    """A folder of full-canvas RGBA PNGs named like REQUIRED_LAYERS (1000x1800 design canvas at `res` px/unit,
    aligned to the rig template) → a NEW immutable version folder. Missing required layers are an error."""
    import storybook as SB
    src = Path(src_dir)
    names = {p.stem: p for p in src.glob("*.png")}
    missing = [n for n in REQUIRED_LAYERS if n not in names]
    if missing:
        raise ValueError(f"missing layers: {missing[:12]}{'…' if len(missing) > 12 else ''}")
    vdir = Path(root) / "characters" / cid / version
    if (vdir / "manifest.json").exists():
        raise FileExistsError(f"{vdir} exists: make a new version")
    tmp = vdir.with_name(version + ".partial")
    (tmp / "layers").mkdir(parents=True, exist_ok=True)
    layers = {}
    for n, p in names.items():
        im = Image.open(p).convert("RGBA")
        bb = im.getbbox()
        if bb is None:
            raise ValueError(f"layer {n} is empty")
        crop = im.crop(bb)
        f = tmp / "layers" / f"{n}.png"
        crop.save(f, optimize=True)
        layers[n] = {"file": f"layers/{n}.png", "offset": [bb[0] / res, bb[1] / res], "size": list(crop.size),
                     "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    import storybook_art as art
    man = {"character": cid, "version": version, "status": "provisional", "rig_version": SB.RIG_VERSION,
           "art_version": f"studio:{provenance.get('model', '?')}", "res": res, "canvas": list(art.CANVAS),
           "provenance": dict(provenance, created=datetime.datetime.now().isoformat(timespec="seconds"),
                              approved_by=None, approved_at=None),
           "palette": provenance.get("palette", {}), "identity": provenance.get("identity", {}),
           "rig": rig or json.loads(json.dumps({"joints": {"shoulder": {s: list(art.SHOULDER[s]) for s in "LR"},
                                                           "elbow": {s: list(art.ELBOW[s]) for s in "LR"},
                                                           "wrist": {s: list(art.WRIST[s]) for s in "LR"},
                                                           "neck": [500, 712], "hips": [500, 1180]},
                                                "limits": SB.JOINT_LIMITS})),
           "poses": SB.POSES, "expressions": {k: list(v) for k, v in SB.EXPRESSIONS.items()},
           "visemes": list(SB.VISEMES), "layers": layers}
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1))
    tmp.rename(vdir)
    return vdir


def rig_template(out, res=1.0):
    """A guide image of the rig canvas (joints, part boxes) to align and cut the generated artwork."""
    import storybook_art as art
    W, H = int(art.CANVAS[0] * res), int(art.CANVAS[1] * res)
    im = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    d = ImageDraw.Draw(im)
    P = lambda p: (p[0] * res, p[1] * res)
    d.ellipse([P((255, 220)), P((745, 720))], outline=(0, 120, 255), width=3)
    for side in "LR":
        for a, b in ((art.SHOULDER[side], art.ELBOW[side]), (art.ELBOW[side], art.WRIST[side])):
            d.line([P(a), P(b)], fill=(255, 80, 0), width=4)
        for j in (art.SHOULDER[side], art.ELBOW[side], art.WRIST[side]):
            x, y = P(j)
            d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(255, 0, 0))
        x, y = P(art.EYE_C[side])
        d.ellipse([x - 60 * res, y - 70 * res, x + 60 * res, y + 70 * res], outline=(0, 160, 0), width=2)
    d.rectangle([P((310, 760)), P((690, 1200))], outline=(120, 0, 200), width=3)
    d.text(P((20, 20)), "rig template: align the artwork to these joints, one PNG per layer", fill=(0, 0, 0))
    im.save(out)
    return out


# ------------------------------------------------------------------------------------------- pose atlas (atlas_v1)
def estimate_head(cutout):
    """Heuristic head geometry of a cut-out FRONT view: (centre_x, head_top, chin, face_box). Skin pixels in the upper
    half of the figure → the face; the chin is the lowest row of that face region. A proposal: the Studio shows it
    drawn on the image and the spec can override it (never trusted blindly)."""
    import numpy as np
    a = np.asarray(cutout.convert("RGBA")).astype(np.int32)
    alpha = a[..., 3] > 128
    ys, xs = np.nonzero(alpha)
    top, bot = int(ys.min()), int(ys.max())
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    skin = alpha & (r > 90) & (r > g + 12) & (g > b + 4) & (r - b > 35) & (r < 250)
    upper = np.zeros_like(skin)
    upper[top:top + int((bot - top) * 0.45)] = True
    face = skin & upper
    rows = np.nonzero(face.sum(1) > max(6, face.sum(1).max() * 0.25))[0]
    if not len(rows):
        raise ValueError("no face found: give the head box by hand in the atlas spec")
    # the face is the biggest band of rows; the chin is where that band ends
    bands, start = [], rows[0]
    for i in range(1, len(rows)):
        if rows[i] != rows[i - 1] + 1:
            bands.append((start, rows[i - 1])); start = rows[i]
    bands.append((start, rows[-1]))
    f0, f1 = max(bands, key=lambda b_: b_[1] - b_[0])
    widths = face.sum(1)
    wmax = widths[f0:f1 + 1].max()
    while f1 > f0 and widths[f1] < 0.55 * wmax:          # the neck is narrower than the face: stop at the chin
        f1 -= 1
    cols = np.nonzero(face[f0:f1 + 1].any(0))[0]
    cx = float((cols.min() + cols.max()) / 2)
    return cx, float(top), float(f1), [float(cols.min()), float(f0), float(cols.max()), float(f1)]


def align_patch(master, edited, box, search=8):
    """Cut `box` (+ margin) out of an EDITED image as a feathered patch for the master, after finding the small shift
    (± search px) that best aligns the edit with the master around the box (edits can drift a few pixels)."""
    import numpy as np
    from PIL import ImageFilter
    m = np.asarray(master.convert("RGB")).astype(np.float32)
    e = np.asarray(edited.convert("RGB").resize(master.size, Image.LANCZOS)).astype(np.float32)
    x0, y0, x1, y1 = [int(v) for v in box]
    pad = 24
    ring = np.ones((y1 - y0 + 2 * pad, x1 - x0 + 2 * pad), bool)
    ring[pad:-pad, pad:-pad] = False                     # compare the surroundings, not the changed face
    best, bd = (0, 0), None
    for dy in range(-search, search + 1, 2):
        for dx in range(-search, search + 1, 2):
            ya, xa = y0 - pad + dy, x0 - pad + dx
            if ya < 0 or xa < 0 or ya + ring.shape[0] > e.shape[0] or xa + ring.shape[1] > e.shape[1]:
                continue
            ref = m[y0 - pad:y1 + pad, x0 - pad:x1 + pad]
            cand = e[ya:ya + ring.shape[0], xa:xa + ring.shape[1]]
            if ref.shape != cand.shape:
                continue
            d = float(np.abs(ref - cand)[ring].mean())
            if bd is None or d < bd:
                bd, best = d, (dx, dy)
    dx, dy = best
    patch = Image.fromarray(e[y0 + dy:y1 + dy, x0 + dx:x1 + dx].astype(np.uint8)).convert("RGBA")
    mask = Image.new("L", patch.size, 0)
    f = max(4, min(patch.size) // 8)
    mask.paste(255, (f, f, patch.width - f, patch.height - f))
    patch.putalpha(mask.filter(ImageFilter.GaussianBlur(f / 2)))
    return patch, {"shift": [dx, dy], "surround_diff": round(bd or 0.0, 2)}


# Edits of the approved full-body master (same seed and settings for every job; the reference image keeps identity).
# kind: pose (a whole new pose image), face (a face patch: blink / mouth shape), expr (an expression patch),
# view (a camera view of the idle pose). wear / hold mark variants the stories need (v3: 3D glasses, popcorn).
_KEEP = (" Preserve the exact same character: face, skin tone, eye colour, hairstyle, accessories, outfit, body "
         "proportions, lighting, art style, camera distance and plain light warm-grey background. Full body, head to "
         "shoes, nothing cropped. Do not redesign the character.")
_FACE = (" Change ONLY the face as described; keep everything else in the image identical: pose, body, hands, "
         "clothes, hair, framing, lighting.")
ATLAS_JOBS = [
    # ---- the 13 storybook poses (nod uses the listen image; the head nod is motion)
    ("idle_front", "pose", "master", None),
    ("listen_front", "pose", "Same character standing relaxed, listening attentively, hands relaxed at his sides, head tilted slightly." + _KEEP, None),
    ("explain_front", "pose", "Same character explaining something, one open hand raised to chest height, palm up, friendly." + _KEEP, None),
    ("open_hands_front", "pose", "Same character with both hands open at waist height, palms up, a gentle questioning shrug." + _KEEP, None),
    ("point_front", "pose", "Same character pointing with one index finger toward the viewer's right, arm extended." + _KEEP, None),
    ("think_front", "pose", "Same character thinking, one hand touching the chin, eyes looking up to the side." + _KEEP, None),
    ("wave_front", "pose", "Same character waving hello with one hand raised beside the head, smiling." + _KEEP, None),
    ("surprise_front", "pose", "Same character surprised, both hands raised near the shoulders, eyes wide, mouth open." + _KEEP, None),
    ("celebrate_front", "pose", "Same character celebrating with both arms raised high, big happy smile." + _KEEP, None),
    ("duck_front", "pose", "Same character ducking down a little with both forearms raised above the head, as if something flies at them, startled but playful." + _KEEP, None),
    ("hold_popcorn_front", "pose", "Same character holding a red-and-white striped popcorn bucket with both hands at the chest." + _KEEP, {"hold": "popcorn", "base": "hold"}),
    ("show_glasses_front", "pose", "Same character holding up a pair of dark 3D cinema glasses in one hand to show them." + _KEEP, {"hold": "3d_glasses", "base": "show"}),
    # ---- what the 3D-glasses story needs (glasses are a face patch, see "wear_3d_glasses" below)
    ("duck_popcorn_front", "pose", "Same character ducking down a little with both forearms raised above the head, still holding a red-and-white striped popcorn bucket in one hand, startled but playful." + _KEEP, {"hold": "popcorn", "base": "duck"}),
    # ---- camera views (of the idle pose)
    ("idle_three_quarter_right", "view", "Same character in a natural three-quarter view, turned slightly toward the viewer's right." + _KEEP, {"view": "three_quarter_right", "base": "idle"}),
    ("idle_three_quarter_left", "view", "Same character in a natural three-quarter view, turned slightly toward the viewer's left." + _KEEP, {"view": "three_quarter_left", "base": "idle"}),
    ("idle_back", "view", "Same character seen from directly behind, standing, same hair and outfit." + _KEEP, {"view": "back", "base": "idle"}),
    # ---- face patches on the master (blink + the mouth shapes the lip sync uses)
    ("blink", "face", "The eyes are gently closed (a natural blink)." + _FACE, None),
    ("mouth_closed", "face", "The lips are pressed together, as when saying 'm'." + _FACE, None),
    ("mouth_small", "face", "The mouth is slightly open, as in quiet speech." + _FACE, None),
    ("mouth_wide", "face", "The mouth is open wide, as when saying 'ah'." + _FACE, None),
    ("mouth_round", "face", "The lips are rounded, as when saying 'oo'." + _FACE, None),
    ("mouth_teeth", "face", "The mouth is wide with the upper teeth showing, as when saying 'ee'." + _FACE, None),
    ("mouth_smile_talk", "face", "Smiling with the mouth open, as when talking happily." + _FACE, None),
    ("wear_3d_glasses", "face", "The character now wears dark 3D cinema glasses with a black frame." + _FACE, {"wear": "3d_glasses"}),
    # ---- expressions (patches; neutral = the master)
] + [(f"expr_{e}", "expr", f"The facial expression is {d}." + _FACE, None) for e, d in (
    ("listening", "calm and attentive"), ("smile", "a gentle closed-mouth smile"), ("happy", "happy, cheeks raised"),
    ("curious", "curious, one eyebrow raised"), ("confused", "confused, eyebrows pulled together"),
    ("surprised", "surprised, eyebrows raised, eyes wide, mouth in a small 'oh'"), ("excited", "excited, big open smile"),
    ("thinking", "thinking, eyes looking up to the side, lips pursed"), ("concerned", "concerned, inner eyebrows raised, small frown"),
    ("laughing", "laughing, eyes closed in happy arcs, mouth open"), ("wink", "winking with one eye, smiling"))]
POSE_MAP_DEFAULT = {"idle": "idle_front", "listen": "listen_front", "explain": "explain_front",
                    "open_hands": "open_hands_front", "point": "point_front", "think": "think_front",
                    "wave": "wave_front", "surprise": "surprise_front", "celebrate": "celebrate_front",
                    "duck": "duck_front", "hold": "hold_popcorn_front", "show": "show_glasses_front",
                    "nod": "listen_front"}          # nod = the listen image + the renderer's head-nod motion


def patch_region(job):
    """Which part of the face an edit is allowed to change: the eye band (blink, glasses), the mouth band (mouth
    shapes) or the whole face (expressions) — so patches can be stacked without erasing each other."""
    if job == "blink" or job.startswith("wear_"):
        return "eyes"
    if job.startswith("mouth_"):
        return "mouth"
    return "face"


def region_box(fbox, region):
    x0, y0, x1, y1 = fbox
    h = y1 - y0
    if region == "eyes":
        return [x0, y0, x1, y0 + 0.6 * h]
    if region == "mouth":
        return [x0 + 0.15 * (x1 - x0), y0 + 0.55 * h, x1 - 0.15 * (x1 - x0), y1 + 0.05 * h]
    return list(fbox)


def build_atlas_source(studio_dir, cid, log=print):
    """Studio outputs (<studio>/<cid>/atlas/<job>.png + the master) → an atlas source folder + spec.json, ready for a
    person to review (and override head/face boxes) before storybook_atlas.import_atlas. Pose images get their
    background removed; face/expression edits become aligned, feathered patches of the master's face box."""
    import numpy as np
    sd = Path(studio_dir) / cid
    src = sd / "atlas_src"
    (src / "poses").mkdir(parents=True, exist_ok=True); (src / "face").mkdir(parents=True, exist_ok=True)
    cut = sd / "cutouts"
    master = Image.open(sd / "atlas" / "idle_front.png")
    mcut = Image.open(cut / "idle_front.png")
    cx, top, chin, fbox = estimate_head(mcut)
    spec = {"poses": {}, "pose_map": {}, "face": {"anchor_pose": "idle_front", "box": fbox, "visemes": {}, "expressions": {}},
            "alignment": {}}
    for job, kind, _, extra in ATLAS_JOBS:
        f = sd / "atlas" / f"{job}.png"
        if not f.exists():
            log(f"{cid}: {job} not generated yet — skipped"); continue
        if kind in ("pose", "view"):
            c = Image.open(cut / f"{job}.png")
            c.save(src / "poses" / f"{job}.png")
            bb = c.getbbox()
            a = np.asarray(c.getchannel("A")) > 128
            cols = np.nonzero(a[bb[3] - 30:bb[3]].any(0))[0]
            entry = {"file": f"poses/{job}.png", "view": (extra or {}).get("view", "front"),
                     "feet": [float(cols.mean()), float(bb[3])], "top": float(bb[1]),
                     "wear": (extra or {}).get("wear", []), "hold": (extra or {}).get("hold")}
            if (extra or {}).get("base"):
                entry["base"] = extra["base"]
            if entry["view"] != "back":
                try:
                    hx, ht, hc, hb = estimate_head(c)
                    entry["head"] = [hx, ht, hc]
                    if entry["view"] == "front":
                        entry["face"] = {"box": hb}
                except ValueError as e:
                    log(f"{cid}/{job}: {e}")
            spec["poses"][job] = entry
        else:
            region = patch_region(job)
            box = region_box(fbox, region)
            patch, info = align_patch(master, Image.open(f), box)
            rel = f"face/{job}.png"
            patch.save(src / rel)
            spec["alignment"][job] = dict(info, region=region, box=box)
            entry = {"file": rel, "box": box}
            if job == "blink":
                spec["face"]["blink"] = entry
            elif job.startswith("wear_"):
                spec["face"].setdefault("wear", {})[job.replace("wear_", "")] = entry
            elif kind == "face":
                spec["face"]["visemes"][job.replace("mouth_", "")] = entry
            else:
                spec["face"]["expressions"][job.replace("expr_", "")] = entry
    spec["pose_map"] = {k: v for k, v in POSE_MAP_DEFAULT.items() if v in spec["poses"]}
    (src / "spec.json").write_text(json.dumps(spec, indent=1))
    log(f"{cid}: atlas source with {len(spec['poses'])} poses, {len(spec['face']['visemes'])} mouth shapes, "
        f"{len(spec['face']['expressions'])} expressions → {src / 'spec.json'} (review, then import)")
    return src, spec

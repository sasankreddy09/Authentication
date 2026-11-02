"""Character Studio engine: runs image jobs (proof of concept, then the pose atlas) once, resumably, with provenance.

Backends — the same model either way, FLUX.2 [klein] 4B (Apache-2.0), distilled, 4 steps, guidance 1.0:
  LocalGen   diffusers on this session's GPU (Colab T4: fp16, 4-bit text encoder, model offload — studio.load_flux);
             records model load time, per-image time, peak VRAM (allocated / reserved) and host RAM.
  HostedGen  the official Hugging Face Space black-forest-labs/FLUX.2-klein-4B (shared ZeroGPU) via gradio_client.
             Optional; quota-limited (measured 2026-10-20: 3 anonymous runs, then "exceeded your ZeroGPU runs
             limit"); a HF token (env HF_TOKEN) gives the account's own free quota. Never needed for videos.
run_jobs(): every job writes <name>.png + <name>.json (prompt, seed, model, revision, backend, size, references and
their sha256, seconds, memory, sha256 of the image). A finished job (image sha matches its sidecar) is skipped, so a
disconnect or an exhausted quota loses nothing; an image is written to *.partial.png and renamed when complete.
Out of memory / quota: the job is reported with the reason and a retry setting; finished images stay; nothing is
replaced by procedural art.
"""
from __future__ import annotations

import gc, hashlib, json, time
from pathlib import Path

SPACE = "black-forest-labs/FLUX.2-klein-4B"
STEPS, GUIDANCE = 4, 1.0

STYLE = ("Premium cinematic 3D-style storybook character art, high-end animated-feature quality, dimensional face "
         "sculpting, gently subsurface-shaded skin, layered natural eye reflections, warm cinematic bounce lighting, "
         "soft golden rim light, nuanced soft shadows, physically believable materials, harmonious colours. "
         "Original character design, stylized child proportions, whimsical but refined, not photorealistic, "
         "not a flat vector cartoon, no text, no logo.")
DESIGN = {
    "arjun": "Use the supplied boy's identity: side-swept tousled medium-brown hair, large warm brown eyes, rounded face and gentle smile. Red zip jacket with pale blue hood lining, yellow striped T-shirt, teal cuffed trousers, red sneakers. Preserve the visible outfit, hair silhouette and proportions while translating the flat reference to dimensional cinematic materials and soft lighting.",
    "meera": "Use the supplied girl's identity: large luminous brown eyes, rosy rounded cheeks, long auburn hair in a high ponytail with a teal hair tie and loose curved bangs. Pale cream short-sleeved floral T-shirt with pink/coral flowers. Preserve her face, ponytail, floral pattern and dimensional warm indoor lighting. Use a simple neutral lower outfit only where the cropped reference does not show it; this needs human review.",
}
IDENTITY = {cid: {"reference_direction": text, "design_revision": "user-references-v2"} for cid, text in DESIGN.items()}
PORTRAIT = "Head-and-shoulders portrait, facing the viewer, softly blurred warm background. "
FULL = ("Full-body production master: the whole character from head to shoes, standing straight in a relaxed "
        "neutral pose facing the viewer, arms slightly away from the body, both hands fully visible, even soft "
        "studio lighting, plain light warm-grey background, centred, nothing cropped. ")
EXPR = ("Using the reference image as the identity reference, create the exact same character with a new facial "
        "expression: natural, joyful surprise (eyebrows raised, eyes wide, mouth open in a happy 'oh'). Preserve "
        "the face structure, skin tone, eye colour, hair design, accessories, clothing, lighting, art style and "
        "camera framing. Do not redesign the character.")
VIEW = ("Using the reference image as the identity reference, show the exact same character in a natural "
        "three-quarter view, turned slightly to the viewer's right, full body head to shoes, plain light warm-grey "
        "background. Preserve the same face, hairstyle, clothing, colour palette, body proportions and accessories, "
        "and the same sculpted 3D-style look and lighting. Do not introduce a different face or hairstyle.")
BACK = ("Using the full-body reference, show the exact same character directly from behind, "
        "head and torso facing away, no face or eyes visible, arms relaxed, full body including shoes. "
        "Preserve hair colour and silhouette, ponytail or hood, outfit colours, proportions, lighting "
        "and plain warm-grey background. Infer only unseen rear clothing details; no mirrored front view.")
REF_NOTE = " The supplied image is the character identity reference. Preserve its face, hairstyle and wardrobe; use dimensional cinematic rendering, not flat outlines."
# proof of concept (claude.md 5.1–5.2): name, prompt builder, own reference job (identity), size, seed
POC_JOBS = [
    ("portrait", lambda c: PORTRAIT + DESIGN[c] + " " + STYLE, None, (640, 640), 11),
    ("full_body_master", lambda c: FULL + DESIGN[c] + " " + STYLE, "portrait", (640, 832), 21),
    ("expression_surprise", lambda c: EXPR, "portrait", (640, 640), 31),
    ("view_three_quarter", lambda c: VIEW, "full_body_master", (640, 832), 41),
    ("view_back", lambda c: BACK, "full_body_master", (640, 832), 51),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class QuotaExhausted(RuntimeError):
    pass


class HostedGen:
    name = "hosted"

    def __init__(self, log=print):
        from gradio_client import Client
        self.client = Client(SPACE)                 # HF_TOKEN from the environment if set (the account's quota)
        self.log = log
        self.revision = "hosted Space (its own pinned weights)"

    def __call__(self, prompt, refs, size, seed):
        from gradio_client import handle_file
        from PIL import Image
        try:
            res, used = self.client.predict(prompt=prompt, input_images=[{"image": handle_file(str(r)), "caption": None}
                                                                         for r in refs],
                                            mode_choice="Distilled (4 steps)", seed=seed, randomize_seed=False,
                                            width=size[0], height=size[1], num_inference_steps=STEPS,
                                            guidance_scale=GUIDANCE, prompt_upsampling=False, api_name="/infer")
        except Exception as e:
            if any(w in str(e).lower() for w in ("quota", "exceeded", "runs limit")):
                raise QuotaExhausted(str(e)[:300]) from e
            raise
        return Image.open(res["path"] if isinstance(res, dict) else res).convert("RGB"), int(used), {}


class LocalGen:
    name = "local"

    def __init__(self, revision, log=print):
        import torch, studio
        self.torch, self.log, self.revision = torch, log, revision
        t0 = time.time()
        self.pipe, self.dtype = studio.load_flux(revision, log)
        self.load_s = round(time.time() - t0, 1)
        log(f"model load: {self.load_s}s ({self.dtype})")

    def __call__(self, prompt, refs, size, seed):
        import studio
        from PIL import Image
        t = self.torch
        if t.cuda.is_available():
            t.cuda.reset_peak_memory_stats()
        try:
            im = studio.generate(self.pipe, prompt, seed, size[0], STEPS,
                                 image=[Image.open(r).convert("RGB") for r in refs] or None, height=size[1])
        except t.cuda.OutOfMemoryError as e:
            self.unload()
            raise MemoryError(f"GPU out of memory at {size[0]}x{size[1]} with {len(refs)} reference image(s): "
                              "retry with SIZE = 640 (finished images are kept)") from e
        mem = {}
        if t.cuda.is_available():
            mem = {"vram_peak_allocated_gb": round(t.cuda.max_memory_allocated() / 2 ** 30, 2),
                   "vram_peak_reserved_gb": round(t.cuda.max_memory_reserved() / 2 ** 30, 2)}
        return im, seed, mem

    def free_cache(self):
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def unload(self):
        if hasattr(self, "pipe"):
            del self.pipe
        self.free_cache()


def host_ram_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2 ** 30, 2)
    except Exception:
        return None


def run_job(gen, out_png, prompt, refs, size, seed, meta_extra, log=print):
    """One image, once. Returns "made" | "skipped" | "waiting" | "failed"; raises QuotaExhausted."""
    out_png = Path(out_png)
    side = out_png.with_suffix(".json")
    if any(not Path(r).is_file() for r in refs):
        raise FileNotFoundError("Missing conditioning reference; no text-only fallback is allowed")
    request = {"prompt": prompt, "reference_sha256": [sha(r) for r in refs], "size": list(size),
               "seed": seed, "revision": gen.revision, "backend": gen.name, "steps": STEPS, "guidance": GUIDANCE}
    request_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    if out_png.exists() or side.exists():
        if out_png.exists() and side.exists():
            old = json.loads(side.read_text())
            if old.get("sha256") == sha(out_png) and old.get("request_key") == request_key:
                return "skipped"
        raise FileExistsError(f"{out_png}: existing output differs or is incomplete; use a new candidate directory")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        im, used, mem = gen(prompt, refs, size, seed)
    except QuotaExhausted:
        raise
    except Exception as e:
        log(f"{out_png.stem}: FAILED after {time.time() - t0:.1f}s — {type(e).__name__}: {str(e)[:300]}")
        return "failed"
    tmp = out_png.with_name(out_png.stem + ".partial.png")
    im.save(tmp)
    tmp.rename(out_png)                                 # the whole image or nothing
    m = dict(meta_extra, request_key=request_key, prompt=prompt, seed=int(used), model="black-forest-labs/FLUX.2-klein-4B",
             revision=gen.revision, backend=gen.name, mode=f"distilled, {STEPS} steps, guidance {GUIDANCE}",
             size=list(size), references=[Path(r).name for r in refs], reference_sha256=[sha(r) for r in refs],
             seconds=round(time.time() - t0, 1), host_ram_gb=host_ram_gb(), sha256=sha(out_png),
             created=time.strftime("%Y-%m-%dT%H:%M:%S"), **mem)
    side_tmp = side.with_suffix(".json.partial")
    side_tmp.write_text(json.dumps(m, indent=1))
    side_tmp.replace(side)
    log(f"{out_png.parent.name}/{out_png.stem}: {m['seconds']}s" + (f", VRAM peak {mem['vram_peak_allocated_gb']} GB" if mem else ""))
    return "made"


def run_poc(gen, out, characters=("arjun", "meera"), steps=("portrait", "full_body_master", "expression_surprise",
                                                             "view_three_quarter", "view_back"), style_refs=None, log=print):
    """The proof-of-concept set (5.2). style_refs: {cid: [paths]} art-direction references for the first images."""
    out = Path(out)
    for cid in characters:
        if not (style_refs or {}).get(cid) or not all(Path(p).is_file() for p in style_refs[cid]):
            raise FileNotFoundError(f"Supply the actual {cid} reference image in style_refs")
    status = {}
    for cid in characters:
        for name, pb, ref, size, seed in POC_JOBS:
            if name not in steps:
                continue
            refs = [out / cid / f"{ref}.png"] if ref else list((style_refs or {}).get(cid, []))
            prompt = pb(cid) + (REF_NOTE if (not ref and refs) else "")
            status[f"{cid}/{name}"] = run_job(gen, out / cid / f"{name}.png", prompt, refs, size, seed,
                                              {"character": cid, "image": name, "stage": "proof_of_concept"}, log)
    return status


def run_atlas(gen, studio_dir, cid, size=(768, 1024), seed=2000, only=None, log=print):
    """All ATLAS_JOBS for one character from its APPROVED full-body master (<studio>/<cid>/atlas/idle_front.png)."""
    import studio
    d = Path(studio_dir) / cid / "atlas"
    master = d / "idle_front.png"
    import cinematic_workflow as CW
    CW.require_review(Path(studio_dir) / cid / "design_approval.json", [master], "design")
    CW.require_review(Path(studio_dir) / cid / "animation_approval.json", [master], "animation")
    if not master.exists():
        raise FileNotFoundError(f"{master}: copy the approved full-body master here first (the proof-of-concept gate)")
    status = {}
    for job, kind, prompt, extra in studio.ATLAS_JOBS:
        if kind == "pose" and prompt == "master":
            continue
        if only and job not in only:
            continue
        status[job] = run_job(gen, d / f"{job}.png", prompt, [master], size, seed,
                              {"character": cid, "image": job, "stage": "atlas", "kind": kind, "variant": extra}, log)
    return status

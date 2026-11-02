# ==== 8. Keyframe images: only made when needed ====
# L4/A100 (image-to-video): every shot starts from a keyframe picture, so they are made now.
# T4 (text-to-video): the video model doesn't use them, so they are skipped (saves the ~7 GB SDXL download, its RAM
# and ~10 min); a keyframe is only made later for a shot whose video failed, as its fallback picture.
from PIL import Image, ImageDraw, ImageFilter, ImageOps

def procedural_space(seed, w=1344, h=768):
    """Fallback image when no GPU is available: stars + soft nebula."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for _ in range(6):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(120, 380)
        col = rng.uniform(0.1, 0.6, 3)
        yy, xx = np.ogrid[:h, :w]
        img += col * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))[..., None] * 0.5
    ys, xs = rng.integers(0, h, 1500), rng.integers(0, w, 1500)
    img[ys, xs] = rng.uniform(0.5, 1.0, (1500, 1))
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))

def load_image_pipe():
    global IMAGE_MODEL
    try:
        import torch
        from diffusers import DiffusionPipeline
    except Exception as e:
        log_exc("diffusers import failed")
        raise RuntimeError("Could not load the image library (diffusers). Click Runtime → Restart session, "
                           "then Runtime → Run all. If it happens again, send this error to Claude:\n" + repr(e)[:500])
    if IMAGE_MODEL == "z-image-turbo":
        try:
            p = DiffusionPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
            p.enable_model_cpu_offload()
            return p
        except Exception as e:
            log_exc("Z-Image-Turbo could not load")
            print("Z-Image-Turbo could not load:", repr(e)[:300], "\n→ falling back to SDXL")
            IMAGE_MODEL = "sdxl"
    kw = dict(torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    try:                                           # load straight onto the GPU (keeps system RAM free on a T4)
        p = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", device_map="cuda", **kw)
    except Exception:
        log_exc("SDXL device_map='cuda' load failed, loading normally")
        p = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", **kw)
    return p.to("cuda")

def needs_image(shot):
    out = PROJ / "keyframes" / f"{shot['shot_id']}.png"
    sc = read_sidecar(out)
    return (not out.exists() or sc is None or sc.get("prompt") != f"{shot['image_prompt']}, {STYLE}"
            or sc.get("seed") != shot["seed"] or (sc.get("model") == "procedural" and IMAGE_MODEL != "procedural"))

def make_keyframes(shots):
    todo = [s for s in shots if s.get("image_prompt") and needs_image(s)]
    if not todo:
        return
    pipe = load_image_pipe() if IMAGE_MODEL != "procedural" else None
    for shot in todo:
        out = PROJ / "keyframes" / f"{shot['shot_id']}.png"
        prompt = f"{shot['image_prompt']}, {STYLE}"
        t0, used = time.time(), IMAGE_MODEL
        try:
            if pipe is None:
                raise RuntimeError("no GPU image model")
            import torch
            g = torch.Generator("cpu").manual_seed(shot["seed"])
            if IMAGE_MODEL == "z-image-turbo":
                img = pipe(prompt=prompt, height=768, width=1344, num_inference_steps=9,
                           guidance_scale=0.0, generator=g).images[0]
            else:
                img = pipe(prompt=prompt, negative_prompt=NEG, height=768, width=1344,
                           num_inference_steps=30, guidance_scale=6.0, generator=g).images[0]
        except Exception as e:
            log_exc(f"{shot['shot_id']}: image generation failed")
            if pipe is not None:
                print(f"  {shot['shot_id']}: image generation failed ({repr(e)[:200]}), using procedural fallback")
            img, used = procedural_space(shot["seed"]), "procedural"
        img.save(out)
        sidecar(out, model=used, prompt=prompt, seed=shot["seed"], seconds=round(time.time() - t0, 1))
        print(f"  {shot['shot_id']}: {used} ({time.time() - t0:.0f}s)")
    del pipe; free_gpu()
    print("Keyframes ready."); ram()

if VIDEO_MODE == "t2v":
    print("Text-to-video mode: keyframe pictures skipped (only made later as fallbacks for shots without video).")
else:
    make_keyframes(SHOTS)

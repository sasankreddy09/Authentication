# ==== 5b. Start the big downloads in the background (they run while the voices are made) ====
# Only what is not cached yet: the text encoder (11.4 GB, only if some prompt has never been encoded) and the
# video model (only if it is not in the Drive model cache). The video step waits for them if they are not done.
import threading
def video_mode():
    if VIDEO_ENGINE != "wan22-5b":
        return "t2v"
    try:
        from diffusers import WanImageToVideoPipeline  # noqa: F401
        return "i2v"                                   # animates each shot's keyframe
    except Exception:
        log_exc("WanImageToVideoPipeline not available, using text-to-video")
        return "t2v"

VIDEO_MODE = video_mode() if VIDEO_ENGINE else None

def real_bf16():
    import torch
    return torch.cuda.get_device_capability(0)[0] >= 8           # T4 = 7.5 (no real bf16), L4/A100 = 8.x

VIDEO_MID = {"wan22-5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "wan21-1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}
NEG_V = NEG + ", static, still image, frozen, slideshow"

def clip_prompt(shot, mode):
    return (f"{shot['motion_prompt']}, {STYLE}" if mode == "i2v"
            else f"{shot['image_prompt']}, {shot['motion_prompt']}, {STYLE}")

# Wan 2.1 and 2.2 share the same UMT5-XXL text encoder and tokenizer. The 2.2 repo stores it in bf16
# (11.4 GB download) while the 2.1 repo stores the same weights as fp32 (22.7 GB), so always take the small one.
TEXT_ENCODER_MIDS = ["Wan-AI/Wan2.2-TI2V-5B-Diffusers", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"]
# encoded prompts, shared by ALL projects (in the model cache on Drive): any prompt encoded once, by any video,
# never needs the 11 GB text encoder again. Older runs kept them per project; those are still found.
EMB_CACHE = MODEL_CACHE / "prompt_cache"
OLD_EMB_CACHE = PROJ / "prompt_cache"

def emb_path(text):
    name = hashlib.sha1(text.encode()).hexdigest()[:16] + ".pt"
    return EMB_CACHE / name if (EMB_CACHE / name).exists() or not (OLD_EMB_CACHE / name).exists() else OLD_EMB_CACHE / name

def prefetch_jobs():
    """(what, repo, snapshot_download arguments) for everything the video step will need and doesn't have."""
    if not VIDEO_ENGINE or not any(s.get("motion_prompt") for s in SHOTS):
        return []
    jobs, mid = [], VIDEO_MID[VIDEO_ENGINE]
    prompts = {clip_prompt(s, VIDEO_MODE) for s in SHOTS if s.get("motion_prompt")} | {NEG_V}
    new = [t for t in prompts if not emb_path(t).exists()]
    if new:
        jobs.append((f"text encoder (11.4 GB) for {len(new)} new prompts", TEXT_ENCODER_MIDS[0],
                     {"allow_patterns": ["text_encoder/*", "tokenizer/*"]}))
    dt_name = "bfloat16" if GPU["bf16"] else "float16"
    states = globals().get("STATES") or {}
    vace = [s for s in SHOTS if s.get("motion_prompt") and VIDEO_RECIPE == "auto" and s["shot_id"] in states]
    if len(vace) < sum(1 for s in SHOTS if s.get("motion_prompt")):   # some shots use plain text-to-video
        skip = ["text_encoder/*"]
        if VIDEO_ENGINE == "wan21-1.3b" and (MODEL_CACHE / f"{VIDEO_ENGINE}-{dt_name}" / "complete").exists():
            skip += ["transformer/*", "vae/*"]    # loads from the Drive model cache: only small config files
        jobs.append((f"video model {mid}", mid, {"ignore_patterns": skip}))
    if vace:                                      # scene-state shots use the VACE layout model, not text-to-video
        skip = ["text_encoder/*"]
        if (MODEL_CACHE / f"wan21-vace-1.3b-{dt_name}" / "complete").exists():
            skip += ["transformer/*", "vae/*"]
        jobs.append(("VACE layout model Wan-AI/Wan2.1-VACE-1.3B-diffusers", "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                     {"ignore_patterns": skip}))
    return jobs

def _prefetch(jobs):
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()                       # no download bars printed into the voice cell's output
    for what, repo, kw in jobs:
        t0 = time.time()
        try:
            snapshot_download(repo, **kw)
            log(f"prefetch: {what} ready in {time.time() - t0:.0f}s")
        except Exception:
            log_exc(f"prefetch of {what} failed (it is downloaded again when needed)")

def wait_for_prefetch():
    th = PREFETCH.get("thread")
    if th is not None and th.is_alive():
        t0 = time.time()
        print("Waiting for the background downloads to finish...")
        th.join()
        print(f"  waited {time.time() - t0:.0f}s")
    try:
        from huggingface_hub.utils import enable_progress_bars
        enable_progress_bars()
    except Exception:
        pass

PREFETCH = {}
_jobs = prefetch_jobs()
if _jobs:
    try:
        need_disk(20 if any("text encoder" in j[0] for j in _jobs) else 8, "the background downloads")
        PREFETCH["thread"] = threading.Thread(target=_prefetch, args=(_jobs,), daemon=True)
        PREFETCH["thread"].start()
        print("Downloading in the background while the voices are made:", "; ".join(j[0] for j in _jobs))
    except Exception as e:
        log_exc("background downloads not started")
        print("Background downloads not started (they happen in the video step instead):", repr(e)[:200])
else:
    print("Nothing to download in the background (no GPU, or everything is cached).")

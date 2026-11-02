# ==== 3. Google Drive, folders and GPU check ====
import os, json, subprocess, shutil, math, random, gc, time, re, difflib, hashlib
# GPU memory fragmentation: lets PyTorch grow and reuse one big block instead of many that can't merge
# ("reserved but unallocated" memory in out-of-memory errors). Must be set before CUDA starts; no speed cost.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Faster parallel Hugging Face downloads, only on big-RAM machines: the free tier (~12.7 GB) downloads in the
# background while the voice models run, so the download keeps its default (smaller) buffers there.
try:
    _RAM_TOTAL_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
except (ValueError, OSError, AttributeError):
    _RAM_TOTAL_GB = 0
if _RAM_TOTAL_GB >= 24:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")      # read at import
from pathlib import Path
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Not running in Colab: using local folders")

PROJ = Path(DRIVE_ROOT) / PROJECT                      # persistent (Drive)
WORK = Path(os.environ.get("WORK_DIR", "/content/work")) / PROJECT   # fast local scratch
for d in ["narration", "keyframes", "clips", "render"]:
    (PROJ / d).mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
# models kept between sessions: on Drive (shared by all projects) or, if turned off, only for this session
MODEL_CACHE = (Path(DRIVE_ROOT) if MODEL_CACHE_ON_DRIVE else WORK.parent) / "_model_cache"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
if not LOG.path.startswith(str(PROJ)):
    LOG.move_to(PROJ / "logs" / os.path.basename(LOG.path))
print("Run log:", LOG.path)

GPU = {"name": None, "vram_gb": 0.0, "bf16": False}
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        GPU = {"name": p.name, "vram_gb": round(p.total_memory / 2**30, 1),
               "bf16": torch.cuda.get_device_capability(0)[0] >= 8,      # real bf16 (T4 only emulates it)
               "sm": "%d.%d" % torch.cuda.get_device_capability(0)}
except ImportError:
    pass

BIG_GPU = GPU["bf16"] and GPU["vram_gb"] >= 20                 # L4, A100, H100, G4 ...
IMAGE_MODEL  = "z-image-turbo" if BIG_GPU else "sdxl" if GPU["vram_gb"] >= 12 else "procedural"
VIDEO_ENGINE = "wan22-5b" if BIG_GPU else "wan21-1.3b" if GPU["vram_gb"] >= 12 else None
if globals().get("RUN_MODE") == "cpu_fallback":            # no AI video or AI images, even if a GPU is there
    VIDEO_ENGINE, IMAGE_MODEL = None, "procedural"
print(f"GPU: {GPU['name'] or 'none'} | VRAM {GPU['vram_gb']} GB | bf16 {GPU['bf16']}"
      + (f" | compute capability {GPU['sm']}" if GPU.get("sm") else ""))
if GPU.get("sm"):
    # T4 = 7.5: fp16 tensor cores; no native bf16 or fp8; FlashAttention-2 needs 8.0+. PyTorch's attention then uses
    # its memory-efficient kernel automatically, so nothing T4-incompatible is ever forced.
    try:
        log(f"GPU kernels: fp16 tensor cores {float(GPU['sm']) >= 7.0}, native bf16 {GPU['bf16']}, "
            f"flash attention possible {float(GPU['sm']) >= 8.0}, SDPA flash/mem-efficient/math enabled: "
            f"{torch.backends.cuda.flash_sdp_enabled()}/{torch.backends.cuda.mem_efficient_sdp_enabled()}/"
            f"{torch.backends.cuda.math_sdp_enabled()}")
    except Exception:
        log_exc("could not read the attention kernel flags")
if not GPU["name"]:
    print("⚠️  No GPU in this session. On the free tier this usually means today's GPU quota is used up, or the runtime\n"
          "    type is CPU. The notebook still finishes (voices, drawn scenes, music, editing), but AI video shots become\n"
          "    still pictures with a camera move. Later: Runtime → Change runtime type → T4 GPU → Run all; only the\n"
          "    missing AI shots are made, everything else is reused.")

def free_tier_estimate(n_ai, n_drawn):
    """Rough time for this video on a free T4, and the free-tier session rules worth knowing."""
    first = 15 + 4 * n_ai + 2                     # downloads (~15 min, once) + ~3–5 min per AI clip + editing
    again = 3 + 4 * n_ai + 2
    print(f"Estimated time on a free T4: ~{first} min the first time (model downloads), ~{again} min in a new session "
          f"once models are cached on Drive; re-runs only remake what changed. ({n_ai} AI clips, {n_drawn} drawn scenes)")
    print("Free tier: keep this browser tab open (idle sessions disconnect), sessions last at most ~12 h, and GPU time\n"
          "per day is limited. If it disconnects: reconnect → Run all; finished work is saved on Drive.")
print(f"Image model: {IMAGE_MODEL} | Video model: {VIDEO_ENGINE or 'none'}")
if VIDEO_ENGINE == "wan22-5b":
    print("✅ Real AI video for every shot: Wan 2.2 (1280×704, 24 fps).")
elif VIDEO_ENGINE == "wan21-1.3b":
    print("⚠️  This GPU (T4) only runs the small Wan 2.1 model: real video, but lower quality (832×480, upscaled)\n"
          "    turbo mode makes it ~3–5 min per shot (25 min without it). For better cinematic video use an L4 or A100:\n"
          "    Runtime → Change runtime type → L4 / A100 (needs Colab Pay-As-You-Go or Pro credits).")
else:
    print("❌ No usable GPU: shots will be camera moves over still images only. "
          "Use Runtime → Change runtime type → GPU.")
try:
    import psutil
    RAM_GB = round(psutil.virtual_memory().total / 2**30, 1)
    print(f"System RAM: {RAM_GB} GB" + ("  (small: the notebook loads big models straight onto the GPU to fit)" if RAM_GB < 16 else ""))
except ImportError:
    pass
print("Project folder:", PROJ)

# ---- Colab budget: free T4 sessions have ~12.7 GB RAM, ~15 GB GPU memory and ~70–110 GB disk ----
def free_disk_gb():
    return shutil.disk_usage("/content" if os.path.exists("/content") else "/").free / 2**30

def purge_hf(repo_ids, subfolders):
    """Delete downloaded model files we no longer need from the Hugging Face cache (frees Colab disk).
    Returns the GB freed. Only the named sub-folders of the named repos are touched."""
    try:
        from huggingface_hub import scan_cache_dir
        freed = 0
        for repo in scan_cache_dir().repos:
            if repo.repo_id not in repo_ids:
                continue
            for rev in repo.revisions:
                for f in rev.files:
                    if any(f"/{sf}/" in str(f.file_path) for sf in subfolders):
                        if Path(f.blob_path).exists():
                            freed += Path(f.blob_path).stat().st_size
                            Path(f.blob_path).unlink()
                        Path(f.file_path).unlink(missing_ok=True)
        log(f"purged {freed / 2**30:.1f} GB from the Hugging Face cache: {repo_ids} {subfolders}")
        return freed / 2**30
    except Exception:
        log_exc("could not clean the Hugging Face cache")
        return 0.0

def need_disk(gb, what, purge=()):
    """Stop with a clear message (instead of a half-written download) when Colab's disk is too full."""
    free = free_disk_gb()
    log(f"disk check for {what}: needs {gb} GB, {free:.1f} GB free")
    for repos, subs in purge:
        if free >= gb:
            break
        purge_hf(repos, subs); free = free_disk_gb()
    if free < gb:
        raise RuntimeError(f"Colab disk too full for {what}: needs ~{gb} GB, only {free:.0f} GB free. "
                           "Runtime → Disconnect and delete runtime, then Run all (finished work is safe on Drive).")

try:
    import psutil
    _vm = psutil.virtual_memory()
    _drive_free = shutil.disk_usage(MODEL_CACHE).free / 2**30
    print(f"Colab budget: RAM {_vm.total / 2**30:.1f} GB ({_vm.available / 2**30:.1f} free) | "
          f"disk {free_disk_gb():.0f} GB free | GPU memory {GPU['vram_gb']} GB | "
          f"{'Google Drive' if MODEL_CACHE_ON_DRIVE else 'cache folder'} {_drive_free:.1f} GB free")
    cached = sorted(p.name for p in MODEL_CACHE.iterdir() if p.is_dir()) if MODEL_CACHE.exists() else []
    print(f"  Model cache ({MODEL_CACHE}): {', '.join(cached) if cached else 'empty (first session: models are downloaded once)'}")
    if MODEL_CACHE_ON_DRIVE and _drive_free < 4 and not (MODEL_CACHE / "wan21-1.3b-float16" / "complete").exists():
        print("  ⚠️ Less than 4 GB free on Google Drive: the video model can't be cached, so it is downloaded every session.")
    print("  This notebook keeps RAM under ~6 GB (big models load straight onto the GPU) and needs ~20 GB of disk\n"
          "  in the first session (11 GB text encoder + 6 GB video model; both are deleted from disk once used).")
    if _vm.available / 2**30 < 6:
        print("  ⚠️ Less than 6 GB RAM free: close other notebooks or use Runtime → Restart session.")
    if free_disk_gb() < 25:
        print("  ⚠️ Less than 25 GB disk free: Runtime → Disconnect and delete runtime gives a fresh disk.")
except ImportError:
    pass
try:                                                   # library versions, for the log
    from importlib.metadata import version as _v
    vers = []
    for pkg in ["torch", "diffusers", "transformers", "accelerate", "huggingface-hub", "onnxruntime-gpu",
                "onnxruntime", "kokoro-onnx", "faster-whisper", "ctranslate2", "numpy"]:
        try:
            vers.append(f"{pkg}={_v(pkg)}")
        except Exception:
            pass
    log("Packages: " + ", ".join(vers))
    log("ffmpeg: " + subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0])
    if GPU["name"]:
        log("nvidia-smi:\n" + subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
except Exception:
    log_exc("could not read versions")

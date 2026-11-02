# ==== 5. Helpers ====
def ff(*args, nice=False):
    """Run ffmpeg; every command, its time and any messages go to the log. Raise with the error text if it fails.
    nice=True: low CPU priority, so background edits don't slow down the GPU work."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)]
    if nice and shutil.which("nice"):
        cmd = ["nice", "-n", "10"] + cmd
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(f"ffmpeg ({time.time() - t0:.1f}s, exit {r.returncode}): " + " ".join(cmd)[:4000], "FFMPG")
    if r.stderr.strip():
        log("ffmpeg said: " + r.stderr[-3000:], "ERROR" if r.returncode else "WARN")
    if r.returncode:
        raise RuntimeError("ffmpeg failed:\n" + " ".join(cmd) + "\n" + r.stderr[-3000:])

import unicodedata
LANGUAGE = PLAN.get("language", "en")       # "en" (Kokoro voices) or "te" (Telugu: Indic Parler-TTS, Hind Guntur)
SCRIPT_FONTS = {"te": ("Hind Guntur", "https://github.com/google/fonts/raw/main/ofl/hindguntur/HindGuntur-{w}.ttf",
                       "ITF Hind Guntur (SIL Open Font License): Telugu + Latin, so 'PNR', 'WL 12', '₹' render too")}
SUB_FONT, FONTS_DIR = "DejaVu Sans", None   # subtitle font (libass) and where its file is

def setup_script_font():
    """Non-English video: one font with the script AND Latin letters for every drawn text and the subtitles.
    Downloaded once (OFL) to the model cache. Telugu needs complex shaping (raqm), or vowel signs break."""
    global SUB_FONT, FONTS_DIR
    if LANGUAGE == "en":
        return
    if LANGUAGE not in SCRIPT_FONTS:
        raise ValueError(f"language '{LANGUAGE}' has no font configured (SCRIPT_FONTS in the helpers cell)")
    from PIL import features
    if not features.check("raqm"):
        raise RuntimeError("Pillow can't shape Telugu text here (no raqm/fribidi): letters would be drawn broken. "
                           "Run: !apt-get -qq install libfribidi0 libraqm0, then Runtime → Restart session.")
    name, url, why = SCRIPT_FONTS[LANGUAGE]
    FONTS_DIR = MODEL_CACHE / "fonts"; FONTS_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for w in ("Regular", "Bold"):
        f = FONTS_DIR / Path(url.format(w=w)).name
        if not f.exists() or f.stat().st_size < 50_000:
            urllib.request.urlretrieve(url.format(w=w), f.with_suffix(".part")); f.with_suffix(".part").rename(f)
        files[w] = f
    drawkit.set_script_font(files["Regular"], files["Bold"])
    SUB_FONT = name
    log(f"script font for '{LANGUAGE}': {name} ({why}); raqm shaping available")
    print(f"Language: {LANGUAGE} | text font: {name}")

import urllib.request
setup_script_font()

def sidecar(path, **info):
    Path(str(path) + ".json").write_text(json.dumps(info, indent=2))
    log(f"saved {Path(path).name}: " + json.dumps({k: v for k, v in info.items() if k != "words"})[:1500])

def read_sidecar(path):
    p = Path(str(path) + ".json")
    return json.loads(p.read_text()) if p.exists() else None

def free_gpu():
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    try:                     # hand freed system RAM back to the OS (Python keeps it after big CPU-side loads otherwise)
        import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def ram_free_gb():
    """System RAM still available (GB); None if it can't be read."""
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except Exception:
        return None

def need_ram(gb, what):
    """Stop with a clear, logged error BEFORE a step that needs `gb` GB of system RAM when less is free: when Colab
    runs out of RAM it restarts the whole session and the error is lost."""
    free_gpu()
    free = ram_free_gb()
    log(f"RAM check for {what}: needs ~{gb:.1f} GB, {'?' if free is None else f'{free:.1f}'} GB free")
    if free is not None and free < gb + 1.0:          # 1 GB headroom for Python, FFmpeg and the notebook itself
        raise MemoryError(f"not enough system RAM for {what}: needs ~{gb:.1f} GB + 1 GB headroom, {free:.1f} GB free")

def gpu_mem():
    """'allocated / reserved / total GB' as PyTorch sees it (live tensors / held by the allocator / card)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "no GPU"
        a, r = torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30
        pk = torch.cuda.max_memory_reserved() / 2**30         # highest point since the last model release
        return (f"{a:.1f} allocated / {r:.1f} reserved / {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB"
                f" (peak {pk:.1f} GB)")
    except Exception:
        return "unknown"

def release_model(m, name="model"):
    """Really give a model's GPU memory back. `del` alone isn't enough when a library (accelerate hooks from
    device_map loading, a pipeline, a cache) still references the model: here every weight's storage is replaced by
    an empty CPU tensor, so the GPU memory is freed even if a hidden reference survives. Use only when done with it."""
    if m is None:
        return
    before = gpu_mem()
    try:
        import torch
        mods = [m] if isinstance(m, torch.nn.Module) else [c for c in getattr(m, "components", {}).values()
                                                           if isinstance(c, torch.nn.Module)]
        try:
            from accelerate.hooks import remove_hook_from_module
            for mod in mods:
                remove_hook_from_module(mod, recurse=True)
        except Exception:
            pass
        with torch.no_grad():
            for mod in mods:
                for t in list(mod.parameters()) + list(mod.buffers()):
                    t.data = torch.empty(0, dtype=t.dtype, device="cpu")
    except Exception:
        log_exc(f"release_model({name}) failed; falling back to garbage collection")
    free_gpu()
    log(f"released {name}: GPU {before} → {gpu_mem()}")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()        # the next model's peak is measured on its own
    except Exception:
        pass

class GpuSampler:
    """Samples GPU utilisation and memory every second in the background (nvidia-smi), so 'is the GPU busy?' is
    answered by the whole interval, not by one snapshot. with GpuSampler() as gs: ...; gs.summary() -> dict."""
    def __init__(self, every=1.0):
        self.every, self.util, self.mem, self._stop = every, [], [], None

    def _run(self):
        while not self._stop.wait(self.every):
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5)
                u, m = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
                self.util.append(u); self.mem.append(m / 1024)
            except Exception:
                pass

    def __enter__(self):
        import threading
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True); self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set(); self._th.join(timeout=10)

    def summary(self):
        if not self.util:
            return {"gpu_util_avg": None, "gpu_util_samples": 0}
        busy = sum(u >= 50 for u in self.util) / len(self.util)
        return {"gpu_util_avg": round(sum(self.util) / len(self.util), 1), "gpu_busy_share": round(busy, 2),
                "gpu_mem_max_gb": round(max(self.mem), 2), "gpu_util_samples": len(self.util)}


def cuda_now():
    """Wall-clock time after all queued GPU work has finished (GPU calls return before the work is done)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    return time.time()


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

def num_words(n):
    """12 -> 'twelve', 100 -> 'one hundred' (so '100' and 'a hundred' count as the same words)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    for size, name in [(10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand"), (100, "hundred")]:
        if n >= size:
            return num_words(n // size) + " " + name + ("" if n % size == 0 else " " + num_words(n % size))

def norm_words(s):
    s = s.lower().replace("-", " ").replace("'", "").replace("\u2019", "")    # restaurant's == restaurants
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)                     # 1,000 -> 1000
    s = re.sub(r"\ba (hundred|thousand|million|billion)\b", r"one \1", s)
    s = re.sub(r"([a-z])(\d)", r"\1 \2", s); s = re.sub(r"(\d)([a-z])", r"\1 \2", s)
    s = re.sub(r"\d+", lambda m: num_words(int(m.group())) if len(m.group()) <= 12 else m.group(), s)
    # letters of ANY script are kept, with their vowel signs (Telugu "సీటు" is 4 code points; 2 are marks, which
    # [a-z0-9] — and even \w — would cut out); everything else is a word break
    return "".join(ch if ch == " " or unicodedata.category(ch)[0] in "LMN" else " " for ch in s).split()

def wer(ref, hyp):
    """Word error rate between two word lists."""
    dp = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(hyp) + 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
    return dp[-1] / max(len(ref), 1)

def speech_error(texts, heard):
    """How wrong the heard speech is, 0..1: the best of the word error rate and 2x the letter error rate (letters
    ignore spaces, so 'backend' / 'back end' or 'WebSocket' / 'web socket' don't count as mistakes).
    Other languages (Telugu): the letter error rate only — words are long and inflected, and Whisper's Telugu is far
    weaker than its English, so word-level scoring would flag correct speech; the threshold is SPEECH_RETRY_OTHER."""
    best = 1.0
    if LANGUAGE != "en" and (globals().get("PLAN") or {}).get("code_mix"):
        import telugu_dialogue                   # code-mixed Telugu: English words in either script, vowel noise
        return min(telugu_dialogue.bilingual_error(t, heard) for t in texts)
    hw = norm_words(heard)
    for t in texts:
        rw = norm_words(t)
        cer = wer(list("".join(rw)), list("".join(hw)))
        best = min(best, cer if LANGUAGE != "en" else min(wer(rw, hw), 2 * cer))
    return best

def ram(label=""):
    """Print system RAM and GPU memory use (Colab restarts the session if system RAM runs out)."""
    msg = resources()
    if msg:
        print(f"   {msg} {label}")

NVENC_OK = False
def pick_video_encoder():
    """NVIDIA hardware encoder (NVENC) when available, else the CPU x264 encoder."""
    global NVENC_OK
    try:
        test = WORK / "nvenc_test.mp4"
        ff("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24", "-frames:v", 24,
           "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", test)
        print("Video encoding: GPU (NVENC)")
        NVENC_OK = True
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0",
                "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", FPS]
    except Exception:
        log_exc("NVENC not available")
        print("Video encoding: CPU (x264)")
        return ["-c:v", "libx264", "-preset", X264_PRESET, "-crf", "18", "-pix_fmt", "yuv420p", "-r", FPS]
VENC = pick_video_encoder()
DEVICE_REPORT["shot + final video encoding"] = "GPU (NVENC)" if NVENC_OK else "CPU (x264)"

STATUS = {}          # shot_id -> how its visual was made (for the QC report)

# ---- Hugging Face models kept on Drive as REAL files ----
# The normal Hugging Face cache stores each file as a symlink into blobs/. Google Drive does not keep symlinks: in a
# later session the "file" is the link text ("../../blobs/…"), and e.g. faster-whisper fails with "Unsupported model
# binary version v774843950" (= the bytes "../."). So models cached on Drive are downloaded as plain files into
# MODEL_CACHE/hf_files/<repo> (snapshot_download with local_dir), checked, and re-downloaded if anything is broken.
WHISPER_REPOS = {"tiny": "Systran/faster-whisper-tiny", "tiny.en": "Systran/faster-whisper-tiny.en",
                 "small.en": "Systran/faster-whisper-small.en",
                 "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"}

def _broken_file(f):
    """A link that didn't survive Drive (tiny text file holding a path) or a file that can't be read."""
    try:
        if f.is_symlink():
            return not f.exists()
        if f.stat().st_size < 1024:                     # real small files are JSON/text, never a bare path
            head = f.read_bytes()[:3]
            return head == b"../" or head[:1] == b"/"
        return False
    except OSError:
        return True

def hf_files(repo, what="model", allow=None):
    """Folder with the repo's real files on Drive (downloaded once, at the licence-verified revision when known)."""
    d = MODEL_CACHE / "hf_files" / repo.replace("/", "--")
    files = [f for f in d.rglob("*") if f.is_file() and ".cache" not in f.parts] if d.exists() else []
    if (d / ".complete").exists() and files and not any(_broken_file(f) for f in files):
        return d
    if d.exists():
        log(f"{what}: cached files of {repo} are incomplete or broken links; downloading again", "WARN")
        shutil.rmtree(d, ignore_errors=True)
    from huggingface_hub import snapshot_download
    t0, rev = time.time(), registry.REVISIONS.get(repo)
    try:
        snapshot_download(repo, local_dir=str(d), revision=rev, allow_patterns=allow)
    except Exception:
        log_exc(f"{what}: download of {repo} at revision {rev} failed; trying the current revision")
        snapshot_download(repo, local_dir=str(d), allow_patterns=allow); rev = "main"
    bad = [str(f) for f in d.rglob("*") if f.is_file() and ".cache" not in f.parts and _broken_file(f)]
    if bad:
        raise RuntimeError(f"{what}: downloaded files of {repo} are broken: {bad[:3]}")
    (d / ".complete").write_text(json.dumps({"repo": repo, "revision": rev, "time": time.strftime("%Y-%m-%d %H:%M")}))
    log(f"{what}: {repo} saved as plain files in {d} ({time.time() - t0:.0f}s)")
    return d

# ---- licence gate: models marked "review" in the registry are refused in strict mode (decided BEFORE any download)
LICENSE_DECISIONS = []
def licence_gate():
    global VIDEO_SPEED, IMAGE_MODEL, CONTINUITY_CHECK
    strict = bool(globals().get("LICENSE_STRICT", False))
    uses = [("Kijai/WanVideo_comfy", "turbo LoRA", globals().get("VIDEO_SPEED") == "turbo"),
            ("stabilityai/stable-diffusion-xl-base-1.0", "SDXL keyframes", globals().get("IMAGE_MODEL") == "sdxl"),
            ("openai/clip-vit-base-patch32", "CLIP continuity check", bool(globals().get("CONTINUITY_CHECK")))]
    for mid, what, used in uses:
        ok, why = registry.model_ok(mid, strict=strict)
        LICENSE_DECISIONS.append({"model": mid, "use": what, "would_load": used, "allowed": ok, "why": why})
        if used and not ok:
            if what == "turbo LoRA":
                VIDEO_SPEED = "fast"           # 25 steps + First Block Cache: ~4x slower per clip, licence-clean
            elif what == "SDXL keyframes":
                IMAGE_MODEL = "procedural"      # fallback pictures are drawn instead
            else:
                CONTINUITY_CHECK = False        # scene-state shots keep the OWLv2 + VLM checks (Apache-2.0)
            print(f"🔒 LICENSE_STRICT: {what} refused ({why}) → " + {"turbo LoRA": "fast video mode",
                  "SDXL keyframes": "drawn fallback pictures", "CLIP continuity check": "check off"}[what])
        log(f"licence: {what}: {'allowed' if ok else 'refused'} ({why})")
licence_gate()

def preflight():
    """What this Colab session can do, measured (nothing assumed), saved to render/preflight.json."""
    import platform
    rep = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(), "gpu": GPU.get("name"),
           "vram_gb": GPU.get("vram_gb"), "compute_capability": GPU.get("sm"), "native_bf16": GPU.get("bf16"),
           "precision": ("bf16" if GPU.get("bf16") else "fp16") if GPU.get("name") else "fp32 (CPU)",
           "ram_free_gb": round(ram_free_gb(), 1), "disk_free_gb": round(free_disk_gb(), 1),
           "model_cache": str(MODEL_CACHE), "video_engine": VIDEO_ENGINE, "image_model": IMAGE_MODEL,
           "video_speed": globals().get("VIDEO_SPEED"), "run_mode": globals().get("RUN_MODE"),
           "story_mode": globals().get("STORY_MODE", "narrator"), "license_strict": globals().get("LICENSE_STRICT"),
           "licences": LICENSE_DECISIONS, "language": LANGUAGE, "cuda": False, "warnings": []}
    try:
        import torch
        rep["cuda"] = torch.cuda.is_available()
        rep["torch"] = torch.__version__
    except ImportError:
        rep["warnings"].append("torch is not installed: no GPU work possible")
    if LANGUAGE != "en":                     # the Telugu voice model is gated: say so now, not after an hour
        try:
            from google.colab import userdata
            has = bool(userdata.get("HF_TOKEN"))
        except Exception:
            has = bool(os.environ.get("HF_TOKEN"))
        rep["hf_token_present"] = has        # presence only; the value is never printed or logged
        if not has:
            rep["warnings"].append("LANGUAGE needs the gated Indic Parler-TTS model: add the HF_TOKEN secret "
                                   "(the voice cell stops with instructions otherwise; it never switches language)")
        elif (globals().get("STRICT_OPEN_SOURCE") or (globals().get("PLAN") or {}).get("code_mix")) and \
                os.environ.get("PARLER_SKIP_ACCESS_CHECK") != "1":
            try:                             # is the gate open for this token? (a HEAD request, nothing downloaded)
                import registry
                try:
                    from google.colab import userdata
                    tok_ = userdata.get("HF_TOKEN")
                except Exception:
                    tok_ = os.environ.get("HF_TOKEN")
                acc = registry.gated_access("ai4bharat/indic-parler-tts", tok_)
                tok_ = None
                rep["parler_access"] = {k: acc[k] for k in ("status", "why")}
                if acc["status"] != "PASS":
                    rep["warnings"].append(f"Indic Parler-TTS access: {acc['status']} ({acc['why']}). {acc['help']}")
            except Exception:
                log_exc("Parler access check failed")
    if rep["gpu"] and not rep["native_bf16"]:
        rep["notes"] = "T4-class GPU: fp16 compute (bf16 is emulated, never assumed)"
    if rep["disk_free_gb"] < 20:
        rep["warnings"].append(f"only {rep['disk_free_gb']} GB disk free")
    if not rep["gpu"]:
        rep["warnings"].append("no GPU: AI video shots fall back to drawn/licensed visuals (CPU fallback)")
    (PROJ / "render").mkdir(exist_ok=True)
    (PROJ / "render" / "preflight.json").write_text(json.dumps(rep, indent=1))
    log("preflight: " + json.dumps(rep)[:3000])
    print("Preflight: " + ", ".join(f"{k}={rep[k]}" for k in ("gpu", "vram_gb", "precision", "ram_free_gb",
                                                                "disk_free_gb", "story_mode", "license_strict"))
          + "".join(f"\n  ⚠️ {w}" for w in rep["warnings"]))
    return rep
PREFLIGHT = preflight()

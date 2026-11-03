# ==== 6. Voices: each character has their own voice; each line is acted with a tone ====
# tone = speed + pitch + loudness + pause (+ an optional effect); character = voice + pitch + effect.
# Every line is checked by speech recognition and re-made if words come out wrong.
import numpy as np, soundfile as sf, urllib.request
SR = 24000
KOKORO_DIR = Path(os.environ.get("KOKORO_DIR", MODEL_CACHE / "kokoro")); KOKORO_DIR.mkdir(parents=True, exist_ok=True)
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
_tts, _asr, _asr_failed, ASR_ON_GPU = None, None, False, False

def kokoro_session(gpu):
    import onnxruntime as rt
    if gpu:
        # DEFAULT = a fixed convolution algorithm, no cuDNN search (the search fails on some T4 setups:
        # "Failed to initialize CUDNN Frontend ... HEURISTIC_QUERY_FAILED")
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = rt.InferenceSession(str(KOKORO_DIR / "kokoro-v1.0.onnx"), providers=providers)
    if gpu:
        sess.disable_fallback()                    # fail loudly instead of switching to CPU behind our back
    log(f"Kokoro session providers: {sess.get_providers()}")
    return sess

def use_cpu_voice(reason):
    """Switch the voice model to the CPU (small model: a few minutes for the whole narration)."""
    global _tts
    from kokoro_onnx import Kokoro
    LOG.write(f"Voice model: GPU failed, switching to CPU. Reason:\n{reason}", "ERROR")
    _tts = Kokoro.from_session(kokoro_session(False), str(KOKORO_DIR / "voices-v1.0.bin"))
    _tts.on_gpu = False
    print("Voice model running on: CPU (the GPU attempt failed; full error in the log file)")
    DEVICE_REPORT["voice (Kokoro)"] = "CPU (GPU attempt failed)"

def get_tts():
    global _tts
    if _tts is None and LANGUAGE != "en":
        if RENDER_CACHED_AUDIO_ONLY:
            raise RuntimeError('A narration WAV or matching sidecar is missing/stale. Allocate a GPU and set '
                               'RENDER_CACHED_AUDIO_ONLY=False to synthesize it.')
        BUDGET.check('Telugu speech model')
        if not GPU['name']:
            raise RuntimeError('New Telugu speech needs an allocated GPU; cached rendering remains available.')
        import torch
        if torch.cuda.mem_get_info()[0] < 9 * 2**30:
            raise RuntimeError('Need at least 9 GiB free GPU memory before loading Parler. Restart the runtime.')
        _tts = ParlerVoice()
    if _tts is None:
        for f in ["kokoro-v1.0.onnx", "voices-v1.0.bin"]:          # ~350 MB, Apache-2.0 model
            if not (KOKORO_DIR / f).exists():                       # once; then kept in the model cache
                print("Downloading", f, "...")
                urllib.request.urlretrieve(KOKORO_URL + f, KOKORO_DIR / (f + ".part"))
                (KOKORO_DIR / (f + ".part")).rename(KOKORO_DIR / f)
        from kokoro_onnx import Kokoro
        import onnxruntime as rt
        log(f"onnxruntime {rt.__version__}, available providers: {rt.get_available_providers()}")
        if "CUDAExecutionProvider" in rt.get_available_providers():
            try:
                rt.preload_dlls()                      # use the CUDA libraries that come with PyTorch
            except Exception:
                log_exc("onnxruntime preload_dlls failed")
            try:
                _tts = Kokoro.from_session(kokoro_session(True), str(KOKORO_DIR / "voices-v1.0.bin"))
                _tts.on_gpu = True
                _tts.create("Testing the voice on the graphics card, with a sentence of ordinary length "
                            "so that every layer of the model runs once.", voice=VOICE, speed=1.0, lang="en-us")
                print("Voice model running on: GPU")
                DEVICE_REPORT["voice (Kokoro)"] = "GPU"
            except Exception:
                use_cpu_voice(traceback.format_exc())
        else:
            use_cpu_voice("onnxruntime has no CUDA support in this session")
        assert VOICE in _tts.get_voices(), f"unknown voice {VOICE}; choose from {_tts.get_voices()}"
    return _tts

def speak(text, speed, voice=None, attempt=0, instruct=""):
    """Kokoro TTS (English); Qwen3-TTS for "qwen:<speaker>" voices (expressive narration, `instruct` = the voice
    direction); Indic Parler-TTS for other languages (attempt = which take: a new seed per retry).
    Kokoro: if the GPU fails on some sentence later on, finish on the CPU instead of stopping."""
    if voice and voice.startswith("gemini:"):          # Telugu via Gemini free tier (made by the voice pass)
        return GEMINI_SPEAK(text, voice.split(":", 1)[1], attempt)
    if (voice or "").startswith("qwen:"):
        return get_qwen_tts().create(text, voice, speed, attempt, instruct)
    if LANGUAGE != "en":
        return get_tts().create(text, voice=voice or VOICE, speed=speed, attempt=attempt)
    try:
        return get_tts().create(text, voice=voice or VOICE, speed=speed, lang="en-us")
    except Exception:
        if not getattr(_tts, "on_gpu", False):
            raise
        use_cpu_voice(traceback.format_exc())
        return _tts.create(text, voice=voice or VOICE, speed=speed, lang="en-us")

def preload_cuda_libs():
    """faster-whisper's GPU backend (CTranslate2 4.x) needs libcublas.so.12 + cuDNN 9. On Colab they live inside the
    pip packages nvidia-cublas-cu12 / nvidia-cudnn-cu12 (installed with torch), outside the loader's search path, so
    the GPU attempt fails ("Library libcublas.so.12 is not found or cannot be loaded") and the check silently ran on
    the CPU. Loading them by full path first (RTLD_GLOBAL) makes them findable. Harmless if absent."""
    import ctypes, glob, importlib.util
    done = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(pkg)
        except Exception:
            spec = None
        for root in (list(spec.submodule_search_locations or []) if spec else []):
            for lib in sorted(glob.glob(os.path.join(root, "lib", "lib*.so*"))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL); done.append(os.path.basename(lib))
                except OSError:
                    pass
    log(f"CUDA libraries preloaded for faster-whisper: {done or 'none found'}")
    return done

def get_asr():
    global _asr, _asr_failed, ASR_ON_GPU
    if _asr_failed:
        return None
    if _asr is None:
        BUDGET.check('speech check')
        try:
            import time
            import torch
            import ctranslate2
            from faster_whisper import WhisperModel
            requested = globals().get('ASR_DEVICE', 'auto')
            if requested not in ('auto', 'cuda', 'cpu'):
                raise ValueError('ASR_DEVICE must be auto, cuda or cpu')
            start = time.monotonic()
            src = str(hf_files('Systran/faster-whisper-small', 'small multilingual speech check'))
            reason = 'CPU explicitly selected'
            use_cuda = False
            if requested != 'cpu':
                if torch.cuda.is_available() and ctranslate2.get_cuda_device_count() > 0:
                    free_gb = torch.cuda.mem_get_info()[0] / 2**30
                    use_cuda = free_gb >= globals().get('ASR_MIN_FREE_GB', 2.0)
                    reason = f'only {free_gb:.2f} GiB GPU memory free; reserving room for TTS'
                else:
                    reason = 'CUDA unavailable to PyTorch or CTranslate2'
                if requested == 'cuda' and not use_cuda:
                    raise RuntimeError('GPU speech check requested but ' + reason)
            if use_cuda:
                try:
                    preload_cuda_libs()
                    supported = ctranslate2.get_supported_compute_types('cuda')
                    compute = 'int8_float16' if 'int8_float16' in supported else 'float16'
                    _asr = WhisperModel(src, device='cuda', compute_type=compute, num_workers=1)
                    ASR_ON_GPU = True
                    DEVICE_REPORT['speech check (Whisper)'] = f'GPU (small multilingual, {compute})'
                except Exception as error:
                    if requested == 'cuda':
                        raise
                    reason = f'GPU initialization failed: {type(error).__name__}: {error}'
                    log(reason)
            if _asr is None:
                log('Speech check falling back to CPU: ' + reason)
                _asr = WhisperModel(src, device='cpu', compute_type='int8', cpu_threads=CPU_THREADS)
                ASR_ON_GPU = False
                DEVICE_REPORT['speech check (Whisper)'] = 'CPU (small multilingual, int8): ' + reason
            log(f"Speech check ready: {DEVICE_REPORT['speech check (Whisper)']}; load/download {time.monotonic()-start:.1f}s")
        except Exception:
            log_exc('speech check unavailable; report remains unverified')
            _asr_failed = True
            ASR_ON_GPU = False
            if globals().get('ASR_DEVICE', 'auto') == 'cuda':
                raise RuntimeError('GPU speech check required but unavailable; see the original error above.') from sys.exc_info()[1]
    return _asr

def listen(path):
    """Transcribe a file -> (text, [[word, start, end], ...]) or (None, None)."""
    global _asr_failed
    import time
    started = time.monotonic()
    asr = get_asr()
    if asr is None:
        return None, None
    try:
        # decode with ffmpeg to 16 kHz mono (avoids a faster-whisper / PyAV version clash)
        p16 = Path(str(path) + ".16k.wav")
        ff("-i", path, "-ar", 16000, "-ac", 1, p16)
        audio, _ = sf.read(str(p16), dtype="float32")
        # names the recogniser can't know (Zomato, Ananya...) are given up front, so it doesn't hear "the motto"
        # beam 5 everywhere: a greedy search was measured only ~6 % faster on CPU (encoder + word timing dominate)
        segs, _ = asr.transcribe(audio, language=LANGUAGE, word_timestamps=True, beam_size=5,
                                 initial_prompt=globals().get("ASR_PROMPT") or None)
        words = [[w.word.strip(), round(w.start, 3), round(w.end, 3)] for seg in segs for w in (seg.words or [])]
        elapsed = time.monotonic() - started
        duration = len(audio) / 16000
        log(f"Speech check {Path(path).name}: {'GPU' if ASR_ON_GPU else 'CPU'}, "
            f"{duration:.1f}s audio checked in {elapsed:.1f}s (includes load if first call)")
        p16.unlink(missing_ok=True)
        return " ".join(w[0] for w in words), words
    except Exception as e:
        log_exc("speech check failed")
        _asr_failed = True
        if globals().get('ASR_DEVICE', 'auto') == 'cuda':
            raise RuntimeError('GPU speech transcription failed; check is not verified.') from e
        print("Speech check failed, continuing without it:", repr(e)[:200])
        return None, None

# ---- other languages (Telugu): Indic Parler-TTS (Apache-2.0) in its own Python environment ----
if LANGUAGE != "en":
    if ASR_MODEL.endswith(".en"):                     # English-only recogniser can't check Telugu
        ASR_MODEL = ASR_MODEL_MULTI
    WER_RETRY = SPEECH_RETRY_OTHER                    # letter error rate; Whisper is much weaker on Telugu than English
    log(f"language {LANGUAGE}: voices = Indic Parler-TTS, speech check = {ASR_MODEL} (letter error, retry above "
        f"{WER_RETRY:.0%})")

def hf_token():
    """Hugging Face token from a Colab secret (🔑 in the left bar) or the HF_TOKEN environment variable. Never in code."""
    tok, why = os.environ.get("HF_TOKEN"), "no HF_TOKEN environment variable"
    if not tok:
        try:
            from google.colab import userdata
            tok = userdata.get("HF_TOKEN")
        except Exception as e:
            log(f"no HF_TOKEN Colab secret: {e!r}", "WARN")
            why = {"SecretNotFoundError": "there is no secret named exactly HF_TOKEN",
                   "NotebookAccessError": "the HF_TOKEN secret exists but 'Notebook access' is OFF for this notebook "
                                          "(or the access pop-up was dismissed)"}.get(type(e).__name__, repr(e)[:200])
    if not tok:
        raise RuntimeError(
            f"Colab gave no Hugging Face token ({why}).\n"
            "The Telugu voice model (ai4bharat/indic-parler-tts) needs a free Hugging Face token:\n"
            "  1. Log in at huggingface.co, open huggingface.co/ai4bharat/indic-parler-tts and accept the terms "
            "(approved instantly).\n  2. huggingface.co/settings/tokens → New token → type 'Read' → copy it.\n"
            "  3. In Colab: 🔑 Secrets (left bar) → Add new secret → name HF_TOKEN, paste the token, turn ON "
            "'Notebook access'.\n  4. Run this cell again (the voice environment is already set up: no reinstall).")
    return tok.strip()                               # pasted with a space or line break = an invalid token (401)

def check_hf_access(tok, repo):
    """Before starting the voice model: is the token valid, and does its account have access to the gated model?
    Hugging Face answers 401 for an invalid token and 403 for terms not accepted; each gets its own fix here.
    The token itself is never printed or logged."""
    if os.environ.get("PARLER_SKIP_ACCESS_CHECK") == "1":       # tests with a stand-in worker
        return
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        name = api.whoami(token=tok).get("name", "?")
    except Exception as e:
        log(f"Hugging Face token check failed: {type(e).__name__}: {str(e)[:300]}", "ERROR")
        raise RuntimeError(
            f"The HF_TOKEN secret is not a valid Hugging Face token (it starts with {tok[:3]!r}, {len(tok)} characters; "
            "a real one starts with 'hf_' and is about 37 characters).\n"
            "  → huggingface.co/settings/tokens → Create new token → type 'Read' → copy the token VALUE (hf_…), not its name.\n"
            "  → Colab 🔑 Secrets → HF_TOKEN → replace the value (no spaces) → Notebook access ON → re-run this cell.")
    try:
        api.auth_check(repo, token=tok)
    except Exception as e:
        log(f"Hugging Face access check for {repo} failed: {type(e).__name__}: {str(e)[:300]}", "ERROR")
        raise RuntimeError(
            f"The token works (Hugging Face account '{name}'), but that account can't download {repo}.\n"
            f"  → While logged in as '{name}', open huggingface.co/{repo} and accept the terms (approved instantly).\n"
            "  → If the token is 'Fine-grained', tick 'Read access to contents of all public gated repos you can access' "
            "(or make a 'Read' token instead).\n  → Re-run this cell.")
    log(f"Hugging Face: account '{name}' has access to {repo}")

def install_tool_env(env, requirements, module, expect_tf, label):
    """One-time per session (~2–4 min): a virtual environment that sees Colab's torch/numpy but has its own
    transformers (the version its tool pins); exact versions from the worker's REQUIREMENTS, installed with
    --no-deps (nothing else is resolved, downloaded or upgraded). Used by the Telugu and the expressive voices."""
    t0, py = time.time(), Path(env) / "bin" / "python"
    print(f"Setting up the {label} environment (one time per session, ~3 min)...")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "uv"], check=True)
    subprocess.run([sys.executable, "-m", "uv", "venv", "--system-site-packages", "--python", sys.executable,
                    str(env)], check=True, capture_output=True)
    req = Path(env) / "requirements.txt"; req.write_text(requirements)
    r = subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", str(py), "--no-deps",
                        "-r", str(req)], capture_output=True, text=True)
    log(f"{label} env install ({time.time() - t0:.0f}s, exit {r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    if r.returncode:
        raise RuntimeError(f"{label} environment could not be installed (details in the log): "
                           + r.stderr.strip().splitlines()[-1][:300])
    chk = subprocess.run([str(py), "-I", "-c", f"import transformers, torch, {module}; "
                          "print(transformers.__version__, torch.__version__)"], capture_output=True, text=True)
    log(f"{label} env check: {chk.stdout.strip()} {chk.stderr[-1500:]}")
    if chk.returncode or not chk.stdout.startswith(expect_tf):
        raise RuntimeError(f"{label} environment is broken (details in the log): " + chk.stderr[-300:])
    (Path(env) / "ready").write_text(chk.stdout)
    print(f"  ready in {time.time() - t0:.0f}s ({chk.stdout.strip()})")

class ParlerVoice:
    """Indic Parler-TTS behind the same interface as Kokoro (create, get_voices). parler-tts pins transformers 4.46,
    the notebook needs 5.x, so it runs in its own environment (Colab's torch, its own transformers) as ONE worker
    process: the model loads once, each sentence is a JSON request. Each character = a trained speaker name + a
    description; each retry = a new seed (Parler samples, so a retry is a genuinely new take)."""
    ENV = Path(os.environ.get("PARLER_ENV", str(WORK.parent / "tts_parler")))

    def __init__(self):
        import parler_worker as pw
        self.pw, self.proc = pw, None
        self.py = self.ENV / "bin" / "python"
        t0 = time.time()
        if not (self.ENV / "ready").exists():
            self.install()
        tok = hf_token()
        check_hf_access(tok, os.environ.get("PARLER_MODEL", pw.MODEL))
        env = dict(os.environ, HF_TOKEN=tok, HF_HOME=str(WORK.parent / "hf_parler"), PYTHONWARNINGS="ignore",
                   TOKENIZERS_PARALLELISM="false")
        dev = "cuda" if GPU["name"] else "cpu"
        dt = "bfloat16" if GPU.get("bf16") else "float32"   # T4: fp16 overflows in the T5 description encoder
        worker = os.environ.get("PARLER_WORKER") or pw.__file__
        self.err_path = Path(LOG.path).with_name("parler_worker.log")         # next to the run log (moved to Drive with it)
        self.err_start = self.err_path.stat().st_size if self.err_path.exists() else 0
        self.err = open(self.err_path, "a")
        self.proc = subprocess.Popen([str(self.py), "-I", worker, os.environ.get("PARLER_MODEL", pw.MODEL), dev, dt],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True,
                                     env=env, bufsize=1)
        try:
            ready = self._read(timeout=BUDGET.timeout(1200))  # first session downloads model weights
        except BaseException:
            self.close()
            raise
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(f"Telugu voice model did not start: {ready} (details: parler_worker.log next to the run log)")
        self.sr = ready["sr"]
        log(f"Parler worker ready in {time.time() - t0:.0f}s: {ready}")
        DEVICE_REPORT["voice (Indic Parler, own process)"] = f"{'GPU' if ready['device'] == 'cuda' else 'CPU'} ({ready['dtype']})"
        print(f"Voice model: Indic Parler-TTS ({ready['device']}, {ready['dtype']}, transformers "
              f"{ready.get('transformers')} in its own environment)")

    def install(self):
        install_tool_env(self.ENV, self.pw.REQUIREMENTS, "parler_tts", "4.46", "Telugu voice")

    def _read(self, timeout=600):
        import select
        r, _, _ = select.select([self.proc.stdout], [], [], BUDGET.timeout(timeout))
        if not r:
            self.close()
            raise RuntimeError(f"Telugu voice model gave no answer in {timeout}s (parler_worker.log next to the run log)")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(self.why_stopped())
            try:
                return json.loads(line)
            except ValueError:
                log(f"parler worker (not protocol): {line.strip()[:300]}", "WARN")

    def why_stopped(self):
        """The worker exited: show its real error on screen (last lines of its log) and what to do about it."""
        try:
            self.proc.wait(timeout=10)
        except Exception:
            pass
        self.err.flush()
        try:
            with open(self.err_path, "rb") as f:
                f.seek(self.err_start); tail = f.read().decode("utf-8", "replace")
        except Exception as e:
            tail = f"(could not read {self.err_path}: {e!r})"
        lines = [l for l in tail.splitlines() if l.strip() and "Warning" not in l and not l.startswith("  warnings.warn")]
        last = "\n".join(lines[-12:])
        low = tail.lower()
        if "gated" in low or " 401" in low or " 403" in low or "unauthorized" in low or "access to model" in low:
            fix = ("Hugging Face refused the download. Open huggingface.co/ai4bharat/indic-parler-tts while logged in and "
                   "ACCEPT the terms; use a token of type 'Read' (a 'Fine-grained' token needs 'Read access to contents "
                   "of all public gated repos you can access' ticked). Then re-run this cell.")
        elif "out of memory" in low:
            fix = "The GPU or RAM was full when the voice model loaded: Runtime → Restart session, then Run all."
        elif "modulenotfounderror" in low or "importerror" in low or "undefined symbol" in low:
            (self.ENV / "ready").unlink(missing_ok=True)
            fix = "The voice environment is broken: it is reinstalled when you re-run this cell. Send this error if it repeats."
        else:
            fix = "Send this error message (and parler_worker.log next to the run log) to Claude."
        log(f"parler worker stopped (exit {self.proc.returncode}):\n{tail[-6000:]}", "ERROR")
        return (f"Telugu voice model stopped (exit code {self.proc.returncode}).\n{fix}\n\nLast lines of its log "
                f"({self.err_path}):\n{last}")

    def get_voices(self):
        return self.pw.SPEAKERS.get(LANGUAGE, [])

    def _one(self, text, desc, seed):
        out = WORK / f"parler_{hashlib.sha1(f'{text}|{desc}|{seed}'.encode()).hexdigest()[:12]}.wav"
        self.proc.stdin.write(json.dumps({"text": text, "description": desc, "seed": seed, "out": str(out)}) + "\n")
        self.proc.stdin.flush()
        r = self._read()
        if not r.get("ok"):
            raise RuntimeError(f"Telugu voice failed: {r.get('error')}")
        return out, r

    def create(self, text, voice, speed=1.0, attempt=0):
        """Sentence by sentence (Parler is most reliable on short inputs), each checked for a plausible length (catches
        cut-off or rambling takes: those get a new seed, up to 3 tries), joined with short pauses; then the line's
        speed (tempo, pitch kept) and Kokoro's sample rate, so everything downstream is unchanged."""
        name, _, style = voice.partition("|")
        desc = (f"{name} speaks {style or 'in a clear, natural, moderately paced voice'}. The recording is of very "
                "high quality, with the speaker's voice sounding clear and very close up, with no background noise.")
        parts = [x.strip() for x in re.split(r"(?<=[.?!।])\s+", text) if x.strip()]
        pieces = []
        for k, part in enumerate(parts):
            letters = sum(1 for ch in part if not ch.isspace())
            for t in range(3):
                seed = int(hashlib.sha1(f"{part}|{name}".encode()).hexdigest()[:6], 16) + 1000 * attempt + 97 * t
                try:
                    out, r = self._one(part, desc, seed)
                except RuntimeError as e:                 # a sampled take can fail inside the model (seen on Colab:
                    if t == 2:                            # "IndexError: tuple index out of range"): new seed
                        raise
                    log(f"Parler take for {part[:40]!r} failed with seed {seed}: {e}; new seed", "WARN")
                    continue
                per = r["seconds"] / max(letters, 1)
                if PARLER_SEC_PER_CHAR[0] <= per <= PARLER_SEC_PER_CHAR[1]:
                    break
                log(f"Parler take for {part[:40]!r}: {r['seconds']}s for {letters} letters ({per:.3f} s/letter) — "
                    "implausible, new seed", "WARN")
            pieces.append(out)
        lst = WORK / f"parler_concat_{os.getpid()}.txt"
        gap = WORK / "parler_gap.wav"
        if not gap.exists():
            ff("-f", "lavfi", "-i", f"anullsrc=r={self.sr}:cl=mono", "-t", 0.18, gap)
        lst.write_text("".join(f"file '{p}'\n" + (f"file '{gap}'\n" if i < len(pieces) - 1 else "")
                               for i, p in enumerate(pieces)))
        joined = WORK / f"parler_line_{os.getpid()}.wav"
        ff("-f", "concat", "-safe", 0, "-i", lst, "-af", f"atempo={min(max(speed, 0.5), 2.0):.4f},aresample={SR}",
           "-ac", 1, "-ar", SR, joined)
        wav, sr = sf.read(str(joined), dtype="float32")
        return wav, sr

    def close(self):
        """Stop the worker: it is a separate process, so its GPU memory is only freed when it exits."""
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close(); self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self.proc:
            for stream in (self.proc.stdin, self.proc.stdout):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass
        if getattr(self, 'err', None) is not None:
            self.err.close()
        self.proc = None

class QwenVoice(ParlerVoice):
    """Expressive English narration: Qwen3-TTS 1.7B CustomVoice (Apache-2.0) in its own environment (qwen-tts pins
    transformers 4.57). ONE fixed premium speaker = the narrator's identity; the voice direction becomes the
    `instruct` (style only). Precision QWEN_TTS_DTYPE (T4: float32 unless VOICE_BENCH shows float16 is stable).
    A whole line (1–3 sentences) is one request, so its prosody stays natural; retries = new seeds."""
    ENV = Path(os.environ.get("QWEN_TTS_ENV", str(WORK.parent / "tts_qwen")))

    def __init__(self):
        import qwen_tts_worker as qw
        self.pw, self.proc = qw, None
        self.py = self.ENV / "bin" / "python"
        t0 = time.time()
        if not (self.ENV / "ready").exists():
            install_tool_env(self.ENV, qw.REQUIREMENTS, "qwen_tts", "4.57", "expressive voice")
        dev = "cuda" if GPU["name"] else "cpu"
        dt = QWEN_TTS_DTYPE if dev == "cuda" else "bfloat16"
        env = dict(os.environ, HF_HOME=str(WORK.parent / "hf_qwen_tts"), PYTHONWARNINGS="ignore", TOKENIZERS_PARALLELISM="false")
        worker = os.environ.get("QWEN_TTS_WORKER") or qw.__file__
        self.err_path = Path(LOG.path).with_name("qwen_tts_worker.log")
        self.err_start = self.err_path.stat().st_size if self.err_path.exists() else 0
        self.err = open(self.err_path, "a")
        self.model = os.environ.get("QWEN_TTS_MODEL", qw.MODEL)
        self.proc = subprocess.Popen([str(self.py), "-I", worker, self.model, dev, dt], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.err, text=True, env=env, bufsize=1)
        ready = self._read(timeout=2400)
        if not ready.get("ready"):
            raise RuntimeError(f"expressive voice model did not start: {ready}")
        self.speakers, self.instruct_ok, self.info = ready.get("speakers", []), ready.get("instruct", False), ready
        log(f"Qwen3-TTS worker ready in {time.time() - t0:.0f}s: {ready}")
        DEVICE_REPORT["voice (Qwen3-TTS, own process)"] = f"{'GPU' if ready['device'] == 'cuda' else 'CPU'} ({ready['dtype']})"
        print(f"Voice model: Qwen3-TTS ({self.model.split('/')[-1]}, {ready['device']}, {ready['dtype']}, "
              f"{'instruction control' if self.instruct_ok else 'NO instruction control: directions not applied'})")

    def get_voices(self):
        return [f"qwen:{s}" for s in self.speakers]

    def create(self, text, voice, speed=1.0, attempt=0, instruct=""):
        spk = voice.split(":", 1)[1]
        if spk not in self.speakers:
            raise ValueError(f"Qwen3-TTS has no speaker '{spk}' (available: {self.speakers})")
        seed = int(hashlib.sha1(f"{text}|{spk}".encode()).hexdigest()[:6], 16) + 1000 * attempt
        out = WORK / f"qwen_{hashlib.sha1(f'{text}|{spk}|{instruct}|{seed}'.encode()).hexdigest()[:12]}.wav"
        self.proc.stdin.write(json.dumps({"text": text, "speaker": spk, "language": "English",
                                          "instruct": instruct if self.instruct_ok else "", "seed": seed,
                                          "out": str(out)}) + "\n")
        self.proc.stdin.flush()
        r = self._read(timeout=900)
        if not r.get("ok"):
            raise RuntimeError(f"expressive voice failed: {r.get('error')}")
        LAST_TTS_INFO.update(r)
        wav, sr = sf.read(str(out), dtype="float32")
        if sr != SR:                                   # Qwen3-TTS is 24 kHz like Kokoro; resample only if not
            res = WORK / f"{out.stem}_{SR}.wav"
            ff("-i", out, "-ar", SR, "-ac", 1, res); wav, sr = sf.read(str(res), dtype="float32")
        return wav, sr

_qwen_tts = None
LAST_TTS_INFO = {}
def get_qwen_tts():
    global _qwen_tts
    if _qwen_tts is None:
        _qwen_tts = QwenVoice()
    return _qwen_tts

def close_tts():
    global _qwen_tts, _tts
    for t_ in (_tts, _qwen_tts):
        if t_ is not None and hasattr(t_, "close"):
            t_.close()
    _qwen_tts = _tts = None

BUDGET.on_exhausted = close_tts

VOICE_FX = {"none":    "",
            "sparkle": "chorus=0.6:0.9:50|60:0.4|0.32:0.25|0.4:2|1.3,aecho=0.8:0.7:40:0.25",   # magical shimmer
            "old":     "lowpass=f=7000,bass=g=2:f=180"}                                         # warmer, older
TONE_FX = {"shaky":   "tremolo=f=5.5:d=0.15",           # trembling (sad / scared)
           "breathy": "highpass=f=200,treble=g=2"}      # airy, whisper-like

# ---- expressive narration: a delivery direction per line, from the story (voice_direction.py) ----
# Words never change. Kokoro can only follow the pace (+ real pauses); Qwen3-TTS follows the whole direction as an
# instruction. What each backend did with each field is reported (prosody_report.json), never assumed.
DIRECTIONS, VQC = {}, {}
if EXPRESSIVE:
    _order = [l["id"] for s_ in SHOTS for l in s_["lines"]]
    _dialogue = PLAN.get("story_mode") == "two_friends_cinematic"
    _profile = "two_friends" if _dialogue else VOICE_PROFILE
    def _direct():
        # two friends: intention from the conversation + response gaps; narrators: beat roles + narration pauses
        d = voice_direction.dialogue_directions(SHOTS, CHARS) if _dialogue else \
            voice_direction.rule_directions(SHOTS, VOICE_PROFILE)
        if VOICE_DIRECTOR == "llm":                     # a local model reads the whole story (GPU, freed after)
            import planner
            llm = planner.LocalLLM(DIRECTOR_MODEL, device="cuda" if GPU["name"] else "cpu",
                                   cache_dir=str(WORK.parent / "hf_planner"), log=log)
            try:
                d = voice_direction.llm_directions(llm, SHOTS, d, planner.ask_json, log)
            finally:
                del llm; free_gpu()
        d, adj = voice_direction.smooth(_order, d, _profile,
                                        EXPRESSIVE_MAX_INTENSITY if VOICE_BACKEND != "kokoro" else 1.0)
        return {"directions": {k: v.model_dump() for k, v in d.items()}, "adjustments": adj}
    _vd = stages.stage(PROJ / "stages", "voice_direction", {
        "lines": [(l["id"], l.get("tts") or l["text"], l.get("tone"), s_.get("beat"), s_.get("chapter"),
                   l.get("voice_direction"), l.get("speaker"), l.get("kind"), l.get("intent"), s_.get("story_beat"))
                  for s_ in SHOTS for l in s_["lines"]],
        "profile": _profile, "director": VOICE_DIRECTOR, "backend": VOICE_BACKEND,
        "cap": EXPRESSIVE_MAX_INTENSITY, "v": 3}, _direct, log)
    DIRECTIONS = {k: voice_direction.VoiceDirection(**v) for k, v in _vd["directions"].items()}
    (PROJ / "render").mkdir(exist_ok=True)
    (PROJ / "render" / "voice_direction.json").write_text(json.dumps(_vd, indent=1))
    print(f"Voice direction ({_profile}, {VOICE_DIRECTOR}): " + " → ".join(
        f"{DIRECTIONS[i].intent or DIRECTIONS[i].emotion}{'*' if DIRECTIONS[i].intensity >= 0.5 else ''}"
        for i in _order if i in DIRECTIONS))
    for a_ in _vd["adjustments"]:
        log("voice direction smoothed: " + a_)

# ---- VOICE_BENCH: the same script, neutral vs expressive, per backend (measured, blind A/B), then stop ----
BENCH_SCRIPT = [   # hook → setup → build-up → suspense question → reveal → calm conclusion (a demo, not published)
    ("hook", "What happens when a giant star finally runs out of fuel?"),
    ("setup", "For millions of years, the star burns hydrogen, pushing outward against its own gravity."),
    ("step", "But when the fuel runs out, nothing holds gravity back anymore."),
    ("turn", "So what happens to all that matter, falling inward at once?"),
    ("resolution", "It collapses into a point so dense that not even light can escape."),
    ("payoff", "Every black hole began as a star, just like the ones you can see tonight.")]
VOICE_BENCH_BACKENDS = ["kokoro", "qwen3-tts"]
# two-friend dialogue benchmark (story plans): one short scene covering the intentions the brief lists. Telugu uses
# lines with the same INTENTIONS, written as natural spoken Telugu (not a word-for-word translation).
DIALOGUE_BENCH = {
    "en": [("meera", "explaining_observation", "fact", "When you look at something close, each of your eyes sees it from a slightly different angle."),
           ("arjun", "genuine_curiosity", "question", "Wait, so my left eye and my right eye see different pictures right now?"),
           ("meera", "reflective_explanation", "fact", "Pretty much... and your brain quietly turns them into one."),
           ("arjun", "playful_disbelief", "observation", "No way. That's why my thumb jumped when I closed one eye!"),
           ("meera", "gentle_correction", "correction", "Almost. Your thumb doesn't move. You just switched which eye was looking."),
           ("arjun", "playful_remark", "reaction", "So I've been watching two movies at once my whole life."),
           ("meera", "satisfying_revelation", "fact", "And that's the whole trick of 3D glasses: each eye gets only its own picture."),
           ("meera", "reflective_explanation", "fact", "The projector puts both pictures on one screen, each lens lets only one through, "
                                                        "and your brain builds the depth from the difference.")],
    "te": [("meera", "explaining_observation", "fact", "ఏదైనా దగ్గరగా ఉన్నదాన్ని చూసినప్పుడు, నీ రెండు కళ్లూ దాన్ని కొంచెం వేర్వేరు కోణాల నుంచి చూస్తాయి."),
           ("arjun", "genuine_curiosity", "question", "ఆగు, అంటే ఇప్పుడు నా ఎడమ కన్నూ కుడి కన్నూ వేర్వేరు చిత్రాలు చూస్తున్నాయా?"),
           ("meera", "reflective_explanation", "fact", "అవును... కానీ నీ మెదడు వాటిని నిశ్శబ్దంగా ఒకటిగా కలిపేస్తుంది."),
           ("arjun", "playful_disbelief", "observation", "అయ్యో! అందుకేనా ఒక కన్ను మూస్తే నా బొటనవేలు దూకింది!"),
           ("meera", "gentle_correction", "correction", "దాదాపు. బొటనవేలు కదలదు, చూసే కన్ను మారుతుంది అంతే."),
           ("arjun", "playful_remark", "reaction", "అంటే నేను జీవితాంతం ఒకేసారి రెండు సినిమాలు చూస్తున్నానన్నమాట."),
           ("meera", "satisfying_revelation", "fact", "త్రీడీ గ్లాసెస్ రహస్యం అదే: ప్రతి కంటికి దాని చిత్రం మాత్రమే చేరుతుంది."),
           ("meera", "reflective_explanation", "fact", "ప్రొజెక్టర్ రెండు చిత్రాలనూ ఒకే స్క్రీన్ మీద వేస్తుంది, ప్రతి లెన్స్ ఒకదాన్నే పంపుతుంది, "
                                                        "అప్పుడు నీ మెదడు ఆ తేడా నుంచి లోతును తానే తయారు చేస్తుంది.")]}
QWEN_BENCH_SPEAKERS = {"arjun": "aiden", "meera": "serena"}   # Meera: no native-English female preset exists in
                                                             # Qwen3-TTS CustomVoice; "serena" is a CANDIDATE to judge


def run_dialogue_bench():
    """VOICE_BENCH for two-friend plans: old (narrator-style) vs new (conversation) direction on Kokoro; Qwen3-TTS
    with conversational instructions; Parler for Telugu. One TTS family at a time. Measured: load/generation time,
    real-time factor, peak VRAM, peak RAM, audio length, speech check, identity per character, QC. Every backend
    gets a status: PASS / FAIL / NOT_TESTED / RESOURCE_UNAVAILABLE / LICENSE_BLOCKED / ACCESS_REQUIRED."""
    import random, resource
    out_dir = PROJ / "render" / "voice_bench"; out_dir.mkdir(parents=True, exist_ok=True)
    lang = LANGUAGE if LANGUAGE in DIALOGUE_BENCH else "en"
    script = DIALOGUE_BENCH[lang]
    shot = {"shot_id": "db", "story_beat": "first_answer",
            "lines": [{"id": f"db_{i + 1}", "speaker": sp, "kind": k, "intent": it, "text": t, "tone": "friendly"}
                      for i, (sp, it, k, t) in enumerate(script)]}
    order = [l["id"] for l in shot["lines"]]
    new, _ = voice_direction.smooth(order, voice_direction.dialogue_directions([shot], CHARS), "two_friends")
    old = voice_direction.rule_directions([dict(shot, beat="step", chapter="c")], VOICE_PROFILE)
    old, _ = voice_direction.smooth(order, old, VOICE_PROFILE)
    configs = ([("kokoro", "old_direction", old), ("kokoro", "new_direction", new), ("qwen3-tts", "new_direction", new)]
               if lang == "en" else [("parler", "new_direction", new)])
    try:
        spk = voice_qc.SpeakerCheck(WORK.parent / "ecapa")
    except Exception:
        log_exc("speaker check unavailable in the voice benchmark"); spk = None
    report, samples = {}, []
    for be, variant, dirs in configs:
        key = f"{be}_{variant}"
        mid = {"kokoro": "hexgrad/Kokoro-82M", "qwen3-tts": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
               "parler": "ai4bharat/indic-parler-tts"}[be]
        ok_lic, why = registry.model_ok(mid, strict=LICENSE_STRICT)
        rec = {"backend": be, "model": mid, "revision": registry.REVISIONS.get(mid), "licence": why,
               "device": "cuda" if GPU["name"] else "cpu"}
        if not ok_lic:
            report[key] = dict(rec, status="LICENSE_BLOCKED"); continue
        if be == "qwen3-tts" and not GPU["name"]:
            report[key] = dict(rec, status="RESOURCE_UNAVAILABLE",
                               why="no GPU: measured ~4–12 min per line on CPU; run the benchmark on a T4"); continue
        if be == "parler":
            try:
                hf_token()
            except Exception as e:
                report[key] = dict(rec, status="ACCESS_REQUIRED", why=str(e)[:300]); continue
        t0, chunks, rows, embs = time.time(), [], {}, {}
        try:
            for l in shot["lines"]:
                d, c = dirs[l["id"]], CHARS[l["speaker"]]
                if be == "qwen3-tts":
                    voice = f"qwen:{c.get('qwen_speaker') or QWEN_BENCH_SPEAKERS[l['speaker']]}"
                    instr, speed = expressive_tts.qwen_params(d, voice)["instruct"], 1.0
                elif be == "parler":
                    voice = "|".join([c["voice"], ", ".join(x for x in (c.get("voice_style"), expressive_tts.parler_style(d)) if x)])
                    instr, speed = "", 1.0
                else:
                    voice, instr = c["voice"], ""
                    speed = expressive_tts.kokoro_params(d, c.get("speed", 1.0))["speed"]
                t1 = time.time()
                wav, sr = speak(l["text"], speed, voice, attempt=0, instruct=instr)
                gen = time.time() - t1
                tmp = out_dir / f"{key}_{l['id']}.wav"; sf.write(tmp, np.asarray(wav, dtype=np.float32), SR)
                heard, _ = listen(tmp)
                err = None if heard is None else speech_error([l["text"]], heard)
                vq = check_line(tmp, l["text"], heard, err)
                vq.update(speaker=l["speaker"], intent=d.intent, gen_s=round(gen, 2),
                          rtf=round(gen / max(len(wav) / SR, 0.1), 2), instruct=instr,
                          peak_vram_gb=LAST_TTS_INFO.get("peak_vram_gb") if be == "qwen3-tts" else None)
                rows[l["id"]] = vq
                if spk:
                    embs.setdefault(l["speaker"], {})[l["id"]] = spk.embed(wav, SR)
                chunks.append((wav, d.pause_before_ms / 1000, d.pause_after_ms / 1000))
            for sp_, e_ in embs.items():
                for lid, sim in voice_qc.consistency(e_).items():
                    rows[lid]["identity"] = sim
            f = assemble(chunks, out_dir / f"dialogue_{key}.wav")
            samples.append((key, f))
            fails = {lid: [c_ for c_, x in r["checks"].items() if x["status"] == "FAIL" and "(diagnostic)" not in c_]
                     for lid, r in rows.items()}
            report[key] = dict(rec, status="FAIL" if any(fails.values()) else "PASS",
                               total_s=round(time.time() - t0, 1), audio_s=round(sum(len(c[0]) for c in chunks) / SR, 1),
                               peak_ram_gb=round(max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                                     resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 2**20, 2),
                               lines=rows, fails=fails, human_naturalness=None)
            print(f"  {key}: {report[key]['status']} {report[key]['total_s']:.0f}s for {report[key]['audio_s']}s of speech")
        except Exception as e:
            log_exc(f"voice bench {key} failed"); report[key] = dict(rec, status="FAIL", error=repr(e)[:300])
            print(f"  {key}: FAILED {repr(e)[:200]}")
        if be != "kokoro":
            close_tts()                                    # one TTS model family at a time
    for k_, r_ in report.items():
        if r_["status"] not in ("PASS", "FAIL"):
            print(f"  {k_}: {r_['status']} — {r_.get('why', r_.get('licence', ''))[:160]}")
    (out_dir / "dialogue_bench.json").write_text(json.dumps(report, indent=1, default=str))
    rnd, key_ = random.Random(), {}
    rnd.shuffle(samples)
    sheet = ["# Blind listening sheet (dialogue benchmark)", "", "Rate each sample 1–5, THEN open blind_test_key.json.", "",
             "| Sample | Human-like | Pronunciation | Conversational flow | Emotion fits | Voices consistent | Overall |",
             "|---|---|---|---|---|---|---|"]
    try:
        from IPython.display import Audio, display
        for lab, (k_, f_) in zip("ABCDEFGH", samples):
            key_[lab] = k_; sheet.append(f"| {lab} | | | | | | |")
            print(f"Sample {lab}:"); display(Audio(str(f_)))
    except Exception:
        for lab, (k_, f_) in zip("ABCDEFGH", samples):
            key_[lab] = k_; sheet.append(f"| {lab} | | | | | | |")
        log_exc("audio players unavailable (not in a notebook)")
    (out_dir / "blind_test_key.json").write_text(json.dumps({k: str(v) for k, v in key_.items()}, indent=1))
    (out_dir / "listening_sheet.md").write_text("\n".join(sheet) + "\n")
    raise VoiceBenchDone(f"Dialogue voice benchmark finished: {out_dir}. Listen and fill listening_sheet.md before "
                         "opening blind_test_key.json; then set VOICE_BENCH = False.")


class VoiceBenchDone(Exception):
    """Raised on purpose after the voice benchmark so 'Run all' stops before making the video."""


def assemble(chunks, out):
    """Chunks [(wav, pause_before_s, pause_after_s)] → one track: real silences, 8 ms edge fades, no per-chunk
    loudness normalisation (the intended dynamics stay)."""
    parts = []
    for w, pb, pa in chunks:
        w = np.asarray(w, dtype=np.float32).copy(); n_ = min(int(SR * 0.008), len(w) // 2)
        if n_:
            r_ = np.linspace(0, 1, n_, dtype=np.float32); w[:n_] *= r_; w[-n_:] *= r_[::-1]
        parts += [np.zeros(int(SR * pb), np.float32), w, np.zeros(int(SR * pa), np.float32)]
    sf.write(out, np.concatenate(parts), SR)
    return out


def run_voice_bench():
    import random
    out_dir = PROJ / "render" / "voice_bench"; out_dir.mkdir(parents=True, exist_ok=True)
    shots = [{"shot_id": f"b{i + 1}", "beat": beat, "chapter": beat, "lines": [{"id": f"b{i + 1}_1", "speaker": "narrator",
              "text": t}]} for i, (beat, t) in enumerate(BENCH_SCRIPT)]
    order = [s_["lines"][0]["id"] for s_ in shots]
    expressive, adj = voice_direction.smooth(order, voice_direction.rule_directions(shots, VOICE_PROFILE), VOICE_PROFILE)
    neutral = voice_direction.neutral_directions(order, expressive)
    text = {s_["lines"][0]["id"]: s_["lines"][0]["text"] for s_ in shots}
    (out_dir / "voice_direction.json").write_text(json.dumps({k: v.model_dump() for k, v in expressive.items()}, indent=1))
    try:
        spk = voice_qc.SpeakerCheck(WORK.parent / "ecapa")
    except Exception:
        log_exc("speaker check unavailable in the voice benchmark"); spk = None
    report, prosody, samples = {}, {}, []
    for be in VOICE_BENCH_BACKENDS:
        voice = f"qwen:{NARRATOR_QWEN_SPEAKER}" if be == "qwen3-tts" else CHARS["narrator"]["voice"]
        for variant, dirs in (("neutral", neutral), ("expressive", expressive)):
            key = f"{be}_{variant}"
            t0, chunks, rows, embs = time.time(), [], {}, {}
            try:
                if be == "qwen3-tts":
                    get_qwen_tts()
                load_s = time.time() - t0
                for lid in order:
                    d = dirs[lid]
                    instr = expressive_tts.qwen_params(d, voice)["instruct"] if be == "qwen3-tts" else ""
                    speed = expressive_tts.kokoro_params(d, 1.0)["speed"] if be == "kokoro" else 1.0
                    t1 = time.time()
                    wav, sr = speak(text[lid], speed, voice, attempt=0, instruct=instr)
                    gen = time.time() - t1
                    tmp = out_dir / f"{key}_{lid}.wav"; sf.write(tmp, np.asarray(wav, dtype=np.float32), SR)
                    heard, _ = listen(tmp)
                    err = None if heard is None else speech_error([text[lid]], heard)
                    vq = check_line(tmp, text[lid], heard, err)
                    vq.update(gen_s=round(gen, 2), rtf=round(gen / max(len(wav) / SR, 0.1), 2),
                              peak_vram_gb=LAST_TTS_INFO.get("peak_vram_gb") if be == "qwen3-tts" else None)
                    rows[lid] = vq
                    if spk:
                        embs[lid] = spk.embed(wav, SR)
                    chunks.append((wav, d.pause_before_ms / 1000, d.pause_after_ms / 1000))
                    prosody[f"{key}/{lid}"] = {"direction": d.model_dump(), "applied": expressive_tts.report(d, be),
                                               "instruct": instr, "speed": speed}
                if spk:
                    for lid, sim in voice_qc.consistency(embs).items():
                        rows[lid]["checks"]["same narrator voice"] = {"status": "UNKNOWN" if sim is None else
                            "PASS" if sim >= voice_qc.SPEAKER_TH.get(be, 0.8) else "FAIL", "value": sim}
                f = assemble(chunks, out_dir / f"narration_{variant}_{be}.wav")
                samples.append((key, f))
                report[key] = {"load_s": round(load_s, 1), "total_s": round(time.time() - t0, 1),
                               "audio_s": round(sum(len(c[0]) for c in chunks) / SR, 1), "lines": rows,
                               "fails": {lid: [c for c, x in r["checks"].items() if x["status"] == "FAIL"] for lid, r in rows.items()}}
                print(f"  {key}: {report[key]['total_s']:.0f}s for {report[key]['audio_s']}s of speech | fails: "
                      + (", ".join(f"{k}:{v}" for k, v in report[key]["fails"].items() if v) or "none"))
            except Exception as e:
                log_exc(f"voice bench {key} failed"); report[key] = {"error": repr(e)[:300]}
                print(f"  {key}: FAILED {repr(e)[:200]}")
        if be == "qwen3-tts":
            close_tts()                                        # one TTS model family at a time
    (out_dir / "prosody_report.json").write_text(json.dumps(prosody, indent=1, default=str))
    (out_dir / "voice_qc_report.json").write_text(json.dumps(report, indent=1, default=str))
    # blind A/B: each backend's neutral and expressive take in random order; the key is saved, not shown
    rnd = random.Random(); key_ = {}
    try:
        from IPython.display import Audio, display
        for be in VOICE_BENCH_BACKENDS:
            pair = [x for x in samples if x[0].startswith(be)]
            rnd.shuffle(pair)
            for lab, (k_, f_) in zip("AB", pair):
                key_[f"{be} {lab}"] = k_
                print(f"{be} — sample {lab}:"); display(Audio(str(f_)))
    except Exception:
        log_exc("audio preview unavailable")
    (out_dir / "blind_test_key.json").write_text(json.dumps(key_, indent=1))
    raise VoiceBenchDone(f"Voice benchmark finished: {out_dir}. Listen to A/B before opening blind_test_key.json; "
                         "then set VOICE_BENCH = False and VOICE_BACKEND to the winner.")


def backend_of(voice):
    return ("qwen3-tts" if voice.startswith("qwen:") else "gemini" if voice.startswith("gemini:")
            else "parler" if LANGUAGE != "en" else "kokoro")
RETAKE_BACKENDS = ("qwen3-tts", "parler")   # local sampling backends: a new seed is a genuinely new take. Gemini lines
                                            # come only from the planned two-speaker pass (budget, conversational flow)

def line_settings(l):
    """Everything that decides how a line sounds (also the cache key: change any of it and the line is remade)."""
    c, t = CHARS[l["speaker"]], TONES[l["tone"]]
    d = DIRECTIONS.get(l["id"])
    # with a direction, emotion is never faked by DSP: no tone pitch shift / gain / tremolo (character pitch stays:
    # that is the character's identity, not an emotion)
    pitch = round(c.get("pitch", 0) + (0 if d else t.get("pitch", 0)), 2)
    gain = round(c.get("gain_db", 0) + (0 if d else t.get("gain_db", 0)), 2)
    chain = []
    if abs(pitch) > 0.01:                             # pitch shift that keeps the length (and word timings)
        r = 2 ** (pitch / 12)
        chain += [f"asetrate={round(SR * r)}", f"aresample={SR}", f"atempo={1 / r:.5f}"]
    for fx in [VOICE_FX[c.get("fx", "none")], "" if d else TONE_FX.get(t.get("fx"), "")]:
        if fx:
            chain.append(fx)
    if gain:
        chain.append(f"volume={gain}dB")
    chain.append("alimiter=limit=0.95")
    if GEMINI_ACTIVE and c.get("gemini_voice"):      # Telugu via Gemini free tier (its own pronunciation rules)
        return {"text": gem_text(l["text"]), "voice": f"gemini:{c['gemini_voice']}",
                "speed": 1.0, "fx": "alimiter=limit=0.95", **({"direction": d.model_dump(), "instruct": ""} if d else {})}
    if LANGUAGE != "en":
        voice = "|".join([c["voice"], ", ".join(x for x in (c.get("voice_style"),
                                                           expressive_tts.parler_style(d) if d else t.get("describe")) if x)])
    elif EXPRESSIVE and VOICE_BACKEND == "qwen3-tts" and l["speaker"] == "narrator":
        voice = f"qwen:{NARRATOR_QWEN_SPEAKER}"         # one fixed premium speaker = the narrator's identity
    elif EXPRESSIVE and VOICE_BACKEND == "qwen3-tts" and c.get("qwen_speaker"):
        voice = f"qwen:{c['qwen_speaker']}"             # a friend with an approved Qwen speaker in the Bible
    else:
        voice = c["voice"]
    want = {"text": l.get("tts") or l["text"], "voice": voice,
            "speed": round(c.get("speed", 1.0) * (d.pace_multiplier if d else t.get("speed", 1.0)) * SPEED_SCALE, 3),
            "fx": ",".join(chain)}
    if d:
        want["direction"] = d.model_dump()
        want["instruct"] = expressive_tts.qwen_params(d, voice)["instruct"] if voice.startswith("qwen:") else ""
    return want

def check_line(path, text, heard, err):
    """Voice QC of one finished line (voice_qc.py): signal checks + missing/repeated words."""
    wav, sr = sf.read(str(path), dtype="float32")
    m = voice_qc.measure(wav, sr, len(text.split()), text=text)
    diff = voice_qc.word_diff(norm_words(text), norm_words(heard)) if heard else None
    return voice_qc.verdict(m, err, WER_RETRY, None, diff) | {"metrics": m}

# ---- Telugu via Google Gemini FREE TIER (optional): preflight → two-speaker chunks → split per turn → checks.
# Only the approved dialogue text (+ short style words) is sent; story, facts and checks stay local. If ANY condition
# or line fails, the whole video uses the local Indic Parler voices (one voice per character per video).
import telugu_dialogue, gemini_tts, importlib
GEMINI_ACTIVE, GEMINI_REPORT, GEMINI_CLIENT, GEMINI_KEY = False, None, None, ""
def gemini_wanted():
    if LANGUAGE != "te":
        return False
    if globals().get("STRICT_OPEN_SOURCE"):          # strict open source: Gemini only on an explicit opt-in
        return TELUGU_TTS_BACKEND == "gemini_free"
    return TELUGU_TTS_BACKEND == "gemini_free" or (TELUGU_TTS_BACKEND == "auto" and PLAN.get("tts_backend") == "gemini_free")
def gem_text(text):
    """What Gemini reads (subtitles keep `text`): the form chosen by the listening benchmark."""
    form = "gemini_free" if globals().get("GEMINI_TRANSCRIPT_FORM", "display") == "display" else "gemini_free:telugu_script"
    return telugu_dialogue.tts_text(text, form)
VOICE_POLICY = PLAN.get("voice_policy", {})         # per-story voice rules (absent: the earlier behaviour)
CODE_MIX = PLAN.get("code_mix")
def gemini_key():
    """GEMINI_API_KEY from Colab Secrets (or the environment). Never printed, logged or saved."""
    try:
        from google.colab import userdata
        k = userdata.get("GEMINI_API_KEY")
    except Exception:
        k = None
    return (k or os.environ.get("GEMINI_API_KEY", "")).strip()
GEMINI_CFG = dict(enabled=gemini_wanted(), model=GEMINI_TTS_MODEL, free_tier_confirmed=GEMINI_FREE_TIER_CONFIRMED,
                  free_tier_only=GEMINI_FREE_TIER_ONLY, max_videos_per_day=MAX_VIDEOS_PER_DAY,
                  max_calls_per_video=MAX_GEMINI_TTS_CALLS_PER_VIDEO, max_retries=MAX_GEMINI_RETRIES_PER_SEGMENT,
                  mode=GEMINI_MODE, reserve_calls=2, asr_repairs=VOICE_POLICY.get("gemini_asr_repairs", True))
GEMINI_LEDGER = gemini_tts.Ledger(Path(DRIVE_ROOT) / "_gemini_usage.json")
GEMINI_CACHE = PROJ / "gemini_audio"

def gemini_client():
    """The Gemini REST client (no package to install; tests inject a stand-in)."""
    global GEMINI_CLIENT, GEMINI_KEY
    if GEMINI_CLIENT is None:
        GEMINI_KEY = gemini_key()
        if not GEMINI_KEY:
            return None
        try:                                           # REST with Colab's own `requests`: nothing is installed
            GEMINI_CLIENT = (globals().get("GEMINI_CLIENT_FACTORY") or gemini_tts.make_client)(GEMINI_KEY)
        except Exception as e:
            log("Gemini client could not start: " + gemini_tts.redact(e, GEMINI_KEY)[:300], "ERROR")
    return GEMINI_CLIENT

def GEMINI_SPEAK(text, voice, attempt=0):
    """One line on demand (only if a Gemini line must be remade): cached, budgeted, never a paid route."""
    who = next(k for k, c in CHARS.items() if c.get("gemini_voice") == voice)
    p_, info = gemini_tts.synthesize(gemini_client(), GEMINI_CFG, [{"speaker": who, "text": text}], {who: voice},
                                     GEMINI_CACHE, GEMINI_LEDGER, PROJECT, GEMINI_KEY, log, label=f"single {attempt}")
    tmp_ = WORK / f"gemini_single_{os.getpid()}.wav"; ff("-i", p_, "-ar", SR, "-ac", 1, tmp_)
    return sf.read(str(tmp_), dtype="float32")

def run_gemini_voices():
    global GEMINI_ACTIVE, GEMINI_REPORT
    ok_scope, why_scope = registry.hosted_ok(GEMINI_TTS_MODEL, "tts")
    lines = [{"id": l["id"], "scene": s_["shot_id"], "speaker": l["speaker"], "intent": l.get("intent") or
              (DIRECTIONS[l["id"]].intent if l["id"] in DIRECTIONS else None),
              "text": gem_text(l["text"])} for s_ in SHOTS for l in s_["lines"]]
    voices = {k: c["gemini_voice"] for k, c in CHARS.items() if c.get("gemini_voice")}
    done = all((read_sidecar(PROJ / "narration" / f"{l['id']}.wav") or {}).get("backend") == "gemini"
               and (read_sidecar(PROJ / "narration" / f"{l['id']}.wav") or {}).get("text") == l["text"]
               and (PROJ / "narration" / f"{l['id']}.wav").exists() for l in lines)
    if done and (PROJ / "render" / "gemini_report.json").exists():
        GEMINI_REPORT = json.loads((PROJ / "render" / "gemini_report.json").read_text()); GEMINI_ACTIVE = True
        print("Telugu voices: Gemini lines already made (cache), no API call"); return
    pf = gemini_tts.preflight(GEMINI_CFG, gemini_key(), GEMINI_LEDGER, PROJECT,
                              gemini_client() if gemini_key() else None)
    if not ok_scope:
        pf["checks"]["hosted exception scope"] = ("POLICY_BLOCKED", why_scope); pf["status"] = "POLICY_BLOCKED"
    print("Gemini free-tier preflight: " + pf["status"])
    for k_, (st_, why_) in pf["checks"].items():
        print(f"  {'✅' if st_ == 'PASS' else '❌'} {k_}: {st_}" + (f" — {why_}" if why_ and st_ != "PASS" else ""))
    GEMINI_REPORT = {"status": "FALLBACK", "preflight": pf, "why": f"preflight {pf['status']}"}
    if pf["status"] == "PASS":
        asr = globals().get("GEMINI_ASR_FN") or listen
        spk = globals().get("GEMINI_SPEAKER_FN")          # (tests inject a stand-in; normally ECAPA)
        if spk is None:
            try:
                _sc = voice_qc.SpeakerCheck(WORK.parent / "ecapa"); spk = _sc.embed
            except Exception:
                log_exc("speaker check unavailable for the Gemini attribution check")
        t0 = time.time()
        rep = gemini_tts.run_voices(lines, voices, GEMINI_CFG, gemini_client(), GEMINI_CACHE, GEMINI_LEDGER, PROJECT,
                                    WORK / "gemini_lines", asr_fn=asr, key=GEMINI_KEY, log=log,
                                    retry_at=SPEECH_RETRY_OTHER, speaker_fn=spk,
                                    err_fn=telugu_dialogue.bilingual_error if CODE_MIX else None)
        rep.update(preflight=pf, seconds=round(time.time() - t0, 1))
        GEMINI_REPORT = rep
    if GEMINI_REPORT["status"] == "PASS":
        GEMINI_ACTIVE = True
        DEVICE_REPORT["voice (Gemini free tier)"] = "Google's servers (no local GPU or RAM)"
        for l in lines:
            if l["id"] in GEMINI_REPORT["gaps"] and l["id"] in DIRECTIONS:      # keep the model's own response timing
                DIRECTIONS[l["id"]].pause_before_ms = min(GEMINI_REPORT["gaps"][l["id"]], 1200)
        by_line = {l["id"]: l for s_ in SHOTS for l in s_["lines"]}
        for lid, m in GEMINI_REPORT["lines"].items():
            out = PROJ / "narration" / f"{lid}.wav"
            ff("-i", WORK / "gemini_lines" / f"{lid}.wav", "-af", "alimiter=limit=0.95", "-ar", SR, "-ac", 1, out)
            want = line_settings(by_line[lid])
            sidecar(out, **want, backend="gemini", model=GEMINI_TTS_MODEL, speaker=by_line[lid]["speaker"],
                    tone=by_line[lid]["tone"], words=m.get("words") or [], heard=m.get("heard"), wer=m.get("err"),
                    flagged=bool(m.get("review")), how=m.get("how"),
                    takes={"takes": [{"take": 1}], "selected_take_id": f"{lid}_t1"})
        print(f"Telugu voices: Gemini {GEMINI_TTS_MODEL} — {GEMINI_REPORT['calls']} API calls for this video, "
              f"{len(GEMINI_REPORT['chunks'])} conversation chunks ({GEMINI_REPORT['seconds']}s)")
    else:
        print(f"Telugu voices: local Indic Parler-TTS (Gemini not used: {GEMINI_REPORT['why'][:300]})")
    for ch_ in GEMINI_REPORT.get("chunks", []):              # what happened to each conversation chunk
        print(f"  chunk {ch_['lines'][0]}..{ch_['lines'][-1]}: split {ch_.get('align', '-')}"
              + (f" ({ch_['why']})" if ch_.get("why") else "")
              + (f", cut without pause at {ch_['cuts_without_pause']}" if ch_.get("cuts_without_pause") else "")
              + (f", repaired {ch_['repaired']}" if ch_.get("repaired") else "")
              + (f", flagged for listening {ch_['flagged_for_review']}" if ch_.get("flagged_for_review") else "")
              + (f", made per line" if ch_.get("mode") == "per-turn" else ""))
    (PROJ / "render").mkdir(exist_ok=True)
    (PROJ / "render" / "gemini_report.json").write_text(json.dumps(GEMINI_REPORT, indent=1, default=str, ensure_ascii=False))
    (PROJ / "render" / "gemini_usage_today.json").write_text(json.dumps(GEMINI_LEDGER.today(), indent=1))

# ---- TELUGU_VOICE_BENCH / TELUGU_VOICE_AUDITION (one-off, then stop): same Telugu dialogue, several voices ----
AUDITION_PAIRS = [("Puck", "Kore"), ("Achird", "Sulafat"), ("Fenrir", "Aoede"), ("Zubenelgenubi", "Vindemiatrix")]
def run_telugu_bench():
    import random
    out_dir = PROJ / "render" / "telugu_voice_bench"; out_dir.mkdir(parents=True, exist_ok=True)
    first = [s_ for s_ in SHOTS][:3]                       # opening, question, demonstration (+ explanation)
    lines = [{"id": l["id"], "scene": s_["shot_id"], "speaker": l["speaker"], "text": l["text"], "tone": l["tone"],
              "intent": DIRECTIONS[l["id"]].intent if l["id"] in DIRECTIONS else None} for s_ in first for l in s_["lines"]]
    rows, samples = {}, []
    project = f"bench:{PROJECT}"
    cfg = dict(GEMINI_CFG, enabled=True, max_calls_per_video=MAX_GEMINI_BENCH_CALLS)
    pf = gemini_tts.preflight(cfg, gemini_key(), GEMINI_LEDGER, project, gemini_client() if gemini_key() else None)
    def assemble_lines(paths, gaps, out):
        parts = []
        for f_, g_ in zip(paths, gaps):
            w_, r_ = sf.read(str(f_), dtype="float32")
            if r_ != SR:
                t_ = WORK / "bench_rs.wav"; ff("-i", f_, "-ar", SR, "-ac", 1, t_); w_, _ = sf.read(str(t_), dtype="float32")
            parts += [np.zeros(int(SR * g_), np.float32), w_]
        sf.write(out, np.concatenate(parts), SR); return out
    if TELUGU_VOICE_BENCH:
        # A. local Indic Parler (existing voices, per line, the director's gaps)
        try:
            t0, paths, errs = time.time(), [], {}
            for l in lines:
                c = CHARS[l["speaker"]]
                wav, sr = speak(telugu_dialogue.tts_text(l["text"], "indic_parler"), 1.0,
                                "|".join([c["voice"], c.get("voice_style") or ""]), attempt=0)
                f_ = out_dir / f"parler_{l['id']}.wav"; sf.write(f_, np.asarray(wav, np.float32), SR); paths.append(f_)
                h_, _ = listen(f_); errs[l["id"]] = None if h_ is None else gemini_tts.letter_error(l["text"], h_)
            gaps = [0] + [DIRECTIONS[l["id"]].pause_before_ms / 1000 if l["id"] in DIRECTIONS else 0.3 for l in lines[1:]]
            samples.append(("indic_parler", assemble_lines(paths, gaps, out_dir / "A_parler.wav")))
            rows["indic_parler"] = {"status": "PASS", "seconds": round(time.time() - t0, 1), "api_calls": 0, "letter_error": errs}
        except Exception as e:
            log_exc("Telugu bench: Parler failed")
            rows["indic_parler"] = {"status": "ACCESS_REQUIRED" if "HF_TOKEN" in str(e) or "gated" in str(e) else "FAIL",
                                    "why": str(e)[:200]}
        close_tts()
        # B/C/D. Gemini (two-speaker chunk, per turn, optional Flash-Lite)
        variants = [("gemini_flash_scene", GEMINI_TTS_MODEL, "scene"), ("gemini_flash_turn", GEMINI_TTS_MODEL, "turn")]
        if TELUGU_BENCH_LITE:
            variants.append(("gemini_lite_scene", "gemini-3.8-flash-lite-tts", "scene"))
        voices = {k: c["gemini_voice"] for k, c in CHARS.items() if c.get("gemini_voice")}
        for name, model, mode in variants:
            if pf["status"] != "PASS":
                rows[name] = {"status": pf["status"], "why": {k: v for k, v in pf["checks"].items() if v[0] != "PASS"}}
                continue
            t0 = time.time()
            cfg_v = dict(cfg, model=model, mode=mode, reserve_calls=0)
            calls0 = GEMINI_LEDGER.calls(project)
            rep = gemini_tts.run_voices([dict(l, text=gem_text(l["text"])) for l in lines],
                                        voices, cfg_v, gemini_client(), GEMINI_CACHE, GEMINI_LEDGER, project,
                                        out_dir / name, asr_fn=listen, key=GEMINI_KEY, log=log, retry_at=SPEECH_RETRY_OTHER)
            st_ = "PASS" if rep["status"] == "PASS" else ("FREE_QUOTA_EXHAUSTED" if "QUOTA" in rep["why"] else
                                                          "ACCESS_REQUIRED" if "ACCESS" in rep["why"] else "FAIL")
            rows[name] = {"status": st_, "why": rep["why"], "seconds": round(time.time() - t0, 1),
                          "api_calls": GEMINI_LEDGER.calls(project) - calls0, "chunks": rep["chunks"],
                          "letter_error": {k: v.get("err") for k, v in rep["lines"].items()}}
            if rep["status"] == "PASS":
                gaps = [0] + [rep["gaps"].get(l["id"], 300) / 1000 for l in lines[1:]]
                samples.append((name, assemble_lines([out_dir / name / f"{l['id']}.wav" for l in lines], gaps,
                                                     out_dir / f"{name}.wav")))
    if TELUGU_VOICE_AUDITION and pf["status"] == "PASS":
        ex = lines[:4]
        for a_, m_ in AUDITION_PAIRS:
            try:
                p_, info = gemini_tts.synthesize(gemini_client(), dict(cfg, mode="scene"),
                                                 [{"speaker": l["speaker"], "text": gem_text(l["text"])}
                                                  for l in ex], {"arjun": a_, "meera": m_}, GEMINI_CACHE, GEMINI_LEDGER,
                                                 project, GEMINI_KEY, log, label=f"audition {a_}/{m_}")
                samples.append((f"audition Arjun={a_} Meera={m_}", p_))
                rows[f"audition {a_}/{m_}"] = {"status": "PASS", "duration_s": info["duration_s"]}
            except gemini_tts.GeminiBlocked as e:
                rows[f"audition {a_}/{m_}"] = {"status": e.status, "why": e.why}; break
    elif TELUGU_VOICE_AUDITION:
        rows["audition"] = {"status": pf["status"], "why": {k: v for k, v in pf["checks"].items() if v[0] != "PASS"}}
    (out_dir / "bench_report.json").write_text(json.dumps({"preflight": pf, "results": rows,
        "dialogue": [(l["id"], l["speaker"], l["text"]) for l in lines]}, indent=1, ensure_ascii=False, default=str))
    bench = [x for x in samples if not x[0].startswith("audition")]
    random.Random().shuffle(bench)
    key_ = {}
    sheet = ["# Telugu listening sheet (native speakers, 1–5)", "", "Dialogue:", ""] + \
            [f"- {CHARS[l['speaker']]['name']}: {l['text']}" for l in lines] + ["",
            "| Sample | Pronunciation | Natural delivery | Emotion fits | Voice consistency | Characters believable | "
            "Timing | Overall | Sounds like two real people? | Notes |", "|---|---|---|---|---|---|---|---|---|---|"]
    try:
        from IPython.display import Audio, display
        print("Telugu dialogue:\n" + "\n".join(f"  {CHARS[l['speaker']]['name']}: {l['text']}" for l in lines))
    except Exception:
        Audio = display = None
    for lab, (name, f_) in zip("ABCDEFGH", bench):
        key_[lab] = name; sheet.append(f"| {lab} | | | | | | | | | |")
        if display:
            print(f"Sample {lab}:"); display(Audio(str(f_)))
    for name, f_ in samples:
        if name.startswith("audition"):
            sheet.append(f"| {name} | | | | | | | | | |")
            if display:
                print(name + ":"); display(Audio(str(f_)))
    (out_dir / "blind_key.json").write_text(json.dumps(key_, indent=1))
    (out_dir / "listening_sheet.md").write_text("\n".join(sheet) + "\n")
    for k_, r_ in rows.items():
        print(f"  {k_}: {r_['status']}" + (f" — {str(r_.get('why'))[:150]}" if r_["status"] != "PASS" else
                                           f" ({r_.get('api_calls', 0)} API calls)"))
    raise VoiceBenchDone(f"Telugu voice benchmark/audition finished: {out_dir}. Native speakers fill "
                         "listening_sheet.md before opening blind_key.json; then set the flags back to False.")

# ---- TELUGU_PRON_BENCH (one-off, then stop): the same short code-mixed exchange in each pronunciation form ----
PRON_QUESTIONS = ["Is this how friends actually talk?", "Are the English words natural?", "Are both voices understandable?",
                  "Do the reactions sound genuine?", "Are any pronunciations strange? (which words)"]
def run_pron_bench():
    import random
    out_dir = PROJ / "render" / "pron_bench"; out_dir.mkdir(parents=True, exist_ok=True)
    allL = [dict(l, scene=s_["shot_id"]) for s_ in SHOTS for l in s_["lines"]]
    lines = allL[:4]                                     # the opening exchange: popcorn, test, 3D effect
    rows, samples = {}, []
    forms = [("parler_respelled", "indic_parler:code_mix"), ("parler_latin", None)]
    try:
        acc = registry.gated_access("ai4bharat/indic-parler-tts", hf_token())
    except Exception as e:
        acc = {"status": "TOKEN_MISSING", "why": str(e)[:200], "help": registry.ACCESS_HELP}
    for name, form in forms:
        if acc["status"] != "PASS":
            rows[name] = {"status": acc["status"], "why": acc["why"]}; continue
        try:
            paths = []
            for l in lines:
                c = CHARS[l["speaker"]]
                txt = telugu_dialogue.tts_text(l["text"], form) if form else l["text"]
                wav, sr = speak(txt, 1.0, "|".join([c["voice"], c.get("voice_style") or ""]), attempt=0)
                f_ = out_dir / f"{name}_{l['id']}.wav"; sf.write(f_, np.asarray(wav, np.float32), SR); paths.append(f_)
            gap = np.zeros(int(SR * 0.35), np.float32)
            mix = np.concatenate([x for f_ in paths for x in (sf.read(str(f_), dtype="float32")[0], gap)])
            sf.write(out_dir / f"{name}.wav", mix, SR); samples.append((name, out_dir / f"{name}.wav"))
            rows[name] = {"status": "PASS", "tts_text": [telugu_dialogue.tts_text(l["text"], form) if form else l["text"] for l in lines]}
        except Exception as e:
            log_exc(f"pronunciation bench {name} failed"); rows[name] = {"status": "FAIL", "why": repr(e)[:200]}
    close_tts()
    if gemini_wanted():                                  # opt-in only: 2 scene requests under the benchmark budget
        cfg = dict(GEMINI_CFG, enabled=True, max_calls_per_video=MAX_GEMINI_BENCH_CALLS, mode="scene")
        project = f"bench:{PROJECT}"
        pf = gemini_tts.preflight(cfg, gemini_key(), GEMINI_LEDGER, project, gemini_client() if gemini_key() else None)
        voices = {k: c["gemini_voice"] for k, c in CHARS.items() if c.get("gemini_voice")}
        for name, form in (("gemini_display", "gemini_free"), ("gemini_telugu_script", "gemini_free:telugu_script")):
            if pf["status"] != "PASS":
                rows[name] = {"status": pf["status"]}; continue
            try:
                p_, info = gemini_tts.synthesize(gemini_client(), cfg, [{"speaker": l["speaker"],
                                                 "text": telugu_dialogue.tts_text(l["text"], form)} for l in lines],
                                                 voices, GEMINI_CACHE, GEMINI_LEDGER, project, GEMINI_KEY, log, label=f"pron {name}")
                samples.append((name, p_)); rows[name] = {"status": "PASS", "cache_hit": info.get("cache_hit")}
            except gemini_tts.GeminiBlocked as e:
                rows[name] = {"status": e.status, "why": e.why}; break
    random.Random().shuffle(samples)
    key_ = {lab: n for lab, (n, _) in zip("ABCDEFGH", samples)}
    sheet = ["# Telugu–English pronunciation listening sheet (native speakers)", "",
             "Same four lines in each sample. Answer 1–5 (5 = yes, completely) and name strange words.", ""] + \
            [f"- {CHARS[l['speaker']]['name']}: {l['text']}" for l in lines] + ["",
             "| Sample | " + " | ".join(PRON_QUESTIONS) + " | Notes |", "|---" * (len(PRON_QUESTIONS) + 2) + "|"] + \
            [f"| {lab} |" + " |" * (len(PRON_QUESTIONS) + 1) for lab in key_]
    (out_dir / "listening_sheet.md").write_text("\n".join(sheet) + "\n")
    (out_dir / "blind_key.json").write_text(json.dumps(key_, indent=1))
    (out_dir / "pron_report.json").write_text(json.dumps({"parler_access": acc, "results": rows}, indent=1,
                                                          ensure_ascii=False, default=str))
    try:
        from IPython.display import Audio, display
        for lab, (n, f_) in zip("ABCDEFGH", samples):
            print(f"Sample {lab}:"); display(Audio(str(f_)))
    except Exception:
        pass
    for k_, r_ in rows.items():
        print(f"  {k_}: {r_['status']}" + (f" — {r_.get('why', '')}" if r_["status"] != "PASS" else ""))
    if acc["status"] != "PASS":
        print(acc["help"])
    raise VoiceBenchDone(f"pronunciation benchmark finished: {out_dir}. Fill listening_sheet.md, then open blind_key.json; "
                         "set the winning form (GEMINI_TRANSCRIPT_FORM / the Parler table) and TELUGU_PRON_BENCH = False.")

def parler_gate():
    """Before any local Telugu synthesis (and before the benchmarks): is the gated Indic Parler-TTS open for this
    token? Uses the worker's own check (whoami + gated-repo access; 401 and 403 each get their fix) — it stops with
    the steps, never bypasses the gate and never switches a character to another voice."""
    tok_ = hf_token()                                    # no token: stops with the Colab Secrets how-to
    check_hf_access(tok_, os.environ.get("PARLER_MODEL", "ai4bharat/indic-parler-tts"))
    tok_ = None
    DEVICE_REPORT["Parler access"] = "checked before synthesis"
    return "PASS"

if VOICE_BENCH:
    run_dialogue_bench() if PLAN.get("story_mode") == "two_friends_cinematic" else run_voice_bench()
if TELUGU_VOICE_BENCH or TELUGU_VOICE_AUDITION:
    run_telugu_bench()
if TELUGU_PRON_BENCH:
    run_pron_bench()
if gemini_wanted():
    run_gemini_voices()
elif LANGUAGE == "te" and STRICT_OPEN_SOURCE:
    print("Telugu voices: local Indic Parler-TTS (STRICT_OPEN_SOURCE: Gemini is opt-in — set TELUGU_TTS_BACKEND = "
          "\"gemini_free\" to use it)")
if not RENDER_CACHED_AUDIO_ONLY and LANGUAGE != "en" and not GEMINI_ACTIVE and (STRICT_OPEN_SOURCE or CODE_MIX) and any(
        not (PROJ / "narration" / f"{l['id']}.wav").exists() for s_ in SHOTS for l in s_["lines"]):
    PARLER_ACCESS = parler_gate()                        # only when a line still has to be made locally
RETAKE_AT = VOICE_POLICY.get("asr_retake_at", WER_RETRY)  # code-mixed Telugu: only a clearly wrong take is remade;
                                                          # a doubtful one is kept and flagged (WER_RETRY) for listening
FLAGGED, _voices_checked, TAKES = [], False, {}
for shot in SHOTS:
    for l in shot["lines"]:
        BUDGET.check("next narration line")
        want = line_settings(l)
        who = CHARS[l["speaker"]]["name"]
        out = PROJ / "narration" / f"{l['id']}.wav"
        old = read_sidecar(out)
        if out.exists() and old and all(old.get(k) == v for k, v in want.items()) \
                and (old.get("words") is not None or get_asr() is None):
            if old.get("flagged"):
                FLAGGED.append(l["id"])
            if len((old.get("takes") or {}).get("takes", [])) > 1:
                TAKES[l["id"]] = old["takes"]                 # approved takes are reused from the cache
            continue                                          # already made with these settings
        if not _voices_checked:
            missing = {c["voice"] for c in CHARS.values()} - set(get_tts().get_voices())
            assert not missing, f"voices not in this voice model: {missing} (available: {get_tts().get_voices()})"
            _voices_checked = True
        best = None
        # Kokoro gives the same audio for the same input, so retries change the speed slightly
        for attempt, sp in enumerate([want["speed"], round(want["speed"] - 0.05, 3), round(want["speed"] + 0.04, 3)][:MAX_TTS_ATTEMPTS]):
            BUDGET.check("next speech take")
            wav, sr = speak(want["text"], sp, want["voice"], attempt=attempt, instruct=want.get("instruct", ""))
            assert sr == SR, sr
            tmp = WORK / f"{l['id']}_try{attempt}.wav"
            sf.write(tmp, np.asarray(wav, dtype=np.float32), SR)
            heard, words = listen(tmp)
            err = None if heard is None else speech_error([l["text"], want["text"]], heard)
            if best is None or (err is not None and err < best["err"]):
                best = {"err": err if err is not None else 0.0, "tmp": tmp, "words": words, "speed": sp, "heard": heard}
            if err is None or err <= RETAKE_AT:
                break
            print(f"  {l['id']} ({who}): speech check heard {err:.0%} wrong words, trying again")
            log(f"    heard: {heard!r} | expected: {l['text']!r}", "WARN")
        ff("-i", best["tmp"], "-af", want["fx"], "-ar", SR, "-ac", 1, out)      # character voice + tone
        d_ = DIRECTIONS.get(l["id"])
        vq = check_line(out, want["text"], best["heard"], best["err"] if best["heard"] is not None else None)
        bad = [k for k, x in vq["checks"].items() if x["status"] == "FAIL" and not k.startswith(("words", "no missing"))
               and "(diagnostic)" not in k]
        if ALLOW_CALMER_RETAKE and bad and d_ is not None and backend_of(want["voice"]) in RETAKE_BACKENDS and d_.intensity > 0.15:
            # expressive take failed a signal check (odd pace, long gap, clipping...): once more, toned down
            calmer = d_.model_copy(update={"intensity": round(d_.intensity * 0.5, 2), "style_instruction": ""})
            calmer.style_instruction = voice_direction.instruction(calmer)
            wav2, _ = speak(want["text"], want["speed"], want["voice"], attempt=5, instruct=calmer.style_instruction)
            tmp2 = WORK / f"{l['id']}_calmer.wav"; sf.write(tmp2, np.asarray(wav2, dtype=np.float32), SR)
            heard2, words2 = listen(tmp2)
            err2 = None if heard2 is None else speech_error([l["text"], want["text"]], heard2)
            out2 = WORK / f"{l['id']}_calmer_fx.wav"; ff("-i", tmp2, "-af", want["fx"], "-ar", SR, "-ac", 1, out2)
            vq2 = check_line(out2, want["text"], heard2, err2)
            n_fail = lambda v: sum(x["status"] == "FAIL" for x in v["checks"].values())
            log(f"{l['id']}: expressive take failed {bad}; toned-down retake fails {n_fail(vq2)} vs {n_fail(vq)}")
            if n_fail(vq2) < n_fail(vq):
                shutil.copy(out2, out); vq = vq2
                best.update(heard=heard2, words=words2, err=err2 if err2 is not None else 0.0)
                vq["toned_down"] = calmer.intensity
        # ---- hero lines (opening, central question, reveal, ending): a bounded second take on backends whose takes
        # differ (Qwen3-TTS, Parler). Chosen by the checks only (fewer wrong words, then fewer failed checks), never
        # by "more pitch variation"; both takes are kept for the A/B listen below. Kokoro is deterministic: 1 take.
        takes = [{"take": 1, "err": best["err"] if best["heard"] is not None else None,
                  "fails": sum(x["status"] == "FAIL" for x in vq["checks"].values())}]
        n_takes = MAX_VOICE_TAKES_HERO if voice_direction.is_hero(l, shot, d_) else MAX_VOICE_TAKES_NORMAL
        if n_takes > 1 and get_asr() is not None and not ASR_ON_GPU:
            n_takes = 1                                   # each extra take costs a slow CPU speech check: skipped
            log(f"{l['id']}: speech check on CPU → no extra hero take")
        if n_takes > 1 and backend_of(want["voice"]) in RETAKE_BACKENDS:
            tdir = PROJ / "narration" / "takes"; tdir.mkdir(exist_ok=True)
            shutil.copy(out, tdir / f"{l['id']}_t1.wav")
            for k_ in range(2, n_takes + 1):
                try:                                      # an extra take is optional: its failure never stops the run
                    wk, _ = speak(want["text"], want["speed"], want["voice"], attempt=10 + k_, instruct=want.get("instruct", ""))
                except Exception as e:
                    log_exc(f"{l['id']}: extra take {k_} failed; keeping the takes already made")
                    takes.append({"take": k_, "err": None, "fails": 99, "error": repr(e)[:200]})
                    continue
                tk = WORK / f"{l['id']}_take{k_}.wav"; sf.write(tk, np.asarray(wk, dtype=np.float32), SR)
                tkf = tdir / f"{l['id']}_t{k_}.wav"; ff("-i", tk, "-af", want["fx"], "-ar", SR, "-ac", 1, tkf)
                hk, wdk = listen(tkf)
                ek = None if hk is None else speech_error([l["text"], want["text"]], hk)
                vqk = check_line(tkf, want["text"], hk, ek)
                takes.append({"take": k_, "err": ek, "fails": sum(x["status"] == "FAIL" for x in vqk["checks"].values())})
                if voice_qc.pick_take(takes, WER_RETRY) is takes[-1]:
                    shutil.copy(tkf, out); vq = vqk
                    best.update(heard=hk, words=wdk, err=ek if ek is not None else 0.0)
            sel = voice_qc.pick_take([t for t in takes if "error" not in t], WER_RETRY)["take"]
            log(f"{l['id']}: hero line, {len(takes)} takes {takes} → take {sel}")
            TAKES[l["id"]] = {"takes": takes, "selected_take_id": f"{l['id']}_t{sel}"}
        VQC[l["id"]] = vq
        flagged = best["heard"] is not None and best["err"] > WER_RETRY
        if flagged:
            FLAGGED.append(l["id"])
        sidecar(out, model={"kokoro": "kokoro-82M (onnx v1.0)", "parler": "ai4bharat/indic-parler-tts",
                            "qwen3-tts": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                            "gemini": GEMINI_TTS_MODEL}[backend_of(want["voice"])],
                backend=backend_of(want["voice"]), applied=expressive_tts.report(d_, backend_of(want["voice"])),
                voice_qc=vq, tts_info=dict(LAST_TTS_INFO) if backend_of(want["voice"]) == "qwen3-tts" else None, **want, speaker=l["speaker"], tone=l["tone"],
                takes=TAKES.get(l["id"], {"takes": [{"take": 1}], "selected_take_id": f"{l['id']}_t1"}),
                speed_used=best["speed"], wer=None if best["heard"] is None else round(best["err"], 3),
                heard=best["heard"], words=best["words"], flagged=flagged)
        info = sf.info(str(out))
        print(f"  {l['id']} {who:<9} [{l['tone']}] {info.frames / info.samplerate:.1f}s"
              + ("" if best["heard"] is None else f"  speech check {best['err']:.0%} wrong words")
              + ("  ⚠️ CHECK THIS LINE" if flagged else ""))
# ---- the same narrator throughout? (ECAPA leave-one-out similarity; calibrated in tests/test_voice.py) ----
if EXPRESSIVE:
    # Do not co-load ECAPA with Parler on a free T4 or start extra identity retakes.
    DEVICE_REPORT['speaker identity'] = 'Not run: manual listening required in this preset'
    print('Speaker identity requires listening review; ECAPA is omitted from the free-tier preset.')
    _lines = [l for s_ in SHOTS for l in s_["lines"]]
    _sc = {l["id"]: read_sidecar(PROJ / "narration" / f"{l['id']}.wav") or {} for l in _lines}
    (PROJ / "render" / "prosody_report.json").write_text(json.dumps({
        "backend": {k: expressive_tts.BACKENDS.get(v.get("backend", "kokoro"), {}) for k, v in _sc.items()},
        "lines": {k: {"text": v.get("text"), "direction": v.get("direction"), "applied": v.get("applied"),
                      "instruct": v.get("instruct"), "tts_info": v.get("tts_info")} for k, v in _sc.items()}},
        indent=1, default=str))
    (PROJ / "render" / "voice_qc_report.json").write_text(json.dumps(
        {k: VQC.get(k) or v.get("voice_qc") for k, v in _sc.items()}, indent=1, default=str))
    _fails = {k: [c for c, x in (VQC.get(k) or v.get("voice_qc") or {"checks": {}})["checks"].items()
                  if x["status"] == "FAIL" and "(diagnostic)" not in c] for k, v in _sc.items()}
    print("Voice QC: " + ("all lines pass" if not any(_fails.values()) else
                          "; ".join(f"{k}: {', '.join(f)}" for k, f in _fails.items() if f)))
close_tts(); _tts = _asr = None; free_gpu()          # free voice + speech models before the big image/video models
if TAKES:                                             # A/B listening for hero lines (inline, no web server)
    (PROJ / "render" / "voice_takes.json").write_text(json.dumps(TAKES, indent=1))
    try:
        from IPython.display import Audio, display
        for lid_, t_ in TAKES.items():
            print(f"{lid_}: selected {t_['selected_take_id']} (checks: {t_['takes']})")
            for x_ in t_["takes"]:
                if "error" not in x_:
                    display(Audio(str(PROJ / "narration" / "takes" / f"{lid_}_t{x_['take']}.wav")))
    except Exception:
        log_exc("inline A/B players unavailable (not in a notebook)")
print("Voices ready." + (f" ⚠️ Listen to: {FLAGGED}" if FLAGGED else "")); ram()

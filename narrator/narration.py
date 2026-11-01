# ==== 6. Narration: Kokoro voice per sentence, speed from the mood, checked by speech recognition ====
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
        try:
            from faster_whisper import WhisperModel
            src = str(hf_files(WHISPER_REPOS.get(ASR_MODEL, ASR_MODEL), "speech check")) \
                if (ASR_MODEL in WHISPER_REPOS or "/" in ASR_MODEL) and not os.path.isdir(ASR_MODEL) else ASR_MODEL
            try:                                           # GPU first; test it, because missing CUDA libs only fail at use
                if not GPU["name"]:
                    raise RuntimeError("no GPU in this session")
                preload_cuda_libs()
                _asr = WhisperModel(src, device="cuda", compute_type="float16")
                list(_asr.transcribe(np.zeros(16000, np.float32), language=LANGUAGE)[0])   # runs lazily, so consume it
                ASR_ON_GPU = True
                print("Speech check running on: GPU")
                DEVICE_REPORT["speech check (Whisper)"] = "GPU (float16)"
            except Exception as e_gpu:
                log_exc("faster-whisper on GPU failed, using CPU")
                ASR_ON_GPU = False
                _asr = WhisperModel(src, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 2)
                DEVICE_REPORT["speech check (Whisper)"] = f"CPU ({str(e_gpu)[:80]})"
                print(f"Speech check running on: CPU — the GPU attempt failed: {str(e_gpu)[:160]}\n"
                      "  (much slower; extra hero takes are skipped to save time; send the run log if this keeps happening)")
        except Exception as e:
            log_exc("speech check could not load")
            _asr_failed = True
            print("Speech check unavailable (subtitles will use estimated timing):", repr(e)[:200])
    return _asr

def listen(path):
    """Transcribe a file -> (text, [[word, start, end], ...]) or (None, None)."""
    global _asr_failed
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
        return " ".join(w[0] for w in words), words
    except Exception as e:
        log_exc("speech check failed")
        _asr_failed = True
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
        ready = self._read(timeout=1800)                 # first session: ~3.7 GB model download
        if not ready.get("ready"):
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
        r, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not r:
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
                self.proc.stdin.close(); self.proc.wait(timeout=60)
            except Exception:
                self.proc.kill()
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
    global _qwen_tts
    for t_ in (_tts, _qwen_tts):
        if t_ is not None and hasattr(t_, "close"):
            t_.close()
    _qwen_tts = None

FLAGGED = []
for shot in SHOTS:
    m = mood(shot)
    for s in shot["sentences"]:
        spoken = s.get("tts") or s["text"]
        speed = round(s.get("speed", m["speed"]) * SPEED_SCALE, 3)
        want = {"text": spoken, "voice": VOICE, "speed": speed}
        out = PROJ / "narration" / f"{s['id']}.wav"
        old = read_sidecar(out)
        if out.exists() and old and all(old.get(k) == v for k, v in want.items()) \
                and (old.get("words") is not None or get_asr() is None):
            if old.get("flagged"):
                FLAGGED.append(s["id"])
            continue                                          # already made with these settings
        best = None
        # Kokoro gives the same audio for the same input, so retries change the speed slightly
        for attempt, sp in enumerate([speed, round(speed - 0.05, 3), round(speed + 0.04, 3)]):
            wav, sr = speak(spoken, sp, attempt=attempt)
            assert sr == SR, sr
            tmp = WORK / f"{s['id']}_try{attempt}.wav"
            sf.write(tmp, np.asarray(wav, dtype=np.float32), SR)
            heard, words = listen(tmp)
            err = None if heard is None else speech_error([s["text"], spoken], heard)
            if best is None or (err is not None and err < best["err"]):
                best = {"err": err if err is not None else 0.0, "tmp": tmp, "words": words, "speed": sp, "heard": heard}
            if err is None or err <= WER_RETRY:
                break
            print(f"  {s['id']}: speech check heard {err:.0%} wrong words, trying again")
            log(f"    heard: {heard!r} | expected: {s['text']!r}", "WARN")
        shutil.copy(best["tmp"], out)
        flagged = best["heard"] is not None and best["err"] > WER_RETRY
        if flagged:
            FLAGGED.append(s["id"])
        sidecar(out, model="kokoro-82M (onnx v1.0)", **want, speed_used=best["speed"],
                wer=None if best["heard"] is None else round(best["err"], 3),
                heard=best["heard"], words=best["words"], flagged=flagged)
        info = sf.info(str(out))
        print(f"  {s['id']}: {info.frames / info.samplerate:.1f}s  speed {best['speed']}"
              + ("" if best["heard"] is None else f"  speech check {best['err']:.0%} wrong words")
              + ("  ⚠️ CHECK THIS LINE" if flagged else ""))
close_tts(); _tts = _asr = None; free_gpu()          # free voice + speech models before the big image/video models
print("Narration ready." + (f" ⚠️ Listen to: {FLAGGED}" if FLAGGED else "")); ram()

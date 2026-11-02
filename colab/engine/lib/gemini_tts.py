"""Google Gemini free-tier TTS: the project's ONE hosted exception, allowed for speech synthesis only.

Verified against the official docs on 2026-10-19 (ai.google.dev/gemini-api/docs: speech-generation, pricing,
rate-limits, billing, terms):
  * models gemini-3.8-flash-tts / gemini-3.8-flash-lite-tts: Telugu + up to 2 speakers (prebuilt voices) per request;
  * REST POST /v1beta/interactions (= SDK client.interactions.create; the SDK itself is not installed, see
    RestClient), per-turn `speech_metadata` (speaker, style),
    generation_config.speech_config = [{"voice"}] (one speaker) or {"speakers": [{"speaker", "voice"}]} (two);
  * unary responses are WAV (44-byte RIFF header, 24 kHz, mono, 16-bit) — never wrapped in a second header here;
  * Free tier = Standard (and Priority) only; Batch/Flex are NOT free → this module never sets a service tier;
  * free-tier input/output "used to improve our products" (human review possible) → only the approved dialogue text
    and short style words are sent; no research, no files, no credentials;
  * free-tier rate limits are per PROJECT and shown in AI Studio, not published per model → local budget below;
  * there is NO documented API to read a project's billing status → the user must confirm it (fails closed).

Statuses: PASS, FAIL, NOT_TESTED, RESOURCE_UNAVAILABLE, FREE_QUOTA_EXHAUSTED, ACCESS_REQUIRED, MANUAL_REVIEW_REQUIRED,
POLICY_BLOCKED.
"""
from __future__ import annotations

import base64, datetime, hashlib, io, json, os, re, time, wave
from pathlib import Path

FREE_TTS_MODELS = ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts")     # free tier verified on the pricing page
MIN_SDK = (2, 25, 0)
PREBUILT_VOICES = ("Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
                   "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
                   "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
                   "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat")
INLINE_TAGS = ("<short pause>", "<long pause>", "<laugh>", "<laughter>", "<chuckle>", "<giggle>", "<sigh>", "<gasp>")
# conversational intention → short turn-level style (an EXPERIMENT; empty = the model's own natural reading)
STYLE_FOR_INTENT = {"explaining_observation": "casual and conversational", "genuine_curiosity": "naturally curious",
                    "reflective_explanation": "warm and relaxed", "playful_disbelief": "lightly amused",
                    "quiet_realization": "thoughtful and reflective", "enthusiastic_discovery": "subtly excited",
                    "gentle_correction": "calm and reassuring", "satisfying_revelation": "warm and relaxed",
                    "guiding_demo": "casual and conversational", "playful_remark": "lightly amused",
                    "asking_confirmation": "naturally curious", "thinking_aloud": "thoughtful and reflective"}


class GeminiBlocked(Exception):
    """A policy / access / quota condition: the caller falls back to the local voice (never to a paid route)."""
    def __init__(self, status, why):
        super().__init__(f"{status}: {why}")
        self.status, self.why = status, why


def redact(text, secret):
    """Never let the key appear in a message, log or file."""
    text = str(text)
    if secret:
        text = text.replace(secret, "***")
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "AIza***", text)


# ------------------------------------------------------------------ preflight (no synthesis call)
def sdk_version():
    try:
        import importlib.metadata as md
        v = md.version("google-genai")
        return tuple(int(x) for x in re.findall(r"\d+", v)[:3]), v
    except Exception:
        return None, None


def preflight(cfg, key, ledger, project, client=None):
    """Every condition that must hold BEFORE any Gemini request. Returns {"status", "checks": {name: (status, why)}}.
    cfg: {"enabled", "model", "free_tier_confirmed", "free_tier_only", "max_videos_per_day", "max_calls_per_video"}.
    The key is only checked for presence (never printed). Model availability: one metadata call (models.get) when a
    client is given — not a synthesis call."""
    ck = {}
    ck["enabled in settings"] = ("PASS", "") if cfg.get("enabled") else ("NOT_TESTED", "Gemini TTS is off in the settings")
    ok_model = cfg.get("model") in FREE_TTS_MODELS
    ck["approved free-tier model"] = ("PASS", cfg.get("model")) if ok_model else (
        "POLICY_BLOCKED", f"{cfg.get('model')!r} is not an approved free-tier TTS model {FREE_TTS_MODELS}")
    ck["free tier confirmed by the user"] = ("PASS", "") if cfg.get("free_tier_confirmed") and cfg.get("free_tier_only", True) \
        else ("POLICY_BLOCKED", "set GEMINI_FREE_TIER_CONFIRMED = True only after AI Studio → Projects shows this "
                                "project's Billing Tier as Free (a key alone does not prove the project is unbilled)")
    ck["API key present"] = ("PASS", "") if key else ("ACCESS_REQUIRED", "add the Colab secret GEMINI_API_KEY")
    try:
        import requests  # noqa: F401  (REST client: no extra package, Colab's environment stays untouched)
        ck["HTTP client"] = ("PASS", "requests")
    except ImportError:
        ck["HTTP client"] = ("RESOURCE_UNAVAILABLE", "requests is not installed")
    today = ledger.today()
    # a voice audition / benchmark ("bench:…") is a one-off test, not a video: it never uses up the daily video
    others = [p for p in today.get("videos", []) if p != project and not p.startswith("bench:")]
    ck["daily video budget"] = ("PASS", f"{len(others)} other videos today") if len(others) < cfg.get("max_videos_per_day", 1) \
        else ("FREE_QUOTA_EXHAUSTED", f"MAX_VIDEOS_PER_DAY={cfg.get('max_videos_per_day', 1)} already used today: {others}")
    used = ledger.calls(project)
    ck["calls left for this video"] = ("PASS", f"{used} used") if used < cfg.get("max_calls_per_video", 6) \
        else ("FREE_QUOTA_EXHAUSTED", f"MAX_GEMINI_TTS_CALLS_PER_VIDEO={cfg.get('max_calls_per_video', 6)} used today: "
                                      "only audio already made (cached) can be used; no new request")
    if client is not None and all(s == "PASS" for s, _ in ck.values()):
        try:
            client.models.get(model=cfg["model"])
            ck["model available to this key"] = ("PASS", "")
        except Exception as e:
            st = classify(e)                       # a bad key / no access stops here; anything else is decided by
            ck["model available to this key"] = (  # the first real request (its errors are classified the same way)
                "ACCESS_REQUIRED" if st == "ACCESS_REQUIRED" else "NOT_TESTED", redact(e, key)[:200])
    # "calls left" is information: audio already made (cached on Drive) is reused for free, and every NEW request is
    # still refused by synthesize() once the budget is used — so a used-up budget can't block re-using cached audio
    soft = {("model available to this key", "NOT_TESTED"), ("calls left for this video", "FREE_QUOTA_EXHAUSTED")}
    bad = [s for k, (s, _) in ck.items() if s != "PASS" and (k, s) not in soft]
    return {"status": "PASS" if not bad else bad[0], "checks": ck}


def classify(e):
    """Map an API error to a status (429 → FREE_QUOTA_EXHAUSTED; 401/403 → ACCESS_REQUIRED; 404 → POLICY_BLOCKED)."""
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    msg = str(e)
    if code == 429 or "RESOURCE_EXHAUSTED" in msg or re.search(r"\b429\b", msg):
        return "FREE_QUOTA_EXHAUSTED"
    if code in (401, 403) or re.search(r"\b(401|403)\b|PERMISSION_DENIED|API key not valid", msg):
        return "ACCESS_REQUIRED"
    if code == 404 or "NOT_FOUND" in msg:
        return "POLICY_BLOCKED"
    return "FAIL"


def retry_delay_s(e):
    """The delay the API asks for (RetryInfo 'retryDelay': '23s'), if any."""
    m = re.search(r"retry(?:Delay|_delay)['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(e))
    return float(m.group(1)) if m else None


# ------------------------------------------------------------------ usage ledger (persistent, on Drive)
class Ledger:
    """Local safety budget, NOT Google's quota: per day the videos that used Gemini and every request made."""
    def __init__(self, path, clock=None):
        self.path, self.clock = Path(path), clock or (lambda: datetime.datetime.now())
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _day(self):
        return self.clock().strftime("%Y-%m-%d")

    def today(self):
        return self.data.setdefault(self._day(), {"videos": [], "requests": []})

    def calls(self, project):
        return sum(1 for r in self.today()["requests"] if r["project"] == project and not r.get("cache_hit"))

    def record(self, project, **info):
        d = self.today()
        if project not in d["videos"] and not info.get("cache_hit") and not project.startswith("bench:"):
            d["videos"].append(project)
        d["requests"].append({"project": project, "time": self.clock().strftime("%H:%M:%S"), **info})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))


# ------------------------------------------------------------------ request / response
def build_request(model, turns, voices, style_on=True):
    """turns: [{"speaker": "arjun", "text": "...", "style": "..."}] (1–2 distinct speakers); voices: {speaker: voice}.
    Only the transcript and short style words are sent (privacy)."""
    if model not in FREE_TTS_MODELS:
        raise GeminiBlocked("POLICY_BLOCKED", f"model {model!r} is not an approved free-tier TTS model")
    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    if len(speakers) > 2:
        raise ValueError("Gemini multi-speaker TTS takes at most 2 speakers per request")
    for s in speakers:
        if voices.get(s) not in PREBUILT_VOICES:
            raise ValueError(f"speaker {s}: {voices.get(s)!r} is not a prebuilt Gemini voice")
    content = []
    for t in turns:
        meta = {"type": "speech_metadata"}
        if len(speakers) == 2:
            meta["speaker"] = t["speaker"]
        if style_on and t.get("style"):
            meta["style"] = t["style"]
        item = {"type": "text", "text": t["text"]}
        if len(meta) > 1:
            item["annotations"] = [meta]
        content.append(item)
    speech = ({"mode": "conversational", "speakers": [{"speaker": s, "voice": voices[s]} for s in speakers]}
              if len(speakers) == 2 else [{"voice": voices[speakers[0]]}])   # as in Google's two-speaker example
    return {"model": model, "input": [{"type": "user_input", "content": content}],
            "response_format": {"type": "audio"}, "generation_config": {"speech_config": speech}}


def decode_audio(interaction):
    """The response audio as WAV bytes + its real format. Unary responses are WAV; raw PCM (l16) is wrapped ONCE."""
    out = getattr(interaction, "output_audio", None)
    data = getattr(out, "data", None) if out is not None else None
    if data is None and isinstance(interaction, dict):
        data = (interaction.get("output_audio") or {}).get("data")
    if not data:
        raise ValueError("the response has no audio")
    raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
    if raw[:4] != b"RIFF":
        mime = getattr(out, "mime_type", "") or ""
        if "l16" not in mime and "pcm" not in mime:
            raise ValueError(f"unexpected audio format (first bytes {raw[:4]!r}, mime {mime!r})")
        b = io.BytesIO()
        with wave.open(b, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(raw)
        raw = b.getvalue()
    return raw, wav_info(raw)


def wav_info(raw):
    """Validate WAV bytes; return the actual format (never assumed)."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    with wave.open(io.BytesIO(raw)) as w:
        n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
    if n == 0:
        raise ValueError("empty audio")
    if raw[44:48] == b"RIFF":
        raise ValueError("double WAV header (PCM was wrapped twice)")
    return {"sample_rate": sr, "channels": ch, "bit_depth": 8 * sw, "frames": n, "duration_s": round(n / sr, 3)}


# ------------------------------------------------------------------ cached synthesis with budget + 429 policy
def cache_key(model, turns, voices, style_on):
    return hashlib.sha1(json.dumps([model, [(t["speaker"], t["text"], t.get("style", "") if style_on else "")
                                            for t in turns], {k: voices[k] for k in sorted(voices)}, style_on],
                                   ensure_ascii=False).encode()).hexdigest()[:16]


def synthesize(client, cfg, turns, voices, cache_dir, ledger, project, key="", log=print, sleep=time.sleep,
               style_on=True, label=""):
    """One request (cached). Returns (wav path, info). Raises GeminiBlocked on policy/quota/access, so the caller
    falls back to the local voice. Retries at most cfg['max_retries'] times, only on 429 with a short server delay."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    k = cache_key(cfg["model"], turns, voices, style_on)
    wav_path, meta_path = cache_dir / f"{k}.wav", cache_dir / f"{k}.json"
    if wav_path.exists() and meta_path.exists():
        info = json.loads(meta_path.read_text())
        ledger.record(project, label=label, model=cfg["model"], key=k, cache_hit=True, status="PASS")
        return wav_path, dict(info, cache_hit=True)
    if ledger.calls(project) >= cfg.get("max_calls_per_video", 6):
        raise GeminiBlocked("FREE_QUOTA_EXHAUSTED", f"local budget: {cfg.get('max_calls_per_video', 6)} calls for this video used")
    req = build_request(cfg["model"], turns, voices, style_on)
    retries = 0
    while True:
        t0 = time.time()
        try:
            inter = client.interactions.create(**req)
        except Exception as e:
            st, why = classify(e), redact(e, key)[:300]
            ledger.record(project, label=label, model=cfg["model"], key=k, status=st, error=why,
                          retry=retries, transcript_hash=k, speakers=voices)
            d = retry_delay_s(e)
            if st == "FREE_QUOTA_EXHAUSTED" and retries < cfg.get("max_retries", 1) and d is not None and d <= 60:
                retries += 1
                log(f"Gemini 429: waiting {d:.0f}s as the API asks (retry {retries})")
                sleep(d)
                if ledger.calls(project) >= cfg.get("max_calls_per_video", 6):
                    raise GeminiBlocked("FREE_QUOTA_EXHAUSTED", "local budget used during retries")
                continue
            if st in ("FREE_QUOTA_EXHAUSTED", "ACCESS_REQUIRED", "POLICY_BLOCKED"):
                raise GeminiBlocked(st, why)
            raise GeminiBlocked("FAIL", why)
        raw, info = decode_audio(inter)
        wav_path.write_bytes(raw)                          # the ORIGINAL audio, before any resampling
        info.update(model=cfg["model"], voices=voices, turns=[(t["speaker"], t["text"]) for t in turns],
                    style_on=style_on, gen_s=round(time.time() - t0, 2), created=time.strftime("%Y-%m-%d %H:%M:%S"))
        meta_path.write_text(json.dumps(info, ensure_ascii=False, indent=1))
        ledger.record(project, label=label, model=cfg["model"], key=k, status="PASS", transcript_hash=k,
                      speakers=voices, duration_s=info["duration_s"], retry=retries)
        return wav_path, dict(info, cache_hit=False)


def plan_chunks(scenes, max_calls, reserve=2, max_s=40.0, sec_per_char=0.075):
    """Group consecutive scenes into conversation chunks: as few requests as possible (each ≤ ~max_s seconds of
    estimated speech), leaving `reserve` calls of the budget for repairs. scenes: [[(line_id, speaker, text)]]."""
    est = lambda lines: sum(len(re.sub(r"\s", "", t)) for _, _, t in lines) * sec_per_char
    chunks, cur = [], []
    for sc in scenes:
        if cur and est(cur + sc) > max_s:
            chunks.append(cur); cur = []
        cur = cur + sc
    if cur:
        chunks.append(cur)
    while len(chunks) > max(max_calls - reserve, 1) and len(chunks) > 1:      # merge the two shortest neighbours
        i = min(range(len(chunks) - 1), key=lambda j: est(chunks[j]) + est(chunks[j + 1]))
        chunks[i:i + 2] = [chunks[i] + chunks[i + 1]]
    return chunks


class ApiError(Exception):
    """An HTTP error from the Gemini API (code = HTTP status; the message carries Google's error JSON)."""
    def __init__(self, code, message):
        super().__init__(f"{code} {message}")
        self.code = code


class RestClient:
    """The documented REST form of the same API (POST /v1beta/interactions), with Colab's own `requests`.
    Why not the google-genai SDK: every version with TTS (2.25–2.31) needs google-auth >= 2.56, while Colab's
    google-colab pins google-auth == 2.49.0 — installing it breaks Colab's own package. The key travels only in
    the x-goog-api-key header (never in a URL, so it can't end up in a log line)."""
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, key, timeout=240, session=None):
        import types
        import requests
        self._key, self._timeout = key, timeout
        self._http = session or requests.Session()
        self.interactions = types.SimpleNamespace(create=self._create)
        self.models = types.SimpleNamespace(get=self._get)

    def _call(self, method, path, body=None):
        r = self._http.request(method, f"{self.BASE}/{path}", json=body, timeout=self._timeout,
                               headers={"x-goog-api-key": self._key, "Content-Type": "application/json"})
        if r.status_code >= 400:
            raise ApiError(r.status_code, r.text[:2000])
        return r.json()

    def _get(self, model):
        return self._call("GET", f"models/{model}")

    def _create(self, **req):
        import types
        d = self._call("POST", "interactions", req)
        audio = [c for st in d.get("steps", []) if st.get("type") == "model_output"
                 for c in st.get("content", []) if c.get("type") == "audio"]
        if not audio:
            raise ValueError("the response has no audio (no model_output step with audio content)")
        a = audio[-1]
        return types.SimpleNamespace(output_audio=types.SimpleNamespace(
            data=a.get("data"), mime_type=a.get("mime_type") or a.get("mimeType") or "audio/wav"))


def make_client(key):
    """The Gemini client used by the notebook (REST; see RestClient for why not the SDK)."""
    return RestClient(key)


# ------------------------------------------------------------------ the whole voice pass for one video
def _read_wav(path):
    import numpy as np, soundfile as sf
    w, sr = sf.read(str(path), dtype="float32")
    return (w.mean(1) if w.ndim > 1 else w).astype(np.float32), sr


def letter_error(expected, heard):
    """Share of the expected letters the recogniser did not hear (any script)."""
    import difflib
    from turn_align import letters
    e, h = letters(expected), letters(heard or "")
    if not e:
        return 0.0
    m = sum(b.size for b in difflib.SequenceMatcher(None, e, h, autojunk=False).get_matching_blocks())
    return round(1 - m / len(e), 3)


def run_voices(lines, voices, cfg, client, cache_dir, ledger, project, out_dir, asr_fn=None, key="", log=print,
               retry_at=0.35, speaker_fn=None, sleep=time.sleep, err_fn=None):
    """lines: [{"id", "scene", "speaker", "text" (what is SPOKEN), "intent"}] in order. Makes every line with
    Gemini (two-speaker chunks first, per-turn as the repair), splits and checks it, and writes
    out_dir/<id>.wav (original sample rate) — ONLY if every line succeeded (one voice per character for the whole
    video). Returns the report: {"status": PASS | FALLBACK, "why", "lines": {...}, "chunks": [...], "calls", "gaps"}.
    cfg["asr_repairs"] = False: a line the recogniser doubts is kept and flagged for listening, never re-requested
    (code-switched speech fools Whisper; an API call is not spent on the recogniser's doubt). err_fn(expected, heard)
    replaces the letter error (e.g. telugu_dialogue.bilingual_error for code-mixed lines)."""
    import numpy as np, soundfile as sf
    import turn_align
    letter_error_ = err_fn or letter_error
    asr_repairs = cfg.get("asr_repairs", True)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(cache_dir) / "split"; tmp.mkdir(parents=True, exist_ok=True)
    style = lambda l: STYLE_FOR_INTENT.get(l.get("intent"), "") if cfg.get("styles", True) else ""
    scenes = []
    for l in lines:
        if not scenes or scenes[-1][0]["scene"] != l["scene"]:
            scenes.append([])
        scenes[-1].append(l)
    by_id = {l["id"]: l for l in lines}
    groups = plan_chunks([[(l["id"], l["speaker"], l["text"]) for l in sc] for sc in scenes],
                         cfg.get("max_calls_per_video", 6), reserve=cfg.get("reserve_calls", 2))
    report = {"status": "PASS", "why": "", "lines": {}, "chunks": [], "gaps": {}, "mode": cfg.get("mode", "scene")}
    pieces = {}

    def check_piece(lid, wav, sr, words):
        heard = " ".join(w[0] for w in words) if words is not None else None
        err = None if heard is None else letter_error_(by_id[lid]["text"], heard)
        return heard, err

    def per_turn(lid):
        l = by_id[lid]
        for attempt in range(1 + cfg.get("max_retries", 1)):
            p, info = synthesize(client, cfg, [{"speaker": l["speaker"], "text": l["text"], "style": style(l)}],
                                 {l["speaker"]: voices[l["speaker"]]}, cache_dir, ledger, project, key, log, sleep,
                                 label=f"turn {lid}" + (f" retry {attempt}" if attempt else ""),
                                 style_on=attempt == 0)            # the retry drops the style: plain reading
            wav, sr = _read_wav(p)
            res = turn_align.align(wav, sr, [l["text"]])
            seg = turn_align.cut(wav, sr, res["spans"])[0]
            f = tmp / f"{lid}_turn.wav"; sf.write(f, seg, sr)
            heard, words = asr_fn(f) if asr_fn else (None, None)
            err = None if heard is None else letter_error_(l["text"], heard)
            if err is None or err <= retry_at:
                return seg, sr, {"how": "per-turn", "err": err, "heard": heard, "words": words, "cache": info.get("cache_hit")}
            if not asr_repairs:                            # the recogniser's doubt costs no second request
                log(f"Gemini {lid}: {err:.0%} of the letters not heard → kept, flagged for listening")
                return seg, sr, {"how": "per-turn", "err": err, "heard": heard, "words": words,
                                 "cache": info.get("cache_hit"), "review": True}
            log(f"Gemini {lid}: {err:.0%} of the letters not heard; " + ("retrying without style" if attempt == 0 else "giving up"))
        return None, None, {"how": "per-turn", "err": err, "heard": heard}

    for g in groups:
        ids = [lid for lid, _, _ in g]
        rec = {"lines": ids, "mode": cfg.get("mode", "scene")}
        speakers = {by_id[i]["speaker"] for i in ids}
        ok_chunk = False
        if cfg.get("mode", "scene") == "scene" and len(speakers) <= 2 and len(ids) > 1:
            try:
                p, info = synthesize(client, cfg, [{"speaker": by_id[i]["speaker"], "text": by_id[i]["text"],
                                                    "style": style(by_id[i])} for i in ids],
                                     {s: voices[s] for s in speakers}, cache_dir, ledger, project, key, log, sleep,
                                     label=f"scene {ids[0]}..{ids[-1]}")
                wav, sr = _read_wav(p)
                heard, words = asr_fn(p) if asr_fn else (None, None)
                res = turn_align.align(wav, sr, [by_id[i]["text"] for i in ids], words=words)
                rec.update(align=res["status"], source=res.get("source"), why=res.get("why"), duration_s=info["duration_s"],
                           cache_hit=info.get("cache_hit"))
                if res["status"] == "PASS":
                    segs = turn_align.cut(wav, sr, res["spans"])
                    bad = []
                    for i, (lid, seg, (a, b)) in enumerate(zip(ids, segs, res["spans"])):
                        ws = [w for w in (words or []) if a - 0.05 <= w[1] < b] if words is not None else None
                        ht = " ".join(w[0] for w in ws) if ws is not None else None
                        err = None if ht is None else letter_error_(by_id[lid]["text"], ht)
                        pieces[lid] = (seg, sr, {"how": "two-speaker chunk", "err": err, "heard": ht,
                                                 "words": [[w[0], round(w[1] - a, 3), round(w[2] - a, 3)] for w in ws] if ws else ws})
                        if i:
                            report["gaps"][lid] = int(1000 * res["gaps"][i - 1])
                        if err is not None and err > retry_at:
                            bad.append(lid)
                    ok_chunk = True
                    rec["cuts_without_pause"] = res.get("cuts_without_pause", [])
                    left = cfg.get("max_calls_per_video", 6) - ledger.calls(project)
                    if bad and (len(bad) > left or not asr_repairs):
                        # not enough calls to repair them all: keep the conversation take, a person listens to these
                        # lines (Whisper's Telugu is weak: a "miss" is often the recogniser, not the voice)
                        for lid in bad:
                            pieces[lid][2]["review"] = True
                        rec["flagged_for_review"] = bad
                        log(f"Gemini: {len(bad)} lines not clearly heard, {left} calls left → kept, flagged for listening")
                        bad = []
                    for lid in bad:                        # targeted repair of one turn, not the whole chunk
                        seg, sr2, m = per_turn(lid)
                        if seg is None:
                            ok_chunk = False; pieces.pop(lid, None)
                        else:
                            pieces[lid] = (seg, sr2, dict(m, how="per-turn repair"))
                    rec["repaired"] = bad
            except GeminiBlocked as e:
                report.update(status="FALLBACK", why=f"{e.status}: {e.why}"); report["chunks"].append(rec); break
        if not ok_chunk and report["status"] == "PASS":
            need = len([i for i in ids if i not in pieces])
            left = cfg.get("max_calls_per_video", 6) - ledger.calls(project)
            if need > left:                                 # don't spend calls on a job that can't be finished
                report.update(status="FALLBACK", why=f"chunk {ids[0]}..{ids[-1]} could not be split "
                              f"({rec.get('why', '')}) and per-line requests need {need} calls, only {left} left")
                report["chunks"].append(rec)
                break
            for lid in ids:                                 # the chunk couldn't be split reliably: per turn
                if lid in pieces:
                    continue
                try:
                    seg, sr2, m = per_turn(lid)
                except GeminiBlocked as e:
                    report.update(status="FALLBACK", why=f"{e.status}: {e.why}"); break
                if seg is None:
                    report.update(status="FALLBACK", why=f"{lid}: words not heard correctly after a retry"); break
                pieces[lid] = (seg, sr2, m)
            rec["mode"] = "per-turn"
        report["chunks"].append(rec)
        if report["status"] != "PASS":
            break
    if report["status"] == "PASS" and speaker_fn is not None:     # who is speaking? (voice embeddings)
        embs = {lid: speaker_fn(seg, sr) for lid, (seg, sr, _) in pieces.items()}
        cents = {}
        for s in {l["speaker"] for l in lines}:
            cents[s] = [embs[i] for i in embs if by_id[i]["speaker"] == s]
        wrong = []
        for lid, e in embs.items():
            sims = {}
            for s, E in cents.items():
                others = [x for x, i in zip(E, [i for i in embs if by_id[i]["speaker"] == s]) if i != lid]
                if others:
                    c = np.mean(others, axis=0); sims[s] = float(e @ (c / (np.linalg.norm(c) + 1e-9)))
            if sims and max(sims, key=sims.get) != by_id[lid]["speaker"]:
                wrong.append((lid, sims))
        report["attribution"] = {"misattributed": wrong}
        if wrong:
            report.update(status="FALLBACK", why=f"speaker attribution unclear for {[w[0] for w in wrong]}")
    if report["status"] == "PASS" and len(pieces) == len(lines):
        for lid, (seg, sr, m) in pieces.items():
            sf.write(out_dir / f"{lid}.wav", seg, sr)
            report["lines"][lid] = {k: v for k, v in m.items() if k != "words"} | {"words": m.get("words")}
    elif report["status"] == "PASS":
        report.update(status="FALLBACK", why="not every line could be made")
    report["calls"] = ledger.calls(project)
    return report

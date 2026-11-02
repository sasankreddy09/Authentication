"""Voice quality control for narration chunks: objective measurements, each check PASS / FAIL / UNKNOWN.

Measured per chunk: spoken words vs the approved text (missing / repeated words), duration and speaking rate,
RMS / peak / clipping, silence at the edges and long gaps inside, pitch distribution and jumps (librosa pYIN),
and voice identity (ECAPA speaker embedding similarity to the narrator's reference; SpeechBrain, Apache-2.0).
Pitch and loudness are DIAGNOSTIC: variation is not proof that an emotion came across. Thresholds = TH (settings).
"""
from __future__ import annotations

import difflib, re

import numpy as np

TH = {"clip_level": 0.995, "clip_max_fraction": 0.001, "silence_db": -45.0, "edge_silence_max_s": 1.0,
      "gap_max_s": 1.2, "rate_wps": (1.4, 4.2), "rate_sps": (2.0, 7.0), "pitch_jump_semitones": 12.0,
      "min_rms_db": -40.0,
      "speaker_min_similarity": 0.80}
# speaker_min_similarity: each chunk vs the leave-one-out mean of the narrator's other chunks. Calibrated on Kokoro
# (6 voices, 4 sentences, 2 speeds): same voice ≥ 0.856, closest different voice (af_heart vs af_bella) ≤ 0.737.
# Chunk-vs-chunk pairs overlap (0.69 vs 0.76), so single comparisons are NOT used. Re-check per backend.


def _db(x):
    return 20 * np.log10(max(float(x), 1e-9))


UNITS = {"cm": "centimetres", "centimeters": "centimetres", "mm": "millimetres", "millimeters": "millimetres",
         "km": "kilometres", "kilometers": "kilometres", "kg": "kilograms", "g": "grams", "m": "metres",
         "meters": "metres", "mph": "miles per hour", "hz": "hertz", "s": "seconds", "min": "minutes", "hrs": "hours"}


def syllables(text: str) -> int:
    """Rough English syllable count (vowel groups, silent final e): speaking rate in syllables/s doesn't punish a
    sentence of short words the way words/s does ('So how does each eye get only its own picture?')."""
    n = 0
    for w in re.findall(r"[a-z]+", text.lower()):
        k = len(re.findall(r"[aeiouy]+", w))
        if w.endswith("e") and not w.endswith(("le", "ee")) and k > 1:
            k -= 1
        n += max(k, 1)
    return n


def measure(wav: np.ndarray, sr: int, words: int, th=TH, text: str | None = None) -> dict:
    """Signal measurements of one chunk (mono float32). With the text, the pace is also in syllables/s."""
    x = np.asarray(wav, dtype=np.float32).squeeze()
    dur = len(x) / sr
    out = {"duration_s": round(dur, 3), "words": words, "empty": len(x) < sr * 0.1}
    if out["empty"]:
        return out
    hop = int(sr * 0.02)
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    lev = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    loud = lev > th["silence_db"]
    out.update(rms_db=round(_db(np.sqrt((x ** 2).mean())), 1), peak_db=round(_db(np.abs(x).max()), 1),
               clip_fraction=round(float((np.abs(x) >= th["clip_level"]).mean()), 5))
    if loud.any():
        first, last = int(np.argmax(loud)), len(loud) - 1 - int(np.argmax(loud[::-1]))
        out["lead_silence_s"], out["tail_silence_s"] = round(first * 0.02, 2), round((len(loud) - 1 - last) * 0.02, 2)
        inner, gap, longest = loud[first:last + 1], 0, 0
        for v in inner:
            gap = 0 if v else gap + 1
            longest = max(longest, gap)
        out["longest_gap_s"] = round(longest * 0.02, 2)
        speech_s = (last - first + 1) * 0.02
        out["speaking_rate_wps"] = round(words / max(speech_s, 0.1), 2)
        if text:
            out["speaking_rate_sps"] = round(syllables(text) / max(speech_s, 0.1), 2)
    try:
        import librosa
        y = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
        f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=16000, frame_length=1024)
        f0 = f0[voiced & np.isfinite(f0)]
        if len(f0) > 5:
            st = 12 * np.log2(f0 / np.median(f0))
            out.update(f0_median_hz=round(float(np.median(f0)), 1), f0_spread_semitones=round(float(np.std(st)), 2),
                       f0_max_jump_semitones=round(float(np.abs(np.diff(st)).max()), 1),
                       voiced_fraction=round(float(voiced.mean()), 2))
    except ImportError:
        out["pitch"] = "librosa not available"
    return out


# words that SOUND the same: a recogniser can't tell them apart, so hearing one for the other is not a missing word
HOMOPHONES = [{"eye", "i", "aye"}, {"eyes", "is"}, {"two", "to", "too"}, {"four", "for"}, {"right", "write"},
              {"their", "there", "they're"}, {"here", "hear"}, {"one", "won"}, {"know", "no"}, {"see", "sea"},
              {"whole", "hole"}, {"its", "it's"}, {"your", "you're"}, {"weight", "wait"}, {"by", "buy", "bye"}]


def _homophones(a, b):
    return a == b or any(a in g and b in g for g in HOMOPHONES)


def word_diff(expected: list[str], heard: list[str]) -> dict:
    """Missing / extra / repeated words between the approved text and what the recogniser heard (normalised words)."""
    unit = lambda ws: [x for w in ws for x in UNITS.get(w, w).split()]      # "6 cm" was heard for "six centimetres"
    expected, heard = unit(expected), unit(heard)
    sm = difflib.SequenceMatcher(None, expected, heard, autojunk=False)
    missing, extra, variants = [], [], []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "replace":
            # the same sounds written differently is not a missing word: berth/birth, travelling/traveling,
            # AM/"a m", waitlisted/"weight listed" (measured false alarms) — compared letter by letter
            e_, h_ = "".join(expected[a0:a1]), "".join(heard[b0:b1])
            same_sound = a1 - a0 == b1 - b0 and all(_homophones(x, y) for x, y in zip(expected[a0:a1], heard[b0:b1]))
            if same_sound or difflib.SequenceMatcher(None, e_, h_, autojunk=False).ratio() >= 0.6:
                variants.append((" ".join(expected[a0:a1]), " ".join(heard[b0:b1]))); continue
        if op in ("delete", "replace"):
            missing += expected[a0:a1]
        if op in ("insert", "replace"):
            extra += heard[b0:b1]
    repeated = [w for i, w in enumerate(heard[1:], 1) if w == heard[i - 1] and len(w) > 2
                and not any(expected[j] == expected[j - 1] == w for j in range(1, len(expected)))]
    return {"missing": missing, "extra": extra, "repeated": repeated, "heard_as": variants}


def verdict(m: dict, speech_err: float | None, retry_at: float, similarity: float | None, diff: dict | None,
            th=TH) -> dict:
    """Checks for one chunk → {check: {status, value}}; 'ok' = no FAIL (UNKNOWN is reported, the gate decides)."""
    C = lambda st, v=None: {"status": st, "value": v}
    c = {}
    c["audio present"] = C("FAIL" if m.get("empty") else "PASS", m.get("duration_s"))
    if m.get("empty"):
        return {"checks": c, "ok": False}
    c["words correct (speech check)"] = C("UNKNOWN" if speech_err is None else "PASS" if speech_err <= retry_at else "FAIL",
                                          speech_err)
    if diff is not None:
        c["no missing or repeated words"] = C("PASS" if not diff["missing"] and not diff["repeated"] else "FAIL", diff)
    lo, hi = th["rate_sps"] if "speaking_rate_sps" in m else th["rate_wps"]
    r = m.get("speaking_rate_sps", m.get("speaking_rate_wps"))
    c["speaking rate natural"] = C("UNKNOWN" if r is None else "PASS" if lo <= r <= hi else "FAIL", r)
    c["no clipping"] = C("PASS" if m["clip_fraction"] <= th["clip_max_fraction"] else "FAIL", m["clip_fraction"])
    c["audible level"] = C("PASS" if m["rms_db"] >= th["min_rms_db"] else "FAIL", m["rms_db"])
    c["no long silence inside"] = C("PASS" if m.get("longest_gap_s", 0) <= th["gap_max_s"] else "FAIL", m.get("longest_gap_s"))
    c["no long silence at the edges"] = C("PASS" if max(m.get("lead_silence_s", 0), m.get("tail_silence_s", 0))
                                         <= th["edge_silence_max_s"] else "FAIL",
                                         [m.get("lead_silence_s"), m.get("tail_silence_s")])
    j = m.get("f0_max_jump_semitones")
    c["no extreme pitch jumps (diagnostic)"] = C("UNKNOWN" if j is None else "PASS" if j <= th["pitch_jump_semitones"]
                                                 else "FAIL", j)
    c["same narrator voice"] = C("UNKNOWN" if similarity is None else
                                 "PASS" if similarity >= th["speaker_min_similarity"] else "FAIL", similarity)
    # diagnostic checks are reported, but never decide (pitch movement is not proof of anything)
    return {"checks": c, "ok": not any(x["status"] == "FAIL" for k, x in c.items() if "(diagnostic)" not in k)}


class SpeakerCheck:
    """ECAPA speaker embeddings (speechbrain/spkrec-ecapa-voxceleb, Apache-2.0), unit length. Used by
    consistency(): a seed alone doesn't guarantee the same voice, so every chunk is measured."""
    MODEL = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self, savedir, device="auto"):
        from speechbrain.inference.speaker import EncoderClassifier
        if device == "auto":                              # the GPU when there is one (small model, ~20 MB)
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"   # speechbrain wants "cuda:<index>"
        self.device = device
        self.m = EncoderClassifier.from_hparams(self.MODEL, savedir=str(savedir), run_opts={"device": device})

    def embed(self, wav, sr):
        import torch
        x = np.asarray(wav, dtype=np.float32).squeeze()
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        with torch.inference_mode():
            e = self.m.encode_batch(torch.from_numpy(x)[None]).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)



# per backend: deterministic Kokoro varies little between chunks (calibrated: 0.80); Qwen3-TTS samples, so even
# two plain takes of one speaker differ more (measured LOO 0.77–0.81 over 3 takes incl. styled ones): PROVISIONAL
# 0.72 — recalibrate from VOICE_BENCH's voice_qc_report.json on the T4.
SPEAKER_TH = {"kokoro": 0.80, "qwen3-tts": 0.72, "parler": 0.72, "gemini": 0.72}   # gemini: provisional, uncalibrated


MIN_IDENTITY_S = 2.0      # shorter clips are not judged (and not used as reference)
SHORT_CLIP_S, SHORT_TH = 3.5, 0.70
# Calibration on two-friend dialogue (2026-10-18, 37 Kokoro lines, am_michael + af_heart, ECAPA leave-one-out):
#   same voice: < 3 s min 0.684 (median 0.81); 3–3.5 s min 0.822; ≥ 3.5 s min 0.874. Other friend: max 0.18.
# The 0.80 bar (calibrated on longer narrator chunks vs SIMILAR voices) only made false alarms on short lines.


def identity_threshold(backend: str, duration_s: float) -> float | None:
    """Pass mark for the identity similarity of one line (None = too short to judge)."""
    if duration_s < MIN_IDENTITY_S:
        return None
    base = SPEAKER_TH.get(backend, TH["speaker_min_similarity"])
    return min(base, SHORT_TH) if duration_s < SHORT_CLIP_S else base


def consistency(embs: dict, th=TH, durations: dict | None = None) -> dict:
    """Narrator identity over the whole narration: each chunk vs the mean of ALL the speaker's OTHER chunks
    (leave-one-out). {chunk id: similarity}; below th["speaker_min_similarity"] = suspicious (regenerate/review).
    Needs ≥ 3 chunks; with fewer, nothing can be judged (UNKNOWN). With durations, chunks shorter than
    MIN_IDENTITY_S are not judged (None) and not used as reference."""
    short = {i for i, d in (durations or {}).items() if d < MIN_IDENTITY_S}
    if short:
        out = consistency({i: e for i, e in embs.items() if i not in short}, th)
        return {i: out.get(i) for i in embs}
    ids = list(embs)
    if len(ids) < 3:
        return {i: None for i in ids}
    E = np.stack([embs[i] for i in ids])
    total = E.sum(axis=0)
    out = {}
    for k, i in enumerate(ids):
        r = (total - E[k]) / (len(ids) - 1)
        out[i] = round(float(E[k] @ (r / (np.linalg.norm(r) + 1e-9))), 3)
    return out


def conversation_diagnostics(lines, max_gap_s=1.5):
    """Timing of a dialogue as a whole. lines: [{"id", "speaker", "start", "end", "rms_db", "sps"}] in order.
    Technical diagnostics only (a person still judges whether it sounds like a real conversation)."""
    gaps = [round(b["start"] - a["end"], 3) for a, b in zip(lines, lines[1:])]
    out = {"n_gaps": len(gaps), "issues": []}
    if gaps:
        g = np.array(gaps)
        out.update(gap_mean_s=round(float(g.mean()), 3), gap_sd_s=round(float(g.std()), 3),
                   gap_min_s=round(float(g.min()), 3), gap_max_s=round(float(g.max()), 3),
                   gap_cv=round(float(g.std() / max(g.mean(), 1e-6)), 3))
        if len(gaps) >= 6 and out["gap_cv"] < 0.2:
            out["issues"].append(f"mechanical rhythm: gaps vary only {out['gap_cv']:.0%} (sd/mean)")
        for a, b, x in zip(lines, lines[1:], gaps):
            if x > max_gap_s:
                out["issues"].append(f"dead air {x:.2f}s before {b['id']}")
            if x < -0.02:
                out["issues"].append(f"overlap {-x:.2f}s: {a['id']} / {b['id']}")
    by = {}
    for l in lines:
        by.setdefault(l["speaker"], []).append(l)
    for sp, ls in by.items():
        rates = [l["sps"] for l in ls if l.get("sps")]
        if len(rates) >= 3:
            med = float(np.median(rates))
            for l in ls:
                if l.get("sps") and abs(l["sps"] - med) / med > 0.35:
                    out["issues"].append(f"{l['id']}: speaking rate {l['sps']:.1f} syll/s vs {sp}'s usual {med:.1f}")
        for a, b in zip(ls, ls[1:]):
            if a.get("rms_db") is not None and b.get("rms_db") is not None and abs(a["rms_db"] - b["rms_db"]) > 4:
                out["issues"].append(f"{b['id']}: level jump {b['rms_db'] - a['rms_db']:+.1f} dB vs {sp}'s previous line")
    return out


def pick_take(takes: list[dict], retry_at: float) -> dict:
    """Best of several takes of one line by the CHECKS only: first acceptable words (≤ retry_at wrong), then fewer
    wrong words, then fewer failed checks; earlier take on a tie. Never by pitch variation or "liveliness"."""
    return min(takes, key=lambda t: ((t.get("err") or 0.0) > retry_at, t.get("err") or 0.0, t.get("fails", 0),
                                     t.get("take", 0)))

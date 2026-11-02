"""Expressive TTS backends behind one interface, and an HONEST mapping of voice direction to what each backend can do.

Every direction field is reported per line as one of:
  "native"       the backend has a real control for it (e.g. Kokoro speed),
  "instruction"  requested in a natural-language instruction the model was trained to follow — an effect is likely
                 but not guaranteed; the voice QC measures what actually came out,
  "description"  the same for description-conditioned models (Parler),
  "silence"      done by inserting real silence between chunks (pauses),
  "unsupported"  the backend can't do it: NOT applied, and reported as such (never claimed as applied).
DSP pitch shifting is never used to fake emotion.
"""
from __future__ import annotations

from voice_direction import EMOTION_WORDS, VoiceDirection, instruction

FIELDS = ("emotion", "intensity", "energy", "pace_multiplier", "emphasis_words", "pause_before_ms", "pause_after_ms")
CAPS = {
    "kokoro":    {"emotion": "unsupported", "intensity": "unsupported", "energy": "unsupported",
                  "pace_multiplier": "native", "emphasis_words": "unsupported",
                  "pause_before_ms": "silence", "pause_after_ms": "silence"},
    "qwen3-tts": {"emotion": "instruction", "intensity": "instruction", "energy": "instruction",
                  "pace_multiplier": "instruction", "emphasis_words": "instruction",
                  "pause_before_ms": "silence", "pause_after_ms": "silence"},
    "gemini":    {"emotion": "style", "intensity": "unsupported", "energy": "unsupported",
                  "pace_multiplier": "unsupported", "emphasis_words": "unsupported",
                  "pause_before_ms": "natural", "pause_after_ms": "silence"},     # style = speech_metadata.style;
                  # natural = the model's own gap between turns inside a two-speaker request (kept as measured)
    "parler":    {"emotion": "description", "intensity": "description", "energy": "description",
                  "pace_multiplier": "description", "emphasis_words": "unsupported",
                  "pause_before_ms": "silence", "pause_after_ms": "silence"},
}
BACKENDS = {
    "gemini": {"model": "gemini-3.8-flash-tts (Google Gemini API, free tier: hosted exception, speech only)",
               "license": "proprietary hosted service (free-tier terms)", "languages": ["te", "en", "..."],
               "identity": "prebuilt voice per character", "instruction": True, "params": "n/a (hosted)",
               "vram_gb": "0 (remote)"},
    "kokoro": {"model": "hexgrad/Kokoro-82M (kokoro-onnx)", "license": "apache-2.0", "languages": ["en"],
               "identity": "fixed stock voice", "instruction": False, "params": "82M", "vram_gb": "~0.3 (or CPU)"},
    "qwen3-tts": {"model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "license": "apache-2.0",
                  "languages": ["en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
                  "identity": "fixed premium speaker (e.g. aiden, ryan)", "instruction": True, "params": "1.7B",
                  "vram_gb": "~7 fp32 (T4: measure with VOICE_BENCH)"},
    "parler": {"model": "ai4bharat/indic-parler-tts", "license": "apache-2.0", "languages": ["te", "hi", "en", "..."],
               "identity": "trained named speaker + description", "instruction": "description", "params": "0.9B",
               "vram_gb": "~4 fp32"},
}


def report(d: VoiceDirection | None, backend: str) -> dict:
    """What happened to each direction field on this backend (see the module docstring)."""
    if d is None:
        return {"direction": "none"}
    caps = CAPS[backend]
    out = {}
    for f in FIELDS:
        v = getattr(d, f)
        neutral = (f == "emotion" and v == "neutral") or (f == "intensity" and v == 0) or \
                  (f == "pace_multiplier" and v == 1.0) or (f == "emphasis_words" and not v) or \
                  (f == "pause_before_ms" and v == 0) or (f == "energy" and v == "medium")
        out[f] = {"value": v, "how": "not needed" if neutral else caps[f]}
    out["unsupported"] = [f for f, x in out.items() if isinstance(x, dict) and x["how"] == "unsupported"]
    return out


def kokoro_params(d: VoiceDirection | None, base_speed: float) -> dict:
    """Kokoro can only take the pace natively (speed); everything else is reported unsupported."""
    return {"speed": round(base_speed * (d.pace_multiplier if d else 1.0), 3)}


def qwen_params(d: VoiceDirection | None, speaker: str) -> dict:
    """Qwen3-TTS CustomVoice: one fixed speaker (identity) + the direction as an instruction (style)."""
    directed = d is not None and (d.emotion != "neutral" or d.intensity > 0)
    return {"speaker": speaker, "language": "English",
            "instruct": (d.style_instruction or instruction(d)) if directed else ""}


def parler_style(d: VoiceDirection | None) -> str:
    """Indic Parler-TTS: the direction becomes part of the speaker description (emotion support for Telugu is
    'not extensively tested' per its model card: the QC decides)."""
    if d is None or d.emotion == "neutral":
        return ""
    pace = "slightly slowly" if d.pace_multiplier < 0.97 else "slightly quickly" if d.pace_multiplier > 1.03 else "at a moderate pace"
    expr = "slightly expressive" if d.intensity < 0.4 else "expressive"
    return f"with {EMOTION_WORDS[d.emotion]}, {expr}, {pace}"

"""Final quality control: measurements of the rendered video (FFmpeg / ffprobe) and the approval gate.

Every check is a dict {"status": PASS|FAIL|UNKNOWN, "mandatory": bool, "evidence": ..., "shots": [...], "fix": str}.
`approved(checks)` is True only when every MANDATORY check is PASS: an UNKNOWN (couldn't be measured) blocks
approval exactly like a FAIL, so nothing unverified slips into an "approved" render.
Thresholds are engineering settings (THRESHOLDS), not universal truths.
"""
from __future__ import annotations

import json, re, subprocess
from pathlib import Path

THRESHOLDS = {"true_peak_max_dbtp": -1.0,      # above this the mix may clip after AAC encoding
              "loudness_target_lufs": -14.0,  # configurable target, not a platform requirement
              "loudness_tolerance_lu": 1.5,
              "silence_db": -45.0,            # quieter than this for...
              "silence_min_s": 1.5,           # ...this long = unintended silence (the music bed should fill pauses)
              "black_min_s": 0.5,
              "freeze_noise": 0.003, "freeze_min_s": 0.8,
              "duration_drift_s": 0.5,
              "duck_min_db": 4.0}             # music must be at least this much lower while someone speaks


def check(status, mandatory=True, evidence=None, shots=None, fix=""):
    assert status in ("PASS", "FAIL", "UNKNOWN"), status
    return {"status": status, "mandatory": mandatory, "evidence": evidence, "shots": shots or [], "fix": fix}


def approved(checks: dict) -> bool:
    return all(c["status"] == "PASS" for c in checks.values() if c["mandatory"])


def _run(cmd):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True)


def probe(path) -> dict:
    """Streams of a media file: video codec/size/fps/pix_fmt, audio codec/rate/channels, durations."""
    r = _run(["ffprobe", "-v", "error", "-show_entries",
              "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt,sample_rate,channels,duration",
              "-of", "json", path])
    if r.returncode:
        raise RuntimeError(f"ffprobe failed: {r.stderr.strip()[:300]}")
    d = json.loads(r.stdout)
    out = {"duration": float(d["format"]["duration"]), "video": None, "audio": None}
    for s in d["streams"]:
        if s["codec_type"] == "video" and out["video"] is None:
            num, den = (int(x) for x in s["r_frame_rate"].split("/"))
            out["video"] = {"codec": s["codec_name"], "width": s["width"], "height": s["height"],
                            "fps": round(num / max(den, 1), 3), "pix_fmt": s.get("pix_fmt"),
                            "duration": float(s.get("duration") or out["duration"])}
        elif s["codec_type"] == "audio" and out["audio"] is None:
            out["audio"] = {"codec": s["codec_name"], "sample_rate": int(s["sample_rate"]), "channels": s["channels"],
                            "duration": float(s.get("duration") or out["duration"])}
    return out


def video_events(path, th=THRESHOLDS) -> dict:
    """Black and frozen stretches, in ONE decoding pass."""
    r = _run(["ffmpeg", "-hide_banner", "-i", path, "-vf",
              f"blackdetect=d={th['black_min_s']}:pix_th=0.10,freezedetect=n={th['freeze_noise']}:d={th['freeze_min_s']}",
              "-an", "-f", "null", "-"])
    black = [(float(a), float(b)) for a, b in re.findall(r"black_start:(\S+) black_end:(\S+)", r.stderr)]
    fs = [float(x) for x in re.findall(r"freeze_start: (\S+)", r.stderr)]
    fe = [float(x) for x in re.findall(r"freeze_end: (\S+)", r.stderr)]
    dur = probe(path)["duration"]
    freezes = [(a, fe[i] if i < len(fe) else dur) for i, a in enumerate(fs)]   # a freeze to the end has no end line
    return {"black": black, "freezes": freezes}


def audio_stats(path, th=THRESHOLDS) -> dict:
    """Integrated loudness, true peak and silent stretches, in ONE decoding pass."""
    r = _run(["ffmpeg", "-hide_banner", "-i", path, "-vn", "-af",
              f"ebur128=peak=true,silencedetect=n={th['silence_db']}dB:d={th['silence_min_s']}", "-f", "null", "-"])
    lufs = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    peaks = re.findall(r"Peak:\s+(-?[\d.]+|-inf) dBFS", r.stderr)
    ss = [float(x) for x in re.findall(r"silence_start: (\S+)", r.stderr)]
    se = [float(x) for x in re.findall(r"silence_end: (\S+)", r.stderr)]
    dur = probe(path)["duration"]
    return {"integrated_lufs": float(lufs[-1]) if lufs else None,
            "true_peak_dbtp": (max(float(p) for p in peaks if p != "-inf") if any(p != "-inf" for p in peaks) else None),
            "silences": [(a, se[i] if i < len(se) else dur) for i, a in enumerate(ss)]}


def _levels(path, sr=8000, hop=0.1):
    """RMS level in dB every `hop` seconds (mono, via FFmpeg → raw float samples)."""
    import numpy as np
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(sr),
                        "-f", "f32le", "-"], capture_output=True)
    x = np.frombuffer(r.stdout, np.float32)
    n = int(sr * hop)
    x = x[: len(x) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt((x ** 2).mean(axis=1)) + 1e-9)


def ducking(voice_wav, ducked_music_wav, voice_on_db=-35.0) -> dict:
    """How much quieter the (already ducked) music bed is while the voice speaks than in the pauses.
    Measured, not assumed: a broken sidechain shows ~0 dB."""
    import numpy as np
    v, m = _levels(voice_wav), _levels(ducked_music_wav)
    n = min(len(v), len(m))
    v, m = v[:n], m[:n]
    speak, pause = v > voice_on_db, v < voice_on_db - 15
    if speak.sum() < 5 or pause.sum() < 5:
        return {"reduction_db": None, "why": "not enough speech or pauses to compare"}
    return {"reduction_db": round(float(np.median(m[pause]) - np.median(m[speak])), 1),
            "speech_s": round(float(speak.sum()) * 0.1, 1), "pause_s": round(float(pause.sum()) * 0.1, 1)}


def contact_sheet(video, out, n=12, cols=4, width=1600):
    """n evenly spaced frames in a grid, for a person to look over before publishing."""
    dur = probe(video)["duration"]
    step = max(dur / n, 0.1)
    rows = -(-n // cols)
    r = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video, "-vf",
              f"fps=1/{step:.3f},scale={width // cols}:-2,tile={cols}x{rows}", "-frames:v", "1", "-q:v", "3", out])
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[:300])
    return Path(out)


def overlaps(spans, a, b, min_s=0.0):
    """Total seconds of `spans` [(start, end)] inside [a, b]; True if more than min_s."""
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in spans) > min_s

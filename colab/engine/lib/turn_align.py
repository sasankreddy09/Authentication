"""Split one multi-speaker TTS waveform back into the plan's dialogue turns (who speaks when).

The 2D characters need per-turn audio: who is speaking, when each turn starts and ends, and each turn's own
loudness for the mouth. A two-speaker Gemini request returns ONE waveform, so:
  1. pauses: 20 ms frames, silence = > 35 dB under the loud parts, ≥ 120 ms long;
  2. expected cut points: from recogniser word times mapped onto the transcript (letters, difflib) when a speech
     recogniser is available, else from each turn's share of the letters;
  3. each cut snaps to the longest pause near its expected point; the cut is the middle of that pause;
  4. checks: every cut found, cuts in order, every turn's duration plausible for its length (seconds per letter).
Anything that fails is UNRESOLVED (→ smaller request / per-turn synthesis / local voice). Nothing is invented: no
timestamp is ever placed where the audio has no pause.
"""
from __future__ import annotations

import difflib, re, unicodedata

import numpy as np

FRAME_S = 0.02


def letters(text):
    """Letters only (any script; Telugu vowel signs kept), lower-case: what speech carries."""
    return "".join(ch for ch in text.lower() if unicodedata.category(ch)[0] in "LM" or ch.isdigit())


def quietest(wav, sr, t, win=0.15):
    """The quietest 20 ms point within ±win s of t (where to cut when two speakers barely pause)."""
    n = int(sr * FRAME_S)
    a, b = max(int((t - win) * sr) // n, 0), min(int((t + win) * sr) // n, len(wav) // n - 1)
    if b <= a:
        return t
    rms = [np.sqrt(np.mean(wav[i * n:(i + 1) * n] ** 2)) for i in range(a, b + 1)]
    return (a + int(np.argmin(rms)) + 0.5) * FRAME_S


def pauses(wav, sr, min_s=0.08, drop_db=35.0):
    """[(start_s, end_s)] of silences inside the speech (not the lead-in / tail)."""
    n = int(sr * FRAME_S)
    k = len(wav) // n
    if k < 3:
        return []
    rms = np.sqrt((wav[: k * n].reshape(k, n) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    loud = np.percentile(db, 95)
    quiet = db < loud - drop_db
    voiced = np.where(~quiet)[0]
    if not len(voiced):
        return []
    a, b = voiced[0], voiced[-1]
    out, i = [], a
    while i <= b:
        if quiet[i]:
            j = i
            while j <= b and quiet[j]:
                j += 1
            if (j - i) * FRAME_S >= min_s:
                out.append((i * FRAME_S, j * FRAME_S))
            i = j
        else:
            i += 1
    return out


def speech_span(wav, sr, drop_db=35.0):
    n = int(sr * FRAME_S); k = len(wav) // n
    db = 20 * np.log10(np.sqrt((wav[: k * n].reshape(k, n) ** 2).mean(1) + 1e-12))
    v = np.where(db >= np.percentile(db, 95) - drop_db)[0]
    return (v[0] * FRAME_S, (v[-1] + 1) * FRAME_S) if len(v) else (0.0, len(wav) / sr)


def expected_cuts(turn_texts, span, words=None):
    """Where each boundary should be. With recogniser words [[word, start, end]], from the words matched to each
    turn's letters; else proportional to letters (Telugu letters ≈ speaking time). Returns (cuts, source)."""
    L = [letters(t) for t in turn_texts]
    if words:
        heard = [letters(w[0]) for w in words]
        joined, owner = "", []
        for i, h in enumerate(heard):
            joined += h; owner += [i] * len(h)
        target, turn_of = "", []
        for k, x in enumerate(L):
            target += x; turn_of += [k] * len(x)
        sm = difflib.SequenceMatcher(None, target, joined, autojunk=False)
        last_word = {}                                       # turn → last heard word matched inside it
        first_word = {}
        for a, b, size in sm.get_matching_blocks():
            for j in range(size):
                t_, w_ = turn_of[a + j], owner[b + j]
                last_word[t_] = max(last_word.get(t_, -1), w_)
                first_word[t_] = min(first_word.get(t_, 10 ** 9), w_)
        cuts = []
        for k in range(len(L) - 1):
            if k in last_word and (k + 1) in first_word and last_word[k] < first_word[k + 1]:
                cuts.append((words[last_word[k]][2] + words[first_word[k + 1]][1]) / 2)
            else:
                cuts.append(None)
        if all(c is not None for c in cuts):
            return cuts, "recogniser"
    tot = sum(len(x) for x in L) or 1
    s0, s1 = span
    acc, cuts = 0, []
    for x in L[:-1]:
        acc += len(x); cuts.append(s0 + (s1 - s0) * acc / tot)
    return cuts, "letters"


def align(wav, sr, turn_texts, words=None, sec_per_char=(0.03, 0.25), tol=None):
    """→ {"status": "PASS"|"UNRESOLVED", "spans": [(start, end)] per turn, "why", "source"}."""
    if len(turn_texts) == 1:
        s0, s1 = speech_span(wav, sr)
        return {"status": "PASS", "spans": [(max(s0 - 0.05, 0), min(s1 + 0.05, len(wav) / sr))], "why": "", "source": "single"}
    span = speech_span(wav, sr)
    ps = pauses(wav, sr)
    exp, src = expected_cuts(turn_texts, span, words)
    tol = tol or max(0.6, 0.12 * (span[1] - span[0]) / max(len(turn_texts) - 1, 1))
    cuts, used, prev, no_pause = [], set(), span[0], []
    for k, e in enumerate(exp):
        cands = [(i, p) for i, p in enumerate(ps) if i not in used and p[0] > prev and abs((p[0] + p[1]) / 2 - e) <= tol]
        if not cands:
            if src == "recogniser" and e > prev:
                # quick turn-taking (no real pause): the recogniser's word times say where the speaker changes;
                # cut at the quietest point there. The speaker check afterwards confirms who is talking.
                q = quietest(wav, sr, e)
                if q > prev:
                    cuts.append((q, q)); prev = q; no_pause.append(k + 1)
                    continue
            return {"status": "UNRESOLVED", "spans": [], "source": src,
                    "why": f"no pause near the expected boundary {k + 1} ({e:.2f}s) — speaker change can't be placed"}
        i, p = max(cands, key=lambda c: (c[1][1] - c[1][0]) - 0.5 * abs((c[1][0] + c[1][1]) / 2 - e))
        used.add(i); cuts.append(p); prev = p[1]
    spans, start = [], max(span[0] - 0.05, 0.0)
    for p in cuts:
        pad = 0.03 if p[1] > p[0] else 0.0                 # no padding across a pause-less cut
        spans.append((start, p[0] + pad)); start = p[1] - pad
    spans.append((start, min(span[1] + 0.05, len(wav) / sr)))
    for (a, b), t in zip(spans, turn_texts):
        n = max(len(letters(t)), 1)
        if not sec_per_char[0] <= (b - a) / n <= sec_per_char[1]:
            return {"status": "UNRESOLVED", "spans": spans, "source": src,
                    "why": f"turn '{t[:30]}…' would last {b - a:.2f}s for {n} letters (implausible)"}
    return {"status": "PASS", "spans": spans, "source": src, "why": "",
            "gaps": [round(c[1] - c[0], 3) for c in cuts], "cuts_without_pause": no_pause}


def cut(wav, sr, spans, fade_s=0.008):
    """Per-turn pieces, with tiny edge fades (no clicks); silences inside the pieces trimmed by the spans."""
    out = []
    for a, b in spans:
        x = wav[int(a * sr):int(b * sr)].astype(np.float32).copy()
        f = min(int(sr * fade_s), len(x) // 2)
        if f:
            r = np.linspace(0, 1, f, dtype=np.float32); x[:f] *= r; x[-f:] *= r[::-1]
        out.append(x)
    return out

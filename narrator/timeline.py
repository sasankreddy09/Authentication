# ==== 7. Timeline: shot lengths come from the narration, pauses come from the mood ====
PRE, TAIL = 0.25, 1.2                               # default lead-in per shot, extra hold at the very end
t, TIMELINE, PLACED = 0.0, [], []
for i, shot in enumerate(SHOTS):
    m, start = mood(shot), t
    if not shot["sentences"]:
        dur = float(shot.get("fixed_s", 3.0))
    else:
        cur = start + float(shot.get("pre_s", PRE))
        for s in shot["sentences"]:
            out = PROJ / "narration" / f"{s['id']}.wav"
            info = sf.info(str(out)); d = info.frames / info.samplerate
            PLACED.append({"id": s["id"], "text": s["text"], "start": cur, "end": cur + d,
                           "words": (read_sidecar(out) or {}).get("words")})
            cur += d + float(s.get("pause_after", m["pause_after"]))   # breathing space after the line
        dur = cur - start
    if i == len(SHOTS) - 1:
        dur += TAIL
    dur = round(dur * FPS) / FPS                    # whole frames
    TIMELINE.append({"shot_id": shot["shot_id"], "mood": shot.get("mood", "calm"), "start": round(start, 3),
                     "dur": dur, "frames": round(dur * FPS)})
    t = start + dur
TOTAL = t

# full narration track, each sentence placed at its exact time (never stretched)
track = np.zeros(int(math.ceil(TOTAL * SR)) + SR // 10, dtype=np.float32)
for p in PLACED:
    w, _ = sf.read(str(PROJ / "narration" / f"{p['id']}.wav"), dtype="float32")
    a = int(p["start"] * SR); track[a:a + len(w)] += w[: len(track) - a]
sf.write(WORK / "narration_full.wav", track, SR)

# subtitles: one line of at most 42 characters per cue, timed to the spoken words
def _lines(text, maxlen=42):
    out, cur = [], ""
    for wd in text.split():
        if cur and len(cur) + 1 + len(wd) > maxlen:
            out.append(cur); cur = wd
        else:
            cur = (cur + " " + wd).strip()
    return out + ([cur] if cur else [])

def token_times(p):
    """Start time (relative to the sentence) of every word of the script text."""
    toks, dur = p["text"].split(), p["end"] - p["start"]
    cp = np.cumsum([0] + [len(tk) + 1 for tk in toks])[:-1].astype(float)
    total = cp[-1] + len(toks[-1])
    xs, ys = [0.0], [0.0]
    if p.get("words"):                               # match script words to what the recogniser heard
        dn = [" ".join(norm_words(tk)) for tk in toks]
        an = [" ".join(norm_words(w[0])) for w in p["words"]]
        for a, b, size in difflib.SequenceMatcher(None, dn, an, autojunk=False).get_matching_blocks():
            for k in range(size):
                if cp[a + k] > xs[-1]:
                    xs.append(cp[a + k]); ys.append(p["words"][b + k][1])
    xs.append(total); ys.append(dur)
    ys = np.maximum.accumulate(np.clip(ys, 0, dur))
    return np.interp(cp, xs, ys)                     # unmatched words: interpolated by position

def _ts(x):
    ms = int(round(x * 1000)); h, ms = divmod(ms, 3600000); m_, ms = divmod(ms, 60000); s_, ms = divmod(ms, 1000)
    return f"{h:02}:{m_:02}:{s_:02},{ms:03}"

cues, word_timed, ass_cues = [], 0, []
for p in PLACED:
    tt, k, lines = token_times(p), 0, _lines(p["text"])
    word_timed += bool(p.get("words"))
    starts = []
    for l in lines:
        starts.append(p["start"] + (0.0 if k == 0 else tt[k])); k += len(l.split())
    ends = starts[1:] + [p["end"]]
    cues += [(a, max(b, a + 0.6), l) for a, b, l in zip(starts, ends, lines)]
    k = 0                                            # the same cues, word by word, for the highlighted ASS subtitles
    for a, b, l in zip(starts, ends, lines):
        ws = l.split()
        ass_cues.append((a, max(b, a + 0.6), [(w, p["start"] + tt[k + j]) for j, w in enumerate(ws)],
                         p.get("speaker", "narrator")))
        k += len(ws)
srt = "\n".join(f"{k}\n{_ts(a)} --> {_ts(b)}\n{txt}\n" for k, (a, b, txt) in enumerate(cues, 1))
(WORK / "subtitles.srt").write_text(srt); (PROJ / "render" / "subtitles.srt").write_text(srt)
_emph = [w for w in PLAN.get("vocabulary", []) if " " not in w]
_cols = {k: c.get("color", "#FFFFFF") for k, c in (CHARS.items() if "CHARS" in globals() else [])}
n_ass = subtitles.write_ass(WORK / "subtitles.ass", ass_cues, W, H, font=SUB_FONT, colors=_cols, emphasis=_emph)
shutil.copy(WORK / "subtitles.ass", PROJ / "render" / "subtitles.ass")
(PROJ / "render" / "timeline.json").write_text(json.dumps({"total_s": TOTAL, "shots": TIMELINE,
    "sentences": [{k: v for k, v in p.items() if k != "words"} for p in PLACED]}, indent=2))
for row in TIMELINE:
    print(f"  {row['shot_id']}: {row['start']:6.2f}s  +{row['dur']:.2f}s  [{row['mood']}]")
print(f"Total length: {TOTAL:.1f}s, {len(cues)} subtitle cues "
      f"({word_timed}/{len(PLACED)} sentences timed word by word)")

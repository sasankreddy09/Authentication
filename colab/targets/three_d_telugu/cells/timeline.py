# ==== 7. Timeline: shot lengths come from the lines; pauses come from the tone (or the line) ====
PRE, TAIL = 0.3, 1.5                                 # lead-in per shot, extra hold at the very end
t, TIMELINE, PLACED = 0.0, [], []
for i, shot in enumerate(SHOTS):
    m, start = mood(shot), t
    cur = start + float(shot.get("pre_s", PRE))
    for l in shot["lines"]:
        out = PROJ / "narration" / f"{l['id']}.wav"
        info = sf.info(str(out)); d = info.frames / info.samplerate
        dv = (globals().get("DIRECTIONS") or {}).get(l["id"])
        if dv is not None:
            cur += dv.pause_before_ms / 1000            # a meaningful silence before a reveal / question
        PLACED.append({"id": l["id"], "speaker": l["speaker"], "tone": l["tone"], "text": l["text"],
                       "start": cur, "end": cur + d, "words": (read_sidecar(out) or {}).get("words")})
        pause = l.get("pause_after", dv.pause_after_ms / 1000 if dv is not None else
                      TONES[l["tone"]].get("pause_after", m["pause_after"]))
        cur += d + float(pause)                       # breathing space after the line
    dur = max(cur - start, float(shot.get("min_s", 0)))
    if i == len(SHOTS) - 1:
        dur += TAIL
    dur = round(dur * FPS) / FPS                    # whole frames
    TIMELINE.append({"shot_id": shot["shot_id"], "mood": shot.get("mood", "calm"), "start": round(start, 3),
                     "dur": dur, "frames": round(dur * FPS)})
    t = start + dur
TOTAL = t
if TOTAL > MAX_VIDEO_DURATION_SECONDS:
    raise RuntimeError(f'Measured narration makes a {TOTAL:.1f}s video, over the {MAX_VIDEO_DURATION_SECONDS}s '
                       'local limit. Review the saved narration before changing the budget.')

# all voices on one track, each line at its exact time (never stretched)
track = np.zeros(int(math.ceil(TOTAL * SR)) + SR // 10, dtype=np.float32)
for p in PLACED:
    w, _ = sf.read(str(PROJ / "narration" / f"{p['id']}.wav"), dtype="float32")
    n_ = min(int(SR * 0.008), len(w) // 2)            # 8 ms fades at the chunk's (silent) edges: no clicks
    if n_:
        w = w.copy(); ramp = np.linspace(0, 1, n_, dtype=np.float32); w[:n_] *= ramp; w[-n_:] *= ramp[::-1]
    a = int(p["start"] * SR); track[a:a + len(w)] += w[: len(track) - a]
sf.write(WORK / "narration_full.wav", track, SR)

# subtitles: at most 42 characters per cue, timed to the spoken words, coloured per character (narrator in italics)
def _lines(text, maxlen=42):
    out, cur = [], ""
    for wd in text.split():
        if cur and len(cur) + 1 + len(wd) > maxlen:
            out.append(cur); cur = wd
        else:
            cur = (cur + " " + wd).strip()
    return out + ([cur] if cur else [])

def token_times(p):
    """Start time (relative to the line) of every word of the script text."""
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

def styled(txt, speaker):
    c = CHARS[speaker]
    return f"<i>{txt}</i>" if speaker == "narrator" else f'<font color="{c.get("color", "#FFFFFF")}">{txt}</font>'

cues, word_timed, ass_cues = [], 0, []
for p in PLACED:
    tt, k, lines = token_times(p), 0, _lines(p["text"])
    word_timed += bool(p.get("words"))
    starts = []
    for l in lines:
        starts.append(p["start"] + (0.0 if k == 0 else tt[k])); k += len(l.split())
    ends = starts[1:] + [p["end"]]
    cues += [(a, max(b, a + 0.6), styled(l, p["speaker"])) for a, b, l in zip(starts, ends, lines)]
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

# the script, screenplay style (handy for the YouTube description or for checking the story)
script = [PLAN["title"].upper(), ""]
for shot in SHOTS:
    script.append(f"[{shot['shot_id']}: " + (shot.get("scene") or f"{shot.get('recipe')}: {shot.get('label', '')}") + "]")
    for l in shot["lines"]:
        script.append(f"  {CHARS[l['speaker']]['name'].upper()} ({l['tone']}): {l['text']}")
    script.append("")
(PROJ / "render" / "script.txt").write_text("\n".join(script))

def describe(shot):
    """What the viewer sees, in one line (for the storyboard)."""
    r = shot.get("recipe", "video")
    if r == "video":
        return f"AI video: {shot['scene']}. {shot['action']}"
    if r == "ui":
        heads = [w.get("text") for sc in shot["ui"]["screens"] for w in sc if w.get("type") == "header"]
        acts = [e["do"] + (f" {e.get('target', '')}" if e.get("target") else "") for e in shot["ui"].get("events", [])]
        return f"App screen ({' → '.join(heads)}): " + ", ".join(acts)
    if r == "journey":
        names = {s_["id"]: s_["label"] for s_ in PLAN["journey"]["stations"]}
        return "Journey: " + " → ".join(names[x] for x in shot["journey"].get("path", [])) + (
            f" (calendar: {shot['journey']['calendar']['days']} days)" if shot["journey"].get("calendar") else "")
    if r == "card":
        return "Card: " + " / ".join(shot["card"]["lines"])
    if r == "stock":
        return f"Licensed image ({shot['stock']['license']}): {shot['stock'].get('description', '')} — {shot['stock']['credit']}"
    if r == "story":                                  # the scene's own visual timeline (cuts inside the dialogue)
        return " → ".join(f"[{v['intent']} · {v['medium']}] {v.get('description', '')}"
                          + (f" ({v['label']})" if v.get("label") else "") for v in shot["story"]["visuals"])
    return r

_cues = musiclib.arrange(PLAN, TIMELINE, TOTAL) if MUSIC_STYLE == "library" else []
def cue_at(t):
    return next((c["track"] for c in _cues if c["start"] <= t < c["end"]), "generated score" if not _cues else "")
_by_shot = {}
for i in globals().get("ISSUES", []):
    _by_shot.setdefault(i["shot"], []).append(f"{i['level']} {i['rule']}")
claims_by_id = {c["id"]: c for c in PLAN.get("claims", [])}
sb = [f"# Storyboard: {PLAN['title']}", "", f"Total {TOTAL:.1f} s, {len(SHOTS)} scenes. Generated by the notebook "
      "from the plan (do not edit; edit plan.json).", "",
      "| # | Time | Beat / step | What the viewer sees | Narration / dialogue | Music · SFX | Sources | Checks |",
      "|---|---|---|---|---|---|---|---|"]
for row, shot in zip(TIMELINE, SHOTS):
    said = " ".join(f"**{CHARS[l['speaker']]['name']}** ({l['tone']}): {l['text']}" for l in shot["lines"])
    sfx_ = ", ".join(c["type"] for c in shot.get("sfx", []))
    src = ", ".join(shot.get("claims", []))
    sb.append(f"| {shot['shot_id']} | {row['start']:.1f}–{row['start'] + row['dur']:.1f}s "
              f"| {shot.get('beat', '')} {('· ' + shot['step']) if shot.get('step') else ''} "
              f"| {describe(shot)}{(' · label: ' + shot['label']) if shot.get('label') else ''} | {said} "
              f"| {cue_at(row['start'] + 0.01)}{(' · ' + sfx_) if sfx_ else ''} | {src} "
              f"| {', '.join(_by_shot.get(shot['shot_id'], [])) or 'ok'} |")
if PLAN.get("claims"):
    sb += ["", "## Sources", ""] + [f"- **{c['id']}**: {c['text']} — " + ", ".join(c["sources"]) for c in PLAN["claims"]]
(PROJ / "render" / "storyboard.md").write_text("\n".join(sb) + "\n")
(PROJ / "render" / "timeline.json").write_text(json.dumps({"total_s": TOTAL, "shots": TIMELINE,
    "lines": [{k: v for k, v in p.items() if k != "words"} for p in PLACED]}, indent=2))
for row, shot in zip(TIMELINE, SHOTS):
    who = ", ".join(CHARS[l["speaker"]]["name"] for l in shot["lines"])
    long = row["dur"] > SHOT_BUDGET + 0.6 and not shot.get("allow_long") and shot.get("recipe") not in DRAWN_RECIPES | {"stock"}
    print(f"  {row['shot_id']}: {row['start']:6.2f}s  +{row['dur']:.2f}s  [{row['mood']}]  {who}"
          + (f"  ⚠️ longer than one clip ({SHOT_BUDGET:.1f}s): the last frame will be held; split this shot" if long else ""))
# short-form limit: two-friend videos (MAX_VIDEO_DURATION_SECONDS) or a plan's own "max_duration_s"
MAX_DURATION = PLAN.get("max_duration_s") or (globals().get("MAX_VIDEO_DURATION_SECONDS")
                                               if PLAN.get("story_mode") == "two_friends_cinematic" else None)
if MAX_DURATION and TOTAL > MAX_DURATION:
    # never sped up or cut to fit: the script must be shortened (fact-preserving revision), then Run all again
    print(f"⚠️ {TOTAL:.1f}s is longer than the {MAX_DURATION}s limit: shorten the script (remove or merge a beat, "
          "keep every sourced fact) — the final check will not approve it as is.")
    log(f"video {TOTAL:.1f}s exceeds the {MAX_DURATION}s limit", "WARN")
print(f"Total length: {TOTAL:.1f}s, {len(cues)} subtitle cues "
      f"({word_timed}/{len(PLACED)} lines timed word by word)")

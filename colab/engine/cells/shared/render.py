# ==== 12. Edit: every shot (cached ones are reused) → join → mix voices, music, effects → final.mp4 ====
t_wait = time.time()
for j in EDIT_JOBS:                                   # let background edits finish first
    j.result()
wait_for_diagrams()
if EDIT_JOBS:
    log(f"waited {time.time() - t_wait:.1f}s for {len(EDIT_JOBS)} background edits")
segs, n_video, n_cached = [], 0, 0
for shot, row in zip(SHOTS, TIMELINE):
    out, how, is_video, cached = render_segment(shot, row)
    n_video += is_video; n_cached += cached
    hold = HOLDS.get(shot["shot_id"], 0)
    STATUS[shot["shot_id"]] = f"{how} [{row['mood']}]" + (f"  ⚠️ last frame held {hold:.1f}s" if hold > 0.6 else "")
    segs.append(out)
    print(f"  {shot['shot_id']}: {STATUS[shot['shot_id']]}" + ("  (already edited)" if cached else ""))
N_VIDEO_SHOTS = sum(s.get("recipe", "video") == "video" for s in SHOTS)   # drawn and licensed-image shots aren't AI video
print(f"{n_video}/{N_VIDEO_SHOTS} video shots are real AI video, {len(SHOTS) - N_VIDEO_SHOTS} drawn or licensed-image scenes. "
      f"{n_cached} shots reused from the edit cache.")

lst = WORK / "segments.txt"
lst.write_text("".join(f"file '{p}'\n" for p in segs))
video = WORK / "video_only.mp4"
ff("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video)

final = WORK / "final.mp4"
def duck_filter():
    """The music ducking (narration = sidechain control signal); the QC rebuilds the same bed to measure it."""
    return "sidechaincompress=" + ":".join(f"{k}={v}" for k, v in DUCK.items())

AF = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"   # FFmpeg 4.4 (Colab's Ubuntu 22.04) can't
#                                     negotiate formats for sidechaincompress by itself: state them on every input
fc = (f"[1:a]aresample=48000,pan=stereo|c0=c0|c1=c0,{AF},asplit=2[vo][sc];"
      f"[2:a]aresample=48000,{AF},volume={MUSIC_GAIN}[m];"
      f"[m][sc]{duck_filter()}[duck];"
      f"[3:a]aresample=48000,{AF}[fx];"
      "[vo][duck][fx]amix=inputs=3:duration=first:normalize=0,"
      "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[a]")
def drawn_subtitles(srt_path):
    """Subtitles drawn by Pillow + raqm (the same shaping as the cards) as ONE transparent image stream, for scripts
    the FFmpeg subtitle burner may not shape: libass in the tested FFmpeg 4.4 build detached Telugu vowel signs
    (మ ీరు instead of మీరు). Same place, size, outline and speaker colours as the English subtitles; no italics
    (Telugu has no true italic). Returns the concat file for FFmpeg's image-sequence input."""
    from PIL import ImageDraw
    ts = lambda x: (lambda h, m, r: int(h) * 3600 + int(m) * 60 + float(r.replace(",", ".")))(*x.split(":"))
    blocks = re.findall(r"^\d+\n(\S+) --> (\S+)\n(.*?)(?=\n\n|\Z)", srt_path.read_text(), re.S | re.M)
    d = WORK / "subs_png"; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    blank = d / "blank.png"; Image.new("RGBA", (W, H), (0, 0, 0, 0)).save(blank)
    f = drawkit.font(round(H * 0.052), bold=True)
    items, t = [], 0.0
    for k, (a, b, txt) in enumerate(blocks):
        a, b = max(ts(a), t), ts(b)
        if b <= a:
            continue
        m = re.search(r'color="(#[0-9A-Fa-f]{6})"', txt)
        plain = re.sub(r"<[^>]+>", "", txt).strip()
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((W / 2, H - H * 30 / 288), plain, font=f, fill=m.group(1) if m else "#FFFFFF",
                                anchor="ms", stroke_width=max(2, H // 270), stroke_fill="black")
        png = d / f"{k:04d}.png"; im.save(png)
        if a > t:
            items.append((blank, a - t))
        items.append((png, b - a)); t = b
    items += [(blank, 1.0), (blank, None)]          # the concat demuxer ignores the last entry's duration
    lst = d / "subs.txt"
    lst.write_text("".join(f"file '{p}'\n" + (f"duration {du:.3f}\n" if du is not None else "") for p, du in items))
    log(f"drawn subtitles: {len(blocks)} cues, font {SUB_FONT}")
    return lst

extra_in = []
if BURN_SUBS and LANGUAGE != "en":                    # non-Latin script: draw the subtitles ourselves (shaped)
    extra_in = ["-f", "concat", "-safe", 0, "-i", drawn_subtitles(WORK / "subtitles.srt")]
    fc += ";[0:v][4:v]overlay=0:0:eof_action=pass:format=auto[v]"
    vmap, vcodec = "[v]", VENC
elif BURN_SUBS and SUB_STYLE == "karaoke":           # word-highlighted ASS (libass), one pass in the final encode
    fc += f";[0:v]ass={WORK / 'subtitles.ass'}" + (f":fontsdir={FONTS_DIR}" if FONTS_DIR else "") + "[v]"
    vmap, vcodec = "[v]", VENC
elif BURN_SUBS:
    fc += (f";[0:v]subtitles={WORK / 'subtitles.srt'}" + (f":fontsdir={FONTS_DIR}" if FONTS_DIR else "")
           + f":force_style='FontName={SUB_FONT},FontSize=18,Outline=1,Shadow=0,MarginV=30'[v]")
    vmap, vcodec = "[v]", VENC
else:
    vmap, vcodec = "0:v", ["-c:v", "copy"]
ff("-i", video, "-i", WORK / "narration_full.wav", "-i", WORK / "music.wav", "-i", WORK / "sfx_track.wav", *extra_in,
   "-filter_complex", fc, "-map", vmap, "-map", "[a]", *vcodec, "-c:a", "aac", "-b:a", "192k",
   "-movflags", "+faststart", "-shortest", final)
shutil.copy(final, PROJ / "render" / "final.mp4")
print("✅ Saved:", PROJ / "render" / "final.mp4")

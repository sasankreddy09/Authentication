# ==== 8b. Editing: each shot is cut to its length, slowed if needed, colour graded and titled ====
# A finished shot is cached on Drive (segments/), keyed by everything that shapes it, so re-runs only re-edit
# shots that changed. While the GPU makes clips, finished shots are edited in the background on the CPU.
from PIL import ImageFont
from concurrent.futures import ThreadPoolExecutor
GRADES = {"neutral":  "eq=contrast=1.0",
          "cool":     "colorbalance=bs=0.06:bm=0.03:bh=0.02",
          "contrast": "eq=contrast=1.08:saturation=1.05",
          "hot":      "eq=contrast=1.12:saturation=1.15,colorbalance=rh=0.05:rm=0.04",
          "deep":     "eq=contrast=1.1:brightness=-0.03:saturation=0.9"}

def zoompan(move, n, amt):
    N = max(n - 1, 1)
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    z, x, y = {"push_in":   (f"1+{amt}*on/{N}", cx, cy),
               "pull_out":  (f"{1 + amt}-{amt}*on/{N}", cx, cy),
               "pan_right": (f"{1 + amt}", f"(iw-iw/zoom)*on/{N}", cy),
               "pan_left":  (f"{1 + amt}", f"(iw-iw/zoom)*(1-on/{N})", cy),
               "tilt_up":   (f"{1 + amt}", cx, f"(ih-ih/zoom)*(1-on/{N})"),
               "tilt_down": (f"{1 + amt}", cx, f"(ih-ih/zoom)*on/{N}")}.get(move, ("1.0", "0", "0"))
    return f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={W}x{H}:fps={FPS}"

def look(shot, dur, d=0.6):
    f = [GRADES[shot.get("grade", mood(shot)["grade"])]]
    if shot.get("flash_in"):  f.append("fade=t=in:st=0:d=0.35:color=white")
    elif shot.get("fade_in"): f.append(f"fade=t=in:st=0:d={d}")
    if shot.get("fade_out"):  f.append(f"fade=t=out:st={max(dur - d, 0):.3f}:d={d}")
    return f

def font(size):
    for p in [drawkit.font_file("DejaVuSans-Bold.ttf"), "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if p and os.path.exists(p):
            return ImageFont.truetype(p, size)
    log("no TrueType font found: titles use Pillow's default font", "WARN")
    return ImageFont.load_default(size=size)

def overlay_png(shot, kind):
    """Transparent PNG laid over the moving video: "title" (big, centred) or "label" (step badge, top left).
    Text is drawn here, never by the AI model, so it is always sharp and spelled right."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if kind == "title":
        f, words, lines = font(84), shot["title"].split(), [""]
        for wd in words:                              # wrap so a long title never runs off the screen
            trial = (lines[-1] + " " + wd).strip()
            if lines[-1] and d.textlength(trial, font=f) > W - 240:
                lines.append(wd)
            else:
                lines[-1] = trial
        text = "\n".join(lines)
        box = d.multiline_textbbox((0, 0), text, font=f, spacing=18, align="center")
        xy = ((W - (box[2] - box[0])) // 2 - box[0], (H - (box[3] - box[1])) // 2 - box[1])
        d.multiline_text(xy, text, font=f, fill=(245, 245, 250, 255), spacing=18, align="center",
                         stroke_width=3, stroke_fill=(0, 0, 0, 230))
    elif kind == "credit":                            # licensed image credit (title, author, source, licence), top right
        f = font(max(18, H // 50))
        text = shot["stock"]["credit"]
        while d.textlength(text, font=f) > W * 0.6 and len(text) > 20:
            text = text[:-4].rstrip() + "…"
        tw = d.textlength(text, font=f)
        x, y, pad = W - tw - 40, 28, 10
        d.rounded_rectangle([x - pad, y - pad // 2, x + tw + pad, y + f.size + pad], radius=8, fill=(0, 0, 0, 140))
        d.text((x, y), text, font=f, fill=(235, 235, 240, 255))
    else:
        f, x, y, pad = font(44), 60, 56, 22
        box = d.textbbox((0, 0), shot["label"], font=f)
        tw, th = box[2] - box[0], box[3] - box[1]
        d.rounded_rectangle([x, y, x + tw + 2 * pad + 16, y + th + 2 * pad], radius=18, fill=(14, 14, 20, 200))
        d.rounded_rectangle([x, y, x + 10, y + th + 2 * pad], radius=5, fill=(232, 72, 85, 255))
        d.text((x + 16 + pad - box[0], y + pad - box[1]), shot["label"], font=f, fill=(255, 255, 255, 255))
    out = WORK / f"{shot['shot_id']}_{kind}.png"; im.save(out); return out

def still_frame(shot):
    kf = PROJ / "keyframes" / f"{shot['shot_id']}.png"
    return kf if kf.exists() else None

SEG_DIR = PROJ / "segments"; SEG_DIR.mkdir(exist_ok=True)
if not BURN_SUBS:                    # segments are joined without re-encoding: they need the final quality
    SEG_VENC = VENC
elif NVENC_OK:                       # segments are re-encoded once more with the subtitles: fast, near-lossless
    SEG_VENC = ["-c:v", "h264_nvenc", "-preset", "p2", "-rc", "vbr", "-cq", "16", "-b:v", "0", "-pix_fmt", "yuv420p", "-r", FPS]
else:
    SEG_VENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-r", FPS]

HOLDS = {}                           # shot -> seconds its last frame is frozen (should be ~0)
REVIEW = {}                          # shot -> why a person should look at it before publishing
def segment_recipe(shot, row):
    """ffmpeg inputs + filter graph for one shot, and what it depends on (for the cache key)."""
    sid, n, dur = shot["shot_id"], row["frames"], row["dur"]
    info = clip_info(shot) if shot.get("motion_prompt") else None
    inputs, k = [], 0
    fb = None
    if shot.get("motion_prompt") and not info and "logic_shot" in globals() and logic_shot(shot):
        # scene-state shot without an accepted/approved clip: never used silently; fallback + listed for review
        REVIEW[sid] = logic_usable(shot)[1]
        if shot.get("fallback"):
            info = {"use_fallback": True}
    if info and info.get("use_fallback") and shot.get("fallback"):
        # the AI clip failed the continuity / relevance check even after a remake: use the plan's drawn fallback
        fb = {"shot_id": f"{sid}_fallback", **shot["fallback"]}
        DIAGRAM_FILES[sid] = make_diagram(fb, row)[0]
        REVIEW.setdefault(sid, "AI clip didn't show the right scene/characters; drawn fallback used")
    if shot.get("recipe") == "stock":                     # licensed real image (Commons), fitted + slow camera move
        src = PROJ / "assets" / shot["stock"]["file"]
        inputs = ["-i", src]
        amt = shot.get("move_amount", mood(shot)["move_amount"])
        # fitted inside the frame over a blurred copy of itself: nothing of the picture is cropped away
        fc = ("[0:v]format=rgb24,split[bg][fg];"
              f"[bg]scale={2 * W}:{2 * H}:force_original_aspect_ratio=increase,crop={2 * W}:{2 * H},boxblur=30:3,"
              "eq=brightness=-0.12[b];"
              f"[fg]scale={2 * W}:{2 * H}:force_original_aspect_ratio=decrease[f];"
              "[b][f]overlay=(W-w)/2:(H-h)/2," + ",".join([zoompan(shot.get("move", "push_in"), n, amt),
                                                          *look(shot, dur), "format=yuv420p"]) + "[v]")
        k, how, info = 1, f"licensed image ({shot['stock'].get('license', '?')}) + camera move", None
    elif shot.get("recipe") in DRAWN or fb:               # drawn scene (no AI), exact length
        dg = DIAGRAM_FILES[sid]
        inputs = ["-i", dg]
        fc = "[0:v]" + ",".join([f"scale={W}:{H}:flags=lanczos", "tpad=stop_mode=clone:stop_duration=30",
                                 *look(shot, dur)[1:], "format=yuv420p"]) + "[v]"
        k, how = 1, (f"{fb['recipe']} fallback (AI clip rejected, review)" if fb else f"{shot['recipe']} (drawn, no AI)")
        info = None
    elif info:                                            # real AI video
        parts = [PROJ / "clips" / p for p in info["parts"]]
        for p in parts:
            inputs += ["-i", p]
        src_len = len(parts) * info["frames_per_part"] / info["fps"]
        stretch = min(max(dur / src_len, 1.0), MAX_SLOWMO)
        HOLDS[sid] = round(max(0.0, dur - src_len * stretch), 2)   # seconds the last frame is held (a freeze)
        chain = []
        if stretch > 1.01:
            chain.append(f"setpts={stretch:.4f}*PTS")
        if (stretch > 1.01 or info["fps"] != FPS) and SMOOTH_MOTION == "mci":
            chain.append(f"minterpolate=fps={FPS}:mi_mode=mci:mc_mode=aobmc:vsbmc=1")   # can warp shapes
        elif (stretch > 1.01 or info["fps"] != FPS) and SMOOTH_MOTION == "blend":
            chain.append(f"minterpolate=fps={FPS}:mi_mode=blend")                    # soft cross-fade, no warping
        else:
            chain.append(f"fps={FPS}")
        chain += [f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos", f"crop={W}:{H}"]
        if SHARPEN > 0:                                   # the AI clips are small (832x480): sharpen after upscaling
            chain.append(f"unsharp=5:5:{SHARPEN}:5:5:0")
        chain += ["tpad=stop_mode=clone:stop_duration=30", *look(shot, dur), "format=yuv420p"]
        head = "".join(f"[{i}:v]" for i in range(len(parts)))
        fc = (f"{head}concat=n={len(parts)}:v=1:a=0," if len(parts) > 1 else "[0:v]") + ",".join(chain) + "[v]"
        k = len(parts)
        how = f"AI video ({info['engine']}, {len(parts)} clip{'s' if len(parts) > 1 else ''}" + \
              (f", slowed {stretch:.2f}x" if stretch > 1.01 else "") + ")"
    else:                                                 # fallback: camera move over the still
        src = still_frame(shot)
        if src is None:
            src = WORK / f"{sid}_blank.png"; Image.new("RGB", (W, H), (5, 5, 12)).save(src)
        inputs = ["-i", src]
        amt = shot.get("move_amount", mood(shot)["move_amount"])
        fc = ("[0:v]" + ",".join([f"scale={2 * W}:{2 * H}:force_original_aspect_ratio=increase", f"crop={2 * W}:{2 * H}",
                                  zoompan(shot.get("move", "push_in"), n, amt), *look(shot, dur), "format=yuv420p"]) + "[v]")
        k = 1
        how = f"⚠️ still image + camera move (no AI video for this shot)"
    vout = "[v]"
    for kind in ("label", "title", "credit"):
        if not (shot.get(kind) or (kind == "credit" and shot.get("recipe") == "stock")):
            continue
        inputs += ["-loop", "1", "-i", overlay_png(shot, kind)]
        fades = (f"fade=t=in:st=0.6:d=0.8:alpha=1,fade=t=out:st={max(dur - 1.2, 1.5):.3f}:d=0.7:alpha=1"
                 if kind == "title" else f"fade=t=in:st=0.15:d=0.4:alpha=1,fade=t=out:st={max(dur - 0.35, 0.6):.3f}:d=0.3:alpha=1")
        fc += f";[{k}:v]format=rgba,{fades}[o{k}];{vout}[o{k}]overlay=0:0:shortest=0[v{k}]"
        vout = f"[v{k}]"; k += 1
    if shot.get("recipe") == "stock":
        deps = shot["stock"]["file"] + "|" + shot["stock"]["credit"]   # file name = content hash of the image
    elif shot.get("recipe") in DRAWN or fb:
        deps = DIAGRAM_FILES[sid].name                    # the file name carries the diagram's content hash
    else:
        dep = (PROJ / "clips" / f"{sid}.json") if info else (PROJ / "keyframes" / f"{sid}.png.json")
        deps = dep.read_text() if dep.exists() else "none"
    return inputs, fc, vout, how, bool(info), deps

SEG_RECIPE_VERSION = 3               # bump when the editing code changes, to re-edit every shot
def render_segment(shot, row, background=False):
    """Edit one shot (or reuse the cached result). Returns (file, how it was made, is AI video, was cached)."""
    sid = shot["shot_id"]
    inputs, fc, vout, how, is_video, deps = segment_recipe(shot, row)
    key = hashlib.sha1(json.dumps([SEG_RECIPE_VERSION, [str(x) for x in inputs], fc, row["frames"],
                                   [str(x) for x in SEG_VENC], shot.get("title"), shot.get("label"), deps]).encode()).hexdigest()[:12]
    out = SEG_DIR / f"{sid}_{key}.mp4"
    if out.exists():
        return out, how, is_video, True
    tmp = WORK / f"seg_{sid}.mp4"
    ff(*inputs, "-filter_complex", fc, "-map", vout, "-frames:v", row["frames"], *SEG_VENC, "-an", tmp, nice=background)
    shutil.copy(tmp, out)
    for old in SEG_DIR.glob(f"{sid}_*.mp4"):
        if old != out:
            old.unlink()
    log(f"{sid}: edited ({'background' if background else 'foreground'}) -> {out.name}")
    return out, how, is_video, False

EDITOR, EDIT_JOBS = ThreadPoolExecutor(max_workers=1), []
TL_ROW = {r["shot_id"]: r for r in TIMELINE}
def edit_in_background(shot):
    def job():
        try:
            return render_segment(shot, TL_ROW[shot["shot_id"]], background=True)
        except Exception:
            log_exc(f"{shot['shot_id']}: background edit failed (it is retried in the edit step)")
    EDIT_JOBS.append(EDITOR.submit(job))

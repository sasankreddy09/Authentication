# ==== 7b. Diagram shots: system-design animations drawn here (no AI, no GPU) ====
# Boxes for services / databases / queues, arrows, requests travelling between them, database rows changing state
# and code notes. Text is exact and sharp, and a shot takes seconds. Layout: 4 columns x 4 rows of boxes on the
# left, a panel (table / code) on the right. Frames are streamed straight into ffmpeg: no video is held in RAM.
# Finished diagrams are cached on Drive and rendered in the background while the GPU makes the AI clips.
import threading
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont
DIAGRAM_VERSION = 1
DW, DH = 1920, 1080
KIND_COLORS = {"client": (59, 130, 246), "service": (139, 92, 246), "store": (16, 185, 129),
               "queue": (245, 158, 11), "cache": (239, 68, 68), "external": (120, 128, 145)}
BG, BOX_W, BOX_H = (13, 17, 33), 250, 86
PACKET, PULSE, TRAVEL = (255, 214, 102), (255, 214, 102), 0.9

def dfont(size, mono=False, bold=True):
    names = (["DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf"] if mono else ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"])
    for n in (names if bold else names[::-1]):
        p = drawkit.font_file(n)                     # system DejaVu, else matplotlib's copy (always on Colab)
        if p:
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)
F_NODE, F_SUB, F_PKT, F_TAB, F_TABH, F_NOTE = dfont(26), dfont(19, bold=False), dfont(22, mono=True), \
    dfont(23, mono=True, bold=False), dfont(23, mono=True), dfont(22, mono=True, bold=False)

def mix(c, k):                                       # blend a colour toward the background (k=0 colour, 1 = bg)
    return tuple(int(a * (1 - k) + b * k) for a, b in zip(c, BG))

_BG_IMG = None
def background():
    global _BG_IMG
    if _BG_IMG is None:
        y = np.linspace(0, 1, DH)[:, None, None]
        top, bot = np.array([18, 24, 48]), np.array([8, 10, 22])
        _BG_IMG = Image.fromarray(np.broadcast_to(top * (1 - y) + bot * y, (DH, DW, 3)).astype(np.uint8))
    return _BG_IMG

def border_point(a, b, pad=8):
    """Where the line from box centre a toward b leaves box a."""
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return a
    t = min((BOX_W / 2 + pad) / abs(dx) if dx else 1e9, (BOX_H / 2 + pad) / abs(dy) if dy else 1e9)
    return ax + dx * t, ay + dy * t

def arrow(d, p, q, color, width):
    d.line([p, q], fill=color, width=width)
    ang = math.atan2(q[1] - p[1], q[0] - p[0])
    h = 10 + width * 2
    d.polygon([q, (q[0] - h * math.cos(ang - 0.4), q[1] - h * math.sin(ang - 0.4)),
               (q[0] - h * math.cos(ang + 0.4), q[1] - h * math.sin(ang + 0.4))], fill=color)

def draw_node(d, spec, state):
    x, y = spec["x"], spec["y"]
    col = KIND_COLORS[spec["kind"]]
    k = {"ghost": 0.88, "dim": 0.62, "normal": 0.0, "focus": 0.0}[state]
    body, edge = mix(mix(col, 0.72), k), mix(col, k)
    box = [x - BOX_W // 2, y - BOX_H // 2, x + BOX_W // 2, y + BOX_H // 2]
    d.rounded_rectangle(box, radius=16, fill=body, outline=edge, width=6 if state == "focus" else 3)
    if spec["kind"] in ("store", "cache"):               # a little database "cylinder" mark
        d.ellipse([box[0] + 12, box[1] + 10, box[0] + 36, box[1] + 20], outline=edge, width=2)
    txt = mix((255, 255, 255), k)
    sub = spec.get("sub")
    d.text((x, y - (11 if sub else 0)), spec["label"], font=F_NODE, fill=txt, anchor="mm")
    if sub:
        d.text((x, y + 22), sub, font=F_SUB, fill=mix((200, 210, 230), k), anchor="mm")

def table_img(tab, values, hot):
    """The right-hand table panel: title, header, rows; `hot` = cells changed in the last moment."""
    w, rh = 620, 46
    cols = tab["columns"]
    h = 58 + rh * (1 + len(values))
    im = Image.new("RGB", (w, h), (22, 28, 52)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=14, outline=(16, 185, 129), width=3)
    d.text((18, 30), "TABLE  " + tab["name"], font=F_TABH, fill=(110, 231, 183), anchor="lm")
    cw = (w - 24) // len(cols)
    for j, c in enumerate(cols):
        d.text((14 + j * cw, 58 + rh // 2), c, font=F_TABH, fill=(180, 190, 215), anchor="lm")
    d.line([12, 58 + rh, w - 12, 58 + rh], fill=(60, 70, 100), width=2)
    for i, row in enumerate(values):
        yy = 58 + rh * (i + 1)
        for j, v in enumerate(row):
            if (i, j) in hot:
                d.rounded_rectangle([10 + j * cw, yy + 5, 6 + (j + 1) * cw, yy + rh - 5], radius=8, fill=(120, 95, 20))
            d.text((14 + j * cw, yy + rh // 2), str(v), font=F_TAB,
                   fill=(255, 230, 150) if (i, j) in hot else (230, 235, 245), anchor="lm")
    return im

def note_img(lines):
    w, lh = 620, 34
    h = 52 + lh * len(lines)
    im = Image.new("RGB", (w, h), (16, 20, 36)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=14, outline=(139, 92, 246), width=3)
    d.text((18, 26), "CODE", font=F_TABH, fill=(196, 181, 253), anchor="lm")
    for i, l in enumerate(lines):
        d.text((18, 52 + lh * i + lh // 2), l, font=F_NOTE, fill=(230, 235, 245), anchor="lm")
    return im

def diagram_timing(dg, dur):
    """Start time of each flow, table change and node reveal (spread over the shot unless given)."""
    flows, changes = dg.get("flows", []), (dg.get("table") or {}).get("changes", [])
    span0, span1 = 0.4, max(dur - 1.2, 0.8)
    ft = [f.get("at", span0 + (span1 - span0) * i / max(len(flows), 1)) for i, f in enumerate(flows)]
    ct = [c.get("at", span0 + 0.6 + (span1 - span0) * (i + 0.5) / max(len(changes), 1)) for i, c in enumerate(changes)]
    return ft, ct

def render_diagram(shot, frames, out):
    arch, dg = PLAN["architecture"], shot["diagram"]
    nodes, dur = arch["nodes"], frames / FPS
    show = set(nodes) if dg.get("show", "all") == "all" else set(dg["show"])
    focus = set(dg.get("focus", []))
    flows = dg.get("flows", [])
    ft, ct = diagram_timing(dg, dur)
    reveal = [n for n in nodes if n in show] if dg.get("reveal") else []
    rt = {n: 0.2 + i * min(0.5, dur * 0.6 / max(len(reveal), 1)) for i, n in enumerate(reveal)}
    tab = dg.get("table")
    note = note_img(dg["note"]) if dg.get("note") else None
    base_cache = {}

    def base(t):
        shown = {n for n in show if n not in rt or t >= rt[n]}
        tstate = sum(t >= c for c in ct)
        hot = frozenset((c["row"], tab["columns"].index(c["col"])) for c, c0 in zip(tab["changes"], ct)
                        if c0 <= t < c0 + 1.2) if tab else frozenset()
        key = (frozenset(shown), tstate, hot)
        if key not in base_cache:
            im = background().copy(); d = ImageDraw.Draw(im)
            for a, b in arch["edges"]:
                if a in shown and b in shown:
                    p, q = border_point((nodes[a]["x"], nodes[a]["y"]), (nodes[b]["x"], nodes[b]["y"])), \
                           border_point((nodes[b]["x"], nodes[b]["y"]), (nodes[a]["x"], nodes[a]["y"]))
                    lit = not focus or (a in focus and b in focus)
                    arrow(d, p, q, (95, 110, 150) if lit else (45, 52, 78), 3)
            for n, spec in nodes.items():
                st = "ghost" if n not in shown else ("focus" if n in focus else ("dim" if focus else "normal"))
                draw_node(d, spec, st)
            y0 = 180
            if tab:
                vals = [list(r) for r in tab["rows"]]
                for c in tab.get("changes", [])[:tstate]:
                    vals[c["row"]][tab["columns"].index(c["col"])] = c["value"]
                ti = table_img(tab, vals, hot); im.paste(ti, (1270, y0)); y0 += ti.height + 30
            if note is not None:
                im.paste(note, (1270, y0))
            if len(base_cache) > 64:
                base_cache.clear()
            base_cache[key] = im
        return base_cache[key]

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{DW}x{DH}", "-r", str(FPS), "-i", "-", *X264_DRAWN, str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(frames):
            t = i / FPS
            im = base(t).copy(); d = ImageDraw.Draw(im)
            for f, t0 in zip(flows, ft):
                a, b = nodes[f["from"]], nodes[f["to"]]
                p, q = border_point((a["x"], a["y"]), (b["x"], b["y"])), border_point((b["x"], b["y"]), (a["x"], a["y"]))
                if t0 <= t < t0 + TRAVEL:                  # the request travelling along the arrow
                    u = (t - t0) / TRAVEL; u = u * u * (3 - 2 * u)
                    arrow(d, p, q, (255, 214, 102), 4)
                    cx, cy = p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u
                    d.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=PACKET, outline=(255, 255, 255), width=2)
                    if f.get("label"):
                        bb = d.textbbox((cx, cy - 34), f["label"], font=F_PKT, anchor="mm")
                        d.rounded_rectangle([bb[0] - 10, bb[1] - 6, bb[2] + 10, bb[3] + 6], radius=10,
                                            fill=(30, 34, 58), outline=PACKET, width=2)
                        d.text((cx, cy - 34), f["label"], font=F_PKT, fill=(255, 236, 170), anchor="mm")
                elif t0 + TRAVEL <= t < t0 + TRAVEL + 0.5:  # the receiving box lights up
                    x, y = b["x"], b["y"]
                    d.rounded_rectangle([x - BOX_W // 2 - 6, y - BOX_H // 2 - 6, x + BOX_W // 2 + 6, y + BOX_H // 2 + 6],
                                        radius=20, outline=PULSE, width=5)
            proc.stdin.write(im.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait():
            raise RuntimeError("diagram encoding failed: " + err[-1500:])
    finally:
        if proc.poll() is None:
            proc.kill()

# ---- every drawn scene: diagram (above), and app screens / journey / cards from engine/lib/drawkit.py ----
DRAWN = globals().get("DRAWN_RECIPES", {"diagram"})
DRAWKIT_VERSION = 3                                   # bump when drawkit's look changes, to redraw every scene
CHARANIM_VERSION = 4                                  # bump when the friends' drawings / demos change

# ---- storybook plans: the approved (or provisional) character and environment asset versions, once per session ----
STORYBOOK = PLAN.get("visual_style_version") == "storybook_v1"
STORYBOOK_VERSION = 1                                 # bump when the storybook renderer's drawing changes
STORYBOOK_REPORT = {}
if STORYBOOK:
    import storybook as SB, envfactory
    os.environ.setdefault("STORYBOOK_ASSETS", str(Path(DRIVE_ROOT) / "_assets"))   # the library lives on Drive
    _t0 = time.time()
    for _cid, _a in PLAN["character_assets"].items():
        _vdir = SB.library_root() / "characters" / _cid / _a["version"]
        if _a.get("status") == "approved":
            if not (_vdir / "manifest.json").exists():
                raise RuntimeError(f"approved character asset {_cid} {_a['version']} is not on Drive ({_vdir}): copy it "
                                   "from the Character Studio output; the video never falls back to simple shapes")
            SB.load(_cid, _a["version"], expect_sha=_a.get("manifest_sha256"), expect_content=_a.get("content_sha256"))
        else:                                         # provisional, code-drawn: prepared in the plan cell, cached on Drive
            SB.ensure(_cid, _a["version"], log=log)
        _m = json.loads((_vdir / "manifest.json").read_text())
        STORYBOOK_REPORT[_cid] = {"version": _a["version"], "status": _m["status"], "art_version": _m["art_version"],
                                  "verify": SB.verify(_vdir), "layers": len(_m["layers"])}
    for _e, _ver in PLAN["environments"].items():
        envfactory.ensure(_e, _ver, SB.library_root(), log=log)
        STORYBOOK_REPORT[f"env:{_e}"] = {"version": _ver, "verify": envfactory.verify(SB.library_root() / "environments" / _e / _ver)}
    log(f"storybook assets ready in {time.time() - _t0:.1f}s: {json.dumps(STORYBOOK_REPORT)[:1500]}")
    print("Storybook characters: " + ", ".join(f"{k} {v['version']} ({v.get('status', 'pack')})" for k, v in STORYBOOK_REPORT.items()))
    DEVICE_REPORT["storybook characters + environments"] = "CPU (Pillow), assets cached on Drive"
STORYBOOK_MOUTH = {}                                  # shot -> mouth check (listener mouths, speech coverage)

# ---- two-friend scenes: the scene's VISUAL timeline, cut inside the dialogue (the audio is already placed) ----
STORY_SPECS, STORY_ROWS = {}, {}                      # shot -> drawable spec / [visual, intent, medium, start, dur]
_TL = {r["shot_id"]: r for r in TIMELINE}

def speech_energy(shot, frames):
    """Per character, per video frame: how loud that character is speaking (0..1). Each line is normalised by its
    own loud parts, so a quiet voice still opens the mouth; silence (pauses, the other speaker) is 0."""
    t0, out = _TL[shot["shot_id"]]["start"], {}
    placed = {p["id"]: p for p in PLACED}
    for l in shot["lines"]:
        p = placed.get(l["id"])
        if p is None:
            continue
        w, sr_ = sf.read(str(PROJ / "narration" / f"{l['id']}.wav"), dtype="float32")
        w = w.mean(1) if w.ndim > 1 else w
        n = int(len(w) / (sr_ / FPS))
        rms = np.array([np.sqrt(np.mean(w[int(i * sr_ / FPS):int((i + 1) * sr_ / FPS)] ** 2) + 1e-12) for i in range(n)])
        ref = np.percentile(rms, 90) if n else 1.0
        e = np.clip((rms / max(ref, 1e-6) - 0.08) / 0.92, 0, 1)
        e = np.maximum(e, np.r_[0, e[:-1]] * 0.6)        # a little hold: no flicker between syllables
        arr = out.setdefault(l["speaker"], np.zeros(frames, np.float32))
        a = int(round((p["start"] - t0) * FPS))
        k = max(0, min(len(e), frames - a))
        if k > 0 and a >= 0:
            arr[a:a + k] = np.maximum(arr[a:a + k], e[:k])
    return out

def speech_visemes(shot, frames):
    """Per character, per video frame: the mouth shape from that speaker's own audio (storybook plans). Silence and
    the other speaker's lines are "rest" — a listener's mouth never moves with someone else's voice."""
    t0, out = _TL[shot["shot_id"]]["start"], {}
    placed = {p["id"]: p for p in PLACED}
    for l in shot["lines"]:
        p = placed.get(l["id"])
        if p is None:
            continue
        w, sr_ = sf.read(str(PROJ / "narration" / f"{l['id']}.wav"), dtype="float32")
        vv, _ = SB.visemes_from_audio(w, sr_, FPS)
        arr = out.setdefault(l["speaker"], ["rest"] * frames)
        a = int(round((p["start"] - t0) * FPS))
        for i, v in enumerate(vv):
            if 0 <= a + i < frames and v != "rest":
                arr[a + i] = v
    return out

def story_spec(shot, frames):
    sid = shot["shot_id"]
    if sid in STORY_SPECS and STORY_SPECS[sid][0] == frames:
        return STORY_SPECS[sid][1]
    t0, placed = _TL[sid]["start"], {p["id"]: p for p in PLACED}
    vis = shot["story"]["visuals"]
    starts = []
    for v in vis:                                      # a visual starts at a point INSIDE a line
        p = placed[shot["lines"][v["start"]["line"]]["id"]]
        starts.append((p["start"] - t0) + v["start"]["at"] * (p["end"] - p["start"]))
    starts[0] = 0.0
    fr = [min(max(int(round(x * FPS)), 0), frames) for x in starts] + [frames]
    for i in range(1, len(fr)):
        fr[i] = max(fr[i], fr[i - 1])
    energy = speech_energy(shot, frames) if globals().get("LIP_SYNC", "energy") == "energy" else {}
    vis_all = speech_visemes(shot, frames) if STORYBOOK and energy else {}
    for c, vv in vis_all.items():                       # one silence gate for both tracks: the loudness decides
        e = energy.get(c, np.zeros(frames))
        vis_all[c] = [v if e[i] > 0.06 else "rest" for i, v in enumerate(vv)]
    visuals, rows = [], []
    for v, a, b in zip(vis, fr, fr[1:]):
        if b <= a:
            log(f"{sid}: visual {v['id']} gets no time (starts too close to the next one); skipped", "WARN")
            continue
        spec = dict(v[v["recipe"]])
        if v["recipe"] == "character":
            spec.update(_t0=round(t0 + a / FPS, 3), _dur=round((b - a) / FPS, 3),
                        _speech={c: [round(float(x), 2) for x in e[a:b]] for c, e in energy.items()})
            if vis_all:
                spec["_visemes"] = {c: list(vv[a:b]) for c, vv in vis_all.items()}
        visuals.append({"id": v["id"], "recipe": v["recipe"], v["recipe"]: spec, "frames": b - a})
        rows.append({"shot": sid, "visual": v["id"], "intent": v.get("intent", "?"), "medium": v.get("medium", "?"),
                     "recipe": v["recipe"], "start": round(t0 + a / FPS, 2), "dur": round((b - a) / FPS, 2),
                     "label": v.get("label", ""), "description": v.get("description", "")})
    if vis_all:                                        # mouth check: only the speaker's mouth moves
        rep_ = {}
        for c, vv in vis_all.items():
            e = energy.get(c, np.zeros(frames))
            talk = [i for i in range(frames) if e[i] > 0.06]
            rep_[c] = {"speech_frames": len(talk), "open_in_speech": sum(vv[i] not in ("rest", "closed") for i in talk),
                       "open_in_silence": sum(1 for i in range(frames) if e[i] <= 0.06 and vv[i] not in ("rest",))}
        STORYBOOK_MOUTH[sid] = rep_
    STORY_SPECS[sid], STORY_ROWS[sid] = (frames, {"visuals": visuals}), rows
    return {"visuals": visuals}

# drawn frames are encoded on the GPU (NVENC) when the session has one: the CPU stays free for drawing
X264_DRAWN = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-tune", "animation", "-pix_fmt", "yuv420p"]
NVENC_DRAWN = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "16", "-b:v", "0", "-pix_fmt", "yuv420p"]
DRAWN_VENC = NVENC_DRAWN if globals().get("NVENC_OK") else X264_DRAWN
log(f"drawn scenes are encoded with {'NVENC (GPU)' if DRAWN_VENC is NVENC_DRAWN else 'x264 (CPU)'}")
DEVICE_REPORT["drawn scenes: drawing (Pillow)"] = "CPU (Pillow has no GPU path)"
DEVICE_REPORT["drawn scenes: encoding"] = "GPU (NVENC)" if DRAWN_VENC is NVENC_DRAWN else "CPU (x264)"

def render_drawn(shot, frames, out, venc=None):
    kind = shot.get("recipe")
    if kind == "diagram":
        return render_diagram(shot, frames, out)
    venc = venc or DRAWN_VENC
    try:
        return _render_drawn(shot, frames, out, venc)
    except RuntimeError:
        if venc is X264_DRAWN:
            raise
        log_exc(f"{shot['shot_id']}: GPU encoder failed for a drawn scene, encoding it on the CPU")
        return _render_drawn(shot, frames, out, X264_DRAWN)

def _render_drawn(shot, frames, out, venc):
    kind = shot.get("recipe")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", *venc, str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        spec = story_spec(shot, frames) if kind == "story" else \
            dict(shot[kind], _has_label=bool(shot.get("label")))      # layout keeps clear of the step label
        for frame in drawkit.render(kind, spec, PLAN, frames, FPS, W, H):
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait():
            raise RuntimeError(f"{kind} encoding failed: " + err[-1500:])
    finally:
        if proc.poll() is None:
            proc.kill()

DIAGRAM_DIR = PROJ / "diagrams"; DIAGRAM_DIR.mkdir(exist_ok=True)
DIAGRAM_FILES = {}

def diagram_path(shot, frames):
    kind = shot.get("recipe")
    shared_bits = PLAN.get("architecture") if kind == "diagram" else [PLAN.get("journey"), PLAN.get("look")]
    if kind in ("story", "character", "demo"):         # the friends' look + this scene's cut points and mouths
        shared_bits = [shared_bits, PLAN.get("bible"), CHARANIM_VERSION]
        if STORYBOOK:                                  # + the exact asset versions and the renderer version
            shared_bits = shared_bits + [PLAN.get("character_assets"), PLAN.get("environments"), STORYBOOK_VERSION]
    body = story_spec(shot, frames) if kind == "story" else shot[kind]
    key = hashlib.sha1(json.dumps([DIAGRAM_VERSION, DRAWKIT_VERSION, kind, shared_bits, body, bool(shot.get("label")),
                                   frames, FPS, W, H],
                                  sort_keys=True).encode()).hexdigest()[:12]
    return DIAGRAM_DIR / f"{shot['shot_id']}_{key}.mp4"

def make_diagram(shot, row):
    out = diagram_path(shot, row["frames"])
    if out.exists():
        return out, True
    t0, tmp = time.time(), WORK / f"diagram_{shot['shot_id']}.mp4"
    render_drawn(shot, row["frames"], tmp)
    shutil.copy(tmp, out)
    for old in DIAGRAM_DIR.glob(f"{shot['shot_id']}_*.mp4"):
        if old != out:
            old.unlink()
    log(f"{shot['shot_id']}: {shot.get('recipe')} drawn in {time.time() - t0:.1f}s ({row['frames']} frames)")
    return out, False

DIAGRAM_SHOTS = [(s, r) for s, r in zip(SHOTS, TIMELINE) if s.get("recipe") in DRAWN]
def _diagram_job(shot, row):
    try:
        DIAGRAM_FILES[shot["shot_id"]] = make_diagram(shot, row)[0]
    except Exception:
        log_exc(f"{shot['shot_id']}: diagram failed (it is retried in the edit step)")

DIAGRAM_POOL, DIAGRAM_JOBS = ThreadPoolExecutor(max_workers=1), []
for s, r in DIAGRAM_SHOTS:
    DIAGRAM_JOBS.append(DIAGRAM_POOL.submit(_diagram_job, s, r))

def wait_for_diagrams():
    t0 = time.time()
    for j in DIAGRAM_JOBS:
        j.result()
    for s, r in DIAGRAM_SHOTS:                          # anything that failed in the background: try once more here
        if s["shot_id"] not in DIAGRAM_FILES:
            DIAGRAM_FILES[s["shot_id"]] = make_diagram(s, r)[0]
    if DIAGRAM_SHOTS:
        log(f"waited {time.time() - t0:.1f}s for {len(DIAGRAM_SHOTS)} diagram shots")
print(f"{len(DIAGRAM_SHOTS)} drawn scenes (app screens, journeys, cards, diagrams) are being drawn in the "
      "background (no GPU needed)." if DIAGRAM_SHOTS else "No drawn scenes in this plan.")

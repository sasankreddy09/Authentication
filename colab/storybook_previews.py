"""Visual evidence for the storybook upgrade (claude.md §22, tests A–D) — rendered on CPU, no GPU, no model.

Usage:  <python with numpy, pillow, soundfile, espeakng_loader> colab/storybook_previews.py <out dir> [--draft]
  A  old_vs_new.png           the legacy code-drawn friends beside the storybook characters (same scene spec)
     character_sheet_*.png    poses x expressions contact sheets (for the human approval step)
  B  B_animation.mp4          ~12 s: blinking, speaking, listening, pointing, thinking, a surprised reaction
  C  C_environment.mp4        ~8 s: the space-observatory pack, characters placed and lit, slow camera move
  D  D_educational.mp4        ~30 s: dialogue in the cinema → camera cut into the eye-views demonstration → back
Speech is robotic espeak-ng test audio (ESPK = short path to espeak-ng-data) — only to drive and check lip sync.
Everything is logged to <out dir>/previews.log with timings.
"""
import argparse, logging, subprocess, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine" / "lib")); sys.path.insert(0, str(HERE / "tests"))
import charanim, demos, storybook as SB, storybook_scene as SC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--draft", action="store_true")
ap.add_argument("--only", default="ABCD")
ap.add_argument("--atlas", metavar="VERSION", help="test F: a conversation with these character versions (e.g. v2), "
                "from STORYBOOK_ASSETS — the approval preview for a new atlas version")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
import tempfile
TMP = Path(tempfile.mkdtemp(prefix="sb_prev_"))          # test speech goes here, not next to the previews
logging.basicConfig(filename=str(OUT / "previews.log"), level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = lambda m: (print(m), logging.info(m))
W, H = (1280, 720) if a.draft else (1920, 1080)
FPS = 24
PLAN = {"language": "en", "visual_style_version": "storybook_v1",
        "character_assets": {c: {"version": "v1"} for c in ("arjun", "meera")},
        "environments": {e: "v1" for e in ("cinema", "lobby", "studio", "space_observatory")}}
LEGACY = {"language": "en"}


def speak(text, voice, pitch, out):
    import espeak_parler_worker as ew
    x = ew.say(text, voice, pitch=pitch)
    sf.write(out, x, ew.SR)
    return x, ew.SR


def track(lines, total_s):
    """lines: [(cid, start_s, wav, sr)] → per-character energy + viseme per frame, and the mixed audio."""
    n = int(total_s * FPS)
    energy = {c: np.zeros(n, np.float32) for c in ("arjun", "meera")}
    vis = {c: ["rest"] * n for c in ("arjun", "meera")}
    mix = np.zeros(int(total_s * 22050) + 1, np.float32)
    for cid, t0, wav, sr in lines:
        vv, e = SB.visemes_from_audio(wav, sr, FPS)
        a0 = int(t0 * FPS)
        for i, (v, ee) in enumerate(zip(vv, e)):
            if a0 + i < n:
                vis[cid][a0 + i] = v
                energy[cid][a0 + i] = min(float(ee), 1.0) if v != "rest" else 0.0
        s0 = int(t0 * 22050)
        mix[s0:s0 + len(wav)] += wav[:len(mix) - s0]
    return energy, vis, mix


def encode(frames_iter, n, wav, out):
    tmpa = TMP / (out.stem + ".wav")
    if wav is not None:
        sf.write(tmpa, wav, 22050)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"] + (["-i", str(tmpa)] if wav is not None else []) + \
          ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"] + \
          (["-c:a", "aac", "-b:a", "128k", "-shortest"] if wav is not None else []) + [str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for k, f in enumerate(frames_iter):
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed for {out}")
    if wav is not None:
        tmpa.unlink()
    log(f"{out.name}: {n} frames in {time.time() - t0:.1f}s ({(time.time() - t0) / max(n, 1):.3f} s/frame)")


def shot_frames(spec, n, t0_s, energy, vis):
    a0 = int(t0_s * FPS)
    sp = dict(spec, _n=n, _dur=n / FPS, _t0=t0_s,
              _speech={c: [round(float(x), 2) for x in e[a0:a0 + n]] for c, e in energy.items()},
              _visemes={c: v[a0:a0 + n] for c, v in vis.items()})
    for i in range(n):
        yield SC.character_frame(sp, PLAN, i / max(n - 1, 1), W, H)


# ---------------------------------------------------------------- A
def test_a():
    t0 = time.time()
    spec = {"world": "cinema_seats", "shot": "two_shot", "screen": "clean",
            "chars": {"arjun": {"wear": ["3d_glasses"], "expr": "surprised", "gesture": "point"},
                      "meera": {"wear": ["3d_glasses"], "expr": "smile", "gesture": "explain"}}}
    old = charanim.character_frame(dict(spec, _n=1), LEGACY, 0.5, 1920, 1080, None, None)
    new = SC.character_frame(dict(spec, _n=1), PLAN, 0.5, 1920, 1080)
    sheet = Image.new("RGB", (1920, 600), (250, 250, 250))
    sheet.paste(Image.fromarray(old).resize((960, 540)), (0, 60))
    sheet.paste(Image.fromarray(new).resize((960, 540)), (960, 60))
    d = ImageDraw.Draw(sheet)
    d.text((20, 20), "BEFORE: legacy code-drawn friends (charanim)", fill=(0, 0, 0))
    d.text((980, 20), "AFTER: storybook rig, provisional art v1 (not yet approved)", fill=(0, 0, 0))
    sheet.save(OUT / "A_old_vs_new.png")
    for cid in ("arjun", "meera"):
        ch = SB.load(cid, "v1")
        cells = []
        for pose in ("idle", "explain", "point", "think", "wave", "surprise", "celebrate", "duck", "hold", "open_hands"):
            im = Image.new("RGBA", (300, 540), (232, 236, 245, 255))
            expr = {"surprise": "surprised", "celebrate": "laughing", "think": "thinking", "duck": "surprised",
                    "wave": "happy"}.get(pose, "neutral")
            e, b, m, iris = SB.EXPRESSIONS[expr]
            SB.Painter(ch, 0.3).draw(im, (0, 0), SB.pose_of(pose), {"eyes": e, "brows": b, "mouth": m, "gaze": (0, 0),
                                                                    "turn": 0, "glasses": False, "iris": iris},
                                     hold="popcorn" if pose == "hold" else None)
            ImageDraw.Draw(im).text((8, 8), pose, fill=(30, 30, 40))
            cells.append(im)
        for expr in SB.EXPRESSIONS:
            big = Image.new("RGBA", (500, 900), (0, 0, 0, 0))
            e, b, m, iris = SB.EXPRESSIONS[expr]
            SB.Painter(ch, 0.5).draw(big, (0, 0), SB.pose_of("idle"), {"eyes": e, "brows": b, "mouth": m, "gaze": (0, 0),
                                                                       "turn": 0.25, "glasses": False, "iris": iris})
            im = Image.new("RGBA", (300, 540), (232, 236, 245, 255))
            im.alpha_composite(big.crop((100, 80, 400, 380)).resize((300, 300)), (0, 120))
            ImageDraw.Draw(im).text((8, 8), expr, fill=(30, 30, 40))
            cells.append(im)
        for v in SB.VISEMES:
            big = Image.new("RGBA", (500, 900), (0, 0, 0, 0))
            SB.Painter(ch, 0.5).draw(big, (0, 0), SB.pose_of("idle"), {"eyes": "open", "brows": "neutral", "mouth": v,
                                                                       "gaze": (0, 0), "turn": 0, "glasses": False, "iris": ""})
            im = Image.new("RGBA", (300, 540), (232, 236, 245, 255))
            im.alpha_composite(big.crop((175, 260, 325, 410)).resize((300, 300)), (0, 120))
            ImageDraw.Draw(im).text((8, 8), f"mouth: {v}", fill=(30, 30, 40))
            cells.append(im)
        cols = 10
        sheet = Image.new("RGBA", (300 * cols, 540 * ((len(cells) + cols - 1) // cols)), (255, 255, 255, 255))
        for i, c in enumerate(cells):
            sheet.paste(c, ((i % cols) * 300, (i // cols) * 540))
        sheet.convert("RGB").save(OUT / f"A_character_sheet_{cid}.png")
    log(f"test A: comparison + character sheets in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------- B
def test_b():
    lines, t = [], 0.6
    for cid, text, pitch in (("arjun", "Wait, the popcorn jumped right out of the screen!", 38),
                             ("meera", "It didn't. Close one eye and look again.", 72),
                             ("arjun", "Hmm. Now it looks flat.", 38)):
        x, sr = speak(text, "en", pitch, TMP / f"B_{cid}_{len(lines)}.wav")
        lines.append((cid, t, x, sr)); t += len(x) / sr + 0.55
    total = max(12.0, t + 0.8)
    energy, vis, mix = track(lines, total)
    l1, l2, l3 = (ln[1] / total for ln in lines)
    spec = {"world": "cinema_seats", "shot": "two_shot", "screen": "clean+popcorn",
            "chars": {"arjun": {"wear": ["3d_glasses"], "expr": "excited", "gesture": "point"},
                      "meera": {"wear": ["3d_glasses"], "expr": "listening", "gesture": "listen"}},
            "changes": [{"t": l2, "chars": {"arjun": {"expr": "listening", "gesture": "listen"},
                                            "meera": {"expr": "smile", "gesture": "explain"}}},
                        {"t": l2 + 0.12, "chars": {"arjun": {"gesture": "nod"}}},
                        {"t": l3, "chars": {"arjun": {"expr": "thinking", "gesture": "think"},
                                            "meera": {"expr": "listening", "gesture": "listen"}}},
                        {"t": min(l3 + 0.18, 0.95), "chars": {"arjun": {"expr": "surprised", "gesture": "surprise"}}}]}
    n = int(total * FPS)
    encode(shot_frames(spec, n, 0.0, energy, vis), n, mix, OUT / "B_animation.mp4")


# ---------------------------------------------------------------- C
def test_c():
    x, sr = speak("Look! That glowing ring is where light bends around the black hole.", "en", 72, TMP / "C_meera.wav")
    total = max(8.0, len(x) / sr + 1.5)
    energy, vis, mix = track([("meera", 0.7, x, sr)], total)
    spec = {"world": "space_observatory", "shot": "two_shot", "camera": {"movement": "slow_push_in"},
            "chars": {"arjun": {"expr": "curious", "gesture": "listen"}, "meera": {"expr": "excited", "gesture": "point"}},
            "changes": [{"t": 0.55, "chars": {"arjun": {"expr": "surprised", "gesture": "surprise"}}}]}
    n = int(total * FPS)
    encode(shot_frames(spec, n, 0.0, energy, vis), n, mix, OUT / "C_environment.mp4")


# ---------------------------------------------------------------- D
def test_d():
    import espeak_parler_worker as ew
    say = [("arjun", "అరే! ఇప్పుడు 3D effect తగ్గిపోయింది. Popcorn screen దగ్గరే ఆగిపోయింది.", 38),
           ("meera", "నీ రెండు కళ్లూ కొంచెం వేరే వేరే angles నుంచి చూస్తాయి.", 72),
           ("meera", "అందుకే ఒక్కో కంటికి slightly different view వస్తుంది.", 72),
           ("arjun", "అయితే ఒక కన్ను మూస్తే, ఒక view miss అయిపోతుంది కదా!", 38),
           ("meera", "Exactly. Brain ఆ రెండు viewsని కలిపి ఒకే బొమ్మ చేస్తుంది.", 72)]
    lines, t = [], 0.4
    for cid, text, pitch in say:
        x = ew.say(text, "te", pitch=pitch)
        lines.append((cid, t, x, ew.SR)); t += len(x) / ew.SR + 0.45
    total = max(30.0, t + 0.6)
    energy, vis, mix = track(lines, total)
    t1, t2, t3, t4 = lines[1][1], lines[3][1], lines[4][1], total
    shots = [  # (seconds, kind, spec)
        (t1, "char", {"world": "cinema_seats", "shot": "closeup", "subject": "arjun", "screen": "plain+popcorn", "push": True,
                      "chars": {"arjun": {"wear": ["3d_glasses"], "expr": "wink", "gesture": "point"},
                                "meera": {"wear": ["3d_glasses"]}}}),
        (t2 - t1, "demo", {"template": "eye_views"}),
        (t3 - t2, "char", {"world": "cinema_seats", "shot": "two_shot", "screen": "clean+popcorn",
                           "chars": {"arjun": {"wear": ["3d_glasses"], "expr": "thinking", "gesture": "think"},
                                     "meera": {"wear": ["3d_glasses"], "expr": "listening", "gesture": "listen"}}}),
        (t4 - t3, "char", {"world": "lobby", "shot": "two_shot", "camera": {"movement": "slow_push_in"},
                           "chars": {"arjun": {"expr": "happy", "gesture": "celebrate"},
                                     "meera": {"expr": "smile", "gesture": "explain"}}}),
    ]
    PLAN["language"] = "te"; PLAN["code_mix"] = "natural"

    def frames():
        start = 0.0
        prev_last = None
        for secs, kind, spec in shots:
            n = int(round(secs * FPS))
            if kind == "demo":
                for i in range(n):
                    f = np.asarray(demos.demo_frame(dict(spec, _n=n), PLAN, i / max(n - 1, 1), 1920, 1080, None, None))
                    f = np.asarray(Image.fromarray(f).resize((W, H), Image.LANCZOS))
                    if prev_last is not None and i < 8:   # a short dissolve: camera cut INTO the demonstration
                        f = (prev_last * (1 - i / 8) + f * (i / 8)).astype(np.uint8)
                    yield f
            else:
                last = None
                for f in shot_frames(spec, n, start, energy, vis):
                    last = f
                    yield f
                prev_last = last.astype(np.float32) if last is not None else None
            start += n / FPS
    n = sum(int(round(s * FPS)) for s, _, _ in shots)
    encode(frames(), n, mix, OUT / "D_educational.mp4")


def test_f():
    """F: the approval preview of a character version (claude.md 11.4): Meera explains while Arjun listens and nods,
    Arjun thinks, then a surprised reaction; both blink; mouths from each speaker's own audio; a pose transition."""
    ver = a.atlas
    PLAN["character_assets"] = {c: {"version": ver} for c in ("arjun", "meera")}
    lines, t = [], 0.5
    for cid, text, pitch in (("meera", "Close one eye and look at the screen again.", 72),
                             ("arjun", "Hmm, now it looks flat. Why?", 38),
                             ("meera", "Each eye sees a slightly different picture, and your brain combines them.", 72),
                             ("arjun", "Wait, really? That is amazing!", 38)):
        x, sr = speak(text, "en", pitch, TMP / f"F_{len(lines)}.wav")
        lines.append((cid, t, x, sr)); t += len(x) / sr + 0.5
    total = t + 0.6
    energy, vis, mix = track(lines, total)
    f = [ln[1] / total for ln in lines]
    spec = {"world": "lobby", "shot": "two_shot",
            "chars": {"arjun": {"expr": "listening", "gesture": "listen"}, "meera": {"expr": "smile", "gesture": "explain"}},
            "changes": [{"t": f[0] + 0.04, "chars": {"arjun": {"gesture": "nod"}}},
                        {"t": f[1], "chars": {"arjun": {"expr": "thinking", "gesture": "think"}, "meera": {"expr": "listening", "gesture": "listen"}}},
                        {"t": f[2], "chars": {"arjun": {"expr": "curious", "gesture": "listen"}, "meera": {"expr": "happy", "gesture": "open_hands"}}},
                        {"t": f[3], "chars": {"arjun": {"expr": "surprised", "gesture": "surprise"}, "meera": {"expr": "smile", "gesture": "listen"}}}]}
    n = int(total * FPS)
    encode(shot_frames(spec, n, 0.0, energy, vis), n, mix, OUT / f"F_conversation_{ver}.mp4")


for k, fn in (("A", test_a), ("B", test_b), ("C", test_c), ("D", test_d), ("F", test_f)):
    if k in a.only:
        t0 = time.time()
        try:
            fn()
        except Exception:
            logging.exception(f"test {k} failed")
            raise
        log(f"test {k} done in {time.time() - t0:.1f}s")

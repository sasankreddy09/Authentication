# ==== 10. Music: licensed library tracks per chapter (default), or a generated score as fallback ====
MSR = 48000
def synth(out, inputs, fc, dur):
    args = []
    for src in inputs:
        args += ["-f", "lavfi", "-i", src.format(d=f"{dur:.3f}")]
    ff(*args, "-filter_complex", fc, "-ar", MSR, "-ac", 2, "-t", f"{dur:.3f}", out)
    a, _ = sf.read(str(out), dtype="float32", always_2d=True)
    return a

def curious_layer(dur, bpm=92, seed=7):
    """Playful, inquisitive motif: plucked-string arpeggios (Karplus-Strong), a soft bass pluck, music-box bells."""
    rng = np.random.default_rng(seed)
    eighth = 60 / bpm / 2
    chords = [(38, [57, 62, 65, 69, 76]),      # Dm9
              (43, [62, 67, 70, 74, 76]),      # Gm6
              (34, [62, 65, 69, 72, 77]),      # Bbmaj7
              (33, [64, 67, 69, 73, 76])]      # A7
    patterns = [[0, 2, 1, 3, -1, 2, 4, 1], [3, 1, -1, 2, 0, 2, -1, 4],
                [0, 1, 2, 4, 3, -1, 2, 1], [4, 2, -1, 1, 3, 1, 0, -1]]
    cache = {}
    def pluck(midi, length=0.9, damp=3.0):
        if midi not in cache:
            f = 440 * 2 ** ((midi - 69) / 12); N = max(int(MSR / f), 2); n = int(length * MSR)
            buf, out, pos = rng.uniform(-1, 1, N).astype(np.float32), np.empty(n, np.float32), 0
            while pos < n:
                k = min(N, n - pos); out[pos:pos + k] = buf[:k]
                buf = 0.996 * 0.5 * (buf + np.roll(buf, -1)); pos += k
            cache[midi] = out * np.exp(-np.arange(n) / MSR * damp)
        return cache[midi]
    def bell(midi, length=1.8):
        f = 440 * 2 ** ((midi - 69) / 12); t = np.arange(int(length * MSR)) / MSR
        return ((np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
                 + 0.1 * np.sin(2 * np.pi * 3.01 * f * t)) * np.exp(-t * 2.5) * 0.35).astype(np.float32)
    out = np.zeros((int((dur + 3) * MSR), 2), np.float32)
    def place(sig, t, gain, pan):
        a = int(t * MSR); seg = out[a:a + len(sig)]
        seg[:, 0] += sig[:len(seg)] * gain * (1 - pan); seg[:, 1] += sig[:len(seg)] * gain * pan
    step = 0
    while step * eighth < dur:
        bar = step // 8
        root, tones = chords[bar % 4]
        idx = patterns[(bar // 4 + bar) % 4][step % 8]
        t = step * eighth + rng.uniform(-0.008, 0.008)                    # tiny human timing variation
        if step % 8 == 0:
            place(pluck(root + 12, 1.6, 1.5), t, 0.55, 0.5)                # bass
        if idx >= 0:
            place(pluck(tones[idx]), t, rng.uniform(0.22, 0.38), rng.uniform(0.25, 0.75))
        if step % 16 == 4 and bar % 2 == 1:
            place(bell(tones[-1] + 12), t, 0.5, rng.uniform(0.3, 0.7))   # music-box sparkle
        step += 1
    raw = WORK / "m_curious_raw.wav"
    sf.write(raw, out[: int(dur * MSR)], MSR)
    ff("-i", raw, "-af", "aecho=0.8:0.5:120|260:0.25|0.15", WORK / "m_curious.wav")
    a, _ = sf.read(str(WORK / "m_curious.wav"), dtype="float32", always_2d=True)
    return a

def rms_norm(a, target):
    return a * (target / max(float(np.sqrt((a ** 2).mean())), 1e-6))

def envelope(key, n_samples, smooth_s=1.5):
    """Per-shot target level for one layer, smoothed so changes swell instead of jumping."""
    rate = 100
    tgt = np.zeros(int(TOTAL * rate) + 1, np.float32)
    for shot, row in zip(SHOTS, TIMELINE):
        lv = {**mood(shot)["music"], **shot.get("music", {})}.get(key, 0.0)
        if key == "curious" and MUSIC_STYLE not in ("curious", "library"):
            lv = 0.0
        tgt[int(row["start"] * rate): int((row["start"] + row["dur"]) * rate) + 1] = lv
    k = int(smooth_s * rate)
    sm = np.convolve(np.pad(tgt, (k, k), mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    return np.interp(np.arange(n_samples) / MSR, np.arange(len(sm)) / rate, sm).astype(np.float32)

D = TOTAL + 1.0
MUSIC_SOURCE, MUSIC_CUES, MUSIC_CREDITS = "generated", [], ""

def lufs(path):
    """Integrated loudness of a file (EBU R128)."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.findall(r"I:\s+(-?\d+\.\d) LUFS", r.stderr)
    return float(m[-1]) if m else None

def fetch_track(title):
    """Licensed track from the library, downloaded once into the model cache (Drive) and reused."""
    import urllib.request
    f = MODEL_CACHE / "music" / f"{title}.mp3"
    f.parent.mkdir(parents=True, exist_ok=True)
    want = musiclib.TRACKS[title]["bytes"]
    if f.exists() and f.stat().st_size == want:
        return f
    t0 = time.time()
    req = urllib.request.Request(musiclib.url(title), headers={"User-Agent": "Mozilla/5.0 (ai-video notebook)"})
    with urllib.request.urlopen(req, timeout=60) as r, open(f.with_suffix(".part"), "wb") as out:
        shutil.copyfileobj(r, out)
    got = f.with_suffix(".part").stat().st_size
    if got < 100_000:
        raise RuntimeError(f"download of '{title}' looks broken ({got} bytes)")
    if got != want:
        log(f"'{title}': {got} bytes, catalogue says {want} (file may have been updated); using it", "WARN")
    f.with_suffix(".part").rename(f)
    log(f"music '{title}' downloaded in {time.time() - t0:.1f}s ({got / 2**20:.1f} MB)")
    return f

def library_music():
    """One licensed track per chapter, 2 s crossfades at chapter changes (never inside a scene)."""
    cues = musiclib.arrange(PLAN, TIMELINE, TOTAL)
    # story-derived restraint: under mostly serious / empathetic narration the music steps back 4 dB (it never
    # gets LOUDER because the voice is excited; the sidechain ducking keeps every word clear either way)
    dirs = globals().get("DIRECTIONS") or {}
    for c in cues:
        inside = [dirs[p["id"]].emotion for p in PLACED if p["id"] in dirs and c["start"] <= p["start"] < c["end"]]
        if inside and sum(e in ("serious", "empathetic") for e in inside) / len(inside) >= 0.5:
            c["gain_db"] = c.get("gain_db", 0.0) - 4.0
            log(f"music: '{c['track']}' {c['start']:.1f}–{c['end']:.1f}s 4 dB lower (serious narration)")
    XF, n = 2.0, int(D * MSR)
    bed = np.zeros((n, 2), np.float32)
    for i, c in enumerate(cues):
        a0, a1 = max(c["start"] - XF / 2, 0.0), min(c["end"] + XF / 2, D)
        seg = WORK / f"m_cue{i}.wav"
        ff("-stream_loop", "-1", "-ss", f"{c.get('offset', 0.0):.2f}", "-i", fetch_track(c["track"]),
           "-t", f"{a1 - a0:.3f}", "-ar", MSR, "-ac", 2, seg)
        a, _ = sf.read(str(seg), dtype="float32", always_2d=True)
        a = a * 10 ** (c.get("gain_db", 0.0) / 20)
        k = min(int(XF * MSR), len(a) // 2)
        if i > 0:
            a[:k] *= np.linspace(0, 1, k)[:, None]
        if i < len(cues) - 1:
            a[-k:] *= np.linspace(1, 0, k)[:, None]
        s0 = int(a0 * MSR); bed[s0:s0 + len(a)] += a[: n - s0]
        print(f"  music {c['start']:6.1f}s–{c['end']:6.1f}s: \"{c['track']}\" ({musiclib.TRACKS[c['track']]['feel']})")
    return bed, cues

user_track = next((p for p in [PROJ / "music.mp3", PROJ / "music.wav"] if p.exists()), None)
mus = None
if not user_track and MUSIC_STYLE == "library":
    try:
        mus, MUSIC_CUES = library_music()
        MUSIC_SOURCE, MUSIC_CREDITS = "library", musiclib.credits(MUSIC_CUES)
    except Exception as e:
        log_exc("music library failed; using the generated score instead")
        print("⚠️ Music library unavailable (details in the log); using the generated score instead.")
if mus is not None:
    pass
elif user_track:
    print("Using your music:", user_track.name, "(make sure you have the rights to it)")
    ff("-stream_loop", "-1", "-i", user_track, "-t", f"{D:.3f}", "-ar", MSR, "-ac", 2, WORK / "user_music.wav")
    mus, _ = sf.read(str(WORK / "user_music.wav"), dtype="float32", always_2d=True)
    n = len(mus)
    level = np.clip(envelope("base", n) + envelope("curious", n) + 0.6 * envelope("tension", n)
                    + 0.4 * envelope("shimmer", n), 0.15, 1.0)
    mus = mus * level[:, None]
else:
    layers = {
        "base": synth(WORK / "m_base.wav",
            ["sine=frequency=55:sample_rate=48000:duration={d}", "sine=frequency=82.41:sample_rate=48000:duration={d}",
             "sine=frequency=110:sample_rate=48000:duration={d}", "anoisesrc=color=brown:sample_rate=48000:duration={d}:amplitude=0.3"],
            "[0]volume=0.5[a];[1]volume=0.3,tremolo=f=0.1:d=0.6[b];[2]volume=0.15,tremolo=f=0.15:d=0.8[c];"
            "[3]lowpass=f=300,volume=0.4[e];[a][b][c][e]amix=inputs=4:normalize=0,"
            "aecho=0.8:0.7:1000|1800:0.3|0.25,pan=stereo|c0=c0|c1=c0", D),
        "tension": synth(WORK / "m_tension.wav",
            ["sine=frequency=41.2:sample_rate=48000:duration={d}", "sine=frequency=58.27:sample_rate=48000:duration={d}",
             "anoisesrc=color=brown:sample_rate=48000:duration={d}:amplitude=0.5"],
            "[0]volume=0.6,tremolo=f=1.6:d=0.9[a];[1]volume=0.25,tremolo=f=1.6:d=0.9[b];"
            "[2]lowpass=f=180,volume=0.5,tremolo=f=1.6:d=0.7[c];[a][b][c]amix=inputs=3:normalize=0,pan=stereo|c0=c0|c1=c0", D),
        "shimmer": synth(WORK / "m_shimmer.wav",
            ["sine=frequency=440:sample_rate=48000:duration={d}", "sine=frequency=659.25:sample_rate=48000:duration={d}",
             "sine=frequency=880:sample_rate=48000:duration={d}"],
            "[0]volume=0.10,tremolo=f=0.2:d=0.6[a];[1]volume=0.07,tremolo=f=0.27:d=0.6[b];[2]volume=0.04,tremolo=f=0.33:d=0.7[c];"
            "[a][b][c]amix=inputs=3:normalize=0,aecho=0.8:0.8:700|1300:0.35|0.25,pan=stereo|c0=c0|c1=c0", D),
        "curious": curious_layer(D),
    }
    targets = {"base": 0.10, "tension": 0.10, "shimmer": 0.06, "curious": 0.12}   # loudness balance between layers
    n = min(len(a) for a in layers.values())
    mus = sum(rms_norm(a[:n], targets[k]) * envelope(k, n)[:, None] for k, a in layers.items())
    print(f"Generated a 4-layer score that follows the moods (style: {MUSIC_STYLE}; royalty-free, made here).")
if MUSIC_SOURCE == "library":
    # level from the VOICE: the bed sits MUSIC_UNDER_VOICE_DB under the voice (ducking adds more while someone talks)
    sf.write(WORK / "music_bed.wav", mus, MSR)
    v, m = lufs(WORK / "narration_full.wav"), lufs(WORK / "music_bed.wav")
    if v is not None and m is not None:
        target = v - MUSIC_UNDER_VOICE_DB - 20 * math.log10(max(MUSIC_GAIN, 1e-3))
        mus = mus * 10 ** ((target - m) / 20)
        log(f"music level: voice {v} LUFS, bed {m} → {target:.1f} LUFS before the {MUSIC_GAIN} gain and ducking")
    peak = float(np.abs(mus).max())
    if peak > 0.98:
        mus = mus * (0.98 / peak)
else:
    mus = mus / max(float(np.abs(mus).max()), 1e-6) * 0.8
fi, fo = int(3 * MSR), int(4 * MSR)
mus[:fi] *= np.linspace(0, 1, fi)[:, None]
mus = mus[: int(TOTAL * MSR)]
mus[-fo:] *= np.linspace(1, 0, min(fo, len(mus)))[:, None]
sf.write(WORK / "music.wav", mus, MSR)
(PROJ / "render" / "music_credits.txt").write_text(MUSIC_CREDITS or "Generated in the notebook (no licence needed).")

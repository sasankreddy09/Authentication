# ==== 11. Sound effects at story moments (generated here, no licences needed) ====
def make_sfx(kind, dur, out):
    d = f"{dur:.3f}"
    if kind == "boom":
        synth(out, ["anoisesrc=color=brown:sample_rate=48000:duration={d}:amplitude=1", "sine=frequency=45:sample_rate=48000:duration={d}"],
              f"[0]lowpass=f=160,volume=1.2[a];[1]volume=0.8[b];[a][b]amix=inputs=2:normalize=0,"
              f"afade=t=out:st=0.05:d={dur - 0.05:.3f}:curve=exp,pan=stereo|c0=c0|c1=c0", dur)
    elif kind == "rumble":
        synth(out, ["anoisesrc=color=brown:sample_rate=48000:duration={d}:amplitude=0.8"],
              f"[0]lowpass=f=90,volume=1.5,tremolo=f=6:d=0.3,afade=t=in:d={dur / 3:.3f},"
              f"afade=t=out:st={dur * 2 / 3:.3f}:d={dur / 3:.3f},pan=stereo|c0=c0|c1=c0", dur)
    elif kind == "whoosh":
        synth(out, ["anoisesrc=color=pink:sample_rate=48000:duration={d}:amplitude=0.8"],
              f"[0]highpass=f=300,lowpass=f=3500,afade=t=in:d={dur * 0.6:.3f}:curve=exp,"
              f"afade=t=out:st={dur * 0.6:.3f}:d={dur * 0.4:.3f},pan=stereo|c0=c0|c1=c0", dur)
    elif kind == "click":                                                # finger tap on a phone button
        synth(out, ["anoisesrc=color=white:sample_rate=48000:duration={d}:amplitude=0.6"],
              f"[0]highpass=f=2500,lowpass=f=9000,afade=t=out:st=0:d={min(dur, 0.05):.3f}:curve=exp,"
              f"volume=0.8,pan=stereo|c0=c0|c1=c0", dur)
    elif kind == "pop":                                                  # notification / item appearing
        synth(out, ["sine=frequency=880:sample_rate=48000:duration={d}"],
              f"[0]vibrato=f=12:d=0.3,afade=t=in:d=0.01,afade=t=out:st=0.02:d={max(dur - 0.02, 0.04):.3f}:curve=exp,"
              f"volume=0.6,pan=stereo|c0=c0|c1=c0", dur)
    elif kind == "coins":                                                # money arriving
        synth(out, ["sine=frequency=2093:sample_rate=48000:duration={d}", "sine=frequency=2637:sample_rate=48000:duration={d}",
                    "sine=frequency=3136:sample_rate=48000:duration={d}"],
              f"[0]afade=t=out:st=0:d=0.25:curve=exp[a];[1]adelay=90|90,afade=t=out:st=0.09:d=0.25:curve=exp[b];"
              f"[2]adelay=180|180,afade=t=out:st=0.18:d={max(dur - 0.18, 0.1):.3f}:curve=exp[c];"
              f"[a][b][c]amix=inputs=3:normalize=0,volume=0.35,pan=stereo|c0=c0|c1=c0", dur)
    elif kind == "ding":                                                 # app notification
        synth(out, ["sine=frequency=1318.5:sample_rate=48000:duration={d}", "sine=frequency=1975.5:sample_rate=48000:duration={d}"],
              f"[0]volume=0.5,afade=t=out:st=0.0:d={dur:.3f}:curve=exp[a];"
              f"[1]volume=0.35,adelay=110|110,afade=t=out:st=0.11:d={max(dur - 0.11, 0.05):.3f}:curve=exp[b];"
              f"[a][b]amix=inputs=2:normalize=0,pan=stereo|c0=c0|c1=c0", dur)
    elif kind == "rustle":                                               # popcorn / paper bag: soft crackle
        synth(out, ["anoisesrc=color=white:sample_rate=48000:duration={d}:amplitude=0.5"],
              f"[0]highpass=f=1800,lowpass=f=7000,tremolo=f=23:d=0.9,tremolo=f=7:d=0.6,afade=t=in:d=0.04,"
              f"afade=t=out:st={dur * 0.5:.3f}:d={dur * 0.5:.3f},volume=0.7,pan=stereo|c0=c0|c1=c0", dur)
    elif kind == "shimmer":
        synth(out, ["sine=frequency=1318.5:sample_rate=48000:duration={d}", "sine=frequency=1760:sample_rate=48000:duration={d}"],
              f"[0]volume=0.3,tremolo=f=7:d=0.5[a];[1]volume=0.2,tremolo=f=9:d=0.5[b];[a][b]amix=inputs=2:normalize=0,"
              f"aecho=0.8:0.8:400|900:0.3|0.2,afade=t=in:d={dur / 2:.3f},afade=t=out:st={dur / 2:.3f}:d={dur / 2:.3f},"
              f"pan=stereo|c0=c0|c1=c0", dur)
    a, _ = sf.read(str(out), dtype="float32", always_2d=True)
    return a

fx = np.zeros((int(TOTAL * MSR) + MSR, 2), np.float32)
n_fx = 0
for shot, row in zip(SHOTS, TIMELINE):
    for k, cue in enumerate(shot.get("sfx", [])):
        a = make_sfx(cue["type"], float(cue.get("dur", 3.0)), WORK / f"sfx_{shot['shot_id']}_{k}.wav")
        a *= 10 ** ((cue.get("gain_db", -12) + SFX_GAIN_DB) / 20)
        s0 = int((row["start"] + cue.get("at", 0.0)) * MSR)
        seg = fx[s0:s0 + len(a)]; seg += a[:len(seg)]
        n_fx += 1
        print(f"  {shot['shot_id']}: {cue['type']} at {row['start'] + cue.get('at', 0.0):.1f}s")
sf.write(WORK / "sfx_track.wav", fx[: int(TOTAL * MSR)], MSR)
print(f"{n_fx} sound effects placed.")

# ==== 13. Quality checks ====
final = PROJ / "render" / "final.mp4"
qc = {"shots": STATUS, "ai_video_shots": f"{n_video}/{N_VIDEO_SHOTS}", "expected_s": round(TOTAL, 2), "story_arc": [r["mood"] for r in TIMELINE]}
side = {p["id"]: read_sidecar(PROJ / "narration" / f"{p['id']}.wav") or {} for p in PLACED}
qc["speech_check"] = {k: v.get("wer") for k, v in side.items()}
qc["lines_to_listen_to"] = FLAGGED
qc["frozen_last_frame_s"] = {k: v for k, v in HOLDS.items() if v > 0}
qc["continuity"] = {s["shot_id"]: (read_sidecar(PROJ / "clips" / s["shot_id"]) or {}).get("continuity")
                    for s in SHOTS if s.get("must_show")}
qc["continuity_warnings"] = {sid: [m for c in (v or []) for m in c["missing"]]
                             for sid, v in qc["continuity"].items() if any(c["missing"] for c in (v or []))}
_iss = globals().get("ISSUES", [])
qc["validation"] = {"errors": len([i for i in _iss if i["level"] == "error"]),
                    "warnings": [f"[{i['rule']}] {i['shot'] or 'plan'}: {i['message']}" for i in _iss if i["level"] == "warn"]}
qc["review"] = dict(REVIEW, **{sid: "still missing after a remake: " + ", ".join(m)
                               for sid, m in qc.get("continuity_warnings", {}).items() if sid not in REVIEW})
REVIEW_DIR = PROJ / "render" / "review"; REVIEW_DIR.mkdir(parents=True, exist_ok=True)
for _sid in qc["review"]:
    _sc = read_sidecar(PROJ / "clips" / _sid) or {}
    if not _sc.get("parts"):
        continue
    try:
        _n = max(int(_sc.get("frames_per_part", 81)) // 4, 1)
        ff("-i", PROJ / "clips" / _sc["parts"][0], "-vf", f"select=not(mod(n\\,{_n})),scale=480:-2,tile=4x1",
           "-frames:v", 1, "-q:v", 3, REVIEW_DIR / f"{_sid}.jpg")
        _scores = "; ".join(f"{k}: seen in {v['seen']:.0%} of frames" for c in _sc.get("continuity", [])
                            for k, v in (c.get("presence") or {}).items())
        qc["review"][_sid] += f" | picture: render/review/{_sid}.jpg | scores: {_scores}"
    except Exception:
        log_exc(f"{_sid}: could not make the review picture")
if qc["review"]:
    print("Shots to look at before publishing (pictures in render/review/):")
    for _sid, _why in qc["review"].items():
        print(f"  {_sid}: {_why}")
qc["music"] = {"source": MUSIC_SOURCE, "cues": MUSIC_CUES, "credits": MUSIC_CREDITS}
# logical consistency: per scene-state shot, the decision and the evidence
_logic = {}
for s in SHOTS:
    if "logic_shot" in globals() and logic_shot(s):
        sc = read_sidecar(PROJ / "clips" / s["shot_id"]) or {}
        ok, why = logic_usable(s)
        _logic[s["shot_id"]] = {"used_in_video": ok, "why": why, "decision": sc.get("decision"),
                                "failed": sc.get("failed", []), "uncertain": sc.get("uncertain", []),
                                "repairs": sc.get("repairs", []), "attempts": len(sc.get("attempts", []))}
qc["logic"] = _logic
qc["not_logically_verified"] = [s["shot_id"] for s in SHOTS if s.get("recipe", "video") == "video"
                                and not ("logic_shot" in globals() and logic_shot(s))]
qc["subtitles_word_timed"] = f"{sum(bool(p.get('words')) for p in PLACED)}/{len(PLACED)}"
# ---- measurements of the rendered file (qclib: ffprobe + FFmpeg detectors) ----
TH = dict(qclib.THRESHOLDS, loudness_target_lufs=-14.0)
C = qclib.check
checks = {}
try:
    pr = qclib.probe(final)
    v, a = pr["video"] or {}, pr["audio"] or {}
    qc["probe"] = pr
    ok = (v.get("codec") == "h264" and v.get("pix_fmt") == "yuv420p" and a.get("codec") == "aac"
          and (v.get("width"), v.get("height")) == (W, H) and abs(v.get("fps", 0) - FPS) < 0.01)
    checks["playable H.264/AAC stream, right size and fps"] = C("PASS" if ok else "FAIL", evidence={"video": v, "audio": a},
        fix=f"expected h264 yuv420p {W}x{H} @ {FPS} fps + AAC")
    drift = abs(pr["duration"] - TOTAL)
    checks["no duration drift (video vs timeline)"] = C("PASS" if drift <= TH["duration_drift_s"] else "FAIL",
        evidence={"file_s": round(pr["duration"], 2), "timeline_s": round(TOTAL, 2)}, fix="re-run the render cell")
    cut = (v.get("duration", 0) - a.get("duration", 0)) > 0.3
    last_line_end = max((pp["end"] for pp in PLACED), default=0.0)
    checks["ending not cut off (audio covers the video, last line ends in time)"] = C(
        "FAIL" if cut or last_line_end > pr["duration"] - 0.2 else "PASS",
        evidence={"audio_s": a.get("duration"), "video_s": v.get("duration"), "last_line_end_s": round(last_line_end, 2)})
except Exception as e:
    log_exc("ffprobe of the final video failed")
    checks["playable H.264/AAC stream, right size and fps"] = C("UNKNOWN", evidence=repr(e)[:200], fix="check FFmpeg")
try:
    ev = qclib.video_events(final, TH)
    qc["black_segments"], qc["freezes"] = ev["black"], ev["freezes"]
    bad_black = [(x, y) for x, y in ev["black"] if not (x < 0.7 or y > TOTAL - 1.0)]   # fades at start/end are planned
    checks["no unplanned black frames"] = C("FAIL" if bad_black else "PASS", evidence=bad_black)
    # freezes only matter inside AI-video shots: drawn cards and stills hold still on purpose
    frozen = [r["shot_id"] for r, sh in zip(TIMELINE, SHOTS) if sh.get("recipe", "video") == "video"
              and qclib.overlaps(ev["freezes"], r["start"], r["start"] + r["dur"], 0.6)]
    checks["no frozen frames inside AI-video shots"] = C("FAIL" if frozen else "PASS", shots=frozen,
        evidence=ev["freezes"], fix="regenerate those clips or use their fallback")
except Exception as e:
    log_exc("video event detection failed")
    checks["no unplanned black frames"] = C("UNKNOWN", evidence=repr(e)[:200])
try:
    au = qclib.audio_stats(final, TH)
    qc["audio"] = au; qc["loudness_lufs"] = au["integrated_lufs"]
    lu = au["integrated_lufs"]
    checks["loudness on target"] = C("UNKNOWN" if lu is None else
        "PASS" if abs(lu - TH["loudness_target_lufs"]) <= TH["loudness_tolerance_lu"] else "FAIL",
        evidence={"integrated_lufs": lu, "target": TH["loudness_target_lufs"]})
    tp = au["true_peak_dbtp"]
    checks["no clipping (true peak)"] = C("UNKNOWN" if tp is None else "PASS" if tp <= TH["true_peak_max_dbtp"] else "FAIL",
                                          evidence={"true_peak_dbtp": tp, "max": TH["true_peak_max_dbtp"]})
    checks["no unintended silence"] = C("FAIL" if au["silences"] else "PASS", evidence=au["silences"],
                                        fix="a gap in narration and music: check the timeline and music cues")
except Exception as e:
    log_exc("audio analysis failed")
    checks["loudness on target"] = C("UNKNOWN", evidence=repr(e)[:200])
try:                                                  # the ducked music bed, rebuilt with the same filter, measured
    duck_wav = WORK / "qc_music_ducked.wav"
    ff("-i", WORK / "narration_full.wav", "-i", WORK / "music.wav", "-filter_complex",
       f"[0:a]aresample=48000,pan=stereo|c0=c0|c1=c0,{AF}[sc];[1:a]aresample=48000,{AF},volume={MUSIC_GAIN}[m];"
       f"[m][sc]{duck_filter()}[d]", "-map", "[d]", duck_wav)
    dk = qclib.ducking(WORK / "narration_full.wav", duck_wav)
    qc["ducking"] = dk
    checks["music ducks under the voice"] = C("UNKNOWN" if dk["reduction_db"] is None else
        "PASS" if dk["reduction_db"] >= TH["duck_min_db"] else "FAIL", evidence=dk,
        fix="raise DUCK ratio or lower DUCK threshold")
except Exception as e:
    log_exc("ducking check failed")
    checks["music ducks under the voice"] = C("UNKNOWN", evidence=repr(e)[:200])
try:
    sheet = qclib.contact_sheet(final, PROJ / "render" / "contact_sheet.jpg")
    qc["contact_sheet"] = str(sheet)
except Exception:
    log_exc("contact sheet failed")

# ---- checks from the pipeline's own records ----
checks["narration complete and heard correctly"] = C("PASS" if not FLAGGED and all(
    v is not None for v in qc["speech_check"].values()) else "FAIL" if FLAGGED else "UNKNOWN",
    shots=FLAGGED, evidence=qc["speech_check"], fix="listen to these lines; re-run the narration cell")
checks["every video shot is verified AI video (no unchecked stand-ins)"] = C(
    "PASS" if n_video == N_VIDEO_SHOTS else "FAIL", evidence=STATUS,
    shots=[s["shot_id"] for s in SHOTS if s.get("recipe", "video") == "video" and "AI video" not in STATUS.get(s["shot_id"], "")])
checks["no frozen last frames (clip shorter than its shot)"] = C(
    "PASS" if all(v <= 0.6 for v in HOLDS.values()) else "FAIL", shots=[k for k, v in HOLDS.items() if v > 0.6], evidence=HOLDS)
checks["characters stay on screen"] = C("PASS" if not qc.get("continuity_warnings") else "FAIL",
                                       shots=list(qc.get("continuity_warnings", {})), evidence=qc.get("continuity_warnings"))
checks["music licensed and credited"] = C("PASS" if MUSIC_SOURCE != "library" or bool(MUSIC_CREDITS) else "FAIL",
                                          evidence=MUSIC_CREDITS)
checks["nothing flagged for review"] = C("PASS" if not qc.get("review") else "FAIL", shots=list(qc.get("review", {})),
                                         evidence=qc.get("review"), fix="look at render/review/ and approve or fix")
checks["every scene-state shot verified or approved"] = C(
    "PASS" if all(x["used_in_video"] for x in qc.get("logic", {}).values()) else "FAIL",
    shots=[k for k, x in qc.get("logic", {}).items() if not x["used_in_video"]], evidence=qc.get("logic"))
checks["plan validation: no errors"] = C("PASS" if not qc["validation"]["errors"] else "FAIL", evidence=qc["validation"])
checks["AI shots without logical verification (information)"] = C(
    "PASS" if not qc["not_logically_verified"] else "UNKNOWN", mandatory=False, shots=qc["not_logically_verified"])
if globals().get("FACTCHECK") is not None:            # topic videos: every narration sentence vs its sources
    bad = {k: x for k, x in FACTCHECK.items() if x["status"] != "PASS"}
    checks["every narrated fact supported by its sources"] = C("PASS" if not bad else "FAIL", shots=sorted(bad),
        evidence=bad, fix="rewrite or remove these sentences (render/factcheck.json)")
if globals().get("ASSET_MANIFEST") is not None:      # licensed images: allowed licence + provenance recorded
    badl = [a["id"] for a in ASSET_MANIFEST if not a.get("license_ok")]
    checks["every image has an allowed licence and its credit"] = C("PASS" if not badl else "FAIL", shots=badl,
        evidence={a["id"]: a.get("license") for a in ASSET_MANIFEST})
if PLAN.get("story_mode") == "two_friends_cinematic":  # ---- storytelling checks (two-friend videos) ----
    MEDIUM_OF = {"video": "ai_video", "stock": "licensed_media", "card": "presentation_graphic", "ui": "presentation_graphic",
                 "diagram": "presentation_graphic", "journey": "animated_demo", "character": "character_scene",
                 "demo": "animated_demo"}
    _rows = []
    for s, r in zip(SHOTS, TIMELINE):
        if s.get("recipe") == "story":
            _rows += STORY_ROWS.get(s["shot_id"], [])
        else:
            _rows.append({"shot": s["shot_id"], "visual": s["shot_id"], "intent": s.get("intent", s.get("recipe", "video")),
                          "medium": MEDIUM_OF.get(s.get("recipe", "video"), "?"), "start": r["start"], "dur": r["dur"],
                          "label": s.get("label", ""), "description": describe(s)})
    stats = story.statistics(_rows)
    qc["story_statistics"] = stats
    (PROJ / "render" / "story_stats.json").write_text(json.dumps({"statistics": stats, "visuals": _rows}, indent=1))
    _ov = [(a["id"], b["id"]) for a, b in zip(PLACED, PLACED[1:]) if b["start"] < a["end"] - 0.02]
    _order = [p_["id"] for p_ in PLACED] == [l["id"] for s in SHOTS for l in s["lines"]]
    checks["dialogue: turns in script order, no two friends talking at once"] = C(
        "PASS" if not _ov and _order else "FAIL", evidence={"overlaps": _ov, "in_order": _order})
    _fact = [l for s in SHOTS for l in s["lines"] if l.get("kind") in story.FACTUAL_KINDS]
    _claims = {c["id"]: c for c in PLAN.get("claims", [])}
    _ungrounded = [l["id"] for l in _fact if not l.get("claims") or not all(
        any(str(u).startswith("https://") for u in _claims.get(c, {}).get("sources", [])) for c in l["claims"])]
    checks["every factual dialogue line cites a sourced fact"] = C("PASS" if not _ungrounded else "FAIL",
        shots=_ungrounded, evidence={"factual_lines": len(_fact)}, fix="cite a claim with a source URL or rephrase")
    checks["presentation graphics at most 20% of screen time"] = C(
        "PASS" if stats["presentation_graphics_pct"] <= 20 else "FAIL", evidence=stats["rendering_medium_pct"],
        fix="replace cards/diagrams with the friends, objects or a demonstration")
    _unl = [f"{r['shot']}/{r['visual']}" for r in _rows if r["intent"] in ("reenactment", "real_event")
            and r["medium"] in ("character_scene", "ai_video") and r.get("label") != "Illustration"]
    checks["re-enactments are labelled 'Illustration' (never shown as real footage)"] = C(
        "PASS" if not _unl else "FAIL", shots=_unl)
    _beats = [s.get("story_beat") for s in SHOTS if s.get("story_beat")]
    _miss = [b for b in ("question", "closing") if b not in _beats]
    _answered = any(b in _beats for b in ("first_answer", "demonstration", "correction", "reveal"))
    checks["story complete: question, answer, closing"] = C("PASS" if not _miss and _answered else "FAIL",
        evidence={"beats": _beats, "missing": _miss, "answered": _answered})
    _long = [f"{r['shot']}/{r['visual']} {r['dur']}s" for r in _rows if r["dur"] > 12]
    checks["visual variety: no single visual longer than 12 s (information)"] = C(
        "PASS" if not _long else "UNKNOWN", mandatory=False, shots=_long)
    checks["holistic review by a person: contact sheet + dialogue/shot timeline (information)"] = C(
        "UNKNOWN", mandatory=False, evidence="render/contact_sheet.jpg, render/dialogue_shot_timeline.md")
    _cl = []
    for p_ in PLACED:
        _m = ((side.get(p_["id"]) or {}).get("voice_qc") or {}).get("metrics", {})
        _cl.append({"id": p_["id"], "speaker": p_["speaker"], "start": p_["start"], "end": p_["end"],
                    "rms_db": _m.get("rms_db"), "sps": _m.get("speaking_rate_sps")})
    _cd = voice_qc.conversation_diagnostics(_cl)
    qc["conversation_timing"] = _cd
    checks["conversation timing: no dead air, overlaps, mechanical gaps or level jumps (diagnostic)"] = C(
        "PASS" if not _cd["issues"] else "UNKNOWN", mandatory=False, evidence=_cd,
        fix="listen to these places; a person decides whether it sounds like a real conversation")
    _md = [f"# Dialogue and shot timeline: {PLAN['title']}", "", "## Audio timeline (dialogue turns)", "",
           "| Start | End | Speaker | Line | Kind | Sources |", "|---|---|---|---|---|---|"]
    _lk = {l["id"]: l for s in SHOTS for l in s["lines"]}
    for p_ in PLACED:
        l = _lk[p_["id"]]
        _md.append(f"| {p_['start']:.2f} | {p_['end']:.2f} | {CHARS[p_['speaker']]['name']} | {p_['text']} | "
                   f"{l.get('kind', '')} | {', '.join(l.get('claims', []))} |")
    _md += ["", "## Visual timeline (independent of the turns)", "",
            "| Start | Length | Scene / visual | Purpose | Medium | What is seen |", "|---|---|---|---|---|---|"]
    for r in _rows:
        _md.append(f"| {r['start']:.2f} | {r['dur']:.2f} | {r['shot']}/{r['visual']} | {r['intent']} | {r['medium']} | "
                   f"{r.get('description', '')}{(' (' + r['label'] + ')') if r.get('label') else ''} |")
    _md += ["", "## Statistics (screen time)", "", "**Narrative purpose:** " + ", ".join(
        f"{k} {v}%" for k, v in stats["narrative_purpose_pct"].items()), "", "**Rendering medium:** " + ", ".join(
        f"{k} {v}%" for k, v in stats["rendering_medium_pct"].items())]
    (PROJ / "render" / "dialogue_shot_timeline.md").write_text("\n".join(_md) + "\n")
    print("Story statistics — purpose: " + ", ".join(f"{k} {v}%" for k, v in stats["narrative_purpose_pct"].items())
          + "\n                   medium:  " + ", ".join(f"{k} {v}%" for k, v in stats["rendering_medium_pct"].items()))
if globals().get("MAX_DURATION"):                     # short-form limit (timeline cell), measured on the real file
    _dur = (qc.get("probe") or {}).get("duration") or TOTAL
    checks[f"video no longer than {MAX_DURATION} s"] = C(
        "PASS" if _dur <= MAX_DURATION + 0.5 else "FAIL", evidence={"duration_s": round(_dur, 2)},
        fix="shorten the script with a fact-preserving revision (never speed up the voices or cut the ending)")
if globals().get("STORYBOOK"):                        # illustrated characters + environment packs
    _sr = globals().get("STORYBOOK_REPORT", {})
    _bad = {k: v["verify"] for k, v in _sr.items() if v.get("verify")}
    checks["storybook assets: exact versions present, every layer unchanged"] = C(
        "PASS" if _sr and not _bad else "FAIL", evidence=_sr, fix="restore the asset version folder on Drive")
    _prov = sorted(k for k, v in _sr.items() if not k.startswith("env:") and v.get("status") != "approved")
    checks["character artwork approved by a person (not provisional)"] = C(
        "PASS" if _sr and not _prov else "UNKNOWN", shots=_prov,
        evidence={k: v.get("status") for k, v in _sr.items() if not k.startswith("env:")},
        fix="MANUAL_REVIEW_REQUIRED: approve a Character Studio version (colab/approve_character.py), then pin it")
    _mo = globals().get("STORYBOOK_MOUTH", {})
    _silent_open = {sid: {c: r["open_in_silence"] for c, r in v.items() if r["open_in_silence"]} for sid, v in _mo.items()}
    _silent_open = {k: v for k, v in _silent_open.items() if v}
    _sp = sum(r["speech_frames"] for v in _mo.values() for r in v.values())
    _op = sum(r["open_in_speech"] for v in _mo.values() for r in v.values())
    checks["lip sync: only the speaker's mouth moves, and it moves while speaking"] = C(
        "UNKNOWN" if not _mo else "FAIL" if _silent_open or (_sp and _op / _sp < 0.3) else "PASS",
        shots=list(_silent_open), evidence={"speech_frames": _sp, "open_mouth_in_speech": _op, "open_in_silence": _silent_open},
        fix="check the viseme track (storybook.visemes_from_audio) for these scenes")
if PLAN.get("localization"):                          # Telugu lines vs their verified English claims
    _loc = PLAN["localization"]
    _rev = [k for k, r in _loc.items() if r.get("status") != "PASS"]
    checks["Telugu factual lines keep the verified claims (numbers, names, key terms)"] = C(
        "PASS" if not _rev else "UNKNOWN", shots=_rev, evidence=_loc,
        fix="MANUAL_REVIEW_REQUIRED: a Telugu speaker compares these lines with their source_text")
_gr = globals().get("GEMINI_REPORT")
if _gr:                                                # the hosted free-tier TTS exception, if it was used
    qc["gemini_tts"] = {k: v for k, v in _gr.items() if k != "lines"}
    _used = _gr.get("status") == "PASS"
    checks["Gemini free-tier TTS: approved model, within the local budget (information)"] = C(
        "PASS" if (not _used or (GEMINI_TTS_MODEL in gemini_tts.FREE_TTS_MODELS and
                                 _gr.get("calls", 0) <= MAX_GEMINI_TTS_CALLS_PER_VIDEO)) else "FAIL",
        mandatory=False, evidence={"used": _used, "calls": _gr.get("calls"), "why_not": _gr.get("why")})
    if _used:
        _mis = (_gr.get("attribution") or {}).get("misattributed", [])
        checks["Gemini voices: every turn split, attributed to the right friend and heard correctly"] = C(
            "PASS" if not _mis and all(c.get("align", "PASS") == "PASS" or c.get("mode") == "per-turn"
                                       for c in _gr.get("chunks", [])) else "FAIL",
            evidence={"chunks": _gr.get("chunks"), "attribution": _gr.get("attribution")})
qc["devices"] = dict(DEVICE_REPORT)                   # where each step actually ran, and the whole-run peaks
qc["peaks"] = dict(RUN_PEAKS)
print("\nWhere each step ran:")
for k_, v_ in DEVICE_REPORT.items():
    print(f"  {k_}: {v_}")
print(f"  peak system RAM {RUN_PEAKS['ram_gb']} GB | peak GPU memory {RUN_PEAKS['gpu_gb']} GB (all processes)")
checks['Speaker identity (manual listening needed)'] = C(
    'UNKNOWN', mandatory=False, evidence='ECAPA and its extra retakes are omitted from the free-tier preset.')
if PLAN.get('story_revision') == 'three_d_friends_te_v4':
    checks['Revised Telugu script and pronunciation reviewed'] = C(
        'UNKNOWN', evidence='New v4 dialogue needs fluent-speaker listening and factual review; old approvals do not apply.')
qc["checks"] = checks
qc["approved"] = qclib.approved(checks)
qc["thresholds"] = TH
(PROJ / "render" / "qc_report.json").write_text(json.dumps(qc, indent=2, default=str))
log("qc: " + json.dumps({k: c["status"] for k, c in checks.items()}))
print("\nFinal checks:")
for k, c in checks.items():
    mark = {"PASS": "✅", "FAIL": "❌", "UNKNOWN": "❔"}[c["status"]]
    print(f"  {mark} {k}" + ("" if c["mandatory"] else " (info)") + (f"  → {', '.join(c['shots'])}" if c["shots"] else ""))
print(f"\n{'APPROVED ✅ every mandatory check passed' if qc['approved'] else 'NOT APPROVED ⚠️ fix or review the ❌/❔ items'}"
      f" | report: render/qc_report.json | contact sheet: render/contact_sheet.jpg")

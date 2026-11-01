# ==== 4. The story plan (written by Claude; edit plan.json on Drive to change it) ====
PLAN_EMBEDDED = json.loads(r"""
__PLAN_JSON__
""")

plan_path = PROJ / "plan.json"
if PLAN_EMBEDDED is None:                    # topic notebook: the plan was just made (and saved) by the topic cell
    PLAN = TOPIC_PLAN
else:
    PLAN = PLAN_EMBEDDED
    if plan_path.exists():
        on_drive = json.loads(plan_path.read_text())
        if on_drive.get("plan_version", 1) >= PLAN_EMBEDDED["plan_version"]:
            PLAN = on_drive
            print("Using plan.json from Drive (your edits). Delete it to restore the built-in story.")
        else:
            backup = PROJ / f"plan.v{on_drive.get('plan_version', 1)}.backup.json"
            plan_path.rename(backup)
            print(f"Found an older plan on Drive; kept it as {backup.name} and switched to the new built-in story.")
    if PLAN is PLAN_EMBEDDED:
        plan_path.write_text(json.dumps(PLAN, indent=2, ensure_ascii=False))

STYLE, NEG = PLAN["style"]["suffix"], PLAN["style"]["negative"]
SHOTS, MOODS, CHARS, TONES = PLAN["shots"], PLAN["moods"], PLAN["characters"], PLAN["tones"]
SFX_TYPES = {"boom", "rumble", "whoosh", "shimmer", "ding", "click", "pop", "coins", "rustle"}
DRAWN_RECIPES = {"diagram", "ui", "journey", "card", "story", "character", "demo"}   # made by code (no AI, no GPU)
if PLAN.get("bible"):                                  # two-friend videos: the plan's own Character Bible
    charanim.set_bible(characters.load(PLAN["bible"]))
if PLAN.get("visual_style_version"):                   # storybook: install the pinned asset versions before checks
    os.environ.setdefault("STORYBOOK_ASSETS", str(Path(DRIVE_ROOT) / "_assets"))
    STORYBOOK_PREP = storybook.prepare_assets(PLAN, json.loads(r"""__STORYBOOK_PACK__"""), log=log)
    print("Storybook assets: " + "; ".join(f"{k}: {v}" for k, v in STORYBOOK_PREP.items()))
VOICE_FX_TYPES, TONE_FX_TYPES = {"none", "sparkle", "old"}, {"shaky", "breathy"}
KOKORO_EN_VOICES = {"af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
                    "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
                    "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
                    "bm_daniel", "bm_fable", "bm_george", "bm_lewis"}
def mood(shot):
    return MOODS[shot.get("mood", "calm")]

# checks: the plan may only use characters, voices, tones and effects this notebook knows
assert len({s["shot_id"] for s in SHOTS}) == len(SHOTS), "duplicate shot ids"
PLAN_LANGUAGE = PLAN.get("language", "en")
if PLAN_LANGUAGE == "en":
    VOICE_NAMES = KOKORO_EN_VOICES
else:                                                  # Indic Parler-TTS speakers trained for this language
    import parler_worker
    VOICE_NAMES = set(parler_worker.SPEAKERS.get(PLAN_LANGUAGE, []))
    assert VOICE_NAMES, f"no voice model configured for language '{PLAN_LANGUAGE}'"
for cid, c in CHARS.items():
    assert c["voice"] in VOICE_NAMES, f"{cid}: unknown voice {c['voice']}; choose from {sorted(VOICE_NAMES)}"
    assert c.get("fx", "none") in VOICE_FX_TYPES, f"{cid}: unknown voice effect {c.get('fx')}"
for tid, t in TONES.items():
    assert t.get("fx") in TONE_FX_TYPES | {None}, f"tone {tid}: unknown effect {t.get('fx')}"
for s in SHOTS:
    sid = s["shot_id"]
    assert s.get("mood", "calm") in MOODS, f"{sid}: unknown mood {s.get('mood')}"
    for c in s.get("cast", []):
        assert c in CHARS, f"{sid}: unknown character {c}"
    for l in s["lines"]:
        assert l["speaker"] in CHARS, f"{sid}: unknown speaker {l['speaker']}"
        assert l["tone"] in TONES, f"{sid}: unknown tone {l['tone']}"
    for c in s.get("sfx", []):
        assert c["type"] in SFX_TYPES, f"{sid}: unknown sound effect {c['type']}"

# ---- continuity check of the plan (before any GPU time is spent) ----
# 1. whoever speaks (except the narrator) must be on screen in that shot
# 2. nobody may vanish for one shot and come back in the same place (A-B-A), unless the shot is a cutaway
# 3. every shot must fit in ONE continuous AI clip (5 s, slowed down at most MAX_SLOWMO): a longer shot would need
#    a second, unrelated clip, and characters would jump or vanish in the middle of it
import unicodedata
EST_SEC_PER_LETTER = 0.1                             # Telugu speaking time per letter incl. vowel signs (conservative)
CLIP_SECONDS = 5.04                                  # Wan clips: 121 frames @ 24 fps or 81 frames @ 16 fps
SHOT_BUDGET = CLIP_SECONDS * MAX_SLOWMO              # longest shot that is still one continuous clip
def est_seconds(shot):
    """Conservative speaking-time estimate (the real one comes after the voices are made)."""
    t = float(shot.get("pre_s", 0.3))
    for l in shot["lines"]:
        c, tn = CHARS[l["speaker"]], TONES[l["tone"]]
        txt = l.get("tts") or l["text"]
        speed = c.get("speed", 1.0) * tn.get("speed", 1.0) * SPEED_SCALE
        if PLAN_LANGUAGE == "en":
            t += len(txt.split()) / (3.0 * speed)
        else:                                        # long, inflected words: estimate from letters instead
            t += EST_SEC_PER_LETTER * sum(1 for ch in txt if ch.isalnum() or unicodedata.category(ch)[0] == "M") / speed
        t += 0.45 * txt.count("...") + 0.12 * len(re.findall(r"[.!?,;:]", txt))
        t += float(l.get("pause_after", tn.get("pause_after", MOODS[shot.get("mood", "calm")]["pause_after"])))
    return t

problems, place = [], lambda s: s.get("location", s.get("scene", s["shot_id"]))
ARCH = PLAN.get("architecture", {"nodes": {}, "edges": []})
for s in SHOTS:
    if s.get("recipe", "video") not in {"video", "stock"} | DRAWN_RECIPES:
        problems.append(f"{s['shot_id']}: unknown recipe {s.get('recipe')} (use video, stock, ui, journey, card, "
                        "diagram, story, character or demo)")
    if s.get("recipe") == "stock":                  # licensed image: file + credit + an allowed licence, any length
        st = s.get("stock", {})
        if not st.get("file") or not (PROJ / "assets" / st["file"]).exists():
            problems.append(f"{s['shot_id']}: licensed image file missing (assets/{st.get('file')})")
        if not st.get("credit") or not st.get("license_ok"):
            problems.append(f"{s['shot_id']}: licensed image needs its credit and an allowed licence")
        continue
    if s.get("recipe") in ("story", "character", "demo"):  # two-friend scenes: drawn by code, any length
        for v in (s.get("story") or {}).get("visuals", []):
            if v.get("recipe") == "photo" and not os.path.isabs(v["photo"].get("file", "")):
                v["photo"]["file"] = str(PROJ / "assets" / v["photo"].get("file", ""))   # licensed image on Drive
            st_ = v.get("start", {})
            if not 0 <= st_.get("line", -1) < len(s["lines"]) or not 0 <= st_.get("at", -1) <= 1:
                problems.append(f"{s['shot_id']}: visual {v.get('id')} starts at {st_} (no such line/fraction)")
        problems += [f"{s['shot_id']}: {m}" for m in drawkit.check_spec(s["recipe"], s.get(s["recipe"], {}), PLAN)]
        for l in s["lines"]:
            if l["speaker"] != "narrator" and l["speaker"] not in s.get("cast", []):
                problems.append(f"{s['shot_id']}: {CHARS[l['speaker']]['name']} speaks but is not in the scene's cast")
        continue
    if s.get("recipe") in ("ui", "journey", "card"):        # drawn scenes: check the spec, no AI clip budget
        problems += [f"{s['shot_id']}: {m}" for m in drawkit.check_spec(s["recipe"], s.get(s["recipe"], {}), PLAN)]
        for l in s["lines"]:
            if l["speaker"] != "narrator" and l["speaker"] not in s.get("cast", []):
                problems.append(f"{s['shot_id']}: {CHARS[l['speaker']]['name']} speaks but is not on screen")
        continue
    if s.get("recipe") == "diagram":                # drawn shots: check they only use boxes that exist
        dg = s.get("diagram", {})
        used = set(dg.get("focus", [])) | ({*dg["show"]} if isinstance(dg.get("show"), list) else set())
        used |= {f[k] for f in dg.get("flows", []) for k in ("from", "to")}
        for n in sorted(used - set(ARCH["nodes"])):
            problems.append(f"{s['shot_id']}: diagram uses unknown box '{n}'")
        tab = dg.get("table")
        for c in (tab or {}).get("changes", []):
            if c["col"] not in tab["columns"] or c["row"] >= len(tab["rows"]):
                problems.append(f"{s['shot_id']}: table change {c} does not match the table")
        if s.get("cast"):
            problems.append(f"{s['shot_id']}: a diagram shot can't have a cast")
        continue
    for l in s["lines"]:
        if l["speaker"] != "narrator" and l["speaker"] not in s.get("cast", []):
            problems.append(f"{s['shot_id']}: {CHARS[l['speaker']]['name']} speaks but is not on screen (add to cast)")
    if est_seconds(s) > SHOT_BUDGET and not s.get("allow_long"):
        problems.append(f"{s['shot_id']}: ~{est_seconds(s):.1f}s of speech is longer than one clip "
                        f"({SHOT_BUDGET:.1f}s): split it into two shots")
for a, b in ARCH["edges"]:
    if a not in ARCH["nodes"] or b not in ARCH["nodes"]:
        problems.append(f"architecture: arrow {a} -> {b} uses an unknown box")
for a, b, c in zip(SHOTS, SHOTS[1:], SHOTS[2:]):
    if place(a) == place(b) == place(c) and not b.get("cutaway"):
        for who in set(a.get("cast", [])) & set(c.get("cast", [])) - set(b.get("cast", [])):
            problems.append(f"{b['shot_id']}: {CHARS[who]['name']} is in {a['shot_id']} and {c['shot_id']} at the same "
                            f"place but vanishes in {b['shot_id']} (add them, or mark the shot \"cutaway\": true)")
# ---- scene validation (engine/lib/validate.py): logic, facts, process accuracy, narration ↔ visual, consistency
ISSUES = validate.validate(PLAN, {"words_per_second": 2.6})
if PLAN.get("story_mode") == "two_friends_cinematic":
    # dialogue rules of the Story Director (sourced factual turns, no monologue, real questions, corrected guesses)
    # re-checked here on the plan as it will be made (plan.json may have been edited on Drive)
    try:
        _st = story.Story(
            title=PLAN["title"], topic=PLAN["title"], language=PLAN.get("language", "en"),
            facts=[story.StoryFact(id=c["id"], text=c["text"], source_url=c["sources"][0]) for c in PLAN.get("claims", [])],
            scenes=[story.Scene(id=s["shot_id"], beat=s.get("story_beat", "first_answer"), world=s.get("location", "studio"),
                                turns=[story.Turn(speaker=l["speaker"], text=l["text"], kind=l.get("kind", "observation"),
                                                  fact_ids=l.get("claims", []),
                                                  tone=l["tone"] if l["tone"] in story.TONES else "friendly")
                                       for l in s["lines"]],
                                visuals=[story.Visual(id=v["id"], intent=v["intent"], turn=v["start"]["line"],
                                                      at=v["start"]["at"]) for v in s["story"]["visuals"]])
                    for s in SHOTS if s.get("recipe") == "story"])
    except Exception as e:                             # pydantic ValidationError: report it like any plan problem
        _st = None
        ISSUES.append({"level": "error", "rule": "D0", "shot": None, "message": f"story structure: {e}"[:1500],
                       "fix": "see colab/engine/lib/story.py for the schema"})
    ISSUES += story.check_story(_st, charanim.bible()) if _st else []
    n_ai = sum(s.get("recipe", "video") == "video" for s in SHOTS)
    if n_ai > MAX_AI_VIDEO_SHOTS:
        ISSUES.append({"level": "error", "rule": "X6", "shot": None, "message": f"{n_ai} AI video shots in a story-mode "
                       f"plan (MAX_AI_VIDEO_SHOTS = {MAX_AI_VIDEO_SHOTS})", "fix": "use character scenes, demonstrations "
                       "or licensed media, or raise MAX_AI_VIDEO_SHOTS"})
# ---- scene-state engine (engine/lib/scene_state.py): positions, facing, camera side, ownership, ordered events.
# Simulated shot by shot; contradictions (crossing the line, reversals, objects changing hands without an event,
# acting from across the room, too many actions for one clip, actions ending at the cut) stop the run here.
STATES, SCENE_ENTS, _st_problems = scene_state.parse_plan(PLAN)
STATES = {st.shot_id: st for st in STATES}
_worlds, _st_issues = scene_state.check_sequence(list(STATES.values()), SCENE_ENTS)
ISSUES += [i.model_dump() for i in _st_problems + _st_issues]
for s in SHOTS:
    if s.get("state") and s.get("recipe", "video") == "video" and not s.get("fallback"):
        ISSUES.append({"level": "warn", "rule": "X5", "shot": s["shot_id"], "message": "AI shot with required actions has "
                       "no drawn \"fallback\": if its clip fails the checks, a still picture is used instead", "fix":
                       "add \"fallback\": {\"recipe\": \"card\", ...}"})
if STATES:
    print(f"Scene-state: {len(STATES)} shots with positions, camera side and ordered events; "
          f"{sum(len(st.events) for st in STATES.values())} events simulated")
ISSUES += [{"level": "error", "rule": "C", "shot": p.split(":")[0], "message": p, "fix": ""} for p in problems]
(PROJ / "render").mkdir(exist_ok=True)
(PROJ / "render" / "validation_report.json").write_text(json.dumps(ISSUES, indent=2, ensure_ascii=False))
(PROJ / "render" / "validation_report.txt").write_text(validate.report(ISSUES) or "No issues.")
if ISSUES:
    print("Scene validation:\n" + validate.report(ISSUES))
if validate.errors(ISSUES):
    log("plan validation errors:\n" + validate.report(validate.errors(ISSUES)), "ERROR")
    raise ValueError(f"The plan failed validation ({len(validate.errors(ISSUES))} errors, listed above and in "
                     "render/validation_report.txt). Nothing was generated: fix plan.json and run again.")
n_dg = sum(s.get("recipe") in DRAWN_RECIPES for s in SHOTS)
# names and special words for the speech check (it would otherwise hear "Zomato" as "the motto")
ASR_PROMPT = ", ".join(dict.fromkeys([c["name"] for k, c in CHARS.items() if k != "narrator"] + PLAN.get("vocabulary", [])))
log(f"speech-check vocabulary: {ASR_PROMPT}")
print(f"Continuity check of the plan: OK (every AI shot fits one {SHOT_BUDGET:.1f}s clip; speakers on screen; "
      f"nobody vanishes; {n_dg} drawn scenes checked; {len(validate.errors(ISSUES))} validation errors, "
      f"{len(ISSUES) - len(validate.errors(ISSUES))} warnings)")

# turn the plan into what the shared video engine uses: the same character description in every prompt
# keeps each character looking (roughly) the same from shot to shot; "must_show" is what the clip check looks for
for s in SHOTS:
    for k, l in enumerate(s["lines"], 1):
        l["id"] = f"{s['shot_id']}_{k}"
    if s.get("recipe") in DRAWN_RECIPES | {"stock"}:  # drawn scenes and licensed images: no AI prompts, never AI video
        s["must_show"] = []
        continue
    looks = [CHARS[c]["look"] for c in s.get("cast", []) if CHARS[c].get("look")]
    props = PLAN.get("props", {})
    looks += [props[p]["look"] for p in s.get("props", []) if p in props]   # same object, same words, every shot
    s["recipe"] = "video"
    s["image_prompt"] = ", ".join([s["scene"], *looks])
    s["motion_prompt"] = s["action"]
    if s["shot_id"] in STATES:                        # spatial wording from the scene-state (the layout video steers)
        s["motion_prompt"] = s["action"] + ", " + scene_state.prompt_hints(STATES[s["shot_id"]], SCENE_ENTS)
    s["must_show"] = [CHARS[c].get("tag") or CHARS[c]["look"].split(",")[0]
                      for c in s.get("cast", []) if CHARS[c].get("look") and not s.get("skip_check")]
    s["must_show"] += [props[p]["tag"] for p in s.get("props", [])            # small objects: only if "check": true
                       if props.get(p, {}).get("tag") and props[p].get("check") and not s.get("skip_check")]

n_words = sum(len(l["text"].split()) for s in SHOTS for l in s["lines"])
print(f'"{PLAN["title"]}": {len(SHOTS)} shots, {sum(len(s["lines"]) for s in SHOTS)} lines, {n_words} words')
for cid, c in CHARS.items():
    n = sum(l["speaker"] == cid for s in SHOTS for l in s["lines"])
    print(f"  {c['name']:<9} voice {c['voice']:<10} pitch {c.get('pitch', 0):+.1f}  effect {c.get('fx', 'none'):<8} {n} lines")
print("Story arc:", " → ".join(s.get("mood", "calm") for s in SHOTS))
log("Prompts:\n" + "\n".join(f"{s['shot_id']}: {s['image_prompt']} | {s['motion_prompt']}"
                              for s in SHOTS if s.get("motion_prompt")))
free_tier_estimate(sum(1 for s in SHOTS if s.get("motion_prompt")), sum(1 for s in SHOTS if s.get("recipe") in
                   globals().get("DRAWN_RECIPES", {"diagram"})))

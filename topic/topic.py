# ==== 3b. Topic → sources → fact-checked script → licensed images → plan (local models, no paid API) ====
# Each stage is saved with a hash of its inputs (PROJ/stages/): after a disconnect, Run all reuses finished stages.
#   research: Wikipedia articles → numbered facts with URL + revision
#   script:   local LLM (PLANNER_MODEL) writes beats citing fact ids → every sentence fact-checked (numbers, names,
#             NLI entailment) → one rewrite, then dropped; too thin = stop (nothing unsupported is narrated)
#   images:   Wikimedia Commons → licence policy → SigLIP 2 relevance vs distractors → best or a drawn card
#   plan:     built by fixed code; the notebook's validator then checks it again in the next cell
import gc, research, factcheck, planner, assets, registry, stages
STAGE_DIR, ASSET_DIR = PROJ / "stages", PROJ / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)
for _m in (PLANNER_MODEL, factcheck.NLI_MODEL, assets.RANK_MODEL):
    _ok, _why = registry.model_ok(_m, strict=LICENSE_STRICT)
    log(f"licence: {_why}")
    if not _ok:
        raise RuntimeError(f"Model not allowed by the licence registry: {_why}")
WIKI = research.Wiki(WIKI_CONTACT, WORK.parent / "wiki_cache", log=log)
COMMONS = research.Wiki(WIKI_CONTACT, WORK.parent / "commons_cache", log=log, api=assets.COMMONS_API)
DEV = "cuda" if GPU["name"] else "cpu"
HF_CACHE = str(WORK.parent / "hf_planner")          # session disk (Drive space is kept for the video models)


def _free():
    """Models are local to the stage functions: once they return, collecting garbage frees their GPU memory."""
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

print(f"Topic: {TOPIC!r} → project {PROJECT}")
sheet = research.FactSheet.model_validate(stages.stage(STAGE_DIR, "research",
    {"topic": TOPIC, "v": 1}, lambda: research.research(TOPIC, WIKI, n_articles=3, max_facts=40).model_dump(), log))
print(f"Research: {len(sheet.facts)} facts from " + ", ".join(s["title"] for s in sheet.sources))
if len(sheet.facts) < 8:
    raise RuntimeError("Wikipedia has too little on this topic to build a sourced video; try a broader topic.")


if STORY_MODE != "two_friends_cinematic":   # ---- one narrator (the classic topic video) ----
    def _write_script():
        t0 = time.time()
        llm = planner.LocalLLM(PLANNER_MODEL, device=DEV, cache_dir=HF_CACHE, log=log)
        log(f"planner {PLANNER_MODEL} loaded in {time.time() - t0:.0f}s on {DEV}, GPU {gpu_mem() if 'gpu_mem' in globals() else ''}")
        nli = factcheck.NLI(device=DEV, cache_dir=HF_CACHE)
        try:
            script, checks = planner.make_script(TOPIC, sheet, llm, nli, target_s=TARGET_SECONDS, log=log)
        finally:
            del llm, nli; _free()
        return {"script": script, "checks": checks}

    _s = stages.stage(STAGE_DIR, "script", {"facts": [f.text for f in sheet.facts], "model": PLANNER_MODEL,
                                             "target": TARGET_SECONDS, "planner": 3, "nli": factcheck.NLI_MODEL,
                                             "th": factcheck.TH}, _write_script, log)
    SCRIPT, CHECKS = _s["script"], _s["checks"]
    print("\n" + planner.script_text(SCRIPT) + "\n")
    (PROJ / "render").mkdir(exist_ok=True)
    (PROJ / "render" / "script.txt").write_text(planner.script_text(SCRIPT))


    def _find_images():
        beats = SCRIPT["beats"]
        descs = [b["visual"].get("description", "") for b in beats]
        ranker = assets.Ranker(device=DEV, cache_dir=HF_CACHE)
        chosen, tried = [], []
        try:
            for i, b in enumerate(beats):
                v = b["visual"]
                if not USE_LICENSED_IMAGES or v["kind"] != "photo" or not v.get("description"):
                    chosen.append(None); continue
                pick = None
                words = [w for w in re.findall(r"[A-Za-z]{4,}", v["description"])][:4]
                for q in [v["description"], " ".join(words)]:          # refine the query once before giving up
                    cands = [c for c in assets.search(COMMONS, q, n=12, allow_sharealike=ALLOW_SHAREALIKE)]
                    tried += [{k: c[k] for k in ("id", "title", "license", "license_ok", "license_why", "query")} for c in cands]
                    ok = [c for c in cands if c["license_ok"]][:6]
                    files = []
                    for c in ok:
                        try:
                            files.append(assets.download(c, ASSET_DIR, COMMONS.ua))
                        except Exception as e:
                            log(f"image download failed ({c['title']}): {e!r}"); c["license_ok"] = False
                    ok = [c for c in ok if c["license_ok"]]
                    said = " ".join(x["text"] for x in b["narration"])
                    ranked = assets.rank(ranker, files, v["description"], [d for j, d in enumerate(descs) if j != i],
                                         narration=said, other_narration=[" ".join(x["text"] for x in bb["narration"])
                                                                          for j, bb in enumerate(beats) if j != i])
                    for j, own, margin, acc, why in ranked:
                        ok[j].update(score=own, margin=margin, decision="accepted" if acc else "rejected", why=why)
                    best = next((ok[j] for j, _own, _margin, accepted, _why in ranked if accepted), None)
                    if best:
                        best["credit_line"] = assets.credit_line(best); pick = best; break
                log(f"beat {i + 1}: " + (f"image {pick['title']} ({pick['license']}, logit {pick['score']}, margin "
                                         f"{pick['margin']})" if pick else "no licensed image fits → drawn card"))
                chosen.append(pick)
        finally:
            del ranker; _free()
        return {"chosen": chosen, "tried": tried}

    _a = stages.stage(STAGE_DIR, "images", {"script": SCRIPT, "allow_sa": ALLOW_SHAREALIKE, "use": USE_LICENSED_IMAGES,
                                             "th": assets.TH, "model": assets.RANK_MODEL, "v": 2}, _find_images, log)
    CHOSEN = _a["chosen"]
    ASSET_MANIFEST = [a for a in CHOSEN if a]
    (ASSET_DIR / "manifest.json").write_text(json.dumps({"used": ASSET_MANIFEST, "considered": _a["tried"]}, indent=1))
    print(f"Images: {len(ASSET_MANIFEST)} licensed images used, {sum(a is None for a in CHOSEN)} beats as drawn cards")

    TOPIC_PLAN = planner.build_plan(TOPIC, SCRIPT, sheet, CHOSEN, voice=NARRATOR_VOICE, project=PROJECT)
    _said = {l["text"] for s in TOPIC_PLAN["shots"] for l in s["lines"]}
    FACTCHECK = {k: v for k, v in CHECKS.items() if v["text"] in _said}      # exactly the sentences that are narrated
    (PROJ / "render" / "factcheck.json").write_text(json.dumps(CHECKS, indent=1, default=str))
    (PROJ / "plan.json").write_text(json.dumps(TOPIC_PLAN, indent=2, ensure_ascii=False))
    print(f"Plan: {len(TOPIC_PLAN['shots'])} shots, {len(TOPIC_PLAN['claims'])} sourced claims, "
          f"{sum(v['status'] == 'PASS' for v in FACTCHECK.values())}/{len(FACTCHECK)} narrated sentences fact-checked PASS")

else:   # ---- two friends: facts → beats → dialogue → visual intentions → grounding → licensed media → plan ----
    if LANGUAGE != "en":
        # the fact check (NLI) is English-only and the 4B planner's Telugu is unverified: a Telugu story could not be
        # checked against its sources, so it is not written automatically (never a silent switch to English)
        raise RuntimeError(f"STORY_MODE two_friends_cinematic + LANGUAGE '{LANGUAGE}': automatic stories are English-only "
                           "for now (facts can't be verified in Telugu). Write the Telugu story JSON and build it with "
                           "colab/build_story_plan.py, or set LANGUAGE = \"en\".")
    BIBLE = characters.load()

    # one local model for every story stage, loaded only if some stage isn't cached yet, freed at the end
    _LLM, _NLI = [None], [None]
    def _llm():
        if _LLM[0] is None:
            t0 = time.time()
            _LLM[0] = planner.LocalLLM(PLANNER_MODEL, device=DEV, cache_dir=HF_CACHE, log=log)
            log(f"planner {PLANNER_MODEL} loaded in {time.time() - t0:.0f}s on {DEV}")
        return _LLM[0]
    def _nli():
        if _NLI[0] is None:
            _NLI[0] = factcheck.NLI(device=DEV, cache_dir=HF_CACHE)
        return _NLI[0]
    # ---- STORY_BENCH: old one-pass writer vs the staged writer, same facts, same local model, 3 topics; then stop
    STORY_BENCH_TOPICS = ["Why do 3D movies look real?", "Why do we hear thunder after seeing lightning?",
                          "What happens when you book a train ticket?"]       # the last: an existing repository topic
    class StoryBenchDone(Exception):
        """Raised on purpose after the story benchmark so 'Run all' stops before making a video."""
    def run_story_bench():
        out = PROJ / "render" / "story_bench"; out.mkdir(parents=True, exist_ok=True)
        rows, md = [], ["# Story benchmark: old writer vs staged writer (same facts, same model)", "",
                        "Read both scripts of each topic **without looking at which is which first** (A/B is shuffled "
                        "per topic; the key is in story_bench.json), rate 1–5, then compare.", ""]
        import random
        for ti, topic_ in enumerate(STORY_BENCH_TOPICS):
            sh = research.FactSheet.model_validate(stages.stage(STAGE_DIR, f"bench_research_{ti}", {"topic": topic_, "v": 1},
                lambda: research.research(topic_, WIKI, n_articles=3, max_facts=40).model_dump(), log))
            def _both():
                old_st = story.make_story(topic_, sh, _llm(), BIBLE, target_s=TARGET_SECONDS, log=log)
                new_st, rec = story.make_story_v2(topic_, sh, _llm(), BIBLE, target_s=TARGET_SECONDS, log=log,
                                                  n_premises=MAX_STORY_PREMISES, max_revisions=MAX_STORY_REVISIONS)
                return {"old": old_st.model_dump(), "new": new_st.model_dump(), "record": rec}
            r = stages.stage(STAGE_DIR, f"bench_story_{ti}", {"topic": topic_, "facts": [f.text for f in sh.facts],
                             "model": PLANNER_MODEL, "target": TARGET_SECONDS, "v": 1}, _both, log)
            versions = {k: story.Story.model_validate(r[k]) for k in ("old", "new")}
            metrics = {k: story.story_metrics(v, BIBLE) for k, v in versions.items()}
            for k, v in versions.items():               # grounding of every claim (same NLI for both)
                g = story.ground(v, _nli(), topic=topic_)
                metrics[k]["claims_pass"] = f"{sum(x['status'] == 'PASS' for x in g.values())}/{len(g)}"
            order = ["old", "new"]; random.Random(ti * 7919 + len(topic_)).shuffle(order)
            rows.append({"topic": topic_, "A": order[0], "B": order[1], "metrics": metrics,
                         "premise": r["record"]["premise"]})
            md += [f"## {topic_}", ""]
            for lab, k in zip("AB", order):
                md += [f"### Script {lab}", "", "```", story.story_text(versions[k], BIBLE).strip(), "```", ""]
            md += ["| Script | Opening curiosity | Originality | Causal flow | Natural dialogue | Chemistry | Clarity | "
                   "Demonstrations | Factual accuracy | Pacing | Ending payoff | Overall interest |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|", "| A | | | | | | | | | | | |",
                   "| B | | | | | | | | | | | |", ""]
        (out / "story_bench.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str))
        (out / "story_bench.md").write_text("\n".join(md) + "\n")
        for r_ in rows:
            print(f"{r_['topic']}: " + " | ".join(f"{k}: {m['required_weaknesses']} critic issues, claims {m['claims_pass']}, "
                                                    f"Arjun questions {m['curious_question_share']:.0%}"
                                                    for k, m in r_["metrics"].items()))
        raise StoryBenchDone(f"Story benchmark written to {out}/story_bench.md — rate the scripts (blind), then set "
                             "STORY_BENCH = False. The critic numbers inform; people decide.")
    if STORY_BENCH:
        try:
            run_story_bench()
        finally:
            _LLM[0] = _NLI[0] = None; _free()
    _FACTS = story.select_facts(sheet.facts)                                     # A: facts a viewer can follow
    _key = {"facts": [f.text for f in _FACTS], "model": PLANNER_MODEL, "target": TARGET_SECONDS, "story": 2,
            "bible": BIBLE.model_dump()}
    try:
        _pr = stages.stage(STAGE_DIR, "story_premise", {**_key, "n": MAX_STORY_PREMISES},     # B + C
                           lambda: story.stage_premise(TOPIC, _FACTS, _llm(), BIBLE, MAX_STORY_PREMISES, log=log), log)
        PREMISE = story.Premise.model_validate(_pr["premise"])
        print(f"Story premise ({PREMISE.format}): {PREMISE.question}\n  scores: " +
              "; ".join(f"{x['format']} {x['score']}" for x in _pr["scores"]))
        _ol = stages.stage(STAGE_DIR, "story_outline", {**_key, "premise": _pr["premise"]},       # D
                           lambda: story.stage_outline(PREMISE, _FACTS, _llm(), BIBLE, TARGET_SECONDS, log=log), log)
        OUTLINE = story.Outline.model_validate(_ol)
        _dl = stages.stage(STAGE_DIR, "story_dialogue", {**_key, "outline": _ol},                  # E
                           lambda: story.stage_dialogue(TOPIC, PREMISE, OUTLINE, _FACTS, _llm(), BIBLE, TARGET_SECONDS,
                                                        log=log).model_dump(), log)
        def _revise():                                                                          # F + G
            st = story.Story.model_validate(_dl)
            before = story.critique(st, BIBLE, llm=_llm(), log=log, ask_json=planner.ask_json)
            st, revised = story.revise(st, before, _llm(), BIBLE, sheet, log=log, max_revisions=MAX_STORY_REVISIONS)
            return {"story": st.model_dump(), "before": [w.model_dump() for w in before], "revised": revised,
                    "after": [w.model_dump() for w in story.critique(st, BIBLE)]}
        _rv = stages.stage(STAGE_DIR, "story_revision", {**_key, "dialogue": _dl, "rev": MAX_STORY_REVISIONS}, _revise, log)
        def _ground():                                                                          # H: facts + repair
            st, res, rep = story.repair_grounding(story.Story.model_validate(_rv["story"]), _nli(), BIBLE,
                                                  llm=_llm(), log=log)
            return {"story": st.model_dump(), "grounding": res, "repair": rep,
                    "dropped": [x["turn"] for x in rep["removed"] + rep["dependents_removed"]]}
        _s = stages.stage(STAGE_DIR, "story_grounding", {**_key, "revised": _rv["story"], "nli": factcheck.NLI_MODEL,
                                                         "th": factcheck.TH}, _ground, log)
    finally:
        _LLM[0] = _NLI[0] = None; _free()
    print("Critique before revision: " + ("; ".join(f"{w['rule']} {w['where']}: {w['problem']}" for w in _rv["before"])
                                          or "none") + f" | revised: {_rv['revised'] or 'nothing'}")
    (PROJ / "render").mkdir(exist_ok=True)
    (PROJ / "render" / "story_record.json").write_text(json.dumps({"premises": _pr, "outline": _ol, "critique": _rv,
                                                                    "repair": _s["repair"]}, indent=1, default=str))
    STORY = story.Story.model_validate(_s["story"])
    _iss = story.check_story(STORY, BIBLE, target_s=TARGET_SECONDS)
    for _i in _iss:
        log(f"story check {_i['level']} {_i['rule']} {_i['shot']}: {_i['message']}")
    if [i for i in _iss if i["level"] == "error"]:
        raise RuntimeError("The generated story breaks the dialogue rules even after removing unsupported turns:\n"
                           + "\n".join(f"  {i['rule']} {i['shot'] or ''}: {i['message']}" for i in _iss if i["level"] == "error")
                           + "\nDelete the stages/story_*.json files on Drive and Run all to write a new one (or try a broader topic).")
    print("\n" + story.story_text(STORY, BIBLE))
    (PROJ / "render").mkdir(exist_ok=True)
    (PROJ / "render" / "script.txt").write_text(story.story_text(STORY, BIBLE))

    def _find_media():
        """Licensed real images for 'real event' / 'licensed media' visuals (same licence + relevance gates)."""
        want = [(s_, v) for s_ in STORY.scenes for v in s_.visuals if v.intent in ("real_event", "licensed_media")
                and v.description]
        if not (USE_LICENSED_IMAGES and want):
            return {"media": {}, "tried": []}
        ranker = assets.Ranker(device=DEV, cache_dir=HF_CACHE)
        media, tried = {}, []
        try:
            descs = [v.description for _, v in want]
            for i, (s_, v) in enumerate(want):
                cands = assets.search(COMMONS, v.description, n=12, allow_sharealike=ALLOW_SHAREALIKE)
                tried += [{k: c[k] for k in ("id", "title", "license", "license_ok", "license_why", "query")} for c in cands]
                ok = [c for c in cands if c["license_ok"]][:6]
                files = []
                for c in ok:
                    try:
                        files.append(assets.download(c, ASSET_DIR, COMMONS.ua))
                    except Exception as e:
                        log(f"image download failed ({c['title']}): {e!r}"); c["license_ok"] = False
                ok = [c for c in ok if c["license_ok"]]
                said = " ".join(t.text for t in s_.turns)
                ranked = assets.rank(ranker, files, v.description, [d for j, d in enumerate(descs) if j != i], narration=said)
                best = next((ok[j] for j, _o, _m, accepted, _w in ranked if accepted), None)
                if best:
                    media[f"{s_.id}/{v.id}"] = {"file": Path(files[ok.index(best)]).name, "credit": assets.credit_line(best),
                                                "license": best["license"], "id": best["id"], "license_ok": True}
                log(f"{s_.id}/{v.id}: " + (f"licensed image {best['title']} ({best['license']})" if best else
                                         "no licensed image passed the gates → fallback chain"))
        finally:
            del ranker; _free()
        return {"media": media, "tried": tried}

    _a = stages.stage(STAGE_DIR, "story_media", {"story": STORY.model_dump(), "allow_sa": ALLOW_SHAREALIKE,
                                                  "use": USE_LICENSED_IMAGES, "th": assets.TH, "v": 1}, _find_media, log)
    TOPIC_PLAN = story.build_plan(STORY, BIBLE, PROJECT, media=_a["media"], plan_version=1)
    ASSET_MANIFEST = list(_a["media"].values())
    (ASSET_DIR / "manifest.json").write_text(json.dumps({"used": ASSET_MANIFEST, "considered": _a["tried"]}, indent=1))
    FACTCHECK = {k: v for k, v in _s["grounding"].items() if v["status"] == "PASS"}   # what is said: all PASS
    (PROJ / "render" / "factcheck.json").write_text(json.dumps({"grounding": _s["grounding"], "dropped": _s["dropped"]},
                                                               indent=1, default=str))
    (PROJ / "plan.json").write_text(json.dumps(TOPIC_PLAN, indent=2, ensure_ascii=False))
    print(f"Plan: {len(TOPIC_PLAN['shots'])} scenes, {sum(len(s_['lines']) for s_ in TOPIC_PLAN['shots'])} turns, "
          f"{len(TOPIC_PLAN['claims'])} sourced facts, {len(_s['dropped'])} unsupported turns removed, "
          f"{len(ASSET_MANIFEST)} licensed images")

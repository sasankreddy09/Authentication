"""Two-friend story JSON → checked cast plan JSON (then build the notebook with build.py).

Usage:  <python with pydantic> colab/build_story_plan.py colab/stories/<x>_story.json colab/<x>_plan.json \
            --project <drive folder> [--version N] [--target 75]

The story JSON is exactly what the Story Director's LLM stages produce (story.Story); writing it by hand is the
scripted stand-in for those stages. Rule violations (unsourced factual turns, monologues, empty reactions,
uncorrected misconceptions, missing question/closing) stop the build; warnings are printed.
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine" / "lib"))
import characters, story  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("story"); ap.add_argument("plan"); ap.add_argument("--project", required=True)
ap.add_argument("--version", type=int, default=1); ap.add_argument("--target", type=int, default=75)
ap.add_argument("--script-dir", default=str(HERE / "scripts"), help="where the readable script is written")
ap.add_argument("--visual", choices=["storybook_v1"], help="storybook_v1: illustrated rigged characters in "
                "environment packs (asset versions from colab/assets are pinned in the plan); default: the legacy drawing")
ap.add_argument("--char-version", default="v1"); ap.add_argument("--env-version", default="v1")
ap.add_argument("--ground", metavar="HF_CACHE", help="fact check every claim-making turn with the NLI model "
                "(factcheck.NLI_MODEL, needs transformers); anything not PASS stops the build")
a = ap.parse_args()

st = story.Story.model_validate(json.loads(Path(a.story).read_text()))
bible = characters.load()
code_mix = json.loads(Path(a.story).read_text()).get("code_mix")    # Telugu–English mode (None: earlier stories)
if code_mix:                                      # the local voice reads the Telugu-script respelling; subtitles keep text
    import telugu_dialogue as td
    assert code_mix in td.CODE_MIX_MODES, f"code_mix must be one of {list(td.CODE_MIX_MODES)}"
    for s_ in st.scenes:
        for t in s_.turns:
            if t.tts is None:
                say = td.tts_text(t.text, "indic_parler:code_mix")
                t.tts = say if say != t.text else None
issues = story.check_story(st, bible, target_s=a.target)
for i in issues:
    print(f"{i['level'].upper():5} {i['rule']} {i['shot'] or '-'}: {i['message']}" + (f"  → {i['fix']}" if i['fix'] else ""))
if any(i["level"] == "error" for i in issues):
    sys.exit("story rejected: fix the errors above")
if a.ground:                                      # the same check the topic notebook enforces on LLM stories
    import factcheck
    res = story.ground(st, factcheck.NLI(device="cpu", cache_dir=a.ground))
    for k, r in res.items():
        print(f"{r['status']:7} {k}: {r['text'][:70]}" + (f"  ({r['why']}, {r.get('scores')})" if r["status"] != "PASS" else ""))
    sp = Path(a.story)                            # stories/<x>_story.json → stories/<x>_story_grounding.json;
    gp = (HERE / "stories" / (sp.stem + "_grounding.json") if sp.stem != "story"   # stories/<folder>/story.json →
          else sp.parent / "grounding.json")      # next to it (several folders share the file name story.json)
    gp.write_text(json.dumps(res, indent=1, default=str))
    if any(r["status"] != "PASS" for r in res.values()):
        sys.exit("story rejected: some claims are not supported by their cited facts (reword them closer to the source)")
raw = json.loads(Path(a.story).read_text())
extra = {}
if st.language != "en":                           # non-English: Telugu critic + cross-language fact preservation
    import telugu_dialogue as td
    tc = td.critique(st, bible, code_mix=code_mix)
    for w in tc:
        print(f"{'ERROR' if w['required'] else 'WARN '} {w['rule']} {w['where']}: {w['problem']} | {w['evidence'][:70]}")
    if any(w["required"] for w in tc):
        sys.exit("story rejected: fix the Telugu dialogue issues above")
    loc = {}
    for s_ in st.scenes:
        for k, t in enumerate(s_.turns):
            if t.kind in story.FACTUAL_KINDS:
                loc[f"{s_.id}_{k + 1}"] = (td.localization_check(t.text, t.source_text, raw.get("glossary", {}))
                                           if t.source_text else {"status": "UNKNOWN", "why": "no canonical English claim"})
    for k, r in loc.items():
        print(f"{r['status']:8} localization {k}" + ("" if r["status"] == "PASS" else f"  {r}"))
    extra["localization"] = loc
    extra["review"] = [k for k, r in loc.items() if r["status"] != "PASS"]
    if code_mix:
        extra["code_mix"] = code_mix
        left = {f"{s_.id}_{k + 1}": td.latin_left(t.text, "indic_parler:code_mix")
                for s_ in st.scenes for k, t in enumerate(s_.turns)}
        left = {k: v for k, v in left.items() if v}
        for k, v in left.items():
            print(f"WARN  pronunciation {k}: no Parler respelling for {v} (Latin letters reach the voice)")
        extra["pronunciation"] = {"display": "text (subtitles)", "indic_parler": "tts (Telugu-script respelling)",
                                  "status": td.PRONUNCIATION_STATUS, "latin_left": left}
if raw.get("tts_backend"):
    extra["tts_backend"] = raw["tts_backend"]
for k in ("strict_open_source", "voice_policy"):  # optional per-story policies (absent: earlier behaviour)
    if k in raw:
        extra[k] = raw[k]
if a.visual:                                      # pin the exact character / environment asset versions
    import envfactory, storybook as SB
    root = SB.library_root()
    ca = {}
    for cid in bible.characters:
        vdir = root / "characters" / cid / a.char_version
        man = json.loads((vdir / "manifest.json").read_text())
        probs = SB.verify(vdir)
        if probs:
            sys.exit(f"character asset {cid} {a.char_version} is damaged: {probs[:3]}")
        ca[cid] = {"version": a.char_version, "status": man["status"], "art_version": man["art_version"],
                   # pinned byte-for-byte (the notebook carries these exact files: build.storybook_pack)
                   "manifest_sha256": SB.manifest_sha(vdir / "manifest.json"),
                   "content_sha256": SB.content_fingerprint(man)}
    worlds = {s_.world for s_ in st.scenes} | {v.world for s_ in st.scenes for v in s_.visuals if v.world}
    envs = sorted({envfactory.WORLD_ENV[w] for w in worlds if w in envfactory.WORLD_ENV})
    missing = sorted(w for w in worlds if w not in envfactory.WORLD_ENV)
    if missing:
        sys.exit(f"no environment pack for worlds {missing} (envfactory.WORLD_ENV)")
    for e in envs:
        probs = envfactory.verify(root / "environments" / e / a.env_version)
        if probs:
            sys.exit(f"environment {e} {a.env_version}: {probs[:3]}")
    extra.update(visual_style_version=a.visual, character_assets=ca, environments={e: a.env_version for e in envs})
    print(f"visual style {a.visual}: characters {ca}, environments {extra['environments']}")
plan = story.build_plan(st, bible, a.project, plan_version=a.version, extra=extra or None)
if a.visual:                                      # every character visual must be drawable by the storybook renderer
    import charanim
    errs = [f"{sh['shot_id']}/{v['id']}: {m}" for sh in plan["shots"] for v in sh["story"]["visuals"]
            if v["recipe"] == "character" for m in charanim.check_spec(v["character"], plan)]
    if errs:
        sys.exit("storybook visuals rejected:\n  " + "\n  ".join(errs))
Path(a.plan).write_text(json.dumps(plan, indent=1, ensure_ascii=False))
script = Path(a.script_dir) / (Path(a.plan).stem.replace("_plan", "") + "_script.md")
script.parent.mkdir(exist_ok=True)
script.write_text("```\n" + story.story_text(st, bible) + "```\n")
n_words = sum(len(t.text.split()) for s in st.scenes for t in s.turns)
print(f"plan: {len(plan['shots'])} scenes, {sum(len(s['story']['visuals']) for s in plan['shots'])} visuals, "
      f"{n_words} words (~{n_words / story.WORDS_PER_SECOND:.0f}s of speech) → {a.plan}; script → {script}")

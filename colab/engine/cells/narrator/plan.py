# ==== 4. The video plan (written by Claude; edit plan.json on Drive to change it) ====
PLAN_EMBEDDED = json.loads(r"""
__PLAN_JSON__
""")

plan_path = PROJ / "plan.json"
PLAN = PLAN_EMBEDDED
if plan_path.exists():
    on_drive = json.loads(plan_path.read_text())
    if on_drive.get("plan_version", 1) >= PLAN_EMBEDDED["plan_version"]:
        PLAN = on_drive
        print("Using plan.json from Drive (your edits). Delete it to restore the built-in plan.")
    else:
        backup = PROJ / f"plan.v{on_drive.get('plan_version', 1)}.backup.json"
        plan_path.rename(backup)
        print(f"Found an older plan on Drive; kept it as {backup.name} and switched to the new built-in plan.")
if PLAN is PLAN_EMBEDDED:
    plan_path.write_text(json.dumps(PLAN, indent=2, ensure_ascii=False))

STYLE, NEG, SHOTS, MOODS = PLAN["style"]["suffix"], PLAN["style"]["negative"], PLAN["shots"], PLAN["moods"]
SFX_TYPES = {"boom", "rumble", "whoosh", "shimmer", "ding", "click", "pop", "coins"}
# names and special words for the speech check (otherwise it may hear "Zomato" as "the motto")
ASR_PROMPT = ", ".join(PLAN.get("vocabulary", []))
def mood(shot):
    return MOODS[shot.get("mood", "calm")]

# checks: the plan must only use things this notebook knows how to make
assert len({s["shot_id"] for s in SHOTS}) == len(SHOTS), "duplicate shot ids"
for s in SHOTS:
    assert s["recipe"] in ("video", "title_card", "still_move", "i2v_clip"), f"{s['shot_id']}: unknown recipe {s['recipe']}"
    assert s.get("mood", "calm") in MOODS, f"{s['shot_id']}: unknown mood {s.get('mood')}"
    for c in s.get("sfx", []):
        assert c["type"] in SFX_TYPES, f"{s['shot_id']}: unknown sound effect {c['type']}"
n_words = sum(len(x["text"].split()) for s in SHOTS for x in s["sentences"])
print(f"{len(SHOTS)} shots, {n_words} words of narration")
print("Story arc:", " → ".join(s.get("mood", "calm") for s in SHOTS))
for a, b in zip(SHOTS, SHOTS[1:]):
    if a.get("move") == b.get("move") and a.get("mood") == b.get("mood"):
        print(f"  note: {a['shot_id']} and {b['shot_id']} have the same move and mood (could feel repetitive)")
free_tier_estimate(sum(1 for s in SHOTS if s.get("motion_prompt")), sum(1 for s in SHOTS if s.get("recipe") in
                   globals().get("DRAWN_RECIPES", {"diagram"})))

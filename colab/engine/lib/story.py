"""Two-friend storytelling: Story Director + Visual Story Director (fixed code; the LLM only writes data).

Pipeline (each stage small, structured, validated with Pydantic, retried with the error fed back):
    facts (research.FactSheet)  →  beats  →  dialogue turns (each factual turn cites fact ids)
    →  visual intentions (per turn, independent of the turns' boundaries)  →  checks  →  plan (cast format)
The plan is the existing cast plan with one new recipe, "story": a SHOT is a dialogue scene (its lines are the
audio units, placed narration-first as before) and `story.visuals` is that scene's own VISUAL timeline: each
visual starts at a point inside a turn ({"line": k, "at": 0..1}), so one turn can span a cutaway and a cut need
not happen at a line boundary. The audio track never depends on the visuals.

Visual Story Director: maps each visual intention to the cheapest medium that is faithful, with an explicit
fallback chain — character scene → licensed media → animated demonstration → AI video (only when allowed and
within budget) → minimal graphic. Every decision is recorded (medium, why, fallback trail), and two independent
statistics are reported: NARRATIVE PURPOSE (what a visual is for) and RENDERING MEDIUM (how it was made).
"""
from __future__ import annotations

import json, re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

import characters as charbible

BEATS = ("observation", "question", "first_answer", "demonstration", "misconception", "correction", "reveal",
         "real_world", "closing")
TURN_KINDS = ("fact", "question", "analogy", "reaction", "misconception", "correction", "instruction", "observation",
              "prediction", "challenge", "hypothetical", "opinion")   # the last four: Arjun is more than a questioner
CLAIM_FREE = ("reaction", "instruction", "observation", "misconception", "prediction", "opinion")  # not NLI-checked
# conversational intentions (voice_direction.INTENTS): what the line is FOR, used by the voice performance director
TURN_INTENTS = ("genuine_curiosity", "thinking_aloud", "playful_disbelief", "gentle_correction", "explaining_observation",
                "asking_confirmation", "quiet_realization", "enthusiastic_discovery", "sharing_surprising_fact",
                "offering_reassurance", "reflective_explanation", "satisfying_revelation", "guiding_demo", "playful_remark")
FACTUAL_KINDS = ("fact", "correction")
TONES = ("curious", "explainer", "warm", "happy", "friendly", "awe", "serious")
INTENTS = ("establishing", "two_friend", "closeup", "ots", "reaction", "real_event", "object_demo", "cutaway",
           "reenactment", "analogy", "licensed_media", "animated_illustration", "minimal_overlay")
MEDIA = ("character_scene", "licensed_media", "animated_demo", "ai_video", "presentation_graphic")
PRESENTATION = ("minimal_overlay",)                       # intents that are "slides" rather than scenes
# shot-level beat used by the existing engine (voice direction BEAT_ROLE, validator S7)
ENGINE_BEAT = {"observation": "hook", "question": "turn", "first_answer": "setup", "demonstration": "step",
               "misconception": "turn", "correction": "resolution", "reveal": "reveal", "real_world": "step",
               "closing": "payoff"}
MAX_TURN_WORDS = 30
MIN_FACT_TURNS = 2                       # D8: at least this many sourced factual turns per story
MAX_SENTENCE_WORDS = 22                  # = validate.py S4
WORDS_PER_SECOND = 2.6


# ------------------------------------------------------------------ the data
class Turn(BaseModel):
    speaker: str
    text: str
    kind: Literal[TURN_KINDS] = "observation"
    fact_ids: list[str] = Field(default_factory=list)
    tone: Literal[TONES] = "friendly"
    tts: Optional[str] = None
    source_text: Optional[str] = None        # non-English turns: the English wording it translates (fact-checked)
    intent: Optional[Literal[TURN_INTENTS]] = None   # why the character says it (else derived from the context)
    expr: Optional[str] = None               # speaker's expression while saying it
    gesture: Optional[str] = None

    @field_validator("text")
    @classmethod
    def short(cls, v):
        v = v.strip()
        if not v or len(v.split()) > MAX_TURN_WORDS:
            raise ValueError(f"each turn must have 1–{MAX_TURN_WORDS} words (a friend, not a lecture)")
        for sent in re.split(r"(?<=[.!?])\s+", v):           # the plan validator's S4 limit, enforced early
            if len(sent.split()) > MAX_SENTENCE_WORDS:
                raise ValueError(f"sentence of {len(sent.split())} words (max {MAX_SENTENCE_WORDS}): split it")
        return v


class Visual(BaseModel):
    id: str
    intent: Literal[INTENTS]
    turn: int = 0                            # index of the turn (in this scene) during which the visual starts
    at: float = Field(0.0, ge=0.0, le=1.0)   # how far into that turn
    subject: Optional[str] = None            # character the shot is about (closeup / ots / reaction)
    template: Optional[str] = None           # demonstration template (demos.TEMPLATES)
    description: str = ""                    # what should be seen (concrete nouns: also the licensed-media query)
    lines: list[str] = Field(default_factory=list)     # minimal overlay text
    chars: dict = Field(default_factory=dict)          # per character: expr / gesture / wear / hold
    changes: list[dict] = Field(default_factory=list)
    screen: Optional[str] = None
    shot: Optional[str] = None               # explicit camera shot for character scenes
    world: Optional[str] = None              # another place than the scene's (e.g. the screen they are watching)
    camera: Optional[dict] = None            # storybook plans: {"movement": "static|slow_push_in|slow_pull_back|pan_*"}
    environment_variant: Optional[str] = None   # storybook plans: a camera preset of the environment pack


class Scene(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,15}$")
    beat: Literal[BEATS]
    world: str
    turns: list[Turn] = Field(min_length=1, max_length=6)
    visuals: list[Visual] = Field(min_length=1)
    transition: Optional[str] = None         # "time_skip" / "cut" when the friends change place
    sfx: list[dict] = Field(default_factory=list)      # optional sparse sound cues {"type", "at" s, "dur", "gain_db"}

    @model_validator(mode="after")
    def visuals_inside(self):
        prev = (-1, -1.0)
        for v in self.visuals:
            if not 0 <= v.turn < len(self.turns):
                raise ValueError(f"scene {self.id}: visual {v.id} starts in turn {v.turn}, which doesn't exist")
            if (v.turn, v.at) <= prev:
                raise ValueError(f"scene {self.id}: visuals must start in order (visual {v.id})")
            prev = (v.turn, v.at)
        if (self.visuals[0].turn, self.visuals[0].at) != (0, 0.0):
            raise ValueError(f"scene {self.id}: the first visual must start with the first turn (turn 0, at 0)")
        return self


class StoryFact(BaseModel):
    id: str
    text: str
    source_url: str
    title: str = ""
    revid: Optional[int] = None


class Story(BaseModel):
    title: str
    topic: str
    language: str = "en"
    logline: str = ""
    facts: list[StoryFact]
    scenes: list[Scene] = Field(min_length=3, max_length=14)
    vocabulary: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ LLM stages (small, structured)
class BeatOutline(BaseModel):
    title: str
    beats: list[dict] = Field(min_length=4, max_length=10)       # {"beat": BEATS, "goal": str, "fact_ids": [...]}

    @field_validator("beats")
    @classmethod
    def known(cls, v):
        for b in v:
            if b.get("beat") not in BEATS:
                raise ValueError(f"unknown beat {b.get('beat')}; use one of {BEATS}")
        return v


class SceneDialogue(BaseModel):
    world: str
    turns: list[Turn] = Field(min_length=1, max_length=6)


class SceneVisuals(BaseModel):
    visuals: list[Visual] = Field(min_length=1, max_length=4)


SYSTEM = """You write short educational scenes where two friends talk.
{curious} ({curious_p}) notices things and asks; {expert} ({expert_p}) explains with everyday examples and small
demonstrations. Rules:
- Use ONLY the numbered facts given. A turn of kind "fact" or "correction" must cite the facts that support it.
- Never add a number, name, date or claim that is not in the cited facts.
- Natural, short turns (at most 25 words). No lectures, no "Wow!"-only replies, no mocking.
- {curious} may guess wrong (kind "misconception"); {expert} then corrects it with sourced facts (kind "correction").
- Answer with ONE JSON object and nothing else."""


def system_prompt(bible):
    a, b = bible.by_role("curious"), bible.by_role("expert")
    return SYSTEM.format(curious=a.name, curious_p=a.personality, expert=b.name, expert_p=b.personality)


def beats_prompt(topic, facts, target_s):
    lines = "\n".join(f"{f.id}: {f.text}" for f in facts)
    return (f"Topic: {topic}\nTarget: about {target_s} seconds.\nPlan 5–8 beats from: {', '.join(BEATS)}. Start with "
            "an everyday observation, then the question. A misconception + correction is optional. End with closing.\n"
            f"Facts:\n{lines}\n\nReturn: {{\"title\": \"max 8 words\", \"beats\": [{{\"beat\": \"observation\", "
            "\"goal\": \"what the viewer learns or feels\", \"fact_ids\": [\"F1\"]}}]}}")


def dialogue_prompt(bible, beat, facts, words, worlds):
    a, b = bible.by_role("curious"), bible.by_role("expert")
    lines = "\n".join(f"{f.id}: {f.text}" for f in facts)
    return (f"Beat: {beat['beat']} — {beat['goal']}\nAbout {words} spoken words in 2–4 turns.\nSpeakers: \"{a.id}\" "
            f"({a.name}), \"{b.id}\" ({b.name}).\nPlace (world): one of {', '.join(worlds)}.\nFacts:\n{lines}\n\nReturn: "
            "{\"world\": \"...\", \"turns\": [{\"speaker\": \"" + a.id + "\", \"text\": \"...\", \"kind\": \"question\", "
            "\"fact_ids\": [], \"tone\": \"curious\"}]}\nkinds: " + ", ".join(TURN_KINDS) + "; tones: " + ", ".join(TONES))


def visuals_prompt(scene_turns, templates):
    t = "\n".join(f"{i}: {x.speaker}: {x.text}" for i, x in enumerate(scene_turns))
    return (f"Turns:\n{t}\n\nChoose 1–3 visuals that SHOW what is being said (not slides). Intents: {', '.join(INTENTS)}. "
            f"Demonstration templates: {', '.join(templates)}. A visual starts during a turn (\"turn\": index, \"at\": "
            "0..1). The first starts at turn 0, at 0. Prefer the friends, real objects and demonstrations; use "
            "\"minimal_overlay\" only for a key number or term.\nReturn: {\"visuals\": [{\"id\": \"v1\", \"intent\": "
            "\"two_friend\", \"turn\": 0, \"at\": 0.0, \"description\": \"...\"}]}")


def make_story(topic, sheet, llm, bible, target_s=75, language="en", log=print, max_facts=24, ask_json=None):
    """facts → beats → dialogue per beat → visual intentions per scene. Returns a validated Story (not yet checked
    against the rules: call check_story / ground)."""
    import demos
    if ask_json is None:
        from planner import ask_json
    facts = sheet.facts[:max_facts]
    sysp = system_prompt(bible)
    outline = ask_json(llm, sysp, beats_prompt(topic, facts, target_s), BeatOutline, log=log)
    worlds = ("living_room", "cinema_seats", "cinema_screen", "lobby", "studio")
    per = max(int(target_s * WORDS_PER_SECOND / len(outline.beats)), 12)
    by_id = {f.id: f for f in facts}
    scenes = []
    for k, b in enumerate(outline.beats):
        rel = [by_id[i] for i in b.get("fact_ids", []) if i in by_id] or facts[:6]
        d = ask_json(llm, sysp, dialogue_prompt(bible, b, rel, per, worlds), SceneDialogue, log=log)
        v = ask_json(llm, sysp, visuals_prompt(d.turns, demos.TEMPLATES), SceneVisuals, log=log)
        scenes.append(Scene(id=f"sc{k + 1:02d}", beat=b["beat"], world=d.world if d.world in worlds else "studio",
                            turns=d.turns, visuals=v.visuals))
    used = {i for s in scenes for t in s.turns for i in t.fact_ids}
    return Story(title=outline.title, topic=topic, language=language,
                 facts=[StoryFact(id=f.id, text=f.text, source_url=f.source_url, title=f.title, revid=f.revid)
                        for f in facts if f.id in used], scenes=scenes)


# ------------------------------------------------------------------ rule checks (no model)
def _issue(level, rule, scene, msg, fix=""):
    return {"level": level, "rule": rule, "shot": scene, "message": msg, "fix": fix}


EMPTY_REACTIONS = re.compile(r"^\s*(wow|cool|oh|okay|ok|nice|amazing|really|great|whoa|i see)[\s!.?,]*$", re.I)


def check_story(story: Story, bible, target_s=None):
    """Dialogue quality, grounding structure, beat order and visual-variety rules. Returns validate-style issues."""
    out = []
    ids = set(bible.characters)
    cur, exp = bible.by_role("curious").id, bible.by_role("expert").id
    fids = {f.id for f in story.facts}
    turns = [(s, t) for s in story.scenes for t in s.turns]
    for s, t in turns:
        if t.speaker not in ids:
            out.append(_issue("error", "D0", s.id, f"unknown speaker '{t.speaker}'", f"use {sorted(ids)}"))
        for f in t.fact_ids:
            if f not in fids:
                out.append(_issue("error", "D1", s.id, f"turn cites unknown fact '{f}'"))
        if t.kind in FACTUAL_KINDS and not t.fact_ids:
            out.append(_issue("error", "D1", s.id, f"factual turn cites no fact: '{t.text[:60]}'",
                              "cite the fact ids it states, or make it a question/reaction/analogy"))
        if t.kind == "reaction" and EMPTY_REACTIONS.match(t.text):
            out.append(_issue("error", "D3", s.id, f"empty reaction '{t.text}'",
                              "react with a real follow-up thought or question"))
    said = []
    for s, t in turns:                                    # the viewer must not hear the same fact twice
        if t.kind not in FACTUAL_KINDS:
            continue
        w = set(re.findall(r"[a-z]{4,}", t.text.lower()))
        for s0, w0 in said:
            if len(w & w0) / max(len(w | w0), 1) >= 0.8:
                out.append(_issue("error", "D7", s.id, f"repeats what was already said in {s0}: '{t.text[:60]}'",
                                  "say something new, or ask/react instead"))
                break
        said.append((s.id, w))
    for i, (s, t) in enumerate(turns):
        if t.kind == "misconception":
            # corrected by the other friend within the next 4 turns (a test may come first), with sourced facts
            nxt = [x for _, x in turns[i + 1:i + 5]]
            if not any(x.kind in FACTUAL_KINDS and x.speaker != t.speaker for x in nxt):
                out.append(_issue("error", "D4", s.id, "a misconception is not corrected within the next four turns",
                                  "the other friend corrects it with a sourced 'correction' or 'fact' turn"))
    run, prev = 0, None
    for s, t in turns:
        run = run + 1 if t.speaker == prev else 1
        prev = t.speaker
        if run > 2:
            out.append(_issue("error", "D2", s.id, f"{run} turns in a row by the same friend (a monologue)",
                              "let the other friend ask or react in between"))
    words = {c: sum(len(t.text.split()) for _, t in turns if t.speaker == c) for c in ids}
    total = max(sum(words.values()), 1)
    if words.get(exp, 0) / total > 0.78:
        out.append(_issue("warn", "D2", None, f"{bible.characters[exp].name} speaks {words[exp] / total:.0%} of the words",
                          "give the curious friend more real questions and guesses"))
    qs = sum(t.kind == "question" and t.speaker == cur for _, t in turns)
    if qs < 2:
        out.append(_issue("error", "D3", None, f"the curious friend asks only {qs} question(s)", "at least 2 real questions"))
    n_facts = sum(t.kind in FACTUAL_KINDS and bool(t.fact_ids) for _, t in turns)
    if n_facts < MIN_FACT_TURNS:                          # an explainer that explains nothing is not a video we ship
        out.append(_issue("error", "D8", None, f"only {n_facts} sourced factual turn(s) (min {MIN_FACT_TURNS})",
                          "let the expert friend explain with the cited facts"))
    beats = [s.beat for s in story.scenes]
    if "question" not in beats:
        out.append(_issue("error", "D5", None, "no scene asks the central question (beat 'question')"))
    if beats[-1] != "closing":
        out.append(_issue("error", "D5", story.scenes[-1].id, "the story must end with a 'closing' scene"))
    first_q = next((i for i, b in enumerate(beats) if b == "question"), 99)
    first_fact = next((i for i, (s, t) in enumerate(turns) if t.kind in FACTUAL_KINDS), 99)
    first_q_turn = sum(len(s.turns) for s in story.scenes[:first_q])
    if first_fact < first_q_turn:
        out.append(_issue("warn", "D5", None, "a fact is explained before the question is asked",
                          "observation → question → answer"))
    if target_s:
        est = total / WORDS_PER_SECOND + 0.6 * len(turns)
        if not 0.7 * target_s <= est <= 1.35 * target_s:
            out.append(_issue("warn", "D6", None, f"estimated length {est:.0f}s vs target {target_s}s",
                              "add or remove a beat"))
    # visual rules (before the director: the intentions themselves)
    vis = [(s, v) for s in story.scenes for v in s.visuals]
    pres = sum(v.intent in PRESENTATION for _, v in vis)
    if pres / max(len(vis), 1) > 0.2:
        out.append(_issue("warn", "V5", None, f"{pres}/{len(vis)} visuals are text overlays (presentation graphics)",
                          "show the friends, objects or a demonstration instead"))
    same, prevk = 1, None
    for s, v in vis:
        key = (v.intent, v.shot, s.world)
        same = same + 1 if key == prevk else 1
        prevk = key
        if same >= 3:
            out.append(_issue("warn", "V6", s.id, f"{same} visuals in a row are the same ({v.intent} in {s.world})",
                              "change the angle, use a reaction or a cutaway"))
    for s, v in vis:
        if v.intent == "reenactment" and not v.description:
            out.append(_issue("warn", "V7", s.id, f"reenactment {v.id} has no description of what it re-creates"))
    return out


def ground(story: Story, nli, topic=""):
    """Fact check every factual turn against the facts it cites (factcheck.check_sentences: numbers, names, NLI).
    Analogies must be worded as comparisons; questions make no claim. Returns {turn key: result}."""
    import factcheck
    by_id = {f.id: f.text for f in story.facts}
    items = []
    for s in story.scenes:
        for k, t in enumerate(s.turns):
            if t.kind in CLAIM_FREE:
                continue                                    # no claim (a misconception is corrected by D4)
            # the NLI model reads English: a translated turn is checked through the English wording it translates
            items.append({"id": f"{s.id}_{k + 1}", "text": t.source_text or t.text, "evidence": [by_id[i] for i in t.fact_ids if i in by_id],
                          "question": t.kind in ("question", "challenge"), "analogy": t.kind in ("analogy", "hypothetical")})
    return factcheck.check_sentences(items, nli, topic=topic)


def enforce_grounding(st: Story, nli, bible, log=print):
    """Fact check, then DROP every claim-making turn that is not PASS (nothing unsupported is said). Visuals that
    started in a dropped turn move to the next surviving turn. Returns (story, results, dropped keys)."""
    res = ground(st, nli, topic=st.topic)
    bad = {k for k, r in res.items() if r["status"] != "PASS"}
    if not bad:
        return st, res, []
    scenes = []
    for s in st.scenes:
        keep = [k for k in range(len(s.turns)) if f"{s.id}_{k + 1}" not in bad]
        if not keep:
            log(f"story: scene {s.id} dropped (none of its turns is supported)")
            continue
        new_of = {}
        for k in range(len(s.turns)):                    # dropped turn → the next kept one (or the last kept)
            new_of[k] = next((i for i, kk in enumerate(keep) if kk >= k), len(keep) - 1)
        vis, seen = [], set()
        for v in s.visuals:
            nv = v.model_copy(update={"turn": new_of[v.turn], "at": v.at if v.turn in keep else 0.0})
            if (nv.turn, nv.at) in seen or (vis and (nv.turn, nv.at) <= (vis[-1].turn, vis[-1].at)):
                continue
            seen.add((nv.turn, nv.at)); vis.append(nv)
        vis[0] = vis[0].model_copy(update={"turn": 0, "at": 0.0})
        scenes.append(s.model_copy(update={"turns": [s.turns[k] for k in keep], "visuals": vis}))
    for k in sorted(bad):
        log(f"story: dropped unsupported turn {k}: {res[k]['text'][:80]!r} ({res[k]['why']})")
    out = Story.model_validate(st.model_copy(update={"scenes": scenes}).model_dump())
    return out, res, sorted(bad)


# ------------------------------------------------------------------ Visual Story Director
INTENT_SHOT = {"establishing": "establishing", "two_friend": "two_shot", "closeup": "closeup", "ots": "ots",
               "reaction": "reaction", "reenactment": "wide", "cutaway": "wide"}
DEMO_FOR = {"object_demo", "animated_illustration", "analogy"}
MEDIA_FOR = {"real_event", "licensed_media"}


ILLUSTRATION = {"en": "Illustration", "te": "దృష్టాంతం"}     # on-screen tag of a re-enactment, per language


def direct(story: Story, bible, media=None, allow_ai=False, ai_budget=0):
    """Choose how every visual is made. media: {visual id: {"file", "credit", "license"}} for licensed images that
    passed the licence + relevance gates. Returns {scene id: [visual spec for the plan]} and the decisions."""
    import demos
    media = media or {}
    plans, decisions = {}, []
    ai_used = 0
    for s in story.scenes:
        out = []
        speakers = [t.speaker for t in s.turns]
        present = list(dict.fromkeys([bible.by_role("curious").id, bible.by_role("expert").id]))
        for v in s.visuals:
            trail, rec, spec, medium, label = [], None, None, None, None
            want = v.intent
            if want in MEDIA_FOR:
                m = media.get(f"{s.id}/{v.id}") or media.get(v.id)      # "scene/visual" (ids repeat per scene)
                if m:
                    rec, medium = "photo", "licensed_media"
                    spec = {"file": m["file"], "credit": m["credit"], "move": "push_in"}
                else:
                    trail.append("licensed media: none passed the licence/relevance gates")
                    if want == "real_event" and allow_ai and ai_used < ai_budget:
                        trail.append("AI video: allowed, but AI cutaways inside a dialogue scene are not built yet")
                    want = "object_demo" if v.template else "reenactment" if want == "real_event" else "two_friend"
            if rec is None and want in DEMO_FOR:
                if v.template in demos.TEMPLATES:
                    rec, medium = "demo", "animated_demo"
                    spec = {"template": v.template, "world": v.world or s.world}
                else:
                    trail.append(f"demonstration: no template for '{v.description[:40]}'")
                    want = "two_friend"
            if rec is None and want == "minimal_overlay":
                lines = [ln for ln in v.lines if len(ln.split()) <= 6][:3]
                if lines:
                    rec, medium = "card", "presentation_graphic"
                    spec = {"lines": lines, "headline": True, "reveal": len(lines) > 1}
                else:
                    trail.append("overlay: no short text given")
                    want = "two_friend"
            if rec is None:                                   # character scene (the default, never fails)
                rec, medium = "character", "character_scene"
                shot = v.shot or INTENT_SHOT.get(want, "two_shot")
                subj = v.subject
                if shot in ("closeup", "ots") and subj is None:
                    subj = speakers[min(v.turn, len(speakers) - 1)]
                if shot == "reaction" and subj is None:          # the one listening
                    sp = speakers[min(v.turn, len(speakers) - 1)]
                    subj = next(c for c in present if c != sp)
                chars = {c: dict(v.chars.get(c, {})) for c in present}
                spec = {"world": v.world or s.world, "shot": shot, "chars": chars}
                if subj:
                    spec["subject"] = subj
                if v.screen:
                    spec["screen"] = v.screen
                if v.changes:
                    spec["changes"] = v.changes
                if v.intent == "reenactment" or (v.intent == "real_event" and medium == "character_scene"):
                    label = ILLUSTRATION.get(story.language, "Illustration")   # never shown as documentary footage
                    spec["label"] = label
                if shot in ("closeup", "reaction"):
                    spec["push"] = True
                if v.camera:
                    spec["camera"] = dict(v.camera)
                if v.environment_variant:
                    spec["environment_variant"] = v.environment_variant
            out.append({"id": v.id, "intent": v.intent, "medium": medium, "recipe": rec, rec: spec,
                        "start": {"line": v.turn, "at": v.at}, "description": v.description,
                        **({"label": label} if label else {}), **({"fallback_trail": trail} if trail else {})})
            decisions.append({"scene": s.id, "visual": v.id, "intent": v.intent, "medium": medium, "recipe": rec,
                              "trail": trail})
        plans[s.id] = out
    return plans, decisions


def statistics(rows):
    """Two independent statistics over visual screen time. rows: [{"intent", "medium", "dur"}]."""
    tot = sum(r["dur"] for r in rows) or 1.0
    purpose, medium = {}, {}
    for r in rows:
        purpose[r["intent"]] = purpose.get(r["intent"], 0) + r["dur"]
        medium[r["medium"]] = medium.get(r["medium"], 0) + r["dur"]
    pct = lambda d: {k: round(100 * v / tot, 1) for k, v in sorted(d.items(), key=lambda kv: -kv[1])}
    return {"total_s": round(tot, 2), "narrative_purpose_pct": pct(purpose), "rendering_medium_pct": pct(medium),
            "presentation_graphics_pct": round(100 * medium.get("presentation_graphic", 0) / tot, 1)}


# ------------------------------------------------------------------ plan (existing cast format + "story" shots)
TONE_TABLE = {"curious": {"speed": 0.98, "pitch": 0.0, "gain_db": 0.0, "pause_after": 0.45},
              "explainer": {"speed": 1.0, "pitch": 0.0, "gain_db": 0.0, "pause_after": 0.4},
              "warm": {"speed": 0.97, "pitch": 0.0, "gain_db": 0.0, "pause_after": 0.5},
              "happy": {"speed": 1.0, "pitch": 0.0, "gain_db": 0.0, "pause_after": 0.4},
              "friendly": {"speed": 1.0, "pitch": 0.0, "gain_db": 0.0, "pause_after": 0.35},
              "awe": {"speed": 0.96, "pitch": 0.0, "gain_db": 0.0, "pause_after": 0.6},
              "serious": {"speed": 0.97, "pitch": 0.0, "gain_db": 0.0, "pause_after": 0.5}}
MOOD_FOR = {"observation": "curious", "question": "curious", "first_answer": "calm", "demonstration": "busy",
            "misconception": "curious", "correction": "calm", "reveal": "joy", "real_world": "calm", "closing": "joy"}
MOODS = {"curious": {"pause_after": 0.5, "grade": "neutral", "move_amount": 0.04,
                     "music": {"base": 0.5, "curious": 0.7, "tension": 0.1, "shimmer": 0.3}},
         "calm": {"pause_after": 0.45, "grade": "neutral", "move_amount": 0.04,
                  "music": {"base": 0.4, "curious": 0.8, "tension": 0.0, "shimmer": 0.2}},
         "busy": {"pause_after": 0.4, "grade": "neutral", "move_amount": 0.05,
                  "music": {"base": 0.5, "curious": 0.6, "tension": 0.2, "shimmer": 0.1}},
         "joy": {"pause_after": 0.45, "grade": "neutral", "move_amount": 0.05,
                 "music": {"base": 0.3, "curious": 1.0, "tension": 0.0, "shimmer": 0.4}}}


def build_plan(story: Story, bible, project, media=None, allow_ai=False, ai_budget=0, plan_version=1, extra=None):
    """Story → cast-format plan the notebook runs (validated again there). Facts become sourced claims; each
    factual line cites them; the visuals carry their intent and medium for the QC statistics."""
    visuals, decisions = direct(story, bible, media, allow_ai, ai_budget)
    claim_of = {f.id: f"c_{f.id.lower()}" for f in story.facts}
    chars = charbible.cast_characters(bible, story.language)
    shots = []
    for s in story.scenes:
        lines, cited = [], []
        for t in s.turns:
            ln = {"speaker": t.speaker, "tone": t.tone, "text": t.text, "kind": t.kind}
            tts = t.tts or charbible.speak_text(t.text, bible.characters[t.speaker].voices[story.language]) \
                if story.language in bible.characters[t.speaker].voices else t.tts
            if tts:
                ln["tts"] = tts                          # pronunciation profile (e.g. 3D → త్రీడీ); subtitles keep text
            if t.intent:
                ln["intent"] = t.intent
            if t.fact_ids:
                ln["claims"] = [claim_of[f] for f in t.fact_ids if f in claim_of]
                cited += ln["claims"]
            if t.expr or t.gesture:
                ln["act"] = {k: v for k, v in (("expr", t.expr), ("gesture", t.gesture)) if v}
            lines.append(ln)
        vis = visuals[s.id]
        _apply_acting(vis, s, bible)
        shots.append({"shot_id": s.id, "recipe": "story", "beat": ENGINE_BEAT[s.beat], "story_beat": s.beat,
                      "mood": MOOD_FOR[s.beat], "location": s.world, "cast": list(bible.characters),
                      "claims": list(dict.fromkeys(cited)), "lines": lines,
                      **({"transition": s.transition} if s.transition else {}),
                      **({"sfx": s.sfx} if s.sfx else {}),
                      "story": {"world": s.world, "visuals": vis}})
    plan = {
        "plan_version": plan_version, "project": project, "title": story.title, "language": story.language,
        "story_mode": "two_friends_cinematic", "visual_style": "stylized_2d", "license_strict": True,
        "factual_dialogue": True, "logline": story.logline,
        "notebook": {"emoji": "🎬", "name": "Two-Friend Explainer", "length": "about 1–1.5 min"},
        "style": {"suffix": "stylized 2D animation, flat colours, two original cartoon friends", "negative":
                  "text, watermark, photorealistic, deformed, blurry"},
        "look": {"bg_top": "#1B2440", "bg_bottom": "#28406A"},
        "characters": chars, "bible": bible.model_dump(), "tones": dict(TONE_TABLE), "moods": dict(MOODS),
        "claims": [{"id": claim_of[f.id], "text": f.text, "sources": [f.source_url],
                    **({"source_revision": f.revid} if f.revid else {})} for f in story.facts],
        "vocabulary": list(dict.fromkeys([c.name for c in bible.characters.values()] + story.vocabulary)),
        "entities": {}, "glossary": {}, "travel": {}, "shots": shots, "director": decisions,
        "music": {"cues": _music_cues(story)},
        "metadata": _metadata(story, bible),
    }
    if extra:
        plan.update(extra)
    return plan


MUSIC_FOR = {"observation": "Investigations", "demonstration": "Easy Lemon", "reveal": "Carefree"}


def _music_cues(story):
    """One licensed track per chapter: curious at the start, light during the demonstration, bright at the reveal."""
    cues, last = [], None
    for s in story.scenes:
        tr = MUSIC_FOR.get(s.beat)
        if tr and tr != last:
            cues.append({"from": s.id, "track": tr}); last = tr
    if not cues or cues[0]["from"] != story.scenes[0].id:
        cues.insert(0, {"from": story.scenes[0].id, "track": "Investigations"})
    return cues


def _metadata(story, bible):
    names = " and ".join(c.name for c in bible.characters.values())
    titles = sorted({f.title for f in story.facts if f.title})
    disclaimer = (f"Simplified explainer based on Wikipedia ({', '.join(titles)}). {names} are fictional, "
                  "original animated characters; scenes marked 'Illustration' are re-enactments, not footage.")
    if story.language == "te":
        tnames = " మరియు ".join(c.names.get("te", c.name) for c in bible.characters.values())
        disclaimer = (f"వికీపీడియా ({', '.join(titles)}) ఆధారంగా సరళంగా వివరించిన వీడియో. {tnames} కల్పిత, సొంతంగా "
                      "గీసిన యానిమేషన్ పాత్రలు; 'దృష్టాంతం' అని ఉన్న దృశ్యాలు పునర్నిర్మాణాలు, నిజమైన ఫుటేజ్ కాదు.\n" + disclaimer)
    return {"title": story.title,
            "summary": story.logline or f"{names} work out: {story.topic}",
            "disclaimer": disclaimer,
            "category": "Education", "audience": "Not made for kids",
            "tags": list(dict.fromkeys([story.topic.lower().rstrip("?"), "science explained", "explainer"]
                                       + [v.lower() for v in story.vocabulary]))[:12]}


def _apply_acting(vis, scene, bible):
    """Turn-level acting (expression/gesture of the speaker) becomes timed 'changes' in the character visuals
    that are on screen while that turn is spoken; the listener reacts with a nod at the start of each turn."""
    starts = [(v["start"]["line"], v["start"]["at"]) for v in vis] + [(len(scene.turns), 0.0)]
    for i, v in enumerate(vis):
        if v["recipe"] != "character":
            continue
        a, b = starts[i], starts[i + 1]
        span = max((b[0] + b[1]) - (a[0] + a[1]), 1e-6)
        spec = v["character"]
        ch = list(spec.get("changes", []))
        for k, t in enumerate(scene.turns):
            if not a[0] <= k < b[0] + (1 if b[1] > 0 else 0):
                continue
            tt = max(0.0, (k - (a[0] + a[1])) / span)
            if tt > 1:
                continue
            listener = next((c for c in bible.characters if c != t.speaker), None)
            st = {t.speaker: {k2: v2 for k2, v2 in (("expr", t.expr), ("gesture", t.gesture or
                                                                         ("explain" if t.kind in FACTUAL_KINDS else
                                                                          "point" if t.kind == "instruction" else "idle")))
                              if v2}}
            if listener:
                st[listener] = {"gesture": "nod" if k > 0 else "idle", "expr": "listening"}
            ch.append({"t": round(min(tt, 1.0), 4), "chars": st})
        spec["changes"] = sorted(ch, key=lambda c: c["t"])


def story_text(story: Story, bible):
    out = [story.title.upper(), ""]
    for s in story.scenes:
        out.append(f"[{s.id} · {s.beat} · {s.world}]")
        for t in s.turns:
            name = bible.characters[t.speaker].name if t.speaker in bible.characters else t.speaker
            out.append(f"  {name.upper()}: {t.text}" + (f"   ({', '.join(t.fact_ids)})" if t.fact_ids else ""))
        out.append("")
    return "\n".join(out)


# ================================================================== v2: premise → outline → dialogue with memory
# → critique → targeted revision → fact-aware repair. Each stage is small, structured (Pydantic) and cacheable; the
# LLM writes data only. The critic's deterministic checks run without any model (CPU tests); the LLM critic is
# optional and can only ADD weaknesses, never approve a story by itself.
FORMATS = ("everyday_mystery", "experiment", "misconception", "what_if", "journey", "historical_discovery",
           "comparison", "observation_reveal")


class Premise(BaseModel):
    format: Literal[FORMATS]
    opening: str                              # the concrete situation the viewer sees first
    question: str                             # the central curiosity
    relevance: str                            # why a viewer cares
    motivation: str                           # why Arjun cares, in this moment
    discovery: str                            # what they find out
    misunderstanding: str = ""                # a plausible wrong idea to correct (optional)
    key_fact: str                             # the most interesting VERIFIED fact (its id in fact_ids)
    demo: str = ""                            # a physical demonstration opportunity
    payoff: str                               # how the ending answers / reframes the opening
    fact_ids: list[str] = Field(min_length=1)
    risks: str = ""                           # what could become inaccurate


class Premises(BaseModel):
    premises: list[Premise] = Field(min_length=1, max_length=3)


class OutlineBeat(BaseModel):
    id: str = Field(pattern=r"^b[0-9]{1,2}$")
    role: Literal[BEATS]
    setting: str                              # a world of charanim.WORLDS
    objective: str                            # what the characters want here
    observation: str = ""
    happens: str
    changes: str                              # what is different afterwards (cause → effect)
    learns: str = ""
    open_question: str = ""                   # what pulls the viewer into the next beat
    why_next: str = ""
    fact_ids: list[str] = Field(default_factory=list)
    demo: str = ""                            # demos.TEMPLATES name, or ""


class Outline(BaseModel):
    title: str
    beats: list[OutlineBeat] = Field(min_length=3, max_length=10)

    @model_validator(mode="after")
    def shape(self):
        if self.beats[-1].role != "closing":
            raise ValueError("the last beat must be the closing")
        if len({b.id for b in self.beats}) != len(self.beats):
            raise ValueError("beat ids must be unique")
        return self


class Weakness(BaseModel):
    where: str                                # scene id or "sc03_2" (turn) or "story"
    problem: str
    evidence: str = ""
    fix: str = ""
    required: bool = True
    rule: str = ""


class Critique(BaseModel):
    weaknesses: list[Weakness] = Field(default_factory=list)


# ------------------------------------------------------------------ Stage A: fact selection
def select_facts(facts, max_facts=16, max_words=45):   # cited Wikipedia sentences run 30–40 words
    """Facts a viewer can follow when spoken: short enough, no duplicates (first source wins)."""
    out, seen = [], set()
    for f in facts:
        w = set(re.findall(r"[a-z]{4,}", f.text.lower()))
        if len(f.text.split()) > max_words or any(len(w & s) / max(len(w | s), 1) > 0.7 for s in seen):
            continue
        out.append(f); seen.add(frozenset(w))
        if len(out) >= max_facts:
            break
    return out


# ------------------------------------------------------------------ Stage B/C: premises and their choice
PREMISE_HINT = """{"premises": [{"format": "everyday_mystery|experiment|misconception|what_if|journey|historical_discovery|comparison|observation_reveal",
  "opening": "a concrete moment the friends are in", "question": "the central question", "relevance": "why a viewer cares",
  "motivation": "why Arjun cares right now", "discovery": "what they find out", "misunderstanding": "a plausible wrong idea (or empty)",
  "key_fact": "the most surprising sourced fact, in plain words", "demo": "a physical demonstration they could do",
  "payoff": "how the ending answers the opening", "fact_ids": ["F1"], "risks": "what could become inaccurate"}]}"""


def premises_prompt(topic, facts, n, used_formats=()):
    lines = "\n".join(f"{f.id}: {f.text}" for f in facts)
    avoid = f" Avoid these formats used recently: {', '.join(used_formats)}." if used_formats else ""
    return (f"Topic: {topic}\nPropose {n} DIFFERENT story concepts (different formats) for a short video where two "
            f"friends discover this together. Use only these facts; never invent danger or evidence.{avoid}\n"
            f"Facts:\n{lines}\n\nReturn JSON: {PREMISE_HINT}")


def score_premise(p: Premise, fact_ids, templates=()):
    """Deterministic premise score (no model judges its own work). Returns (score, reasons). Unsupported → -inf."""
    reasons, sc = [], 0.0
    missing = [f for f in p.fact_ids if f not in fact_ids]
    if missing:
        return float("-inf"), [f"cites unknown facts {missing}"]
    concrete = len(re.findall(r"[a-z]{4,}", p.opening.lower()))
    if 4 <= concrete <= 30:
        sc += 1; reasons.append("concrete opening")
    if p.question.strip().endswith("?"):
        sc += 1; reasons.append("a real question")
    if p.demo:
        sc += 1; reasons.append("has a demonstration")
        if any(t.replace("_", " ") in p.demo.lower() or t.split("_")[0] in p.demo.lower() for t in templates):
            sc += 1; reasons.append("demonstration can be drawn")
    if p.misunderstanding:
        sc += 0.5; reasons.append("a misconception to correct")
    if len(p.fact_ids) >= 3:
        sc += 1; reasons.append(f"{len(p.fact_ids)} sourced facts")
    q = set(re.findall(r"[a-z]{4,}", p.question.lower())); pay = set(re.findall(r"[a-z]{4,}", p.payoff.lower()))
    if q & pay:
        sc += 1; reasons.append("payoff answers the question")
    if p.format in ("experiment", "what_if", "observation_reveal", "comparison"):
        sc += 0.5; reasons.append("the curious friend acts (predicts / tests / compares)")
    if re.search(r"\b(today we|did you know|let me explain|welcome)\b", p.opening, re.I):
        sc -= 2; reasons.append("formulaic opening")
    return sc, reasons


def choose_premise(premises, fact_ids, templates=()):
    scored = [(score_premise(p, fact_ids, templates), i, p) for i, p in enumerate(premises)]
    (best_sc, why), i, p = max(scored, key=lambda x: (x[0][0], -x[1]))
    if best_sc == float("-inf"):
        raise ValueError("no premise is supported by the facts")
    return p, [{"index": j, "format": q.format, "score": s, "reasons": r} for (s, r), j, q in scored]


# ------------------------------------------------------------------ Stage D: outline
OUTLINE_HINT = """{"title": "max 8 words", "beats": [{"id": "b1", "role": "observation", "setting": "cinema_seats",
  "objective": "...", "observation": "...", "happens": "...", "changes": "...", "learns": "...",
  "open_question": "...", "why_next": "...", "fact_ids": ["F1"], "demo": "finger_test or empty"}]}"""


def outline_prompt(premise: Premise, facts, target_s, worlds, templates):
    lines = "\n".join(f"{f.id}: {f.text}" for f in facts)
    return (f"Premise ({premise.format}): opening: {premise.opening} | question: {premise.question} | discovery: "
            f"{premise.discovery} | misunderstanding: {premise.misunderstanding} | demo: {premise.demo} | payoff: "
            f"{premise.payoff}\nPlan the story in 4–8 beats for about {target_s} seconds. Roles: {', '.join(BEATS)}. "
            "Each beat must CAUSE the next (an observation, prediction, test or misunderstanding leads on) — never "
            "'another interesting fact'. The last beat is 'closing' and answers the opening question.\n"
            f"Settings: {', '.join(worlds)}. Demonstrations that can be drawn: {', '.join(templates)}.\n"
            f"Facts:\n{lines}\n\nReturn JSON: {OUTLINE_HINT}")


# ------------------------------------------------------------------ story state (memory between beats)
class StoryState:
    """What has happened so far, for the next beat's prompt and for continuity checks."""
    def __init__(self, bible):
        self.cur, self.exp = bible.by_role("curious"), bible.by_role("expert")
        self.facts, self.open_q, self.misconceptions, self.corrected, self.demos = [], [], [], [], []
        self.location, self.recent = None, []

    def update(self, scene: "Scene"):
        if self.location and scene.world != self.location:
            self.recent.append(f"(they move from {self.location} to {scene.world})")
        self.location = scene.world
        for t in scene.turns:
            for f in t.fact_ids:
                if f not in self.facts:
                    self.facts.append(f)
            if t.kind == "question" and t.speaker == self.cur.id:
                self.open_q.append(t.text)
            elif t.speaker == self.exp.id and t.kind in FACTUAL_KINDS and self.open_q:
                self.open_q.pop(0)                          # the oldest question got its answer
            if t.kind == "misconception":
                self.misconceptions.append(t.text)
            if t.kind == "correction" and self.misconceptions:
                self.corrected.append(self.misconceptions.pop(0))
            self.recent.append(f"{t.speaker}: {t.text}")
        for v in scene.visuals:
            if v.template and v.template not in self.demos:
                self.demos.append(v.template)
        self.recent = self.recent[-5:]

    def summary(self):
        return (f"Location: {self.location or 'not set'}. Facts already explained: {', '.join(self.facts) or 'none'} "
                f"(don't repeat them without adding something). {self.cur.name} still wonders: "
                f"{' | '.join(self.open_q) or 'nothing open'}. {self.cur.name} still believes (uncorrected): "
                f"{' | '.join(self.misconceptions) or 'nothing'}. Demonstrations done: {', '.join(self.demos) or 'none'}. "
                f"Last lines: {' / '.join(self.recent) or '(start)'}")


# ------------------------------------------------------------------ Stage E: dialogue, beat by beat with memory
DIALOGUE_RULES = """Write it to be SPOKEN by two friends, not read:
- {cur} is active: he observes, predicts, guesses wrong, proposes a test, notices contradictions, helps with the
  demonstration, remembers earlier points. Not just "Wow!", "Really?", "Tell me more", "What happens next?".
- {exp} doesn't lecture: she asks him to predict, demonstrates, uses everyday analogies, corrects gently, builds on
  what he said, admits uncertainty when the facts don't say.
- Short natural sentences, contractions, a little humour when it fits. No "Today we'll learn", "Great question",
  "Did you know", "Let me explain step by step", "In conclusion", "Another interesting fact".
- Every line does something: advances the story, explains, asks something relevant, reveals character, fixes a
  misunderstanding, builds curiosity or pays off. No filler.
- Turn kinds: fact | correction (must cite fact_ids) | question | challenge | prediction | misconception |
  hypothetical (start with "Imagine" or "What if") | analogy (worded as a comparison) | observation | instruction |
  reaction | opinion."""


def dialogue_prompt_v2(bible, premise, beat: OutlineBeat, facts, words, state: StoryState):
    lines = "\n".join(f"{f.id}: {f.text}" for f in facts)
    a, b = bible.by_role("curious"), bible.by_role("expert")
    return (DIALOGUE_RULES.format(cur=a.name, exp=b.name) +
            f"\n\nCentral question: {premise.question}\nStory so far: {state.summary()}\n"
            f"This beat ({beat.role}, setting {beat.setting}): objective: {beat.objective}; observation: "
            f"{beat.observation}; what happens: {beat.happens}; what changes: {beat.changes}; viewer learns: "
            f"{beat.learns}; leaves open: {beat.open_question}.\nAbout {words} spoken words in 2–5 turns. Speakers: "
            f"\"{a.id}\" ({a.name}), \"{b.id}\" ({b.name}).\nFacts:\n{lines}\n\nReturn JSON: {{\"world\": \"{beat.setting}\", "
            "\"turns\": [{\"speaker\": \"...\", \"text\": \"...\", \"kind\": \"...\", \"fact_ids\": [], \"tone\": \"friendly\"}]}"
            f"\ntones: {', '.join(TONES)}")


WORLDS = ("living_room", "cinema_seats", "cinema_screen", "lobby", "studio")


def stage_premise(topic, facts, llm, bible, n_premises=2, used_formats=(), ask_json=None, log=print):
    import demos
    if ask_json is None:
        from planner import ask_json
    ps = ask_json(llm, system_prompt(bible), premises_prompt(topic, facts, n_premises, used_formats), Premises, log=log)
    premise, scores = choose_premise(ps.premises[:max(n_premises, 1)], {f.id for f in facts}, demos.TEMPLATES)
    return {"premises": [p.model_dump() for p in ps.premises], "scores": scores, "premise": premise.model_dump()}


def stage_outline(premise, facts, llm, bible, target_s, ask_json=None, log=print):
    import demos
    if ask_json is None:
        from planner import ask_json
    rel = [f for f in facts if f.id in premise.fact_ids] + [f for f in facts if f.id not in premise.fact_ids][:4]
    return ask_json(llm, system_prompt(bible), outline_prompt(premise, rel, target_s, WORLDS, demos.TEMPLATES),
                    Outline, log=log).model_dump()


def stage_dialogue(topic, premise, outline, facts, llm, bible, target_s, language="en", ask_json=None, log=print):
    """Stage E: beat by beat with the story state (memory) in every prompt. Returns a Story."""
    import demos
    if ask_json is None:
        from planner import ask_json
    sysp = system_prompt(bible)
    te_note = ""
    if language == "te":                                  # natural spoken Telugu writer (same schema)
        import telugu_dialogue as td
        sysp = td.system_prompt(bible)
        te_note = ("\nWrite every turn's \"text\" in natural spoken Telugu. Every fact/correction turn also gives "
                   "\"source_text\": the same claim in plain English using only the cited facts. Format: "
                   + td.DIALOGUE_HINT)
    rel = [f for f in facts if f.id in premise.fact_ids] + [f for f in facts if f.id not in premise.fact_ids][:4]
    state, scenes = StoryState(bible), []
    per = max(int(target_s * WORDS_PER_SECOND / len(outline.beats)), 14)
    by_id = {f.id: f for f in facts}
    for k, b in enumerate(outline.beats):
        bf = [by_id[i] for i in b.fact_ids if i in by_id] or rel[:4]
        d = ask_json(llm, sysp, dialogue_prompt_v2(bible, premise, b, bf, per, state) + te_note, SceneDialogue, log=log)
        v = ask_json(llm, sysp, visuals_prompt(d.turns, demos.TEMPLATES), SceneVisuals, log=log)
        vis = list(v.visuals)
        if b.demo in demos.TEMPLATES and not any(x.template == b.demo for x in vis):     # the planned demo is shown
            last = vis[-1]
            nxt = (max(last.turn, len(d.turns) - 1), 0.5)
            if nxt <= (last.turn, last.at):
                nxt = (last.turn, min(last.at + 0.25, 1.0))
            if nxt > (last.turn, last.at):
                vis.append(Visual(id=f"v{len(vis) + 1}", intent="object_demo", turn=nxt[0], at=nxt[1],
                                  template=b.demo, description=b.happens))
        sc = Scene(id=f"sc{k + 1:02d}", beat=b.role, world=d.world if d.world in WORLDS else b.setting
                   if b.setting in WORLDS else "studio", turns=d.turns, visuals=vis,
                   transition="time_skip" if scenes and scenes[-1].world != (d.world if d.world in WORLDS else b.setting) else None)
        scenes.append(sc); state.update(sc)
    used = {i for s in scenes for t in s.turns for i in t.fact_ids}
    return Story(title=outline.title, topic=topic, language=language, logline=premise.opening,
                 facts=[StoryFact(id=f.id, text=f.text, source_url=f.source_url, title=f.title, revid=f.revid)
                        for f in facts if f.id in used], scenes=scenes)


def make_story_v2(topic, sheet, llm, bible, target_s=75, language="en", log=print, ask_json=None,
                  n_premises=2, max_revisions=1, used_formats=()):
    """Stages A–G. Returns (story, record) where record keeps every intermediate artifact (premises + scores, the
    chosen premise, outline, critique before/after) for caching and the story benchmark."""
    if ask_json is None:
        from planner import ask_json
    facts = select_facts(sheet.facts)                                                     # A
    pr = stage_premise(topic, facts, llm, bible, n_premises, used_formats, ask_json, log)  # B + C
    premise = Premise.model_validate(pr["premise"]); ps = Premises(premises=pr["premises"]); scores = pr["scores"]
    log(f"story: premise '{premise.format}' chosen: {premise.question} ({scores})")
    outline = Outline.model_validate(stage_outline(premise, facts, llm, bible, target_s, ask_json, log))   # D
    st = stage_dialogue(topic, premise, outline, facts, llm, bible, target_s, language, ask_json, log)      # E
    before = critique(st, bible, llm=llm, ask_json=ask_json, log=log)                     # F (rules + LLM critic)
    st, revised = revise(st, before, llm, bible, sheet, ask_json=ask_json, log=log, max_revisions=max_revisions)  # G
    after = critique(st, bible)
    record = {"facts": [f.id for f in facts], "premises": [p.model_dump() for p in ps.premises], "scores": scores,
              "premise": premise.model_dump(), "outline": outline.model_dump(),
              "critique_before": [w.model_dump() for w in before], "revised_scenes": revised,
              "critique_after": [w.model_dump() for w in after]}
    return st, record


# ------------------------------------------------------------------ Stage F: critic
FORMULAS = [r"\btoday (we|we'll|we will|i'll)\b", r"\bwelcome (back|to)\b", r"\bgreat question\b", r"\bdid you know\b",
            r"\blet me explain\b", r"\bstep by step\b", r"\bin conclusion\b", r"\bhello (everyone|guys)\b"]
ARTIFICIAL = [r"\banother interesting fact\b", r"\bnow let me tell you\b", r"\bmoving on\b", r"\blet'?s move on\b",
              r"\bnext,? let'?s\b", r"\bhere'?s another\b"]
EMPTY_PROMPTS = re.compile(r"^\s*(wow|really|that'?s (amazing|cool|interesting)|tell me more|what happens next|"
                           r"go on|and then)\W*$", re.I)


def _w(text):
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def critique(st: Story, bible, llm=None, ask_json=None, log=print):
    """Concrete weaknesses with evidence (deterministic). With an llm, its critique is ADDED (validated JSON)."""
    out = []
    cur, exp = bible.by_role("curious").id, bible.by_role("expert").id
    turns = [(s, k, t) for s in st.scenes for k, t in enumerate(s.turns)]
    tid = lambda s, k: f"{s.id}_{k + 1}"
    for s, k, t in turns:
        for pat in FORMULAS:
            if re.search(pat, t.text, re.I):
                out.append(Weakness(where=tid(s, k), rule="C1", problem="formulaic phrase", evidence=t.text,
                                    fix="say it the way a friend would"))
        for pat in ARTIFICIAL:
            if re.search(pat, t.text, re.I):
                out.append(Weakness(where=tid(s, k), rule="C5", problem="artificial transition", evidence=t.text,
                                    fix="let the next point arise from something they see, predict or misunderstand"))
        if t.speaker == cur and EMPTY_PROMPTS.match(t.text):
            out.append(Weakness(where=tid(s, k), rule="C3", problem="Arjun only prompts Meera to continue",
                                evidence=t.text, fix="give him an observation, a guess or a prediction"))
    run, start = 0, None                                   # C2: question → answer → question → answer ... loop
    for i in range(1, len(turns)):
        (s0, k0, a), (s1, k1, b) = turns[i - 1], turns[i]
        qa = a.speaker == cur and a.kind == "question" and b.speaker == exp and b.kind in FACTUAL_KINDS
        if qa:
            run += 1; start = start or tid(s0, k0)
            if run == 3:
                out.append(Weakness(where=start, rule="C2", problem="Q&A loop: three question→fact exchanges in a row",
                                    evidence="; ".join(x.text[:40] for _, _, x in turns[i - 5:i + 1]),
                                    fix="replace one exchange with a prediction, a test or a misunderstanding"))
        elif not (a.speaker == exp and b.speaker == cur):
            run, start = 0, None
    kinds = [t.kind for _, _, t in turns if t.speaker == cur]
    active = {"observation", "prediction", "challenge", "hypothetical", "misconception", "opinion"}
    if kinds and sum(x == "question" for x in kinds) / len(kinds) > 0.7:
        out.append(Weakness(where="story", rule="C3", problem=f"Arjun mostly asks ({kinds.count('question')}/{len(kinds)})",
                            evidence=", ".join(kinds), fix="let him observe, predict, challenge or help demonstrate"))
    if not set(kinds) & active:
        out.append(Weakness(where="story", rule="C3", problem="Arjun never observes, predicts or challenges",
                            evidence=", ".join(kinds), fix="add one prediction or observation of his own"))
    words = {c: sum(len(t.text.split()) for _, _, t in turns if t.speaker == c) for c in (cur, exp)}
    if words[exp] / max(sum(words.values()), 1) > 0.72:
        out.append(Weakness(where="story", rule="C4", problem="Meera lectures", required=False,
                            evidence=f"{words[exp] / max(sum(words.values()), 1):.0%} of the words",
                            fix="hand some discoveries to Arjun"))
    first = st.scenes[0].turns[0]
    if first.kind in FACTUAL_KINDS or (first.kind == "question" and first.speaker == cur and
                                        len(_w(first.text) & _w(st.topic)) >= 2 and not st.scenes[0].world):
        out.append(Weakness(where=f"{st.scenes[0].id}_1", rule="C6", problem="opens with the textbook question/fact",
                            evidence=first.text, fix="open on a concrete moment or observation"))
    opening = " ".join(t.text for s in st.scenes[:2] for t in s.turns)
    closing = " ".join(t.text for t in st.scenes[-1].turns) + " " + " ".join(t.text for t in st.scenes[-2].turns)
    if not (_w(opening) & _w(closing)):
        out.append(Weakness(where=st.scenes[-1].id, rule="C7", problem="the ending doesn't pay off the opening",
                            evidence=closing[:120], fix="return to the opening moment with the new understanding"))
    if not any(v.template or v.intent in ("object_demo", "animated_illustration") for s in st.scenes for v in s.visuals) \
            and not any(t.kind == "instruction" for _, _, t in turns):
        out.append(Weakness(where="story", rule="C8", problem="no demonstration at all", evidence="",
                            fix="let them test or show the idea"))
    grams = {}
    for s, k, t in turns:                                  # C9 repeated wording across lines
        ws = re.findall(r"[a-z']+", t.text.lower())
        for g in {" ".join(ws[i:i + 4]) for i in range(len(ws) - 3)}:
            grams.setdefault(g, []).append(tid(s, k))
    for g, where in grams.items():
        if len(set(where)) > 1 and len(_w(g)) >= 2:
            out.append(Weakness(where=where[-1], rule="C9", problem="repeated wording", evidence=f"'{g}' in {where}",
                                fix="vary it or cut the repeat", required=False))
    for i, (s, k, t) in enumerate(turns):                  # C10: a question nobody answers
        if t.kind == "question" and t.speaker == cur:
            nxt = [x for _, _, x in turns[i + 1:i + 3]]
            if not any(x.speaker == exp and x.kind not in ("question",) for x in nxt):
                out.append(Weakness(where=tid(s, k), rule="C10", problem="question left without an answer",
                                    evidence=t.text, fix="answer it, or turn it into a test"))
    starts = [t.text.split()[0].lower().strip(",.!?") for _, _, t in turns if t.speaker == cur and t.kind == "question"]
    for w0 in set(starts):
        if starts.count(w0) >= 3:
            out.append(Weakness(where="story", rule="C11", problem=f"Arjun's questions keep starting with '{w0}'",
                                evidence=str(starts), fix="vary his rhythm", required=False))
    qs = [i for i, (_, _, t) in enumerate(turns) if t.kind == "question" and t.speaker == cur]
    instant = [i for i in qs if i + 1 < len(turns) and turns[i + 1][2].speaker == exp
               and turns[i + 1][2].kind in FACTUAL_KINDS]
    if len(qs) >= 2 and len(instant) / len(qs) >= 0.5 and not {"prediction", "hypothetical"} & set(kinds):
        out.append(Weakness(where="story", rule="C12", required=False,
                            problem=f"{len(instant)}/{len(qs)} questions are answered at once and Arjun never predicts",
                            evidence="; ".join(turns[i][2].text[:40] for i in instant),
                            fix="let him guess or predict first, then test it"))
    unprompted = [tid(s, k) for i, (s, k, t) in enumerate(turns) if t.speaker == exp and t.kind in FACTUAL_KINDS
                  and (i == 0 or turns[i - 1][2].speaker == exp)]
    if unprompted:
        out.append(Weakness(where=unprompted[0].split("_")[0], rule="C13", required=len(unprompted) >= 2,
                            problem="an explanation arrives unprompted (nothing Arjun said or saw leads to it)",
                            evidence=", ".join(unprompted), fix="let an observation, guess or test trigger it"))
    last = st.scenes[-1].turns[-1].text
    if re.search(r"^\W*(\w+\s+){0,3}\w+,\s+(\w+\s+){0,3}\w+,\s+(and\s+)?", last) and "?" not in last:
        out.append(Weakness(where=f"{st.scenes[-1].id}_{len(st.scenes[-1].turns)}", rule="C14", required=False,
                            problem="the ending is a recap list", evidence=last,
                            fix="end on a moment (a callback to the opening) instead of a summary"))
    if llm is not None and ask_json is not None:
        try:
            more = ask_json(llm, "You are a demanding script editor. Find real weaknesses; never praise.",
                            critic_prompt(st, bible), Critique, log=log)
            ids = {s.id for s in st.scenes} | {tid(s, k) for s, k, _ in turns} | {"story"}
            out += [w.model_copy(update={"rule": w.rule or "LLM"}) for w in more.weaknesses if w.where in ids]
        except Exception as e:
            log(f"story: LLM critic unavailable ({e}); deterministic critique only")
    return out


def critic_prompt(st: Story, bible):
    script = "\n".join(f"{s.id}_{k + 1} [{s.beat}] {bible.characters[t.speaker].name}: {t.text}"
                       for s in st.scenes for k, t in enumerate(s.turns))
    return ("Judge this two-friend educational script. Is the opening interesting and concrete? Does each scene cause "
            "the next? Does Arjun contribute ideas? Does Meera explain naturally? Repetition, artificial transitions, "
            "formal wording, missing demonstrations, unprepared discoveries, an unsatisfying ending, generic Q&A?\n"
            f"{script}\n\nReturn JSON: {{\"weaknesses\": [{{\"where\": \"sc02 or sc02_1 or story\", \"problem\": \"...\", "
            "\"evidence\": \"quote\", \"fix\": \"...\", \"required\": true}]}. An empty list only if there is truly nothing.")


# ------------------------------------------------------------------ Stage G: targeted revision
def revise(st: Story, weaknesses, llm, bible, sheet, ask_json=None, log=print, max_revisions=1):
    """Rewrite ONLY the scenes with required weaknesses (bounded); keep a rewrite only if it has fewer required
    weaknesses and still passes the dialogue rules. Returns (story, revised scene ids)."""
    if llm is None or max_revisions <= 0:
        return st, []
    if ask_json is None:
        from planner import ask_json
    by_scene = {}
    for w in weaknesses:
        if w.required:
            sid = w.where.split("_")[0] if w.where != "story" else None
            by_scene.setdefault(sid, []).append(w)
    targets = [s for s in st.scenes if s.id in by_scene]
    if None in by_scene and not targets:                   # story-level problem: rewrite the weakest middle scene
        targets = st.scenes[1:2]
        by_scene[targets[0].id] = by_scene[None]
    facts = {f.id: f for f in getattr(sheet, "facts", [])}
    revised = []
    for _ in range(max_revisions):
        for s in targets:
            ws = by_scene.get(s.id, [])
            fl = "\n".join(f"{f.id}: {f.text}" for i, f in facts.items() if i in {x for t in s.turns for x in t.fact_ids}) \
                or "\n".join(f"{f.id}: {f.text}" for f in list(facts.values())[:6])
            problems = "\n".join(f"- {w.problem}: {w.evidence} → {w.fix}" for w in ws)
            prompt = (DIALOGUE_RULES.format(cur=bible.by_role("curious").name, exp=bible.by_role("expert").name)
                      + f"\n\nRewrite this scene to fix ONLY these problems:\n{problems}\nScene ({s.beat}, {s.world}):\n"
                      + "\n".join(f"{t.speaker} [{t.kind}] {t.text}" for t in s.turns)
                      + f"\nFacts:\n{fl}\n\nReturn JSON: {{\"world\": \"{s.world}\", \"turns\": [...]}} (same format, "
                        f"{len(s.turns)}±1 turns).")
            try:
                d = ask_json(llm, system_prompt(bible), prompt, SceneDialogue, log=log)
            except Exception as e:
                log(f"story: revision of {s.id} failed ({e}); kept the original"); continue
            new_vis = [v.model_copy(update={"turn": min(v.turn, len(d.turns) - 1)}) for v in s.visuals]
            dedup, last = [], None
            for v in new_vis:
                if last is None or (v.turn, v.at) > last:
                    dedup.append(v); last = (v.turn, v.at)
            cand_scene = s.model_copy(update={"turns": d.turns, "visuals": dedup})
            cand = st.model_copy(update={"scenes": [cand_scene if x.id == s.id else x for x in st.scenes]})
            try:
                cand = Story.model_validate(cand.model_dump())
            except Exception as e:
                log(f"story: revised {s.id} is invalid ({e}); kept the original"); continue
            req = lambda story_: sum(w.required for w in critique(story_, bible) if w.where.startswith(s.id))
            errs = [i for i in check_story(cand, bible) if i["level"] == "error"]
            if req(cand) < req(st) and not errs:
                st = cand; revised.append(s.id)
                log(f"story: revised {s.id} ({len(ws)} problems)")
            else:
                log(f"story: revision of {s.id} not better; kept the original")
    return st, revised


# ------------------------------------------------------------------ fact-aware repair (replaces plain dropping)
REWRITE_HINT = '{"text": "...", "fact_ids": ["F1"]}'


class TurnRewrite(BaseModel):
    text: str
    fact_ids: list[str] = Field(default_factory=list)


def repair_grounding(st: Story, nli, bible, llm=None, ask_json=None, log=print, max_rewrites=1):
    """1 find unsupported claims → 2 bounded rewrite from the cited facts → 3 re-check → 4 still unsupported: remove
    → 5–6 remove what depended on it (the question it answered, a reaction to it) → 7 re-check the story.
    Returns (story, results, report)."""
    res = ground(st, nli, topic=st.topic)
    bad = {k for k, r in res.items() if r["status"] != "PASS"}
    report = {"rewritten": [], "removed": [], "dependents_removed": []}
    if not bad:
        return st, res, report
    by_id = {f.id: f for f in st.facts}
    scenes = {s.id: s for s in st.scenes}
    if llm is not None:
        if ask_json is None:
            from planner import ask_json
        for key in sorted(bad):
            sid, k = key.rsplit("_", 1); k = int(k) - 1
            t = scenes[sid].turns[k]
            ev = "\n".join(f"{i}: {by_id[i].text}" for i in t.fact_ids if i in by_id)
            for _ in range(max_rewrites):
                try:
                    rw = ask_json(llm, system_prompt(bible),
                                  f"Rewrite this line so it says ONLY what these facts support, in the same "
                                  f"conversational voice ({bible.characters[t.speaker].name}, {t.kind}).\nLine: {t.text}\n"
                                  f"Facts:\n{ev}\nReturn JSON: {REWRITE_HINT}", TurnRewrite, log=log)
                    nt = t.model_copy(update={"text": rw.text, "fact_ids": [i for i in rw.fact_ids if i in by_id] or t.fact_ids})
                    Turn.model_validate(nt.model_dump())
                except Exception as e:
                    log(f"story: rewrite of {key} failed ({e})"); continue
                import factcheck
                r2 = factcheck.check_sentences([{"id": key, "text": nt.source_text or nt.text,
                                                 "evidence": [by_id[i].text for i in nt.fact_ids if i in by_id],
                                                 "question": nt.kind in ("question", "challenge"),
                                                 "analogy": nt.kind in ("analogy", "hypothetical")}], nli, topic=st.topic)
                if r2[key]["status"] == "PASS":
                    turns = list(scenes[sid].turns); turns[k] = nt
                    scenes[sid] = scenes[sid].model_copy(update={"turns": turns})
                    report["rewritten"].append({"turn": key, "from": t.text, "to": nt.text}); bad.discard(key)
                    break
    drop = set(bad)
    for key in sorted(bad):                                # dependents of what is removed
        sid, k = key.rsplit("_", 1); k = int(k) - 1
        turns = scenes[sid].turns
        t = turns[k]
        if k > 0 and turns[k - 1].kind == "question" and turns[k - 1].speaker != t.speaker:
            drop.add(f"{sid}_{k}"); report["dependents_removed"].append({"turn": f"{sid}_{k}", "why": f"its answer {key} was removed"})
        if k + 1 < len(turns) and turns[k + 1].kind in ("reaction", "observation", "opinion") \
                and turns[k + 1].speaker != t.speaker:
            drop.add(f"{sid}_{k + 2}"); report["dependents_removed"].append({"turn": f"{sid}_{k + 2}", "why": f"it reacts to {key}"})
        report["removed"].append({"turn": key, "text": t.text, "why": res[key]["why"]})
    st = Story.model_validate(st.model_copy(update={"scenes": [scenes[s.id] for s in st.scenes]}).model_dump())
    if drop:
        st = _remove_turns(st, drop, log)
    for k in report["removed"]:
        log(f"story: removed unsupported turn {k['turn']}: {k['text'][:80]!r} ({k['why']})")
    for k in report["dependents_removed"]:
        log(f"story: removed {k['turn']} ({k['why']})")
    return st, ground(st, nli, topic=st.topic), report


def _remove_turns(st: Story, keys, log=print):
    scenes = []
    for s in st.scenes:
        keep = [k for k in range(len(s.turns)) if f"{s.id}_{k + 1}" not in keys]
        if not keep:
            log(f"story: scene {s.id} dropped (nothing left)"); continue
        new_of = {k: next((i for i, kk in enumerate(keep) if kk >= k), len(keep) - 1) for k in range(len(s.turns))}
        vis = []
        for v in s.visuals:
            nv = v.model_copy(update={"turn": new_of[v.turn], "at": v.at if v.turn in keep else 0.0})
            if not vis or (nv.turn, nv.at) > (vis[-1].turn, vis[-1].at):
                vis.append(nv)
        vis[0] = vis[0].model_copy(update={"turn": 0, "at": 0.0})
        scenes.append(s.model_copy(update={"turns": [s.turns[k] for k in keep], "visuals": vis}))
    return Story.model_validate(st.model_copy(update={"scenes": scenes}).model_dump())


# ------------------------------------------------------------------ story benchmark helpers (old vs new script)
def story_metrics(st: Story, bible):
    """Structural numbers for the story benchmark (they inform, they don't decide: human ratings decide)."""
    cur, exp = bible.by_role("curious").id, bible.by_role("expert").id
    turns = [t for s in st.scenes for t in s.turns]
    ck = critique(st, bible)
    kinds = [t.kind for t in turns if t.speaker == cur]
    words = {c: sum(len(t.text.split()) for t in turns if t.speaker == c) for c in (cur, exp)}
    return {"turns": len(turns), "words": sum(words.values()),
            "curious_kinds": {k: kinds.count(k) for k in sorted(set(kinds))},
            "curious_question_share": round(kinds.count("question") / max(len(kinds), 1), 2),
            "expert_word_share": round(words[exp] / max(sum(words.values()), 1), 2),
            "demonstrations": sum(1 for s in st.scenes for v in s.visuals if v.template),
            "required_weaknesses": sum(w.required for w in ck), "weaknesses": [f"{w.rule} {w.where}: {w.problem}" for w in ck]}

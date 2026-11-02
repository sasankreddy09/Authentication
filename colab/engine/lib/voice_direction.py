"""Voice direction: how each narration line should be DELIVERED, decided from the story, never from punctuation
marks or keywords. The words never change (emphasis words must be words of the line).

Where a line's direction comes from (first that applies):
  1. "voice_direction" written in the plan (validated);
  2. the author's tone label on the line (existing plans: curious, explainer, warm, worried, happy, friendly);
  3. the line's narrative ROLE: the shot's beat (hook, setup, step, aside, turn, resolution, payoff) and what the
     sentence does (asks, explains, reveals, concludes), with the neighbouring lines as context;
  4. optionally a local LLM director (LLM_SYSTEM below) that reads the whole story and returns validated JSON.
Then `smooth()` enforces the arc rules for every source: conservative intensity (profile range), small changes
inside a beat, justified jumps only at beat boundaries, no runs of dramatic emotions, rare peaks.
"""
from __future__ import annotations

import json, re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Emotion = Literal["neutral", "curious", "conversational", "suspenseful", "excited", "surprised", "serious",
                  "empathetic", "reassuring", "confident", "reflective", "inspirational"]
Role = Literal["hook", "question", "explain", "reveal", "compare", "consequence", "conclude", "dialogue"]
DRAMATIC = {"suspenseful", "excited", "surprised", "inspirational"}
# emotions that can sit next to each other (or blend) without sounding like a jump
COMPATIBLE = {
    "neutral": {"conversational", "curious", "serious", "confident", "reflective", "reassuring"},
    "curious": {"neutral", "conversational", "suspenseful", "surprised", "excited"},
    "conversational": {"neutral", "curious", "reassuring", "confident", "excited"},
    "suspenseful": {"curious", "serious", "surprised"},
    "excited": {"curious", "surprised", "inspirational", "conversational", "confident"},
    "surprised": {"curious", "excited", "suspenseful", "serious"},
    "serious": {"neutral", "empathetic", "suspenseful", "reflective", "confident"},
    "empathetic": {"serious", "reassuring", "reflective"},
    "reassuring": {"conversational", "empathetic", "confident", "neutral"},
    "confident": {"neutral", "conversational", "reassuring", "inspirational", "reflective", "serious"},
    "reflective": {"neutral", "serious", "confident", "inspirational", "empathetic"},
    "inspirational": {"confident", "reflective", "excited"},
}


class VoiceDirection(BaseModel):
    speaker_id: str = "narrator"
    emotion: Emotion = "neutral"
    blend: Emotion | None = None
    intensity: float = Field(0.25, ge=0.0, le=1.0)
    delivery: str = "clear and natural"
    pace_multiplier: float = Field(1.0, ge=0.8, le=1.15)
    energy: Literal["low", "medium", "high"] = "medium"
    emphasis_words: list[str] = Field(default_factory=list, max_length=3)
    pause_before_ms: int = Field(0, ge=0, le=1200)
    pause_after_ms: int = Field(450, ge=0, le=1500)
    style_instruction: str = ""
    transition: Literal["steady", "gradual", "shift"] = "steady"
    role: Role = "explain"
    source: str = "rules"                       # plan | tone | rules | llm | dialogue (+ "smoothed")
    intent: str | None = None                   # two-friend dialogue: the conversational intention of the line

    @field_validator("blend")
    @classmethod
    def blend_compatible(cls, v, info):
        e = info.data.get("emotion")
        if v is not None and e is not None and v != e and v not in COMPATIBLE.get(e, set()):
            raise ValueError(f"blend {v} is not compatible with {e}")
        return v


PROFILES = {
    # base emotion, allowed intensity range, base pace, pause scale, emotions allowed, preferred backend, nonverbal
    "science_explainer":     dict(base="conversational", lo=0.15, hi=0.6, pace=1.0, pause=1.0, backend="qwen3-tts",
                                  allowed=None, nonverbal=[]),
    "beginner_education":    dict(base="reassuring", lo=0.1, hi=0.5, pace=0.96, pause=1.15, backend="qwen3-tts",
                                  allowed=None, nonverbal=[]),
    "cinematic_documentary": dict(base="neutral", lo=0.15, hi=0.7, pace=0.97, pause=1.2, backend="qwen3-tts",
                                  allowed=None, nonverbal=[]),
    "mystery_documentary":   dict(base="curious", lo=0.2, hi=0.7, pace=0.95, pause=1.25, backend="qwen3-tts",
                                  allowed=None, nonverbal=[]),
    "historical_narration":  dict(base="reflective", lo=0.1, hi=0.55, pace=0.96, pause=1.15, backend="qwen3-tts",
                                  allowed={"neutral", "reflective", "serious", "curious", "confident", "empathetic",
                                           "suspenseful", "surprised"}, nonverbal=[]),
    "inspirational_story":   dict(base="confident", lo=0.2, hi=0.7, pace=1.0, pause=1.1, backend="qwen3-tts",
                                  allowed=None, nonverbal=[]),
    # two friends talking: mostly natural, low intensity; stronger only on a few hero lines; pauses = response gaps
    "two_friends":           dict(base="conversational", lo=0.1, hi=0.5, pace=1.0, pause=1.0, backend="qwen3-tts",
                                  allowed=None, nonverbal=[]),
}

# the author's tone labels (existing plans) → (emotion, intensity, energy)
TONE_MAP = {"curious": ("curious", 0.4, "medium"), "explainer": ("conversational", 0.25, "medium"),
            "warm": ("reassuring", 0.3, "medium"), "worried": ("serious", 0.4, "low"),
            "happy": ("excited", 0.45, "high"), "friendly": ("conversational", 0.35, "medium"),
            "sad": ("empathetic", 0.4, "low"), "tense": ("suspenseful", 0.45, "low"),
            "awe": ("surprised", 0.45, "medium"), "serious": ("serious", 0.35, "low")}
# a shot's narrative beat → role, emotion, intensity, pace, pauses
BEAT_ROLE = {"hook": ("hook", "curious", 0.45, 1.03, 0, 550), "setup": ("explain", "conversational", 0.3, 1.0, 0, 450),
             "step": ("explain", "conversational", 0.25, 0.98, 0, 450), "aside": ("explain", "neutral", 0.2, 1.0, 0, 400),
             "turn": ("question", "suspenseful", 0.45, 0.94, 150, 650), "resolution": ("reveal", "surprised", 0.45, 0.97, 250, 600),
             "reveal": ("reveal", "surprised", 0.5, 0.96, 300, 650), "consequence": ("consequence", "serious", 0.4, 0.95, 100, 600),
             "payoff": ("conclude", "confident", 0.35, 0.96, 150, 900)}
EMOTION_WORDS = {"neutral": "a clear, even tone", "curious": "quiet curiosity", "conversational": "a warm, conversational tone",
                 "suspenseful": "restrained suspense", "excited": "controlled excitement", "surprised": "genuine but controlled surprise",
                 "serious": "a serious, credible tone", "empathetic": "serious empathy and restrained emotion",
                 "reassuring": "a calm, reassuring tone", "confident": "calm confidence", "reflective": "a thoughtful, reflective tone",
                 "inspirational": "quiet inspiration"}


def words_of(text):
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", text)


def asks(text):
    """The sentence's FUNCTION (syntax), not its punctuation: does it ask something?"""
    return bool(re.match(r"^\s*(what|why|how|who|where|when|which|will|would|can|could|do|does|did|is|are|was|were|"
                         r"have|has|should|but what|so what|and what)\b", text, re.I)) or text.rstrip().endswith("?")


def instruction(d: VoiceDirection, narrator_desc=None):
    """Natural-language direction for instruction-following TTS (built from the fields, never free-form)."""
    pace = ("slightly slower than usual" if d.pace_multiplier < 0.97 else
            "slightly faster than usual" if d.pace_multiplier > 1.03 else "at a natural pace")
    if narrator_desc is None:
        narrator_desc = "the same narrator voice" if d.speaker_id == "narrator" else "the same voice"
    if d.intent in INTENT_WORDS:                       # a friend in a conversation, not a narrator reading
        s = f"Maintain {narrator_desc}. Talking to a friend, {INTENT_WORDS[d.intent]}, with {EMOTION_WORDS[d.emotion]}"
        s += f", {pace}, {'subtle' if d.intensity < 0.35 else 'moderate'} expression"
        if d.emphasis_words:
            s += ", lightly stressing " + ", ".join(f"'{w}'" for w in d.emphasis_words)
        return s + ". Casual and believable, like real speech, never theatrical or announcer-like."
    s = f"Maintain {narrator_desc}. Speak with {EMOTION_WORDS[d.emotion]}"
    if d.blend:
        s += f" and a touch of {EMOTION_WORDS[d.blend]}"
    s += f", {pace}, {'subtle' if d.intensity < 0.35 else 'moderate' if d.intensity < 0.6 else 'clear'} expression"
    if d.emphasis_words:
        s += ", gently emphasizing " + ", ".join(f"'{w}'" for w in d.emphasis_words)
    if d.role in ("question", "hook") and d.pause_before_ms:
        s += ", with a natural pause before the final phrase"
    return s + ". Natural and believable, never theatrical."


def pick_emphasis(text, n=2):
    """Most informative words: numbers, names, long content words (deterministic; the LLM director may override)."""
    stop = {"about", "after", "their", "there", "these", "those", "which", "where", "while", "would", "could",
            "should", "because", "before", "every", "other", "being", "still", "until", "since", "under", "into"}
    ws = words_of(text)
    scored = []
    for i, w in enumerate(ws):
        lw = w.lower()
        if lw in stop or len(lw) < 4 and not re.search(r"\d", w):
            continue
        sc = (3 if re.search(r"\d", w) else 0) + (2 if i > 0 and w[0].isupper() else 0) + min(len(w), 10) / 5
        scored.append((sc, i, w))
    return [w for _, _, w in sorted(sorted(scored, reverse=True)[:n], key=lambda x: x[1])]


def beats_of(shots):
    """Story beats: consecutive shots sharing the same (chapter, beat role); a beat may span several shots."""
    beats, cur, key = [], [], None
    for s in shots:
        k = (s.get("chapter"), s.get("beat"))
        if cur and k != key:
            beats.append(cur); cur = []
        cur.append(s); key = k
    if cur:
        beats.append(cur)
    return beats


def rule_directions(shots, profile="science_explainer", narrator_only=False):
    """{line_id: VoiceDirection} from plan directions / author tones / narrative roles (+ context)."""
    P = PROFILES[profile]
    out = {}
    beats = beats_of(shots)
    for bi, beat in enumerate(beats):
        last_beat = bi == len(beats) - 1
        for s in beat:
            for li, l in enumerate(s["lines"]):
                lid = l.get("id") or f"{s['shot_id']}_{li + 1}"
                if l.get("voice_direction"):
                    out[lid] = VoiceDirection(**{**l["voice_direction"], "source": "plan"}); continue
                if narrator_only and l.get("speaker") != "narrator":
                    continue
                beat_name = s.get("beat") or ("hook" if bi == 0 else "payoff" if last_beat else "step")
                role, emo, inten, pace, pb, pa = BEAT_ROLE.get(beat_name, BEAT_ROLE["step"])
                src = "rules"
                if l.get("speaker", "narrator") != "narrator":
                    role = "dialogue"
                if asks(l["text"]) and role not in ("hook",):
                    role, emo, pb = "question", ("suspenseful" if beat_name in ("turn",) else "curious"), max(pb, 100)
                if l.get("tone") in TONE_MAP:                         # the author's intent outranks the rules
                    emo, inten, energy = TONE_MAP[l["tone"]]; src = "tone"
                else:
                    energy = "low" if emo in ("serious", "empathetic", "suspenseful", "reflective") else \
                        "high" if emo in ("excited",) else "medium"
                if P["allowed"] and emo not in P["allowed"]:
                    emo = P["base"]
                d = VoiceDirection(speaker_id=l.get("speaker", "narrator"), emotion=emo, intensity=inten, role=role,
                                   delivery=EMOTION_WORDS[emo], pace_multiplier=round(pace * P["pace"], 3), energy=energy,
                                   emphasis_words=pick_emphasis(l.get("tts") or l["text"]),
                                   pause_before_ms=int(pb * P["pause"]), pause_after_ms=int(pa * P["pause"]),
                                   transition="gradual" if li or s is not beat[0] else "shift", source=src)
                out[lid] = d
    return out


def smooth(order, dirs, profile="science_explainer", max_intensity=1.0):
    """Arc rules for ANY source. order: line ids in story order. max_intensity: a backend cap (measured: strong
    instructions to Qwen3-TTS shifted the narrator's timbre and even changed a word). Returns (dirs, adjustments)."""
    P, adj, run = dict(PROFILES[profile]), [], 0
    P["hi"] = min(P["hi"], max_intensity)
    peaks, last, runs = 0, {}, {}
    for lid in order:
        d = dirs.get(lid)
        if d is None:
            continue
        # continuity is a property of ONE voice: Meera's calm correction must not be forced into Arjun's surprise
        prev, run = last.get(d.speaker_id), runs.get(d.speaker_id, 0)
        lo, hi = P["lo"], P["hi"]
        if d.intensity > hi or d.intensity < lo:
            adj.append(f"{lid}: intensity {d.intensity} → within the {profile} range [{lo}, {hi}]")
            d.intensity = min(max(d.intensity, lo), hi)
        if prev is not None:
            same_beat = d.transition != "shift"
            step = 0.15 if same_beat else 0.3
            if abs(d.intensity - prev.intensity) > step:
                new = round(prev.intensity + step * (1 if d.intensity > prev.intensity else -1), 2)
                adj.append(f"{lid}: intensity {d.intensity} → {new} (no sudden jump {'inside a beat' if same_beat else 'between beats'})")
                d.intensity = new
            if same_beat and d.emotion != prev.emotion and d.emotion not in COMPATIBLE[prev.emotion] \
                    and d.source not in ("plan", "dialogue"):           # a friend's intention is a real change
                adj.append(f"{lid}: {d.emotion} → {prev.emotion} + blend (incompatible change inside a beat)")
                d.blend, d.emotion = (d.emotion if d.emotion in COMPATIBLE[prev.emotion] else None), prev.emotion
        run = run + 1 if d.emotion in DRAMATIC else 0
        if run > 2:                                               # no long runs of drama
            adj.append(f"{lid}: {d.emotion} → conversational (third dramatic line in a row)")
            d.emotion, d.intensity, run = "conversational", min(d.intensity, 0.35), 0
        if d.intensity >= 0.6:
            peaks += 1
            if peaks > max(1, len(order) // 8):                   # peaks are rare
                adj.append(f"{lid}: intensity {d.intensity} → 0.55 (too many peaks)")
                d.intensity = 0.55
        if not d.style_instruction or d.source.startswith("dialogue"):
            d.style_instruction = instruction(d)
        if adj and adj[-1].startswith(lid) and "smoothed" not in d.source:
            d.source += "+smoothed"
        last[d.speaker_id], runs[d.speaker_id] = d, run
    return dirs, adj


def neutral_directions(order, dirs):
    """The neutral baseline for the A/B benchmark: same voice, no expression, no extra pauses."""
    return {k: VoiceDirection(speaker_id=d.speaker_id, emotion="neutral", intensity=0.0, role=d.role,
                              pause_before_ms=0, pause_after_ms=450, source="neutral-baseline",
                              style_instruction="") for k, d in dirs.items()}


LLM_SYSTEM = """You are a documentary voice director. For each narration line, decide HOW it should be spoken so the
story feels alive but natural: read the whole story first, use each line's meaning and its neighbours, and keep one
narrator. Never change the words. Most lines are calm teaching (intensity 0.15–0.35); save stronger delivery for real
turns and reveals; never alternate excited and calm every line; serious facts are never exciting.
Return ONE JSON object: {"lines": [{"id": "...", "emotion": one of [neutral, curious, conversational, suspenseful,
excited, surprised, serious, empathetic, reassuring, confident, reflective, inspirational], "intensity": 0-1,
"pace_multiplier": 0.85-1.1, "energy": "low|medium|high", "emphasis_words": [up to 2 words copied from the line],
"pause_before_ms": 0-600, "pause_after_ms": 200-1200, "role": "hook|question|explain|reveal|compare|consequence|conclude|dialogue",
"delivery": "a few words"}]}"""


class _LLMOut(BaseModel):
    lines: list[dict]


def llm_directions(llm, shots, base, ask_json, log=print):
    """The LLM proposes; every field is validated; emphasis words must be in the line; failures keep the rule
    direction for that line (logged)."""
    story = []
    for s in shots:
        for li, l in enumerate(s["lines"]):
            lid = l.get("id") or f"{s['shot_id']}_{li + 1}"
            story.append({"id": lid, "beat": s.get("beat") or s.get("chapter"), "speaker": l.get("speaker", "narrator"),
                          "text": l.get("tts") or l["text"]})
    out = ask_json(llm, LLM_SYSTEM, "Story, in order:\n" + json.dumps(story, ensure_ascii=False, indent=0), _LLMOut,
                   log=log, max_new_tokens=2400)
    texts = {x["id"]: x["text"] for x in story}
    dirs = dict(base)
    for row in out.lines:
        lid = row.get("id")
        if lid not in texts or lid not in base:
            continue
        try:
            words = {w.lower() for w in words_of(texts[lid])}
            row["emphasis_words"] = [w for w in row.get("emphasis_words", []) if w.lower() in words][:3]
            d = VoiceDirection(**{**base[lid].model_dump(), **{k: v for k, v in row.items() if k != "id"},
                                  "speaker_id": base[lid].speaker_id, "source": "llm", "style_instruction": ""})
            dirs[lid] = d
        except Exception as e:
            log(f"voice director: line {lid} kept the rule direction ({str(e)[:120]})")
    return dirs


# ---------------------------------------------------------------- two friends talking (story plans)
# conversational intention → (emotion, intensity, pace, energy). Most lines stay natural and low-intensity;
# the stronger ones are reserved for hero moments (discovery, revelation).
INTENTS = {
    "genuine_curiosity":       ("curious", 0.3, 1.0, "medium"),
    "thinking_aloud":          ("reflective", 0.2, 0.96, "low"),
    "playful_disbelief":       ("surprised", 0.35, 1.02, "medium"),
    "gentle_correction":       ("reassuring", 0.25, 0.98, "medium"),
    "explaining_observation":  ("conversational", 0.2, 1.0, "medium"),
    "asking_confirmation":     ("curious", 0.25, 1.03, "medium"),
    "quiet_realization":       ("reflective", 0.3, 0.95, "low"),
    "enthusiastic_discovery":  ("excited", 0.45, 1.04, "high"),
    "sharing_surprising_fact": ("surprised", 0.3, 1.0, "medium"),
    "offering_reassurance":    ("reassuring", 0.25, 0.98, "medium"),
    "reflective_explanation":  ("reflective", 0.2, 0.97, "low"),
    "satisfying_revelation":   ("confident", 0.4, 0.97, "medium"),
    "guiding_demo":            ("conversational", 0.3, 1.02, "medium"),
    "playful_remark":          ("conversational", 0.3, 1.04, "medium"),
}
INTENT_WORDS = {
    "genuine_curiosity": "genuinely wondering about something", "thinking_aloud": "thinking it through out loud",
    "playful_disbelief": "playfully doubtful", "gentle_correction": "gently putting a friend right",
    "explaining_observation": "explaining what you both just saw", "asking_confirmation": "checking you understood",
    "quiet_realization": "quietly realising something", "enthusiastic_discovery": "excited by what you just noticed",
    "sharing_surprising_fact": "sharing something surprising", "offering_reassurance": "reassuring a friend",
    "reflective_explanation": "explaining calmly", "satisfying_revelation": "revealing the answer, pleased",
    "guiding_demo": "inviting your friend to try something", "playful_remark": "making a light, friendly remark"}
HERO_INTENTS = {"enthusiastic_discovery", "satisfying_revelation", "playful_disbelief"}
HERO_BEATS = {"observation", "question", "reveal", "closing"}


def derive_intent(turn, prev, speaker_role, story_beat, first_in_scene):
    """The line's conversational intention from WHAT IT DOES and WHAT CAME JUST BEFORE (never punctuation alone).
    turn/prev: plan lines (dicts with speaker, kind, text); speaker_role: curious | expert."""
    if turn.get("intent") in INTENTS:
        return turn["intent"]
    k = turn.get("kind", "observation")
    pk = (prev or {}).get("kind")
    other = prev is not None and prev.get("speaker") != turn.get("speaker")
    follow = re.match(r"^\s*(so|and|then|wait|but|you mean|hold on)\b", turn.get("text", ""), re.I)
    if k == "question":
        if other and pk in ("fact", "correction") and follow:
            return "asking_confirmation"
        if other and pk in ("fact", "correction", "observation") and story_beat in ("reveal", "correction"):
            return "playful_disbelief" if follow else "genuine_curiosity"
        return "genuine_curiosity"
    if k in ("misconception", "prediction", "hypothetical"):
        return "thinking_aloud"
    if k == "challenge":
        return "playful_disbelief"
    if k == "correction":
        return "gentle_correction"
    if k == "fact":
        if story_beat in ("reveal",) or (speaker_role == "curious" and other):
            return "satisfying_revelation" if speaker_role == "expert" else "quiet_realization"
        return "explaining_observation" if other and pk == "question" else "reflective_explanation"
    if k == "instruction":
        return "guiding_demo"
    if k == "analogy":
        return "explaining_observation"
    if k in ("reaction", "opinion"):
        return "playful_remark" if speaker_role == "expert" or story_beat == "closing" else "quiet_realization"
    # observation
    if speaker_role == "curious" and (pk == "instruction" or (first_in_scene and story_beat == "observation")):
        return "enthusiastic_discovery"
    if story_beat == "closing":
        return "offering_reassurance" if speaker_role == "expert" else "playful_remark"
    return "explaining_observation" if speaker_role == "expert" else "thinking_aloud"


def _jitter(key, lo, hi):
    """Deterministic value in [lo, hi] (no identical, mechanical gaps; the same line always gets the same gap)."""
    import hashlib
    f = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return int(lo + (hi - lo) * f)


def response_gap_ms(turn, prev, intent, prev_intent, first_in_scene, key):
    """Silence before a line. Conversation research: the typical gap between speakers is ~200 ms (Stivers et al.
    2009, PNAS, 10 languages); longer gaps carry meaning (thinking, trying a demonstration, a reveal)."""
    if prev is None:
        return 0
    if prev.get("kind") == "instruction":
        return _jitter(key, 650, 950)                  # time to actually try it (also across a cut)
    if first_in_scene:
        return _jitter(key, 100, 250)                  # the shot's own lead-in (pre_s) adds ~300 ms
    if intent == "satisfying_revelation":
        return _jitter(key, 450, 650)
    if intent in ("quiet_realization", "thinking_aloud"):
        return _jitter(key, 400, 650)
    if prev.get("speaker") == turn.get("speaker"):
        return _jitter(key, 250, 450)                  # same person, next thought
    if intent in ("playful_disbelief", "enthusiastic_discovery", "playful_remark"):
        return _jitter(key, 120, 260)                  # quick reaction
    if prev.get("kind") == "question":
        return _jitter(key, 200, 380)                  # answering
    return _jitter(key, 180, 380)


def dialogue_directions(shots, characters):
    """{line_id: VoiceDirection} for two-friend plans: intention from the story context, response gaps instead of
    narrator pauses (pause_after = 0), per-character baseline (the curious friend a little livelier)."""
    out, prev, prev_intent = {}, None, None
    for s in shots:
        beat = s.get("story_beat") or s.get("beat") or ""
        for li, l in enumerate(s["lines"]):
            lid = l.get("id") or f"{s['shot_id']}_{li + 1}"
            if l.get("voice_direction"):
                out[lid] = VoiceDirection(**{**l["voice_direction"], "source": "plan"})
                prev, prev_intent = l, out[lid].intent
                continue
            role = (characters.get(l.get("speaker"), {}) or {}).get("role", "expert")
            intent = derive_intent(l, prev, role, beat, li == 0)
            emo, inten, pace, energy = INTENTS[intent]
            if role == "curious" and energy != "low":
                inten, pace = min(inten + 0.05, 0.5), pace * 1.01       # Arjun: a bit livelier, not cartoonish
            hero = intent in HERO_INTENTS and beat in HERO_BEATS
            if not hero:
                inten = min(inten, 0.35)
            gap = response_gap_ms(l, prev, intent, prev_intent, li == 0, lid)
            d = VoiceDirection(speaker_id=l.get("speaker", "narrator"), emotion=emo, intensity=round(inten, 2),
                               role="question" if l.get("kind") == "question" else "dialogue",
                               delivery=INTENT_WORDS[intent], pace_multiplier=round(min(max(pace, 0.9), 1.08), 3),
                               energy=energy, emphasis_words=pick_emphasis(l.get("tts") or l["text"], n=1),
                               pause_before_ms=gap, pause_after_ms=0, transition="gradual" if li else "shift",
                               source="dialogue", intent=intent)
            d.style_instruction = instruction(d)
            out[lid] = d
            prev, prev_intent = l, intent
    return out


def is_hero(line, shot, direction=None):
    """Lines that may get a second take: the opening, the central question, the reveal, the ending, and
    discovery/revelation moments."""
    return (shot.get("story_beat") in HERO_BEATS and line is shot["lines"][0]) or \
        (direction is not None and direction.intent in HERO_INTENTS)

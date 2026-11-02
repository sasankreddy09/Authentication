"""Character Bible: the two recurring friends of the storytelling videos.

Both characters are ORIGINAL artwork drawn by code (charanim.py) from the canonical parameters below — no image
model, no third-party character design, no real person's likeness — so the artwork is ours to use. Reference
images are rendered once per (character, expression, gesture) and stored with stable content-hash asset IDs: a
character is never "re-imagined" from a text prompt. Outfits/settings may vary per video; identity fields may not.
"""
from __future__ import annotations

import hashlib, json
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

EXPRESSIONS = ("neutral", "smile", "curious", "surprised", "confused", "thinking", "happy", "listening", "wink")
GESTURES = ("idle", "explain", "point", "think", "open_hands", "nod", "duck")   # duck: dodge something flying at you


class Voice(BaseModel):
    """One character's voice in one language. A preset of an approved model (no cloning, no real person)."""
    backend: Literal["kokoro", "parler", "qwen3-tts"]
    voice: str                                   # Kokoro voice / Parler trained speaker / Qwen speaker
    style: str = ""                              # Parler description of how they speak
    pitch: float = 0.0                           # identity pitch offset (semitones), never used for emotion
    qwen_speaker: Optional[str] = None           # Qwen3-TTS CustomVoice preset used when VOICE_BACKEND="qwen3-tts"
    gemini_voice: Optional[str] = None           # Gemini prebuilt voice (backend "gemini_free"); provisional until
    gemini_voice_status: str = ""                # "provisional" | "auditioned" (Telugu audition 2026-10-19: the user
                                                 # chose Arjun = Achird, Meera = Sulafat)
    emotion_control: Literal["none", "instruction", "description"] = "none"   # what the backend can follow
    rate_range: tuple[float, float] = (0.92, 1.08)                          # allowed pace multipliers
    intensity_range: tuple[float, float] = (0.1, 0.5)                       # validated expressive range
    accent: str = ""
    baseline: str = ""                           # baseline delivery, in words
    reference_sample: str = ""                   # speaker-verification reference (made on the first run)
    pronunciation: dict[str, str] = Field(default_factory=dict)            # written word → what the TTS reads


class Look(BaseModel):
    skin: str
    hair: str
    hair_style: Literal["short_messy", "long_tied", "bob", "short_neat"]
    top: str
    top_detail: str                              # collar / trim colour
    eyes: str = "#2B1D14"
    glasses: bool = False


class Character(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,23}$")
    name: str
    names: dict[str, str] = Field(default_factory=dict)         # per language: {"te": "అర్జున్"}
    role: Literal["curious", "expert"]
    personality: str
    look: Look
    voices: dict[str, Voice]                                    # per language code
    expressions: list[str] = Field(default_factory=lambda: list(EXPRESSIONS))
    gestures: list[str] = Field(default_factory=lambda: list(GESTURES))
    tag: str                                                    # detector query (scene checks)
    color: str                                                  # subtitle colour

    @field_validator("expressions")
    @classmethod
    def known_expr(cls, v):
        bad = [e for e in v if e not in EXPRESSIONS]
        if bad:
            raise ValueError(f"unknown expressions {bad}")
        return v

    @field_validator("gestures")
    @classmethod
    def known_gest(cls, v):
        bad = [g for g in v if g not in GESTURES]
        if bad:
            raise ValueError(f"unknown gestures {bad}")
        return v


class Bible(BaseModel):
    version: int = 1
    relationship: str
    characters: dict[str, Character]

    @field_validator("characters")
    @classmethod
    def one_of_each(cls, v):
        roles = sorted(c.role for c in v.values())
        if roles != ["curious", "expert"]:
            raise ValueError(f"the bible needs exactly one curious and one expert friend, got {roles}")
        for k, c in v.items():
            if k != c.id:
                raise ValueError(f"character key {k} != id {c.id}")
        return v

    def by_role(self, role):
        return next(c for c in self.characters.values() if c.role == role)


DEFAULT_BIBLE = {
    "version": 1,
    "relationship": "Close friends of the same age who hang out often. Arjun notices things and asks; Meera loves "
                    "explaining with real examples and small demonstrations. She never mocks him and never lectures "
                    "for long; he asks real follow-ups, sometimes with a plausible wrong guess, never just 'Wow'.",
    "characters": {
        "arjun": {
            "id": "arjun", "name": "Arjun", "names": {"te": "అర్జున్"}, "role": "curious",
            "personality": "curious, observant, relatable, occasionally funny; asks the question the viewer is thinking",
            "look": {"skin": "#C68B59", "hair": "#1F1A17", "hair_style": "short_messy", "top": "#F4A23B",
                     "top_detail": "#C97A12"},
            "voices": {"en": {"backend": "kokoro", "voice": "am_michael", "qwen_speaker": "aiden", "accent": "American",
                              "baseline": "friendly, curious, a little energetic; questions rise naturally, never cartoonish",
                              "rate_range": [0.95, 1.08], "intensity_range": [0.1, 0.5]},
                       "te": {"backend": "parler", "voice": "Prakash", "emotion_control": "description",
                              "gemini_voice": "Achird", "gemini_voice_status": "auditioned",
                              "style": "in a young, lively and curious voice at a moderate pace",
                              "baseline": "friendly and curious", "pronunciation": {"3D": "త్రీడీ"}}},
            "tag": "a young man in an orange hoodie", "color": "#FFD166"},
        "meera": {
            "id": "meera", "name": "Meera", "names": {"te": "మీరా"}, "role": "expert",
            "personality": "knowledgeable, friendly, patient; simplifies with familiar examples and demonstrations",
            "look": {"skin": "#A86B4A", "hair": "#2B1B14", "hair_style": "long_tied", "top": "#2E8C8C",
                     "top_detail": "#F2F2F2", "glasses": True},
            # Qwen3-TTS CustomVoice has no native-English female preset (only aiden/ryan are English natives):
            # Meera stays on Kokoro af_heart until a female preset is measured and approved (see the voice benchmark)
            "voices": {"en": {"backend": "kokoro", "voice": "af_heart", "accent": "American",
                              "baseline": "warm, patient, confident; calm explanations that brighten on discoveries",
                              "rate_range": [0.93, 1.05], "intensity_range": [0.1, 0.45]},
                       "te": {"backend": "parler", "voice": "Lalitha", "emotion_control": "description",
                              "gemini_voice": "Sulafat", "gemini_voice_status": "auditioned",
                              "style": "in a warm, clear and friendly explaining voice at a moderate pace",
                              "baseline": "warm and clear", "pronunciation": {"3D": "త్రీడీ"}}},
            "tag": "a young woman in a teal jacket", "color": "#7FDBDA"},
    },
}


def load(bible=None) -> Bible:
    return Bible.model_validate(bible or DEFAULT_BIBLE)


def asset_id(char: Character, expression: str, gesture: str, extra: str = "") -> str:
    """Stable ID of one reference image: same look + expression + gesture → same ID, on every machine."""
    key = json.dumps([char.id, char.look.model_dump(), expression, gesture, extra], sort_keys=True)
    return f"{char.id}_{expression}_{gesture}_{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def cast_characters(bible: Bible, language: str = "en") -> dict:
    """The plan's `characters` block (existing cast format) for this bible and language."""
    out = {}
    for c in bible.characters.values():
        v = c.voices.get(language)
        if v is None:
            raise ValueError(f"{c.name} has no voice for language '{language}' (add it to the Character Bible)")
        entry = {"name": c.names.get(language, c.name), "voice": v.voice, "speed": 1.0, "pitch": v.pitch,
                 "fx": "none", "color": c.color, "look": f"original 2D character {c.name}", "tag": c.tag,
                 "role": c.role}
        if v.style:
            entry["voice_style"] = v.style
        if v.qwen_speaker:
            entry["qwen_speaker"] = v.qwen_speaker
        if v.gemini_voice:
            entry["gemini_voice"] = v.gemini_voice
        out[c.id] = entry
    return out


def speak_text(text: str, voice: Voice) -> str | None:
    """The TTS string for a line from the character's pronunciation profile (None when it equals the text)."""
    out = text
    for w, say in voice.pronunciation.items():
        out = out.replace(w, say)
    return out if out != text else None

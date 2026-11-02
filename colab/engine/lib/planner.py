"""Topic → fact-checked script → plan JSON, with a LOCAL open model (no paid API).

The LLM only writes DATA (a Draft: title, hook candidates, beats with narration + cited fact ids + a visual intent).
Everything else is fixed, tested code:
  parse + validate (Pydantic, with the error fed back for up to 2 retries)
  → fact check every sentence against the facts it cites (factcheck.py), one rewrite for failures, then drop
  → choose the hook by rules (sourced, short, a question or a surprising fact, names the topic)
  → build the plan deterministically (cards / licensed images, claims with source URLs, music, metadata),
    which the notebook's existing validator then checks again.
Default model: Qwen/Qwen3-4B-Instruct-2507 (Apache-2.0, ~8 GB fp16 → fits a T4 on its own);
Qwen/Qwen3-1.7B for low memory. Both verified Apache-2.0 on the Hub.
"""
from __future__ import annotations

import json, re, textwrap
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
WORDS_PER_SECOND = 2.5           # Kokoro narration pace incl. pauses (measured on the existing videos ≈ 2.4–2.7)


# ------------------------------------------------------------------ the data the LLM writes
class Sentence(BaseModel):
    text: str
    kind: Literal["fact", "question", "analogy"] = "fact"
    fact_ids: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def short(cls, v):
        v = v.strip()
        if not v or len(v.split()) > 24:
            raise ValueError("each sentence must have 1–24 words")
        return v


class Visual(BaseModel):
    kind: Literal["photo", "card"]
    description: str = ""                      # photo: what the picture shows (concrete nouns)
    card_lines: list[str] = Field(default_factory=list)


class Beat(BaseModel):
    narration: list[Sentence] = Field(min_length=1, max_length=3)
    visual: Visual


class Draft(BaseModel):
    title: str
    hooks: list[Sentence] = Field(min_length=2, max_length=4)
    beats: list[Beat] = Field(min_length=3, max_length=10)
    takeaway: Sentence

    @field_validator("title")
    @classmethod
    def title_len(cls, v):
        if len(v.split()) > 10:
            raise ValueError("title: at most 10 words")
        return v.strip()


class Rewrite(BaseModel):
    text: str
    fact_ids: list[str] = Field(default_factory=list)       # missing → the original sentence's facts


MIN_LENGTH_FRACTION = 0.5        # a script under half the target length after the fact check is refused, not shipped


def symbols(text):
    """Notation a beginner can't read aloud or follow (M☉, ∝, ≈ ...)."""
    import unicodedata
    return sorted({ch for ch in text if unicodedata.category(ch) in ("So", "Sm") and ch not in "+=%°<>"})


def _content(text):
    return {w for w in re.findall(r"[a-z]{4,}", text.lower())}


def similar(a, b):
    """Word overlap (Jaccard) of two sentences' content words."""
    x, y = _content(a), _content(b)
    return len(x & y) / max(len(x | y), 1)


SYSTEM = """You write scripts for short educational explainer videos for complete beginners.
Rules:
- Use ONLY the numbered facts given. Every "fact" sentence must be fully supported by the facts it cites in fact_ids.
- Never add a number, name, date or claim that is not in the cited facts. If unsure, leave it out.
- Simple words. Explain an unfamiliar term in the same sentence. At most 20 words per sentence.
- An everyday comparison is allowed as kind "analogy" (it must not add new facts). Questions use kind "question".
- Build curiosity: question → explanation → payoff. No greetings, no filler, no repetition.
- Answer with ONE JSON object and nothing else."""

SCHEMA_HINT = """{
  "title": "short title, max 8 words",
  "hooks": [{"text": "a gripping opening question or surprising sourced fact", "kind": "question|fact", "fact_ids": ["F2"]}, ... 3 candidates],
  "beats": [
    {"narration": [{"text": "...", "kind": "fact|question|analogy", "fact_ids": ["F1"]}],
     "visual": {"kind": "photo", "description": "what a real photo/illustration should show, concrete nouns"}},
    {"narration": [...], "visual": {"kind": "card", "card_lines": ["max 6 words", "max 6 words"]}}
  ],
  "takeaway": {"text": "one closing sentence", "kind": "fact", "fact_ids": ["F3"]}
}"""


def draft_prompt(topic, facts, target_s):
    words = int(target_s * WORDS_PER_SECOND)
    lines = "\n".join(f"{f.id}: {f.text}" for f in facts)
    return (f"Topic: {topic}\nTarget length: about {target_s} seconds = about {words} spoken words in total, "
            f"in 5–7 beats of 1–2 sentences.\nUse "
            f"\"photo\" visuals for things a real photo or illustration can show, \"card\" for numbers, steps or "
            f"comparisons.\n\nFacts:\n{lines}\n\nReturn JSON in exactly this shape:\n{SCHEMA_HINT}")


# ------------------------------------------------------------------ the local model
class LocalLLM:
    def __init__(self, model_id=DEFAULT_MODEL, device="cuda", cache_dir=None, log=print):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device, self.log, self.model_id = torch, device, log, model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        dt = torch.float16 if device == "cuda" else torch.float32
        kw = {"device_map": {"": 0}} if device == "cuda" else {}
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, cache_dir=cache_dir, **kw).eval()
        if device != "cuda":
            self.model.to(device)

    def chat(self, system, user, max_new_tokens=1600):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        ans = self.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return re.sub(r"(?s)<think>.*?</think>", "", ans).strip()


def parse_json(text, model_cls):
    """The first complete JSON object in the answer, validated by the Pydantic model."""
    s = text.find("{")
    if s < 0:
        raise ValueError("no JSON object in the answer")
    depth, inside, esc = 0, False, False
    for i, ch in enumerate(text[s:], s):
        if inside:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                inside = False
            continue
        if ch == '"':
            inside = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return model_cls.model_validate(json.loads(text[s:i + 1]))
    raise ValueError("the JSON object is not closed (answer cut off?)")


def ask_json(llm, system, user, model_cls, tries=3, log=print, max_new_tokens=1600):
    err = None
    for k in range(tries):
        prompt = user if err is None else (f"{user}\n\nYour previous answer was invalid: {err}\n"
                                           "Return corrected JSON only.")
        ans = llm.chat(system, prompt, max_new_tokens)
        try:
            return parse_json(ans, model_cls)
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            err = str(e)[:600]
            log(f"planner: answer {k + 1} invalid: {err[:200]}")
    raise RuntimeError(f"the planner model gave no valid JSON after {tries} tries: {err}")


# ------------------------------------------------------------------ fact-checked script
def make_script(topic, sheet, llm, nli, target_s=60, log=print, max_facts=30):
    """Draft → fact check → one rewrite → drop what is still unsupported. Returns (script dict, checks dict)."""
    import factcheck
    facts = sheet.facts[:max_facts]
    by_id = {f.id: f for f in facts}
    draft = ask_json(llm, SYSTEM, draft_prompt(topic, facts, target_s), Draft, log=log)
    sentences = []                                             # (key, Sentence)
    for i, h in enumerate(draft.hooks):
        sentences.append((f"hook{i}", h))
    for b, beat in enumerate(draft.beats):
        for j, s in enumerate(beat.narration):
            sentences.append((f"b{b}s{j}", s))
    sentences.append(("take", draft.takeaway))

    def items(pairs):
        out = []
        for key, s in pairs:
            ids = [x for x in s.fact_ids if x in by_id]
            out.append({"id": key, "text": s.text, "question": s.kind == "question" or s.text.endswith("?"),
                        "analogy": s.kind == "analogy",
                        "evidence": [by_id[x].text for x in ids] or ([] if s.kind == "fact" else [f.text for f in facts[:8]])})
        return out

    checks = factcheck.check_sentences(items(sentences), nli, topic)
    for key, s in sentences:                                 # beginner readability: no unexplained notation
        if checks[key]["status"] == "PASS" and symbols(s.text):
            checks[key].update(status="FAIL", why=f"symbols a beginner can't read: {' '.join(symbols(s.text))}")
    bad = [(k, s) for k, s in sentences if checks[k]["status"] != "PASS" and not k.startswith("hook")]
    for key, s in bad:                                         # one rewrite, restricted to the sentence's facts
        ids = [x for x in s.fact_ids if x in by_id] or [f.id for f in facts[:6]]
        user = ("Rewrite this sentence for a beginner so that it says ONLY what these facts say. Keep it short.\n"
                f"Sentence: {s.text}\nFacts:\n" + "\n".join(f"{x}: {by_id[x].text}" for x in ids) +
                '\nReturn JSON: {"text": "...", "fact_ids": ["..."]}')
        try:
            rw = ask_json(llm, SYSTEM, user, Rewrite, tries=2, log=log, max_new_tokens=200)
        except RuntimeError:
            continue
        # the rewrite must stay on the SAME facts: citing other facts would change what the beat says
        new = Sentence(text=rw.text, kind="fact", fact_ids=[x for x in rw.fact_ids if x in ids] or ids)
        r = factcheck.check_sentences(items([(key, new)]), nli, topic)[key]
        if r["status"] == "PASS" and symbols(new.text):
            r.update(status="FAIL", why=f"symbols a beginner can't read: {' '.join(symbols(new.text))}")
        r["rewritten_from"] = s.text
        checks[key] = r
        if r["status"] == "PASS":
            _replace(draft, key, new)
    # hook: the best SUPPORTED candidate (questions and sourced facts), short, naming the topic
    tw = set(re.findall(r"[a-z]{4,}", topic.lower()))
    def hook_score(i, h):
        c = checks[f"hook{i}"]
        return (c["status"] == "PASS", h.kind == "question" or bool(re.search(r"\d", h.text)),
                bool(tw & set(re.findall(r"[a-z]{4,}", h.text.lower()))), -len(h.text.split()))
    hi, hook = max(enumerate(draft.hooks), key=lambda x: hook_score(*x))
    if checks[f"hook{hi}"]["status"] != "PASS":
        raise RuntimeError("no hook candidate is supported by the sources; re-run or choose another topic")
    beats, said = [], [hook.text]
    for b, beat in enumerate(draft.beats):
        keep = []
        for j, s in enumerate(beat.narration):
            key = f"b{b}s{j}"
            if checks[key]["status"] != "PASS":
                log(f"planner: dropped unsupported sentence in beat {b + 1}: {s.text!r} ({checks[key]['why']})")
            elif any(similar(s.text, x) >= 0.6 for x in said) or (
                    s.kind == "fact" and any(factcheck.same_claim(nli, s.text, x) for x in said)):   # a repeat
                checks[key].update(status="DROPPED", why="repeats an earlier sentence")
                log(f"planner: dropped repetition in beat {b + 1}: {s.text!r}")
            else:
                keep.append(s); said.append(s.text)
        if keep:
            beats.append({"narration": [s.model_dump() for s in keep], "visual": beat.visual.model_dump()})
    take = draft.takeaway.model_dump() if checks["take"]["status"] == "PASS" and not any(
        similar(draft.takeaway.text, x) >= 0.6 for x in said) else None
    words = sum(len(s["text"].split()) for b in beats for s in b["narration"]) + len(hook.text.split()) + \
        (len(take["text"].split()) if take else 0)
    target_words = target_s * WORDS_PER_SECOND
    if len(beats) < 2 or words < MIN_LENGTH_FRACTION * target_words:
        raise RuntimeError(f"only {words} words ({len(beats)} beats) survived the fact check, for a target of "
                           f"~{target_words:.0f}: too thin to publish. Use the 4B planner model, a shorter target, "
                           "or a topic Wikipedia covers better.")
    script = {"title": draft.title, "hook": hook.model_dump(), "beats": beats, "takeaway": take,
              "model": getattr(llm, "model_id", "?")}
    return script, checks


def _replace(draft, key, new):
    if key == "take":
        draft.takeaway = new
    elif key.startswith("b"):
        b, j = (int(x) for x in re.findall(r"\d+", key))
        draft.beats[b].narration[j] = new


# ------------------------------------------------------------------ deterministic plan builder
MOODS = {"curious": {"pause_after": 0.55, "grade": "cool", "move_amount": 0.06,
                     "music": {"base": 0.5, "curious": 0.7, "tension": 0.1, "shimmer": 0.4}},
         "calm": {"pause_after": 0.5, "grade": "neutral", "move_amount": 0.05,
                  "music": {"base": 0.4, "curious": 0.8, "tension": 0.0, "shimmer": 0.2}},
         "reflective": {"pause_after": 0.6, "grade": "neutral", "move_amount": 0.05,
                        "music": {"base": 0.5, "curious": 0.6, "tension": 0.0, "shimmer": 0.4}}}
TONES = {"curious": {"speed": 0.96, "pitch": 0.2, "gain_db": 0.0, "pause_after": 0.55},
         "explainer": {"speed": 1.0, "pitch": 0.0, "gain_db": 0.0, "pause_after": 0.45},
         "warm": {"speed": 0.95, "pitch": 0.0, "gain_db": -0.5, "pause_after": 0.6}}
LOOK = {"bg_top": "#10233F", "bg_bottom": "#1B4060", "accent": "#19C3A6", "accent2": "#FF6B6B", "gold": "#FFC43D"}


def card_lines(text, n=3, width=6):
    """Up to n short lines (≤ 9 words each, the card limit) of WHOLE clauses from a sentence: clauses are joined
    until there are ≥ 4 words ("However," alone isn't a card), and a clause that doesn't fit is left out rather
    than cut mid-phrase."""
    clauses, cur = [], []
    for part in re.split(r"(?<=[,;:])\s+", re.sub(r"[.!]$", "", text.strip())):
        cur += part.rstrip(",;:").split()
        if len(cur) >= 4:
            clauses.append(cur); cur = []
    if cur:
        clauses.append(cur)
    words = []
    for c in clauses:
        if len(words) + len(c) > n * width and words:
            break
        words += c
    words = words[:n * width]
    return [" ".join(words[i:i + width]) for i in range(0, len(words), width)]


def card_supported(line, narration):
    """On-screen text must say what the narration says: its numbers and names appear in the narration, and it
    shares at least one content word with it ("Nothing escapes" on a beat about gas clouds is refused)."""
    import factcheck
    bad_nums, bad_names = factcheck.static_checks(line, narration)
    words = {w for w in re.findall(r"[a-z]{4,}", line.lower())}
    said = {w for w in re.findall(r"[a-z]{4,}", narration.lower())}
    stem = lambda w: w[:5]
    return not bad_nums and not bad_names and bool({stem(w) for w in words} & {stem(w) for w in said})


def build_plan(topic, script, sheet, beat_assets, voice="af_heart", project=None):
    """Script + chosen images → the cast-format plan the notebook renders. beat_assets[i]: manifest entry or None."""
    by_id = sheet.by_id()
    used, shots = [], []

    def line(s, tone):
        ids = [x for x in s.get("fact_ids", []) if x in by_id]
        used.extend(ids)
        return {"speaker": "narrator", "tone": tone, "text": s["text"], **({"story": True} if s["kind"] != "fact" else {})}, ids

    h, ids = line(script["hook"], "curious")
    shots.append({"shot_id": "sh01", "chapter": "hook", "recipe": "card", "mood": "curious", "seed": 1,
                  "card": {"icon": "question", "lines": card_lines(script["hook"]["text"], 2, 7), "bullets": False},
                  "lines": [h], "claims": ids, "sfx": [{"type": "pop", "at": 0.5, "dur": 0.4, "gain_db": -14}]})
    for i, beat in enumerate(script["beats"]):
        sid = f"sh{len(shots) + 1:02d}"
        tone = "curious" if beat["narration"][0]["kind"] == "question" else "explainer"
        ls, cid = [], []
        for s in beat["narration"]:
            l, ids = line(s, tone); ls.append(l); cid += ids
        shot = {"shot_id": sid, "chapter": f"beat{i + 1}", "mood": "calm" if i % 2 else "curious", "seed": i + 2,
                "lines": ls, "claims": sorted(set(cid))}
        a = beat_assets[i] if i < len(beat_assets) else None
        if a is not None:
            shot.update(recipe="stock", move="push_in" if i % 2 else "pan_right",
                        stock={"file": a["file"], "credit": a["credit_line"], "license": a["license"],
                               "license_ok": a["license_ok"], "asset": a["id"], "page": a["page"],
                               "description": beat["visual"].get("description", "")})
        else:
            said = " ".join(x["text"] for x in beat["narration"])
            lines = [ln for ln in beat["visual"].get("card_lines", []) if 0 < len(ln.split()) <= 9][:4]
            if not lines or not all(card_supported(ln, said) for ln in lines):
                lines = card_lines(beat["narration"][0]["text"])      # text taken from what is actually said
            shot.update(recipe="card", card={"lines": lines, "bullets": len(lines) > 2})
        shots.append(shot)
    if script.get("takeaway"):
        t, ids = line(script["takeaway"], "warm")
        shots.append({"shot_id": f"sh{len(shots) + 1:02d}", "chapter": "payoff", "recipe": "card", "mood": "reflective",
                      "seed": 99, "card": {"icon": "check", "lines": card_lines(script["takeaway"]["text"]),
                                           "bullets": False}, "lines": [t], "claims": ids})
    claims = [{"id": f.id, "text": f.text, "sources": [f.source_url], "confidence": "sourced",
               "source_revision": f.revid} for f in sheet.facts if f.id in set(used)]
    images = [a for a in beat_assets if a]
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:40]
    return {
        "plan_version": 1, "project": project or f"topic_{slug}", "language": "en", "title": script["title"],
        "logline": f"Auto-generated explainer: {topic}", "made_by": f"local planner ({script.get('model')})",
        "style": {"suffix": "clean educational explainer", "negative": "text, watermark, logo"},
        "look": LOOK, "characters": {"narrator": {"name": "Narrator", "voice": voice, "speed": 1.0, "pitch": 0,
                                                 "fx": "none", "color": "#FFFFFF", "look": None}},
        "tones": TONES, "moods": MOODS, "claims": claims, "entities": {}, "glossary": {},
        "vocabulary": sorted({w for f in sheet.facts for w in re.findall(r"\b[A-Z][a-z]{3,}\b", f.text)})[:40],
        "music": {"cues": [{"from": "sh01", "track": "Investigations"}, {"from": "sh02", "track": "Easy Lemon"},
                           {"from": shots[-1]["shot_id"], "track": "Carefree"}]},
        "shots": shots,
        "sources": sheet.sources,
        "metadata": {
            "title": script["title"],
            "summary": " ".join(s["text"] for b in script["beats"] for s in b["narration"])[:600],
            "disclaimer": ("Automatically researched from Wikipedia and fact-checked against the cited sentences by a "
                           "local model; simplified for beginners. Check the sources listed below."),
            "category": "Education", "audience": "Not made for kids",
            "tags": sorted(set(re.findall(r"[a-z]{4,}", topic.lower())))[:10],
            "sources": [s["url"] for s in sheet.sources],
            "image_credits": [a["credit_line"] + f" ({a['page']})" for a in images]}}


def script_text(script):
    """Readable script (for the person reviewing it)."""
    out = [script["title"].upper(), "", "HOOK: " + script["hook"]["text"], ""]
    for i, b in enumerate(script["beats"], 1):
        v = b["visual"]
        out.append(f"[{i}] " + " ".join(s["text"] for s in b["narration"]))
        out.append("     visual: " + (v.get("description") or " / ".join(v.get("card_lines", []))))
    if script.get("takeaway"):
        out += ["", "TAKEAWAY: " + script["takeaway"]["text"]]
    return "\n".join(textwrap.fill(x, 110, subsequent_indent="     ") if x else x for x in out)

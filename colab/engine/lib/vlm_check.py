"""Video-language-model event checks: did each required action actually happen, and when?

The VLM (default Qwen/Qwen3-VL-2B-Instruct, Apache-2.0; its technical report describes explicit text timestamps for
temporal grounding) gets sampled frames WITH their times and one narrow yes/no question per event. The answer must
be strict JSON with a time and what it saw; anything else becomes UNKNOWN, never PASS. Geometric checks
(motion_check.py) stay the primary evidence for positions and directions; the VLM covers what boxes can't see
(a hand placing a small case, a scan) and is cross-checked in acceptance.py.

question()/parse() are pure Python and unit-tested; QwenJudge needs a GPU (not testable on the CPU machine).
"""
from __future__ import annotations

import json, re

VERSION = 1
QUESTION = """You are checking an AI-generated video clip against a storyboard. Frames are listed with their times.
Question: {q}
Answer ONLY with JSON: {{"happened": "yes" | "no" | "unsure", "start_s": number or null, "end_s": number or null,
"evidence": "one short sentence describing what you see"}}"""


def question(event, names: dict[str, str]) -> str:
    who = names.get(event.actor, event.actor)
    obj = names.get(event.obj, event.obj) if event.obj else ""
    tgt = names.get(event.target, event.target) if event.target else ""
    return {
        "enter": f"Does {who} enter the frame?",
        "walk_to": f"Does {who} walk towards {tgt or 'the destination'} and arrive there?",
        "stop": f"Does {who} come to a stop?",
        "take_out": f"Does {who} take {obj} out of a bag or pocket?",
        "place": f"Does {who} put {obj} down on {tgt}, and does it stay there?",
        "pick_up": f"Does {who} pick up {obj}?",
        "hand_over": f"Does {who} hand {obj} to {tgt}?",
        "scan": f"Does {who} scan {obj} with a handheld scanner?",
        "open": f"Does {who} open {obj}?",
        "sit": f"Does {who} sit down?", "stand": f"Does {who} stand up?",
        "look_at": f"Does {who} look at {obj or tgt}?",
        "talk": f"Does {who} speak (mouth moving)?",
        "exit": f"Does {who} leave the frame?",
    }.get(event.type, event.describe or f"Does {who} {event.type.replace('_', ' ')}?")


def parse(text: str):
    """Strict parse of the VLM answer -> (status, evidence). Malformed or hedged answers -> UNKNOWN."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return "UNKNOWN", {"raw": (text or "")[:200], "why": "no JSON in the answer"}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "UNKNOWN", {"raw": text[:200], "why": "invalid JSON"}
    h = str(d.get("happened", "")).strip().lower()
    ev = {"start_s": d.get("start_s"), "end_s": d.get("end_s"), "evidence": str(d.get("evidence", ""))[:200]}
    if h == "yes" and isinstance(d.get("end_s"), (int, float)) and ev["evidence"]:
        return "PASS", ev
    if h == "no" and ev["evidence"]:
        return "FAIL", ev
    return "UNKNOWN", dict(ev, why="unsure, or 'yes' without a time and evidence")


class QwenJudge:
    """Qwen3-VL on the GPU. Frames are passed as images, each preceded by its time."""
    MODEL = "Qwen/Qwen3-VL-2B-Instruct"

    def __init__(self, device="cuda", cache_dir=None, model=None):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch, self.device = torch, device
        mid = model or self.MODEL
        self.proc = AutoProcessor.from_pretrained(mid, cache_dir=cache_dir)
        from trackers import load_on            # straight onto the GPU: no 4.3 GB copy in system RAM
        self.model = load_on(AutoModelForImageTextToText, mid, device,
                             torch_dtype=torch.float16 if device == "cuda" else torch.float32, cache_dir=cache_dir)

    def ask(self, frames, times, q: str, max_new_tokens=120) -> str:
        from PIL import Image
        content = []
        for f, t in zip(frames, times):
            content += [{"type": "text", "text": f"t={t:.1f}s"}, {"type": "image", "image": Image.fromarray(f)}]
        content.append({"type": "text", "text": QUESTION.format(q=q)})
        inp = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                            add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]

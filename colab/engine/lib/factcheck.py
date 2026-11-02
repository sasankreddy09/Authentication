"""Is each narrated sentence supported by the facts it cites? Two deterministic gates + one model gate.

  1. numbers: every number in the sentence (digits, "^" exponents, number words) appears in its evidence;
  2. names: every proper noun / acronym in the sentence appears in its evidence (or the topic);
  3. entailment: an NLI cross-encoder (cross-encoder/nli-deberta-v3-base, Apache-2.0, runs on the CPU) reads
     evidence → sentence: PASS if entailed, FAIL if contradicted, UNKNOWN if neither.
Questions (hooks) make no claim: only gates 1–2 apply. Analogies: gates 1–2 + they must be worded as a comparison
(NLI can't judge them: a correct analogy is "about something else", which it scores as contradiction). FAIL and UNKNOWN both block automatic approval; the planner
gets one rewrite, then the sentence is dropped. No model can guarantee a claim is true — this guarantees an
unsupported claim can't reach an approved render silently. Thresholds are settings (TH), to be calibrated.
"""
from __future__ import annotations

import re

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"     # chosen on a labelled set: 13/14, 0 false passes (small: 12/14)
TH = {"entail_min": 0.5, "contradict_max": 0.5}
NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
             "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000,
             "dozen": 12, "half": 0.5}
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
GENERIC = {"I", "We", "You", "It", "This", "That", "These", "Those", "There", "Here", "What", "Why", "How", "When",
           "Where", "Which", "Who", "But", "And", "So", "Then", "Now", "Imagine", "Think", "Ever", "Picture"}


def numbers(text):
    """Values mentioned: '2×10^30' → 2e30, '4.6 billion' → 4.6e9, '1,045' → 1045, 'three' → 3. Years kept as is."""
    t = text.replace("−", "-").replace(",", "")
    out = set()
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:×|x|\*)\s*10\^(-?\d+)", t):
        out.add(float(m.group(1)) * 10 ** int(m.group(2)))
    t2 = re.sub(r"(\d+(?:\.\d+)?)\s*(?:×|x|\*)\s*10\^(-?\d+)", " ", t)
    for m in re.finditer(r"10\^(-?\d+)", t2):
        out.add(10.0 ** int(m.group(1)))
    t2 = re.sub(r"10\^(-?\d+)", " ", t2)
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(thousand|million|billion|trillion)?", t2, re.I):
        v = float(m.group(1)) * (SCALE[m.group(2).lower()] if m.group(2) else 1)
        out.add(v)
    for w, v in NUM_WORDS.items():
        for m in re.finditer(rf"\b{w}\b\s*(thousand|million|billion|trillion)?", t2, re.I):
            out.add(v * (SCALE[m.group(1).lower()] if m.group(1) else 1))
    return out


def names(text):
    """Proper nouns and acronyms, not counting the first word of a sentence or generic capitalised words."""
    found = set()
    for sent in re.split(r"(?<=[.!?])\s+", text):
        toks = re.findall(r"[A-Za-z][A-Za-z'*-]*", sent)
        for i, w in enumerate(toks):
            if (w.isupper() and len(w) >= 2) or (i > 0 and w[0].isupper() and w not in GENERIC):
                found.add(w.strip("'s").rstrip("'"))
    return found


def _close(a, b):
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1) or (b and abs(a / b - 1) < 0.02)


def static_checks(sentence, evidence, topic=""):
    ev_nums, ev_low = numbers(evidence), (evidence + " " + topic).lower()
    bad_nums = sorted(n for n in numbers(sentence) if not any(_close(n, e) for e in ev_nums))
    bad_names = sorted(n for n in names(sentence) if n.lower() not in ev_low)
    return bad_nums, bad_names


class NLI:
    def __init__(self, model=NLI_MODEL, device="cpu", cache_dir=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, cache_dir=cache_dir).to(device).eval()
        lab = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.ix = (lab["entailment"], lab["contradiction"], lab["neutral"])

    def scores(self, pairs):
        """[(premise, hypothesis)] → [(p_entail, p_contradict, p_neutral)], batched."""
        if not pairs:
            return []
        with self.torch.inference_mode():
            enc = self.tok([p for p, _ in pairs], [h for _, h in pairs], truncation=True, max_length=512,
                           padding=True, return_tensors="pt").to(self.device)
            pr = self.model(**enc).logits.softmax(-1).cpu().tolist()
        e, c, n = self.ix
        return [(r[e], r[c], r[n]) for r in pr]


def check_sentences(items, nli, topic="", th=TH):
    """items: [{"id", "text", "evidence": [fact texts], "question": bool}] → {id: {"status", "why", "scores"...}}."""
    out, todo = {}, []
    for it in items:
        ev = " ".join(it["evidence"])
        bad_nums, bad_names = static_checks(it["text"], ev, topic)
        r = {"text": it["text"], "evidence": it["evidence"], "bad_numbers": bad_nums, "unsupported_names": bad_names}
        if bad_nums or bad_names:
            r.update(status="FAIL", why="numbers/names not in the cited facts")
        elif it.get("question"):
            r.update(status="PASS", why="question: makes no claim; numbers and names are sourced")
        elif it.get("analogy"):
            # a comparison is about something ELSE, which NLI reads as "contradiction" (measured: 0.84 for a correct
            # waterfall analogy), so NLI can't judge it. It must SAY it's a comparison, and gates 1–2 already keep
            # out new numbers and names, so no factual claim can be smuggled in this way.
            if re.search(r"\b(like|as if|as though|imagine|think of|picture|similar to|compare|what if|suppose)\b", it["text"], re.I):
                r.update(status="PASS", why="analogy: marked as a comparison, no unsourced numbers or names")
            else:
                r.update(status="FAIL", why="called an analogy but not worded as a comparison ('like', 'imagine'...)")
        elif not it["evidence"]:
            r.update(status="FAIL", why="a statement with no cited fact")
        else:
            todo.append((it["id"], ev, it["text"]))
        out[it["id"]] = r
    for (sid, ev, text), (pe, pc, pn) in zip(todo, nli.scores([(ev, t) for _, ev, t in todo])):
        r = out[sid]
        r["scores"] = {"entail": round(pe, 3), "contradict": round(pc, 3), "neutral": round(pn, 3)}
        if pc >= th["contradict_max"]:
            r.update(status="FAIL", why="contradicted by its cited facts")
        elif pe >= th["entail_min"]:
            r.update(status="PASS", why="entailed by its cited facts")
        else:
            r.update(status="UNKNOWN", why="not clearly supported by its cited facts")
    return out


def same_claim(nli, a, b, th=0.9):
    """A repeat for the viewer: either sentence strongly entails the other (measured: a reworded repeat scores
    1.00 one way and 0.01 the other, because it adds a detail; unrelated sentences score 0.00 both ways)."""
    (e1, _, _), (e2, _, _) = nli.scores([(a, b), (b, a)])
    return max(e1, e2) >= th

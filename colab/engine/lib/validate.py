"""Scene validation: checks a video plan BEFORE any voice or GPU work.

Pure Python (no Colab, no GPU) so it is unit-tested locally (tests/test_validate.py).
validate(plan, settings) -> list of issues; each issue is a dict:
    {"level": "error" | "warn", "rule": "P2", "shot": "sh05", "message": "...", "fix": "..."}
Errors stop the run before anything is generated (the plan must be fixed); warnings are reported.

Rule groups (the letters are used in reports and in the documentation):
  S  storytelling / engagement   (research: Mayer's multimedia principles, Guo et al. 2014)
  P  process and factual accuracy (claims with sources, conditional steps, numbers, invented internals)
  A  narration ↔ visual alignment (what is said must be what is shown)
  V  visual consistency           (characters, props, places, time)
"""
import re

# ---------------------------------------------------------------- vocab
HEDGES = {"if", "may", "might", "some", "sometimes", "depending", "depends", "usually", "can", "could",
          "often", "eligible", "unless", "when", "either", "or", "most", "many", "certain", "where"}
# Telugu: conditionals and "may/can" are word ENDINGS (ఉంటే = "if it is", జరగొచ్చు = "may happen"), plus a few words
TE_HEDGE_WORDS = {"ఒకవేళ", "కొన్నిసార్లు", "సాధారణంగా", "బహుశా", "లేదా", "కొన్ని"}
TE_HEDGE_ENDINGS = ("తే", "టే", "చ్చు")


def hedged(text):
    """Does the narration word a conditional fact conditionally? (English words, or Telugu words / endings)"""
    ws = set(words(text))
    return bool((content_words(text) | ws) & HEDGES or ws & TE_HEDGE_WORDS
                or any(w.endswith(TE_HEDGE_ENDINGS) for w in ws))
TIME_UNITS = (r"(seconds?|minutes?|hours?|days?|business days?|working days?|weeks?|months?"
              r"|సెకన్|నిమిష|గంట|రోజు|రోజుల|వారం|వారాల|నెల)")      # + Telugu stems (suffixes follow: రోజుల ముందు)
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30}
# phrases that claim knowledge of a company's internal systems; allowed only with a cited claim
INTERNAL_DETAIL = ["algorithm", "database", "server", "servers", "machine learning", "ai model", "microservice",
                   "internal system", "backend", "kafka", "redis", "api", "queue", "neural"]
# words a non-technical viewer may not know; each must be in the plan's glossary (explained with an analogy)
JARGON = ["acquirer", "issuer", "issuing bank", "card network", "payment gateway", "payment processor",
          "settlement", "clearing", "authorization", "authorisation", "chargeback", "ledger", "ach", "neft",
          "upi", "arn", "acquirer reference number", "reversal", "batch", "interchange", "api", "webhook",
          "idempotency", "database", "microservice", "transaction id"]
STOP = set("""a an the and or but if then so of to in on at by for with from as is are was were be been being it its
this that these those you your yours he she they them his her their we our us i me my not no yes do does did done
can could may might will would shall should just only also very more most some any each every all into out up down
over under again there here when where why how what which who whom whose than too now then still yet even""".split())

DRAWN = {"diagram", "ui", "journey", "card", "story", "character", "demo"}


import unicodedata
# a phrase can't start right after a letter/digit of any script or an Indic vowel sign (marks aren't \w in Python)
_NOT_START = r"[\w\u0300-\u036F\u0900-\u0DFF]"


def words(text):
    """Words of any script ("Riya's" → riya's; Telugu keeps its vowel signs)."""
    t = "".join(ch if ch == "'" or unicodedata.category(ch)[0] in "LMN" else " " for ch in text.lower())
    return [w for w in t.split() if w.strip("'")]


def content_words(text):
    return {w.strip("'") for w in words(text) if w not in STOP and len(w) > 2}


def numbers_in(text):
    """Numbers used with a time unit, e.g. '3-5 business days' -> {3, 5}; 'ten business days' -> {10}."""
    t = text.lower()
    found = set()
    for m in re.finditer(r"(\d+)\s*(?:-|–|to)\s*(\d+)\s*" + TIME_UNITS, t):
        found |= {int(m.group(1)), int(m.group(2))}
    for m in re.finditer(r"(\d+)\s*" + TIME_UNITS, t):
        found.add(int(m.group(1)))
    for w, n in NUMBER_WORDS.items():
        if re.search(rf"\b{w}\b\s*" + TIME_UNITS, t):
            found.add(n)
    return found


def all_numbers(text):
    t = text.lower()
    return {int(x) for x in re.findall(r"\d+", t)} | {n for w, n in NUMBER_WORDS.items() if re.search(rf"\b{w}\b", t)}


def has_phrase(text, phrase):
    """Whole-word match. Telugu attaches case endings to words (PNRలో = "in the PNR", రియాకి = "to Riya"), so after a
    phrase only a following LATIN letter/digit breaks the match (English "car" still doesn't match "cart")."""
    phrase = phrase.lower()
    right = r"(?![a-z0-9])" if phrase[-1:].isascii() else ""
    return re.search(rf"(?<!{_NOT_START}){re.escape(phrase)}{right}", text.lower()) is not None


def visual_text(shot, plan):
    """Everything the viewer SEES in a shot, as text (for alignment checks)."""
    parts = [shot.get("scene", ""), shot.get("action", ""), shot.get("shows", ""), shot.get("label", ""),
             shot.get("title", "")]
    chars = plan.get("characters", {})
    for c in shot.get("cast", []):
        parts += [chars.get(c, {}).get("look") or "", chars.get(c, {}).get("tag") or "", chars.get(c, {}).get("name", "")]
    props = plan.get("props", {})
    for p in shot.get("props", []):
        parts += [props.get(p, {}).get("look", ""), p]
    for key in ("ui", "journey", "card", "diagram", "stock", "character", "demo"):
        if key in shot:
            parts.append(_flatten(shot[key]))
    for v in (shot.get("story") or {}).get("visuals", []):        # two-friend scenes: every visual of the scene
        parts += [v.get("description", ""), v.get("intent", ""), _flatten(v.get(v.get("recipe"), {}))]
        if v.get("recipe") == "demo":
            parts.append(DEMO_SHOWS.get(v["demo"].get("template"), ""))
    if shot.get("recipe") == "journey":
        st = {s["id"]: s for s in plan.get("journey", {}).get("stations", [])}
        for sid in shot.get("journey", {}).get("path", []):
            parts.append(st.get(sid, {}).get("label", "") + " " + st.get(sid, {}).get("icon", ""))
    return " ".join(parts)


# what each drawn demonstration shows (demos.py), for the narration ↔ visual check
DEMO_SHOWS = {"finger_test": "thumb finger hand left eye right eye close one eye jumps near far window",
              "eye_views": "two eyes left eye right eye two pictures ball tree brain one picture depth",
              "projector_glasses": "projector screen two pictures light glasses lenses filter left eye right eye "
                                   "polarized polarization twist"}


def _flatten(x):
    if isinstance(x, dict):
        return " ".join(_flatten(v) for v in x.values())
    if isinstance(x, list):
        return " ".join(_flatten(v) for v in x)
    return str(x)


def spoken(shot):
    return " ".join(l.get("text", "") for l in shot.get("lines", []))


def entity_hits(text, entities):
    """Which plan entities (with synonyms) a text mentions."""
    out = set()
    for name, syns in entities.items():
        if any(has_phrase(text, s) for s in [name] + list(syns)):
            out.add(name)
    return out


# ---------------------------------------------------------------- the checks
def validate(plan, settings=None):
    s = {"words_per_second": 2.6, "max_sentence_words": 22, "max_label_words": 6, "max_title_words": 9,
         "max_diagram_share": 0.25, "max_diagram_run": 2, "max_same_visual_run": 3, "hook_seconds": 10,
         "wpm_range": (125, 185)}
    s.update(plan.get("rules", {}))
    s.update(settings or {})
    issues = []

    def add(level, rule, shot, msg, fix=""):
        issues.append({"level": level, "rule": rule, "shot": shot, "message": msg, "fix": fix})

    shots = plan.get("shots", [])
    if not shots:
        add("error", "S0", None, "the plan has no shots")
        return issues
    ids = [x["shot_id"] for x in shots]
    claims = {c["id"]: c for c in plan.get("claims", [])}
    steps = {st["id"]: st for st in plan.get("process", [])}
    step_order = [st["id"] for st in plan.get("process", [])]
    entities = plan.get("entities", {})
    glossary = {k.lower(): v for k, v in plan.get("glossary", {}).items()}
    company = plan.get("real_company")                      # topic is a real company: no invented internals

    # ---- S: storytelling and engagement
    first_lines = " ".join(spoken(x) for x in shots[:2])
    if "?" not in first_lines and not shots[0].get("hook"):
        add("warn", "S1", ids[0], "the first two shots don't pose a question (no hook)",
            "open with the viewer's problem as a question, e.g. 'So why is the money not there yet?'")
    recipes = [x.get("recipe", "video") for x in shots]
    n_dg = sum(r == "diagram" for r in recipes)
    if n_dg / len(shots) > s["max_diagram_share"]:
        add("error", "S2", None, f"{n_dg}/{len(shots)} shots are box-and-arrow diagrams (max "
            f"{s['max_diagram_share']:.0%}): the video becomes a flowchart presentation",
            "show the process with people, UI screens and the journey recipe instead")
    run = 0
    for x, r in zip(shots, recipes):
        run = run + 1 if r == "diagram" else 0
        if run > s["max_diagram_run"]:
            add("error", "S2", x["shot_id"], f"{run} diagram shots in a row", "break the run with a story or UI shot")
    same, prev = 1, None
    for x in shots:
        key = (x.get("recipe", "video"), x.get("location"),       # story scenes: also their own cuts
               tuple(v.get("intent") for v in (x.get("story") or {}).get("visuals", [])))
        same = same + 1 if key == prev else 1
        prev = key
        if same > s["max_same_visual_run"]:
            add("warn", "S3", x["shot_id"], f"{same} shots in a row look the same ({key[0]} at {key[1]})",
                "change the angle, the place or the shot type to keep attention")
    for x in shots:
        for l in x.get("lines", []):
            for sent in re.split(r"(?<=[.!?])\s+", l.get("text", "")):
                n = len(words(sent))
                if n > s["max_sentence_words"]:
                    add("error", "S4", x["shot_id"], f"sentence of {n} words (max {s['max_sentence_words']}): '{sent[:60]}...'",
                        "split it: one idea per sentence")
        if len(words(x.get("label", ""))) > s["max_label_words"]:
            add("warn", "S5", x["shot_id"], f"step label has {len(words(x['label']))} words (max {s['max_label_words']})",
                "shorten it (on-screen text competes with narration)")
        if len(words(x.get("title", ""))) > s["max_title_words"]:
            add("warn", "S5", x["shot_id"], "title is long", "keep titles under ~9 words")
        txt = spoken(x).lower()
        for term in JARGON:
            if has_phrase(txt, term) and term not in glossary:
                add("error", "S6", x["shot_id"], f"jargon '{term}' is not in the plan's glossary",
                    "say it in everyday words, or add it to 'glossary' with a simple analogy")
    seen_terms = set()
    for x in shots:                                        # a glossary term must be explained where it first appears
        txt = spoken(x).lower()
        for term, g in glossary.items():
            if has_phrase(txt, term) and term not in seen_terms:
                seen_terms.add(term)
                if not (x.get("explains") == term or (x.get("explains") or []) and term in x.get("explains")):
                    add("warn", "S6", x["shot_id"], f"'{term}' first appears here without being explained",
                        f"mark this shot \"explains\": \"{term}\" and give the analogy ({g})")
    if shots[-1].get("beat") not in ("payoff", "recap"):
        add("warn", "S7", ids[-1], "the last shot is not marked as the payoff/recap",
            "end with the answer to the opening question (beat: \"payoff\")")

    # ---- P: process and factual accuracy
    pos = -1
    covered = set()
    for x in shots:
        st = x.get("step")
        if st is None:
            continue
        if st not in steps:
            add("error", "P1", x["shot_id"], f"unknown process step '{st}'", f"use one of {step_order}")
            continue
        covered.add(st)
        i = step_order.index(st)
        if i < pos and not x.get("flashback"):
            add("error", "P1", x["shot_id"], f"step '{st}' comes after a later step: the process runs backwards",
                "reorder the shots, or mark the shot \"flashback\": true")
        pos = max(pos, i)
        if steps[st].get("conditional"):
            if not hedged(spoken(x)) and not x.get("lines") == []:
                add("error", "P2", x["shot_id"], f"step '{st}' only happens sometimes "
                    f"({steps[st].get('condition', 'conditional')}) but the narration states it as always true",
                    "word it conditionally: 'if…', 'for some items…', 'usually…', 'depending on…'")
    for st in step_order:
        if not steps[st].get("conditional") and not steps[st].get("optional") and st not in covered:
            add("error", "P3", None, f"required process step '{st}' ({steps[st].get('text', '')}) is never shown",
                "add a shot for it, or don't call it a required step")
    for c in claims.values():
        if not any(str(u).startswith("https://") for u in c.get("sources", [])):
            add("error", "P4", None, f"claim {c['id']} has no source URL", "add the page where this fact is stated")
    for x in shots:
        cited = x.get("claims", [])
        for cid in cited:
            if cid not in claims:
                add("error", "P4", x["shot_id"], f"cites unknown claim '{cid}'")
        cited_text = " ".join(claims[c]["text"] for c in cited if c in claims)
        for l in x.get("lines", []):
            factual = l.get("speaker") == "narrator" and not l.get("story") and x.get("step") is not None
            # two-friend explainers: the friends' factual lines ARE the explanation, so they need sources too
            friend_fact = bool(plan.get("factual_dialogue")) and l.get("kind") in ("fact", "correction")
            if friend_fact:
                for cid in l.get("claims", []):
                    if cid not in claims:
                        add("error", "P4", x["shot_id"], f"line cites unknown claim '{cid}'")
                if not l.get("claims"):
                    add("error", "P4", x["shot_id"], f"a factual line cites no claim: '{l.get('text', '')[:60]}'",
                        "add \"claims\": [ids] to the line, or make it a question / reaction / analogy")
                line_cited = " ".join(claims[c]["text"] for c in l.get("claims", []) if c in claims)
                nums = {int(n) for n in re.findall(r"\d+", l.get("text", ""))}
                if nums and not nums <= all_numbers(line_cited):
                    add("error", "P5", x["shot_id"], f"number(s) {sorted(nums)} are not in the line's cited claims",
                        "only use numbers that appear in a sourced claim")
                continue
            if factual and not cited:
                add("error", "P4", x["shot_id"], "a factual narrator line cites no claim",
                    "add \"claims\": [ids] for the facts it states (or mark the line \"story\": true)")
            nums = numbers_in(l.get("text", "")) if l.get("speaker") == "narrator" else set()   # dialogue is fiction
            if nums and not nums <= all_numbers(cited_text):
                add("error", "P5", x["shot_id"], f"timing {sorted(nums)} is not in the cited claims",
                    "only use numbers that appear in a sourced claim")
            if company and l.get("speaker") == "narrator":
                for ph in INTERNAL_DETAIL:
                    if has_phrase(l.get("text", ""), ph) and not cited:
                        add("error", "P6", x["shot_id"], f"'{ph}' describes {company}'s internal systems without a source",
                            "describe only what is publicly documented, or cite a claim")
    days = []
    for x in shots:
        if "day" in x:
            if days and x["day"] < days[-1][1] and not x.get("flashback"):
                add("error", "P7", x["shot_id"], f"time runs backwards (day {x['day']} after day {days[-1][1]})",
                    "fix the order or mark \"flashback\": true")
            days.append((x["shot_id"], x["day"]))

    # ---- A: narration ↔ visual alignment
    for x in shots:
        said, seen = spoken(x), visual_text(x, plan)
        if not said.strip():
            continue
        missing = entity_hits(said, entities) - entity_hits(seen, entities) - set(x.get("offscreen", []))
        if missing:
            add("error", "A1", x["shot_id"], f"the narration talks about {sorted(missing)} but the shot doesn't show it",
                "show it (scene/action/ui/journey), or list it in \"offscreen\" if it is only mentioned")
        if x.get("recipe", "video") not in DRAWN and not (content_words(said) & content_words(seen)) and not entities:
            add("warn", "A2", x["shot_id"], "no word of the narration matches the visual description",
                "make the visual show what is being said")
    for x in shots:
        if x.get("recipe", "video") in DRAWN | {"video"} and not (x.get("scene") or x.get("shows") or
                                                                 x.get("recipe") in DRAWN):
            add("error", "A3", x["shot_id"], "shot has no visual description")

    # ---- V: visual consistency
    chars = plan.get("characters", {})
    for a, b in zip(shots, shots[1:]):
        if a.get("location") and b.get("location") and a["location"] != b["location"] and not b.get("transition"):
            for who in set(a.get("cast", [])) & set(b.get("cast", [])):
                trav = plan.get("travel", {})
                if b["location"] not in trav.get(a["location"], []):
                    add("error", "V1", b["shot_id"], f"{chars.get(who, {}).get('name', who)} jumps from "
                        f"'{a['location']}' to '{b['location']}' with no transition",
                        "add \"transition\": \"cut\"/\"time_skip\" with a reason, or a travel shot")
    props = plan.get("props", {})
    state = {}
    for x in shots:
        for p in x.get("props", []):
            if p not in props:
                add("error", "V2", x["shot_id"], f"unknown prop '{p}'", "define it in \"props\" with one fixed look")
        for p, st in x.get("prop_state", {}).items():
            path = props.get(p, {}).get("path", [])
            if st not in path:
                add("error", "V3", x["shot_id"], f"prop '{p}' state '{st}' is not in its path {path}")
                continue
            if p in state and path.index(st) < path.index(state[p]) and not x.get("flashback"):
                add("error", "V3", x["shot_id"], f"prop '{p}' goes back from '{state[p]}' to '{st}' "
                    "(an object can't be in two places / move backwards)")
            state[p] = st
    for x in shots:
        for c in x.get("cast", []):
            if c not in chars:
                add("error", "V4", x["shot_id"], f"unknown character '{c}'")
    return issues


def errors(issues):
    return [i for i in issues if i["level"] == "error"]


def report(issues):
    """Readable text, one line per issue, errors first."""
    lines = []
    for i in sorted(issues, key=lambda i: (i["level"] != "error", i["shot"] or "")):
        lines.append(f"{'❌' if i['level'] == 'error' else '⚠️ '} [{i['rule']}] {i['shot'] or 'plan'}: {i['message']}"
                     + (f"  → {i['fix']}" if i["fix"] else ""))
    return "\n".join(lines)

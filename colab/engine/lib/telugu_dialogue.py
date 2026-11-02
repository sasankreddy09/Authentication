"""Natural spoken Telugu for Arjun and Meera: writer prompt, Telugu critic, cross-language fact check, pronunciation.

Writer: an extension of the Story Director's dialogue stage (story.py); same schema. A factual Telugu turn carries
`source_text` = the canonical ENGLISH wording of the claim; the English NLI checks only that English text (its
inputs are genuinely English). Whether the Telugu line SAYS the same thing is a separate, conservative check
(`localization_check`): numbers, names and the claim's key terms must be present, else UNKNOWN → manual review.
An English-only entailment model is never run on Telugu text.
"""
from __future__ import annotations

import re, unicodedata

TELUGU = re.compile(r"[ఀ-౿]")

SYSTEM = """You are an expert Telugu conversational screenplay writer.
Write an educational conversation between {cur} and {exp} that sounds like two real Telugu-speaking friends.
Use natural spoken Telugu (clear, contemporary, understood in Telangana and Andhra Pradesh) — not literary Telugu,
not a word-for-word translation of English, no exaggerated slang.
{cur} is curious, observant and friendly: he notices things, predicts, sometimes misunderstands, tries small
experiments, remembers earlier points. Not every line of his is a question.
{exp} is warm, knowledgeable and relaxed: she explains simply, uses everyday examples, asks him to predict, suggests
demonstrations, corrects gently. Never a newsreader, never long textbook paragraphs.
Begin with an interesting observation or situation, build curiosity through actions and discoveries, end with a
satisfying realization. No repetitive questions, generic praise, artificial jokes, formal introductions or lectures.
Use English words only where Telugu speakers naturally do (3D, screen, glasses, projector).
Every factual turn must be supported by the given facts: cite fact_ids and give "source_text" = the same claim in
plain English using only those facts. Keep names, numbers, quantities and scientific meaning exactly.
No emotion instructions inside the spoken text. Return valid JSON only."""


def system_prompt(bible, code_mix=None):
    """The writer's system prompt. code_mix (None = the original mostly-Telugu writer, unchanged) adds the
    Telugu–English style guide for that mode (CODE_MIX_MODES): short enough for the local Qwen3-4B planner."""
    a, b = bible.by_role("curious"), bible.by_role("expert")
    out = SYSTEM.format(cur=a.names.get("te", a.name), exp=b.names.get("te", b.name))
    if code_mix:
        out = out.replace("Use English words only where Telugu speakers naturally do (3D, screen, glasses, projector).\n",
                          "") + "\n" + code_mix_prompt(code_mix)
    return out


DIALOGUE_HINT = ('{"world": "...", "turns": [{"speaker": "arjun", "text": "<spoken Telugu>", "kind": "observation", '
                 '"fact_ids": [], "source_text": null, "tone": "friendly"}, {"speaker": "meera", "text": "<Telugu>", '
                 '"kind": "fact", "fact_ids": ["F1"], "source_text": "<the same claim in English>", "tone": "explainer"}]}')


# ------------------------------------------------------------------ Telugu critic
# bookish / translated vocabulary that sounds like a textbook when spoken (with a spoken alternative)
FORMAL = {"యొక్క": "the possessive ending (…లో / …కి)", "త్రిమితీయ": "3D", "చలనచిత్రం": "సినిమా",
          "చలనచిత్రాలు": "సినిమాలు", "అనుభూతిని": "అనిపించడం", "గ్రహిస్తుంది": "అర్థం చేసుకుంటుంది / తెలుసుకుంటుంది",
          "భిన్నమైన": "వేరువేరు", "కలిగిస్తాయి": "అనిపిస్తాయి", "ద్వారా": "తో / నుంచి (often)",
          "అనగా": "అంటే", "కావున": "అందుకే", "మరియు": "(join with -ఊ / ఇంకా)", "ఉపయోగించబడుతుంది": "వాడతారు",
          "చేయబడింది": "చేశారు", "వీక్షించు": "చూడు", "ప్రేక్షకులు": "జనం / చూసేవాళ్లు (in dialogue)",
          "చదునైన": "ఫ్లాట్", "దృశ్యం": "కనిపించేది / వ్యూ", "చిత్రాలను": "బొమ్మల్ని", "చిత్రాలు": "బొమ్మలు",
          "చిత్రాన్ని": "బొమ్మని", "చిత్రం": "బొమ్మ", "దిశల": "వైపుల", "దిశలో": "వైపు"}
# English idioms translated word for word (they don't exist in spoken Telugu)
CALQUES = {"తెలివైన జత": "'a clever pair of …'", "సైన్స్ కోసం": "'for science!'", "నీ పనిని చేయి": "'do your thing'",
           "మంచి ప్రశ్న": "'good question'", "రోజు చివరలో": "'at the end of the day'", "ఆనందించండి": "'enjoy' (formal)"}
REACTIONS = ("వావ్", "అద్భుతం", "అరే", "నిజంగానా", "ఓహ్", "సూపర్")
FORMULAS_TE = (r"ఈ రోజు మనం", r"నేర్చుకుందాం", r"చాలా మంచి ప్రశ్న", r"మీకు తెలుసా", r"ముగింపులో", r"స్వాగతం")


def _words(text):
    t = "".join(ch if unicodedata.category(ch)[0] in "LMN" or ch == "'" else " " for ch in text)
    return [w for w in t.split() if w]


def english_share(text):
    ws = _words(text)
    return sum(1 for w in ws if re.match(r"^[A-Za-z]", w)) / max(len(ws), 1)


def critique(st, bible, max_words=24, code_mix=None):
    """Telugu-specific weaknesses with turn ids (story.Weakness-compatible dicts). code_mix=None keeps the original
    rules exactly (T4 = English share); a code-mix mode replaces T4 with the structural Telugu–English rules
    (code_mix_critique: English-dominant line / awkward insertion / ...), never a fixed English ratio."""
    out = []
    turns = [(s, k, t) for s in st.scenes for k, t in enumerate(s.turns)]
    tid = lambda s, k: f"{s.id}_{k + 1}"
    react = {}
    for s, k, t in turns:
        if not TELUGU.search(t.text):
            out.append({"where": tid(s, k), "rule": "T0", "problem": "not Telugu", "evidence": t.text, "required": True,
                        "fix": "write the line in spoken Telugu"})
            continue
        hits = [w for w in FORMAL if w in t.text]
        if hits:
            out.append({"where": tid(s, k), "rule": "T1", "problem": "bookish / translated vocabulary",
                        "evidence": ", ".join(f"{h} → {FORMAL[h]}" for h in hits), "required": True,
                        "fix": "say it the way friends talk"})
        cal = [c for c in CALQUES if c in t.text]
        if cal:
            out.append({"where": tid(s, k), "rule": "T8", "problem": "English idiom translated word for word",
                        "evidence": ", ".join(f"{c} = {CALQUES[c]}" for c in cal), "required": True,
                        "fix": "say what a Telugu friend would actually say here"})
        for pat in FORMULAS_TE:
            if re.search(pat, t.text):
                out.append({"where": tid(s, k), "rule": "T2", "problem": "formulaic phrase", "evidence": t.text,
                            "required": True, "fix": "start from the situation, not an announcement"})
        n = len(_words(t.text))
        if n > max_words:
            out.append({"where": tid(s, k), "rule": "T3", "problem": f"long speech ({n} words)", "evidence": t.text[:60],
                        "required": t.speaker == bible.by_role("expert").id, "fix": "split it or let Arjun react"})
        sh = english_share(t.text)
        if code_mix is None and sh > 0.35:
            out.append({"where": tid(s, k), "rule": "T4", "problem": f"too much English ({sh:.0%} of words)",
                        "evidence": t.text, "required": False, "fix": "keep English only for words people really use"})
        for r in REACTIONS:
            if t.text.startswith(r) or f" {r}" in t.text:
                react.setdefault(r, []).append(tid(s, k))
        if re.search(r"<[^>]+>|\(.*?(happily|sadly|laugh|నవ్వ).*?\)", t.text):
            out.append({"where": tid(s, k), "rule": "T5", "problem": "stage direction inside spoken text",
                        "evidence": t.text, "required": True, "fix": "put delivery in the style field, not the text"})
    for r, where in react.items():
        if len(where) >= 2:
            out.append({"where": where[-1], "rule": "T6", "problem": f"reaction '{r}' repeated", "evidence": str(where),
                        "required": len(where) >= 3, "fix": "react with what he actually noticed"})
    starts = [_words(t.text)[0] for _, _, t in turns if t.speaker == bible.by_role("curious").id
              and t.kind == "question" and _words(t.text)]
    for w0 in set(starts):
        if starts.count(w0) >= 3:
            out.append({"where": "story", "rule": "T7", "problem": f"Arjun's questions keep starting with '{w0}'",
                        "evidence": str(starts), "required": False, "fix": "vary his rhythm"})
    if code_mix is not None:
        out += code_mix_critique(st, bible, code_mix)
    return out


# ------------------------------------------------------------------ cross-language fact preservation
TE_NUM = {"ఒక": 1, "ఒకటి": 1, "రెండు": 2, "మూడు": 3, "నాలుగు": 4, "ఐదు": 5, "ఆరు": 6, "ఏడు": 7, "ఎనిమిది": 8,
          "తొమ్మిది": 9, "పది": 10, "వంద": 100}
EN_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
          "hundred": 100, "single": 1, "both": 2}


def numbers(text, table):
    out = {int(x) for x in re.findall(r"\d+", text)}
    for w in _words(text.lower()):
        for k, v in table.items():
            if w == k or (TELUGU.search(k) and w.startswith(k)):
                out.add(v)
    return out


def localization_check(te_text, en_text, glossary):
    """Does the Telugu line keep the English claim's numbers and key terms? glossary: {english term: [Telugu
    renderings]}. PASS only when everything required is found; otherwise UNKNOWN (→ manual review), never FAIL-safe
    approval. Returns {"status", "missing", "numbers"}."""
    missing = []
    en_l = en_text.lower()
    for term, te_forms in glossary.items():
        if re.search(rf"(?<![a-z]){re.escape(term.lower())}(?![a-z])", en_l):
            if not any(f in te_text or (re.search(r"[A-Za-z]", f) and f.lower() in te_text.lower()) for f in te_forms):
                missing.append(term)
    n_en = numbers(en_text, EN_NUM) - {1, 2}       # "one/two/both/single" are too ambiguous across languages
    n_te = numbers(te_text, TE_NUM) - {1, 2}
    extra = sorted(n_te - n_en)
    lost = sorted(n_en - n_te)
    status = "PASS" if not missing and not extra and not lost else "UNKNOWN"
    return {"status": status, "missing_terms": missing, "numbers_lost": lost, "numbers_added": extra}


# ------------------------------------------------------------------ pronunciation per backend
# Display text (subtitles) and TTS input are different strings: subtitles keep the natural code-mixed line
# ("ఇప్పుడు 3D effect తగ్గిపోయింది"), the TTS may get a respelling. Indic Parler is trained on Telugu script, so
# English words are respelled the way Telugu speakers say them; Gemini gets the form that wins the listening
# benchmark (TELUGU_PRON_BENCH), until then the display text. Every entry is a CANDIDATE until a native speaker
# has listened (PRONUNCIATION_STATUS); only the words listed are changed — nothing is transliterated blindly.
_PARLER_RESPELL = {
    "3D": "త్రీడీ", "effect": "ఎఫెక్ట్", "screen": "స్క్రీన్", "screens": "స్క్రీన్స్", "movie": "మూవీ",
    "brain": "బ్రెయిన్", "projector": "ప్రొజెక్టర్", "polarization": "పోలరైజేషన్", "polarizing": "పోలరైజింగ్",
    "left eye": "లెఫ్ట్ ఐ", "right eye": "రైట్ ఐ", "left lens": "లెఫ్ట్ లెన్స్", "right lens": "రైట్ లెన్స్",
    "eyes": "ఐస్", "glasses": "గ్లాసెస్", "filter": "ఫిల్టర్", "filters": "ఫిల్టర్స్", "test": "టెస్ట్",
    "popcorn": "పాప్‌కార్న్", "view": "వ్యూ", "views": "వ్యూస్", "picture": "పిక్చర్", "pictures": "పిక్చర్స్",
    "lens": "లెన్స్", "exactly": "ఎగ్జాక్ట్‌లీ", "correct": "కరెక్ట్", "seriously": "సీరియస్‌లీ",
    "actually": "యాక్చువల్లీ", "wait": "వెయిట్", "slightly": "స్లైట్‌లీ", "different": "డిఫరెంట్",
    "drama": "డ్రామా", "scene": "సీన్", "flat": "ఫ్లాట్", "depth": "డెప్త్", "size": "సైజ్", "real": "రియల్",
    "act": "యాక్ట్", "miss": "మిస్", "combine": "కంబైన్", "separate": "సెపరేట్", "technology": "టెక్నాలజీ",
    "angle": "యాంగిల్", "angles": "యాంగిల్స్", "clue": "క్లూ", "clues": "క్లూస్", "show": "షో", "simple": "సింపుల్",
    "feeling": "ఫీలింగ్", "hero": "హీరో", "logic": "లాజిక్", "problem": "ప్రాబ్లమ్", "try": "ట్రై",
    "super": "సూపర్", "full": "ఫుల్", "pass": "పాస్", "check": "చెక్", "confirm": "కన్ఫర్మ్", "bro": "బ్రో",
    "OK": "ఓకే", "okay": "ఓకే", "shadow": "షాడో", "shadows": "షాడోస్", "TV": "టీవీ", "tiny": "టైనీ",
    "mini": "మినీ", "nose": "నోస్", "basically": "బేసిక్‌గా", "literally": "లిటరల్లీ", "light": "లైట్", "feel": "ఫీల్",
    "eye": "ఐ", "lenses": "లెన్సెస్",
}
PRONUNCIATION = {"indic_parler": {"3D": "త్రీడీ"}, "gemini_free": {},
                 # code-mixed stories (plan "code_mix"): the full respelling table for the local voice
                 "indic_parler:code_mix": dict(_PARLER_RESPELL),
                 # Gemini candidate B for the listening benchmark: the same Telugu-script respelling
                 "gemini_free:telugu_script": dict(_PARLER_RESPELL)}
PRONUNCIATION_STATUS = {
    "indic_parler": "adopted for 3D only (earlier Telugu videos)",
    "indic_parler:code_mix": "CANDIDATE — Telugu-script respelling; confirm by listening (TELUGU_PRON_BENCH)",
    "gemini_free": "CANDIDATE — display text as written (Latin English words); confirm by listening",
    "gemini_free:telugu_script": "CANDIDATE — Telugu-script respelling for Gemini; the benchmark decides",
}


def _sub(text, table):
    """Longest key first; Latin keys only where a Latin word starts/ends (Telugu endings may follow:
    'eyesకి' → 'ఐస్కి'); case-insensitive for Latin."""
    out = text
    for w in sorted(table, key=len, reverse=True):
        say = table[w]
        if re.search(r"[A-Za-z0-9]", w):
            out = re.sub(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", lambda m, s=say: s, out, flags=re.I)
        else:
            out = out.replace(w, say)
    return out


def tts_text(text, backend, extra=None):
    """The string sent to the TTS for this backend (subtitles always keep `text`)."""
    return _sub(text, {**PRONUNCIATION.get(backend, {}), **(extra or {})})


def latin_left(text, backend, extra=None):
    """English (Latin) words the backend would still receive in Latin letters → a review note for Parler,
    whose Telugu training doesn't cover Latin script (add the word to the table after listening)."""
    return re.findall(r"[A-Za-z][A-Za-z']*", tts_text(text, backend, extra))


# ================================================================== Telugu–English code-mixing (Tenglish)
# Style guide (docs/16_TENGLISH_STYLE_GUIDE.md has the long version and the research behind it):
#  * Telugu is the grammatical base (matrix language): Telugu verbs, case endings, word order; a sentence ends in
#    Telugu (verb / copula / particle) — "3D effect తగ్గిపోయింది", "correct కదా?".
#  * English supplies content words friends actually say in English (movie, screen, glasses, test, effect, brain,
#    filter, projector, different) and a few discourse reactions (actually, wait, seriously, exactly, correct).
#  * English verbs enter through Telugu light verbs: "try చేద్దాం", "combine చేస్తుంది", "miss అవుతుంది";
#    English nouns take Telugu endings: "glassesలో", "eyesకి", "viewsని".
#  * English words are written in English letters in the display text; Telugu-script respelling is a TTS concern.
#  * Not: whole English sentences, English function words inside a Telugu sentence ("the", "is", "because"),
#    bookish Telugu, word-for-word idioms, forced slang, an English word in every line, academic English
#    ("perceive", "stereoscopic") where a friend would say it simply. Some reactions stay fully Telugu.
CODE_MIX_MODES = {
    "light":   {"label": "mostly Telugu with occasional English", "en_content_max": 2,
                "note": "Mostly Telugu: at most one or two common English words in a line; many lines fully Telugu."},
    "natural": {"label": "balanced, natural Telugu–English", "en_content_max": 4,
                "note": "Balanced: English for everyday terms and a few reactions, Telugu for everything else."},
    "casual":  {"label": "urban casual Telugu–English", "en_content_max": 6,
                "note": "Urban casual: English terms and reactions are frequent, but the grammar stays Telugu."},
}
_GUIDE = """Telugu–English code-mixing ({label}). {note}
- Telugu grammar always: Telugu verbs, case endings and word order; every sentence ends in Telugu.
- English only for words friends really say in English (movie, screen, glasses, test, effect, brain, filter,
  projector, different, actually, wait, seriously, exactly, correct). Write them in English letters.
- English verbs with Telugu helpers: "try చేద్దాం", "combine చేస్తుంది". English nouns take Telugu endings: "glassesలో".
- Never: a full English sentence, English "the/is/of/because" inside Telugu, bookish Telugu, translated idioms,
  forced slang, English in every line. Some reactions stay fully Telugu ("అబ్బా!", "అవునా?").
Good: "అరే! ఇప్పుడు 3D effect తగ్గిపోయింది." | "ఒక చిన్న test చేద్దాం." | "Correct! ప్రతి lensకి వేరే filter ఉంటుంది."
Bad: "Actually the projector is showing two images." | "త్రిమితీయ చలనచిత్రం" | "Bro, literally full crazy vibe!\""""


def code_mix_prompt(mode):
    m = CODE_MIX_MODES[mode]
    return _GUIDE.format(label=m["label"], note=m["note"])


# English function words: inside a Telugu sentence they mean an English clause was pasted in (awkward insertion)
EN_FUNCTION = {"the", "a", "an", "is", "are", "was", "were", "am", "be", "been", "of", "to", "in", "on", "at", "for",
               "with", "from", "by", "and", "or", "it", "its", "this", "that", "these", "those", "i", "you", "he",
               "she", "we", "they", "my", "your", "his", "her", "our", "their", "me", "him", "us", "them", "do",
               "does", "did", "will", "would", "can", "could", "should", "have", "has", "had", "not", "very",
               "also", "because", "then", "than", "what", "why", "how", "when", "where", "which", "who", "just",
               "about", "into"}
# discourse words Telugu speakers really use in English (not counted as awkward)
EN_DISCOURSE = {"so", "but", "ok", "okay", "actually", "basically", "wait", "seriously", "exactly", "correct",
                "right", "sorry", "please", "bro", "super", "simple", "yes", "no", "literally"}
# academic English where a friend says it simply
EN_ACADEMIC = {"perceive": "feel అవుతుంది / తెలుస్తుంది", "perception": "feeling", "stereoscopic": "3D",
               "binocular": "రెండు కళ్లతో", "monocular": "ఒక్క కన్నుతో", "retinal": "కంటిలో",
               "superimposed": "ఒకదాని మీద ఒకటి", "designated": "దాని", "visual angle": "size",
               "polarized": "filter ఉన్న"}
# English words written in Telugu letters: in a code-mixed display line they read better in English letters
TRANSLITERATED_EN = {"గ్లాసెస్": "glasses", "స్క్రీన్": "screen", "టెస్ట్": "test", "ఎఫెక్ట్": "effect",
                     "ప్రొజెక్టర్": "projector", "ఫిల్టర్": "filter", "పాప్‌కార్న్": "popcorn", "లెన్స్": "lens",
                     "ఫ్లాట్": "flat", "ఫ్రేమ్": "frame", "కరెక్ట్": "correct", "గెస్": "guess"}
REACTIONS_EN = ("exactly", "seriously", "actually", "wait", "correct", "super", "wow", "ok", "okay")


def token_lang(w):
    """'te' | 'en' | 'mix' (English stem + Telugu ending: 'eyesకి') | 'num'."""
    te, la = bool(TELUGU.search(w)), bool(re.search(r"[A-Za-z]", w))
    if te and la:
        return "mix"
    if la:
        return "en"
    return "te" if te else "num"


def _sentences(text):
    return [x for x in re.split(r"(?<=[.!?…])\s+|\.\.\.", text) if _words(x)]


def line_profile(text):
    """Structure of one line: Telugu / English / mixed tokens, English function words, the longest run of
    English words, whether each sentence ends in Telugu (Telugu grammar) — no ratio threshold involved."""
    ws = _words(text)
    langs = [token_lang(w) for w in ws]
    run = best = 0
    for g in langs:
        run = run + 1 if g == "en" else 0
        best = max(best, run)
    sents = []
    for snt in _sentences(text):
        sw = _words(snt)
        sl = [token_lang(w) for w in sw]
        fn = [w for w, g in zip(sw, sl) if g == "en" and w.lower() in EN_FUNCTION]
        sents.append({"text": snt, "n": len(sw), "ends_te": sl[-1] in ("te", "mix"), "fn": fn,
                      "te": sum(g in ("te", "mix") for g in sl), "en": sum(g == "en" for g in sl)})
    en_words = [w for w, g in zip(ws, langs) if g in ("en", "mix")]
    return {"n": len(ws), "te": langs.count("te"), "en": langs.count("en"), "mix": langs.count("mix"),
            "en_run": best, "sentences": sents,
            "en_content": [w for w in en_words if re.sub(r"[ఀ-౿]", "", w).lower() not in EN_FUNCTION | EN_DISCOURSE],
            "en_function": [w for s_ in sents for w in s_["fn"]], "has_en": bool(en_words)}


def english_dominant(text):
    """A sentence that is no longer Telugu: English grammar (an English function word AND an English ending), a run
    of 6+ English words, or no Telugu at all in a sentence of 3+ words. Returns the offending sentences."""
    bad = []
    for s_ in line_profile(text)["sentences"]:
        if s_["n"] >= 3 and (s_["te"] == 0 or (s_["fn"] and not s_["ends_te"])):
            bad.append(s_["text"])
    p = line_profile(text)
    if p["en_run"] >= 6 and not bad:
        bad.append(text)
    return bad


def code_mix_critique(st, bible, mode):
    """The code-mixing rules (story.Weakness-compatible dicts):
    T4  English-dominant sentence (required): the line stopped being Telugu.
    T9  awkward insertion: English function words inside a Telugu sentence (required from 2, else a warning).
    T10 English crowding for this mode (warning): more English content words in one line than the mode suggests.
    T11 forced / missing mixing over the story (warnings): English in every line; or almost none in 'natural'/'casual'.
    T12 English written in Telugu letters in the display text (warning; not in 'light' mode).
    T13 academic English where a friend would say it simply (warning).
    T6  (extended) the same English reaction repeated."""
    m = CODE_MIX_MODES[mode]
    out, tid = [], (lambda s, k: f"{s.id}_{k + 1}")
    turns = [(s, k, t) for s in st.scenes for k, t in enumerate(s.turns)]
    react = {}
    for s, k, t in turns:
        if not TELUGU.search(t.text):
            continue                                   # T0 already reports it
        p = line_profile(t.text)
        dom = english_dominant(t.text)
        if dom:
            out.append({"where": tid(s, k), "rule": "T4", "problem": "English sentence, not Telugu with English words",
                        "evidence": " | ".join(dom)[:120], "required": True,
                        "fix": "keep Telugu grammar: Telugu verb at the end, English only for the key words"})
        fn = [w for s_ in p["sentences"] if s_["ends_te"] for w in s_["fn"]]
        if fn:
            out.append({"where": tid(s, k), "rule": "T9", "problem": "awkward English insertion (function words)",
                        "evidence": ", ".join(fn), "required": len(fn) >= 2,
                        "fix": "say the small words in Telugu (the/is/of/because → -ని, ఉంది, -లో, ఎందుకంటే)"})
        if len(p["en_content"]) > m["en_content_max"]:
            out.append({"where": tid(s, k), "rule": "T10", "problem": f"{len(p['en_content'])} English words in one line "
                        f"(mode '{mode}': up to ~{m['en_content_max']})", "evidence": ", ".join(p["en_content"]),
                        "required": False, "fix": "keep English for the words friends really say in English"})
        tr = [f"{w} → {e}" for w, e in TRANSLITERATED_EN.items() if w in t.text]
        if tr and mode != "light":                     # mostly-Telugu style may keep loanwords in Telugu letters
            out.append({"where": tid(s, k), "rule": "T12", "problem": "English word written in Telugu letters",
                        "evidence": ", ".join(tr), "required": False,
                        "fix": "write it in English in the display text (the TTS respelling is separate)"})
        ac = [f"{w} → {e}" for w, e in EN_ACADEMIC.items() if re.search(rf"(?<![A-Za-z]){w}", t.text, re.I)]
        if ac:
            out.append({"where": tid(s, k), "rule": "T13", "problem": "academic English", "evidence": ", ".join(ac),
                        "required": False, "fix": "say it the way a friend explains it"})
        first = _words(t.text)[:1]
        for r in REACTIONS_EN:
            if first and first[0].lower() == r:
                react.setdefault(r, []).append(tid(s, k))
    for r, where in react.items():
        if len(where) >= 2:
            out.append({"where": where[-1], "rule": "T6", "problem": f"reaction '{r}' repeated", "evidence": str(where),
                        "required": len(where) >= 3, "fix": "react with what was actually noticed"})
    lines = [t for _, _, t in turns if TELUGU.search(t.text)]
    with_en = [t for t in lines if line_profile(t.text)["has_en"]]
    if lines and len(with_en) == len(lines) and mode != "casual":
        out.append({"where": "story", "rule": "T11", "problem": "English forced into every line", "evidence":
                    f"{len(with_en)}/{len(lines)} lines", "required": False,
                    "fix": "let some reactions stay fully Telugu"})
    if lines and mode in ("natural", "casual") and len(with_en) < 0.25 * len(lines):
        out.append({"where": "story", "rule": "T11", "problem": "hardly any code-mixing for this mode", "evidence":
                    f"{len(with_en)}/{len(lines)} lines have an English word", "required": False,
                    "fix": "use the English words friends really say (screen, test, effect, glasses...)"})
    return out


# ------------------------------------------------------------------ the six-way evaluation (candidates A/B/C)
CRITERIA = ("naturalness", "clarity", "personality", "humour", "code_mixing", "science")
SCIENCE_POINTS = {        # brief §3: what the explanation must contain, as the fact ids that support each point
    "two different views for the two eyes": {"F1"},
    "the projector/glasses separate the views": {"F3", "F4", "F5"},
    "the brain combines them into depth": {"F2"},
    "one eye weakens stereo depth, other depth cues remain": {"F7"},
}


def evaluate(st, bible, mode, glossary=None):
    """Deterministic 0–5 proxies for the six criteria + the evidence behind each (a person still judges humour and
    naturalness: these numbers only catch what is measurable). Returns {"scores", "evidence", "issues"}."""
    cur, exp = bible.by_role("curious").id, bible.by_role("expert").id
    iss = critique(st, bible, code_mix=mode)
    req = [w for w in iss if w["required"]]
    turns = [t for s in st.scenes for t in s.turns]
    nat_bad = [w for w in iss if w["rule"] in ("T1", "T2", "T8", "T9", "T13", "T4", "T12")]
    words = [len(_words(t.text)) for t in turns]
    long_ = [w for w in iss if w["rule"] == "T3"]
    ak = [t.kind for t in turns if t.speaker == cur]
    mk = [t.kind for t in turns if t.speaker == exp]
    guesses = sum(k in ("misconception", "prediction", "hypothetical") for k in ak)
    tease = sum(k == "reaction" for k in mk)
    pop0 = "popcorn" in " ".join(t.text for t in st.scenes[0].turns).lower() or "పాప్‌కార్న్" in st.scenes[0].turns[0].text
    pop1 = "popcorn" in " ".join(t.text for t in st.scenes[-1].turns).lower() or \
        "పాప్‌కార్న్" in " ".join(t.text for t in st.scenes[-1].turns)
    reacts = [_words(t.text)[0].lower() for t in turns if t.kind == "reaction" and _words(t.text)]
    prof = [line_profile(t.text) for t in turns]
    mix_lines = sum(p["has_en"] for p in prof)
    te_only = sum(not p["has_en"] for p in prof)
    mix_bad = [w for w in iss if w["rule"] in ("T4", "T9", "T10", "T11", "T12")]
    cited = {f for t in turns for f in t.fact_ids}
    covered = [k for k, ids in SCIENCE_POINTS.items() if ids & cited]
    loc = {}
    for s in st.scenes:
        for k, t in enumerate(s.turns):
            if t.fact_ids:
                loc[f"{s.id}_{k + 1}"] = localization_check(t.text, t.source_text or "", glossary or {})["status"] \
                    if t.source_text else "UNKNOWN"
    clamp = lambda x: max(0, min(5, round(x, 1)))
    scores = {
        "naturalness": clamp(5 - sum(1.0 if w["required"] else 0.25 for w in nat_bad)),
        "clarity": clamp(5 - 1.0 * len(long_) - max(0, (sum(words) / max(len(words), 1)) - 12) * 0.3),
        "personality": clamp(2 + min(guesses, 2) + min(tease, 1) - (ak.count("question") / max(len(ak), 1) > 0.6)),
        "humour": clamp(1 + 2 * pop0 + 2 * pop1 - (len(reacts) - len(set(reacts)))),
        "code_mixing": clamp((5 if 0 < te_only < len(turns) else 3) - sum(1.0 if w["required"] else 0.25 for w in mix_bad)),
        "science": clamp(5 * len(covered) / len(SCIENCE_POINTS) - sum(v != "PASS" for v in loc.values())),
    }
    ev = {"naturalness": [f"{w['rule']} {w['where']}: {w['problem']}" for w in nat_bad],
          "clarity": f"{sum(words)} words, {sum(words) / max(len(words), 1):.1f} per turn, {len(long_)} long turns",
          "personality": f"Arjun: {guesses} guesses/predictions, kinds {ak}; Meera: {tease} teasing reactions",
          "humour": f"popcorn opening {pop0}, popcorn callback {pop1}, reaction openers {reacts}",
          "code_mixing": f"{mix_lines} lines with English, {te_only} fully Telugu; " +
                         "; ".join(f"{w['rule']} {w['where']}" for w in mix_bad),
          "science": f"points covered {len(covered)}/{len(SCIENCE_POINTS)}: {covered}; localization {loc}"}
    return {"scores": scores, "evidence": ev, "issues": iss, "required": req}


# LLM rubric (optional, local Qwen3-4B): it can only ADD weaknesses; it never approves a story or a claim.
RUBRIC_PROMPT = """Rate this Telugu–English dialogue between two friends (1–5 each) and list concrete problems.
Criteria: naturalness (do Telugu friends talk like this?), clarity, personality (Arjun curious and funny,
Meera relaxed and teasing), humour, code_mixing ({label}: Telugu grammar, natural English words only),
science (only what the cited facts say).
Return JSON: {{"scores": {{"naturalness": n, "clarity": n, "personality": n, "humour": n, "code_mixing": n,
"science": n}}, "problems": [{{"where": "sc01_2", "problem": "...", "fix": "..."}}]}}
Dialogue:
{dialogue}"""


def rubric_prompt(st, bible, mode):
    lines = [f"{s.id}_{k + 1} {bible.characters[t.speaker].name}: {t.text}" for s in st.scenes
             for k, t in enumerate(s.turns)]
    return RUBRIC_PROMPT.format(label=CODE_MIX_MODES[mode]["label"], dialogue="\n".join(lines))


def llm_rubric(st, bible, mode, llm, ask_json):
    """The optional LLM opinion as extra (non-required) weaknesses; scores are reported, never used to approve."""
    from pydantic import BaseModel, Field

    class Problem(BaseModel):
        where: str = "story"
        problem: str
        fix: str = ""

    class Rubric(BaseModel):
        scores: dict = Field(default_factory=dict)
        problems: list[Problem] = Field(default_factory=list)

    r = ask_json(llm, "You are a strict Telugu dialogue editor. Return valid JSON only.",
                 rubric_prompt(st, bible, mode), Rubric)
    return {"scores": r.scores, "issues": [{"where": p.where, "rule": "T-LLM", "problem": p.problem, "evidence": "",
                                            "required": False, "fix": p.fix} for p in r.problems]}


# ------------------------------------------------------------------ speech check for code-mixed lines
# Whisper writes an English word either in Latin ("effect") or in Telugu letters ("ఎఫెక్టు"), and confuses Telugu
# vowel lengths. Both sides are brought to one representation before comparing: known English words → their
# Telugu-script respelling, then letters only; a consonant skeleton (vowel signs dropped) shows when the
# "error" is only vowel noise. A bad score is a REVIEW flag for a person, never proof the voice is wrong.
_VOWEL_SIGNS = re.compile(r"[ా-్ౕౖఁ-ఃఅ-ఔ]")


def bilingual_key(text):
    import turn_align
    return turn_align.letters(_sub(text, _PARLER_RESPELL))


def _err(e, h):
    import difflib
    if not e:
        return 0.0
    m = sum(b.size for b in difflib.SequenceMatcher(None, e, h, autojunk=False).get_matching_blocks())
    return round(1 - m / len(e), 3)


def compare_speech(expected, heard, threshold=0.35):
    """→ {"err", "err_skeleton", "status": PASS | REVIEW | NO_ASR, "why"} for a code-mixed line."""
    if heard is None:
        return {"err": None, "err_skeleton": None, "status": "NO_ASR", "why": "no recogniser result"}
    e, h = bilingual_key(expected), bilingual_key(heard)
    err = _err(e, h)
    sk = _err(_VOWEL_SIGNS.sub("", e), _VOWEL_SIGNS.sub("", h))
    if err <= threshold:
        return {"err": err, "err_skeleton": sk, "status": "PASS", "why": ""}
    if sk <= threshold:
        return {"err": err, "err_skeleton": sk, "status": "PASS",
                "why": "consonants match; the difference is vowel-level recogniser noise"}
    return {"err": err, "err_skeleton": sk, "status": "REVIEW",
            "why": "the recogniser disagrees: a person listens (Whisper's Telugu/English mixing is weak)"}


def bilingual_error(expected, heard):
    """Drop-in error for code-mixed lines: the smaller of the normalized letter error and the skeleton error."""
    r = compare_speech(expected, heard)
    return None if r["err"] is None else min(r["err"], r["err_skeleton"])

# 16. Telugu–English (Tenglish) dialogue: style guide, critic and voice handling

Written 2026-10-20 for the 3D glasses Telugu video, version 3 (`three_d_friends_te_v3`).

Code:
- `colab/engine/lib/telugu_dialogue.py`
- `colab/stories/three_d_friends_te_v3/`
- `colab/tenglish_candidates.py`
- `colab/tests/test_tenglish.py`

## 1. What the research says (and what we took from it)

| Source | What it shows | What we use |
|---|---|---|
| [Code-mixed Nepali/Telugu–English datasets (arXiv 2504.21026)](https://arxiv.org/abs/2504.21026) | A social-media dataset built for abusive-language detection, not dialogue. Its examples are real everyday Telugu–English mixing: a Telugu frame with English words inserted. | Telugu is the base. English words go into Telugu sentences, not the other way round. |
| [Word-level language identification for English–Telugu (arXiv 2010.04482)](https://arxiv.org/abs/2010.04482) | Each word can be labelled Telugu, English or mixed (English stem + Telugu ending). | `token_lang()` labels each word `te` / `en` / `mix`, e.g. `eyesకి` is `mix`. |
| [Universal Dependencies Telugu–English treebank](https://universaldependencies.org/qte/) | Code-switched sentences tagged per word. Most switches are single nouns or short phrases. | English mostly enters as nouns and short reactions. |
| Myers-Scotton's Matrix Language Frame model (used in [CALCS 2021, Telugu political speech](https://aclanthology.org/2021.calcs-1.1/)) | One language gives the grammar frame: word order, endings, function words. The other inserts content words. | An English function word (the, is, of, because) inside a Telugu sentence is the sign of an awkward insertion. A sentence ending in English with English grammar is no longer Telugu. |
| IIIT Hyderabad work on Telugu–English dialogue data | There is very little natural dialogue data. | No corpus-trained model. Small rules plus a native speaker's ear. |

### The five registers the brief asks us to tell apart

| Register | Example | In this video? |
|---|---|---|
| Formal written Telugu | త్రిమితీయ చలనచిత్రం యొక్క అనుభూతి | No (critic rule T1) |
| Everyday spoken Telugu | అబ్బా! అది నా మీదకే వచ్చింది | Yes: some lines stay fully Telugu |
| Telugu grammar + English content words | ఇప్పుడు 3D effect తగ్గిపోయింది | Yes: the main style |
| Romanized Telugu | ippudu 3D effect taggipoyindi | No. Subtitles use Telugu script with English words in English letters. |
| Excessive English | Then both eyes open చేస్తే why so much more? | No (critic rule T4) |

## 2. The style guide (used by the writer prompt and the critic)

1. **Telugu grammar always.** Use Telugu verbs, case endings and word order, and end every sentence in Telugu (a verb, copula or particle).
   - Good: "3D effect తగ్గిపోయింది", "correct కదా?".
2. **English only for words friends really say in English.**
   - Nouns and terms: movie, screen, glasses, test, effect, brain, filter, projector, view, depth, popcorn.
   - A few reactions: actually, wait, seriously, exactly, correct.
   - These are examples, not a required list.
3. **English verbs take a Telugu helper verb:** "try చేద్దాం", "combine చేస్తుంది", "miss అవుతుంది".
4. **English nouns take Telugu endings:** "glassesలో", "eyesకి", "viewsని", "pictureగా".
5. **Display text writes English words in English letters.** How the voice pronounces them is a separate TTS concern (§5).
6. **Some reactions stay fully Telugu:** "అబ్బా!", "అవునా?", "అమ్మో, వద్దు!".
7. **Never use:**
   - whole English sentences;
   - English function words inside Telugu;
   - bookish Telugu;
   - idioms translated word for word;
   - forced slang ("Bro, literally full vibe");
   - English in every line;
   - academic English where a friend would say it simply ("perceive" becomes "feel అవుతుంది / తెలుస్తుంది").
8. **No strong regional dialect.** Use contemporary Telugu understood in both Andhra Pradesh and Telangana ("రా" between friends is fine).

### Code-mix modes (`code_mix` in the story or plan; default absent, so earlier stories are unchanged)

| Mode | Label | English content words per line (soft hint, warning only) |
|---|---|---|
| `light` | mostly Telugu with occasional English | ~2 |
| `natural` | balanced, natural Telugu–English (selected for v3) | ~4 |
| `casual` | urban casual Telugu–English | ~6 |

There is no fixed English ratio. The hard rules are structural (T4, T9); the per-mode count is only a warning (T10).

## 3. The critic (`telugu_dialogue.critique(st, bible, code_mix=...)`)

The existing rules are kept unchanged: T0, T1 (bookish), T2 (formulaic), T3 (long speech), T5, T6 (repeated reactions), T7 and T8 (calques).

- **With `code_mix=None`** (all earlier stories), T4 keeps its old meaning: English share above 35%, warning only.
- **With a code-mix mode**, T4 and the new rules are:

| Rule | Meaning | Required? |
|---|---|---|
| T4 | **English-dominant sentence**: it has English function words AND ends in English, or has 6+ English words in a row, or has no Telugu at all in 3+ words | yes |
| T9 | **Awkward insertion**: English function words inside a sentence that ends in Telugu | from 2 words (1 = warning) |
| T10 | More English content words in one line than the mode suggests | warning |
| T11 | English forced into every line (light/natural modes), or almost no English at all (natural/casual modes) | warning |
| T12 | An English word written in Telugu letters in the display text (not checked in `light` mode) | warning |
| T13 | Academic English ("perceive", "stereoscopic") | warning |
| T6+ | The same English reaction repeated ("Exactly" ×3) | at 3 |

### Six-way evaluation and LLM rubric

- `evaluate()` gives deterministic 0–5 proxies, with evidence, for naturalness, clarity, personality, humour, code-mixing and science.
- `llm_rubric()` is an optional opinion from Qwen3-4B. It can only add non-required issues; it never approves anything.
- Facts stay as before:
  - every factual turn has `fact_ids` plus an English `source_text`;
  - the NLI model checks the English claim only;
  - `localization_check` only checks that numbers and terms are preserved (PASS / UNKNOWN). Similar terminology never verifies a claim.

## 4. Candidates A / B / C (text only, no voice spent)

All three share the same scenes, beats, fact ids and English claims (`stories/three_d_friends_te_v3/candidates.json`). The full table is in `reports/tenglish_candidates.md`.

| | A light | **B natural (selected)** | C casual |
|---|---|---|---|
| Critic errors in own mode | 0 | 0 | 3 (English sentences) |
| Science points covered | 4/4 (one line drops "3D", so localization is UNKNOWN) | 4/4, all localization PASS | 4/4 |
| Main weakness | "తెర", "కళ్లద్దాలు", "పరీక్ష", "సూచనల" read as textbook to a young viewer | none measurable; a native speaker must listen | "Then both eyes open చేస్తే why so much more?" is no longer Telugu; "perceive"; "Bro" ×2 |

**Selection:**
- B is preferred, as the brief asks.
- A is close on the proxies. The difference between A and B is register, which the proxies can't judge, so a native speaker decides.
- C is rejected: three lines fail T4.

## 5. Display text vs TTS text

- **Subtitles always show `text`.** The voice gets `tts`, built at plan time by `build_story_plan.py`. The timeline reads only `l["text"]`.
- **Indic Parler-TTS (default, local, Apache-2.0)** gets a Telugu-script respelling of a fixed list of English words (`PRONUNCIATION["indic_parler:code_mix"]`):
  - 3D → త్రీడీ, effect → ఎఫెక్ట్, screen → స్క్రీన్, movie → మూవీ, brain → బ్రెయిన్, projector → ప్రొజెక్టర్, polarization → పోలరైజేషన్, left eye → లెఫ్ట్ ఐ, right eye → రైట్ ఐ, plus the other words this script uses.
  - Matching is word-bounded on the Latin side and longest-first, and keeps Telugu endings (`eyesకి` → `ఐస్కి`).
  - Only listed words are changed. A Latin word with no entry is reported at build time (`latin_left`); none is left in v3.
- **Gemini (opt-in only)** reads the `display` form by default; the alternative is `telugu_script`. The setting is `GEMINI_TRANSCRIPT_FORM`.
- **Every entry is a CANDIDATE** (`PRONUNCIATION_STATUS`) until `TELUGU_PRON_BENCH = True` has been run and a native speaker has filled its blind listening sheet.
  - The sheet asks the brief's five questions: Is this how friends talk? Are the English words natural? Are both voices understandable? Do the reactions sound genuine? Are any pronunciations strange?

## 6. Voice checks for code-switched speech

- **`compare_speech(expected, heard)`:**
  - Known English words are mapped to one script on both sides, so `effect` and `ఎఫెక్టు` match.
  - It also computes a consonant-skeleton error, so vowel-length noise from Whisper doesn't count.
  - Result: PASS / REVIEW / NO_ASR. A REVIEW means a person listens; it never proves the voice wrong.
- **Local Parler:** a line is remade only when the error is above 0.6 (`voice_policy.asr_retake_at`). Above 0.35 it is kept and flagged.
- **QC:** a flagged line fails the mandatory speech check, so the video is not approved until someone listens.
- **Gemini (opt-in):**
  - `voice_policy.gemini_asr_repairs = False`: the recogniser's doubt never triggers another request. The line is kept and flagged.
  - Everything else is unchanged: 6 calls per video, cache by exact transcript, confirmed Free tier, stop when the quota is exhausted.
- **Speaker identity:** unchanged — ECAPA identity plus the Gemini attribution check. One voice per character for the whole video.

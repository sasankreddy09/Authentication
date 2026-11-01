# 13. Natural voices and stronger stories for Arjun and Meera (2026-10-18)

**The brief:** `claude.md`, "Natural Human-Like Voices + Advanced Cinematic Storytelling".

**The aim:** make the two friends sound like a real conversation and the stories feel like a discovery rather than Q&A, while keeping the existing architecture.

This document reports what was measured, what changed, the results, and what is still unverified.

**Status words used below:**
- PASS / FAIL: actually executed.
- NOT_TESTED: not run.
- RESOURCE_UNAVAILABLE: not possible on this machine.
- ACCESS_REQUIRED: needs a token or access we don't have here.
- LICENSE_BLOCKED: refused by the licence registry.

## 1. Architecture audit (before any change)

| # | Component | Where | Finding |
|---|---|---|---|
| 1 | Story-writing stages | `story.make_story` | facts → beats → dialogue → visuals; no premise, no outline, no memory, no critic |
| 2 | Story prompts | `story.SYSTEM`, `beats_prompt`, `dialogue_prompt` | asked for "beats" of a fixed list; nothing on Arjun's active role or spoken style |
| 3 | Dialogue schema | `story.Turn` | 8 kinds; Arjun could only ask, observe, react or hold a misconception |
| 4 | Voice functions | cast `narration.py` (`line_settings`, `speak`) | one request per line; Telugu → Parler, narrator → Qwen (optional), friends → **Kokoro only** |
| 5 | Expressive controls | `voice_direction.rule_directions` | **narrator rules applied to the friends**: pauses come from the shot's narration beat |
| 6 | TTS integrations | Kokoro, Qwen3-TTS 1.7B (narrator only), Indic Parler | Kokoro follows only pace + silence; the measured sample's emotion, intensity and emphasis were all "unsupported" |
| 7 | Speaker identity | `voice_qc.consistency` | 0.80 threshold calibrated on long narrator chunks |
| 8 | Chunking / stitching | one TTS call per line, 8 ms edge fades, final loudnorm only | **already good**: no tiny fragments, minimal processing (kept) |
| 9 | Timing | cast `timeline.py` | gap = previous line's `pause_after` + `pause_before` |
| 10 | Caching | `stages.stage`, sidecars, segment/diagram keys | good (kept); story = one stage |
| 11 | Human review | contact sheet, `dialogue_shot_timeline.md`, narrator-only A/B | no A/B for dialogue, no takes |
| 12 | Limits | — | Qwen CustomVoice has **no native-English female preset** (only aiden/ryan); Qwen on CPU takes 4–12 min per line (measured earlier) |

**Measured on the existing 89 s sample** (`three_d_friends_v1`, Kokoro):
- **Gaps:** the gaps between turns were 0.45–1.18 s (mean 0.85 s). Conversation research reports a typical speaker-transition gap of about 0.2 s (Stivers et al., 2009, PNAS, 10 languages).
- **Cause of the gaps:** narrator beat pauses (450–900 ms after a line, 0–300 ms before it) applied to dialogue.
- **Cross-speaker smoothing:** the "no sudden emotion change" smoothing ran across BOTH speakers. Example: Meera's reassuring correction was forced into Arjun's "curious" (`sc05_2`).
- **Wrong instruction text:** every friend's instruction read "Maintain the same narrator voice".
- **Levels and rates are not the problem:** Arjun ≈ −23 dB and Meera ≈ −22 dB (balanced); speaking rates were 2.2–4.0 words/s.

**Story weaknesses of the existing script:** the new critic found these, and they match a careful read.
- 3 of Arjun's 4 questions are answered at once, and he never predicts anything.
- One explanation arrives unprompted (`sc06`).
- The ending is a recap list.

## 2. Story improvements (new: `engine/lib/story.py`, v2 section)

| Stage | What it does |
|---|---|
| A. `select_facts` | Keeps facts a viewer can follow (≤ 45 words; near-duplicates dropped). |
| B. Premise generation (`premises_prompt`) | 2–3 concepts in different formats (everyday mystery, experiment, misconception, what-if, journey, historical discovery, comparison, observation → reveal). Each concept has: opening, question, relevance, motivation, discovery, misunderstanding, key fact, demo, payoff, fact ids, risks. |
| C. `choose_premise` | **Rule-based score**; the model never grades its own idea. Rewards: concrete opening, a real question, a drawable demonstration, a misconception, ≥ 3 facts, a payoff that answers the question, and the curious friend acting. Penalises formulaic openings. A premise citing unknown facts is never chosen. |
| D. `Outline` | Beats with role, setting, objective, observation, what happens, what changes, what is learned, the open question and why the next beat follows. Must end in `closing`. |
| E. `stage_dialogue` | Beat by beat. Every prompt carries the **story memory** (`StoryState`): facts already said, Arjun's open questions and uncorrected beliefs, demos done, location, last lines. A planned demonstration is always shown. |
| F. `critique` | **Deterministic critic** (rules below). The **optional LLM critic** can only *add* weaknesses, and only at real scene/turn ids. |
| G. `revise` | Rewrites only scenes with required weaknesses (bounded by `MAX_STORY_REVISIONS`). A rewrite is kept only if it has fewer weaknesses AND passes the dialogue rules. |
| H. `repair_grounding` | For each unsupported claim: one bounded rewrite from its facts → re-check → otherwise remove it **plus what depended on it** (the question it answered, a reaction to it) → re-check the story. |

**New turn kinds:** `prediction`, `challenge`, `hypothetical` and `opinion`. Arjun can now predict, challenge and imagine, not just ask. A hypothetical must be worded as one ("Imagine…", "What if…").

**Critic rules:**

| Rule | Detects |
|---|---|
| C1 | formulaic phrases |
| C2 | Q&A loop (3 question→fact exchanges in a row) |
| C3 | Arjun only prompting, or ≥ 70 % questions, or never active |
| C4 | Meera lecturing (> 72 % of the words) |
| C5 | artificial transitions |
| C6 | textbook opening |
| C7 | ending that doesn't pay off the opening |
| C8 | no demonstration |
| C9 | repeated wording |
| C10 | a question left unanswered |
| C11 | repetitive question rhythm |
| C12 | every question answered at once with no prediction |
| C13 | unprompted explanations |
| C14 | a recap-list ending |

- **Rule D4:** a misconception may be corrected by a test first. A sourced correction must follow within 4 turns.
- **Rule D7:** no repeated facts.

**Topic notebook:**
- Every story stage is its own cached stage: `story_premise`, `story_outline`, `story_dialogue`, `story_revision`, `story_grounding`.
- The local model is loaded once, and only if a stage is missing.
- New settings: `MAX_STORY_PREMISES = 2`, `MAX_STORY_REVISIONS = 1`, `STORY_BENCH = False`.

## 3. Voice improvements (`voice_direction.py`, cast `narration.py`, `voice_qc.py`)

1. **Voice Performance Director** (`dialogue_directions`):
   - Each line gets a **conversational intention** from what it does and what came just before. There are 14 intentions: genuine curiosity, thinking aloud, playful disbelief, gentle correction, asking for confirmation, quiet realisation, enthusiastic discovery, satisfying revelation, guiding a demo, playful remark, and others.
   - Example: the same words "Wait, how is that possible?" get different intentions after a fact than at the start of a scene (tested).
   - Intensity stays ≤ 0.35 except on hero moments (≤ 0.5).
   - Arjun gets a slightly livelier baseline.
   - Instructions read "Talking to a friend, …", never "narrator voice".
   - Kokoro follows pace and silence only, reported honestly as before.
2. **Conversational timing:**
   - The pause BEFORE a line is a response gap: 120–380 ms for answers and reactions, 400–650 ms to think, 450–650 ms before a revelation, 650–950 ms to try a demonstration (also across a cut).
   - The pause after a line is 0.
   - Gaps vary deterministically by line id: never identical, and stable on re-runs.
3. **Smoothing per speaker:** continuity belongs to one voice. Narrator-only plans are unaffected.
4. **Bounded takes:**
   - `MAX_VOICE_TAKES_NORMAL = 1` and `MAX_VOICE_TAKES_HERO = 2`.
   - Hero lines are the opening, the central question, the reveal, the ending, and discovery/revelation intents.
   - A second take is made only on backends whose takes differ (Qwen3-TTS, Parler); Kokoro is deterministic.
   - The choice is by checks only (`voice_qc.pick_take`): acceptable words, then fewer wrong words, then fewer failed checks. Never "more pitch variation".
   - Both takes are kept (`narration/takes/`) with inline A/B players.
   - `selected_take_id` goes in the sidecar; cached takes are reused.
5. **Chunking:** one call per line was kept. The audit found no fragment problem; lines are 1–3 sentences.
6. **QC:**
   - Conversation diagnostics: gap mean, spread and range; mechanical rhythm; dead air > 1.5 s; overlaps; per-speaker rate outliers; level jumps > 4 dB. Shown as information.
   - Homophones (eye/I, two/to, …) count as "heard as", not as missing words.
   - **Identity threshold by clip length**, calibrated on 37 two-friend lines:
     - Same voice: < 3 s min 0.68; 3–3.5 s min 0.82; ≥ 3.5 s min 0.87.
     - The other friend: max 0.18.
     - Rule: < 2 s not judged; 2–3.5 s needs 0.70; longer clips need the backend threshold.
7. **Per-character Qwen routing:** with `VOICE_BACKEND = "qwen3-tts"`, a friend with an approved `qwen_speaker` in the Bible uses Qwen; others stay on Kokoro.

## 4. Changed / added files

| File | Change |
|---|---|
| `engine/lib/story.py` | v2 stages (premises, outline, memory, dialogue, critic, revision, fact-aware repair, metrics); new turn kinds + `intent`; D4/D7; pronunciation → `tts` |
| `engine/lib/voice_direction.py` | `two_friends` profile, intentions, `derive_intent`, `response_gap_ms`, `dialogue_directions`, `is_hero`, per-speaker `smooth`, friend-style `instruction` |
| `engine/lib/voice_qc.py` | `identity_threshold` (calibrated), homophones, `conversation_diagnostics`, `pick_take` |
| `engine/lib/characters.py` | Voice profile fields (qwen preset, emotion control, rate/intensity ranges, accent, baseline, reference sample, pronunciation); `wink` expression; `speak_text` |
| `engine/lib/charanim.py` | wink (one eye closed); see-through tinted 3D lenses (eyes stay readable) |
| `engine/lib/factcheck.py` | hypotheticals ("what if", "suppose") count as comparisons |
| `engine/lib/registry.py` | Chatterbox Turbo / Nano recorded as evaluated, not integrated (`review`), with revisions |
| `cells/cast/narration.py` | dialogue directions for story plans; per-character Qwen; hero takes + A/B; dialogue `VOICE_BENCH` (`run_dialogue_bench`); duration-aware identity |
| `cells/cast/settings.py` | `MAX_VOICE_TAKES_NORMAL/HERO` |
| `cells/topic/topic.py`, `settings_head.py` | staged v2 story pipeline; `MAX_STORY_PREMISES`, `MAX_STORY_REVISIONS`, `STORY_BENCH` |
| `cells/shared/qc.py` | conversation-timing diagnostics in the final report |
| `stories/three_d_friends_v2/` | premises, outline, story (the scripted stand-in for the LLM stages) |
| `three_d_friends_v2_plan.json`, `colab_three_d_friends_v2.ipynb`, `scripts/three_d_friends_v2_script.md` | the new sample |
| `samples/v2/`, `samples/voice_bench/` | rendered sample + A/B audio |
| `tests/test_story_v2.py` (new, 42 checks); `test_story.py`, `test_voice.py` (2 checks updated for intended changes) | tests |

## 5. Schema updates (all optional fields, so old plans are unchanged)

- **Turn:** adds `intent` (14 values) and `source_text`. Kinds now include prediction / challenge / hypothetical / opinion.
- **Plan line:** adds `intent`, plus `tts` from the pronunciation profile.
- **VoiceDirection:** adds `intent`. `pause_before_ms` may now be up to 1200 ms.
- **New schemas:** Premise / Premises / OutlineBeat / Outline / Weakness / Critique.
- **Bible `Voice`:** new fields with defaults.
- **Sidecar:** adds `takes` + `selected_take_id`.

## 6. Model evaluation

| Candidate | Licence (Hub, verified) | Revision | Fit | Status here |
|---|---|---|---|---|
| Kokoro-82M | Apache-2.0 | `f3ff357` | baseline; pace + silence only; CPU-fast | **PASS** (benchmarked, production default) |
| Qwen3-TTS 1.7B CustomVoice | Apache-2.0 | `0c0e305` | instruction-following; English natives **aiden, ryan (male only)** | **RESOURCE_UNAVAILABLE** here (no GPU; ~4–12 min/line on CPU); run `VOICE_BENCH` on a T4 |
| Qwen3-TTS 0.6B CustomVoice | Apache-2.0 | `85e237c` | no instruction control (model card) | not useful for expression |
| Chatterbox Turbo / Nano | MIT | `749d1c1` / `71ccd1d` | distinct voices need reference audio; package pins torch 2.6 + transformers 5.2; watermark | **NOT_TESTED**; registry `review` (strict mode refuses) |
| Indic Parler-TTS (Telugu) | Apache-2.0, gated | `660bc53` | Prakash / Lalitha + descriptions | **ACCESS_REQUIRED** here (token is in Colab) |

**Decision:**
- Kokoro stays the production default for both friends.
- Arjun may use Qwen `aiden` once the T4 benchmark and a listening test support it.
- Meera has no native-English female Qwen preset. `serena` is benchmarked as a *candidate* only.
- Mixing backends between the two friends must be judged by ear before use.

## 7. Voice benchmark (actual results, CPU)

`VOICE_BENCH = True` on the v2 notebook runs the 8-line two-friend scene. Results (`samples/voice_bench/dialogue_bench.json`):

| Config | Status | Time | Speech | Peak RAM | Identity (LOO, per friend) | Notes |
|---|---|---|---|---|---|---|
| kokoro, old (narrator) direction | FAIL* | 56.4 s | 35.4 s | 1.43 GB | 0.83–0.92 | scene 39.2 s |
| kokoro, new (conversation) direction | FAIL* | 41.1 s | 35.0 s | 1.47 GB | 0.81–0.92 | scene 37.4 s |
| qwen3-tts (aiden / serena) | RESOURCE_UNAVAILABLE | — | — | — | — | no GPU |

\* Both FAILs were a single homophone ("eye" heard as "I") by the test machine's tiny ASR. That is now handled as "heard as". The run was not repeated afterwards.

**A/B files:** `samples/voice_bench/dialogue_kokoro_old_direction.wav` vs `…_new_direction.wav`.

**Human rating:** use `listening_sheet.md`, then open `blind_test_key.json`. **Not done yet.**

**Full sample timing:**
- **v1 (old story + old direction):** gaps between turns 0.45–1.18 s, mean 0.85 s. Within a scene: 0.45–1.05 s.
- **v2 (new story + new direction):** within a scene 0.12–0.60 s, overall mean 0.37 s, spread 0.14 s, not mechanical.

## 8. Story benchmark (old vs new script, 3D topic)

| Measure | Old (v1) | New (v2) |
|---|---|---|
| Format | linear Q&A explainer | **experiment** (chosen by the scorer) |
| Arjun's turns that are questions | 44 % | 30 % |
| Arjun predicts / tests | no | yes ("My new guess: each lens blocks something"; the one-eye test) |
| Meera's share of words | 59 % | 55 % |
| Critic weaknesses | 3 (C12, C13, C14) | 0 |
| Claims fact-checked by NLI | 10/10 PASS | 9/9 PASS |
| Ending | recap list | callback to the opening (popcorn, head tilt, "No promises.") |
| Length | 89.0 s | 78.0 s |

**What the numbers do and don't show:**
- These numbers come from a critic I wrote. **They do not prove the story is better.** Human ratings are needed: `STORY_BENCH = True` in the topic notebook writes old and new scripts for 3 topics side by side with a rating table.
- STORY_BENCH needs the 4B model, so it is **NOT_TESTED** here. The local runs used a scripted stand-in for the model.

## 9. Character profiles (Bible)

- **Arjun:**
  - en: Kokoro `am_michael` (Qwen preset `aiden` when enabled). Baseline: friendly, curious, a little energetic, questions rise naturally, never cartoonish. Rate 0.95–1.08, intensity 0.1–0.5.
  - te: Parler `Prakash`, "young, lively and curious".
- **Meera:**
  - en: Kokoro `af_heart` (no Qwen preset). Baseline: warm, patient, confident, brightens on discoveries. Rate 0.93–1.05, intensity 0.1–0.45.
  - te: Parler `Lalitha`, "warm, clear and friendly".
- **Both:** the pronunciation profile maps `3D` to `త్రీడీ` in Telugu.
- **Cross-language:** the same personality across languages; the voice timbre is **not** claimed to be the same.

## 10. Integration

- **Final audio drives timing:** voices → placed lines → each scene's visual cut points (unchanged).
- A selected take that changes a line's length only re-times that line's scene: the visual cache key includes the cut points and mouth movement.
- Each story stage, direction, take and line has its own cache.

## 11. Test report (executed on this machine, CPU)

**Suites:**
- Existing suites: voice 24, voice_backend 6, registry 8, qc 14, subtitles 7, planner 16, assets 8, telugu 25, speed 25, logic 27, consistency 62, lib 31 = **253 PASS**.
- `test_story` **53 PASS** and `test_story_v2` **42 PASS**.
- Online licence check: PASS.
- **Total: 348 offline + 1 online, all passing.**

**Updated tests (intended changes):**
- `test_voice`: character lines no longer say "narrator voice".
- `test_story`: D4 allows a test before the correction.

**Bugs found by the new tests and fixed:**
- A 32-word fact limit dropped real Wikipedia facts.
- The demonstration gap was skipped across a cut.

**End-to-end runs:**
- **v2 sample:** first run NOT APPROVED. Two lines were flagged:
  - The identity check on a 3.05 s clip (0.797); fixed by the calibrated threshold.
  - A tiny-ASR mishearing.
  - Re-run with small.en (the Colab default): **APPROVED**, 78.0 s.

## 12. Colab compatibility (measured only where run)

- **CPU draft render of v2:** peak RAM 1.4–1.8 GB; ~3–4 min; 0 AI clips.
- **GPU (T4) runs:** none were executed. **No VRAM or GPU timing is reported.**

## 13. Sample output

- `colab/samples/v2/three_d_friends_v2_draft_720p.mp4` (78 s, 720p draft), plus contact sheet, dialogue/shot timeline, statistics, QC report and the voice directions.
- The old v1 sample is still in `colab/samples/` for comparison.

## 14. How to run

**Sample:**
1. Upload `colab/colab_three_d_friends_v2.ipynb` to Colab.
2. Run all (CPU is fine).
3. For the voice benchmark: set `VOICE_BENCH = True` and choose a T4 GPU, so that Qwen3-TTS is included.
4. Listen to the A/B players and fill in `render/voice_bench/listening_sheet.md`.

**Topic:** in `colab_topic_to_video.ipynb`, set `TOPIC`. To compare the old and new story writers on 3 topics, set `STORY_BENCH = True`.

**Your own story:** `build_story_plan.py <story.json> <plan.json> --project <name> --ground <hf cache>` (as before).

## 15. Remaining work (not done or not verifiable here)

1. **Human listening:**
   - Is the new direction and timing more natural? This needs the blind A/B, and nothing here claims the voices sound human.
   - **Telugu needs a native speaker.**
2. **On a T4:**
   - The Qwen3-TTS dialogue benchmark: Arjun = aiden; Meera = serena as a candidate (non-native accent possible).
   - The VRAM/RAM/time numbers.
   - Hero takes on a real sampling backend. The take logic is unit-tested, but the full stand-in-worker run was not executed.
3. **STORY_BENCH with the real Qwen3-4B on 3 topics + human ratings.** The 4B model's premises and critiques have not been read yet.
4. **Telugu:**
   - Short voice benchmark: written (8 lines, same intentions), but ACCESS_REQUIRED here.
   - The full Telugu story video: not rendered, as the brief says to wait until speech is validated.
5. Chatterbox was not benchmarked (separate environment; reference audio needed).

## 16. Addendum: first Colab run (2026-10-18)

**What happened:** the user's first Colab run of `colab_three_d_friends_v2.ipynb` reached the voices on the GPU, but the speech check failed to load (`Unsupported model binary version … v774843950`).

**Cause:** Hugging Face cache symlinks don't survive Google Drive.

**Fix:** plain-file model folders with validation and automatic re-download (guide §42).

**Verified on CPU:**
- The failure was reproduced and the repair works.
- The notebook loads the speech check from the new folder.
- APPROVED, 78 s.
- Affected suites pass: registry 9, voice 24, voice_backend 6, telugu 25, speed 25, story 53, story_v2 42.

## 17. Correction (2026-10-19)

- **STORY_BENCH:** §2 and §8 described `STORY_BENCH = True` as implemented. It was not: the setting existed but no code used it. It is now implemented in the topic notebook and was run end to end with a scripted model (see `docs/15_ARCHITECTURE_VALIDATION.md`). Real writing-quality scores still need the 4B model in Colab and human ratings.
- **New rule D8:** at least 2 sourced factual turns per story.

# 15. Architecture validation and improvements (2026-10-19)

**How it was validated:** with evidence, not opinion.
1. Static analysis of every built notebook as ONE program (pyflakes over all cells joined, so names shared between cells resolve as they do in Colab).
2. A check that every setting is actually enforced somewhere.
3. The WARN/ERROR lines of the latest full runs, grouped.
4. End-to-end runs of each path:
   - English two-friend;
   - Telugu with stand-in Gemini;
   - Telugu without a key;
   - the topic notebook with the staged writer;
   - `STORY_BENCH`.
5. The full test suite.

## 1. Findings

| # | Finding | Evidence | Severity | Status |
|---|---|---|---|---|
| 1 | **Gemini lines were labelled "parler"** by `backend_of()`, so the identity check ordered a "calmer retake" through the API. That is an unplanned, single-voice request outside the conversation (a 5th call). The next one hit the budget, and the exception **disabled the identity check for all lines** | run log: `GEMINI_SPEAK → GeminiBlocked FREE_QUOTA_EXHAUSTED`, then "speaker consistency check unavailable" | high (spends free quota, hides QC) | **fixed** |
| 2 | **A story with 0 sourced facts was accepted** and rendered as an "explainer" | topic run with the staged writer: "Plan: … 0 sourced facts" → video made | high (product correctness) | **fixed**: rule D8 (≥ 2 sourced factual turns) |
| 3 | `STORY_BENCH` setting existed and was documented (`docs/13` §2, §8), but **was never implemented** | setting used in 0 places | medium (doc overstated) | **fixed**: implemented and run |
| 4 | `MAX_VIDEO_DURATION_SECONDS` declared but **never enforced** | used in 0 places | medium | **fixed**: warning after timing + required QC check (two-friend / short-form plans only, so older long videos are not affected) |
| 5 | The topic notebook's staged-writer path had **never been run end to end** after it was wired in | — | medium | **now run**: APPROVED with the scripted model |
| 6 | Topic error message pointed to the old stage file (`stages/story.json`) | read in the failing run | low | **fixed** |
| 7 | Static analysis: `_MONITOR_STOP`, `TOPIC_PLAN`, `FACTCHECK`, `ASSET_MANIFEST` "undefined" | pyflakes | none | **false positives**: each is guarded at runtime |
| 8 | `VISUAL_STYLE` is not used (only one style exists) | used in 0 places | none | left (informational setting) |
| 9 | NVENC "Unknown encoder" errors in local logs | log | none | expected locally (no NVIDIA GPU); Colab uses NVENC |

## 2. Fixes in code

- `cells/cast/narration.py`:
  - `backend_of()` knows `gemini:` voices.
  - `RETAKE_BACKENDS = ("qwen3-tts", "parler")`: identity retakes, toned-down retakes and hero takes only for local sampling backends. Gemini lines come only from the planned two-speaker pass, and an outlier is flagged for listening.
  - Per-line `try` in the identity check.
  - The model-name table includes Gemini.
- `lib/expressive_tts.py`: Gemini capabilities (style = `speech_metadata.style`; pauses = the model's own gaps, kept as measured).
- `lib/voice_qc.py`: Gemini identity threshold 0.72 (provisional, uncalibrated).
- `lib/story.py`: rule **D8** (`MIN_FACT_TURNS = 2`).
- `cells/cast/timeline.py` + `cells/shared/qc.py`: duration limit. `MAX_DURATION` = the plan's `max_duration_s`, or `MAX_VIDEO_DURATION_SECONDS` for two-friend plans; FAIL if the final file is longer; never sped up or cut.
- `cells/topic/topic.py`:
  - `STORY_BENCH`: 3 topics (3D movies, thunder, train ticket), old vs staged writer on the same facts and model, grounding of both, blind A/B `render/story_bench/story_bench.md` with a rating table; then stops.
  - Corrected stage-file message.
- `tests/run_topic_story.py`: the scripted model answers every staged-writer prompt (premises, outline, dialogue with memory, critic, revision, repair).

## 3. Verification

| Check | Result |
|---|---|
| Full suite (15 suites) | **402 PASS** (voice 24, voice_backend 6, registry 9 incl. online licences, qc 14, subtitles 7, planner 16, assets 8, telugu 26, speed 25, logic 27, consistency 62, lib 31, story 54, story_v2 42, gemini_telugu 51 incl. the real-speech split) |
| Telugu v2, stand-in Gemini, after fix #1 | exactly **4** API requests (the planned chunks); identity check ran for every line; 0 "unavailable" |
| Telugu v2, no key | ACCESS_REQUIRED → all 19 lines local Parler (stand-in) → video rendered |
| English 3D v2 | APPROVED, 78 s, no Gemini code path touched |
| Topic notebook, staged writer | APPROVED; 2 sourced facts; critic reports the missing demonstration |
| Topic notebook, `STORY_BENCH = True` | 3 topics researched; both writers ran; benchmark file written; stopped on purpose. The scores come from a scripted model, so they show the machinery works, **not** writing quality |

## 4. Notebooks rebuilt

- 3D glasses notebooks: `colab_three_d_friends_v2.ipynb`, `colab_three_d_friends_te_v2.ipynb`.
- The topic notebook (`colab_topic_to_video.ipynb`), because fixes #2, #3, #5 and #6 live in it.
- **No other video's notebook or plan was rebuilt or edited.** If one is rebuilt later, it gets the fixes. All of them are inert for non-Gemini, non-two-friend plans, apart from rule D8, which applies only to story plans.

## 5. Recommended next improvements (not done here)

1. **The voice cell is large** (`cells/cast/narration.py`, ~1,150 lines: voices, two benchmarks, the Gemini pass, QC). Moving the benchmark and Gemini orchestration into library modules would make them unit-testable like the rest. It's a refactor with no behaviour change, so it was not risked without a GPU run to compare against.
2. **Per-session downloads in the topic notebook:** the 4B planner (~8 GB), NLI and SigLIP are re-downloaded every Colab session (local cache), roughly 3–5 min each time.
   - Option: plain-file Drive cache (`hf_files`), at the cost of ~9 GB of the 15 GB free Drive.
   - This is the user's trade-off; the default is unchanged.
3. **Calibrate on real audio:** the Gemini identity threshold (0.72 provisional) and the Telugu letter-error threshold (35 %), from the first real Gemini and Parler runs.
4. **GPU-measured numbers:** none yet for the Telugu Parler path, the Gemini path or the 4B planner. The first Colab run's log (`@@`/RES lines, preflight) provides them.

## 6. Addendum: GPU vs CPU audit (2026-10-19)

**Question:** does the Telugu 3D notebook use GPU memory, not only CPU RAM?

**Already on the GPU:**
- Indic Parler voices (own process, `cuda`, float32 on T4).
- Whisper speech check (float16; Colab confirmed "running on: GPU").
- Kokoro (onnxruntime-gpu).
- Shot and final encoding (NVENC).

**Found on the CPU even with a GPU, now fixed:**
- **ECAPA speaker-identity check** (hard-coded `device="cpu"`) → now `auto`: GPU when available.
- **Encoding every drawn scene** (libx264; for the two-friend videos that is every frame) → now NVENC when the session has it, with automatic x264 retry if NVENC fails.
- **Topic notebook NLI fact check** (hard-coded `device="cpu"`) → now the session's device.

**CPU by nature:**
- Drawing Arjun and Meera (Pillow has no GPU path).
- Gemini voices (run on Google's servers, using neither).

**Not changed:** Parler loads its weights into CPU RAM before moving them to the GPU (a few seconds; ~3.5 GB). Avoiding that needs `accelerate` in its pinned environment, which can't be tested here without a GPU.

**New evidence in every run:**
- Each model records where it actually ran (`DEVICE_REPORT`).
- The monitor now also samples GPU memory of all processes every 10 s (`RUN_PEAKS`).
- The final QC prints "Where each step ran" with peak system RAM and peak GPU memory; also saved in `qc_report.json` (`devices`, `peaks`).

**Verified here (CPU only):** whole-notebook static analysis shows no new undefined names; Gemini (61), story and registry suites pass. **The GPU paths themselves are verified only by the first Colab run's report.**

## 7. Addendum: Tenglish v3 validation (2026-10-20)

- `colab/tests/test_tenglish.py`: 93 checks, CPU, no network, all PASS. They cover the claude.md §11 list; see `colab/tests/README.md`.
- No regression:
  - The Telugu v2 and English v2 plans rebuild identical to the committed ones. The only difference is the additive Character Bible fields: the `duck` gesture, plus Gemini voice fields that predate this change in the English v2 plan.
  - Earlier stories get the original critic rules.
- Bug fixed: `build_story_plan.py --ground` wrote every `stories/<folder>/story.json` report to the same `stories/story_grounding.json`. Folder stories now write `stories/<folder>/grounding.json`. The shared file was regenerated from the Telugu v2 story with the same NLI model.
- Test fix: `test_telugu.py` and `test_voice_backend.py` predate `DEVICE_REPORT`; they now give it to the voice functions they run.

## 8. Addendum: storybook visual upgrade validation (2026-10-20)

- **`colab/tests/test_storybook.py`:** asset immutability and tamper detection, approval and pinning, rig completeness and limits, smooth transitions, lip sync (silence, speech, only the speaker), environment packs and presets, every v3 visual rendering, validators, legacy compatibility, licences, cache key.
- **`colab/storybook_previews.py`:** visual tests A–D in `reports/storybook/`.
- **Full CPU draft:** the storybook v3 notebook end to end, with espeak-ng stand-in speech.
- **Not verified here:**
  - the Character Studio (needs a GPU);
  - Godot (not adopted);
  - real Parler / Gemini voices for lip sync;
  - human approval of the artwork.

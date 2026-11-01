# 10. Audit of `claude.md` against the real pipeline, and decisions (2026-10-18)

`claude.md` asks for a free, local, Colab-T4 educational video generator. This file records three things:
1. what the repository **already does**, so it is not rebuilt;
2. which requested changes are **legitimate improvements**, and which are not;
3. **what was decided**, and why.

Every licence and model claim below was checked against the Hugging Face API or the publisher's page on 2026-10-18.

## 1. Audit: requested capability → current state

| # | `claude.md` requirement | State before this work | Where |
|---|---|---|---|
| 1 | No paid APIs | **Met.** No OpenAI, Claude, ElevenLabs, Runway, Modal, RunPod or stock-API code. YouTube upload uses Google's free API with the user's OAuth and uploads private. | whole `colab/engine` (grep) |
| 2 | Models fit a T4 | **Met.** One model family on the GPU at a time (text encoder → VACE → checkers); direct-to-GPU loads; RAM guard. | `clips.py`, `logic.py`, `helpers.py` |
| 3 | No redundant model loading | **Mostly.** Models are reloaded once per repair round, by design (VACE + Qwen3-VL don't fit together). | `clips.py` rounds |
| 4 | Batching | Narration runs one sentence at a time (required for per-line retries); drawn scenes render in a background thread. | `narration.py`, `diagrams.py` |
| 5 | No needless re-encoding | Segments are encoded once (near-lossless) and cached; the final encode is needed for subtitles and the audio mix. **Two encodes in total, by design** (it enables per-shot caching). | `editlib.py`, `render.py` |
| 6 | Irrelevant or illogical scenes | Drawn recipes are deterministic. AI shots get scene-state layout control, OWLv2 + Qwen3-VL checks and a CLIP relevance check; unverified shots → fallback + REVIEW. | `logic.py`, `clips.py`, `acceptance.py` |
| 7 | Narration ↔ visual link | Validator rules A1–A3 (entities said must be shown); the timeline is driven by the narration. | `validate.py`, `timeline.py` |
| 8 | Temporal / spatial checks | Scene-state engine (rules X0–X5, E1–E11); motion checks; camera estimation (improved in guide §36). | `scene_state.py`, `motion_check.py` |
| 9 | Crash recovery / caching | Narration, prompts, clips, segments, model caches and drawn scenes are all keyed by content and resume after a restart. **There is no stage manifest for the new topic stages.** | sidecars everywhere |
| 10 | Hook, pacing, subtitles, music, mix | Hook rule S1, mood-based pacing, word-timed SRT subtitles, licensed Kevin MacLeod music (CC BY) with credits, `sidechaincompress` ducking, `loudnorm`. **No word-level highlighting.** | `validate.py`, `timeline.py`, `music.py`, `render.py` |
| — | **Topic → script without Claude** | **Missing.** Plans are written per video by an assistant. This is the main gap against "fully free". | — |
| — | **Licensed real imagery** | **Missing.** Visuals are drawn or AI-generated only. | — |
| — | **Final QC gates** | Partial: ffprobe, ebur128 loudness, blackdetect, speech check. **No** freeze, silence, clipping, ducking, stream/size/fps/drift checks, contact sheet, or single approval gate. | `qc.py` |
| — | **Runtime modes** | Missing (one fixed 1080p24 mode). | settings |
| — | **No-paid-API / licence tests** | Missing as automated tests. | — |

## 2. Decisions on the requested technologies

| Request | Verdict | Reason (verified) |
|---|---|---|
| Local LLM planner, `Qwen/Qwen3-1.7B` | **Adopt, adapted.** Default **`Qwen/Qwen3-4B-Instruct-2507`** (Apache-2.0, text-only, non-thinking, ~8 GB fp16: fits the T4 on its own). `Qwen3-1.7B` (Apache-2.0) is the low-memory option. | A 1.7B model writes noticeably weaker explanations; the 4B fits because nothing else is loaded during planning. `Qwen3.5-4B` (Apache-2.0) is multimodal (`image-text-to-text`), so it loads a vision tower we don't need. |
| LLM writes JSON, fixed code renders it | **Adopt** (already the project rule) | The LLM writes *beats* (narration + visual intent + fact ids); deterministic Python builds the full plan, and the existing validator gates it. |
| Research from permitted public sources | **Adopt:** Wikipedia (Action API) | The text is CC BY-SA. We extract facts (facts aren't copyrightable), paraphrase, and cite the article URL for every claim. Requests follow Wikimedia's policy: a User-Agent with your contact, `maxlag=5`, serial requests ([access policy](https://www.mediawiki.org/wiki/Wikimedia_APIs/Access_policy)). |
| Fact validation | **Adopt:** two deterministic gates + one model gate | (1) every number and named term in a narration sentence must appear in its cited source sentences; (2) an NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`, Apache-2.0, CPU) must find the sentence *entailed* by its evidence. Unsupported sentences are rewritten once, then blocked. No model can *guarantee* zero hallucination; failures block approval instead. |
| Kokoro via the `kokoro` package | **Reject** | `kokoro-onnx` already runs the same Apache-2.0 Kokoro-82M weights, tested on GPU and CPU. Switching adds dependencies for no gain. |
| `whisperx` alignment | **Reject (keep faster-whisper)** | whisperx pulls pyannote and its own torch pins; faster-whisper word timestamps already drive the subtitles. |
| Qwen3-TTS 0.6B | **Defer** | Apache-2.0, but 10 languages (no Telugu); Kokoro already covers English. |
| Wikimedia Commons imagery | **Adopt** | Licence from `extmetadata` (`LicenseShortName`, `Artist`, `AttributionRequired`, `LicenseUrl`). **Default allow-list: public domain (incl. PD-USGov/NASA), CC0, CC BY.** CC BY-SA is opt-in (`ALLOW_SHAREALIKE`), because an image the video builds on can make the whole video a ShareAlike adaptation ([CC ShareAlike](https://wiki.creativecommons.org/ShareAlike_interpretation)). Unknown or NC/ND licences are always rejected. Provenance goes into `assets_manifest.json`, plus on-screen and description credits. |
| Pexels / Pixabay | **Defer** (optional per `claude.md`) | They need API keys and have their own terms (not open licences). |
| Ranking with OpenCLIP / SigLIP | **Adopt:** **`google/siglip2-base-patch16-224`** (Apache-2.0) through the already-pinned `transformers` | No new `open_clip` dependency. The score is used **relatively** (the candidate must beat distractor captions and a calibrated floor), never as proof of correctness. |
| FLUX.2 klein 4B | **Defer** | Apache-2.0 (4B only; the 9B is non-commercial), but ~13 GB fp16 by the publisher's figure, plus a Qwen3-4B text encoder: no T4 headroom. Diffusion stays optional (existing SDXL path). |
| ACE-Step 1.5 music | **Defer** (optional per `claude.md`) | Reported MIT, but VRAM depends on the variant (4–24 GB), unverified on a T4. The licensed library already works. |
| ASS kinetic subtitles (`pysubs2`) | **Adopt, without `pysubs2`** | ASS is written directly (small, tested, no dependency). Word-by-word highlighting for Latin scripts through libass. Telugu keeps the Pillow-drawn subtitles (libass detached vowel signs in testing, guide §37). |
| 1280×720 @ 30 fps default | **Adapt** | `RUN_MODE`: `draft` = 1280×720, `standard` = 1920×1080 (tested, sharper for drawn text), `cpu_fallback` = 1280×720 with no AI video. **24 fps is kept**: the AI clips are 16 fps and 24 fps keeps drawn-scene work 20 % lower. |
| 17 renamed modules (`config.py`, …) | **Reject renaming; map instead** | `claude.md` itself says "adapt to the existing structure". The notebook builder assembles `engine/cells` + `engine/lib`; new capabilities become new `engine/lib` modules. The mapping is in §4. |
| ResourceManager / checkpoints | **Exists; extend** | Existing: `gpu_mem`, `release_model`, `need_ram`, `GpuSampler`, content-keyed sidecars. Added: a stage manifest for the topic stages (research, plan, fact check, assets) so they resume. |
| Final QC + single approval gate | **Adopt** | Added freeze, silence, true-peak/clipping, ducking effectiveness, stream/resolution/fps/duration-drift checks, a contact sheet, and `approved = all mandatory gates PASS`. |
| T4 benchmarks | **Cannot be measured here** (no GPU) | `BENCHMARK = True` (guide §36) and `perf.json` record them on Colab. **No numbers are invented.** |

## 3. Order of work

The order follows `claude.md`'s priority: correctness → Colab reliability → scene consistency → render speed → engagement.

1. **QC and approval gate** (correctness): `qc.py`, with tests.
2. **Licence registry + no-paid-API test** (correctness): `engine/lib/registry.py`, tests.
3. **Topic pipeline** (the main gap):
   - `research.py` (Wikipedia facts with URLs);
   - `factcheck.py` (number/term check + NLI);
   - `planner.py` (Qwen beats → validated plan);
   - `assets.py` (Commons search, licence filter, SigLIP ranking, manifest);
   - the `stock` recipe;
   - the topic notebook.
4. **ASS word-highlight subtitles** (engagement).
5. **Runtime modes** (reliability / speed).

## 4. Mapping of `claude.md` module names to this repository

| `claude.md` name | Here |
|---|---|
| `config.py` | `engine/cells/*/settings.py` (+ `RUN_MODE`) |
| `research.py` | `engine/lib/research.py` (new) |
| `script_generator.py` | `engine/lib/planner.py` (new) + `engine/cells/topic/*` |
| `video_plan.py` | `engine/lib/planner.py` (Pydantic: FactSheet, NarrativeBeat, Script) + `scene_state.py` |
| `tts_engine.py`, `audio_alignment.py` | `engine/cells/*/narration.py` (Kokoro / Parler + faster-whisper word timing) |
| `asset_provider.py`, `asset_ranker.py` | `engine/lib/assets.py` (new) |
| `scene_validator.py` | `scene_state.py`, `motion_check.py`, `acceptance.py`, `validate.py`, `factcheck.py` |
| `visual_recipes.py` | `drawkit.py`, `diagrams.py`, `editlib.py` (+ `stock` recipe) |
| `subtitle_engine.py` | `engine/lib/subtitles.py` (new) + `timeline.py` |
| `music_engine.py`, `audio_mixer.py` | `musiclib.py`, `music.py`, `sfx.py`, `render.py` |
| `render_engine.py` | `editlib.py` + `render.py` (FFmpeg, NVENC smoke test → libx264) |
| `resource_manager.py`, `checkpoint_manager.py` | `helpers.py` (+ `engine/lib/stages.py`, new) |
| `quality_control.py` | `qc.py` + `engine/lib/qclib.py` (new) |

## 5. What was implemented (2026-10-18), with test results

**New `engine/lib` modules:**

| Module | Purpose |
|---|---|
| `qclib.py` | Final QC: ffprobe; black and freeze detection in one pass; loudness, true peak and silence in one pass; **measured** ducking; contact sheet; PASS/FAIL/UNKNOWN; approval gate. |
| `registry.py` | Verified weight licences for every model; paid-service deny-list; image licence policy. |
| `research.py` | Wikipedia facts with URL and revision. It reads the **HTML extract**, because the plain-text extract turned "2×10<sup>30</sup>" into "2×1030", a wrong number a fact check would have confirmed. |
| `factcheck.py` | Number and name gates plus NLI entailment. |
| `planner.py` | Qwen draft → Pydantic → fact check → one rewrite → drop; repeat, symbol and length gates; deterministic plan builder; card text must match the narration. |
| `assets.py` | Commons search, licence filter, cached download by content hash, SigLIP 2 ranking against description **and** narration. |
| `subtitles.py` | Word-highlighted ASS. |
| `stages.py` | Stage checkpoints keyed by content hash. |

**Notebooks and cells:**
- New notebook `colab/colab_topic_to_video.ipynb` (`python3 build.py --topic …`): type a topic, get a video.
- New cell `engine/cells/topic/topic.py`.
- New `stock` recipe (licensed image, fitted over a blurred copy of itself, with a credit) in `editlib`, `plan`, `timeline`, `render` and `validate`.
- `RUN_MODE` (`standard` / `draft` / `cpu_fallback`).
- `SUB_STYLE` (`karaoke` / `plain`).
- `DUCK` ducking settings.
- Image credits in the YouTube description.

**Decisions measured on real data:**

| Decision | Measurement |
|---|---|
| NLI model | On 14 labelled pairs (paraphrases, negations, wrong numbers, unrelated claims): `nli-deberta-v3-small` 12/14, **`nli-deberta-v3-base` 13/14** (1.35 s on CPU), `DeBERTa-v3-large` 13/14 (16 s). **No model ever passed a false or unsupported claim.** Base was chosen. |
| Analogies | NLI scores a correct analogy as "contradiction" (0.84): it's about something else. Analogies are therefore checked by the number/name gates plus required comparison wording, never by NLI. |
| Repeats | A reworded repeat entails one way only (1.00 / 0.01). Rule: either direction ≥ 0.9. Unrelated sentences score 0.00 both ways. |
| SigLIP 2 thresholds | On real Commons images, right images had a margin over distractors of +3.9 to +6.5; other beats' images −2.7 or lower; unrelated images (cat, mountain) far lower. Thresholds: margin ≥ 1.5, logit ≥ −6. **Calibrated on a small sample; re-calibrate when topics change.** |
| Narration gate | Found by looking at a real render: a galaxy image passed the *description* check under narration about an accretion disk. Images must now also match what is **said**; an adversarial test covers it. |

**Tests** (all passing; 221 checks across 10 suites):

| Suite | Checks |
|---|---|
| `test_registry.py` | 6 (+1 online licence re-check against the Hub) |
| `test_qc.py` | 14 (synthetic media with planted defects) |
| `test_subtitles.py` | 7 (libass render, pixel checks) |
| `test_planner.py` | 16 (scripted LLM + real NLI) |
| `test_assets.py` | 8 (live Commons + SigLIP 2, adversarial) |
| Existing suites (`telugu`, `speed`, `logic_loop`, `consistency`, `lib`) | 25 / 25 / 27 / 62 / 31 |

**End-to-end run of the topic notebook** (Python 3.13, CPU, FFmpeg 4.4; real Wikipedia, Commons, NLI, SigLIP 2, Kokoro, render and QC; a scripted stand-in for the 4B planner, which needs the T4):
- 22 cells, 0 errors.
- **APPROVED**, with every mandatory check PASS. Measured: ducking 10.1 dB, loudness −14.1 LUFS, true peak −1.4 dBTP.
- **Resume:** a second run reused the research and script stages, so the topic cell took 0 s.
- The CPU stage times are not T4 numbers. **No T4 benchmark has been run yet**: use `BENCHMARK = True` and `render/perf.json` on Colab.

**Known limitations:**
- **Planner quality:** the real 4B planner hasn't been run here. Qwen3-0.6B on CPU produced a script the gates correctly refused as too thin. Expect that some topics stop at "too thin to publish" rather than ship weak content.
- **Card wording:** card lines made from the narration (used when the model's card text isn't supported) can still be cut inside a long clause.
- **Facts:** facts come from Wikipedia only, and the NLI judges entailment against those sentences, not real-world truth. Sources are listed in the description for checking.
- **Commons metadata:** `LicenseShortName` is user-entered. Unknown values are rejected, never guessed.
- **Not done (deferred, see §2):** Pexels/Pixabay, ACE-Step, FLUX.2 klein, Qwen3-TTS and whisperx.

## 6. Second `claude.md` (Two-Friend Cinematic), validated and implemented (2026-10-18)

**Decisions after checking the brief's claims:**
- **Turbo LoRA:** no Hub licence tag (it is a repackaging of Apache-2.0 Self-Forcing), so strict mode refuses it.
- **SDXL** (OpenRAIL++) and **OpenAI CLIP** (no tag): refused in strict mode.
- **Rhubarb:** MIT, but not bundled. The mouth follows the speaker's loudness instead, and it is not claimed as lip sync.
- **Commons video:** mostly BY-SA for the sample topic, so images only.
- **Telugu:** always Parler. Unknown languages stop with an error. Automatic Telugu stories are refused, because the fact check is English-only.

**Results (CPU):** 306 offline checks + 1 online check, all passing; the 3D two-friend sample is APPROVED (89 s); the backward-compatibility run is unchanged.
Report: `docs/12_TWO_FRIEND_STORYTELLING.md`.

# 05 — Implementation Roadmap

**Status:** Phase 1 (research and design) is complete with this document set. Nothing below has been implemented yet.
**Date:** 2026-10-11

- **Guiding rule:** build the boring, deterministic spine first (schemas, project folder, FFmpeg renderer, checkpoints), then plug models in behind adapters.
- Each milestone produces a video you can watch, even if the visuals are placeholders at first.
- Effort figures are **[est]** for one experienced Python developer working part-time (about 15–20 h/week).

---

## Overview

| Phase | Name | Goal | Effort [est] | Needs GPU? | Needs credentials? |
|---|---|---|---|---|---|
| 1 | Research and architecture | This document set | done | No | No |
| 2 | Prototype design | Freeze schemas, interfaces, settings, and the 30–60 s prototype spec | 2–3 days | No | No |
| 3a | **M1 — Skeleton renderer** | `main.py create` makes a 30–60 s video from a hand-written plan with placeholder images | 1 week | No | No |
| 3b | **M2 — Planning stages** | LLM produces brief → research → script → storyboard → validated plan | 1 week | No | LLM API key |
| 3c | **M3 — Local audio** | Kokoro TTS, alignment, timing lock, music/SFX library, mix, subtitles | 1 week | No | No |
| 3d | **M4 — Remote GPU worker** | Job protocol + Colab notebook: keyframes (T2I) and I2V clips | 1–2 weeks | Colab | HF token (only if a gated model is chosen) |
| 3e | **M5 — QC and resume** | Asset QC, final QC, hash-based resume, cleanup | 1 week | No | No |
| 3f | **M6 — YouTube publishing** | OAuth, private resumable upload, thumbnail, captions, publish record | 3–5 days | No | Google OAuth client |
| 3g | **M7 — 3-minute documentary** | Full pyramids example (doc 06) end to end | 1 week | Colab | All above |
| 4 | Hardening and production | Rented GPU worker, scheduling, Shorts format, regression tests | ongoing | Rented GPU | — |

Total to a working 3-minute documentary: **about 6–9 weeks** part-time [est].

---

## Phase 2 — Prototype design (minimal working prototype)

### 2.1 Prototype scope: a 30–60 second video

| Item | Prototype choice | Why |
|---|---|---|
| Length | 45 s target (30–60 s accepted), ~9–12 shots | Small enough to iterate in minutes |
| Format | 1920×1080, 24 fps, H.264/AAC | Main target |
| Planning | LLM API (structured JSON), one call per stage | Best plan quality for the laptop |
| Visuals | `graphic` + `still_motion` from **keyframes made on Colab**; **0–3 I2V shots** optional | Works even when Colab gives no GPU: missing keyframes fall back to graphics |
| Narration | **Kokoro-82M on the laptop CPU** | Apache-2.0, fast on CPU, no GPU needed |
| Alignment | faster-whisper `base`/`small` int8 on CPU | Word timestamps for subtitles and timing |
| Music | 3–5 pre-downloaded CC0 / royalty-free tracks in `assets/music/` (generation comes later) | Removes a GPU dependency and a licensing unknown |
| SFX | Small CC0 library in `assets/sfx/` | Same |
| Render | FFmpeg on the laptop CPU | Hardware-independent |
| QC | ffprobe, black/freeze/silence detect, loudness, WER | Deterministic |
| Publish | Not in the prototype's first milestone (M6) | Keeps credentials out until the rest works |

**Prototype acceptance criteria:**
1. `python main.py create --topic "How black holes are formed" --duration 45 --format youtube` produces `final_16x9.mp4` with no manual steps except running the Colab notebook cell.
2. The video is 45 s ± 3 s, 1920×1080, 24 fps, H.264 High + AAC 48 kHz, and plays in VLC and a browser.
3. Narration is intelligible, with WER < 10% against the script, and subtitles are within ±200 ms of speech.
4. No black, frozen or silent gaps beyond the planned ones. Integrated loudness is −14 ± 1 LUFS.
5. Killing the process mid-run and re-running `create` finishes without regenerating completed assets (checked by log counts).
6. If the GPU worker is unavailable, the run still completes, with graphics/still fallbacks and a warning in `qc_report.json`.

### 2.2 Design freeze deliverables (before writing code)

- [ ] `schemas/` Pydantic models: Brief, Research, Script, Storyboard, ProductionPlan, Timeline, Job, QCReport, Metadata, PublishRecord (from `02_ARCHITECTURE.md §5`).
- [ ] `config/settings.yaml` and `config/models.yaml` keys (doc 04 §7).
- [ ] Adapter interfaces: `ImageGen`, `VideoGen`, `TTS`, `MusicGen`, `SFXSource`, `LLMProvider`.
- [ ] Style preset `cinematic_documentary.yaml` (prompt suffix, negative, grade, transitions, pacing).
- [ ] One hand-written `production_plan.json` (45 s) as a fixture for tests.

---

## Phase 3 — Implementation milestones

### M1 — Skeleton renderer (no AI)

**Build:**
- Repo layout from `AI_VIDEO_AUTOMATION_MASTER_PLAN.md §8`, `pyproject.toml` with pinned dependencies (uv), `ruff`, `pytest`.
- `orchestration/pipeline.py`, a stage runner with input hashing and sidecars.
- `rendering/`: timeline builder, normalization, joins, loudnorm, encode profiles (`youtube`, `shorts`, `draft`).
- `rendering/recipes/`: the registry plus the **starter menu of about 9 recipes**: `still_move`, `i2v_clip` (placeholder until M4, falling back to `still_move`), `title_card`, `map_route`, `comparison_panels`, `timeline_graphic`, `lower_third`, `transition`, `music_bed`/`sfx_at`. Each has a params model, a version, a fallback and a golden test. **Do not build more until the backlog asks.**
- `config/recipe_fallbacks.yaml` + validator mapping of unknown recipes → closest recipe, logged as a `wish`.
- `hardware.py` + `main.py doctor`: reports CPU, RAM, disk, GPU/CUDA, ffmpeg version, encoders.

**Acceptance:**
- The fixture plan + placeholder PNGs + a test WAV → a valid MP4 matching the format checks.
- A fixture shot with an unknown recipe (`split_screen`) renders with its fallback, and the wish appears in `recipe_requests.json`. The run does not stop.
- `pytest` covers timeline maths (offsets with transitions) and ffprobe validation.
- `main.py doctor` runs on the laptop.

### M2 — Planning stages

**Build:**
- `LLMProvider` (hosted API backend first; Ollama backend stubbed).
- Stages: Director, Researcher (web search + fetch, sources saved), Scriptwriter, Storyboard (per section), Validator, Critic.
- Prompt files are versioned in `prompts/`. All raw LLM I/O is logged to `logs/llm/`.

**Acceptance:**
- 5 different topics each produce a schema-valid `production_plan.json` with no manual edits.
- Every narrated factual sentence links to ≥ 1 claim with a source URL.
- Re-running with no changes makes no LLM calls (cache hit).
- The JSON Schema given to the planner is generated from the recipe registry. Plans that want missing effects produce `wish` entries, not failures.
- `main.py backlog` lists wishes grouped across projects.

### M3 — Local audio

**Build:**
- Kokoro TTS per sentence, with a pronunciation dictionary.
- faster-whisper WER check with regeneration.
- Word alignment → `words.json`, then the duration lock.
- Music/SFX library loader with license records (`licenses.json`).
- Ducking + two-pass loudnorm. SRT/ASS writer.

**Acceptance:**
- A 45 s plan → narration whose total length matches the timed plan within 3%.
- Subtitle QC rules pass.
- Loudness −14 ± 1 LUFS.
- The first end-to-end video (graphics + placeholders + real voice) is produced.

### M4 — Remote GPU worker (Colab)

**Build:**
- Job file protocol (`02_ARCHITECTURE.md §6.3`) over a shared Google Drive folder (rclone on the laptop, `drive.mount` in Colab).
- `worker/` package with adapters: `t2i.z_image_turbo` / `t2i.flux2_klein_4b` / `t2i.flux_schnell`, `i2v.wan22_ti2v_5b` (and `i2v.wan22_a14b_gguf` for experiments).
- `notebooks/colab_worker.ipynb`: installs pinned requirements, detects GPU, chooses adapters that fit its VRAM, processes the queue, and writes heartbeats.
- `comfy_workflow` adapter type: headless ComfyUI on 127.0.0.1 inside the worker; slot-filled, **pinned** community workflows (`worker/workflows/*.json` + `.lock.yaml` with ComfyUI commit, custom-node commits, model sha256s). Licence of each workflow and node recorded.
- Benchmark the diffusers adapter vs the best pinned community Wan 2.2 I2V workflow (LightX2V) on the same 10 shots. Keep the better one per adapter.
- Local `remote` stage: push jobs, poll, pull results, verify hashes, fall back on timeout.

**Acceptance:**
- 12 keyframe jobs and 3 I2V jobs complete on a Colab GPU and come back to the laptop with sidecars.
- Disconnecting Colab mid-queue and reconnecting resumes without redoing finished jobs.
- Runtime per keyframe and per clip is measured on each GPU type you get, and recorded in `docs/benchmarks.md`.
- The worker refuses to run a workflow whose installed versions differ from its lock file. Each workflow's golden render passes.

### M5 — QC, resume and cleanup

**Build:**
- Asset QC: image checks (resolution, blank/duplicate via imagehash, optional CLIP prompt similarity, optional VLM yes/no), clip checks (frame difference, freeze, flicker), plus retry policy and fallbacks.
- Final QC report.
- `main.py status`, `main.py resume`, `main.py clean --keep final`.

**Acceptance:**
- Deliberately corrupting 2 clips and 1 narration file leads to only those being regenerated.
- The QC report lists every hard and soft check.
- `clean` reduces a finished project to under 400 MB while keeping the plan, final video and logs.

### M6 — YouTube publishing

**Build:** follows `07_YOUTUBE_AUTOMATION.md`.
- `main.py auth youtube` (InstalledAppFlow, loopback).
- Token storage with mode 600 / keyring.
- Resumable upload with a stored session URI, `thumbnails.set`, `captions.insert`, the publish record and idempotency check.

**Acceptance:**
- A test video uploads as **private** to your channel. Its `video_id` and URL are recorded.
- Re-running `publish` does not create a duplicate.
- Killing the upload at 50% and re-running resumes it.
- No credential is in the repo (`gitleaks` passes).

### M7 — Full 3-minute documentary

**Build:**
- Run the pyramids example (doc 06) end to end.
- Tune the style preset, the I2V share and the QC thresholds from what you see.

**Acceptance:**
- A 3-minute video passes final QC and human review, and is uploaded private.
- The total cost and time are recorded in `publish_record.json`.

---

## Phase 4 — Hardening and production

| Item | Detail | Trigger |
|---|---|---|
| Rented GPU worker | Same worker package on RunPod/Vast/Lambda with an S3/R2 bucket instead of Drive | Colab GPU availability becomes the bottleneck |
| Higher-quality I2V | Wan 2.2 I2V-A14B (+ LightX2V distill LoRA) on a 48–80 GB GPU | Visual quality goal not met by 5B |
| Upscaling / interpolation | SeedVR2 or Real-ESRGAN, RIFE | Native 720p looks soft at 1080p |
| Music generation | ACE-Step 1.5 adapter | Library tracks feel repetitive |
| Shorts | Separate 9:16 storyboard pass, ≤ 60–180 s | Channel strategy |
| Scheduling | `--publish-at` after the API audit | API project audited |
| Durable orchestration | DBOS or Prefect if you run several projects concurrently | > 1 video/day |
| Regression suite | 10 golden plans rendered in `draft` profile in CI | Before every model/prompt change |
| New recipes | Built as tested project code from the wish backlog (for example a Blender `cutaway_3d` recipe) | A wish recurs in ≥ 3 videos, or you decide it matters |
| QC repair tool calls | LLM chooses among typed actions (`regenerate`, `swap_recipe`, `retrim`) when deterministic retries fail | Deterministic retry rules leave too many shots on fallback |
| Timeline export | OpenTimelineIO or MLT XML export for manual touch-ups in Resolve/Kdenlive | You want to hand-edit some videos |
| Local GPU | 24 GB NVIDIA card (RTX 3090/4090/5090 class) runs most adapters locally | Ongoing Colab/rental spend exceeds the hardware cost |

---

## Risks to watch during implementation

| Risk | Early signal | Mitigation |
|---|---|---|
| Colab gives no GPU or a GPU that is too small for I2V | M4 benchmarks | Planner lowers the I2V share automatically; consider Colab Pay-as-you-go or a rented GPU |
| Laptop RAM (7.4 GB) during FFmpeg renders | OOM / swap in M1 | Render shot segments one at a time; avoid a single huge filtergraph; use `-threads` limits |
| LLM plans violate timing or schema | M2 failure rate | Stricter schema, smaller per-section calls, validator feedback loop |
| Model license changes | Monthly check | `models.yaml` records the license and checked date; the CI test fails if `commercial_ok` is false for an enabled adapter |
| YouTube API project not audited | Public/scheduled uploads forced to private (official docs currently conflict on whether this still applies; see doc 07 §2.2) | Private is the default anyway. Test once; apply for the audit only if API-driven public uploads are blocked |

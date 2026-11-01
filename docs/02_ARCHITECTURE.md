# 02 — Architecture

**Project:** AI Video Studio, an open-source pipeline that turns a topic into a narrated documentary video and uploads it to YouTube
**Status:** Phase 1 (research and design). No implementation code yet.
**Date:** 2026-10-11

Labels used in all documents:
- **[O]** is verified from an official source (repo, model card, license, docs) during this research.
- **[C]** is community or third-party information.
- **[est]** is an engineering estimate. Measure it before relying on it.
- **[verify]** is uncertain and must be checked before use.

Model and tool choices are justified in `01_RESEARCH_REPORT.md` and `03_TECHNOLOGY_COMPARISON.md`. This document describes the **shape** of the system: components, data, control flow, failure handling and resume.

---

## 1. Design principles

1. **The LLM produces data, not running code.** Every creative decision (script, shots, camera, prompts, music cues, metadata) is a validated JSON document. Fixed, tested Python code turns those documents into media. This is the main reliability decision. There is **no run-time code generation**. Variety comes from a growing menu of tested recipes, with a non-blocking fallback for unsupported requests (§9).
2. **Short shots, then edit.** No open model reliably produces long continuous video. Every shot is 2–8 s, generated on its own and joined by an editor (FFmpeg). Continuity comes from shared keyframes, locked descriptions and editing, not from long generations.
3. **Voice first.** Narration is synthesized and word-aligned **before** any visuals are generated. The narration timing sets each shot's exact duration, so nothing is stretched to fit afterwards.
4. **Every stage leaves an inspectable artifact on disk.** Each stage reads files and writes files. You can open, edit, replace or re-run any stage on its own (for example, hand-edit `script.json` and re-run from there).
5. **Resume is free.** Every generated asset is keyed by a hash of its inputs. Re-running a project skips everything that already exists and is valid.
6. **Models are plugins.** Each capability (T2I, I2V, TTS, music, and so on) is an interface with interchangeable backends, selected in `config/models.yaml`. Swapping Wan for a newer model changes one adapter, not the pipeline.
7. **The laptop is the control plane and the GPU is a worker.** The local machine plans, edits, checks and publishes. GPU work runs as **batch jobs** on Colab or a rented GPU. Credentials for YouTube never leave the laptop.
8. **Private by default, human approval before publication.**

---

## 2. System overview

```text
┌───────────────────────────── LOCAL UBUNTU (control plane, CPU) ───────────────────────────────┐
│                                                                                               │
│  main.py create ──► Orchestrator (plain Python, stage graph, checkpoints, logs)                │
│                       │                                                                       │
│     ┌─────────────────┼───────────────────────────────────────────────┐                       │
│     │ PLANNING (LLM)  │  Director → Researcher → Scriptwriter →        │  LLM provider:        │
│     │                 │  Storyboard → Critic   (typed JSON stages)     │  hosted API or local  │
│     └─────────────────┴───────────────────────────────────────────────┘  open model           │
│                       │                                                                       │
│     Resource planner: detect hardware ─► route each job to LOCAL or REMOTE GPU                │
│                       │                                                                       │
│     LOCAL CPU jobs: TTS (Kokoro/Piper), alignment (faster-whisper), graphics (Pillow),        │
│                     camera moves on stills, FFmpeg edit/mix/encode, QC, thumbnail text        │
│                       │                                                                       │
│     Publisher: OAuth token (local only) ──► YouTube Data API v3 (private upload)              │
└───────────────────────┼───────────────────────────────────────────────────────────────────────┘
                        │ job files + results (rclone ⇄ Google Drive folder, or S3/R2 bucket)
┌───────────────────────▼───────────── REMOTE GPU WORKER (Colab or rented GPU) ─────────────────┐
│  worker notebook / script: reads job JSONs ─► runs model adapters ─► writes results + done.json│
│  T2I keyframes · I2V / T2V clips · (optional) GPU TTS, music, SFX, upscale, interpolation     │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Why this split:** the reference laptop (i5-1335U, 7.4 GB RAM, no NVIDIA GPU, checked 2026-10-11) can plan, synthesize speech on CPU, run FFmpeg and publish, but it cannot run diffusion models in reasonable time. The design does not change when the GPU is local: if the resource planner finds a suitable NVIDIA GPU, the same job files are executed by a local worker process instead.

---

## 3. Agents: one LLM, several typed stages

### 3.1 Single agent vs multiple agents

| Option | Reliability | Debuggability | Cost | Verdict |
|---|---|---|---|---|
| One giant prompt ("make the whole video plan") | Low. A 3-min video has ~30–40 shots, and quality drops with output size | Poor | Lowest | ✖ |
| Free-form multi-agent chat (agents talk to each other) | Low. Non-deterministic, loops, hard to resume | Poor | High | ✖ |
| **Fixed sequence of typed stages, each one LLM call with one JSON schema** | **High.** Each output is validated, checkpointed and retryable on its own | **Good.** Each stage has a file you can read | Low | **✔ Chosen** |

The "agents" in the brief are **roles**, not independent processes. Some are LLM calls, and most are deterministic code:

| Role | Implemented as | LLM? | Output artifact |
|---|---|---|---|
| Director | LLM stage: brief, angle, style, structure, duration budget | ✔ | `01_plan/brief.json` |
| Researcher | LLM stage with web search/fetch tools; collects claims + sources + confidence | ✔ | `01_plan/research.json` |
| Scriptwriter | LLM stage: narrative, narration text per beat | ✔ | `01_plan/script.json` |
| Storyboard / cinematographer | LLM stage, one call per section: shots, camera, lens, movement, prompts, transitions, audio cues | ✔ | `01_plan/storyboard.json` |
| Critic | Deterministic validator (Pydantic + rules) **then** one LLM review pass that returns a list of fixes | Partly | `01_plan/plan_review.json` → `production_plan.json` |
| Resource planner | Deterministic code | ✖ | `02_jobs/routing.json` |
| Visual generator | Deterministic code calling model adapters | ✖ | `03_visuals/…` |
| Audio agent | Deterministic code calling TTS / music / SFX adapters | ✖ | `04_audio/…` |
| Video editor | Deterministic code: timeline builder + FFmpeg renderer | ✖ | `05_render/…` |
| Quality controller | Deterministic checks (FFmpeg, ASR, image metrics) **plus** an optional vision-LLM review of a contact sheet | Partly | `06_qc/qc_report.json` |
| Publisher | LLM stage for metadata, deterministic code for upload | Partly | `07_publish/metadata.json`, `publish_record.json` |

**Framework:** none. The stages are ordinary Python functions called in order by a ~200-line orchestrator. Section 6 of `03_TECHNOLOGY_COMPARISON.md` compares LangGraph, CrewAI, AutoGen, PydanticAI and others and explains why they add little here. If research ever needs a tool loop (search → read → search again), use **PydanticAI** or the LLM vendor's SDK tool loop *inside* the Researcher stage only.

### 3.2 LLM provider

The planning stages talk to an `LLMProvider` interface with one method, `generate_structured(prompt, schema) -> dict`. Backends:

| Backend | Quality for long structured plans | Runs on this laptop | Cost | Use |
|---|---|---|---|---|
| Hosted frontier API (for example Anthropic Claude with structured outputs) | Best | ✔ (API call) | Roughly $0.10–1 per video [est] | **Default for quality** (not open source) |
| Open-weight model via Ollama / llama.cpp **on the laptop** (Qwen3 4B class) | Weak for 30+ shot storyboards [est] | Barely, given 8 GB RAM | Free | Offline fallback, metadata only |
| Open-weight model on a GPU (Qwen3 14B–32B class via vLLM/Ollama on Colab or a rented GPU) | Good | ✖ | GPU time | Fully open-source mode |

Whatever the backend, the output is checked against the same JSON Schema. Invalid output is sent back with the validation errors (max 2 retries), then the stage fails visibly.

---

## 4. Shot render modes: the core visual strategy

Documentaries do not need every second to be generated video. Each shot in the plan picks a **render mode**. The storyboard stage proposes it. The resource planner may downgrade it if no suitable GPU is available.

| Mode | How it is made | Cost | Reliability | Use for |
|---|---|---|---|---|
| `still_motion` | T2I keyframe → camera move rendered by FFmpeg (push-in, pull-out, pan, parallax-like crop) | Lowest | **High** | Establishing shots, landscapes, artefacts, reconstructions, most documentary b-roll |
| `i2v` | T2I keyframe → image-to-video model animates it (camera move + subject motion) | Medium-high | Medium | Hero shots, people working, water, fire, crowds |
| `t2v` | Text-to-video directly | Medium-high | Medium-low | Generic motion b-roll with no recurring subject |
| `graphic` | Deterministic rendering: Pillow/Matplotlib/SVG maps, timelines, diagrams, title cards; then camera move | Lowest | **Highest** | Dates, numbers, maps, diagrams, anything that must be **factually exact** |
| `flf2v` (optional) | First + last keyframe → model interpolates the motion | High | Medium | Designed transitions (for example, a quarry turning into a finished pyramid) |

Rules:
- **Every `i2v` shot starts from a keyframe image.** The keyframe locks composition, style and the look of recurring subjects. This is the main continuity tool.
- If the I2V attempt fails QC twice, the shot **falls back to `still_motion`** with the same keyframe, so the timeline never has a hole.
- **Anything with text, numbers or maps is `graphic`.** Diffusion models misspell text and invent geography.
- **Mix target for a 3-min documentary:** about 50–70% `still_motion`/`graphic` and 30–50% `i2v`. This depends on the GPU budget and is set in `settings.yaml`.

---

## 5. Data flow and the intermediate format

### 5.1 Project folder: the contract between all stages

```text
projects/<slug>/
├── project.json                  # id, topic, settings snapshot, schema versions, status per stage
├── 01_plan/
│   ├── brief.json                # Director
│   ├── research.json             # claims, sources, confidence, uncertainties
│   ├── script.json               # sections → beats → narration sentences
│   ├── storyboard.json           # sections → shots (camera, prompts, modes, cues)
│   ├── plan_review.json          # validator + critic findings and applied fixes
│   └── production_plan.json      # FINAL merged, validated, timed plan (single source of truth)
├── 02_jobs/
│   ├── routing.json              # job → local | remote, with reasons
│   └── remote/<job_id>.json      # job files sent to the GPU worker
├── 03_visuals/
│   ├── keyframes/<shot_id>.png   (+ <shot_id>.json: model, seed, prompt, hash)
│   ├── clips/<shot_id>.mp4       (+ sidecar json)
│   ├── graphics/<shot_id>.png
│   └── rejected/                 # failed attempts kept for debugging (auto-deleted later)
├── 04_audio/
│   ├── narration/<sentence_id>.wav
│   ├── narration.wav  words.json # full voice track + word timestamps
│   ├── music/<section_id>.wav
│   ├── sfx/<cue_id>.wav
│   └── mix.wav                   # ducked, loudness-normalized master
├── 05_render/
│   ├── timeline.json             # exact edit decision list (EDL)
│   ├── shots/<shot_id>.mp4       # normalized per-shot segments
│   ├── subtitles.srt  subtitles.ass
│   ├── final_16x9.mp4  final_9x16.mp4 (optional)
│   └── contact_sheet.jpg
├── 06_qc/qc_report.json
├── 07_publish/
│   ├── metadata.json  thumbnail.png  thumbnail_concepts.json
│   ├── approval.json             # who approved, when, privacy target
│   └── publish_record.json       # video id, URL, upload time, API responses
└── logs/
    ├── pipeline.jsonl            # structured log, one line per event
    └── llm/<stage>_<n>.json      # prompts + raw responses (for audit and prompt tuning)
```

Every JSON document carries `schema_version`. Every media file has a JSON sidecar recording **what made it**: model ID, model revision, adapter version, prompt, seed, parameters, input hashes, wall time and the GPU used.

### 5.2 The production plan schema

The production plan is the hand-off point between planning (LLM) and production (deterministic code). Pydantic models are the source of truth, and the JSON Schema is exported from them and given to the LLM.

```python
# schemas/plan.py (abridged)
from typing import Literal
from pydantic import BaseModel, Field

ShotSize   = Literal["extreme_wide", "wide", "medium", "medium_close", "close_up",
                     "extreme_close_up", "aerial", "insert", "pov", "over_shoulder"]
CameraMove = Literal["static", "push_in", "pull_out", "pan_left", "pan_right", "tilt_up",
                     "tilt_down", "truck_left", "truck_right", "crane_up", "crane_down",
                     "orbit_left", "orbit_right", "handheld", "drone_forward"]
RenderMode = Literal["still_motion", "i2v", "t2v", "graphic", "flf2v"]
Transition = Literal["cut", "crossfade", "dip_to_black", "fade_from_black", "fade_to_black", "wipe"]

class Camera(BaseModel):
    shot_size: ShotSize
    angle: Literal["eye_level", "low", "high", "birds_eye", "dutch", "worms_eye"]
    lens_mm: int = Field(ge=12, le=200)          # 24 wide, 50 normal, 85+ portrait/compression
    move: CameraMove
    move_speed: Literal["very_slow", "slow", "medium", "fast"]
    depth_of_field: Literal["deep", "medium", "shallow"]

class AudioCue(BaseModel):
    cue_id: str
    kind: Literal["sfx", "ambience", "stinger"]
    description: str                             # "distant hammering on limestone, wind"
    offset_s: float = 0.0
    gain_db: float = -18

class RecipeCall(BaseModel):
    recipe: str                                  # must exist in the registry; unknown names → fallback table
    params: dict                                 # validated by that recipe's own Pydantic model

class Wish(BaseModel):
    description: str
    closest_recipe: str
    reason: str

class Shot(BaseModel):
    shot_id: str                                 # "s02_sh03"
    beat_id: str                                 # links to script beat
    narration_ids: list[str]                     # sentences spoken over this shot
    duration_s: float = Field(ge=1.5, le=10)     # locked after TTS alignment
    purpose: str                                 # what the viewer must understand
    render_mode: RenderMode
    camera: Camera
    subject: str
    action: str                                  # what moves (empty for still_motion)
    environment: str
    lighting: str
    time_of_day: str
    continuity_refs: list[str] = []              # character/location/object ids from the bible
    image_prompt: str                            # keyframe prompt (T2I)
    motion_prompt: str = ""                      # I2V/T2V prompt: motion + camera only
    negative_prompt: str = ""
    visual: RecipeCall                           # {recipe: "still_move"|"i2v_clip"|"map_route"|…, params: {...}}
                                                 # validated against the recipe registry (§9.1)
    overlays: list[RecipeCall] = []              # e.g. lower_third
    wish: Wish | None = None                     # what the planner really wanted if the menu lacked it (§9.2)
    on_screen_text: str | None = None            # rendered by code, never by the image model
    transition_in: Transition = "cut"
    audio_cues: list[AudioCue] = []
    factual_status: Literal["verified", "reconstruction", "illustrative"] = "illustrative"
    claim_ids: list[str] = []                    # research claims this shot depicts

class Section(BaseModel):
    section_id: str
    title: str                                   # also used for YouTube chapters
    mood: str
    music_brief: str                             # "low strings, slow tempo, sparse percussion"
    color_grade: str                             # name of a LUT/preset in config
    shots: list[Shot]

class BibleEntry(BaseModel):
    ref_id: str                                  # "loc_giza_plateau", "char_overseer"
    kind: Literal["character", "location", "object"]
    canonical_description: str                   # pasted verbatim into every prompt that uses it
    reference_image: str | None = None           # generated once, reused as I2V/T2I conditioning

class ProductionPlan(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    topic: str
    format: Literal["youtube_16x9", "shorts_9x16"]
    resolution: tuple[int, int]                  # (1920,1080) or (1080,1920)
    fps: Literal[24, 25, 30]
    target_duration_s: int
    style: dict                                  # style preset id + global prompt suffix + negative
    narrator: dict                               # tts backend, voice id, speed, language
    bible: list[BibleEntry]
    sections: list[Section]
    disclosure: dict                             # {"contains_synthetic_media": true, "reason": "..."}
```

The JSON Schema export of this model goes into the LLM prompts. A representative filled shot is in `06_EXAMPLE_VIDEO_WORKFLOW.md`.

### 5.3 Validator rules (deterministic, run before anything is generated)

| Rule | Why |
|---|---|
| Sum of shot durations = narration length ± 3% (after alignment) | Timeline integrity |
| Every `continuity_refs` ID exists in the bible | Consistency |
| Every shot that shows a recurring subject has its bible description in `image_prompt` | Consistency |
| No `on_screen_text` inside `image_prompt` / `motion_prompt` | Models garble text |
| `i2v`/`t2v` shots ≤ the selected model's max clip length (for example ≤ 5 s for Wan at 16 fps [O]) | Model limits |
| Adjacent shots differ in shot size or camera move | Visual rhythm |
| Narration pace 2.0–3.0 words/s (≈ 120–180 wpm) | Comprehension |
| Every factual claim in narration maps to a `research.json` claim with confidence ≥ "medium" or is phrased as uncertain | Factual integrity |
| Shots depicting reconstructions are marked `factual_status: reconstruction` | Honesty, disclosure |
| Prompt length within model token limits | Avoid silent truncation |

---

## 6. Orchestration, checkpoints and resume

### 6.1 Stage graph

```text
plan ─► voice ─► lock_timing ─► route ─┬─► keyframes ─► clips ──┐
                                       ├─► graphics ────────────┤
                                       └─► music, sfx ──────────┼─► shots ─► edit/mix ─► subtitles ─► qc ─► package ─► [approve] ─► publish
```

The orchestrator is a plain Python loop over stages. Each stage declares:
- `inputs`: the files it reads.
- `outputs`: the files it writes.
- `run(project) -> None`.
- `validate(project) -> list[str]`, the post-conditions for its outputs.

### 6.2 Checkpoint rule

A stage (or one item inside a stage, such as one shot) is **skipped** when all of these hold:
1. Its output files exist.
2. Its sidecar `input_hash` equals the hash of its current inputs: upstream file hashes, the relevant config slice, the adapter version and the seed.
3. Its `validate()` passes.

```python
def input_hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:                       # dicts are dumped with sort_keys=True; files are hashed by content
        h.update(_canonical_bytes(p))
    return h.hexdigest()[:16]
```

Consequences:
- **Kill the process at any point and re-run `create`.** Finished shots are reused, and only missing or invalid ones are made.
- **Edit one shot's prompt in `production_plan.json`.** Only that shot's keyframe, clip and segment and the final render are redone.
- **Change the voice.** Narration, timing and everything downstream are redone. Keyframes are reused if the shot list did not change.

`project.json` keeps a per-stage status (`pending | running | done | failed`) for humans and for `main.py status`. The hashes, not the status flags, decide what re-runs.

### 6.3 Remote job protocol (Colab / rented GPU)

GPU work is shipped as **job files**. The worker is stateless.

```json
{
  "job_id": "kf_s02_sh03_7c1e9a",
  "project_id": "pyramids_2026_10_11",
  "kind": "keyframe",
  "adapter": "t2i.z_image_turbo",
  "params": {"prompt": "...", "negative": "...", "width": 1920, "height": 1088, "steps": 8, "seed": 41872},
  "inputs": [],
  "outputs": ["03_visuals/keyframes/s02_sh03.png"],
  "input_hash": "7c1e9a0b2f4d55e1",
  "created_at": "2026-10-11T10:02:11Z"
}
```

1. The **local** orchestrator writes job files to `jobs/pending/` in the shared folder and uploads any input files (for example keyframes for I2V).
2. The **worker** (a Colab notebook cell loop or `python -m worker` on a rented GPU) claims a job by moving it to `jobs/running/` with a lease timestamp. It loads the model **once per job kind** to amortize load time, runs the job, writes outputs and sidecars, and then writes `jobs/done/<job_id>.json` (or `jobs/failed/<job_id>.json` with the traceback).
3. **Local** pulls `done/` results, verifies hashes and sizes, and marks shots complete. Stale leases (worker died) return to `pending/` after a timeout.
4. **Idempotent:** a job whose output already exists with a matching hash is skipped by the worker too.

The transport is a shared folder: Google Drive via `rclone` on the laptop and `drive.mount()` in Colab, or any S3-compatible bucket for rented GPUs. Details and limits are in `04_EXECUTION_STRATEGY.md`.

---

## 7. Failure handling

| Failure | Detection | Automatic response | Escalation |
|---|---|---|---|
| LLM returns invalid JSON / schema error | Pydantic validation | Re-ask with the error list (≤ 2) | Stage fails. You can fix the file by hand and resume |
| Factual claim with low confidence | Research confidence + validator | Script rephrases with hedging ("historians believe…") or drops the claim | Listed in `plan_review.json` |
| Keyframe artifact (garbled text, extra limbs, wrong subject) | Vision check (CLIP-style similarity and/or VLM yes/no) | New seed (≤ 3) | Fallback prompt, then `graphic` or a flag |
| I2V clip broken (warping, freezing, flicker) | Motion and frame-difference stats, VLM check on 3 frames | New seed (≤ 2) | Downgrade to `still_motion` |
| Colab session dies / GPU unavailable | Lease timeout, no heartbeat | Job returns to the queue. The next session continues | Planner downgrades remaining `i2v` to `still_motion` if a deadline is set |
| Out of GPU memory | Worker catches CUDA OOM | Retry with lower resolution / CPU offload / fewer frames | Mark the job `failed` with the reason |
| TTS mispronunciation | ASR word-error rate per sentence > threshold | Regenerate the sentence (≤ 2), using a phonetic respelling list from config | Flag the sentence |
| Render error | FFmpeg exit code + stderr | Re-run that shot segment | Fail with the FFmpeg log |
| Final QC failure | `qc_report.json` | Re-run only the listed shots or stages | Human review |
| YouTube upload interrupted | HTTP error / exception | Resumable upload continues from the last byte, with exponential backoff | Stored session URI allows a later resume |
| YouTube quota exceeded | `quotaExceeded` error | **Do not retry.** Wait for the daily reset | Logged |

---

## 8. Quality control (summary)

Section 5 of `04_EXECUTION_STRATEGY.md` lists the thresholds per stage. The final gate checks:
- **Technical:** resolution, fps, codec, duration, black frames, frozen frames, silence, loudness, true peak.
- **Speech:** ASR transcript of the final mix vs the script.
- **Subtitles:** line length, reading speed, sync.
- **Visual:** duplicate shots (perceptual hash), contact-sheet review by a vision-capable LLM (optional).
- **Content:** every narrated factual claim is backed by a source, and the synthetic-media disclosure flag is set.

The video cannot be published while any **hard** check fails.

---

## 9. Fixed executor, recipe menu and generation workflows (no run-time code generation)

**Decision (2026-10-11): Claude writes only the plan. One fixed, tested executor renders every video.**
- Run-time code generation was the core of the older design (`ai_video_automation_research.md`, Revision 2, now superseded).
- It is dropped. Documentaries share one structure (shots, narration, music, transitions), and only the content varies, so a JSON plan can describe every video.
- Generated code would need a sandbox, policy checks and a repair loop only to make it safe. Without it, this design needs no code sandbox.

The plan's flexibility comes from four mechanisms.

### 9.1 Recipe menu: a small vocabulary of named, tested effects

Every visual, transition and audio treatment in the plan is a **recipe**: a name plus parameters that Pydantic checks. Each recipe is a function in `rendering/recipes/` with its own tests. The LLM picks recipes and fills in their parameters. It never writes code.

**Start with about 9 recipes, the ones the first videos actually need:**

| Recipe | Kind | Parameters (validated) | Implementation |
|---|---|---|---|
| `still_move` | visual | `image`, `move` ∈ {static, push_in, pull_out, pan_left, pan_right, tilt_up, tilt_down}, `speed`, `focus_xy` | FFmpeg zoompan/crop on a 2× supersampled keyframe |
| `i2v_clip` | visual | `keyframe`, `motion_prompt`, `seed`, `trim` | GPU worker (Wan 2.2) → trim/scale; falls back to `still_move` |
| `title_card` | visual | `title`, `subtitle`, `style` | Pillow |
| `map_route` | visual | `places[] {name, lat, lon}`, `route[]`, `extent` | Matplotlib + Natural Earth |
| `comparison_panels` | visual | `panels[] {label, caption, image?}` (2–4) | Pillow layout, sequential reveal |
| `timeline_graphic` | visual | `events[] {date_label, text}` | Matplotlib/Pillow |
| `lower_third` | overlay | `text`, `start_s`, `dur_s` | ASS/drawtext overlay |
| `transition` | edit | `type` ∈ {cut, crossfade, dip_to_black, fade_from_black, fade_to_black}, `dur_s` | FFmpeg `xfade` |
| `music_bed` / `sfx_at` | audio | `track` or `cue`, `gain_db`, `duck_under_voice`, `offset_s` | FFmpeg `sidechaincompress`, `adelay`, `amix` |

**Rules:**
- **Grow the menu only on demand.** A recipe is added only when real plans ask for it (§9.2). There is no up-front catalogue of 40 effects.
- Each recipe has:
  - a **version**, recorded in the shot sidecar, which makes it part of the resume hash;
  - a **golden test**: a fixed input producing a reference output, compared by duration, size and a perceptual hash;
  - a declared **fallback recipe**.
- The JSON Schema given to the LLM is generated from the recipe registry, so the planner sees exactly the menu that exists.

### 9.2 "Unsupported" requests: fall back, log, never block

The planner may want something the menu cannot express, such as a split-screen, a time-lapse build-up or an animated 3D cutaway. **The video is never blocked.** Instead:

1. **The planner always emits a supported recipe** (the closest one), plus an optional `wish` describing what it really wanted:

   ```json
   "visual": {"recipe": "comparison_panels", "params": {"panels": [ ... ]}},
   "wish": {"description": "animated 3D cutaway showing an internal spiral ramp rising through the pyramid",
            "closest_recipe": "comparison_panels", "reason": "no 3D recipe available"}
   ```

2. **The validator never fails on an unknown recipe name.** If the LLM invents one anyway, the validator maps it to the closest recipe through a static fallback table (`config/recipe_fallbacks.yaml`, for example `split_screen → comparison_panels`, `timelapse_build → i2v_clip`, `anything_else → still_move`). It records the substitution as a `wish`.
3. Every `wish` is written to `01_plan/recipe_requests.json` and appended to a global `recipe_backlog.jsonl` (project, shot, description, fallback used, date).
4. `main.py backlog` groups the wishes by similarity and counts them. **A new recipe is built only when a wish recurs**, for example in three or more videos, or when you decide it matters. It is written and tested as normal project code (Claude Code can help at development time), and then it joins the menu.

So every new idea in a plan becomes **backlog data, not a production stop**.

### 9.3 Generation workflows: pinned community ComfyUI workflows as an adapter type

On the generation side, the worker can run **fixed, community-tuned ComfyUI workflows**, which are often better tuned than hand-written diffusers code (samplers, schedulers, LightX2V LoRA strengths, VAE tiling, block swap). Claude still only supplies values.

| Item | Rule |
|---|---|
| Adapter type | `comfy_workflow`, alongside `diffusers` in `config/models.yaml` |
| Workflow file | API-format JSON checked into `worker/workflows/<name>@<version>.json`, with its **source URL, author, licence and sha256** recorded |
| Inputs | Declared **slots** only, such as `prompt`, `negative`, `seed`, `frames`, `width`, `height`, `image`. The adapter fills node inputs by slot mapping and changes nothing else in the graph |
| Pinning | ComfyUI **commit**, each custom node's **repo + commit**, each model file's **filename + sha256**, recorded in `worker/workflows/<name>.lock.yaml`. The worker refuses to run if the installed versions differ |
| Execution | ComfyUI runs **headless inside the worker** (bound to 127.0.0.1, never exposed). The adapter submits the graph to the local API and collects the outputs. On Colab it runs as a normal process inside the notebook: no tunnel, no web UI |
| Testing | One golden render per workflow at a fixed seed (duration, size, perceptual-hash tolerance), re-run whenever anything in the lock file changes |
| Licences | ComfyUI is GPL-3.0, which is fine to run as a separate tool. Check each workflow's and custom node's licence before adoption. **WanGP** is not OSI-licensed and bans hosted/SaaS use, so it is fine for experiments but not as the embedded engine |

**Recommended split:**
- Use **diffusers** adapters for the simplest models (keyframes).
- Use **pinned ComfyUI workflows** where the community tuning clearly wins: Wan 2.2 I2V with LightX2V, A14B GGUF on 16–24 GB, FLF2V, SeedVR2 upscaling.
- Benchmark both paths in Milestone M4 and keep whichever is better per adapter.

### 9.4 Templates and tool calls

- **Templates (fixed code, filled values).** The per-project Colab launcher cell and run parameters are filled from human-written Jinja2 templates.
- **Tool calls only in the QC repair loop (optional, Phase 4).** When a shot fails QC, an LLM may choose among typed, validated actions: `regenerate(shot, seed, prompt_edit)`, `swap_recipe(shot, fallback)`, `retrim(shot, in_s, out_s)`. Each call is logged. Budgets and licences are enforced by the executor. Deterministic retry rules (§7) remain the default.

### 9.5 Standard timeline formats (optional, later)

`timeline.json` can additionally be exported to an editor-readable format for manual touch-ups:

| Format | Licence | Notes |
|---|---|---|
| **OpenTimelineIO** | Apache-2.0 | Converts to Resolve/Premiere/FCP via adapters |
| **MLT XML** (rendered by `melt`) | LGPL-2.1 | Opens in Kdenlive and Shotcut |
| **Editly** JSON | MIT [UNVERIFIED this session] | Node-based declarative editor |
| Remotion | **Custom licence**: free for individuals and companies of ≤ 3 employees; larger companies need a paid company licence | Check the terms before choosing it |

The FFmpeg executor remains the renderer of record. Exports are a convenience.

**Never generated by the LLM:** model-loading code, the GPU worker, recipes, the FFmpeg renderer, QC and the YouTube publisher. All of them are versioned, tested project code.

---

## 10. Security model

| Asset | Where it lives | Who can read it |
|---|---|---|
| YouTube OAuth client secret (`client_secret.json`) | `~/.config/ai-video-studio/` (mode 600), never in the repo | Local publisher only |
| YouTube refresh token | Same folder (mode 600), or the system keyring via `keyring` | Local publisher only |
| LLM API key | `.env` (git-ignored) or environment variable | Local planner only |
| Hugging Face token (gated models only) | Colab **Secrets** panel / environment variable on the GPU host | Worker only |
| Shared job folder (Drive/bucket) | Contains prompts and media, **no secrets** | Laptop + worker |

- `.gitignore` ships with `.env`, `*.json` credential patterns, `projects/*/` media and `secrets/`.
- A pre-commit hook (`detect-secrets` or `gitleaks`) blocks committing secrets.
- The Colab worker never receives YouTube credentials.

---

## 11. Output formats

| Profile | Resolution | FPS | Video | Audio | Notes |
|---|---|---|---|---|---|
| `youtube` | 1920×1080 | 24 or 30 | H.264 High, CRF 18–20, yuv420p, closed GOP, `+faststart` | AAC-LC 48 kHz stereo 192–384 kb/s | Default |
| `shorts` | 1080×1920 | 30 | Same | Same | ≤ 3 min. Separate plan or reframed crop |
| `archive` | 1920×1080 | source | H.265/HEVC CRF 20 or ProRes | PCM/AAC | Optional master for re-edits |
| `draft` | 960×540 | 24 | H.264 `veryfast`, CRF 26 | AAC 128k | Fast preview |

- **Vertical output** is made by a separate plan (the storyboard is re-composed for 9:16), not by blindly cropping 16:9. The cropping fallback uses a centre-weighted crop on `still_motion` shots only.
- The renderer always normalizes every shot to the target resolution, fps, pixel format and SAR before joining.

---

## 12. Architecture as built, and the changes (appended 2026-10-18)

Sections 1–11 describe the **original design**. The implementation evolved, and this section records how it
works now. Where it disagrees with an earlier section, **this section is current**. Earlier text is kept unchanged
because the docs are append-only. Details, measurements and tests for each change are in
`docs/08_COLAB_VIDEO_PIPELINE_GUIDE.md` (sections 30–38), `docs/09_LOGICAL_CONSISTENCY_ARCHITECTURE.md` and
`docs/10_CLAUDE_MD_AUDIT_AND_DECISIONS.md`.

### 12.1 What changed against sections 2–11

| Earlier design (section) | Now |
|---|---|
| Local laptop control plane + remote GPU worker exchanging job files (§2) | **One Google Colab notebook does everything** on the free T4: planning (topic notebook), voices, AI video, drawn scenes, edit, mix, QC. Drive holds the state, so a disconnect resumes. The laptop only builds the notebooks and runs the tests. |
| Planning by a hosted LLM API or a local model (§2, §3) | **Hand-written plan JSON** (one notebook per video), or the **topic notebook**: the local `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0) writes a draft from Wikipedia facts. **No paid or hosted LLM.** `tests/test_registry.py` enforces this. |
| LLM API key in `.env` (§10) | Not needed. The only secret is an optional **Hugging Face read token** (Telugu voice model), in Colab Secrets, never in code. |
| Shot render modes (§4) | The recipe menu is `video` (AI), `card`, `ui`, `journey`, `diagram` (drawn by code) and **`stock`** (licensed real image). Drawn and licensed-image shots need no GPU and are never "hallucinated". |
| Plan schema (§5) | Two variants: **narrator** (sentences per shot) and **cast** (characters, lines, tones). Extensions: **scene states** (§12.4), **claims** with source URL (+ revision for topic videos), `fallback` cards, `language`, a `tts` spelling per line. |
| Checkpoints (§6) | Every artifact is cached by a content key (sidecars: narration, prompt embeddings, clips, segments, drawn scenes, models on Drive). Topic stages are saved by `engine/lib/stages.py`. Run all after a disconnect redoes only what's missing or changed. |
| QC summary (§8) | **A single approval gate** (§12.6): `approved` only if every mandatory check is PASS; an UNKNOWN blocks like a FAIL. |
| Output `youtube` 1920×1080 (§11) | Unchanged default (`RUN_MODE = "standard"`, 24 fps). New `draft` and `cpu_fallback` modes render 1280×720; `cpu_fallback` uses no AI video. Subtitles are burned in **one** final encode; segments are encoded once, near-lossless, and cached. |

### 12.2 Current system overview

```text
┌──────────────────────────── LAPTOP (no GPU needed) ─────────────────────────────┐
│  plan JSON (hand-written)  ──►  colab/build.py  ──►  colab_<video>.ipynb         │
│  python3 build.py --topic   ──────────────────►  colab_topic_to_video.ipynb      │
│  engine/cells/{shared,cast,narrator,topic}/*.py + engine/lib/*.py (tested here)   │
└───────────────────────────────────────────────┬──────────────────────────────────┘
                                                 │ upload notebook
┌────────────────────────── GOOGLE COLAB FREE (T4 GPU, Drive) ─────────────────────┐
│ settings → log → install → setup → lib → [topic] → plan → helpers → prefetch →    │
│ narration → timeline → diagrams → keyframes → editlib → logic → bench → clips →   │
│ music → sfx → render → qc → metadata → preview                                    │
│                                                                                   │
│ GPU, ONE model family at a time (released between stages):                        │
│   planner (Qwen3-4B) → voice (Kokoro / Indic Parler-TTS in its own environment) →  │
│   UMT5 text encoder → VACE / Wan2.1 video → checkers (OWLv2 + Qwen3-VL) → repairs  │
│ CPU: research, fact check (NLI), drawn scenes, FFmpeg edit/mix/encode, QC          │
│ Drive: plan, narration, clips, segments, stages/, model caches, render/, logs/     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 12.3 Topic → video pipeline (new, 2026-10-18)

**Topic** → **research** (`research.py`: Wikipedia Action API, polite client; the HTML extract keeps exponents;
numbered facts with URL + revision) → **draft** (`planner.py`: Qwen writes JSON data only: hook candidates, beats,
cited fact ids, visual intent; Pydantic validation with retries) → **fact check** (`factcheck.py`: numbers and
names must be in the cited facts, plus NLI entailment with `cross-encoder/nli-deberta-v3-base`; one rewrite on the
same facts, then drop; repeat, symbol and length gates; too thin = **stop**) → **images** (`assets.py`: Commons
search → licence policy from `registry.py` → content-hash cache → SigLIP 2 ranking against the description **and**
the narration, distractors and the other beats → image or drawn card) → **plan** (deterministic builder; card
text must match the narration) → the **same** cast pipeline renders it.

### 12.4 Logical and spatial consistency (2026-10-16)

Scene-state shots carry a typed state: camera, positions, facing, holders and timed events, simulated before any
GPU work. They are generated with **Wan2.1-VACE-1.3B** steered by a layout video drawn from the plan, then checked
against the **same** plan:
- OWLv2 detection and motion checks;
- camera motion measured on the background, with planned and detected subjects masked, sub-pixel;
- `enter`/`exit`/`walk_to` judged by geometry;
- other actions confirmed by Qwen3-VL.

Failures get **targeted** repairs (`acceptance.py`); unresolved shots go to REVIEW and are replaced by their
drawn fallback. Full design: `docs/09_LOGICAL_CONSISTENCY_ARCHITECTURE.md`.

### 12.5 Colab T4 resource architecture (2026-10-16/17)

- **One model family on the GPU at a time.** Text encoder → VACE → checkers, in rounds. `release_model()`
  really frees weights, even when hidden references survive; `expandable_segments` is on; an out-of-memory error
  retries the clip with 25 % fewer frames.
- **System RAM** (12.7 GB): models load straight to the GPU (`device_map`); any RAM fallback is guarded by
  `need_ram()`; `malloc_trim` after releases; high-parallel downloads only on big-RAM machines.
- **Speed:**
  - the Self-Forcing turbo LoRA (4 steps, verified compatible: VACE base blocks are bit-identical to T2V-1.3B);
  - a timed, untiled-first VAE;
  - the all-zero VACE latent is reused (bit-identical);
  - per-stage timing and GPU sampling go to `perf.json`;
  - `BENCHMARK = True` gives measured T4 comparisons.
- **Isolated tool environments:** a model with conflicting pins (parler-tts needs `transformers 4.46`) runs in its
  own virtual environment as a JSON worker process, built from a pinned lock that reuses Colab's torch.

### 12.6 Final quality gate (2026-10-18)

`qclib.py` + the `qc` cell → `render/qc_report.json`. Each check has `status`, `evidence`, `shots` and `fix`.

- **Mandatory checks:**
  - a playable H.264/AAC stream at the right size and fps; no duration drift; ending not cut off;
  - no unplanned black frames; no frozen frames inside AI-video shots;
  - loudness on target; no clipping (true peak); no unintended silence; **measured** music ducking;
  - narration heard correctly (speech check); every AI shot verified;
  - every scene-state shot verified or approved; nothing flagged for review; plan validation clean;
  - music licensed and credited;
  - topic videos also: every narrated fact supported, and every image with an allowed licence and its credit.
- **Approval:** `approved = all mandatory PASS`. A contact sheet is saved for a person to look over.

### 12.7 Languages (2026-10-17)

`"language"` in the plan selects the voice, speech check, fonts and validator rules.

| Language | Voice | Speech check | Text and subtitles |
|---|---|---|---|
| **English** | Kokoro stock voices | `small.en` (word error) | word-highlighted ASS subtitles |
| **Telugu** | Indic Parler-TTS stock speakers (gated: needs an HF token) | multilingual Whisper (letter error) | Hind Guntur font (Telugu + Latin), raqm shaping; subtitles drawn with Pillow (libass broke Telugu vowel signs) |

Telugu lines carry a `tts` spelling so acronyms and numbers are pronounced correctly. The validator handles
Telugu word endings, hedges and time units.

### 12.8 Licensing architecture (2026-10-18)

- `engine/lib/registry.py` lists every model with its **verified weight licence** and a status:
  - `ok`: Apache-2.0 / MIT;
  - `review`: a known condition (the turbo LoRA repack has no licence tag; OpenAI CLIP's model card; SDXL's
    OpenRAIL++);
  - `blocked`.
- `LICENSE_STRICT` refuses `review` models.
- **Images:** public domain, CC0 and CC BY by default; CC BY-SA opt-in; NC, ND, GFDL-only and unknown licences
  rejected.
- **Music:** Kevin MacLeod, CC BY 4.0, credited.
- **Credits:** image and music credits go into the YouTube description; image credits also appear on screen.

### 12.9 Module map (current)

| Layer | Files |
|---|---|
| Notebook builder | `colab/build.py` (plan → notebook; `--topic` → topic notebook) |
| Cells | `engine/cells/shared/*` (setup, helpers, clips, logic, editlib, render, qc, …), `cast/*`, `narrator/*`, `topic/*` |
| Planning and facts | `research.py`, `planner.py`, `factcheck.py`, `validate.py` |
| Visuals | `drawkit.py` (drawn scenes, script fonts), `assets.py` (licensed images), `layout_control.py` (VACE layouts) |
| Consistency | `scene_state.py`, `motion_check.py`, `trackers.py`, `vlm_check.py`, `acceptance.py` |
| Audio and text | `musiclib.py`, `subtitles.py`, `parler_worker.py` |
| Quality, licences, resume | `qclib.py`, `registry.py`, `stages.py` |
| Tests | `colab/tests/test_*.py`: 10 suites, 221 checks, all passing on 2026-10-18 |

### 12.10 Still open

- **No measured T4 benchmarks yet.** Run `BENCHMARK = True` on Colab.
- **The 4B planner's output quality** hasn't been run on a T4.
- **SigLIP 2 thresholds** were calibrated on a small sample.
- **Deferred:** Pexels/Pixabay, music generation (ACE-Step), FLUX.2 klein, Qwen3-TTS, a vertical (Shorts) build.

## 13. Expressive narration layer (appended 2026-10-18)

Narration-first is unchanged; a delivery stage is inserted before speech synthesis:

```text
approved lines → story beats (chapter+beat across shots) → VoiceDirection per line
  (plan direction → author tone → narrative role; optional local-LLM director) → smooth() arc rules
  (profile range, backend cap, no jumps / dramatic runs / frequent peaks)
→ backend: Kokoro (pace + real pauses) | Qwen3-TTS 1.7B CustomVoice in its own environment (instruction) |
  Indic Parler-TTS for Telugu (description); DSP pitch/gain/tremolo are off when a direction is present
→ per-line voice QC (words with letter-aware diff, pace, gaps, clipping, pitch as a diagnostic) → calmer retake
→ identity post-pass (ECAPA leave-one-out, per-backend threshold) → calmer retake or flag
→ timeline (pause before/after, 8 ms fades, no per-chunk loudness normalisation) → word alignment → subtitles,
  shot lengths, music (−4 dB under serious narration; sidechain ducking) → render
```

**Modules:**

| Module | Purpose |
|---|---|
| `engine/lib/voice_direction.py` | directions, profiles, beats, arc smoothing, optional LLM director |
| `engine/lib/expressive_tts.py` | backend capability table, parameter mapping, honest per-field report |
| `engine/lib/qwen_tts_worker.py` | Qwen3-TTS worker process with its pinned lock |
| `engine/lib/voice_qc.py` | voice measurements, word diff, ECAPA identity check |
| `QwenVoice`, `install_tool_env` | narration cell: the expressive backend, and one installer shared with the Telugu voice |

**Licences** (`registry.py`): Qwen3-TTS 1.7B/0.6B Apache-2.0, ECAPA (SpeechBrain) Apache-2.0.
Full details: `docs/11_EXPRESSIVE_NARRATION.md`.

## 14. Two-friend storytelling layer (appended 2026-10-18)

Full report: `docs/12_TWO_FRIEND_STORYTELLING.md`.

- **New modules:**
  - `characters.py`: the Character Bible.
  - `charanim.py`: 2D characters drawn on the CPU.
  - `demos.py`: physical demonstrations and licensed photos.
  - `story.py`: Story Director + Visual Story Director.
- **New recipe `story`:**
  - A shot is a dialogue scene. Its `lines` are the audio units, placed narration-first exactly as before.
  - `story.visuals[]` is the scene's own visual timeline: each visual starts inside a line (`{"line": k, "at": 0..1}`). Audio never depends on visuals.
- **Visual fallback chain:** licensed media → demonstration → (budgeted whole-shot AI video) → minimal overlay → character scene. Re-enactments are labelled "Illustration".
- **Strict licences** (`LICENSE_STRICT`, default True for story plans): the turbo LoRA, SDXL and OpenAI CLIP are refused before download. Model revisions are recorded in `registry.REVISIONS`.
- **Preflight** (`render/preflight.json`): GPU, VRAM, precision (bf16 only if native), RAM, disk, HF token presence.
- **QC adds:**
  - dialogue order/overlap;
  - grounding of factual dialogue;
  - presentation-graphics share;
  - re-enactment labels;
  - story completeness;
  - two statistics (narrative purpose vs rendering medium);
  - `dialogue_shot_timeline.md`.
- **Old plans:** unchanged (`STORY_MODE = "narrator"`, `LICENSE_STRICT = False`, verified by an identical end-to-end result).

## 15. Natural voices and story engine v2 (appended 2026-10-18)

Full report: `docs/13_NATURAL_VOICES_AND_STORYTELLING.md`.

- **Story stages** (each cached in the topic notebook):
  - fact selection → premises (rule-scored) → outline (cause → effect) → dialogue beat by beat with story memory → critic (14 rules + an optional LLM critic that can only add weaknesses) → targeted revision → fact-aware repair. The repair rewrites a line first, otherwise removes it together with the question or reaction that depended on it.
- **Voice:**
  - The Voice Performance Director gives every friend's line a conversational intention from its context.
  - Response gaps replace narrator pauses; smoothing runs per speaker.
  - Bounded hero takes on sampling backends, with A/B.
  - The identity threshold depends on clip length; homophones are handled.
  - Conversation-timing diagnostics in QC.
- **The architecture is unchanged:** voice-first timeline, independent visual timeline, renderer, caches, QC gate.

## 16. Telugu voices via Gemini free-tier TTS (3D glasses video only) (appended 2026-10-19)

Full report: `docs/14_TELUGU_GEMINI_TTS.md`.

- **Scope:** only `colab_three_d_friends_te_v2.ipynb` uses it (plan key `tts_backend: "gemini_free"`). Every other plan and notebook is unchanged.
- **The one hosted exception, speech only:**
  - Google Gemini `gemini-3.8-flash-tts` (free tier), listed apart from the open models in `registry.HOSTED_EXCEPTIONS`.
  - Story writing, fact checks, visuals and rendering stay local and open.
- **Flow:**
  1. Fail-closed preflight: manual free-tier confirmation, key, SDK, approved model, local budget.
  2. Two-speaker conversation chunks (≤ 4 requests for 6 scenes).
  3. WAV kept as-is.
  4. Split per turn at real pauses (`turn_align.py`), using recogniser word times.
  5. Per-turn letter check, then speaker attribution.
  6. Per-turn `narration/<id>.wav`, so mouths, subtitles and caches are unchanged.
- **Fallback:** if anything fails, the whole video uses local Indic Parler.
- **Telugu dialogue:**
  - The native Telugu writer prompt and the Telugu critic (T0–T8) live in `telugu_dialogue.py`.
  - Factual Telugu lines are verified through their English canonical claim (NLI). A conservative cross-language term/number check sends anything uncertain to manual review.

## 17. Architecture point by point: the current state (appended 2026-10-19)

How a video is made today, in order. Each point names where it lives.

### A. Notebook and build
1. **One notebook per video, built from tested code:** `colab/build.py <plan>.json <notebook>.ipynb` joins the cells in `colab/engine/cells/` and embeds `colab/engine/lib/*.py`. The LLM writes only data (plans and stories), never code.
2. **Cell order:** settings → log → install → setup → lib → plan → helpers → prefetch → voices (narration) → timeline → drawn scenes (diagrams) → keyframes → editing → logic → bench → clips → music → sfx → render → QC → metadata → preview. The topic notebook adds a "topic" cell after lib.
3. **Plans:** `*_plan.json` holds characters, lines, shots, claims, visuals and music. The plan cell validates everything before any voice or GPU work. A `plan.json` on Drive with the same `plan_version` overrides the embedded plan.

### B. Story (two friends, Arjun and Meera)
4. **Character Bible** (`lib/characters.py`):
   - Two original characters with fixed looks and expressions.
   - Voices per language and backend: Kokoro (en), Indic Parler (te), optional Qwen3-TTS preset, optional Gemini prebuilt voice.
   - Pronunciation profile.
5. **Story Director** (`lib/story.py`, staged and cached): facts → 2 premises (chosen by rules) → cause-and-effect outline → dialogue beat by beat with story memory → critic (14 rules + optional LLM critic) → targeted revision → fact repair (rewrite or remove, plus anything that depended on it).
6. **Telugu writing** (`lib/telugu_dialogue.py`): natural spoken-Telugu writer prompt; Telugu critic T0–T8 (bookish words, calques, formulas, long speeches, too much English, tags in text, repeated reactions).
7. **Facts:**
   - English claims are checked against cited Wikipedia sentences: numbers/names gates + NLI (`lib/factcheck.py`).
   - Telugu lines carry their English claim (`source_text`, NLI-checked) and pass a term/number preservation check. Anything uncertain → manual review.
8. **Visual Story Director:** each visual's intent → character scene / licensed photo / demonstration / (budgeted) AI video / minimal overlay, with the fallback recorded. Re-enactments are labelled "Illustration".

### C. Voices (`cells/cast/narration.py`, `cells/narrator/narration.py`)
9. **Performance director** (`lib/voice_direction.py`):
   - Each line gets a conversational intention from its context.
   - Response gaps of 0.1–0.95 s replace narrator pauses.
   - Smoothing runs per speaker.
10. **Backends:**
    - English: Kokoro (default) or Qwen3-TTS.
    - Telugu: Indic Parler (local, gated: `HF_TOKEN`), or **Gemini free-tier TTS**, the one hosted exception, speech only, opt-in per plan (`tts_backend: gemini_free`).
11. **Gemini path** (`lib/gemini_tts.py`, `lib/turn_align.py`):
    1. Fail-closed preflight (manual free-tier confirmation, key, SDK ≥ 2.25, approved model, local budget).
    2. Two-speaker chunk requests; WAV kept as-is.
    3. Split per turn at real pauses, using recogniser word times.
    4. Per-turn letter check, then speaker attribution.
    5. Anything uncertain → local Parler for the whole video.
12. **Checks per line:**
    - speech check (Whisper: English small.en, Telugu large-v3-turbo, word/letter error);
    - signal checks; identity check by clip length;
    - bounded hero takes on sampling backends (A/B kept).
13. **Robustness:**
    - Hugging Face models on Drive are kept as plain files (no symlinks), validated and re-downloaded if broken.
    - CUDA libraries are preloaded for faster-whisper on the GPU.
    - A failed Parler take is retried with a new seed.

### D. Timing, pictures, video
14. **Voice-first timeline** (`cells/cast/timeline.py`): measured line lengths and gaps place every line. Each scene's visuals are cut at points inside lines (independent visual timeline). Subtitles are timed word by word.
15. **Drawn scenes** (`lib/charanim.py`, `lib/demos.py`, `lib/drawkit.py`):
    - 2D friends on the CPU, cached sprites; breathing, blinks, gestures; the mouth follows that speaker's own loudness.
    - Demonstrations: finger test, eye views, projector + glasses.
    - Licensed photos with on-screen credit.
16. **AI video (optional, GPU):** Wan 2.1 / VACE scene-state shots with OWLv2 and VLM checks and repair rounds. One heavy model family on the GPU at a time. Budgets: `MAX_AI_VIDEO_SHOTS`, `MAX_GPU_MINUTES`.
17. **Editing and render:**
    - Per-shot cached segments.
    - Licensed music with sidechain ducking; synthesised sound effects.
    - Loudness to −14 LUFS; subtitles burned in (Telugu drawn with Pillow for correct shaping).

### E. Safety, licences, resumability
18. **Licences** (`lib/registry.py`):
    - Every model has a verified licence and a pinned revision. `LICENSE_STRICT` refuses "review" models.
    - Media: PD / CC0 / CC BY (BY-SA opt-in).
    - `HOSTED_EXCEPTIONS`: Gemini TTS only, free tier only. No paid services anywhere.
19. **Preflight** (`cells/shared/helpers.py`): GPU, VRAM, precision (bf16 only if native), RAM, disk, token presence → `render/preflight.json`.
20. **Resume:**
    - Every stage, line, take, drawn scene, clip and segment is cached on Drive with content keys.
    - **Run all** after a disconnect continues where it stopped.
    - Gemini audio is cached by transcript hash; usage goes to the ledger `_gemini_usage.json`.

### F. Quality gate and outputs
21. **Final QC** (`cells/shared/qc.py`, `lib/qclib.py`):
    - Measured checks: stream, duration, black/frozen frames, loudness, peaks, silence, ducking.
    - Pipeline checks: speech, review items, validation.
    - Story checks: dialogue order, sourced facts, graphics share, labels, completeness, Telugu fact preservation.
    - Gemini checks.
    - Approved only if every required check is PASS.
22. **Outputs** in `MyDrive/ai-video/<project>/render/`:
    - `final.mp4`, `subtitles.srt/.ass`, `script.txt`, `storyboard.md`, `dialogue_shot_timeline.md`
    - `story_stats.json`, `qc_report.json`, `preflight.json`
    - voice reports (`voice_direction.json`, `voice_qc_report.json`, `gemini_report.json`), `metadata.json`
    - The YouTube upload is manual: Private, "Altered or synthetic content: Yes".

### G. Notebooks in use for the 3D glasses topic
23. **Notebooks:**
    - `colab_three_d_friends_v2.ipynb`: English, the one-eye story, approved on CPU.
    - `colab_three_d_friends_te_v2.ipynb`: Telugu, natively written, Gemini optional, Parler fallback.
    - The v1 notebooks are kept unchanged for comparison.

Details per area: `docs/11` (voices), `docs/12` (two friends), `docs/13` (natural voices + stories), `docs/14` (Telugu + Gemini).

## 18. Architecture validation (appended 2026-10-19)

Full report: `docs/15_ARCHITECTURE_VALIDATION.md`.

**How it was validated:** static analysis of every notebook as one program, a check that every setting is enforced, grouped run-log errors, end-to-end runs of each path, and the full suite.

**Fixed:**
1. Gemini lines mislabelled as Parler caused unplanned API retakes and disabled the identity check. Now only local sampling backends get retakes, and there is per-line error handling.
2. Stories with no sourced facts were accepted. New rule D8: ≥ 2 sourced factual turns.
3. `STORY_BENCH` was documented but missing. Now implemented and run.
4. `MAX_VIDEO_DURATION_SECONDS` was not enforced. Now a required QC check for short-form two-friend plans.
5. The topic notebook's staged writer is now tested end to end.

**Result:** 402 checks PASS. No other video's notebook was rebuilt.

## 19. Telugu–English (Tenglish) 3D glasses video, v3 (appended 2026-10-20)

Full guide: `docs/16_TENGLISH_STYLE_GUIDE.md`. There is no new architecture; these are additive switches, all inactive unless a plan sets them.

- **New files:**
  - Story: `colab/stories/three_d_friends_te_v3/story.json`, plus `candidates.json` and `grounding.json` (NLI results).
  - Plan and notebook: `colab/three_d_friends_te_v3_plan.json` → `colab/colab_three_d_friends_te_v3.ipynb` (project `three_d_friends_te_v3`).
- **Plan keys (new, optional):**
  - `code_mix: "natural"` turns on the Tenglish critic, mixed demo labels and the bilingual speech check.
  - `strict_open_source: true` turns on the notebook setting `STRICT_OPEN_SOURCE`.
  - `voice_policy` sets `asr_retake_at`, `gemini_asr_repairs` and `speech_compare`.
  - `pronunciation` records the display vs TTS forms and their status.
- **Voice routing:**
  - With `STRICT_OPEN_SOURCE` on: Telugu is always local Indic Parler-TTS. Gemini runs only if the user sets `TELUGU_TTS_BACKEND = "gemini_free"`; `auto` and plan hints are ignored.
  - With it off (all earlier plans), routing is exactly as before.
- **Before local synthesis,** `parler_gate()` checks token validity and gated-repo access. It runs only when some line still has to be made, and stops with the fix steps; there is no bypass and no voice switch. The preflight also reports Parler access (`registry.gated_access`, a HEAD request).
- **Display text vs TTS text:**
  - Plan lines keep `text` (subtitles) and get `tts` (the Telugu-script respelling for Parler).
  - Gemini reads `GEMINI_TRANSCRIPT_FORM` (`display` | `telugu_script`).
  - `TELUGU_PRON_BENCH` makes a blind listening sheet for these forms.
- **Speech check for code-mixed lines:**
  - `telugu_dialogue.bilingual_error` is used in `speech_error` and in Gemini `run_voices(err_fn=...)`.
  - Doubtful lines are flagged (QC fails until someone listens). A Parler retake happens only above 0.6; Gemini never repairs on an ASR doubt (`asr_repairs=False`).
- **Visuals:**
  - `duck` gesture: arms up and the body drops.
  - Cinema screen `clean+popcorn` / `plain+popcorn`.
  - `te_mix` demo labels.
- **Sound:** scene-level `sfx` in story JSON (new `rustle` type, plus whoosh and pop), passed to the shot.
- **No new model:** no AI video, no new TTS. The NLI fact check is the existing `cross-encoder/nli-deberta-v3-base`.

## 20. Storybook characters and cinematic environments (appended 2026-10-20)

Full design: `docs/17_STORYBOOK_CHARACTERS.md`. The change is additive: plans without `visual_style_version` render exactly as before.

- **New modules (engine/lib):**
  - `storybook_art.py`: provisional layered artwork.
  - `storybook.py`: asset library, rig, poses, expressions, visemes.
  - `storybook_scene.py`: the renderer for drawkit kind "character" when `visual_style_version == "storybook_v1"`.
  - `envfactory.py`: environment packs, camera presets, parallax, lighting.
  - `studio.py`: Character Studio helpers.
- **Asset library:**
  - Location: `colab/assets/{characters,environments}/<id>/<version>/`, immutable and sha256-verified.
  - Plans pin versions and manifest hashes (`character_assets`, `environments`).
  - `build.py` embeds exactly those versions in the notebook, about 3–5 MB.
  - The diagrams cell installs them on Drive (`_assets`) without overwriting anything, and refuses missing or changed assets. It never falls back to simple shapes.
- **Rendering:**
  - The fixed compositor: Pillow on the CPU, drawn at the output size.
  - The static background plate is cached per shot. The camera move is one crop, and the foreground has its own parallax.
  - Mouth shapes come from each speaker's own audio (`speech_visemes`).
  - The drawn-shot cache key adds the asset versions only for storybook plans.
- **QC (storybook plans only):**
  - assets present and unchanged;
  - "character artwork approved by a person" (UNKNOWN while provisional, so the video is not approved);
  - lip sync: only the speaker's mouth moves.
- **One-time production:**
  - `colab_character_studio.ipynb`: FLUX.2 klein 4B concepts and edits, BiRefNet cut-outs, a semi-manual layer split, import as a new version.
  - `approve_character.py` records the approval.
- **Licences:** registry entries for the Studio models; FLUX.2 klein 9B and RMBG-2.0/1.4 are refused.

## 21. Cinematic character artwork: pose atlas + Character Studio (appended 2026-10-20)

- **New character asset type:** `atlas_v1` (`storybook_atlas.py`), used behind the same storybook interface. The plan schema is unchanged; plans pin `manifest_sha256` + `content_sha256`.
- **One-time production:** `colab_character_studio.ipynb` (`studio_run.py`). The order is proof of concept → human approval gate → 35 images per character → cut-outs → import → previews → approval.
- **Normal renders never import or call the image model;** this is tested.
- **Details:** `docs/17_STORYBOOK_CHARACTERS.md` §11 and `reports/character_poc/REPORT.md`.

> **⚠ SUPERSEDED (2026-10-11).** This document is kept for its reasoning and cost comparisons. Do not use it as the current design.
> The current design is in `docs/` (start with `docs/AI_VIDEO_AUTOMATION_MASTER_PLAN.md`). Main changes:
> 1. **No per-video LLM-written code.** This revision's core idea (Claude writes `produce_<project>.py` against `vidsdk`, plus a sandbox, AST policy, dry-run and repair loop) is dropped. Documentaries share one structure, so Claude writes only the JSON plan. A fixed, tested executor renders it, using a small, growing **recipe menu** with a non-blocking fallback for unsupported effects (`docs/02_ARCHITECTURE.md` §9).
> 2. **No longer "every shot is AI video".** The mix is AI I2V hero shots, AI keyframes with camera moves, and code-drawn graphics. This is cheaper, more reliable, and accurate for maps, dates and numbers.
> 3. **Hardware reality.** The control plane is the local laptop (no NVIDIA GPU). GPU jobs run on Colab (prototype) or rented GPUs (production), rather than Modal sandboxes and a GPU broker.
> 4. **Facts updated since 2026-10-07:**
>    - Qwen-Image-2.1 is non-commercial. Wan 2.2 TI2V-5B is 24 fps / 121 frames.
>    - The YouTube docs now conflict on the "unaudited uploads forced private" rule.
>    - MiniMax H3, Kandinsky 6 and LTX-2.5 have been released.
>
> See `docs/01_RESEARCH_REPORT.md` for current facts.

# Automated Full-AI Video Creation and YouTube Publishing: Architecture Research

- **Revision 2, 2026-10-07.** Rewritten for two requirements:
  1. **Every shot is AI-generated video.** No image slideshows, no Ken Burns, no stock footage.
  2. **The LLM writes new code for every video.**
- **Revision 1, 2026-10-06.** The hybrid stills-plus-video design. It is kept here only as a cost comparison.
- This answers the brief in `~/Documents/Dexur_data_insights/automation.md`.

**Labels used throughout**
- **[O]** comes from official docs, a repo, a model card or a license file opened during research.
- **[C]** comes from community reports or third-party pages.
- **[est]** is my own engineering estimate.
- **[verify]** could not be confirmed and must be checked before you rely on it.

Nothing here is legal advice. Have every model license reviewed before you monetize.

---

## A. Feasibility

**Yes, a fully AI-generated video pipeline can be built today with open-source models and Python.** Three things make it work:

1. **Short clips, many of them.** Open video models produce good clips of about **5 s** (Wan 2.2: 81 frames at 16 fps, up to 720p). A 5-minute video is therefore **about 60 generated clips**, stitched on the narration timeline. Longer shots come from chaining clips (last frame to next clip) or from continuation models.
2. **Generated, but against a fixed SDK.** The LLM writes a **new Python production program for every video**. That program calls a **vetted, versioned SDK** (`vidsdk`) of generation primitives. The code is linted, policy-checked, cost-estimated and run in a sandbox, then repaired automatically if it fails. This keeps the flexibility of per-video code without letting the LLM run arbitrary commands on machines that hold credentials.
3. **Consistency from reference frames.** Recurring characters and locations get a reference frame. It never appears in the video; it is only the first-frame condition for image-to-video. Shots without recurring subjects use plain text-to-video.

| Component | Maturity | Notes |
|---|---|---|
| LLM as writer, director and cinematographer → strict JSON | **Mature** | Anthropic structured outputs (JSON schema / Pydantic) [O] |
| LLM writes per-video Python against a known SDK | **Usable** | Reliable when the API surface is small and documented, with a lint, test and repair loop. Not reliable against raw model repos |
| Text-to-video, 5 s clips, 720p | **Usable** | Wan 2.2 A14B (Apache) is the best commercial-safe option. Expect 10–30% of clips to need a regeneration [est] |
| Image-to-video from a reference frame | **Usable** | Best way to keep identity across clips |
| Shots longer than ~5–10 s | **Unreliable** | Chaining or continuation drifts after 2–3 hops |
| Character consistency across 60 clips | **Partly solved** | LoRA + reference-frame I2V + automated drift rejection |
| Narration TTS, music, SFX | **Mature** | Qwen3-TTS / Chatterbox, ACE-Step 1.5, Stable Audio Open |
| FFmpeg assembly, QC, YouTube API upload | **Mature** | Uploads stay private until the API audit passes [O] |
| **Monetization policy** | **Main risk** | YouTube explicitly names videos that "stitch together unrelated or inconsistent AI clips" as not monetizable [O]. Full-AI video must be narrative-led, consistent and researched |

**Cost:** for a 5-minute full-AI video, roughly **$12–20 of GPU time** at standard quality, or **$100–150** at maximum quality. It takes **about 1 hour** of wall clock on 8 parallel GPUs [est]. See section 23.

---

## B. Recommended architecture

### B.1 End-to-end architecture

```text
 Topic
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR  (plain Python → DBOS → Temporal)   state: Postgres  files: R2/S3 │
└──────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  PHASE 1 — PLAN (Claude, structured outputs)                       CPU
   Brief → Research → Script → Director/Storyboard → Validator → Critic
   ──► VideoPlan JSON  (every shot = a video shot with camera, light, motion)
   │
   ▼  PHASE 2 — VOICE FIRST (sets the timeline)                         GPU (small)
   Qwen3-TTS per sentence → ASR check → WhisperX alignment → exact shot durations
   │
   ▼  PHASE 3 — CODE GENERATION (Claude writes a new program for this video)  CPU
   VideoPlan + SDK reference + model registry + budget
     ──► produce_<project>.py
     ──► gates: ruff/mypy → AST policy → dry-run on mock GPUs (cost + shape check)
     ──► self-repair loop (≤3) on any failure
   │
   ▼  PHASE 4 — EXECUTE IN SANDBOX (generated program runs)             CPU sandbox
   program calls vidsdk ──► GPU broker ──► RunPod Serverless / Modal GPU workers
     ├─ reference frames for characters/locations (Qwen-Image, internal only)
     ├─ video clips: Wan 2.2 T2V / I2V / FLF2V, LTX-2.5, continuation
     ├─ per-clip QC (VLM + quality + drift) → regenerate / reseed / reprompt
     ├─ RIFE 16→32 fps, SeedVR2 upscale 720p→1080p
     ├─ music: ACE-Step 1.5 · SFX/ambience: Stable Audio Open
     └─ edit: FFmpeg graph (cuts, transitions, grade, subtitles, mix, loudnorm)
   │
   ▼  PHASE 5 — QC of final video (black/freeze/silence, loudness, WER, VLM)  CPU+GPU
   │
   ▼  PHASE 6 — PACKAGE (Claude metadata; thumbnail from best frame + text)  CPU
   │      [human approval gate while trust is built]
   ▼  PHASE 7 — PUBLISH (YouTube Data API, resumable, idempotent, synthetic flag)
   ▼
 Published / scheduled video  +  costs, attempts, code, logs stored per project
```

### B.2 Key design decisions

1. **Every visible shot is generated video.** The final video contains no still images. Images exist only as internal conditioning frames.
2. **The LLM writes new code for each video, but against an SDK, not against raw models.** The SDK holds everything that must not vary: model loading, GPU calls, retries, caching, storage, credentials, the FFmpeg primitives and the QC primitives. The generated program holds everything that should vary per video: the shot-by-shot generation strategy, model choice per shot, prompt variants, seeds, chaining, edit rhythm, transitions, grading, sound design and the mix (section 8).
3. **The generated code runs in a sandbox with no secrets.** GPU work goes through a **broker** service that holds the API keys, enforces the budget and only accepts known SDK operations.
4. **Voice comes first.** Narration timing decides each shot's exact length before any video is generated, so clips are generated at the right duration rather than cut to fit.
5. **Everything is idempotent and content-addressed.** Each clip is keyed by (model hash, params, seed, conditioning inputs). Re-running a repaired program reuses every clip that already succeeded.
6. **A license allow-list lives in the SDK.** The generated code cannot call a non-commercial model, because the SDK refuses it.

---

## C. Recommended stack (full-AI video)

| Layer | Choice | Why / alternative |
|---|---|---|
| LLM: planner, director, code writer, repair, metadata | **Claude**: `claude-opus-5-5` for brief, research, critic and code generation; `claude-sonnet-5-5` for storyboard sections and repairs | Structured outputs for plans, and strong code generation. Open-source fallback: Qwen3-Coder / Qwen3 with vLLM |
| Agent framework | **None.** Plain SDK calls + Pydantic + the codegen loop below | PydanticAI later if research becomes a tool loop |
| Orchestrator | Plain Python → **DBOS** (MVP) → **Temporal** (production) | Durable steps, retries, resume |
| Code sandbox | **Modal Sandbox** or **E2B**, or a gVisor/Firecracker container: no network except the broker, no secrets, CPU-only, time-limited | The generated program only orchestrates; GPUs run elsewhere |
| **Main video model** | **Wan 2.2 T2V-A14B / I2V-A14B** (Apache-2.0) + LightX2V 4-step LoRA | Best open quality with no license restrictions [O] |
| Fast / draft video | **Wan 2.2 TI2V-5B** (runs on a 24 GB card) or **LTX-2.5 distilled** (free under $10M ARR) [O] | LTX-2.5 also generates native audio and multi-shot clips |
| Shot transitions / continuity | **Wan 2.1 FLF2V** (first + last frame), **Wan VACE / Phantom** (reference-to-video) | Apache-2.0 |
| Long continuous shots | **LongCat-Video** continuation (MIT), or last-frame → I2V chaining (≤ 2–3 hops) | SkyReels V2 diffusion forcing [verify license] |
| Talking characters (optional) | **InfiniteTalk** / Wan2.2-S2V (Apache) | LatentSync 1.6 for redubbing |
| Internal reference frames | **Qwen-Image-2512 / Edit-2511** (Apache), **Z-Image-Turbo** (Apache) | Used only to condition I2V. Never shown |
| Character identity | **Character LoRA** (ai-toolkit) for Wan + Qwen-Image | Avoid InsightFace-based tools (non-commercial) |
| Frame rate / resolution | **RIFE** (MIT) 16→32 fps, **SeedVR2** (Apache) 720p→1080p | Real-ESRGAN for speed |
| Narration | **Qwen3-TTS 1.7B** (Apache), **Chatterbox** (MIT) | Kokoro as the CPU fallback |
| Alignment / subtitles | **WhisperX** forced alignment → SRT/ASS | faster-whisper for the word-error check |
| Music | **ACE-Step 1.5** (MIT) | Stable Audio Open for stingers |
| SFX / ambience | **Stable Audio Open**, Freesound CC0, or LTX-2.5's native audio | |
| Editor / renderer | **FFmpeg** via **typed-ffmpeg** | |
| QC | ffprobe, FFmpeg detect filters, PySceneDetect, imagehash, **DOVER**, **Qwen3-VL** / Claude vision | |
| State / storage | **Postgres** + **Cloudflare R2** (S3-compatible) | SQLite + local disk for the prototype |
| GPUs | **RunPod Serverless** or **Modal** (H100 / L40S / 4090) | Lambda or RunPod reserved at steady volume. Colab for experiments only |
| YouTube | google-api-python-client, OAuth refresh token, resumable upload | |

---

## D. The flow: what is used at every step

### D.1 Flow diagram

```text
[1] Topic ─► [2] Brief ─► [3] Research ─► [4] Script ─► [5] Storyboard ─► [6] Validate+Critic
                                                                              │
                                                                     VideoPlan JSON
                                                                              │
[7] Narration TTS ─► [8] Align words ─► [9] Lock shot durations ◄─────────────┘
                                              │
                                   [10] Generate program (Claude)
                                              │
                         [11] Static gates ─► [12] Dry-run (mock GPUs) ─► fail? ─► [13] Repair ─┐
                                              │ pass                                           │
                                              ◄────────────────────────────────────────────────┘
                                   [14] Run program in sandbox
                                              │
            ┌───────────────┬─────────────────┼─────────────────┬────────────────┐
     [15] Ref frames   [16] Video clips   [17] Clip QC      [18] Music      [19] SFX
      (internal)       (parallel GPUs)    ↺ regen/reseed     ACE-Step       Stable Audio
            └───────────────┴──────┬──────────┴─────────────────┴────────────────┘
                          [20] Interpolate + upscale
                                   │
                          [21] Edit + mix + subtitles (FFmpeg)
                                   │
                          [22] Final QC ─► fail? ─► re-run failed shots only (back to 14)
                                   │ pass
                 [23] Metadata ─► [24] Thumbnail ─► [25] Approval ─► [26] Upload ─► [27] Record
```

### D.2 Step-by-step: what is used, where it runs, what goes in and out

| # | Step | What is used | Runs on | Input → Output | Time per 5-min video [est] |
|---|---|---|---|---|---|
| 1 | Accept topic | CLI / API (`avs make "topic"`), orchestrator starts the workflow (DBOS/Temporal) | CPU | topic string → `project_id`, workflow | instant |
| 2 | Brief | **Claude Opus**, structured output → `Brief` (Pydantic) | Anthropic API | topic → audience, format, angle, duration, style profile, tone, hook | ~20 s |
| 3 | Research (factual topics) | **Claude** + web search tool → `FactSheet` with source URLs | Anthropic API | brief → facts + citations | 1–3 min |
| 4 | Script | **Claude Opus** → `Script` (beats, narration, pacing) | Anthropic API | brief + facts → ~700 words / 5 min | ~1 min |
| 5 | Storyboard / direction | **Claude Sonnet**, one call per section, in parallel. Uses the **style profile YAML** plus few-shot examples | Anthropic API | script + profile → shots with camera, lens, lighting, motion, characters, transitions, audio cues | ~1 min |
| 6 | Validate + critique | **Pydantic** validator (rules in code) + **Claude Opus** critic, one revise pass | CPU + API | storyboard → **VideoPlan JSON v1** | ~1 min |
| 7 | Narration | **Qwen3-TTS 1.7B** (or Chatterbox), one call per sentence, with a style instruction from the profile | GPU (L4/4090), via vidsdk | narration text → `narration/{i}.wav` | ~2–4 min |
| 8 | Check + align | **faster-whisper** WER check per sentence (regenerate if > 5%), then **WhisperX** forced alignment | GPU/CPU | wavs + text → `words.json` (word timestamps) | ~1 min |
| 9 | Lock durations | Plain Python: shot duration = its narration span (+ pauses), split at > 8 s | CPU | VideoPlan + words → **VideoPlan v2** (timed) | instant |
| 10 | **Generate program** | **Claude Opus** with system prompt + `vidsdk` API stubs + model registry + budget + VideoPlan v2 | Anthropic API | → `produce_<project>.py` (new code for this video) | ~1–2 min |
| 11 | Static gates | **ruff**, **mypy**, **AST policy** (import allow-list, no `os`/`subprocess`/`socket`/`eval`/file I/O outside workdir), size limits | CPU | program → pass / list of errors | seconds |
| 12 | Dry run | Program runs against **MockBackend** (fake GPUs return placeholders instantly) → estimated GPU-minutes, shot coverage, timeline length | Sandbox (CPU) | program → cost estimate, coverage report | seconds |
| 13 | Repair (if needed) | **Claude Sonnet** gets the errors or traceback plus the program → returns a corrected program. Up to 3 loops, then human review | Anthropic API | failing program → fixed program | ~1 min per loop |
| 14 | Execute | Program runs in **Modal Sandbox / E2B** with no secrets. Every GPU call goes through the **GPU broker** (holds keys, enforces budget and license allow-list) | Sandbox (CPU) | program → job graph on GPUs | (wraps steps 15–21) |
| 15 | Reference frames (internal) | **Qwen-Image-2512** / **Z-Image-Turbo** + character **LoRA**; **Qwen-Image-Edit-2511** for variants | GPU (L40S/H100) | character/location bible → `refs/*.png` (never shown in the video) | ~2–5 min |
| 16 | **Video clips** | **Wan 2.2 T2V-A14B** (no recurring subject) · **Wan 2.2 I2V-A14B** from a ref frame (recurring subject) · **FLF2V** (designed transitions) · **last-frame chaining** / **LongCat** (shots > 5 s) · **LTX-2.5** (fast drafts or native-audio shots) · **InfiniteTalk** (speaking character). All with LightX2V 4-step LoRA | GPU workers ×8 (H100/L40S) via **RunPod Serverless / Modal** | shot specs → `shots/{id}/clip_raw.mp4` (720p, 16 fps, ~5 s) | ~60 clips × 4–6 GPU-min ≈ 40–60 min wall clock on 8 GPUs |
| 17 | Clip QC (per clip) | **ffprobe** (shape), **DOVER** (quality), **Qwen3-VL** (matches shot? artifacts? identity?), face/subject similarity to the ref, **imagehash** (duplicates) | GPU (small) + CPU | clip → pass / regen with new seed / reprompt / switch model | ~10–20 s per clip |
| 18 | Music | **ACE-Step 1.5**, one bed per section from the profile's music prompt (or from the curated pool) | GPU | section prompts → `music/{section}.wav` | ~1 min |
| 19 | SFX / ambience | **Stable Audio Open** per audio cue; Freesound CC0 for real sounds | GPU / API | cues → `sfx/{cue}.wav` | ~2 min |
| 20 | Interpolate + upscale | **RIFE** 16→32 fps (or 48), **SeedVR2** 720p→1080p | GPU | `clip_raw.mp4` → `clip.mp4` (1080p) | ~1–2 GPU-min per clip |
| 21 | Edit, mix, subtitles | **FFmpeg** (typed-ffmpeg graph written by the generated program): trim to the narration span, `xfade` transitions, color grade (LUT or `eq`/`curves`), speed ramps, ASS subtitles, `sidechaincompress` ducking, two-pass `loudnorm` −14 LUFS, H.264 High + AAC 48 kHz + faststart | CPU | clips + audio → `final.mp4` | ~3–5 min |
| 22 | Final QC | `blackdetect`, `freezedetect`, `silencedetect`, `ebur128`, **PySceneDetect** cut count, pHash duplicates, **faster-whisper** WER vs script, subtitle rules, **Qwen3-VL / Claude** over a contact sheet (1 frame per 5 s) | CPU + GPU | final.mp4 → `qc_report.json`: pass / list of shots to redo | ~3 min |
| 23 | Metadata | **Claude Sonnet** → title (≤ 100 chars), description (≤ 5000 bytes, chapters from the timeline, sources, AI disclosure line), tags (≤ 500 chars), category, language, playlist | Anthropic API | plan + transcript → `metadata.json` | ~20 s |
| 24 | Thumbnail | Best frame chosen from the generated clips by **Qwen3-VL** scoring (or a dedicated **Qwen-Image** render), **Pillow** text overlay, OCR + 160×90 readability check | CPU/GPU | clips → `thumbnail.png` (1280×720 or larger) | ~1 min |
| 25 | Approval | Human gate (web page or CLI) in early phases; automatic later | — | video + metadata → approve / reject | — |
| 26 | Upload | **YouTube Data API v3**: resumable `videos.insert` (`containsSyntheticMedia=true`, `publishAt`), `thumbnails.set`, `captions.insert` (SRT), `playlistItems.insert`. Idempotency key + stored session URI | CPU | files → `youtube_video_id` | ~2–5 min |
| 27 | Record | Postgres: costs, attempts, seeds, model hashes, **the generated program and every repair**, QC reports, video ID | CPU | — | instant |

**Total for a 5-minute video** [est]: about **1–1.5 h wall clock** with 8 GPU workers, about **8–10 GPU-hours**, and **~$12–20** in GPU cost plus about $1–3 in LLM cost.

---

## 2–3, 7. AI Director: VideoPlan schema, style system, single vs multi-agent

### 2.1 One LLM call or several?

Use **five or six short, typed stages** (steps 2–6 in the flow), not one large call and not a free-form agent swarm. Each stage has one job and one schema, and is checkpointed and retryable on its own. A five-minute video has about 60 shots, which is too much structure for one call to do well. Autonomous agent chat is non-deterministic and hard to resume.

| Stage | Input → Output | Model |
|---|---|---|
| Brief | topic → `Brief` | Claude Opus |
| Research | Brief → `FactSheet` (claims + URLs) | Claude Opus + web search |
| Script | Brief + FactSheet → `Script` | Claude Opus |
| Director / storyboard | Script + StyleProfile → shots | Claude Sonnet, one call per section, in parallel |
| Critic | plan vs brief → issues → one revise | Claude Opus |
| **Programmer** | timed VideoPlan + SDK reference → `produce_<project>.py` | Claude Opus (section 8) |

### 2.2 VideoPlan schema (Pydantic, abridged)

Every shot is a **video shot**. The director describes **motion**: subject action, camera move and pace. `video_mode` is a suggestion, and the generated program may override it.

```python
from pydantic import BaseModel, Field
from typing import Literal

ShotType   = Literal["extreme_wide","wide","medium","close_up","extreme_close_up","over_shoulder","pov","aerial","macro","insert"]
CameraMove = Literal["static","push_in","pull_out","pan_left","pan_right","tilt_up","tilt_down","orbit","dolly","crane_up","handheld","tracking","fpv"]
VideoMode  = Literal["t2v","i2v_ref","flf2v","extend","r2v","talking"]   # all produce video

class Character(BaseModel):
    id: str; name: str; description: str           # locked canonical description
    lora: str | None = None; ref_frame: str | None = None; voice_id: str | None = None

class Location(BaseModel):
    id: str; description: str; ref_frame: str | None = None

class AudioCue(BaseModel):
    kind: Literal["sfx","ambience","music_stinger"]; prompt: str; at: float = 0.0; gain_db: float = -12

class Shot(BaseModel):
    shot_id: str; beat_id: str
    narration: str                                  # words spoken over this shot
    duration_s: float = Field(ge=1.5, le=12)        # locked from TTS timing (step 9)
    intent: str                                     # what the viewer must understand/feel
    importance: Literal["hero","standard","filler"]
    shot_type: ShotType; camera_move: CameraMove; camera_speed: Literal["slow","medium","fast"]
    lens: str                                       # "24mm wide", "85mm shallow DOF"
    lighting: str; environment: str; time_of_day: str
    subject_action: str                             # what moves in frame
    characters: list[str] = []; location: str | None = None
    visual_prompt: str; negative_prompt: str = ""
    video_mode: VideoMode = "t2v"
    continuity_from: str | None = None              # shot whose last frame seeds this one
    transition_in: Literal["cut","crossfade","dip_to_black","whip","match_cut","fade_from_black"] = "cut"
    audio_cues: list[AudioCue] = []

class Section(BaseModel):
    section_id: str; title: str; emotion: str
    music_prompt: str; music_intensity: float = Field(ge=0, le=1)
    grade: str                                      # "cold teal shadows, crushed blacks"
    shots: list[Shot]

class VideoPlan(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    project_id: str; topic: str; title_working: str; audience: str; language: str = "en"
    format: Literal["long_16x9","short_9x16"]; target_duration_s: int
    style_profile_id: str; tone: list[str]
    narrator: dict                                  # {"tts":"qwen3-tts","voice":"...","style_instruction":"..."}
    hook: str
    characters: list[Character] = []; locations: list[Location] = []
    sections: list[Section]
    sources: list[str] = []
    synthetic_realistic: bool                       # → containsSyntheticMedia
    gpu_budget_minutes: int                         # hard cap passed to the GPU broker
```

**Validator rules (in code):**
- Durations sum to the target, within ±10%.
- All character and location IDs resolve.
- Every shot longer than 5 s has `extend` or a `continuity_from` plan.
- Recurring characters use `i2v_ref` or `r2v`, never pure `t2v`.
- No two adjacent shots have identical `shot_type` + `camera_move`, which adds visual variety.
- Narration runs at 2.0–3.2 words per second.
- Prompt token limits are respected.

### 2.3 Style and tone system

**Style profiles are versioned YAML**, chosen by the Brief stage, plus 2–3 few-shot storyboard examples per profile.
- Profiles are built from independent axes: genre, look, tone and pacing.
- No RAG is needed for style.
- No per-style agents are needed.

For full-AI video, the profile also carries **motion and model preferences**:

```yaml
id: cosmic_documentary
genre: documentary
look:
  prompt_suffix: "cinematic photoreal, anamorphic 35mm, volumetric light, deep blacks, subtle film grain"
  negative: "cartoon, text, watermark, extra limbs, warping, flicker, lowres"
  grade_lut: luts/teal_orange_soft.cube
cinematography:
  preferred_shots: [extreme_wide, aerial, macro, close_up]
  camera_moves: {default: slow push_in, tension: slow pull_out, reveal: crane_up, energy: fpv}
  avg_shot_len_s: 5.0
  lighting: "low-key, hard rim light, single source"
video:
  default_model: wan22_a14b            # T2V/I2V
  draft_model: wan22_ti2v_5b
  fps_out: 32                          # RIFE from 16
  resolution: 1080p                    # SeedVR2 from 720p
  motion_strength: medium
transitions: {default: cut, section_break: dip_to_black, time_jump: whip}
narration: {tts: qwen3-tts, style_instruction: "deep, calm documentary narrator; hushed awe; pauses at commas", wpm: 140}
music: {prompt: "slow cinematic ambient, low strings, sub drones, sparse piano", duck_db: -18}
sfx_density: medium
```

Ship about 12 profiles: cinematic, documentary, explainer, horror, sci-fi/cyberpunk, inspirational, humorous, anime-style, nature, historical, children's, and luxury/minimal. A **variation step** randomizes secondary parameters (lens, grade, music tempo, voice) for every video, so no two videos look templated.

---

## 4, 22. Open-source video model research and model selection

### 4.1 Video models (the core of a full-AI pipeline)

| Model | License / commercial use | Params | VRAM | Native output | Modes | Speed | Prod (1–5) |
|---|---|---|---|---|---|---|---|
| [Wan 2.2](https://github.com/Wan-Video/Wan2.2) | **Apache-2.0** ✔ | A14B MoE / TI2V-5B | 14B: 80 GB recommended, ≥24 GB with offload; 5B on a 4090 [O] | 480p / 720p, ~5 s (81 frames, 16 fps) | T2V, I2V, TI2V, S2V, Animate | 5B: 5 s 720p in under 9 min on one GPU [O]; A14B + LightX2V 4-step ~1–3 min for 480p on a 4090 [C] | **5** |
| [Wan 2.1](https://github.com/Wan-Video/Wan2.1) + [VACE](https://github.com/ali-vilab/VACE) | Apache-2.0 ✔ | 1.3B / 14B | 1.3B ≈ 8 GB | 480p / 720p, 81 frames | T2V, I2V, **FLF2V**, ref-to-video, pose/depth control | 1.3B ~4 min for 480p on a 4090 [O] | 4 |
| [LTX-2 / 2.5](https://github.com/Lightricks/LTX-2) | Free under **$10M ARR** ✔ (with conditions) | 19–22B | Distilled ~16 GB with FP8 + offload [O] | Up to 4K with upscaler, **native audio**, multi-shot | T2V, I2V, A+V | Fastest in its class (vendor) | 4 |
| [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | ✖ not valid in EU/UK/South Korea | 8.3B | 14 GB with offload | 480p / 720p, 5 s | T2V, I2V | ~75 s per clip on a 4090, distilled [O] | 4 (license risk) |
| [Kandinsky 5.0](https://github.com/kandinskylab/kandinsky-5) | **MIT** ✔ | 2B / 19B | 12 GB+ | 5–10 s | T2V, I2V | Pro: 560–1241 s on an H100 [O] | 3 |
| [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) | **MIT** ✔ | 13.6B | [verify] | 720p 30 fps, **long continuation** | T2V, I2V, continuation | [verify] | 3 |
| [MAGI-1](https://github.com/SandAI-org/MAGI-1) | Apache-2.0 ✔ | 4.5B / 24B | 4.5B on a 4090 | Autoregressive streaming | T2V, I2V, V2V | — | 2–3 |
| [FramePack](https://github.com/lllyasviel/FramePack) | Hunyuan license ✖ | 13B | 6 GB | Up to 60 s | I2V | 1.5–2.5 s per frame on a 4090 [O] | 3 |
| [SkyReels V2/V3](https://github.com/SkyworkAI/SkyReels-V2) | Skywork [verify] | 1.3–19B | 15–51 GB | Unlimited length (diffusion forcing) | T2V, I2V, extension | — | 3 |
| CogVideoX, Mochi 1, Open-Sora 2, Step-Video, SVD, AnimateDiff | Mixed | — | — | — | — | — | 1–2: superseded or stale |

Wan 2.5–2.7 are **API-only**. Open Wan 3.0 weights are unconfirmed [verify].

### 4.2 Image models: internal reference frames and thumbnails only

| Model | License | VRAM | Role in full-AI pipeline |
|---|---|---|---|
| [Qwen-Image-2512 / Edit-2511](https://github.com/QwenLM/Qwen-Image) | Apache-2.0 ✔ | ~24 GB in FP8 | Character and location reference frames for I2V. Optional thumbnail render (best text) |
| [Z-Image-Turbo](https://github.com/Tongyi-MAI/Z-Image) | Apache-2.0 ✔ | 16 GB | Fast first frames for I2V shots |
| [FLUX.2 klein 4B](https://github.com/black-forest-labs/flux2) | Apache-2.0 ✔ | ~8 GB | Multi-reference alternative |
| FLUX.1/2 dev, Kontext dev | ✖ non-commercial without a BFL license | — | Avoid |

### 4.3 Supporting models

- **Consistency:**
  - Character LoRA trained with [ai-toolkit](https://github.com/ostris/ai-toolkit) (MIT) for Wan and Qwen-Image.
  - [Phantom-Wan](https://github.com/Phantom-video/Phantom) and [VACE](https://github.com/ali-vilab/VACE) for reference-to-video (Apache).
  - Wan FLF2V for transitions.
  - ✖ PuLID, InstantID and IP-Adapter-FaceID: they depend on InsightFace weights, which are non-commercial.
- **Talking characters:**
  - [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) (Apache).
  - Wan2.2-S2V (Apache).
  - [LatentSync 1.6](https://github.com/bytedance/LatentSync) (Apache).
  - [MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk) (MIT).
- **Post-processing:**
  - [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) (MIT).
  - [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) (Apache).
  - [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD).
- **Speed:**
  - [LightX2V](https://github.com/ModelTC/LightX2V): 4-step distilled LoRAs; 14B models in 8 GB VRAM with offload [O]. Distillation slightly reduces motion [C].
  - TeaCache and MagCache: about 1.5–2× faster.
  - FP8 and GGUF quantization.

### 4.4 Model selection: the program chooses, the SDK enforces

The generated program decides which model to use for each shot. The SDK and GPU broker enforce four constraints:
- **License allow-list:** the SDK refuses any model not cleared for `commercial` use.
- **Budget:** the broker rejects a job once `gpu_budget_minutes` would be exceeded.
- **Capability check:** the SDK refuses requests beyond a model's limits, such as a 9 s clip from a 5 s model or I2V without a frame.
- **Fallback chain on repeated failure:** `wan22_a14b → ltx25 → wan22_ti2v_5b`.

Default selection logic, which the codegen prompt gives the LLM as guidance:

| Shot need | Mode | Model |
|---|---|---|
| No recurring subject (landscape, space, crowd, object) | `t2v` | Wan 2.2 T2V-A14B |
| Recurring character or location | `i2v_ref` | Ref frame (Qwen-Image + LoRA) → Wan 2.2 I2V-A14B |
| Designed transition between two compositions | `flf2v` | Wan 2.1 FLF2V |
| Continuous shot > 5 s | `extend` | Last frame → Wan I2V (≤ 2 hops), or LongCat continuation |
| Subject must match a reference closely while moving | `r2v` | Phantom-Wan / VACE |
| Character speaks on camera | `talking` | InfiniteTalk (from ref frame + narration audio) |
| Draft / preview pass | any | Wan 2.2 TI2V-5B or LTX-2.5 distilled at 480p |

---

## 5–6. Audio: TTS, music, SFX

### TTS

| Model | Weights license | Commercial | VRAM | Style control | Cloning | Verdict |
|---|---|---|---|---|---|---|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 0.6B / 1.7B | Apache-2.0 | ✔ | small | **Natural-language instructions** [O] | 3 s | **Primary** |
| [Chatterbox](https://github.com/resemble-ai/chatterbox) Turbo / Multilingual V3 / Nano | MIT | ✔ | 0.1–0.5B | Exaggeration/CFG, `[laugh]` tags | Yes | Strong alternative (watermark [verify]) |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Apache-2.0 | ✔ | CPU | None | No | Fast fallback |
| [CosyVoice 2/3](https://github.com/FunAudioLLM/CosyVoice) | Apache-2.0 | ✔ | 0.5B | Instructions | Yes | Multilingual |
| [IndexTTS2](https://github.com/index-tts/index-tts) | bilibili license | ✔ below 100M MAU / RMB 1B | ~8–12 GB | Emotion + duration control | Yes | Dubbing with timing |
| [Higgs Audio v2](https://github.com/boson-ai/higgs-audio) | Boson Community | ✔ below 100k annual active users (ambiguous) | 24 GB | Very expressive | Yes | Legal review first |
| Zonos, Dia, Orpheus, CSM | Apache | ✔ (Llama caveats) | 4–16 GB | Tags | Varies | Dialogue, not long narration |
| ✖ XTTS-v2 / F5-TTS / Fish Speech | NC licenses | **No** | | | | Avoid |
| Piper / Bark | GPL voices / abandoned | — | | | | Avoid |

Synthesize **one sentence per call** and run an ASR word-error check on each (regenerate if above 5%). All autoregressive TTS models drift on long passages.

### Music and SFX

| Need | Choice | License |
|---|---|---|
| Section score | **[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)**: under 10 s per song on a 3090 [O] | MIT ✔ |
| Stingers, ambience, SFX | **Stable Audio Open** (≤ 47 s) | Free under $1M revenue |
| Shot-synced ambience | **LTX-2.5 native audio** (when that model generates the shot) | LTX license |
| Real SFX | **Freesound API** filtered to CC0 | Check the API commercial terms [O] |
| ✖ Avoid | MusicGen, AudioLDM2, Tango2, MMAudio, ThinkSound | Non-commercial |

---

## 8. Code generation: the LLM writes a new program for every video

### 8.1 Why it is done this way

Letting the LLM write **raw** model code per video (cloning repos, loading weights, CUDA setup) fails often and is unsafe. Letting it write a **new orchestration program per video against a stable SDK** works well. You get per-video creativity in how the film is built, and reliability in how models run.

| Concern | How the design handles it |
|---|---|
| Safety | The program runs in a sandbox: no secrets, no network except the GPU broker, AST import allow-list, CPU-only, time limit |
| Reliability | The SDK is tested and versioned. The program only composes SDK calls. Static gates + dry run + automatic repair before any GPU time is spent |
| Cost control | The dry run estimates GPU minutes. The broker enforces `gpu_budget_minutes` and rejects jobs past it |
| Reproducibility | The program text, SDK version, model hashes and seeds are all stored. Re-running the same program reproduces the video, using cached clips |
| Flexibility | Each video gets its own generation strategy, chaining logic, edit rhythm, transitions, grading and sound design |
| Growth | New capabilities are added as **new SDK functions**. The LLM can draft them in dev mode, but a human reviews and tests them before they enter the SDK |

### 8.2 The SDK surface the LLM writes against (`vidsdk`)

```python
# vidsdk/__init__.py: stable, versioned, documented; this file (stubs + docstrings) is sent to the LLM
from vidsdk.types import Clip, Frame, Audio, Words, Video, QCResult

# --- video generation (all return Clip; cached by content hash; budget/licence enforced) ---
def t2v(prompt: str, *, negative: str = "", seconds: float = 5.0, model: str = "wan22_a14b",
        res: str = "720p", seed: int | None = None, steps: int = 4) -> Clip: ...
def i2v(frame: Frame, prompt: str, *, negative: str = "", seconds: float = 5.0,
        model: str = "wan22_a14b", seed: int | None = None) -> Clip: ...
def flf2v(first: Frame, last: Frame, prompt: str, *, seconds: float = 5.0, seed: int | None = None) -> Clip: ...
def extend(clip: Clip, prompt: str, *, seconds: float = 5.0, method: str = "last_frame_i2v") -> Clip: ...
def r2v(refs: list[Frame], prompt: str, *, seconds: float = 5.0) -> Clip: ...
def talking(frame: Frame, audio: Audio, prompt: str = "") -> Clip: ...

# --- internal reference frames (never placed on the timeline) ---
def ref_frame(prompt: str, *, lora: str | None = None, model: str = "qwen_image", seed: int | None = None) -> Frame: ...
def edit_frame(frame: Frame, instruction: str) -> Frame: ...
def last_frame(clip: Clip) -> Frame: ...

# --- quality ---
def qc_clip(clip: Clip, *, expect: str, ref: Frame | None = None) -> QCResult: ...   # VLM + DOVER + drift
def best_of(candidates: list[Clip], *, expect: str) -> Clip: ...

# --- post ---
def interpolate(clip: Clip, fps: int = 32) -> Clip: ...
def upscale(clip: Clip, res: str = "1080p") -> Clip: ...

# --- audio ---
def tts(text: str, *, voice: str, style: str) -> Audio: ...
def music(prompt: str, seconds: float, *, seed: int | None = None) -> Audio: ...
def sfx(prompt: str, seconds: float) -> Audio: ...

# --- edit (FFmpeg under the hood) ---
class Timeline:
    def __init__(self, fps: int = 30, size: tuple[int, int] = (1920, 1080)): ...
    def add(self, clip: Clip, *, start: float, dur: float, transition: str = "cut", trans_dur: float = 0.5,
            speed: float = 1.0, grade: str | None = None) -> None: ...
    def audio(self, a: Audio, *, start: float, gain_db: float = 0, duck_under_voice: bool = False) -> None: ...
    def voice(self, a: Audio, words: Words) -> None: ...
    def subtitles(self, words: Words, style: str = "lower_third") -> None: ...
    def render(self, *, loudness_lufs: float = -14.0) -> Video: ...

# --- parallelism ---
def gather(*jobs) -> list: ...          # runs SDK calls concurrently on GPU workers
```

### 8.3 Codegen controller (runs in the orchestrator)

```python
# avs/codegen/controller.py
import ast, json, subprocess, tempfile, pathlib
from anthropic import Anthropic

client = Anthropic()
SDK_REF = pathlib.Path("vidsdk/REFERENCE.pyi").read_text()     # stubs + docstrings above
ALLOWED_IMPORTS = {"vidsdk", "vidsdk.types", "math", "random", "json", "itertools", "dataclasses", "typing"}
BANNED_NAMES = {"eval", "exec", "compile", "open", "__import__", "globals", "locals", "getattr", "setattr"}

SYSTEM = """You are a film director and Python engineer. Write ONE complete Python program that produces
the video described by the VideoPlan, using ONLY the vidsdk API given. Every shot on the timeline must be
generated video (t2v / i2v / flf2v / extend / r2v / talking). Reference frames are only conditioning.
Rules: use plan durations exactly; recurring characters use i2v from their ref_frame; run shots in parallel
with gather(); call qc_clip on every clip and retry up to 2 times with a new seed or adjusted prompt, then
fall back to a cheaper model; stay under the GPU budget; finish with Timeline.render() and print its uri.
Return only the code."""

def policy_check(code: str) -> list[str]:
    errors, tree = [], ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            if mod.split(".")[0] not in {m.split(".")[0] for m in ALLOWED_IMPORTS}:
                errors.append(f"line {node.lineno}: import '{mod}' not allowed")
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errors.append(f"line {node.lineno}: '{node.id}' not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"line {node.lineno}: dunder access not allowed")
    return errors

def static_gates(code: str) -> list[str]:
    errs = policy_check(code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    for cmd in (["ruff", "check", f.name], ["mypy", "--strict", f.name]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode: errs.append(r.stdout[-4000:])
    return errs

def generate_program(plan: dict, budget_min: int) -> str:
    msg = client.messages.create(
        model="claude-opus-5-5", max_tokens=32000, system=SYSTEM,
        messages=[{"role": "user", "content":
            f"<sdk>\n{SDK_REF}\n</sdk>\n<budget_gpu_minutes>{budget_min}</budget_gpu_minutes>\n"
            f"<videoplan>\n{json.dumps(plan)}\n</videoplan>"}])
    return strip_fences(msg.content[0].text)

def repair_program(code: str, errors: str) -> str:
    msg = client.messages.create(
        model="claude-sonnet-5-5", max_tokens=32000, system=SYSTEM,
        messages=[{"role": "user", "content":
            f"<sdk>\n{SDK_REF}\n</sdk>\n<program>\n{code}\n</program>\n<errors>\n{errors}\n</errors>\n"
            "Fix the program. Return the full corrected program only."}])
    return strip_fences(msg.content[0].text)

def build_program(plan: dict, budget_min: int, sandbox) -> str:
    code = generate_program(plan, budget_min)
    for attempt in range(4):
        errs = static_gates(code)
        if not errs:
            dry = sandbox.run(code, backend="mock")          # fake GPUs: instant placeholders
            if dry.ok and dry.est_gpu_minutes <= budget_min and dry.timeline_s >= 0.95 * plan["target_duration_s"]:
                return code                                   # ready for the real run
            errs = [dry.traceback or f"estimate {dry.est_gpu_minutes} min / timeline {dry.timeline_s}s fails limits"]
        code = repair_program(code, "\n".join(errs))
    raise NeedsHumanReview(code, errs)
```

**Sandbox run (step 14):** `sandbox.run(code, backend="broker")` executes in a Modal Sandbox or E2B container with the `vidsdk` client installed. The SDK client sends each operation to the **GPU broker**, an HTTP service that holds the RunPod/Modal keys, checks license, capability and budget, de-duplicates by content hash, and dispatches to GPU workers. If the real run throws, the traceback goes through `repair_program` again. Clips that already succeeded are cache hits, so a repair costs only the failed work.

### 8.4 Example of a program the LLM generates (abridged)

This is the kind of new code Claude writes for "What would happen if the Sun disappeared?". It is different for every video.

```python
# produce_prj_7f3a.py: generated by claude-opus-5-5 for VideoPlan prj_7f3a v2 (do not edit by hand)
import random
from vidsdk import (t2v, i2v, flf2v, extend, ref_frame, last_frame, qc_clip, interpolate, upscale,
                    tts, music, sfx, gather, Timeline)
from vidsdk.types import Clip

random.seed(7)
NEG = "text, watermark, cartoon, extra limbs, warping, flicker, lowres"
LOOK = "cinematic photoreal, anamorphic 35mm, volumetric light, deep blacks, subtle film grain"

def make(shot_prompt: str, seconds: float, *, frame=None, expect: str, tries: int = 3) -> Clip:
    """Generate one shot, QC it, retry with new seed then fall back to a cheaper model."""
    for i in range(tries):
        model = "wan22_a14b" if i < 2 else "ltx25"
        seed = random.randint(0, 2**31)
        clip = (i2v(frame, f"{shot_prompt}, {LOOK}", negative=NEG, seconds=seconds, model=model, seed=seed)
                if frame else
                t2v(f"{shot_prompt}, {LOOK}", negative=NEG, seconds=seconds, model=model, seed=seed))
        if qc_clip(clip, expect=expect, ref=frame).passed:
            return upscale(interpolate(clip, fps=32), res="1080p")
    raise RuntimeError(f"shot failed after {tries} tries: {expect}")

# --- recurring location: Earth seen from orbit, locked via an internal reference frame ---
earth_ref = ref_frame("Earth from low orbit at night side terminator, city lights, thin blue atmosphere, " + LOOK, seed=11)

# --- section 1: hook (5 shots) -------------------------------------------------------------
s1 = gather(
    lambda: make("extreme wide, the Sun blazing over a calm ocean horizon, slow push in", 5.2,
                 expect="bright sun over ocean"),
    lambda: make("macro, solar plasma filaments twisting, slow orbit", 4.8, expect="solar surface close-up"),
    lambda: make("aerial, golden afternoon light over a busy city, slow pull out", 5.0, expect="sunlit city aerial"),
    lambda: make("the Sun's disc collapses to a point and vanishes, stars appear, static wide", 4.6,
                 expect="sun disappears into darkness"),
    lambda: make("slow push in toward Earth, sunlight still on its day side", 6.0, frame=earth_ref,
                 expect="earth from orbit, same look as reference"),
)

# --- section 2: the 8 minutes (time-lapse feeling, continuous shot via extend) --------------
crossing = make("wide, Shibuya crossing at 3pm, crowds walking, daylight", 5.0, expect="busy crossing in daylight")
crossing_dark = extend(crossing, "daylight drains from the sky, crowds stop and look up, streetlights flicker on",
                       seconds=5.0)
# designed transition: day Earth → dark Earth using first/last frames
dark_earth = ref_frame("Earth from orbit fully dark, faint city lights only, " + LOOK, seed=12)
earth_fade = flf2v(earth_ref, dark_earth, "the planet's day side fades to black, slow drift", seconds=5.0)

# ... sections 3–7 generated the same way (frozen oceans, vents, drifting Earth) ...

# --- audio ---------------------------------------------------------------------------------
# narration was produced in step 7; PLAN_ASSETS is injected into the program by the sandbox runner
vo, words = PLAN_ASSETS.narration, PLAN_ASSETS.words
score = gather(lambda: music("slow cinematic ambient, low strings, sub drone, building", 52),
               lambda: music("sparse piano, cold, unresolved, slow", 64))
whoosh = sfx("deep sub-bass whoosh fading to silence", 3)

# --- edit ----------------------------------------------------------------------------------
tl = Timeline(fps=32, size=(1920, 1080))
t = 0.0
for clip, dur, tr in zip(s1, [5.2, 4.8, 5.0, 4.6, 6.0], ["fade_from_black", "cut", "cut", "cut", "dip_to_black"]):
    tl.add(clip, start=t, dur=dur, transition=tr, grade="luts/teal_orange_soft.cube"); t += dur
tl.add(crossing, start=t, dur=4.0); t += 4.0
tl.add(crossing_dark, start=t, dur=5.0, speed=0.9, grade="luts/cold_night.cube"); t += 5.0
tl.add(earth_fade, start=t, dur=5.0, transition="crossfade", trans_dur=1.0); t += 5.0
# ... remaining shots ...
tl.voice(vo, words)
tl.audio(score[0], start=0, gain_db=-16, duck_under_voice=True)
tl.audio(score[1], start=52, gain_db=-18, duck_under_voice=True)
tl.audio(whoosh, start=19.6, gain_db=-6)
tl.subtitles(words, style="lower_third")
video = tl.render(loudness_lufs=-14.0)
print(video.uri)
```

### 8.5 What may and may not be generated

| Generated per video by the LLM (new code each time) | Fixed, human-reviewed code (SDK / platform) |
|---|---|
| Shot generation strategy, model choice per shot, prompts, seeds | Model loading, weights, CUDA, containers |
| Chaining / extend / FLF2V logic, retries and fallbacks | GPU broker, budget, license allow-list, caching |
| Edit timeline, transitions, speed ramps, grading choices | FFmpeg primitives, encoding settings, loudness |
| Sound design: music sections, SFX placement, ducking | TTS, alignment, QC detectors |
| Optional: custom FFmpeg filter strings through `Timeline.raw_filter()` (validated against a filter allow-list) | YouTube upload, OAuth, idempotency, database |

---

## 9. Agent frameworks: still not needed

| Framework | Latest (PyPI, 2026-10) | Fit |
|---|---|---|
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/durable-execution) | 1.2.x | Checkpointing and human interrupts. Overlaps with the orchestrator |
| [PydanticAI](https://pydantic.dev/docs/ai/integrations/durable_execution/overview/) | 2.x | Best if you need one: typed output, plugs into Temporal and DBOS |
| CrewAI | 1.15 | Role-play abstraction. Weak durability |
| AutoGen | 0.7.5 | **Maintenance mode** [O] |
| Microsoft Agent Framework | 1.20 | Capable, Azure-leaning |
| OpenAI Agents SDK | 0.23 | 0.x, OpenAI-centric |
| smolagents | 1.26 | Code agents, but no durability. The codegen loop above is the same idea, made durable and gated |
| Claude Agent SDK | — | Built for autonomous file and shell agents. Overkill here |

The "code-writing agent" is the **generate → gate → dry-run → repair loop** in 8.3. It is about 100 lines of your own code, and it runs inside a durable workflow step. You don't need a framework for it.

---

## 18. Workflow orchestration

| Phase | Choice | Why |
|---|---|---|
| **1. Prototype** | Plain Python + SQLite + content-hash cache + `tenacity` | Zero ops |
| **2. MVP** | **[DBOS](https://docs.dbos.dev/)**: durable workflows in Postgres, queues with concurrency limits | Fewest moving parts. Prefect 3 if you want a UI |
| **3. Production** | **[Temporal](https://docs.temporal.io/develop/python)**: heartbeated long activities, task queue per GPU pool, child workflow per shot | Gold standard. [Hatchet](https://docs.hatchet.run/home) is the lighter alternative |
| ✖ | Airflow, n8n, Celery chords, Ray as the orchestrator | Wrong fit |

```python
@workflow
def produce(topic):
    plan    = step(plan_video, topic)                        # steps 2–6
    approve_if(plan, cfg.gate_plan)
    voice   = step(narrate_and_align, plan)                  # steps 7–8
    plan    = step(lock_durations, plan, voice)              # step 9
    program = step(build_program, plan, plan.gpu_budget_minutes)   # steps 10–13 (generate, gate, dry-run, repair)
    video   = step(run_program, program, max_repairs=2)      # step 14–21 (sandbox + GPU broker)
    qc      = step(final_qc, video, plan)                    # step 22
    while qc.redo_shots and qc.rounds < 2:                   # only failed shots regenerate (cached rest)
        program = step(repair_program_for_shots, program, qc.redo_shots)
        video   = step(run_program, program); qc = step(final_qc, video, plan)
    pkg     = step(package, plan, video)                     # steps 23–24
    approve_if(pkg, cfg.gate_publish)                        # step 25
    vid_id  = step(upload_youtube, pkg, idempotency_key=plan.project_id)   # step 26
```

---

## 10. Execution environment

**Approximate on-demand $/GPU-hour** [O unless marked; prices change often]:

| Provider | RTX 4090 | L40S | A100 80GB | H100 | Notes |
|---|---|---|---|---|---|
| [RunPod Pods](https://www.runpod.io/pricing) (Community / Secure) | $0.34 / $0.74 | $0.79 / $1.09 | $1.19–1.59 | $1.99–3.49 | Cheapest reliable hourly option. Network volumes for weights |
| [RunPod Serverless](https://docs.runpod.io/serverless/overview) | $1.10 | $1.75 | $2.72 | $4.79 | Queue API, scales to zero. **MVP pick** |
| [Modal](https://modal.com/pricing) | — | ~$1.95 | ~$2.50 | ~$3.95 | Per-second Python functions + **Sandboxes** for the generated program. **MVP pick** |
| [Vast.ai](https://vast.ai/pricing) | ~$0.25–0.40 [C] | varies | varies | ~$1.5–3.5 [C] | Cheapest, variable reliability |
| [Lambda](https://lambda.ai/pricing) | — | — | A100 40GB $1.99 | $3.29–4.29 | Reserved / steady volume |
| AWS / GCP | — | g6e ~$1.86 [C] | — | $6.9–11 [C] | Expensive |
| [fal](https://fal.ai/pricing) / [Replicate](https://replicate.com/pricing) | — | ~$3.51 | ~$5.04 | ~$5.49 | Hosted fallback |

**Which GPU for full-AI video:**
- **Wan 2.2 A14B at 720p:** an **H100 80 GB** or **A100 80 GB** runs it without offload (fastest per clip). An **L40S 48 GB** or **4090 24 GB** works with FP8 + offload + LightX2V, which is slower per clip but often cheaper per clip.
- **Benchmark both** in Phase 1 and record $/clip in the model registry.

**Google Colab:** **not for production.**
- No job API.
- No guaranteed GPU.
- Session caps (12 h free, 24 h Pro+) and idle kills.
- The free tier forbids remote control ([FAQ](https://research.google.com/colaboratory/faq.html)) [O].
- **Use it for testing models and prompts.**

---

## 11. Video assembly: FFmpeg is the core renderer

- **Timeline → filtergraph.** The generated program builds a `Timeline`. The SDK turns it into FFmpeg graphs via **[typed-ffmpeg](https://github.com/livingbio/typed-ffmpeg)**.
- **Normalize every clip** to 1920×1080, 30/32 fps, yuv420p before joining.
- **Join method:**
  - Cuts use the concat demuxer, with no re-encode.
  - Transitions use `xfade` / `acrossfade`.
  - Grading uses `lut3d` (`.cube` files) or `eq`/`curves`.
  - Speed ramps use `setpts`.
- **Avoid:** MoviePy except for prototypes. Remotion needs a paid license for teams of 4+.

```bash
# trim a generated clip to the narration span and grade it
ffmpeg -i clip.mp4 -ss 0 -t 4.6 -vf "lut3d=luts/teal_orange_soft.cube,scale=1920:1080,fps=32,format=yuv420p" -an -c:v libx264 -crf 16 s01.mp4

# crossfade two generated shots (offset = durA - transition)
ffmpeg -i a.mp4 -i b.mp4 -filter_complex "[0:v][1:v]xfade=transition=fade:duration=1:offset=4[v]" -map "[v]" -c:v libx264 -crf 16 ab.mp4

# slow-motion speed ramp (0.9x) on a shot
ffmpeg -i s07.mp4 -vf "setpts=PTS/0.9" -an s07_slow.mp4

# duck music under narration
ffmpeg -i narration.wav -i music.wav -filter_complex "[1:a]volume=0.35[m];[0:a]asplit=2[sc][vo];[m][sc]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=350[duck];[vo][duck]amix=inputs=2:duration=first:normalize=0[a]" -map "[a]" mix.wav

# loudness pass 1 (measure), then pass 2 with measured_* values and linear=true
ffmpeg -i mix.wav -af loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json -f null -

# final mux with burned subtitles, YouTube settings
ffmpeg -i video.mp4 -i mix_norm.wav -vf "ass=subs.ass" -map 0:v -map 1:a -c:v libx264 -profile:v high -crf 18 -preset slow -pix_fmt yuv420p -c:a aac -b:a 384k -ar 48000 -movflags +faststart final.mp4
```

YouTube recommends H.264 High, closed GOP, 4:2:0, AAC-LC 48 kHz, faststart, and ~8 Mbps at 1080p ([source](https://support.google.com/youtube/answer/1722171)) [O]. −14 LUFS is a community convention, not an official target [C].

---

## 12. Full-AI vs hybrid: why full-AI is chosen, and how to make it work

| | **Full-AI video (chosen)** | Hybrid (revision 1) |
|---|---|---|
| Look | Every second moves. Cinematic | Mostly stills with camera moves |
| Cost, 5-min video | ~$12–20 standard / ~$100–150 max quality [est] | ~$2–3 [est] |
| Wall clock | ~1–1.5 h on 8 GPUs | ~25 min |
| Consistency | Hard: needs refs, LoRA, drift QC | Easy |
| GPU | 48–80 GB preferred | 16–24 GB |
| Policy risk | Higher if clips feel unrelated | Lower |

**Rules that make full-AI viable:**
1. **Narration-led editing.** Every clip illustrates a specific line, so the video never becomes a reel of unrelated clips (the policy risk).
2. **Shots of 3–6 s, cut often.** This matches what the models do well. Long takes only via `extend`, at most 2 hops.
3. **References for anything that recurs.** Pure T2V only for one-off subjects.
4. **Draft-then-final.** Optionally run a 480p TI2V-5B / LTX draft pass of the whole video (cheap). Let a VLM and/or human approve the cut, then render finals at 720p A14B.
5. **QC every clip and generate up to 3 candidates** for hero shots (`best_of`).
6. **Budget caps per video** are enforced by the broker.

---

## 13. Character and scene consistency (full-AI)

1. **Character and location bible** in the plan, with descriptions locked and pasted verbatim into prompts.
2. **Reference frames:** an approved turnaround sheet per character (Qwen-Image), and an establishing frame per location. They are never shown in the video.
3. **Character LoRA** (ai-toolkit, ~20–40 images from the sheet, about 1 h on a 24 GB card [C]), trained for both **Wan** and **Qwen-Image**. This is the most reliable method for recurring characters.
4. **I2V from the reference** for every shot containing a recurring subject. Use `edit_frame` for pose and outfit variants, always from the original reference.
5. **FLF2V** for designed transitions. **Last-frame chaining** for continuous action (≤ 2 hops).
6. **Phantom / VACE reference-to-video** when the subject must move a lot.
7. **Automated drift rejection** in `qc_clip`: subject similarity to the reference plus a VLM "same character, same outfit, same place?" check, with regeneration on failure.
8. **Seeds** are recorded for reproducibility. They are not an identity strategy.

**License caution:** InsightFace face models are non-commercial. Use a commercially licensed embedder [verify], or rely on the VLM check.

---

## 14. Automated QC pipeline

| Gate | Check | Threshold [est, calibrate] | Action on failure |
|---|---|---|---|
| Per clip (inside `qc_clip`) | ffprobe shape; **DOVER** quality; **Qwen3-VL** "does this clip show *expect*? warping? extra limbs? text artifacts? flicker?"; subject similarity vs ref; motion check (frame-difference energy, so the clip is not near-static) | VLM pass; DOVER above the bottom 10%; similarity ≥ calibrated value | New seed → adjusted prompt → fallback model → human |
| Per TTS sentence | faster-whisper WER, clipping, silence | WER ≤ 5% | Regenerate |
| Final video | `blackdetect=d=0.5:pix_th=0.10`, `freezedetect=n=-60dB:d=2`, `silencedetect=n=-50dB:d=1.5`, `ebur128` | No unintended black > 0.5 s, freeze > 2 s or silence > 1.5 s; −14 ±1 LUFS; TP ≤ −1 dBTP | Fix the timeline / regenerate the shot |
| Duplicates | pHash at 1 fps | Hamming ≤ 6 between non-adjacent shots | Regenerate one |
| Structure | PySceneDetect cuts vs timeline; duration vs plan | ±10% | Flag |
| Sync | Word times vs subtitle cues | ±200 ms | Re-align |
| Subtitles | ≤ 42 chars/line, ≤ 2 lines, ≤ 20 chars/s, ≥ 0.8 s | Hard | Re-segment |
| Encode | H.264 High, yuv420p, AAC 48 kHz, faststart | Hard | Re-encode |
| Holistic | VLM on a contact sheet + script + title + thumbnail: coherence, consistency across shots, misleading claims, policy | Pass/flag | Human review |

---

## 15–16. YouTube automation and thumbnails

### API facts [O, checked 2026-10-06]

- **Audit:** uploads from unverified API projects (created after 2020-07-28) are **locked private** until the project passes the YouTube API compliance audit ([videos.insert](https://developers.google.com/youtube/v3/docs/videos/insert), [audit form](https://developers.google.com/youtube/v3/guides/quota_and_compliance_audits)). **Apply in week one.**
- **Quota:**
  - Since 2026-06-01, `videos.insert` has **its own bucket: 100 uploads per day at 1 unit each**.
  - The shared 10,000 units/day covers the rest: `thumbnails.set` 50, `captions.insert` 400, `videos.update` 50, `playlistItems.insert` 50 ([quota](https://developers.google.com/youtube/v3/determine_quota_cost), [revision history](https://developers.google.com/youtube/v3/revision_history)).
  - The quota page intro still shows the old 1600, so confirm in the Cloud console.
- **OAuth:**
  - An app left in "Testing" gets refresh tokens that **expire after 7 days**. Publish the consent screen ([OAuth](https://developers.google.com/identity/protocols/oauth2)).
  - Captions need the `youtube.force-ssl` scope.
- **Resumable upload:**
  - Chunks must be multiples of 256 KB. Resume with `Content-Range: bytes */N` and a 308 reply ([guide](https://developers.google.com/youtube/v3/guides/using_resumable_upload_protocol)).
  - Retry 5xx with backoff. Never retry `quotaExceeded`.
- **Fields:**
  - `privacyStatus`, `publishAt` (private + never published), `selfDeclaredMadeForKids`, **`containsSyntheticMedia`**, `categoryId`, `defaultLanguage`, `defaultAudioLanguage`.
  - Title ≤ 100 chars, description ≤ 5000 bytes, tags ≤ 500 chars total.
- **Thumbnail:** up to 50 MB (since 2026-09-14). Recommended 3840×2160, minimum 640 px wide, and the channel must be verified ([help](https://support.google.com/youtube/answer/72431)).
- **Long uploads:** videos over 15 minutes need a verified account.

### Duplicate prevention

1. Write `publications(project_id, idempotency_key, session_uri)` before uploading.
2. Resume the stored session on retry. Never start a fresh upload.
3. Store the `videoId` on the 201 reply, before setting the thumbnail or captions.
4. Put a `ref:<project_id>` marker in the description. Check the uploads playlist for it before any fresh upload.

### Policies

- **Disclosure:** realistic AI scenes **must be disclosed**. Set `containsSyntheticMedia=true` ([policy](https://support.google.com/youtube/answer/14328491)).
- **Monetization:** "inauthentic content" (since 2025-07-15) excludes mass-produced templated AI content, and explicitly videos that **"stitch together unrelated or inconsistent AI clips"**. It is assessed at channel level ([policy](https://support.google.com/youtube/answer/1311392)).
- **Spam:** the spam policy covers "automated or synthetic mass-production" ([policy](https://support.google.com/youtube/answer/2801973)).
- **For full-AI video this means:**
  - Keep editing narration-led and consistent.
  - Back factual topics with research.
  - Vary profiles, voices and music.
  - Keep a human review early on.
  - Cap volume.

### Thumbnail workflow (no separate image needed by default)

```text
all final clips
  → sample 3 frames per hero shot → Qwen3-VL scores: subject clarity, emotion, contrast, works at 160×90
  → best frame (or, if none scores well, a dedicated Qwen-Image render of the LLM's thumbnail concept)
  → Pillow: crop 16:9, 2–4 word text from the LLM, stroke + safe margins
  → OCR check + VLM "misleading?" check → thumbnails.set
```

---

## 19. Storage architecture

- **Prototype:** local disk.
- **MVP onward:** **Cloudflare R2** (S3-compatible, no egress fees; GPU workers pull and push clips constantly) or MinIO.
- **Not Google Drive.**

```text
s3://videos/
  projects/{project_id}/
    plan/        brief.json factsheet.json script.json storyboard.json videoplan.v1.json videoplan.v2.json
    code/        produce.v1.py produce.v2.py ... repair_log.json gates.json dryrun.json
    refs/        {character_or_location_id}.png        # internal conditioning frames
    shots/{shot_id}/
      attempts/{n}.mp4  qc_{n}.json                     # every candidate kept 30 days
      clip_raw.mp4  clip.mp4 (interpolated + upscaled)
    audio/       narration/{i}.wav words.json music/{section}.wav sfx/{cue}.wav mix.wav
    subtitles/   captions.srt captions.ass
    render/      timeline.json final.mp4 contact_sheet.jpg qc_report.json
    publish/     thumbnail.png metadata.json youtube.json
  cas/{sha256[:2]}/{sha256}.{ext}      # content-addressed cache: same inputs → reuse clip
  loras/{character_id}/wan.safetensors qwen_image.safetensors
  library/music/{profile_id}/...
```

**Footprint** [est]:
- About **3–5 GB** of intermediates per 5-minute full-AI video (raw, interpolated and upscaled clips plus attempts).
- 0.3–0.6 GB for the final file.
- Delete intermediates after 30 days. Keep the plan, code, final video and QC reports forever.

---

## 20. Database / state model (Postgres)

```sql
create table projects (
  id uuid primary key, topic text not null, status text not null,  -- planning|voicing|coding|generating|qc|ready|published|failed
  style_profile_id text, target_duration_s int, gpu_budget_minutes int, license_mode text default 'commercial',
  created_at timestamptz default now(), updated_at timestamptz
);

create table plan_versions (
  id uuid primary key, project_id uuid references projects, kind text, version int,   -- brief|factsheet|script|storyboard|videoplan
  body jsonb not null, llm_model text, prompt_hash text, tokens_in int, tokens_out int, cost_usd numeric,
  approved_by text, created_at timestamptz default now(), unique(project_id, kind, version)
);

create table programs (                         -- every generated / repaired program
  id uuid primary key, project_id uuid references projects, version int,
  parent_id uuid references programs,           -- repair lineage
  code text not null, code_sha text, sdk_version text, llm_model text,
  reason text,                                  -- initial|gate_fail|dryrun_fail|runtime_fail|qc_redo
  gate_errors jsonb, dryrun jsonb,              -- est_gpu_minutes, timeline_s, coverage
  status text,                                  -- generated|passed_gates|dry_run_ok|running|succeeded|failed
  created_at timestamptz default now(), unique(project_id, version)
);

create table scenes (
  id text primary key, project_id uuid references projects, section_id text, ord int,
  spec jsonb not null, status text not null, final_asset_id uuid
);

create table generation_attempts (              -- one row per SDK GPU call (written by the GPU broker)
  id uuid primary key, project_id uuid, scene_id text, program_id uuid references programs,
  op text,                                      -- t2v|i2v|flf2v|extend|r2v|talking|ref_frame|tts|music|sfx|interpolate|upscale
  model text, model_sha text, provider text, gpu_type text,
  params jsonb, prompt text, negative_prompt text, seed bigint,
  input_hash text not null, status text, error text, cache_hit bool,
  gpu_seconds numeric, cost_usd numeric, qc jsonb, output_asset_id uuid,
  created_at timestamptz default now()
);
create index on generation_attempts(input_hash);

create table assets (
  id uuid primary key, sha256 text unique, uri text, mime text, bytes bigint,
  duration_s numeric, width int, height int, fps numeric, license text, source text,
  created_at timestamptz default now()
);

create table qc_reports (id uuid primary key, project_id uuid, scope text, scope_id text, passed bool, report jsonb, created_at timestamptz default now());

create table budgets (project_id uuid primary key references projects, gpu_minutes_limit int, gpu_minutes_used numeric default 0, usd_used numeric default 0);

create table publications (
  id uuid primary key, project_id uuid references projects unique,
  idempotency_key text unique, session_uri text, youtube_video_id text unique,
  status text, privacy text, publish_at timestamptz, metadata jsonb, synthetic_disclosed bool,
  created_at timestamptz default now()
);
```

**Retrying only failed work:**
- Each GPU call's `input_hash` covers (op, model hash, params, seed, input asset hashes).
- When a repaired program runs, every unchanged call is a **cache hit**. Only new or failed calls use GPU time.
- Stage-level resume comes from the orchestrator (DBOS/Temporal).

---

## 23. Cost and performance estimates (full-AI video)

**Assumptions** [est, planning numbers; measure your own in Phase 1]:
- About 1 clip per 5 s.
- **Standard** = Wan 2.2 A14B + LightX2V 4-step at 720p, then RIFE + SeedVR2, ×1.5 for retries. That is **~6–8 GPU-min per final clip**, at ~$2/GPU-h effective (H100 pods / L40S serverless).
- **Max quality** = full-step A14B at 720p on H100, ~20 min per clip ×1.5, at ~$4–4.8/h.
- **Draft** = TI2V-5B / LTX-2.5 distilled at 480p, ~1 GPU-min per clip.
- **Wall clock** assumes 8 GPU workers in parallel.
- **LLM costs** (planning, code generation, repairs, VLM QC via API) add roughly $1–3 per video [est]. Check current API pricing.

| Video length | Clips | Draft pass (optional) | **Standard (recommended)** GPU / cost / wall clock | Max quality GPU / cost |
|---|---|---|---|---|
| 30 s | ~6 | ~6 GPU-min / ~$0.30 | ~1 GPU-h / **~$2–3** / ~20 min | ~3 H100-h / ~$12–15 |
| 1 min | ~12 | ~12 GPU-min / ~$0.50 | ~1.7 GPU-h / **~$3–5** / ~25 min | ~6 H100-h / ~$25–30 |
| 5 min | ~60 | ~1 GPU-h / ~$2 | ~8 GPU-h / **~$12–20** / ~1–1.5 h | ~30 H100-h / ~$120–145 |
| 10 min | ~120 | ~2 GPU-h / ~$4 | ~16 GPU-h / **~$25–40** / ~2–2.5 h | ~60 H100-h / ~$240–290 |

**Storage:** 3–5 GB intermediates for each 5 minutes of video, plus 0.3–0.6 GB final.

**Biggest cost levers:**
1. The LightX2V 4-step LoRA (≈10× fewer steps).
2. Fewer retries through a better first prompt and the draft pass.
3. Cheaper GPUs (RunPod Community, Vast) for non-urgent batches.
4. Generating at 480p and relying on SeedVR2 upscaling [test quality first].

---

## H. Worked example: "What would happen if the Sun disappeared?"

| Step | What happens | Used |
|---|---|---|
| 1. Topic | `avs make "What would happen if the Sun disappeared?"` | CLI → DBOS workflow |
| 2. Brief | audience 16–45, 7-min long-form, angle "minute-by-minute timeline", profile `cosmic_documentary`, tone mysterious/awe, hook "For eight minutes and twenty seconds, nobody on Earth would know." | Claude Opus, structured output |
| 3. Research | Facts with URLs: light and gravity changes take ~8 min 20 s to arrive; Earth continues in a straight line; photosynthesis stops; surface below −18 °C within ~a week; oceans freeze over from the top while the deep ocean stays liquid; hydrothermal vents | Claude + web search |
| 4. Script | 7 sections, ~980 words at 140 wpm | Claude Opus |
| 5. Storyboard | 84 video shots (avg 5 s) with camera, lens, light, motion, transitions, SFX cues. Locations: `loc_earth_orbit`, `loc_tokyo`, `loc_ocean`, `loc_vent` | Claude Sonnet ×7 sections in parallel |
| 6. Validate | Fixes 3 shots > 5 s with no extend plan; rejects 2 adjacent identical shots; critic tightens the hook | Pydantic validator + Claude Opus critic |
| 7–8. Voice | 61 sentences → wavs; 2 regenerated (WER > 5%); word timestamps | Qwen3-TTS ("deep calm documentary narrator, hushed awe"), faster-whisper, WhisperX |
| 9. Lock timing | Shot durations set from word times; total 7:04 | Python |
| 10. Codegen | `produce.v1.py` (~600 lines): ref frames for 4 locations; shot modes **t2v 56, i2v_ref 16, flf2v 5, extend 7**; hero shots use `best_of` 3 candidates; grades per section; music in 7 sections | Claude Opus + vidsdk reference |
| 11–13. Gates and repair | v1 fails mypy (wrong arg type for `flf2v`). v2 passes the gates. The dry run estimates 610 GPU-min (budget 700) and covers 7:04 of timeline → approved | ruff, mypy, AST policy, MockBackend, Claude Sonnet repair |
| 14–16. Generate | 4 ref frames; 84 shots across 8 H100 workers; LightX2V 4-step | Modal Sandbox (program) → GPU broker → Modal/RunPod H100 (Wan 2.2 A14B, Wan 2.1 FLF2V, Qwen-Image) |
| 17. Clip QC | 19 clips regenerated: 11 for crowd faces warping in Tokyo shots, 5 for frozen oceans that looked static, 3 for Earth from orbit drifting from its ref frame (new seed, then passed). 1 Tokyo shot fell back to LTX-2.5 | Qwen3-VL, DOVER, similarity vs ref |
| 18–19. Audio | 7 music sections; 23 SFX (sub-bass whoosh, ice cracking, wind, crowd gasp) | ACE-Step 1.5, Stable Audio Open |
| 20. Post | 16→32 fps, 720p→1080p | RIFE, SeedVR2 |
| 21. Edit | Timeline from the program: cuts on narration phrases, dip-to-black at section breaks, whip on time jumps, cold LUT after the Sun vanishes, 0.9× speed ramp on the dark crossing, ducked music, −14 LUFS | FFmpeg via typed-ffmpeg |
| 22. Final QC | No black/freeze issues, WER 1.8%, −14.2 LUFS, no duplicates, VLM contact sheet OK → pass | FFmpeg detect filters, PySceneDetect, imagehash, faster-whisper, Qwen3-VL |
| 23. Metadata | Title "What If the Sun Disappeared? (Minute by Minute)", chapters, sources, AI disclosure line, tags, category 28 (Science & Technology), playlist "What If..." | Claude Sonnet |
| 24. Thumbnail | Best frame: half-lit frozen Earth + text "8 MINUTES" | Qwen3-VL frame scoring, Pillow, OCR |
| 25. Approval | You approve in the review page | Human |
| 26. Upload | Resumable `videos.insert` (private, `publishAt` = 2026-10-10 15:00 UTC, `containsSyntheticMedia=true`) → `thumbnails.set` → `captions.insert` → `playlistItems.insert` | YouTube Data API v3 |
| 27. Record | ~11 GPU-h, ~$20–25 GPU, program v2 + logs stored, `youtube_video_id` saved | Postgres, R2 |

---

## E. MVP: the smallest full-AI version to build first

```text
topic
 → Claude: Brief → Script → Storyboard (VideoPlan, ≤ 90 s, ~15 shots)            [structured outputs]
 → Qwen3-TTS per sentence → WhisperX word times → lock shot durations
 → Claude writes produce.py against vidsdk v0 (t2v, i2v, ref_frame, qc_clip, Timeline)
 → ruff + AST policy + dry run on MockBackend → repair loop (≤ 3)
 → run in Modal Sandbox → Modal GPU functions: Wan 2.2 TI2V-5B (draft) or A14B + LightX2V (final)
 → RIFE → FFmpeg timeline render (cuts, crossfades, music from a small ACE-Step pool, ducking, loudnorm, SRT)
 → basic QC (ffprobe, black/freeze/silence, WER, VLM on contact sheet)
 → you watch → upload PRIVATE via YouTube API
```

- **Scope:**
  - **vidsdk v0** has ~10 functions.
  - **One GPU provider** (Modal: GPU functions and Sandboxes in one place).
  - **SQLite + local disk.**
  - **No** LoRA training, no talking heads, no orchestrator yet.
- **Effort:** about **3–4 weeks** for an experienced Python developer [est].
- **What it proves:** codegen reliability (the percentage of programs passing within 3 repairs), clip success rate, $/clip, and whether the result is watchable.

## F. Production version

- **Orchestration:** DBOS → Temporal. Per-shot child workflows, resume, and redoing only the shots that failed QC.
- **GPU broker:** a separate service that holds the keys and checks budget, licenses and capability. It runs a content-hash cache and routes across Modal, RunPod and fal (failover).
- **SDK:** **vidsdk v1** adds flf2v, extend, r2v, talking, best_of, edit_frame, upscale, music/sfx, raw filters with an allow-list, and draft mode.
- **Consistency:** character/location bible, an automated LoRA training job (ai-toolkit), and drift QC.
- **Two-pass production:** a 480p draft of the whole video gets VLM and human approval, then the final pass runs at 720p→1080p.
- **Codegen hardening:**
  - A golden-test suite of 50 VideoPlans to regression-test prompt or SDK changes.
  - Tracking of repair-rate and failure-type metrics.
  - SDK docs auto-generated from the stubs.
- **Storage and state:** Postgres + R2, cost dashboards, QC rejection rate per model, and YouTube Analytics retention fed back into profile selection.
- **Compliance:**
  - Pass the API audit and publish the OAuth app.
  - Add a policy-check stage covering synthetic disclosure, sensitive topics and AI expert personas.
  - Set volume caps and keep a human spot-check rate.

## G. Repository structure

```text
ai-video-automation/
├── pyproject.toml
├── config/
│   ├── style_profiles/*.yaml
│   ├── model_registry.yaml        # model → sha, license, vram, max_seconds, modes, measured $/clip
│   └── channels/*.yaml
├── vidsdk/                        # the stable SDK the generated code imports (client side)
│   ├── __init__.py  types.py  timeline.py
│   ├── REFERENCE.pyi              # stubs + docstrings sent to the LLM (generated from code)
│   └── mock.py                    # MockBackend for dry runs
├── broker/                        # GPU broker service: auth, budget, licence, cache, routing
│   ├── app.py  budget.py  cache.py  providers/{modal.py,runpod.py,fal.py}
├── workers/                       # GPU worker images + handlers
│   ├── wan/  ltx/  qwen_image/  tts/  music/  sfx/  post/ (rife, seedvr2)  qc/ (dover, qwen3-vl)
├── src/avs/
│   ├── schemas/                   # Brief, FactSheet, Script, VideoPlan, QC, YT metadata
│   ├── planning/                  # stage prompts (versioned), validator, critic, few-shot/
│   ├── voice/                     # tts per sentence, WER check, alignment, duration locking
│   ├── codegen/                   # controller.py (generate/gates/dry-run/repair), policy.py, prompts/
│   ├── sandbox/                   # modal_sandbox.py / e2b.py runners
│   ├── render/                    # timeline → typed-ffmpeg graphs, subtitles, luts/
│   ├── qc/                        # final-video checks, thresholds.yaml
│   ├── publish/                   # oauth, resumable upload, thumbnails, captions, idempotency
│   ├── storage/  db/              # R2/local, SQLAlchemy + Alembic
│   ├── workflows/produce.py       # plain → DBOS → Temporal
│   └── cli.py                     # avs make "topic" --profile auto --draft
├── tests/
│   ├── golden_plans/              # VideoPlans used to regression-test codegen
│   ├── test_policy.py  test_sdk_mock.py  test_render.py  test_publish_idempotency.py
└── notebooks/                     # Colab/local model experiments only
```

## I. Implementation roadmap

| Phase | Build | Exit criterion |
|---|---|---|
| **0. Setup (week 1)** | GCP project + YouTube OAuth app (published), **apply for the API audit**. In Colab or on a local GPU, run Wan 2.2 (5B and A14B + LightX2V), LTX-2.5, Qwen3-TTS and ACE-Step by hand. Record time and quality per clip | You know $/clip and quality on 2 GPU types |
| **1. Plan + voice (weeks 1–2)** | Schemas, planning stages, validator, critic, TTS + alignment + duration locking | 10 topics → valid timed VideoPlans |
| **2. SDK + codegen (weeks 2–4)** | vidsdk v0 + MockBackend, codegen controller (generate, gates, dry run, repair), Modal Sandbox runner, Modal GPU functions for Wan/TTS/RIFE, FFmpeg Timeline | ≥ 90% of programs pass within 3 repairs; a 60–90 s full-AI video renders end to end |
| **3. Quality loop (weeks 4–6)** | `qc_clip` (VLM + DOVER + motion), retries/fallbacks, final QC, draft pass, private upload | 5 videos per week with no manual steps, watchable |
| **4. Durability + broker (weeks 6–8)** | GPU broker (budget, license, cache), Postgres + R2, DBOS workflow, cost tracking, RunPod as a second provider | Kill anything mid-run; resume reuses all finished clips |
| **5. Consistency + long-form (weeks 8–11)** | Ref frames, LoRA training job, flf2v/extend/r2v, drift QC, 5–10 min videos | 7-min video with consistent recurring subjects |
| **6. Publishing (weeks 11–13)** | Audit approved → scheduled public uploads, metadata, thumbnails from frames, captions, policy checks, approval page | Scheduled public uploads, zero duplicates |
| **7. Production hardening** | Temporal, multi-provider failover, golden-plan regression tests, analytics feedback, talking characters | — |

**Build first:** the `VideoPlan` schema, **vidsdk v0 + MockBackend**, and the **codegen controller**. Once generated programs reliably pass the gates against the mock, you can swap in any video model behind the SDK without changing the codegen.

---

## 24. Sources

### Official documentation and repositories

- **YouTube Data API:**
  - [videos.insert](https://developers.google.com/youtube/v3/docs/videos/insert)
  - [videos resource](https://developers.google.com/youtube/v3/docs/videos)
  - [quota costs](https://developers.google.com/youtube/v3/determine_quota_cost)
  - [revision history](https://developers.google.com/youtube/v3/revision_history)
  - [resumable upload](https://developers.google.com/youtube/v3/guides/using_resumable_upload_protocol)
  - [audits](https://developers.google.com/youtube/v3/guides/quota_and_compliance_audits)
  - [thumbnails.set](https://developers.google.com/youtube/v3/docs/thumbnails/set)
  - [captions.insert](https://developers.google.com/youtube/v3/docs/captions/insert)
  - [errors](https://developers.google.com/youtube/v3/docs/errors)
  - [OAuth 2.0](https://developers.google.com/identity/protocols/oauth2)
- **YouTube policies:**
  - [synthetic content](https://support.google.com/youtube/answer/14328491)
  - [monetization / inauthentic content](https://support.google.com/youtube/answer/1311392)
  - [spam](https://support.google.com/youtube/answer/2801973)
  - [encoding](https://support.google.com/youtube/answer/1722171)
  - [thumbnails](https://support.google.com/youtube/answer/72431)
  - [length limits](https://support.google.com/youtube/answer/71673)
- **Video models:** [Wan 2.2](https://github.com/Wan-Video/Wan2.2), [Wan 2.1](https://github.com/Wan-Video/Wan2.1), [VACE](https://github.com/ali-vilab/VACE), [Phantom](https://github.com/Phantom-video/Phantom), [LTX-2](https://github.com/Lightricks/LTX-2), [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), [Kandinsky 5](https://github.com/kandinskylab/kandinsky-5), [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video), [MAGI-1](https://github.com/SandAI-org/MAGI-1), [FramePack](https://github.com/lllyasviel/FramePack), [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2), [LightX2V](https://github.com/ModelTC/LightX2V)
- **Image models:** [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [Z-Image](https://github.com/Tongyi-MAI/Z-Image), [FLUX.2](https://github.com/black-forest-labs/flux2), [FLUX.1-dev license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md), [ai-toolkit](https://github.com/ostris/ai-toolkit)
- **Talking characters and post-processing:** [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk), [LatentSync](https://github.com/bytedance/LatentSync), [MuseTalk](https://github.com/TMElyralab/MuseTalk), [Practical-RIFE](https://github.com/hzwer/Practical-RIFE), [SeedVR](https://github.com/ByteDance-Seed/SeedVR)
- **Audio:** [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), [Chatterbox](https://github.com/resemble-ai/chatterbox), [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), [IndexTTS](https://github.com/index-tts/index-tts), [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), [Stability license](https://stability.ai/community-license-agreement), [Freesound API terms](https://freesound.org/help/tos_api/)
- **Rendering and QC:** [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html), [typed-ffmpeg](https://github.com/livingbio/typed-ffmpeg), [MoviePy](https://github.com/Zulko/moviepy), [Remotion license](https://www.remotion.pro/license), [PySceneDetect](https://github.com/Breakthrough/PySceneDetect), [DOVER](https://github.com/vqassessment/DOVER), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [WhisperX](https://github.com/m-bain/whisperX)
- **LLM, orchestration and agents:** [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [LangGraph](https://docs.langchain.com/oss/python/langgraph/durable-execution), [PydanticAI durable execution](https://pydantic.dev/docs/ai/integrations/durable_execution/overview/), [AutoGen notice](https://github.com/microsoft/autogen), [Temporal Python](https://docs.temporal.io/develop/python), [DBOS](https://docs.dbos.dev/), [Hatchet](https://docs.hatchet.run/home)
- **GPU providers:** [RunPod pricing](https://www.runpod.io/pricing), [RunPod Serverless](https://docs.runpod.io/serverless/overview), [Modal pricing](https://modal.com/pricing), [Modal cold starts](https://modal.com/docs/guide/cold-start), [Lambda](https://lambda.ai/pricing), [Replicate](https://replicate.com/pricing), [fal](https://fal.ai/pricing), [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- **Reference projects:** [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo), [ShortGPT](https://github.com/RayVentura/ShortGPT), [short-video-maker](https://github.com/gyoridavid/short-video-maker). They are stock-footage-based and have no codegen, QC or consistency, so they are only useful for plumbing ideas.

### Community or third-party sources (treat with caution)

- Vast, AWS and GCP GPU prices.
- Consumer-GPU timings for Wan + LightX2V.
- The LTX-2.5 release coverage.
- Wan 3.0 open-weights claims (unconfirmed).
- The −14 LUFS convention.
- Faceless-channel cost write-ups.

### Still to verify

- The commercial terms for MiniMax-H3, Skywork and CogVideoX.
- The LongCat-Video VRAM requirement.
- Whether Chatterbox watermarks every variant.
- A commercially licensed face embedder for drift QC.
- Shorts thumbnails via the API.
- Modal Sandbox / E2B limits for long-running orchestration programs: the program mostly waits on GPU jobs, so confirm the sandbox timeout covers a 1–2 h run, or split execution per section.

---

> **Note added 2026-10-16 (append only).** Logical, spatial and temporal consistency of generated shots (entry side,
> direction, object hand-offs, action order, camera axis) is now designed and partly implemented in
> `docs/09_LOGICAL_CONSISTENCY_ARCHITECTURE.md`: typed scene-states per shot, VACE layout-controlled generation,
> geometric + video-language-model checks with PASS/FAIL/UNKNOWN evidence, targeted repairs, and an acceptance gate
> that keeps failed or unchecked clips out of the final video. That document and `docs/08_COLAB_VIDEO_PIPELINE_GUIDE.md`
> supersede the generation and QC parts of this file.

---

# Part 2 — The pipeline as built (appended 2026-10-18)

> **Append only.** Everything above this line is the original research (2026-10) and is kept unchanged. This part
> records what was actually built and decided since then, so this file is a single place to read the whole story.
> Where Part 1 and Part 2 disagree, **Part 2 is current**. Detailed write-ups:
> - `docs/02_ARCHITECTURE.md` §12–13 (architecture as built);
> - `docs/08_COLAB_VIDEO_PIPELINE_GUIDE.md` §30–39 (every change, with cause, fix and tests);
> - `docs/09_LOGICAL_CONSISTENCY_ARCHITECTURE.md` (scene consistency);
> - `docs/10_CLAUDE_MD_AUDIT_AND_DECISIONS.md` (audit of `claude.md` and the decisions);
> - `docs/11_EXPRESSIVE_NARRATION.md` (expressive voices).

## P2.1 What changed against Part 1

| Part 1 proposed | What was built, and why |
|---|---|
| Full-AI video for every shot | **Hybrid.** Most shots are drawn by code (cards, app screens, journeys, animated diagrams), plus licensed real images, so their text and numbers are always exact. AI video is used only where a person or place must be seen, and every AI shot is checked against the plan or replaced by a drawn fallback. |
| A laptop control plane + remote GPU workers, a database and an orchestrator | **One Google Colab notebook per video** (free T4) does everything. Google Drive holds the state; every step is cached by content, so after a disconnect, Run all continues where it stopped. The laptop only builds notebooks and runs the tests. |
| A hosted LLM writes scripts and agents plan | **No paid or hosted LLM.** Plans are written as JSON per video, or the **topic notebook** writes them with a local open model (Qwen3-4B-Instruct, Apache-2.0) from Wikipedia facts. The LLM only writes data; fixed, tested code renders it. |
| Paid or closed services in places (TTS, stock media, Modal, …) | **Free and local only.** A test (`test_registry.py`) fails if any engine file imports or calls OpenAI, Anthropic, ElevenLabs, Runway, Replicate, Modal, RunPod, or the Pexels/Pixabay APIs. |
| Generic QC list | **One approval gate.** `qc_report.json` gives PASS/FAIL/UNKNOWN per check with evidence; `approved` is true only when every mandatory check passes. An UNKNOWN blocks approval like a FAIL. |

## P2.2 How it runs (one notebook)

```text
colab/build.py <plan>.json <notebook>.ipynb     (or: build.py --topic colab_topic_to_video.ipynb)
  └─ notebook cells, in order:
     settings → log → install → setup → lib → [topic] → plan → helpers → prefetch → narration → timeline →
     diagrams → keyframes → editlib → logic → bench → clips → music → sfx → render → qc → metadata → preview
```

- **Engine code** lives in real Python files that are unit-tested on the laptop and copied into each notebook:
  `colab/engine/cells/{shared,cast,narrator,topic}/*.py` and `colab/engine/lib/*.py`.
- **Two plan formats:**
  - *narrator*: sentences per shot (older videos);
  - *cast*: characters, lines and tones, plus drawn scenes, scene states and claims (all new videos and the topic
    notebook).
- **Shot recipes:**

  | Recipe | What it is |
  |---|---|
  | `video` | AI video, Wan2.1 / VACE |
  | `card` | text card with an icon and bullets |
  | `ui` | animated phone-app screens |
  | `journey` | stations with a moving token |
  | `diagram` | boxes, arrows and travelling labels |
  | `stock` | licensed real image, fitted over a blurred copy of itself, with a slow camera move and an on-screen credit |

- **Narration-first timeline:** the voice audio decides every shot's length; subtitles are word-timed from the
  speech recogniser.
- **GPU rule:** one model family on the T4 at a time, released between stages:
  1. planner (topic notebook);
  2. voice;
  3. UMT5 text encoder (prompts are cached on Drive);
  4. VACE / Wan video;
  5. checkers (OWLv2 + Qwen3-VL);
  6. repair rounds.

## P2.3 Logical and spatial consistency (2026-10-16)

- **Typed scene states:** shots with a `state` in the plan carry the camera (setup, axis, side), start boxes and
  facing direction, who holds what, and timed events (enter, walk_to, stop, take_out, place, hand_over, look_at,
  gesture, …).
  - These are simulated before any GPU work, with rules X0–X5 and E1–E11: crossing the line, reversals, objects
    teleporting, acting from afar, actions at the cut.
- **Generation:** Wan2.1-VACE-1.3B, steered by a layout video drawn from the plan (coloured boxes moving where the
  plan says).
- **Checks against the SAME plan:**
  - OWLv2 detection;
  - camera motion measured on the background, with planned and detected subjects masked, sub-pixel and anchored;
  - positions, direction, stops, hand-offs and order;
  - Qwen3-VL for actions the boxes can't see.
- **Decisions:** targeted repairs (stronger layout → simpler layout → static camera → new seed → fewer frames) within
  `REPAIR_BUDGET`. Unresolved shots go to REVIEW and are replaced by their drawn fallback; they're never used silently.

## P2.4 Colab T4 reliability and speed (2026-10-16/17)

| Problem | Fix |
|---|---|
| CUDA out of memory (11.8 GB leaked text encoder) | `release_model()` really frees the weights; `expandable_segments`; rounds; an out-of-memory error retries with 25 % fewer frames |
| System RAM restarts (12.7 GB) | models load straight to the GPU (`device_map`); `need_ram()` guards every RAM fallback; `malloc_trim`; VACE prefetched instead of the unused text-to-video model |
| 25-minute clips: turbo failed silently | the Self-Forcing turbo LoRA was verified compatible with VACE (base blocks bit-identical to T2V-1.3B); it is downloaded locally, and its fusion verified, with the error shown on screen; 4 steps instead of 25×2 |
| 10–13 min outside the progress bar | the VAE runs untiled first (tiles only after an out-of-memory error); the all-zero VACE latent is cached (bit-identical, one VAE pass saved per clip); every stage is GPU-timed in `perf.json` |
| False QC rejections | `enter`/`exit` events judged by geometry; camera masks include planned boxes (a missed detection used to fake a 0.2–0.3 camera move; a real slow pan used to read 0.000) |
| No measurements | `BENCHMARK = True` times one shot under each setting on the T4 (`render/bench/benchmark.md`) |

## P2.5 Languages: Telugu (2026-10-17)

- **Voice:** Indic Parler-TTS (ai4bharat, Apache-2.0, gated: needs a free Hugging Face Read token in the Colab secret
  `HF_TOKEN`).
  - It runs in its own environment (it pins transformers 4.46), installed from a pinned lock on top of Colab's own
    torch.
  - The notebook checks the token and the model access before starting, and explains 401 vs 403 errors.
- **Pronunciation:** each line has `text` for the subtitles and `tts` with acronyms and numbers spelled out in Telugu
  ("WL 12" → డబ్ల్యూ ఎల్ పన్నెండు).
- **Text:** the Hind Guntur font (Telugu + Latin) with raqm shaping for every drawn text.
  - Telugu subtitles are drawn with Pillow, because libass detached Telugu vowel signs.
- **Checks:** multilingual Whisper scored by letter error; the validator handles Telugu words, suffixes, hedges and
  time units.
- **Files:** `colab/train_ticket_te_plan.json`, `colab/colab_train_ticket_te_v1.ipynb`,
  `colab/scripts/train_ticket_te_script.md`.

## P2.6 Topic → video, with free local models (2026-10-18, from `claude.md`)

1. **Research:** Wikipedia (Action API, with a User-Agent carrying your contact, `maxlag=5`, cached). Numbered facts
   keep their URL and revision.
   - The HTML extract is used because the plain-text extract turned "2×10³⁰" into "2×1030".
2. **Script:** Qwen3-4B-Instruct writes beats that cite fact IDs (JSON, validated by Pydantic).
3. **Fact check, per sentence:**
   - numbers and names must appear in the cited facts;
   - NLI entailment with `cross-encoder/nli-deberta-v3-base` (picked on a labelled set: 13/14 right, 0 false passes);
   - repeat, unreadable-symbol and too-short gates; one rewrite, then the sentence is dropped;
   - a script that is too thin is **refused**, not shipped.
4. **Images:** Wikimedia Commons, limited to public domain, CC0 and CC BY (CC BY-SA is opt-in; NC, ND and unknown
   licences are refused).
   - SigLIP 2 ranks candidates against the beat's description **and** what the narration says, beating distractors
     and the other beats (calibrated: margin ≥ 1.5).
   - Otherwise the beat becomes a drawn card whose text must match the narration.
5. **Plan:** built by fixed code, then rendered by the same cast cells. Stages are checkpointed in `stages/`.
6. **Result on CPU end to end:** APPROVED. Ducking 10.1 dB, −14.1 LUFS, true peak −1.4 dBTP; a second run reused
   every stage.

**Notebook:** `colab/colab_topic_to_video.ipynb`. Set `TOPIC` and `WIKI_CONTACT`, pick a T4, Run all.

## P2.7 Final quality control and licences (2026-10-18)

- **Mandatory checks:**
  - a playable H.264/AAC stream at the right size and fps; no duration drift; ending not cut;
  - no unplanned black frames; no frozen frames inside AI shots;
  - loudness on target; no clipping; no unintended silence; **measured** ducking;
  - speech heard correctly; AI shots verified; scene-state shots verified or approved; nothing left for review;
  - plan validation clean; music licensed and credited;
  - topic videos: every narrated fact supported, every image licensed and credited.
- A contact sheet is saved for a quick visual review.
- **Subtitles:** word-highlighted ASS ("karaoke": each word lights up as it's spoken; numbers and key terms in the
  accent colour), burned in one final encode.
- **Run modes:** `standard` (1920×1080), `draft` (1280×720), `cpu_fallback` (no AI video).
- **Licence registry** (`colab/engine/lib/registry.py`): every model with its verified weight licence.
  - "review" items: the turbo LoRA repack has no licence tag of its own (upstream Self-Forcing is Apache-2.0);
    OpenAI CLIP's model card restricts deployed use; SDXL is OpenRAIL++.
  - `LICENSE_STRICT` refuses those.

## P2.8 Expressive, scene-aware narration (2026-10-18)

- **Direction:** every line gets a `VoiceDirection` (12 emotions, intensity, pace, pauses before/after, emphasis
  words that must be in the line).
  - It comes from the author's tone label or the line's role in the story (hook, turn, reveal, payoff), never from
    "!" or keywords. A local LLM director is optional.
  - Arc rules then apply: conservative intensity, no sudden jumps, no long dramatic runs, rare peaks.
- **No DSP emotion:** with a direction, emotional pitch shifting, gain and tremolo are off.
- **Backends:**
  - Kokoro (default): follows the pace and real pauses only, and the reports say so.
  - Qwen3-TTS 1.7B CustomVoice: follows the whole direction as an instruction, and runs in its own environment.
- **Measured on CPU** (same speaker, sentence and seed):
  - suspense came out slower and lower (5.76 s vs 3.76 s);
  - excitement came out higher and louder, but changed "But" to "Well";
  - strong styles shifted the narrator's timbre.
  - Hence `EXPRESSIVE_MAX_INTENSITY = 0.5`, calmer retakes, and an ECAPA leave-one-out identity check (Kokoro 0.80
    calibrated; Qwen and Parler 0.72 provisional).
- **Voice QC:** words (letter-aware, units normalised), pace in syllables per second, gaps, clipping, pitch as a
  diagnostic only, narrator identity.
- **`VOICE_BENCH = True`:** neutral vs expressive per backend, blind A/B, reports. Choose the backend from that.

## P2.9 Videos built

| Notebook | Plan | Notes |
|---|---|---|
| `colab_make_video_v15.ipynb` | `black_holes_plan.json` | first full-AI explainer |
| `colab_story_v3.ipynb` | `story_plan.json` | story with characters |
| `colab_zomato_order_v1.ipynb` | `zomato_order_plan.json` | food-order explainer |
| `colab_food_backend_v1.ipynb` | `food_order_backend_plan.json` | system-design diagrams |
| `colab_amazon_refund_v1.ipynb` | `amazon_refund_plan.json` | scene-state AI shots |
| `colab_train_ticket_v1.ipynb` | `train_ticket_plan.json` | IRCTC, Riya, 14 shots |
| `colab_train_ticket_te_v1.ipynb` | `train_ticket_te_plan.json` | the same story, in Telugu |
| `colab_three_d_movies_v1.ipynb` | `three_d_movies_plan.json` | how 3D glasses work: Nina, 15 shots, 91 s, 10 Wikipedia-sourced claims; script in `colab/scripts/three_d_movies_script.md` |
| `colab_topic_to_video.ipynb` | made at run time | any topic |

## P2.10 Models (all downloaded automatically; only one needs approval)

| Use | Model | Licence |
|---|---|---|
| AI video / layout control | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `Wan-AI/Wan2.1-VACE-1.3B-diffusers` (+ the UMT5 encoder from `Wan-AI/Wan2.2-TI2V-5B-Diffusers`) | Apache-2.0 |
| Turbo (4 steps) | `Kijai/WanVideo_comfy` (Self-Forcing LoRA) | review: see P2.7 |
| Scene checks | `google/owlv2-base-patch16-ensemble`, `Qwen/Qwen3-VL-2B-Instruct` | Apache-2.0 |
| Voices | `hexgrad/Kokoro-82M`; `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`; `ai4bharat/indic-parler-tts` (**gated**: accept the terms) | Apache-2.0 |
| Speech check | `Systran/faster-whisper-small.en`, `dropbox-dash/faster-whisper-large-v3-turbo` | MIT |
| Narrator identity | `speechbrain/spkrec-ecapa-voxceleb` | Apache-2.0 |
| Topic planner / fact check / images | `Qwen/Qwen3-4B-Instruct-2507`, `cross-encoder/nli-deberta-v3-base`, `google/siglip2-base-patch16-224` | Apache-2.0 |

## P2.11 Tests (laptop, CPU)

`colab/tests/`: 12 suites, **252 checks, all passing** on 2026-10-18.

| Suite | Checks | Suite | Checks |
|---|---|---|---|
| consistency | 62 | lib | 31 |
| logic loop | 27 | speed | 25 |
| Telugu | 25 | voice | 24 |
| planner | 16 | QC | 14 |
| assets | 8 | subtitles | 7 |
| registry | 7 | voice backend | 6 |

End-to-end CPU runs (Python 3.13, FFmpeg 4.4) pass for the train-ticket (English and Telugu), Amazon, topic and 3D
notebooks.

## P2.12 Still open (needs a Colab T4 or people)

- **Measured T4 numbers:** run `BENCHMARK = True` (video) and `VOICE_BENCH = True` (voices) once.
- **Real runs not yet seen:** the 4B planner's script quality and the VACE + turbo visual quality.
- **Recalibration:** the SigLIP 2 and Qwen-voice identity thresholds, which were set on small samples.
- **Listening:** blind listening for the expressive voice. Retention can only be judged from YouTube analytics.
- **Deferred:** Pexels/Pixabay (API keys, not open licences), FLUX.2 klein (T4 memory), ACE-Step music, CosyVoice,
  Chatterbox, and a vertical (Shorts) build.

---

# Part 3 (2026-10-18): Two-friend cinematic storytelling

**What it is:** two recurring friends discuss the topic:
- **Arjun** is curious and asks.
- **Meera** knows and shows: the finger test, two eyes → two pictures → depth, projector → polarised glasses.

**How it is built:**
- They are original 2D characters drawn by code (`engine/lib/charanim.py`) from a Character Bible (`characters.py`).
- They breathe, blink and gesture. They react, and their mouths open with their own voice.
- The dialogue comes from the Story Director (`story.py`): beats → turns → visuals. Every factual turn cites a Wikipedia fact and must pass the NLI check, otherwise it is removed.
- The Visual Story Director picks each visual. Its order is: licensed media, a demonstration, AI video (budgeted), a minimal overlay, and otherwise the friends themselves. Re-enactments are labelled "Illustration".
- Each scene has its own visual timeline, so a cut can happen in the middle of a line.

**Free and licensed:** strict licence mode (default for these videos) refuses the turbo LoRA, SDXL and OpenAI CLIP. Model revisions are recorded.

**Sample:** `colab/samples/three_d_friends_draft_720p.mp4`, "Why do 3D movies look real?":
- 89 s, approved by the QC gate.
- Made on a CPU in about 3.5 min, and about 25 s to re-run from the cache.
- Peak RAM 1.4 GB.
- Screen time: 73 % the friends, 27 % demonstrations, 0 % slides.

**Tests:** 306 offline checks + 1 online check, all pass.

**Still open:**
- GPU runs.
- The 4B model's dialogue quality.
- Real lip sync.
- Telugu automatic stories.
- AI cutaways inside scenes.
- Viewer judgement.

Details: `docs/12_TWO_FRIEND_STORYTELLING.md`.

---

# Part 4 (2026-10-18): Natural voices and stronger stories

**What was wrong (measured):**
- Arjun and Meera were directed like a narrator.
- Every reply came 0.45–1.2 s late. Real conversation is ~0.2 s (Stivers et al., 2009).
- One friend's mood was smoothed into the other's.
- Kokoro can only follow pace and pauses.
- The stories answered every question immediately; Arjun never guessed or tested anything, and the ending was a recap.

**What changed:**
- **Stories:** concept first (2 premises, chosen by rules), then an outline where each step causes the next. Then dialogue written with a memory of what has been said. Then a critic (14 rules), a targeted rewrite, and a fact repair that never leaves a question without its answer.
- **Voices:** every line has a reason ("guiding the demo", "quiet realisation", "satisfying revelation" …) and its timing follows from it. Replies come in 0.1–0.6 s; trying a demonstration gets ~0.8 s.
  - Hero lines get two takes on Qwen/Parler, chosen by checks and kept for A/B.
  - The identity check now depends on clip length, calibrated on 37 lines.
- **New sample** (`colab/samples/v2/three_d_friends_v2_draft_720p.mp4`, 78 s, approved): the one-eye test leads to the answer.

**Models:** Kokoro stays the default.
- Qwen3-TTS has only male native-English presets, so it can be Arjun's voice once tested on a T4.
- Chatterbox (MIT) is recorded but not integrated (heavy pins, needs reference audio).
- Parler stays for Telugu.

**Tests:** 348 offline checks + 1 online check, all pass.

**Not done:**
- Human listening.
- GPU runs.
- The real 4B story benchmark.
- The Telugu voice benchmark (needs the Colab token).

Details: `docs/13_NATURAL_VOICES_AND_STORYTELLING.md`.

---

# Part 5 (2026-10-19): Natural Telugu + Gemini free-tier voices (3D glasses video)

**Goal:** Telugu videos where Arjun and Meera sound like two real Telugu friends.

**What changed (only the 3D glasses Telugu video):**
- **New Telugu script, written natively** (not translated): the one-eye test, the popcorn callback.
  - A Telugu critic found 11 translation artefacts in the old script (bookish words like చిత్రాలు/దృశ్యం, calques like "సైన్స్ కోసం") and 0 in the new one.
  - Native speakers still need to judge.
- **Facts:** each factual Telugu line keeps its English claim, which is fact-checked (9/9); the Telugu wording is checked to keep the key terms (5/5). Anything uncertain goes to manual review.
- **Optional Google Gemini free-tier TTS** (`gemini-3.8-flash-tts`), the one allowed hosted exception, for speech only:
  - verified against Google's docs;
  - fail-closed free-tier preflight;
  - ≤ 6 calls per video;
  - 429 handled;
  - key never logged;
  - only dialogue text sent.
- **Two-speaker requests** are split back into per-turn audio at real pauses. Tested on real speech: 0.08 s accuracy.
- **Fallback:** anything uncertain → local Indic Parler for the whole video.

**Tests:** 49 new checks, plus end-to-end runs with a stand-in Gemini (4 calls, all turns aligned) and without a key (fallback to Parler).

**Not done:**
- A real Gemini call (needs your key).
- The voice audition and the Parler-vs-Gemini benchmark.
- Native-speaker listening.

Details: `docs/14_TELUGU_GEMINI_TTS.md`.

---

# Part 6 (2026-10-19): Architecture validated

**What was checked:** every notebook as one program, every setting, the run logs, every path end to end, and 402 tests.

**Five real problems found and fixed:**
1. Gemini lines could trigger extra API calls and switch off the voice-identity check.
2. A story with no facts could become a video.
3. `STORY_BENCH` was promised in the docs but missing.
4. The 120-second limit was not enforced.
5. The topic notebook's new story writer had never been run end to end.

**Result:** all 402 checks pass.

**Still recommended:**
- Split the large voice cell into library modules.
- Optionally cache the 8 GB planner on Drive.
- Calibrate thresholds and measure on a real T4 with real Gemini audio.

Details: `docs/15_ARCHITECTURE_VALIDATION.md`.

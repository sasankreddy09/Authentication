> Historical research plan (2026-10-11). For implemented behavior, use [Current architecture](02_ARCHITECTURE.md) and [Colab setup](GOOGLE_COLAB.md).

# AI Video Automation: Master Plan

**Date:** 2026-10-11 · **Phase:** 1 complete (research and architecture). No code, credentials, cloud resources or model downloads yet.

**Detail documents:**

| Doc | Content |
|---|---|
| `01_RESEARCH_REPORT.md` | Models, tools, licences, sources |
| `02_ARCHITECTURE.md` | Components, schemas, checkpoints, security |
| `03_TECHNOLOGY_COMPARISON.md` | Scores, stacks, the three strategies |
| `04_EXECUTION_STRATEGY.md` | Ubuntu/Colab, routing, runtimes, costs, config |
| `05_IMPLEMENTATION_ROADMAP.md` | Milestones and acceptance criteria |
| `06_EXAMPLE_VIDEO_WORKFLOW.md` | Pyramids documentary, end to end |
| `07_YOUTUBE_AUTOMATION.md` | OAuth, uploads, quota, policy |

**Labels:** [O] verified from official sources · [C] community · [est] estimate · [UNVERIFIED] check before relying on it.

---

## 1. Bottom line

1. **It is feasible**, with clear limits:
   - Open models generate **short shots (≈ 5 s)**, not long videos. The system therefore plans a documentary as **30–40 shots**, generates each one separately, and **edits** them together on a narration-driven timeline.
   - Long continuous AI video remains experimental.
2. **The LLM produces data, not running code.**
   - The AI director writes **validated JSON**: brief, research with sources, script, storyboard with camera directions, and production plan.
   - Fixed, tested Python turns that JSON into media.
   - **There is no run-time code generation.** Variety comes from a small **recipe menu**, about 9 named and tested effects to start, grown only when real plans ask for more.
   - When the planner wants an effect the menu lacks, it falls back to the closest recipe and logs a "wish". **Unsupported ideas never block a video** (doc 02 §9).
3. **Your laptop is the control plane.** It has an i5-1335U, 7.4 GB RAM and no NVIDIA GPU. It plans, narrates (Kokoro on CPU), edits (FFmpeg), checks and publishes.
   - **GPU work runs as job files** on Colab (prototype) or a rented GPU (production).
   - YouTube credentials never leave the laptop.
4. **The visual strategy is a mix, not pure text-to-video:**
   - ~30–40% **image-to-video** hero shots (Wan 2.2 TI2V-5B, Apache-2.0).
   - ~45–55% **AI keyframes with deterministic camera moves** (Z-Image-Turbo / FLUX.2 klein 4B + FFmpeg).
   - ~10–20% **code-drawn graphics** for anything that must be exact: maps, dates, numbers, labels.
   - Every I2V shot falls back to a camera move on its keyframe, so a missing GPU never blocks a video.
5. **Every default component is commercially usable**, under Apache-2.0, MIT, CC0 or libraries with similar terms.
   - Models that are excluded in some territories (HunyuanVideo, MiniMax H3) are off by default.
   - Non-commercial models are blocked by default: FLUX-dev, Qwen-Image-2.1, MusicGen, XTTS and others.
6. **YouTube:** uploads are **private by default**, set `containsSyntheticMedia=true`, and wait for explicit human approval. The spam and "inauthentic content" policies target mass-produced AI videos, so the system is built for **researched, varied, reviewed** videos at low volume.
7. **Recommendation:**
   - **Prototype** on **Strategy B** (laptop + Colab, ~$0–2 per video).
   - **Production** on **Strategy C** (laptop + rented GPU, ~$3–10 per 3-min video).
   - Same codebase for both.

---

## 2. Recommended architecture

```text
┌──────────────────────── LOCAL UBUNTU (control plane) ─────────────────────────┐
│ main.py ─► Stage runner (plain Python, hash checkpoints, JSONL logs)         │
│   PLAN (LLM, JSON schemas): Director → Researcher → Scriptwriter →            │
│                             Storyboard (per section) → Validator + Critic     │
│   VOICE: Kokoro TTS per sentence → faster-whisper WER + word timestamps       │
│          → lock every shot's duration to the narration                        │
│   ROUTE: hardware.json + models.yaml → local | remote | fallback              │
│   CPU WORK: graphics, camera moves, edit, mix, loudness, subtitles, encode    │
│   QC: technical + speech + subtitle + duplicate + (optional) vision review    │
│   PUBLISH: metadata + thumbnail → human approval → YouTube API (private)      │
└──────────────┬────────────────────────────────────────────────────────────────┘
               │ job JSON + inputs ⇄ results (rclone ⇄ Google Drive  |  S3 later)
┌──────────────▼──────────── GPU WORKER (Colab notebook / rented GPU) ──────────┐
│ detect GPU → choose adapters → keyframes (Z-Image / FLUX.2 klein / SDXL)      │
│ → I2V (Wan 2.2 TI2V-5B; A14B + LightX2V on big GPUs) → [music, upscale]       │
│ → outputs + sidecars + heartbeat                                              │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Agents are roles, not chatbots.**
- Five LLM stages:
  - Director
  - Researcher
  - Scriptwriter
  - Storyboard/cinematographer
  - Critic, plus the metadata writer.
- The rest is deterministic code:
  - resource planner
  - visual and audio generators
  - editor
  - quality controller
  - publisher.
- No agent framework is needed. LangGraph, CrewAI, AutoGen (now in maintenance mode) and others are compared in doc 03 §6.

**Intermediate format:** a project folder holding JSON documents with `schema_version`. Every media file has a sidecar recording model, revision, prompt, seed, input hash and timing. The hand-off contract is `production_plan.json`. See doc 02 §5 and the filled example scene in doc 06 §7.

---

## 3. Production workflow

```text
topic ─► brief ─► research (claims + sources + confidence) ─► script ─► storyboard
      ─► validate + critic ─► production_plan.json
      ─► narration (per sentence) ─► ASR check ─► word alignment ─► lock shot timing
      ─► route jobs ─┬─ local: graphics
                     ├─ remote GPU: keyframes ─► I2V clips   (fallback: still_motion)
                     └─ library/remote: music, SFX
      ─► asset QC (regenerate only failures)
      ─► shot segments (camera moves, normalize) ─► join + transitions + grade
      ─► mix (ducking, −14 LUFS) ─► subtitles ─► final MP4
      ─► final QC ─► metadata + thumbnail ─► YOU review + approve
      ─► YouTube resumable upload (private, synthetic-media flag) ─► publish_record.json
```

- **Per-step reference:** inputs, outputs, libraries, hardware, time, failure modes and validation are in doc 04 §6.
- **Resume:** hash-based skip at the level of individual shots (doc 02 §6.2).

---

## 4. Selected technologies

| Function | Prototype (Strategy B) | Production (Strategy C) | Licence |
|---|---|---|---|
| Planning LLM | Hosted API with JSON-schema output (behind an `LLMProvider` interface) | Same, or Qwen3-class via vLLM | API terms / Apache-2.0 |
| Research | LLM web search + fetch; sources stored | + PydanticAI tool loop if needed | — |
| Narration | **Kokoro-82M** (CPU) | **Qwen3-TTS 1.7B** (GPU) | Apache-2.0 |
| ASR / alignment | **faster-whisper** | + WhisperX | MIT / BSD-2 |
| Keyframes | **Z-Image-Turbo**, **FLUX.2 klein 4B** (SDXL on T4) | + Qwen-Image-Edit-2511, character LoRAs (ai-toolkit) | Apache-2.0 / OpenRAIL++ / MIT |
| Image-to-video | **Wan 2.2 TI2V-5B** (1280×704, 24 fps, 5 s) | **Wan 2.2 I2V-A14B + LightX2V 4-step**, Wan 2.1 FLF2V | Apache-2.0 |
| Camera moves on stills | **FFmpeg zoompan/crop** (supersampled) | + Blender for 3D explainers | LGPL/GPL |
| Graphics | Pillow, Matplotlib, Natural Earth, as recipes (`title_card`, `map_route`, `comparison_panels`, `timeline_graphic`) | + Blender templates for 3D explainers, added when the backlog asks | permissive |
| Music | Curated **CC0 / royalty-free library** | **ACE-Step 1.5** | MIT |
| SFX | **Freesound CC0** pack | + Stable Audio Open (< $1M revenue) | CC0 / Community |
| Interpolation / upscale | — (TI2V-5B is native 24 fps) | RIFE, SeedVR2 | MIT / Apache-2.0 |
| Edit / mix / encode | **FFmpeg** (xfade, lut3d, sidechaincompress, loudnorm, libx264/AAC) | Same; NVENC for drafts | LGPL/GPL |
| QC | ffprobe, detect filters, faster-whisper, imagehash, PySceneDetect | + vision-LLM review, DOVER | permissive |
| Orchestration | Plain Python stage runner | + DBOS when running several projects concurrently | MIT |
| Transfer | rclone ⇄ Google Drive | S3/R2 bucket | MIT |
| GPU workflow runner | diffusers adapters | + **pinned community ComfyUI workflows** (ComfyUI, custom-node and model hashes locked) for Wan I2V/FLF2V/SeedVR2 | Apache-2.0 / GPL-3.0 (separate tool) |
| Timeline export (optional) | — | OpenTimelineIO (Apache-2.0) or MLT XML (LGPL) for manual touch-ups; avoid Remotion unless its company licence fits | — |
| Publishing | google-api-python-client 2.201, google-auth-oauthlib 1.5 | Same | Apache-2.0 |

**Blocked by policy** (`allow_non_commercial: false`):
- FLUX.1-dev / FLUX.2-dev / klein-9B
- Qwen-Image-2.1
- MusicGen, MMAudio, AudioLDM2
- XTTS-v2, F5-TTS, Fish S2
- InsightFace-based face tools

**Off by default, behind an explicit legal-review flag:**
- HunyuanVideo 1.5 and HunyuanVideo-Foley (EU/UK/KR excluded)
- MiniMax H3 (also excludes the USA)
- LTX-2.x (revenue cap)

---

## 5. Three strategies (summary of doc 03 §9)

| | A — Fully local | **B — Laptop + Colab** | **C — Laptop + rented GPU** |
|---|---|---|---|
| Quality | Current laptop: stills/graphics only. With a 24 GB GPU: medium-high | Medium-high | **High** |
| Up-front cost | $0, or $1.5–3.5k for a GPU box | **$0** | $0 |
| Per 3-min video [est] | ~$0.1–0.5 (LLM) | **~$1–2** | **~$3–10** |
| Wall clock [est] | 0.5–1 h (no AI video) / 1–3 h (24 GB GPU) | 1.5–4 h | 0.5–1.5 h |
| Reliability | High | Medium-low (GPU not guaranteed) | High |
| Complexity | Low | Medium | Medium-high |
| Scales | 1 GPU | 1 session | N GPUs |
| **Use for** | Offline / explainer mode | **Prototype** | **Production** |

---

## 6. Implementation order

| Step | Milestone | Proves | Effort [est, part-time] |
|---|---|---|---|
| 0 | Phase 2 design freeze: schemas, interfaces, settings, a 45 s fixture plan | Contracts are stable | 2–3 days |
| 1 | **M1** Skeleton renderer: fixture plan → MP4 with placeholders | FFmpeg spine, checkpoints, `doctor` | 1 week |
| 2 | **M2** Planning stages | LLM → valid plans with sources | 1 week |
| 3 | **M3** Local audio: Kokoro, alignment, timing lock, mix, subtitles | **First real 45 s video** (graphics + voice) | 1 week |
| 4 | **M4** Colab worker: keyframes + I2V over Drive; benchmarks | AI visuals + graceful fallback | 1–2 weeks |
| 5 | **M5** QC + resume + cleanup | Only failures regenerate | 1 week |
| 6 | **M6** YouTube private upload | Secure OAuth, resumable, idempotent | 3–5 days |
| 7 | **M7** 3-minute pyramids documentary | Full system | 1 week |
| 8 | Phase 4: rented GPU, A14B, ACE-Step, Qwen3-TTS, Shorts, DBOS, regression suite | Production quality and volume | ongoing |

**Prototype definition** (doc 05 §2):
- 30–60 s video, mostly local.
- Graphics + AI keyframes + 0–3 I2V shots.
- Kokoro narration, library music, FFmpeg render, automated QC.
- Completes even with no GPU.

---

## 7. Execution commands (target UX)

```bash
# setup (once)
sudo apt install ffmpeg rclone
uv sync --group core
python main.py doctor
rclone config                       # Drive remote for the Colab worker
python main.py auth youtube         # when you reach M6 (see doc 07 §3)

# make a video
python main.py create --topic "How black holes are formed" --duration 180 \
  --style cinematic --voice english --format youtube
#   then open notebooks/colab_worker.ipynb in Colab (GPU runtime) → Run all
python main.py status  --project projects/black_holes
python main.py resume  --project projects/black_holes
python main.py review  --project projects/black_holes
python main.py approve --project projects/black_holes

# publish (separate, explicit)
python main.py publish --project projects/black_holes --privacy private
python main.py publish --project projects/black_holes --privacy public --confirm-public   # later, after you watch it
python main.py clean   --project projects/black_holes
```

---

## 8. Proposed repository structure (improved)

```text
ai-video-studio/
├── main.py                         # CLI: doctor, create, status, resume, redo, review, approve, publish, clean, auth
├── pyproject.toml  uv.lock         # pinned; dependency groups: core (laptop), worker (GPU), dev
├── .env.example                    # LLM_API_KEY=, LLM_MODEL=  (no real values)
├── .gitignore  .pre-commit-config.yaml   # gitleaks, ruff
├── README.md
├── config/
│   ├── settings.yaml               # formats, routing, QC thresholds, publishing defaults
│   ├── recipe_fallbacks.yaml       # unknown/unsupported effect → closest recipe
│   ├── models.yaml                 # adapters: repo, pinned revision, licence, commercial_ok, VRAM, defaults
│   ├── styles/cinematic_documentary.yaml  …   # prompt suffix, negatives, grade, pacing, transitions
│   └── pronunciations.yaml
├── schemas/                        # Pydantic models = single source of truth (+ exported JSON Schema)
│   ├── brief.py research.py script.py storyboard.py plan.py timeline.py job.py qc.py publish.py
├── llm/
│   ├── provider.py                 # LLMProvider interface: hosted / ollama / vllm backends
│   └── prompts/<stage>.v1.md       # versioned prompt templates
├── agents/                         # LLM stages (pure functions: inputs → JSON)
│   ├── director.py researcher.py scriptwriter.py storyboard.py critic.py metadata_writer.py
├── generators/                     # local (CPU) generators
│   ├── speech.py alignment.py graphics.py music.py sfx.py subtitles.py
├── worker/                         # GPU side — same package on Colab, rented GPU or a local GPU
│   ├── main.py                     # queue loop: claim → run → write outputs + sidecar → heartbeat
│   ├── gpu.py                      # detect GPU, pick adapters that fit
│   ├── adapters/  t2i_z_image.py t2i_flux2_klein.py t2i_sdxl.py i2v_wan22_5b.py comfy_workflow.py music_ace_step.py
│   └── workflows/ <name>@<ver>.json + <name>.lock.yaml   # pinned community ComfyUI workflows (API format)
├── rendering/
│   ├── timeline.py                 # plan + words → timeline.json (EDL)
│   ├── recipes/                    # registry + ~9 starter recipes, each with params model, version, golden test, fallback
│   ├── camera.py                   # still_motion moves (zoompan/crop expressions)
│   ├── shots.py transitions.py grading.py audio_mix.py renderer.py
│   └── luts/*.cube
├── qc/
│   ├── asset_checks.py final_checks.py report.py
├── publishing/
│   ├── youtube.py auth.py metadata.py thumbnail.py
├── orchestration/
│   ├── pipeline.py                 # stage graph + runner
│   ├── checkpoint.py               # input hashing, sidecars, skip logic
│   ├── router.py hardware.py remote.py (rclone/S3 transport) cleanup.py project.py
├── notebooks/
│   └── colab_worker.ipynb          # thin: mount Drive, pip install worker group, run worker.main
├── assets/                         # licensed library: music/, sfx/, fonts/, maps/ + licenses.json
├── projects/                       # one folder per video (git-ignored)
├── tests/
│   ├── fixtures/plan_45s.json
│   ├── test_schemas.py test_timeline.py test_checkpoint.py test_router.py
│   ├── test_render_smoke.py        # renders the 45 s fixture in draft profile
│   └── test_publish_idempotency.py # mocked YouTube API
└── docs/                           # these documents + benchmarks.md
```

**What changed from the brief's structure, and why:**
- **`schemas/`** is separate, because the JSON contracts are the backbone of the system.
- **`worker/`** is split from `generators/` because GPU code runs on a different machine, with different dependencies.
- **`qc/`** is its own package. There is no `sandbox/`, because nothing generated is executed.
- **`rendering/recipes/`** holds the effect vocabulary, and **`worker/workflows/`** holds pinned ComfyUI workflows.
- **`recipe_backlog.jsonl`** at the repo root collects unsupported requests across projects (`main.py backlog`).
- **`llm/prompts`** are versioned.
- **`assets/licenses.json`** is added for auditable music and SFX licensing.
- **`notebooks/`** is kept thin, so the notebook never diverges from the tested worker code.

---

## 9. Expected costs

| Item | Cost [est] |
|---|---|
| Laptop control plane | $0 |
| LLM API per 3-min video (planning, research, critic, metadata) | ~$0.10–1 (depends on model and research depth) |
| Colab GPU per 3-min video (L4/A100) | ~$0.40–0.90 in compute units |
| Rented GPU per 3-min video (Strategy C) | ~$3–10 |
| Monthly at 4 videos (B / C) | ~$7–15 / ~$20–50 |
| Optional 24 GB GPU workstation | ~$1,500–3,500 one-off |
| YouTube API | Free (quota-based) |

GPU prices and Colab compute-unit rates change often. Re-check them before budgeting.

---

## 10. Major technical risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **Colab gives no or small GPUs; sessions drop** | High | Delays | Graceful fallback to still_motion; resumable job queue; move to rented GPUs (Strategy C) |
| 2 | **AI clip quality** (warping, hands, crowds, flicker) | High | Visual quality | Short clips; I2V from approved keyframes; QC + reseed; still_motion fallback; avoid close-up hands/faces in prompts |
| 3 | **Factual errors / hallucinated history** | Medium | Credibility, policy | Research stage with sources; claim-linked narration; hedging for contested claims; graphics drawn by code; human review |
| 4 | **Licence changes or misread licences** | Medium | Legal | `models.yaml` licence + check date; the loader refuses non-commercial adapters; monthly re-check; legal review before monetising |
| 5 | **YouTube policy** (inauthentic/mass-produced AI content) | Medium | Demonetisation / strikes | Low volume, varied styles and voices, research depth, disclosure flag, human approval |
| 6 | **Laptop RAM (7.4 GB) during renders** | Medium | Crashes | Per-shot rendering, segmented concat, thread limits; render on the worker if needed |
| 7 | **Credential leakage** | Low | High | Tokens only in `~/.config` (mode 600) / keyring; gitleaks; never on Colab; log redaction |
| 8 | **Unverified runtime numbers** (no official T4/L4 Wan 2.2 timings) | High | Planning | Benchmark in M4 before fixing budgets |
| 9 | **Plans want effects the menu lacks** | High early on | Low | Closest-recipe fallback + wish backlog; add recipes only when wishes recur |
| 13 | **Community ComfyUI workflows/nodes change or break** | Medium | Medium | Lock ComfyUI, node and model hashes; golden render per workflow; diffusers adapter as fallback |
| 10 | **Drive transfer quotas / >10k-items limits** | Medium | Stalls | Per-project folders, few large files, backoff; S3 in production |
| 11 | **API docs conflict on private-upload restriction** | — | Low | Private default anyway; test once; audit only if needed |
| 12 | **Model churn** (new versions every few months) | High | Maintenance | Adapter interface + pinned revisions; regression suite on golden plans |

---

## 11. Verified vs experimental

- **Build on these (verified, mature):**
  - LLM structured plans
  - Kokoro / faster-whisper on CPU
  - FFmpeg editing, mixing, loudness and encoding
  - Still keyframes with camera moves
  - YouTube resumable private uploads with the synthetic-media flag
  - Hash-based resume
- **Usable with care:**
  - Wan 2.2 TI2V-5B on L4/A100
  - A14B + LightX2V on 48–80 GB GPUs
  - Character LoRAs, SeedVR2, ACE-Step
  - Vision-LLM QC
- **Experimental:**
  - Wan on a Colab T4
  - Long continuous shots (LongCat, SkyReels, FramePack)
  - Kandinsky 6 / LTX-2.5 native audio
  - VACE/Phantom reference-to-video
  - Fully open-source planning on a CPU-only laptop

---

## 12. Next step

**Phase 2 (prototype design freeze).** Write the Pydantic schemas, adapter interfaces, `settings.yaml`/`models.yaml`, the cinematic style preset and a hand-written 45 s fixture plan. Then start M1.

No credentials, GPU installs or model downloads are needed until M2 (LLM key) and M4 (Colab).

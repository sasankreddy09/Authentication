# 03 — Technology Comparison and Recommended Stack

**Date:** 2026-10-11

**Basis:** `01_RESEARCH_REPORT.md` (facts and sources).

**Scoring method:**
- Each criterion is scored **1–5** (5 = best). These are judgement scores based on the verified facts in doc 01. They are **not** benchmark results.
- **Weights:**
  - Quality 25%
  - Licence safety 20%
  - Hardware fit (Colab L4 / 24 GB) 20%
  - Automation ease (diffusers/CLI, deterministic) 15%
  - Maturity / maintenance 10%
  - Speed 10%
- A licence that is non-commercial or excludes territories caps the total at "✖ not eligible", whatever the other scores are.

---

## 1. Video generation (image-to-video and text-to-video)

| Model | Quality | Licence | HW fit | Automation | Maturity | Speed | **Weighted** | Role |
|---|---|---|---|---|---|---|---|---|
| **Wan 2.2 TI2V-5B** | 4 | 5 | 4 | 5 | 5 | 3 | **4.35** | **Default I2V/T2V** (L4/A100/24 GB) |
| **Wan 2.2 A14B + LightX2V 4-step** | 5 | 5 | 2 (80 GB official; GGUF 16–24 GB) | 4 | 4 | 3 | **3.95** | **Quality tier** on rented 48–80 GB GPUs |
| Wan 2.1 1.3B / FLF2V-14B / VACE | 3 | 5 | 4 (1.3B: 8 GB) | 4 | 5 | 3 | 3.95 | Transitions (FLF2V), low-VRAM drafts |
| Kandinsky 6.0 Lite | 3–4 [UNVERIFIED] | 5 | 3 | 4 | 2 | 3 | 3.58 | Watch-list; native audio |
| LTX-2.5 | 4–5 | 3 (< $10M revenue) | 2 (~66 GiB files; FP8 + offload) | 4 | 4 | 4 | 3.53 | Optional quality/audio tier, if the licence fits |
| LongCat-Video | 4 | 5 | 2 | 3 | 3 | 2 | 3.35 | Experimental long shots |
| CogVideoX 1.5-5B | 3 | 2 | 4 | 4 | 3 | 3 | 3.15 | ✖ superseded |
| HunyuanVideo 1.5 | 4 | **✖ territory** | 4 | 4 | 4 | 4 | not eligible | Only if you and your audience are outside the EU/UK/KR, after legal review |
| MiniMax H3 | 5 | **✖ territory (incl. USA)** | 1 | 3 | 3 | 2 | not eligible | — |
| FramePack | 3 | ✖ likely Hunyuan terms | 5 | 3 | 3 | 2 | not eligible [UNVERIFIED] | — |
| MAGI-2 | — | 5 | 1 (8× H100) | 2 | 2 | 1 | — | Impractical |

**Decision:**
- **Wan 2.2 TI2V-5B** is the default video adapter. It produces 1280×704 at 24 fps, 5 s per clip, and the frame rate matches a 24 fps timeline without conversion.
- **Wan 2.2 I2V-A14B + LightX2V** is the upgrade when a 48–80 GB GPU is rented.
- **Wan 2.1 FLF2V** handles designed transitions.

## 2. Image generation (keyframes, thumbnails)

| Model | Quality | Licence | HW fit | Automation | Maturity | Speed | **Weighted** | Role |
|---|---|---|---|---|---|---|---|---|
| **Z-Image-Turbo** | 4.5 | 5 | 4 (16 GB) | 5 | 4 | 5 | **4.58** | **Default keyframes** |
| **FLUX.2 klein 4B** | 4 | 5 | 5 (~13 GB) | 5 | 4 | 5 | **4.65** | Default alternative + multi-reference edits |
| Qwen-Image-2512 / Edit-2511 | 5 | 5 | 2 (~24 GB fp8; 58 GB repo) | 4 | 4 | 2 | 3.85 | Character sheets, edits, thumbnail text (quality tier) |
| FLUX.1-schnell | 3.5 | 5 | 3 | 5 | 5 | 5 | 4.23 | Fallback |
| SDXL (+ LoRAs) | 3 | 4 | 5 (T4 OK) | 5 | 5 | 4 | 4.20 | **T4 fallback** |
| SD 3.5 Large | 4 | 3 (< $1M revenue) | 3 | 5 | 4 | 3 | 3.65 | Optional |
| FLUX.1-dev / FLUX.2-dev / Qwen-Image-2.1 | 5 | **✖ NC** | — | — | — | — | not eligible | — |

**Decision:**
- **Z-Image-Turbo** is the default keyframe model (better photorealism [C]) and **FLUX.2 klein 4B** is the alternative. Their scores are within rounding (klein wins on VRAM); benchmark both on 20 documentary prompts in M4 and keep the better one.
- **SDXL** is the T4 fallback.
- **Qwen-Image-Edit-2511** is used when a character sheet or precise edit is needed.

## 3. Text-to-speech

| Model | Quality | Licence | HW fit (laptop CPU) | Automation | Maturity | Speed | **Weighted** | Role |
|---|---|---|---|---|---|---|---|---|
| **Kokoro-82M** | 4 | 5 | **5** | 5 | 4 | 5 | **4.65** | **Default narrator** (laptop CPU) |
| **Qwen3-TTS 1.7B** | 5 | 5 | 2 (GPU) | 4 | 4 | 3 | 3.95 | **Quality upgrade** on Colab; style instructions |
| Chatterbox (Turbo / Multilingual / Nano) | 4.5 | 5 (watermarked) | 3 (Nano on CPU) | 4 | 4 | 4 | 4.13 | Alternative, multilingual |
| Kyutai Pocket TTS | 3.5 | 4 (CC-BY) | 5 | 4 | 3 | 4 | 3.98 | CPU alternative |
| Piper | 3 | 3 (GPL + per-voice) | 5 | 5 | 4 | 5 | 4.00 | Fallback |
| CosyVoice 3 | 4 | 5 | 2 | 3 | 4 | 3 | 3.55 | Multilingual / dialects |
| XTTS-v2, F5-TTS, Fish S2, Voxtral TTS, Higgs v3 | — | **✖ NC** | — | — | — | — | not eligible | — |

## 4. Music and SFX

| Option | Quality | Licence | HW fit | Automation | **Verdict** |
|---|---|---|---|---|---|
| **Curated CC0 / royalty-free library** | 4 (human-made) | 5 (when logged) | 5 | 5 | **Prototype default** |
| **ACE-Step 1.5** | 4 | 5 (MIT) | 5 (< 4 GB) | 4 | **Generation default (Phase 4)** |
| DiffRhythm 2 | 3.5 | 5 | 4 | 3 | Alternative |
| Stable Audio Open / SA 3.0 Small | 3.5 | 3 (< $1M) | 4 | 4 | SFX and stingers under the cap |
| Freesound CC0 SFX | 4 | 5 | 5 | 4 (API) | **SFX default** |
| MusicGen, MMAudio, AudioLDM2, Tango 2, ThinkSound | — | ✖ NC | — | — | Not eligible |

## 5. Composition, rendering, QC

| Need | Choice | Runner-up | Rejected (why) |
|---|---|---|---|
| Renderer | **FFmpeg 8.x/9.x CLI**, driven by the fixed executor's recipes | typed-ffmpeg graph builder | MoviePy (slow, high RAM, stale); Remotion (Node; **custom licence**, companies above 3 employees need a paid licence) |
| Timeline interchange (optional) | **OpenTimelineIO** (Apache-2.0) | MLT XML / `melt` (LGPL-2.1); Editly (MIT [UNVERIFIED]) | Remotion as the timeline engine (licence) |
| Camera moves on stills | **FFmpeg `zoompan`/`crop` on 2× supersampled image** | Blender camera on an image plane (smoother parallax) | — |
| Transitions | **`xfade`** (fade, fadeblack, wipes) | — | Fancy AI transitions (unreliable) |
| Grading | **`lut3d` with `.cube` presets** | `eq`/`curves` | — |
| Subtitles | **SRT for YouTube captions**; ASS for optional burn-in | — | — |
| Interpolation | **RIFE** (GPU) | FFmpeg `minterpolate` (CPU) | — |
| Upscale | **SeedVR2** (quality) / Real-ESRGAN (fast) | Lanczos scale (baseline) | — |
| Encoding | **libx264 CRF 18** on CPU | NVENC on Colab (drafts), VA-API on Intel (drafts) | — |
| 3D explainer shots | **Blender `bpy` (Phase 4)** | Matplotlib 2.5D | — |
| QC | **ffprobe + FFmpeg detect filters + faster-whisper WER + imagehash + PySceneDetect**; optional vision-LLM contact-sheet review | DOVER (video quality model) | — |

## 6. Orchestration and agents

| Option | Reliability for a fixed pipeline | Resume | Complexity | Fit | Verdict |
|---|---|---|---|---|---|
| **Plain Python stage runner + file checkpoints (hash-based)** | High | ✔ (by design) | **Lowest** | Exact | **✔ Prototype + single-machine production** |
| DBOS (3.2, SQLite/Postgres) | High | ✔ durable | Low | Good | Add when running several concurrent projects |
| Prefect 3 | High | ✔ | Medium | Good, has a UI | Alternative to DBOS |
| Temporal | Very high | ✔ | High (needs a server) | Overkill for one laptop | Large-scale only |
| LangGraph 1.2 | Medium-high | ✔ checkpointer | Medium | Built for LLM loops; duplicates the stage runner | ✖ not needed |
| PydanticAI 2.x | High for typed LLM calls | via DBOS/Temporal | Low | **Inside the Research stage only** (tool loop) | Optional |
| CrewAI 1.15 | Medium | Weak | Medium | Role-play agents add non-determinism | ✖ |
| AutoGen 0.7 | — | — | — | Maintenance mode | ✖ |
| Microsoft Agent Framework / OpenAI Agents SDK / Claude Agent SDK | Medium-high | Varies | Medium | Vendor-oriented agent loops | ✖ for the pipeline; Claude Agent SDK only if you later want an interactive "studio assistant" |

**Why the simplest option wins here:**
- The stages and their order are known in advance.
- Every expensive step produces a file, so checkpointing by file hash is trivial and inspectable.
- Multi-agent chat would add cost, latency and non-determinism without improving any stage's output, because each stage is already one well-specified call with a schema.

## 7. Compute options

| Option | GPU guarantee | Cost | Automation | Max model | Verdict |
|---|---|---|---|---|---|
| Laptop CPU (i5-1335U, 7.4 GB) | n/a | $0 | Full | TTS, ASR, FFmpeg | **Control plane + CPU stages** |
| Colab Free | ✖ none; T4 if lucky | $0 | Attended notebook only; **free tier bans distributed workers and bypassing the UI** [O] | 5B at best on a T4 [UNVERIFIED] | Experiments only |
| Colab Pay-as-you-go / Pro ($9.99 per 100 CU) | Priority, not guaranteed | L4 ≈ $0.17/h, A100-40 ≈ $0.54/h [C, volatile] | Attended notebook loop | 5B comfortably; A14B GGUF | **Prototype GPU** |
| Colab Pro+ / Google AI Ultra | Priority + background execution (Ultra: "uninterrupted background execution") [O] | $49.99/mo+ | Better | Same, plus longer sessions | If Colab becomes the main worker |
| RunPod / Vast pods | ✔ while rented | 4090 $0.34–0.74/h; A100-80 $1.19–1.59/h; H100 $1.99–3.49/h | Full (SSH/API) | Everything | **Production GPU** |
| Modal (serverless) | ✔ | L4 $0.80/h; H100 $3.95/h; $30/mo credit | Full (Python API) | Everything | Production alternative (per-second billing) |
| Own 24 GB GPU (RTX 3090/4090/5090) | ✔ | ~$800–2,500 one-off [C, volatile] | Full | 5B, A14B GGUF, all audio | When monthly cloud spend justifies it |

---

## 8. Recommended stack

### 8.1 Prototype stack (Strategy B: laptop + Colab)

| Layer | Choice |
|---|---|
| Language / env | Python 3.12, **uv** with locked `uv.lock`; separate dependency groups: `core` (laptop), `worker` (GPU) |
| Planning LLM | Hosted API with JSON-schema output, behind `LLMProvider` (Ollama/vLLM backend optional) |
| Schemas | Pydantic v2 → JSON Schema |
| Research | LLM web search/fetch tool loop; claims + URLs + confidence |
| TTS | **Kokoro-82M** (kokoro-onnx, CPU) |
| Alignment / ASR | **faster-whisper** (`small.en`, int8, CPU) |
| Keyframes | **Z-Image-Turbo** / **FLUX.2 klein 4B** on Colab (SDXL on a T4) |
| I2V | **Wan 2.2 TI2V-5B** on Colab L4/A100 (diffusers) |
| Graphics | Pillow + Matplotlib + Natural Earth data |
| Music / SFX | Curated CC0 library with `licenses.json` |
| Render | **FFmpeg** (apt or static build), libx264/AAC |
| QC | ffprobe, FFmpeg detect filters, faster-whisper, imagehash, PySceneDetect |
| Transfer | **rclone** ⇄ Google Drive folder ⇄ `drive.mount()` in Colab |
| Orchestration | Plain Python stage runner + hash checkpoints |
| Publishing | google-api-python-client + google-auth-oauthlib, local only |
| Effects | **Recipe menu** (~9 tested recipes) + closest-recipe fallback for unsupported requests. No run-time code generation |
| GPU workflows | diffusers adapters; **pinned community ComfyUI workflows** where better tuned (benchmark in M4) |

### 8.2 Production stack (Strategy C)

| Layer | Change from the prototype |
|---|---|
| GPU | Rented **L40S 48 GB / A100 80 GB / H100** pod or Modal functions; same worker package |
| Transfer | S3-compatible bucket (Cloudflare R2 or MinIO) instead of Drive |
| I2V | **Wan 2.2 I2V-A14B + LightX2V 4-step**; Wan 2.1 FLF2V for transitions; optional LTX-2.5 if the licence fits |
| Keyframes | Z-Image-Turbo + **Qwen-Image-Edit-2511** for character sheets; character LoRAs via ai-toolkit |
| TTS | **Qwen3-TTS 1.7B** (VoiceDesign for a consistent house narrator) |
| Music | **ACE-Step 1.5** + library |
| Post | RIFE, SeedVR2 |
| Orchestration | Stage runner + **DBOS** (SQLite → Postgres) when running several projects concurrently |
| QC | + vision-LLM contact-sheet review, + DOVER |

---

## 9. Three implementation strategies

### Strategy A — Fully local (Ubuntu)

- **What runs where:** everything on one Ubuntu machine.
- **Required hardware for real AI video:**
  - **Minimum:** NVIDIA GPU with **24 GB VRAM** (RTX 3090/4090/5090 class), **64 GB system RAM** (offloading 5B/14B text encoders and model swapping need it), 1 TB NVMe for weights and caches (Wan 2.2 5B ~34 GB, A14B ~126 GB, image models 8–58 GB each).
  - **Comfortable:** 32 GB+ VRAM (RTX 5090) or a 48 GB workstation card.
- **On the current laptop** (no NVIDIA GPU, 7.4 GB RAM): only the **"no-generative-visuals" variant** works:
  - LLM planning (API), Kokoro TTS, graphics, FFmpeg camera moves on **licensed/public-domain stills**, subtitles, upload.
  - It produces watchable explainer videos, not cinematic AI footage.
- **Quality limits even with a 24 GB card:**
  - 5B at 720p is fine.
  - A14B needs GGUF quantisation and the 4-step LoRA (some motion loss).
  - About 1–3 min per 480p clip on a 4090 with A14B + LightX2V [C]; under 9 min per 720p clip with 5B, unoptimised [O].
  - A 3-min video with 12 I2V shots takes about 0.5–2 h of GPU time [est].

### Strategy B — Hybrid: Ubuntu laptop + Google Colab (**recommended prototype**)

- **What runs where:**
  - **Laptop:** planning, TTS, alignment, graphics, edit, render, QC, publishing.
  - **Colab:** keyframes and I2V (plus optional Qwen3-TTS and ACE-Step), as job files over Drive.
- **Quality:** good stills; 5B I2V for 30–40% of shots; the rest uses camera moves on keyframes and graphics.
- **Limits:**
  - No guaranteed GPU, session caps and idle timeouts.
  - The free tier is unsuitable for an unattended worker.
  - The pipeline must degrade gracefully (I2V → still_motion).

### Strategy C — Best practical open-source stack (**recommended production**)

- **What runs where:**
  - **Laptop or a small always-on box:** control plane and publishing.
  - **Rented GPU** (RunPod/Vast pod or Modal): all generation.
- **Quality:** A14B I2V for most motion shots, FLF2V transitions, character LoRAs, upscaling and interpolation, generated music.
- **Limits:** costs money per video. Needs an S3 bucket and container images. Still bound by 5 s clip limits, so editing remains essential.

### 9.1 Comparison

| Criterion | A — Fully local | B — Laptop + Colab | C — Laptop + rented GPU |
|---|---|---|---|
| **Visual quality** | Current laptop: low (stills + graphics only). With a 24 GB GPU: medium-high | Medium-high (5B I2V + stills) | **High** (A14B, LoRAs, upscaling) |
| **Hardware cost** | $0 now; **$1,500–3,500** for a 24 GB GPU workstation [est] | **$0** | $0 |
| **Operating cost per 3-min video** [est] | Electricity only | **$0–2** (≈ 1–4 h of L4/A100 at $0.17–0.54/h) | **$2–10** (≈ 2–5 GPU-h at $0.8–2/h incl. idle and model loading) |
| **Generation speed (3-min video)** [est] | Laptop: 30–60 min (no AI video). 24 GB GPU: 1–3 h | 1.5–4 h, depends on GPU assigned | **0.5–1.5 h** (parallel GPUs possible) |
| **Reliability** | High (no external dependency) | **Medium-low** (GPU availability, sessions, Drive quotas) | High (paid capacity) |
| **Automation complexity** | Low | Medium (job protocol, attended notebook) | Medium-high (containers, bucket, provider API) |
| **Licensing** | Same models in all three; all defaults are Apache/MIT | Same; also Colab ToS (no deepfakes, free-tier worker limits) | Same; provider ToS |
| **Scalability** | 1 GPU | 1 session at a time; quota-limited | **Horizontal** (N GPUs) |
| **Setup effort** | Low (current laptop) / medium (GPU drivers) | Medium | Medium-high |
| **Fits the "local + Colab" goal** | Partly | **Yes** | Beyond it |

### 9.2 Recommendation

- **Prototype: Strategy B.**
  - It costs nothing up front.
  - It proves the planning, timing, editing, QC and publishing spine on the current laptop.
  - It uses Colab only for the GPU-bound steps.
  - It is designed so that losing the GPU never blocks a video.
- **Production: Strategy C.**
  - It uses the same codebase. The worker package is identical; only the transport (Drive → S3) and the launcher change.
  - Move when any of these holds:
    - (a) you publish more than about 2 videos/week;
    - (b) Colab availability causes missed deadlines;
    - (c) you want A14B quality.
- **Buy a GPU (Strategy A with a 24 GB card) only if** cloud GPU spend stays above roughly **$100–150/month for 12+ months** [est], or you need offline work.

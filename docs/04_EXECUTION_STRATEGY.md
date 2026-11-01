# 04 — Execution Strategy: Ubuntu, Colab, Workload Routing, Runtime and Cost

**Date:** 2026-10-11

**Labels:**
- [O] official
- [C] community
- [est] estimate; measure it in Milestone M4 and record it in `docs/benchmarks.md`
- [UNVERIFIED]

---

## 1. Environments

### 1.1 Local Ubuntu (reference machine, detected 2026-10-11)

| Item | Value | Consequence |
|---|---|---|
| CPU | Intel i5-1335U (10 cores / 12 threads) | Fine for TTS (Kokoro), ASR (faster-whisper small, int8) and FFmpeg |
| RAM | 7.4 GB total (~1.7 GB free during the check) | Render one shot at a time. Avoid giant filtergraphs. Close the browser during renders |
| GPU | Intel UHD (Raptor Lake), **no NVIDIA / CUDA** | No local diffusion. VA-API/QSV encode possible for drafts [UNVERIFIED] |
| Disk | 265 GB free on `/home` | Enough for ~100 projects after cleanup. Keep weights off the laptop |
| Python | 3.12.3 | OK. Use **uv** for an isolated venv |
| Tools missing | ffmpeg, rclone | Install in M1 (`sudo apt install ffmpeg rclone`). Ubuntu's ffmpeg is fine; a static build gives a newer version. No container runtime is needed (no generated code is executed) |

### 1.2 Hardware detection (`main.py doctor`)

The resource planner runs these checks at start-up and writes `hardware.json`:

```python
# orchestration/hardware.py (sketch)
import os, shutil, platform, subprocess, json, psutil

def detect() -> dict:
    info = {
        "os": platform.platform(), "python": platform.python_version(),
        "cpu": {"model": _cpu_model(), "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count()},
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "ram_free_gb": round(psutil.virtual_memory().available / 2**30, 1),
        "disk_free_gb": round(shutil.disk_usage(os.getcwd()).free / 2**30, 1),
        "ffmpeg": _version(["ffmpeg", "-version"]),
        "encoders": _encoders(),                       # libx264, h264_nvenc, h264_vaapi...
        "gpus": [], "cuda": None,
    }
    if shutil.which("nvidia-smi"):
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        for line in q.strip().splitlines():
            name, total, free, drv = [x.strip() for x in line.split(",")]
            info["gpus"].append({"name": name, "vram_gb": int(total) / 1024,
                                 "vram_free_gb": int(free) / 1024, "driver": drv})
    try:
        import torch                                     # only present in the worker env
        info["cuda"] = {"available": torch.cuda.is_available(), "version": torch.version.cuda,
                        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported()}
    except ImportError:
        pass
    return info
```

### 1.3 Google Colab (facts as of 2026-10-11)

| Aspect | Fact | Source |
|---|---|---|
| GPU availability | *"Colab resources are not guaranteed and not unlimited… Colab does not publish these limits."* GPU types "vary over time" | [O] FAQ |
| Session length | *"at most 12 hours"*; *"Colab Pro+ supports continuous code execution for up to 24 hours if you have sufficient compute units"* | [O] FAQ |
| Idle timeout | Applies only when code is not running; length not published (~90 min reported) | [O] / [C] |
| Pricing | Pro / Pay-as-you-go ≈ **$9.99 per 100 CU**; Pro+ $49.99/mo (500–600 CU) [UNVERIFIED]; since 2026-09-22 Colab benefits are bundled with Google AI plans (Ultra: "uninterrupted background execution") | [C] / [O] blog |
| CU burn (volatile) | T4 ~1.2 CU/h (≈ $0.12/h) · L4 ~1.7 (≈ $0.17) · A100-40 ~5.4 (≈ $0.54) · A100-80 ~7.5 · G4/RTX PRO 6000 ~8.7 · H100 listed | [C] |
| When CU runs out | *"Paid users whose compute unit balance is exhausted will revert to the free-of-charge tier policies"* | [O] |
| Banned on all runtimes | Web services / file hosting unrelated to interactive compute, remote proxies, crypto mining, **"creating deepfakes"**, multiple accounts to evade limits, containerisation to circumvent anti-abuse | [O] |
| Banned on free tier / zero balance | Remote control (SSH, remote desktop), **"bypassing the notebook UI to interact primarily via a web UI"**, **"running distributed computing workers"** | [O] |
| RAM / disk | Free ≈ 12 GB RAM, ~100 GB disk [C]; paid tiers can get high-memory VMs | [C]/[O] |
| Drive | Mount fails above ~10k items in a folder; per-user operation and bandwidth quotas cause I/O errors; buying Drive space doesn't add VM disk | [O] |
| Secrets | `google.colab.userdata.get("HF_TOKEN")` from the Secrets panel | [C] |
| Programmatic use | Google released the **Colab CLI** in June 2026 (provision runtimes, run scripts, fetch artifacts). Tier rules [UNVERIFIED] | [C] |

**Operating rules that follow:**
1. **The Colab worker is an attended notebook.** You open it, run "Start worker", and it processes the queue of jobs for **your own** project, inside the notebook UI. It is not a server, has no tunnel, no SSH and no web UI. On the free tier, keep it attended and short. For regular use, have compute units (Pay-as-you-go or Pro); an automated queue on the free tier is risky under *"distributed computing workers"*.
2. **Never assume a GPU type.** The worker detects the GPU and picks adapters that fit (§3.2).
3. **No real-person likenesses or voice clones** (deepfake ban) — this is also a project policy.
4. **Cache weights on Drive?** Usually **no**. Pulling 20–35 GB from Hugging Face each session is typically faster than through the Drive FUSE mount [C/est]. Keep the HF cache on local VM disk (`/content/hf`). Only small LoRAs and configs live on Drive.

### 1.4 Colab Free vs Pro vs local vs rented

| | Colab Free | Colab Pay-as-you-go / Pro | Colab Pro+ / AI Ultra | Laptop (current) | Laptop + 24 GB GPU | Rented GPU |
|---|---|---|---|---|---|---|
| Likely GPU | T4 or none | T4 / **L4** / A100 (priority) | L4 / A100 / H100 / G4, longer sessions | none | 3090/4090/5090 | any |
| Wan 2.2 5B I2V | marginal [UNVERIFIED] | ✔ (L4/A100) | ✔ | ✖ | ✔ | ✔ |
| Wan 2.2 A14B | ✖ practical | GGUF on L4/A100-40; native on A100-80 | ✔ | ✖ | GGUF + LightX2V | ✔ |
| Max session | ≤ 12 h, often less | ≤ 12 h | ≤ 24 h, background | ∞ | ∞ | ∞ |
| Unattended automation | ✖ (ToS) | grey area; keep attended | better | ✔ | ✔ | ✔ |
| Cost | $0 | ~$0.17–0.54/h of GPU | $49.99+/mo | $0 | hardware | $0.34–3.5/h |

---

## 2. Workload routing

### 2.1 Default routing table

| Workload | Default target | Why | Fallback |
|---|---|---|---|
| Planning LLM calls (brief, research, script, storyboard, critic, metadata) | **Local** → hosted API | CPU-trivial; API quality | Ollama on a Colab GPU (open mode) |
| TTS (Kokoro) | **Local CPU** | ~realtime or faster on CPU [est]; no transfer | Piper (local) |
| TTS (Qwen3-TTS, quality mode) | Remote GPU | Needs a GPU | Kokoro |
| ASR check + alignment | **Local CPU** (faster-whisper small int8) | ~0.2–0.5× realtime [est] | whisper.cpp |
| Graphics (maps, titles, diagrams) | **Local CPU** | Deterministic | — |
| Keyframes (T2I) | **Remote GPU** | No CUDA locally | Graphic card or licensed stock still |
| I2V clips | **Remote GPU** (L4/A100 preferred) | VRAM | Downgrade the shot to `still_motion` |
| Music (ACE-Step) | Remote GPU (Phase 4) | — | Curated library (default in prototype) |
| Upscale / interpolate | Remote GPU | — | Lanczos + `fps` filter locally |
| Per-shot segments, edit, mix, encode | **Local CPU** | Files are already local; FFmpeg is CPU | Remote with NVENC if the laptop is too slow |
| QC | **Local CPU** (+ optional vision-LLM API) | — | — |
| YouTube upload | **Local only** | Credentials never leave the laptop | — |

### 2.2 Routing algorithm

```text
for each job in the plan:
    req = models.yaml[job.adapter].requirements        # min_vram_gb, needs_bf16, est_s_per_unit
    if job.kind in LOCAL_ONLY (publish, plan, mix, render, qc): run locally
    elif local GPU satisfies req:                        run locally (worker process)
    elif remote.enabled and remote worker heartbeat < 5 min old:
         if remote GPU satisfies req:                    send to remote queue
         elif a lighter adapter exists (e.g. SDXL instead of Z-Image on T4): send with that adapter
         else:                                           apply fallback_mode (i2v → still_motion)
    else:                                                apply fallback_mode, flag in qc_report
deadline check: if est_remote_time > settings.max_remote_wait, downgrade lowest-priority i2v shots first
```

- Every decision and its reason is written to `02_jobs/routing.json`. Example: `"s02_sh03": {"target": "remote", "adapter": "i2v.wan22_ti2v_5b", "reason": "needs 24GB, remote reports L4 22.5GB+offload"}`.
- The worker reports its GPU in a `heartbeat.json` (GPU name, VRAM, bf16, adapters loaded, queue position). The laptop reads it to route.

---

## 3. File transfer between laptop and Colab

### 3.1 Mechanism: a shared Google Drive folder

```text
Google Drive: /ai-video-studio/
├── jobs/{pending,running,done,failed}/<job_id>.json   # small JSON files
├── inputs/<project_id>/...                            # keyframes for I2V, reference images
├── outputs/<project_id>/...                           # generated keyframes, clips, audio
└── workers/<worker_id>/heartbeat.json
```

- **Laptop side:** `rclone` with a Drive remote, configured once with `rclone config` (OAuth in your browser). The token is stored in `~/.config/rclone/rclone.conf`, mode 600.
  - Push jobs: `rclone copy`.
  - Pull results: `rclone copy --include` for `done/` and `outputs/`.
  - The orchestrator calls rclone via `subprocess` with `--checksum`.
- **Colab side:** `from google.colab import drive; drive.mount('/content/drive')`. The worker reads `jobs/pending` and writes results to `outputs/`.
- **Drive limits respected:**
  - Fewer than 10k items per folder: outputs go in per-project subfolders, and `done/` is archived per project.
  - Few large files rather than many tiny ones.
  - Exponential backoff on I/O errors.
  - Clips are written to local VM disk first, then copied in one go.
- **Integrity:** each output has a sidecar with its `sha256`. The laptop verifies it before accepting.

### 3.2 Worker adapter selection by GPU (Colab or rented)

| GPU detected | Keyframes | I2V | Notes |
|---|---|---|---|
| None / CPU | — (refuse image jobs) | — | Laptop falls back |
| T4 16 GB (no bf16) | SDXL / FLUX.1-schnell fp8-GGUF [est] | Wan 2.2 5B fp16/GGUF **experimental**, low priority | Expect slow runs; prefer still_motion |
| L4 24 GB (bf16, FP8) | Z-Image-Turbo / FLUX.2 klein 4B | **Wan 2.2 TI2V-5B** (offload text encoder) | **Target prototype GPU** |
| A100 40 GB | same | Wan 2.2 5B (fast); A14B GGUF + LightX2V | |
| A100 80 GB / H100 / G4 96 GB | + Qwen-Image | **Wan 2.2 A14B + LightX2V**, FLF2V | Quality tier |

### 3.3 Alternative transports

| Transport | When |
|---|---|
| Google Drive (rclone + mount) | **Prototype** with Colab |
| S3-compatible bucket (Cloudflare R2, Backblaze B2, MinIO) | Rented GPUs (Strategy C); more reliable than Drive FUSE |
| `rsync` over SSH | Your own GPU box on the LAN |
| Hugging Face Hub private dataset | Not recommended (versioning noise) |

---

## 4. Runtime and cost estimates

All figures are **[est]** unless marked. Wan 2.2 5B timing anchor: under 9 min per 5 s 720p clip on a consumer 24 GB GPU, unoptimised [O]. Measure everything in M4.

### 4.1 Per-unit estimates

| Unit | Laptop CPU | Colab T4 | Colab L4 | A100 40/80 | H100 |
|---|---|---|---|---|---|
| Plan (all LLM stages, 3-min video) | 3–8 min (API) | — | — | — | — |
| Kokoro TTS, 180 s of narration | 1–4 min | — | — | — | — |
| faster-whisper small int8, 180 s | 1–3 min | — | — | — | — |
| Keyframe 1344×768 (Z-Image-Turbo / klein 4B) | ✖ | 20–60 s (SDXL) | 5–15 s | 2–6 s | 1–4 s |
| Wan 2.2 TI2V-5B, 5 s 720p clip | ✖ | 20–40 min [UNVERIFIED if it runs] | 6–12 min | 3–6 min | 1.5–4 min |
| Wan 2.2 A14B + LightX2V 4-step, 5 s 480p | ✖ | ✖ | 4–10 min (GGUF) | 2–5 min | 1–3 min |
| Model load (first job of a kind, per session) | — | 3–8 min | 2–5 min | 1–3 min | 1–3 min |
| Still_motion shot segment (5 s, 1080p) | 10–40 s | — | — | — | — |
| Final encode, 180 s 1080p x264 medium | 3–10 min | — | — | — | — |

### 4.2 Per-video estimates (3-minute documentary, ~34 shots, 12 I2V)

| Strategy | GPU time | Wall clock | Cash cost | Notes |
|---|---|---|---|---|
| A, current laptop (no AI video; graphics + licensed stills) | 0 | 30–60 min | ~$0.10–0.50 (LLM API) | Explainer quality |
| B, Colab L4 | ~2–3 h (28 keyframes + 12 × 5B clips + load) | 2.5–4 h | ~$0.40–0.60 CU + LLM ≈ **$1** | If an L4 is granted |
| B, Colab A100 | ~1–1.5 h | 1.5–2.5 h | ~$0.60–0.90 CU + LLM ≈ **$1–2** | |
| B, Colab free T4 | 5–9 h if it works | multiple sessions | $0 | Not recommended for I2V |
| C, RunPod L40S/A100 pod, 5B | ~1 h + 15 min setup | ~1.5 h | ~$1–3 | |
| C, A100-80/H100, A14B + LightX2V, 2 retries avg | ~1.5–3 h (one GPU) / ~25–40 min (4 GPUs) | 0.75–3 h | **~$3–10** | Quality tier |

**Monthly view [est]:**
- **4 videos/month on Strategy B:** ~$5–10 of compute units + ~$2–5 LLM.
- **4 videos/month on Strategy C:** ~$15–45.
- **A 24 GB GPU workstation ($1,500–3,500)** pays for itself only at sustained high volume.

### 4.3 Disk usage and cleanup

| Item | Size [est] | Policy |
|---|---|---|
| Project working files (3-min video) | 0.8–1.5 GB | `main.py clean` after publishing: delete `03_visuals/rejected`, `05_render/shots`, raw clips; keep plan, final MP4, keyframes, logs |
| Kept after cleanup | 200–400 MB | Forever (or archive to Drive) |
| Model weights on the laptop | 0 (none needed) | — |
| Model weights on Colab VM | 20–60 GB per session | Ephemeral |
| Model weights on a rented GPU | 50–200 GB | Network volume, reused across pods |
| Incomplete jobs | Detected by stale leases / missing sidecars | `main.py resume` re-queues them; `main.py clean --orphans` removes partial files without a valid sidecar |

---

## 5. Quality validation thresholds per stage

[est] values; calibrate them in M5.

| Stage | Hard checks (block) | Soft checks (warn) | Retry policy |
|---|---|---|---|
| Plan | Schema valid; durations sum ±10% of target; all refs resolve; every claim sourced | Pace 2.0–3.0 words/s; adjacent-shot variety | LLM re-ask with errors ×2 |
| TTS sentence | File exists, duration > 0.3 s, no clipping (peak < −1 dBFS) | WER ≤ 5% vs text | Regenerate ×2, apply pronunciation dictionary |
| Alignment | Every sentence aligned; monotonic timestamps | — | Fall back to proportional timing |
| Keyframe | Correct size, not blank (std dev > 5), not a near-duplicate of another shot (pHash Hamming > 6) | CLIP/VLM prompt match | New seed ×3 → simplified prompt → fallback |
| I2V clip | Frames ≥ needed; decodes; no frozen run > 1 s; mean frame difference above a motion floor | Flicker metric; VLM "artefacts?" check on 3 frames | New seed ×2 → still_motion |
| Shot segment | Exact duration ±1 frame; 1920×1080, 24 fps, yuv420p | — | Re-render |
| Mix | Integrated −14 ± 1 LUFS; true peak ≤ −1 dBTP | Music under voice by ≥ 12 dB | Re-run loudnorm |
| Subtitles | ≤ 2 lines, ≤ 42 chars/line, cue ≥ 0.8 s | ≤ 20 chars/s | Re-segment |
| Final video | Duration ±3%; codec/profile; `blackdetect` (d ≥ 0.5 s, unplanned), `freezedetect` (d ≥ 2 s), `silencedetect` (−50 dB, ≥ 1.5 s); all shots present | Final-mix WER < 8%; scene cuts ≈ timeline | Re-run failing shots/stages only |
| Publish | Approval hash matches; metadata limits; disclosure flag | Title/thumbnail match content (VLM) | — |

---

## 6. Pipeline step reference

For every step: inputs, outputs, tools, hardware, time for a 3-min video [est], failure modes and validation.

| # | Step | Inputs → Outputs | Libraries / models | Python modules | HW | Time | Failure modes → retry | Validation |
|---|---|---|---|---|---|---|---|---|
| 1 | Topic intake | CLI args → `project.json` | — | `main.py`, `orchestration/project.py` | CPU | <1 s | Bad args → exit with message | Settings schema |
| 2 | Brief | topic → `brief.json` | LLM API | `agents/director.py`, `llm/provider.py` | API | 15–30 s | Invalid JSON → re-ask ×2 | Pydantic |
| 3 | Research | brief → `research.json` | LLM + web search/fetch; `httpx`, `trafilatura` | `agents/researcher.py` | API | 2–5 min | Fetch errors → skip source; too few sources → widen query | ≥ 1 source per claim; confidence labels |
| 4 | Script | brief + research → `script.json` | LLM | `agents/scriptwriter.py` | API | 30–60 s | Over/under length → re-ask | Word count ±10%; claim links |
| 5 | Storyboard | script + style → `storyboard.json` | LLM (per section, parallel) | `agents/storyboard.py` | API | 1–2 min | Schema errors → re-ask per section | Pydantic + validator |
| 6 | Validate + critic | storyboard → `plan_review.json`, `production_plan.json` | Rules + LLM | `agents/critic.py`, `schemas/validate.py` | CPU+API | 30–60 s | Unfixable → stop for human edit | All hard rules pass |
| 7 | Narration | sentences → `narration/*.wav` | Kokoro (kokoro-onnx) | `generators/speech.py` | CPU | 1–4 min | Mispronunciation → dictionary + regen | WER, duration |
| 8 | Align + lock timing | wavs → `words.json`, timed plan | faster-whisper | `generators/alignment.py`, `orchestration/timing.py` | CPU | 1–3 min | Misalignment → proportional fallback | Monotonic, total length |
| 9 | Route | plan + hardware → `routing.json`, job files | — | `orchestration/router.py` | CPU | <1 s | No GPU → fallbacks | Every shot has a target |
| 10 | Graphics | graphic specs → `graphics/*.png` | Pillow, Matplotlib, Natural Earth | `generators/graphics.py` | CPU | <1 min | Missing data → simple text card | Size, non-blank |
| 11 | Keyframes | jobs → `keyframes/*.png` | Z-Image-Turbo / FLUX.2 klein / SDXL (diffusers) | `worker/adapters/t2i_*.py` | GPU | 5–25 min | OOM → lower res/offload; artefact → reseed | §5 keyframe checks |
| 12 | I2V | keyframe + motion prompt → `clips/*.mp4` | Wan 2.2 TI2V-5B (diffusers) | `worker/adapters/i2v_*.py` | GPU | 30 min–3 h | OOM / session loss / bad motion → reseed → still_motion | §5 clip checks |
| 13 | Music / SFX | cues → `music/*.wav`, `sfx/*.wav` | Library (prototype), ACE-Step (later) | `generators/music.py`, `generators/sfx.py` | CPU/GPU | <1–10 min | No match → neutral bed | Licence recorded; duration |
| 14 | Asset QC | assets → `assets_qc.json` | imagehash, PyAV, numpy, optional VLM API | `agents/quality_controller.py` | CPU | 2–5 min | Fails → back to 11/12 for those shots only | §5 |
| 15 | Shot segments | assets + timeline → `shots/*.mp4` | FFmpeg | `rendering/shots.py` | CPU | 5–15 min | FFmpeg error → log + retry once | ffprobe |
| 16 | Audio mix | narration + music + SFX → `mix.wav` | FFmpeg (`sidechaincompress`, `loudnorm`) | `rendering/audio_mix.py` | CPU | <1 min | — | LUFS, true peak |
| 17 | Subtitles | words → `.srt`, `.ass` | Python | `generators/subtitles.py` | CPU | <1 s | — | §5 |
| 18 | Final render | segments + mix + subs → `final_16x9.mp4` | FFmpeg libx264 | `rendering/renderer.py` | CPU | 5–15 min | OOM → segmented concat | ffprobe profile |
| 19 | Final QC | final → `qc_report.json` | FFmpeg detect filters, faster-whisper, PySceneDetect | `agents/quality_controller.py` | CPU | 2–4 min | Fail → targeted re-run | §5 |
| 20 | Metadata + thumbnail | plan + timeline → `metadata.json`, `thumbnail.png` | LLM, Pillow | `publishing/metadata.py`, `publishing/thumbnail.py` | CPU+API | 1–2 min | Over-length → truncate + re-ask | YouTube limits |
| 21 | Approval | you → `approval.json` | — | `main.py review/approve` | — | — | — | Hash bound to the final file |
| 22 | Upload | files → `publish_record.json` | YouTube Data API v3 | `publishing/youtube.py` | CPU/net | 1–5 min | 5xx → backoff/resume; quota → stop | `video_id` stored |
| 23 | Record + cleanup | — → logs, cleaned project | — | `orchestration/cleanup.py` | CPU | <1 min | — | — |

### 6.1 Resume without regenerating completed scenes

The full mechanism is in `02_ARCHITECTURE.md §6.2`. In short:
1. Every artifact has a sidecar with its `input_hash`, built from upstream file hashes, config slice, adapter version and seed.
2. On re-run, a step is skipped when the output exists, the hash matches, and validation passes.
3. Remote jobs carry the same hash. Stale leases (no heartbeat for 10 minutes) are re-queued.
4. Commands:
   - `main.py resume --project X` continues.
   - `main.py redo --project X --shot s03_sh06` invalidates only that shot and its downstream render.
   - `main.py status` prints per-stage and per-shot state.

---

## 7. Configuration keys

### 7.1 `config/settings.yaml`

```yaml
project_root: projects
format_profiles:
  youtube: {width: 1920, height: 1080, fps: 24, vcodec: libx264, crf: 18, preset: medium, acodec: aac, abitrate: 192k}
  shorts:  {width: 1080, height: 1920, fps: 30, vcodec: libx264, crf: 18, preset: medium, acodec: aac, abitrate: 192k}
  draft:   {width: 960,  height: 540,  fps: 24, vcodec: libx264, crf: 26, preset: veryfast, acodec: aac, abitrate: 128k}
visual_mix: {i2v_share_max: 0.40, graphic_share_min: 0.10}
llm: {provider: hosted, model: <set in .env>, max_retries: 2, temperature: 0.4}
tts: {adapter: tts.kokoro, voice: am_michael, speed: 1.0, max_wer: 0.05}
remote:
  enabled: true
  transport: gdrive            # gdrive | s3 | local
  rclone_remote: "gdrive:ai-video-studio"
  heartbeat_timeout_s: 300
  lease_timeout_s: 600
  max_remote_wait_min: 180
qc: {loudness_lufs: -14, true_peak_dbtp: -1.0, max_black_s: 0.5, max_freeze_s: 2.0, final_wer_max: 0.08}
publishing: {default_privacy: private, max_uploads_per_day: 2, category_id: "27", made_for_kids: false}
cleanup: {delete_rejected_after_publish: true, keep_shot_segments: false}
```

### 7.2 `config/models.yaml`

```yaml
adapters:
  t2i.z_image_turbo:
    repo: Tongyi-MAI/Z-Image-Turbo
    revision: <pin commit sha in M4>
    license: apache-2.0
    commercial_ok: true
    license_checked: 2026-10-11
    requirements: {min_vram_gb: 16, needs_bf16: false}
    defaults: {steps: 8, width: 1344, height: 768}
  t2i.flux2_klein_4b:
    repo: black-forest-labs/FLUX.2-klein-4B
    license: apache-2.0
    commercial_ok: true
    requirements: {min_vram_gb: 13}
  t2i.sdxl:
    repo: stabilityai/stable-diffusion-xl-base-1.0
    license: openrail++
    commercial_ok: true
    requirements: {min_vram_gb: 8}
  i2v.wan22_ti2v_5b:
    repo: Wan-AI/Wan2.2-TI2V-5B-Diffusers
    license: apache-2.0
    commercial_ok: true
    requirements: {min_vram_gb: 22, recommended_vram_gb: 24}
    defaults: {width: 1280, height: 704, frames: 121, fps: 24}
  i2v.wan22_a14b_lightx2v:
    repo: Wan-AI/Wan2.2-I2V-A14B-Diffusers
    lora: lightx2v/Wan2.2-Distill-Loras
    license: apache-2.0
    commercial_ok: true
    requirements: {min_vram_gb: 48}
    defaults: {width: 832, height: 480, frames: 81, fps: 16, steps: 4}
  tts.kokoro:
    package: kokoro-onnx==0.6.1
    license: apache-2.0
    commercial_ok: true
    requirements: {cpu_ok: true}
  i2v.wan22_i2v_lightx2v_comfy:          # pinned community ComfyUI workflow (adapter type comfy_workflow)
    type: comfy_workflow
    workflow: worker/workflows/wan22_i2v_lightx2v@1.json
    lock: worker/workflows/wan22_i2v_lightx2v.lock.yaml   # ComfyUI commit, custom-node commits, model sha256s
    source: <URL of the community workflow>   # + author + licence, recorded at adoption
    slots: {prompt: "6.text", negative: "7.text", seed: "3.seed", image: "52.image", frames: "50.length"}   # example node ids
    license: apache-2.0 (models) / check workflow + nodes
    commercial_ok: true
    requirements: {min_vram_gb: 16}
defaults: {t2i: t2i.z_image_turbo, i2v: i2v.wan22_ti2v_5b, tts: tts.kokoro}
policy: {allow_non_commercial: false}    # loader refuses adapters with commercial_ok: false
```

The repo IDs above are to be confirmed and pinned to commit SHAs in M4. `Wan-AI/Wan2.2-TI2V-5B-Diffusers` and the LightX2V LoRA repo name are [UNVERIFIED] spellings.

---

## 8. Execution commands (target UX)

```bash
# one-time
uv sync --group core                              # laptop environment
python main.py doctor                             # hardware + tools report
rclone config                                     # create the "gdrive" remote (browser OAuth)
python main.py auth youtube                       # only when you reach publishing

# per video
python main.py create --topic "How black holes are formed" --duration 180 \
  --style cinematic --voice english --format youtube
#   → plans, narrates, renders graphics, writes GPU jobs, waits for the worker
#   → open notebooks/colab_worker.ipynb in Colab, Runtime → GPU, run all
python main.py status  --project projects/black_holes
python main.py resume  --project projects/black_holes        # after any interruption
python main.py review  --project projects/black_holes
python main.py approve --project projects/black_holes
python main.py publish --project projects/black_holes --privacy private
python main.py clean   --project projects/black_holes
```

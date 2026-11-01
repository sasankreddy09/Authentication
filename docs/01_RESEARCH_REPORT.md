# 01 — Research Report: Open-Source Tools for AI Video Production (October 2026)

**Date:** 2026-10-11
**Method:**
- Facts were checked against GitHub repos, Hugging Face model cards and API metadata (license tags, file sizes, parameter counts), LICENSE files, PyPI and official vendor pages. This was done on 2026-10-06/07 and re-checked on 2026-10-11.
- Nothing was installed or downloaded except metadata.

**Labels:**
- **[O]** comes from an official repo, model card, license or docs.
- **[C]** comes from community or third-party sources.
- **[est]** is an engineering estimate.
- **[UNVERIFIED]** is uncertain; test it before relying on it.

**Not legal advice.** Licenses change. Re-read each LICENSE before monetising, and record the date you checked it in `config/models.yaml`.

---

## 0. Executive summary

1. **Open text/image-to-video is good for 4–10 s shots, not for long videos.** Every competitive open model generates short clips:
   - Wan 2.2: 5 s at 16–24 fps.
   - HunyuanVideo 1.5 and Kandinsky 6: about 5 s.
   - LTX-2.5: about 5 s per generation, extendable.
   - A 3-minute video is therefore **30–40 separately generated shots**, joined by an editor. Long-video models (LongCat-Video, SkyReels V2 diffusion forcing, FramePack) exist but drift and are slow. **[O for clip lengths; drift is C]**
2. **Wan 2.2 (Apache-2.0) is still the best commercially clean open video family.**
   - The **TI2V-5B** model generates 1280×704, 24 fps, 5 s clips and runs on a 24 GB GPU with offload. ComfyUI reports it fitting in about 8 GB with offloading [O/C].
   - The **A14B** MoE models give higher quality but officially want an **80 GB** GPU. Community GGUF quantisation plus the 4-step LightX2V LoRA bring them to 12–24 GB at 480p [O/C].
   - **No Wan 2.5+/3.0 weights exist.** Those versions are API-only [O].
3. **Many headline models are not commercially clean.**
   - **Territory exclusions:** HunyuanVideo 1.5 excludes EU/UK/South Korea. MiniMax H3 excludes EU/UK/Korea **and the USA**.
   - **Non-commercial:** FLUX.1-dev/FLUX.2-dev, Qwen-Image-2.1 (the new Qwen Research License), MusicGen, F5-TTS, XTTS-v2, MMAudio, AudioLDM2 and the InsightFace face models.
   - **Revenue caps:** the LTX-2.x community license applies below $10M ARR. The Stability Community License applies below $1M revenue.
4. **Audio is mature and cheap.**
   - **Kokoro-82M** (Apache-2.0) gives good English narration on a laptop CPU.
   - **Qwen3-TTS** and **Chatterbox** (MIT, watermarked) are the stronger GPU options.
   - **ACE-Step 1.5** (MIT, <4 GB VRAM) is the best commercially clean music model.
   - **MusicGen weights are CC-BY-NC**, so they cannot be used commercially.
5. **FFmpeg remains the right renderer.** MoviePy has slowed (no release in about 17 months). Remotion needs a paid license above 3 employees. Blender is excellent but heavy, so use it only for 3D/diagram scenes.
6. **Colab is useful but not a guaranteed worker.**
   - GPUs are "not guaranteed". Sessions last at most 12 h (24 h on Pro+).
   - The free tier bans "running distributed computing workers" and "bypassing the notebook UI".
   - All tiers ban "creating deepfakes" [O].
   - Design for an **attended notebook worker** with graceful fallback. Rent a GPU for production.
7. **YouTube API:** since 2026-06-01, uploads have their own quota bucket of **100 uploads/day at 1 unit each** [O]. `containsSyntheticMedia` exists in the API. Realistic AI scenes must be disclosed. Mass-produced repetitive AI content is explicitly called out in the spam and YPP policies [O].

---

## 1. AI planning and script generation

### 1.1 Capability assessment

| Task | LLM reliability | Notes |
|---|---|---|
| Topic research with web search + citations | **Good with tools, poor without** | Must fetch sources and attach URLs per claim. LLMs hallucinate dates and numbers if not grounded [C/est] |
| Narrative structure, narration script | **Mature** | Strong across frontier and mid-size open models |
| Storyboards, shot lists, camera language (shot size, lens, angle, move) | **Mature** | Works best per section (8–12 shots per call), not 40 shots in one call [est] |
| Text-to-image / video prompts | **Good** | Needs model-specific style suffixes and negatives from config |
| Timing and subtitle timing | **Do not trust the LLM** | Compute from TTS audio + word alignment instead |
| Character and scene continuity | **Partly** | The LLM writes a "bible" of canonical descriptions; code pastes them verbatim |
| YouTube title, description, tags, thumbnail concept | **Mature** | Validate lengths in code |

### 1.2 Structured JSON production plans: recommended

Use typed JSON with a schema at every stage:
- Pydantic models are the source of truth and are exported to JSON Schema.
- Hosted APIs (Anthropic, OpenAI, Google) support schema-constrained output.
- Open-weight servers also support it:
  - **Ollama**: JSON Schema in `format` [O]. v0.40.3 at the time of research. *"Ollama's Cloud currently does not support structured outputs."*
  - **llama.cpp**: `json_schema`, `response_format`, or `--grammar` [O]. Not every JSON Schema feature is supported.
  - **vLLM** 0.31: `response_format` with `json_schema` via xgrammar/guidance [O]. The old `guided_json` was removed in v0.12.

The full schemas are in `02_ARCHITECTURE.md §5.2`, and a filled example scene is in `06_EXAMPLE_VIDEO_WORKFLOW.md §7`. A minimal shot looks like this:

```json
{
  "shot_id": "s02_sh03", "duration_s": 4.0, "render_mode": "i2v",
  "camera": {"shot_size": "close_up", "angle": "eye_level", "lens_mm": 85, "move": "static",
             "move_speed": "very_slow", "depth_of_field": "shallow"},
  "image_prompt": "...", "motion_prompt": "...", "negative_prompt": "...",
  "continuity_refs": ["char_worker_gang"], "transition_in": "cut",
  "audio_cues": [{"cue_id": "sfx_tap", "kind": "sfx", "description": "metal chisel tap", "offset_s": 0.3}],
  "factual_status": "reconstruction", "claim_ids": ["c07"]
}
```

### 1.3 Which LLM

| Option | Plan quality | Runs on the reference laptop (7.4 GB RAM, no GPU) | Licence |
|---|---|---|---|
| Hosted frontier API (for example Claude) | Best | Yes (API) | Commercial API (not open source) |
| Qwen3 / Qwen3.x 14B–32B class via vLLM/Ollama on a GPU | Good [est] | No (needs a GPU) | Apache-2.0 [O for Qwen3] |
| Small open model (≤ 4B) via Ollama on CPU | Weak for 30-shot storyboards [est] | Barely | Apache-2.0 |

**Recommendation:**
- Use a hosted API for planning in the prototype. It costs cents per video and is not a GPU workload.
- Keep an `LLMProvider` interface so a self-hosted open model can replace it.
- This is the one place where the "open-source only" goal is traded for reliability. Say so explicitly in the project README.

---

## 2. Image and video generation

### 2.1 Video models (verified 2026-10-11)

| Model (latest open) | Licence / commercial | Params | Download | VRAM | Native output | Modes | Practical? |
|---|---|---|---|---|---|---|---|
| **[Wan 2.2 TI2V-5B](https://github.com/Wan-Video/Wan2.2)** | **Apache-2.0** ✔ [O] | 5B | ~34 GB repo (DiT fp32 shards ~20 GB, umT5 11.4 GB) [O] | 24 GB with offload + `--t5_cpu` [O]. ~8 GB in ComfyUI with offload [C] | **1280×704, 24 fps, 121 frames (5 s)** [O] | T2V + I2V | ✔ **Default** for Colab L4/A100 and 24 GB cards. 5 s at 720p in under 9 min on one consumer GPU, unoptimised [O] |
| **[Wan 2.2 A14B](https://github.com/Wan-Video/Wan2.2)** (T2V, I2V) | **Apache-2.0** ✔ [O] | MoE 27B total / 14B active [O] | I2V repo ~126 GB [O] | Official: **80 GB** even with offload [O]. Community GGUF Q4–Q6 + LightX2V: 12–24 GB [C] | 480p/720p, 16 fps, 81 frames | T2V, I2V (FLF2V via ComfyUI) | ✔ Best quality. Rented 48–80 GB GPU. 4090 + 4-step LoRA: ~1–3 min per 5 s 480p clip [C] |
| [Wan 2.1](https://github.com/Wan-Video/Wan2.1) (1.3B, 14B, FLF2V, VACE) | Apache-2.0 ✔ [O] | 1.3B / 14B | I2V-14B-720P repo ~82 GB [O] | 1.3B: **8.19 GB**; ~4 min for 5 s 480p on a 4090 [O] | 480p/720p, 16 fps | T2V, I2V, **FLF2V**, VACE (reference/control) | ✔ for transitions and low-VRAM T2V |
| Wan 2.2-Animate-2-14B, Wan-Dancer-14B (2026) | Apache-2.0 [O] | 14B | — | — | — | Character animation | Niche |
| Wan 2.5 / 2.6 / 2.7 / 3.0 | **API only** [O]. Wan 3.0 public beta 2026-08-06 [C] | — | — | — | — | — | ✖ Not open |
| [LTX-2.5](https://github.com/Lightricks/LTX-2) | **LTX-2.x Community License**: free under **$10M annual revenue** [O] | 22B DiT + Gemma-4-12B text encoder [O] | Transformer 42 GB bf16 (int8 21.5 GB); minimal set ~66 GiB [O] | FP8 + CPU/disk offload supported; no official minimum [O] | Up to 4K, 24 fps, **native audio** [O] | T2V, I2V, keyframes, audio-to-video, retake | ◐ Strong, but heavy. Licence OK for small creators |
| [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | Tencent Community License, **not valid in EU/UK/South Korea**; cap above 100M MAU [O] | 8.3B | 372 GB repo (33 GB fp32 transformer files) [O] | **14 GB** with offload [O] | 480p/720p, 121 frames; SR to 1080p | T2V, I2V; 8–12-step distilled I2V | ◐ Good on modest GPUs. **Excluded if you or your audience are in the EU/UK/KR** |
| [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (open weights 2026-07-28) | Community licence that **excludes EU, UK, Korea and USA**; separate terms above $20M revenue [O] | 33B | ~498 GB repo [O] | Official example uses 4 GPUs [O] | 768p, 24 fps, 4–15 s, audio | T2VA, I2VA, FLF, multi-ref | ✖ Licence/territory |
| [Kandinsky 6.0](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers) (Sept 2026) | **MIT** ✔ [O] | Lite 3.2B / Pro 30.1B [O] | Lite 28 GB, Pro 90 GB [O] | No official minimum; offload examples [O] | 5 s, 24 fps, ~480p + SR, **audio** | T2AV, TI2AV | ◐ Promising new option, lightly tested [UNVERIFIED quality] |
| Kandinsky 5.0 | MIT ✔ [O] | 2B / 19B | 4.6 / 43 GB [O] | — | 5–10 s | T2V, I2V | ◐ |
| [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) | MIT ✔ [O] | 13.6B | 83 GB [O] | Not stated | 720p 30 fps, continuation | T2V, I2V, continuation | ◐ For long continuous shots (experimental) |
| [CogVideoX 1.5-5B](https://huggingface.co/zai-org/CogVideoX-5b-I2V/blob/main/LICENSE) | CogVideoX License: commercial use needs registration, **< 1M visits/month**; **2B is Apache-2.0** [O] | 5.5B / 1.7B | 31 / 14 GB [O] | ~10 GB with offload [C] | 1360×768, 16 fps, 5–10 s | T2V, I2V | ✖ Superseded |
| MAGI-1 / MAGI-2-preview | Apache-2.0 ✔ [O] | 24B, 4.5B / 114B MoE | 246 / 307 GB [O] | MAGI-2 needs **8× Hopper** [O] | — | T2V, I2V | ✖ Impractical |
| SkyReels V2/V3 | Skywork Community License, commercial use allowed under its terms [O] | 14–19B | 52–82 GB | Wan-based | Diffusion forcing gives long video | R2V, V2V, A2V | ◐ Experimental |
| FramePack / F1 | Code Apache-2.0; **weights are a HunyuanVideo fine-tune, so Tencent territory terms likely apply** [UNVERIFIED] | 12.9B | 25.7 GB | ~6 GB [C] | Up to 60–120 s | I2V | ◐ Licence risk |
| Mochi 1, Open-Sora 2.0, Step-Video | Apache / MIT, but **Open-Sora bundles FLUX.1-dev weights (NC)** [O] | 10–30B | 69–133 GB | 20–80 GB | — | — | ✖ Superseded |

**Practical conclusions:**
- For a laptop + Colab setup, **Wan 2.2 TI2V-5B** for image-to-video is the only option that is high quality, commercially clean and fits a 24 GB-class GPU.
- **Colab T4 (16 GB, no bf16/FP8, Turing)** is marginal:
  - The 5B model *probably* runs in ComfyUI/WanGP with fp16/GGUF and an offloaded text encoder. Expect slow runs and possible fp16 artefacts [UNVERIFIED].
  - The free tier's ~12 GB system RAM is the bottleneck [C].
  - A14B on a T4 needs GGUF Q4 + the 4-step LoRA at 480p, and expert swapping on low RAM is the likely failure point [C].
  - **Plan around L4 (24 GB, bf16/FP8) or A100.**
- **No published T4/L4 timings for Wan 2.2 exist** in official sources. **Benchmark in Milestone M4** before committing to the budget.

### 2.2 Image models (keyframes and thumbnails)

| Model | Licence | Size | VRAM | Use |
|---|---|---|---|---|
| **[Z-Image-Turbo](https://github.com/Tongyi-MAI/Z-Image)** | **Apache-2.0** ✔ [O] | 6.15B; ~33 GB repo | Fits 16 GB consumer GPUs; 8 steps [O] | **Default keyframe model** (photoreal, fast) |
| **[FLUX.2 klein 4B](https://github.com/black-forest-labs/flux2)** | **Apache-2.0** ✔ [O] | 3.9B; 7.8 GB | ~13 GB; RTX 3090/4070+ [O] | Default alternative; multi-reference editing |
| [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | Apache-2.0 ✔ [O] | 12B; 23.8 GB | fp8/GGUF to ~12–16 GB [C] | Fallback; 1–4 steps |
| [Qwen-Image-2512 / Edit-2511](https://github.com/QwenLM/Qwen-Image) | **Apache-2.0** ✔ [O] | 20.4B; ~58 GB repo | Offload/GGUF; ~24 GB fp8 [C] | Best text rendering and editing; character sheets |
| Qwen-Image-2.1 / 2.1-Turbo (Sept–Oct 2026) | ✖ **Qwen Research License, non-commercial** [O] | 7.1B | — | Avoid for monetised use |
| FLUX.1-dev, FLUX.2-dev, FLUX.2 klein 9B | ✖ Non-commercial [O] | — | — | Avoid |
| SD 3.5 Large/Medium | Stability Community License (**< $1M revenue**) [O tag] | 8.1B / 2.5B | — | OK under the cap |
| SDXL | CreativeML Open RAIL++-M [O] | 2.6B | 8 GB | Runs on a T4; lower quality; huge LoRA ecosystem |
| HiDream-I1 / HiDream-O1-Image | MIT [O] (I1 uses a Llama-3.1 text encoder under the Llama licence [C]) | 17B / 8.8B | — | Alternative |
| HunyuanImage 3.0 | Tencent licence (territory-excluded) [O] | 83B MoE | Multi-GPU | ✖ |

**CPU-only image generation is impractical** on the reference laptop. A single 1024 px SDXL-Turbo image takes minutes and RAM is 7.4 GB [est]. Keyframes go to Colab or a rented GPU.

### 2.3 Camera movement, consistency, upscaling, interpolation

| Need | Best practical option | Licence | Status |
|---|---|---|---|
| **Cinematic camera movement** | (a) FFmpeg `zoompan`/`crop` on supersampled stills: push, pull, pan, tilt (deterministic, CPU). (b) Motion prompts in Wan I2V ("slow dolly forward"). (c) Wan 2.1 VACE / control (depth, pose) for precise moves | FFmpeg LGPL/GPL; Wan Apache | (a) **Verified, reliable**. (b) Usable, not exact. (c) Experimental |
| **Character / object consistency** | Bible with canonical text → same keyframe model + seed family → I2V from the approved keyframe; **character LoRA** (ai-toolkit, MIT, active) for recurring people; Qwen-Image-Edit / FLUX.2 klein multi-reference for variants | Apache/MIT | Usable. The LoRA approach is most reliable [C] |
| **Reference-image-based scenes** | Wan VACE (Apache, 17.3B, 75 GB), Phantom (Apache), FLUX.2 klein / Qwen-Image-Edit multi-reference | Apache | Experimental → usable |
| ✖ Face-ID tools | IP-Adapter-FaceID, InstantID, PuLID, ReActor all depend on **InsightFace models, which are non-commercial research only** [O] | NC | Avoid |
| **Video upscaling** | **SeedVR2** 3B/7B (Apache): ≤ 8 GB with GGUF + block swap, 12–16 GB FP8 [O-node]; **Real-ESRGAN** (BSD-3, fast, dormant since 2024) | Apache / BSD | Usable |
| **Frame interpolation** | **Practical-RIFE** (MIT, updated Aug 2026) [O]; FFmpeg `minterpolate` as a CPU fallback (lower quality) | MIT | **Mature** |
| **Motion consistency** | Keep clips ≤ 5 s; I2V from keyframes; avoid chaining more than 2 hops; QC with frame-difference and flicker metrics | — | Design rule |
| **Speed-ups** | LightX2V 4-step distill LoRAs for Wan (Apache): 14B in **8 GB VRAM + 16 GB RAM** at 480p/720p [O]; TeaCache/MagCache; FP8/GGUF | Apache | Usable; distillation slightly reduces motion [C] |
| **Low-VRAM runner** | **Wan2GP / WanGP** v17.17 (2026-10-07). Supports GTX 10xx/RTX 20xx, from 6 GB [O] | **WanGP Community License 2.0, not OSI**: free use, outputs sellable, **no hosted/SaaS/embedded use** without a licence [O] | Good for experiments. Check the licence before embedding it in a pipeline |
| **Node-based runner** | ComfyUI (GPL-3.0, Comfy-Org) with API mode | GPL-3.0 | Popular; heavier to automate than diffusers |

### 2.4 Open-source vs open-weight: the distinction

| Category | Meaning | Examples |
|---|---|---|
| **Open source (OSI licence on code + weights)** | Use, modify and redistribute, including commercially | Wan 2.1/2.2 (Apache), Kandinsky 5/6 (MIT), LongCat (MIT), Z-Image (Apache), FLUX.2 klein 4B (Apache), FLUX.1-schnell (Apache), Kokoro (Apache), ACE-Step 1.5 (MIT), Chatterbox (MIT), Qwen3-TTS (Apache) |
| **Open weights, custom licence with caps** | Downloadable, but commercial use is limited by revenue, users or registration | LTX-2.x ($10M), Stability Community ($1M), CogVideoX-5B (registration, 1M visits), IndexTTS (100M MAU), WanGP (no SaaS) |
| **Open weights, territory-excluded** | Licence does not apply in some regions, so use there is unlicensed | HunyuanVideo 1.5 / FramePack weights / HunyuanVideo-Foley (EU, UK, KR); MiniMax H3 (EU, UK, KR, **US**) |
| **Open weights, non-commercial** | Research or personal use only | FLUX.1-dev, FLUX.2-dev, Qwen-Image-2.1, MusicGen, AudioLDM2, Tango 2, MMAudio, F5-TTS, XTTS-v2, Fish S2, Higgs v3, Voxtral-4B-TTS, InsightFace models |

---

## 3. Audio generation

### 3.1 Text-to-speech

| Model | Code / weights licence | Commercial | Hardware | Cloning | Status | Verdict |
|---|---|---|---|---|---|---|
| **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** | Apache-2.0 / Apache-2.0 [O]; `kokoro-onnx` 0.6.1 (2026-08-19, MIT) | ✔ | **CPU** | No | Model v1.0 (2025-01) is frozen; packages maintained | **Default narrator** on the laptop. 8 languages, 54 voices; trained on permissive/synthetic data [O] |
| **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** 0.6B / 1.7B (2026-01) | Apache-2.0 / Apache-2.0 [O] | ✔ | GPU | 3 s reference | Active | **Best quality upgrade.** Natural-language style instructions, 10 languages |
| **[Chatterbox](https://github.com/resemble-ai/chatterbox)** Turbo / Multilingual V3 / Nano | MIT / MIT [O] | ✔ | GPU; Nano ~3× realtime on 8 CPU cores [O] | Zero-shot | Active | Strong. **All outputs carry the PerTh watermark** [O]. This is acceptable, and arguably good for disclosure |
| [CosyVoice 3](https://github.com/FunAudioLLM/CosyVoice) | Apache-2.0 [O] | ✔ (check each HF card) | GPU | Cross-lingual | Active | Multilingual alternative |
| [Kyutai TTS 1.6B](https://huggingface.co/kyutai/tts-1.6b-en_fr) / Pocket TTS 100M | Code MIT/Apache; **weights CC-BY-4.0** [O] | ✔ with attribution | Pocket: **CPU**, 2.3–2.5× realtime on 4 vCPU [O] | Restricted by design | Active | Good CPU alternative (English/French) |
| [Dia / Dia2](https://github.com/nari-labs/dia) | Apache-2.0 [O] | ✔ | ~4.4 GB GPU | Yes (README bans imitating people without consent) | Active | Dialogue rather than narration |
| [IndexTTS-2 / 2.5](https://github.com/index-tts/index-tts) | bilibili licence [O] | ✔ below 100M MAU / RMB 1B | ~8–12 GB | Yes; emotion + duration control | Active | Useful for timed dubbing |
| [Piper](https://github.com/OHF-Voice/piper1-gpl) (piper1-gpl v1.8.0) | **GPL-3.0** code [O]; **each voice has its own dataset licence** | ◐ GPL obligations if distributed; check each voice | CPU (ONNX) | No | Active | OK fallback; robotic compared with Kokoro |
| [Coqui TTS (idiap fork)](https://github.com/idiap/coqui-ai-TTS) v0.27.5 | MPL-2.0 code; **XTTS-v2 weights are CPML, non-commercial**, and Coqui is defunct [O/C] | ✖ for XTTS | GPU | Yes | Fork maintained | Avoid XTTS |
| F5-TTS, Fish Speech S2, Higgs Audio v3, Voxtral-4B-TTS | NC licences [O] | ✖ | — | — | — | Avoid |
| VibeVoice TTS (Microsoft) | **TTS code pulled 2025-09-05** [O] | ✖ | — | — | — | Avoid |
| MeloTTS, StyleTTS2, Orpheus, Bark | Stale / extra conditions / Llama licence | ◐ | — | — | Stale | Avoid |

**Voice cloning policy for this project:**
- **Do not clone real people's voices.** Colab bans "creating deepfakes" on all runtimes [O]. YouTube's likeness-detection tools now cover creators. Several model licences forbid imitation without consent.
- Use stock voices, or a voice designed from a text description (Qwen3-TTS VoiceDesign).
- Clone only **your own** voice, with a recorded consent note.

**Narration consistency and alignment:**
- **One voice ID per project**, stored in the plan.
- Synthesize **sentence by sentence**, which avoids drift on long passages and allows retries.
- Run an ASR word-error check per sentence with **faster-whisper** (MIT, 1.2.1).
- Get word timestamps from faster-whisper or **WhisperX** (BSD-2, 3.8.6, wav2vec2 alignment). **stable-ts** (MIT) can force-align to a known script.
- **aeneas** is AGPL and has been unmaintained since 2017, so avoid it.

### 3.2 Music, SFX, enhancement, mixing

| Need | Choice | Licence | Notes |
|---|---|---|---|
| **Music** | **[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)** | **MIT** ✔ [O] | Base model needs **< 4 GB VRAM**; under 10 s per song on a 3090 [O]. XL needs 20 GB+ |
| Music (alt) | DiffRhythm / DiffRhythm 2 | Apache-2.0 [O/C] | 8 GB with `--chunked`; full songs |
| Music (alt) | Stable Audio Open 1.0 / Stable Audio 3.0 Small & Medium | Stability Community, **< $1M revenue** [O] | ≤ 47 s (1.0); trained on licensed/CC data |
| ✖ Music | MusicGen (weights CC-BY-NC), YuE (NC except individual creators) | NC [O] | Avoid |
| **SFX** | **Freesound CC0** files (check the API terms), Stable Audio Open / SA3 Small-SFX (< $1M) | CC0 / Community | |
| ✖ SFX | MMAudio, AudioLDM2, Tango 2, ThinkSound (NC); HunyuanVideo-Foley (territory-excluded) | | Avoid |
| **Speech enhancement** | DeepFilterNet (MIT/Apache, real-time CPU, 48 kHz) | ✔ | Usually unnecessary for clean TTS |
| **Loudness + mix** | FFmpeg `loudnorm` (two-pass), `ebur128`, `sidechaincompress`, `amix`; **pyloudnorm** 0.2.0 for measurement | LGPL/GPL, MIT | **Mature.** −14 LUFS is a community convention for YouTube [C] |
| ✖ Separation | Demucs (archived 2025-01-01) | MIT | Only if needed; use the adefossez fork |

**Prototype recommendation:**
- Use a **curated library** of 10–20 CC0 or royalty-free music beds and a CC0 SFX pack, recording each file's licence in `licenses.json`.
- Add ACE-Step generation in Phase 4.
- This removes a GPU dependency and makes licensing auditable.

---

## 4. Video composition and editing

| Tool | Version (Oct 2026) | Licence | Role | Verdict |
|---|---|---|---|---|
| **[FFmpeg](https://ffmpeg.org/download.html)** | 9.0.2 (8.1.x / 8.0.x maintained) [O] | LGPL/GPL | Trims, scaling, `zoompan` camera moves, `xfade` transitions, `lut3d` grading, `ass` subtitles, audio mix, loudness, encoding | **Core renderer** |
| **[typed-ffmpeg](https://pypi.org/project/typed-ffmpeg/)** | 4.5 (2026-08) [O] | MIT | Typed Python builder for filter graphs | **Use**, instead of the unmaintained ffmpeg-python (0.2.0, 2019) |
| [PyAV](https://pypi.org/project/av/) | 19.0.1 (2026-10-03) [O] | BSD-3 | Frame-level access (QC metrics, custom effects) | Use in QC |
| [MoviePy](https://pypi.org/project/moviepy/) | 2.2.1 (2025-05) [O] | MIT | Simple Python compositing | ✖ for the pipeline: slow, high RAM, no release in 17 months |
| [Remotion](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md) | 4.0.534 [O] | Custom: free for individuals and ≤ 3-employee companies | React-based motion graphics | Optional for animated graphics; Node dependency |
| [Blender / bpy](https://pypi.org/project/bpy/) | 5.2.2 LTS; `bpy` wheel 5.2.2 [O] | GPL-3.0 | Scripted 3D scenes (pyramid construction diagrams, orbits, maps in 3D), headless render | Phase 4, for explanatory 3D shots. CPU Cycles is slow on the laptop; EEVEE needs a GPU |
| [PySceneDetect](https://pypi.org/project/scenedetect/) | 0.7.1 [O] | BSD-3 | Cut detection in QC | Use |
| [auto-editor](https://github.com/WyattBlue/auto-editor) | 31.7.2 (GitHub) [O] | Unlicense | Silence cutting | Not needed |
| Pillow / Matplotlib / Cairo | — | permissive | Title cards, maps, timelines, diagrams | Use |

**Encoding choices:**
- **CPU libx264** (CRF 18, preset slow/medium) is the default. It is deterministic, high quality, and fast enough for 1–5 minute videos on the laptop: about 0.5–2× realtime at 1080p preset medium on an i5-1335U [est].
- **GPU NVENC:**
  - T4: 1 engine, H.264/HEVC, no AV1. L4: 2 engines plus AV1 [O].
  - Use it on Colab when the render runs there. Whether Colab's stock FFmpeg includes `h264_nvenc` is [UNVERIFIED]; check with `ffmpeg -encoders | grep nvenc`.
  - NVENC needs a higher bitrate than x264 for the same quality.
- **Intel Quick Sync (VA-API)** on the laptop's UHD graphics can encode H.264/HEVC via `h264_vaapi`/`h264_qsv` [est, untested]. It is an optional speed-up for drafts.
- **When to use Blender:** only for shots that are 3D diagrams or explanations needing geometric accuracy, which diffusion models cannot guarantee. Examples: ramp theories, a cutaway of the pyramid interior, star-field camera paths.

---

## 5. Agent frameworks and orchestration (summary; detail in doc 03 §6)

| Framework | Version (2026-10) | Status |
|---|---|---|
| LangGraph | 1.2.14 | Active, MIT; checkpointing + human-in-the-loop |
| CrewAI | 1.15.27 | Active; role-play abstraction |
| AutoGen | 0.7.5 | **Maintenance mode**. Microsoft points new users to Agent Framework (1.21.0) [O] |
| PydanticAI | 2.55.0 | Active, MIT; typed outputs, durable-execution integrations |
| smolagents | 1.26.0 | Slowing |
| OpenAI Agents SDK / Claude Agent SDK | 0.23.1 / 0.2.165 | Active; vendor-centric |
| Prefect / DBOS / Temporal / Hatchet | 3.8.8 / 3.2.0 / 1.34.0 / 1.42.1 | Durable workflow engines |

**Finding:** the workflow is a fixed pipeline with known stages. A plain Python stage runner with file checkpoints is the simplest reliable choice. Frameworks add value only for (a) multi-machine durable execution, which is DBOS/Prefect territory, or (b) open-ended tool loops inside research, which is PydanticAI or the LLM vendor's tool loop.

---

## 6. Execution environments (summary; detail in doc 04)

| Environment | Facts |
|---|---|
| **Reference laptop** (checked 2026-10-11) | i5-1335U (10 cores / 12 threads), **7.4 GB RAM** (~1.7 GB free at check time), Intel UHD (no CUDA), 265 GB free disk, Python 3.12.3. **FFmpeg, Docker/Podman and rclone are not installed** |
| **Colab** | Session ≤ 12 h (24 h on Pro+ with compute units); idle timeouts; *"resources are not guaranteed"*; GPU types vary [O]. Paid plans are about $9.99 per 100 compute units [C]. Approximate burn: T4 ~1.2 CU/h, L4 ~1.7, A100-40 ~5.4, A100-80 ~7.5, G4 (RTX PRO 6000) ~8.7 [C, volatile]. **Free tier bans "running distributed computing workers" and "bypassing the notebook UI"**; all tiers ban deepfakes and web-service hosting [O]. Since 2026-09-22 Colab benefits are included in Google AI plans [O]. Google released a Colab CLI in June 2026 (tier requirements [UNVERIFIED]) |
| **Google Drive** | Mount fails if a folder has more than ~10k items; per-user operation and bandwidth quotas cause I/O errors [O]. Buying more Drive space does not increase Colab disk |
| **Rented GPUs** (volatile) | RunPod 4090 $0.34–0.74/h, L40S $0.79–1.09, A100-80 $1.19–1.59, H100 $1.99–3.49; Vast H100 ~$1.67 [C]; Modal L4 $0.80, A100-40 $2.10, H100 $3.95, with $30/month free credit [O] |

---

## 7. YouTube (summary; detail in doc 07)

- **Quota:** `videos.insert` uses a dedicated bucket (100 calls/day, 1 unit each). The other operations share 10,000 units/day [O].
- **Private restriction for unaudited projects:** the official pages conflict. The old 2020 rule forces uploads to private; the current `videos.insert` page says unverified projects are *not* restricted [O]. This is moot for a private-by-default design.
- **Disclosure:** `containsSyntheticMedia` exists (added 2024-10-30). The disclosure policy requires it for realistic synthetic scenes [O].
- **Monetisation:** the policy is now called **"inauthentic content"** (2025-07-15). The spam policy explicitly targets *"automated or synthetic mass-production"* [O].

---

## 8. Reference projects

| Project | What it shows | Limitation |
|---|---|---|
| [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) | Topic → script → stock footage → TTS → subtitles → MP4 | Stock footage, no generation, no QC |
| [ShortGPT](https://github.com/RayVentura/ShortGPT) | Shorts automation framework | Stale, stock-based |
| [short-video-maker](https://github.com/gyoridavid/short-video-maker) | Remotion + Kokoro shorts | Short form only |
| ComfyUI workflows for Wan 2.2 | Practical I2V settings | GUI-centred |

None of these covers research-grounded planning, generated visuals with continuity, QC and resume together. Their value is in plumbing ideas: subtitle styling, TTS chunking and FFmpeg recipes.

---

## 9. Verified vs experimental

| Verified / mature (build on these) | Usable with care | Experimental (Phase 4+) |
|---|---|---|
| LLM structured JSON plans | Wan 2.2 TI2V-5B I2V on L4/A100 | Long continuous shots (LongCat, SkyReels DF, FramePack) |
| Kokoro / Qwen3-TTS narration, faster-whisper alignment | Wan 2.2 A14B GGUF on 16–24 GB | Wan 2.2 on a Colab T4 |
| Still keyframes + FFmpeg camera moves | Character LoRA consistency | Kandinsky 6 / LTX-2.5 native audio |
| FFmpeg edit, mix, loudness, subtitles, encode | SeedVR2 upscaling | VACE/Phantom reference-to-video |
| YouTube resumable upload, OAuth, private default | ACE-Step music | Blender scripted 3D explainer shots |
| Hash-based resume, job files over Drive | VLM-based visual QC | Fully local open-LLM planning on CPU |

---

## 10. Open items to verify in Phase 2/3

1. Actual Wan 2.2 TI2V-5B time per clip on Colab L4, A100 and T4 (none published).
2. Whether Colab's FFmpeg has NVENC, and real Colab RAM/disk on your plan.
3. Colab CLI tier requirements, and whether an attended notebook loop processing a queue is acceptable on your tier.
4. Custom thumbnail eligibility on your channel, and whether unaudited API uploads can be made public.
5. FramePack weights licence; Kandinsky 6 practical VRAM; LTX-2.5 minimum VRAM.
6. Licence of each chosen Piper voice, if Piper is used.

---

## 11. Sources

**Video:**
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) · [Wan2.1](https://github.com/Wan-Video/Wan2.1) · [Wan-AI on HF](https://huggingface.co/Wan-AI)
- [LTX-2 + LICENSE-2_x](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x)
- [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) · [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Kandinsky 6 Lite](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers) · [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) · [CogVideoX licence](https://huggingface.co/zai-org/CogVideoX-5b-I2V/blob/main/LICENSE)
- [MAGI-2-preview](https://huggingface.co/sand-ai/MAGI-2-preview) · [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) · [FramePack](https://github.com/lllyasviel/FramePack)
- [LightX2V](https://github.com/ModelTC/LightX2V) · [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) · [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- [Wan 3.0 coverage (C)](https://datanorth.ai/news/alibaba-launches-wan3-0-video-model) · [Wan2.1 GPU timings (C)](https://instasd.com/post/wan2-1-performance-testing-across-gpus) · [T4 GGUF repo (C)](https://huggingface.co/Rob1221rib/wan22-fp16-i2v-gguf)

**Image:**
- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) · [FLUX.2](https://github.com/black-forest-labs/flux2) · [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) · [Qwen-Image](https://github.com/QwenLM/Qwen-Image)
- [InsightFace (licence note)](https://github.com/deepinsight/insightface) · [ai-toolkit](https://github.com/ostris/ai-toolkit) · [VACE](https://github.com/ali-vilab/VACE) · [Phantom](https://github.com/Phantom-video/Phantom)

**Post-processing:**
- [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) · [SeedVR](https://github.com/ByteDance-Seed/SeedVR) · [SeedVR2 ComfyUI node](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) · [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

**Audio:**
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) · [kokoro-onnx](https://pypi.org/project/kokoro-onnx/) · [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [Chatterbox](https://github.com/resemble-ai/chatterbox) · [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [Kyutai TTS](https://huggingface.co/kyutai/tts-1.6b-en_fr) · [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) · [Dia](https://github.com/nari-labs/dia) · [IndexTTS](https://github.com/index-tts/index-tts)
- [Piper (piper1-gpl)](https://github.com/OHF-Voice/piper1-gpl) · [Piper voice card example](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/lessac/medium/MODEL_CARD) · [idiap/coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS)
- [F5-TTS](https://github.com/SWivid/F5-TTS) · [Fish Speech licence](https://raw.githubusercontent.com/fishaudio/fish-speech/main/LICENSE) · [VibeVoice](https://github.com/microsoft/VibeVoice)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [WhisperX](https://pypi.org/project/whisperx/) · [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) · [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0) · [Stability licence update](https://stability.ai/news/license-update) · [DiffRhythm](https://github.com/ASLP-lab/DiffRhythm) · [YuE](https://github.com/multimodal-art-projection/YuE)
- [MMAudio](https://github.com/hkchengrex/MMAudio) · [HunyuanVideo-Foley licence](https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanVideo-Foley/main/LICENSE) · [ThinkSound](https://github.com/FunAudioLLM/ThinkSound)
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) · [pyloudnorm](https://pypi.org/project/pyloudnorm/)

**Composition:**
- [FFmpeg](https://ffmpeg.org/download.html) · [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html) · [typed-ffmpeg](https://pypi.org/project/typed-ffmpeg/) · [PyAV](https://pypi.org/project/av/) · [MoviePy](https://pypi.org/project/moviepy/)
- [Remotion licence](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md) · [bpy](https://pypi.org/project/bpy/) · [NVENC support matrix](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new) · [PySceneDetect](https://pypi.org/project/scenedetect/)

**LLM, orchestration and agents:**
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs) · [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) · [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)
- [AutoGen notice](https://github.com/microsoft/autogen) · [LangGraph](https://docs.langchain.com/oss/python/langgraph/durable-execution) · [DBOS](https://docs.dbos.dev/) · [Temporal](https://docs.temporal.io/develop/python)

**Compute:**
- [Colab FAQ](https://research.google.com/colaboratory/faq.html) · [Colab in Google AI plans](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/) · [Colab CLI](https://github.com/googlecolab/google-colab-cli) · [Colab Enterprise pricing](https://cloud.google.com/colab/pricing)
- [RunPod](https://www.runpod.io/pricing) · [Vast](https://vast.ai/pricing) · [Lambda](https://lambda.ai/pricing) · [Modal](https://modal.com/pricing)
- [Colab CU rates (C)](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)

**YouTube:** see `07_YOUTUBE_AUTOMATION.md §8`.

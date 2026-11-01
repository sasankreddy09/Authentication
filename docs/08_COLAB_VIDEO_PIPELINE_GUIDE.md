# The Colab AI Video Pipeline — Complete Guide

How the video engine in `colab/` works, why it is built this way, every problem met since the first video and how each was solved. Read this before building a new video or changing the engine.

> Current versions: `build_notebook.py` → notebook **v15** (narrated explainers), `build_story_notebook.py` → **story v5** (characters with voices, optional diagram shots). Both share one engine.

---

## 0. Context for AI assistants (read this first)

> Paste this whole file into another AI. This section is the short version: enough to work on the project safely. The rest of the document is the detail.

**What this is.** A pipeline that makes YouTube explainer / story videos in **Google Colab** (free T4 GPU) from a JSON **plan**. Output: `final.mp4` (1080p, voices, AI video, drawn diagrams, music, sound effects, subtitles) + upload text. The user runs the notebook in Colab; development happens on a Linux machine **without a GPU**.

**The one rule of the architecture.** The AI writes **data** (a plan JSON); a **fixed, tested engine** turns it into a video. Do **not** generate new Python per video. Engine changes go into the two builder scripts, are tested, and apply to every video. Shots use a small recipe menu (`video` = AI clip, `diagram` = drawn by code; fallback = still image + camera move). Unsupported requests fall back and are reported — they never block the run.

**Where things are.**

| Path | What |
|---|---|
| `colab/build_notebook.py` (~2000 lines) | **The engine.** Every shared notebook cell as a string: `code(src, tag)`. Run directly → narrator-only notebook. Exposes `CELLS` and `write_notebook()` for import. |
| `colab/build_story_notebook.py` (~420 lines) | Notebooks for plans with **characters** and/or **diagrams**. Imports the shared cells via `runpy`, replaces `settings`, `plan`, `narration` (voices), `timeline`, `metadata`. |
| `colab/*_plan.json` | One plan per video (`black_holes_plan.json` narrator-only, `story_plan.json` characters, `zomato_order_plan.json`, `food_order_backend_plan.json` characters + diagrams). |
| `colab/colab_*.ipynb` | Generated notebooks — never edit by hand, rebuild. |
| `colab/tests/` | CPU test runner and targeted tests (`README.md` there has the exact commands). |
| `docs/08_COLAB_VIDEO_PIPELINE_GUIDE.md` | This file. `docs/01–07` = original research/plans (older; this file wins where they differ). |

**Build:** `python3 build_notebook.py plan.json out.ipynb` or `python3 build_story_notebook.py plan.json out.ipynb`. Bump `NOTEBOOK_VERSION` (engine: `"v15 (...)"`, story: `"story v5 (...)"`) and the markdown header when behaviour changes; the user checks it in cell 1.

**Notebook cell order (tags):** `settings → log → install → setup → plan → helpers → prefetch → narration → timeline → diagrams → keyframes → editlib → clips → music → sfx → render → qc → metadata → preview`. Cells run top to bottom in one Python namespace (later cells use globals from earlier ones).

**Key names you will meet in the code.**

| Name | Meaning |
|---|---|
| `PLAN`, `SHOTS`, `TIMELINE`, `TOTAL`, `PLACED` | Plan, shot list, per-shot `{shot_id, start, dur, frames, mood}`, video length, placed voice lines |
| `PROJ`, `WORK`, `MODEL_CACHE` | Drive project folder, local scratch, shared model cache on Drive |
| `VIDEO_ENGINE` (`"wan21-1.3b"` T4 / `"wan22-5b"` L4+ / `None`), `VIDEO_MODE` (`t2v`/`i2v`), `GPU` | Chosen hardware path |
| `ff()`, `sidecar()`, `read_sidecar()`, `log()`, `log_exc()`, `ram()`, `free_gpu()` | FFmpeg runner (logged), JSON next to a file (`x.wav.json`), logging, readouts |
| `speak()`, `listen()`, `speech_error()` | Kokoro TTS, Whisper transcription, mismatch score (word + letter level) |
| `encode_prompts()`, `emb_path()`, `video_parts()`, `load_video_model()`, `speed_up()`, `make_clip()`, `frames_for()`, `write_clip()` | Video pipeline: prompt cache, model cache, turbo LoRA, generation, frames on demand, safe clip writer |
| `clip_key()`, `clip_info()` | Is a finished clip valid for the current plan? (sidecar comparison) |
| `presence()`, `missing_from()` | CLIP continuity check (character visible?) |
| `render_diagram()`, `make_diagram()` | Drawn diagram shots |
| `segment_recipe()`, `render_segment()`, `edit_in_background()` | Per-shot edit (FFmpeg graph) with content-hash cache |
| `prefetch_jobs()`, `wait_for_prefetch()`, `need_disk()`, `purge_hf()` | Background downloads, disk guard and cleanup |
| `CLIP_WRITER`, `SEG_RECIPE_VERSION`, `DIAGRAM_VERSION` | Bump to invalidate old cached clips / edits / diagrams |

**Conventions (follow them in any change).**

- **Cache by content:** every output has a sidecar or a hash in its file name; re-runs remake only what changed. If you change how something is made, bump the matching version constant.
- **Never block:** wrap risky steps, `log_exc()` the full traceback, fall back, report in `STATUS` / `qc_report.json`. No bare `except: pass`.
- **Log everything** (user requirement): prints, commands, tracebacks, timings, RAM/GPU go to `logs/run_*.log`.
- **Colab limits:** keep system RAM < ~6 GB (load models with `device_map={"": 0}`, never through RAM), one big model on the GPU at a time, check disk before downloads, delete big downloads once cached.
- **Text on screen is drawn by code**, never by the AI model. AI prompts never contain text, logos or brand names.
- **Plans are validated** in the plan cell before any GPU work; add a check there for any new plan field.
- **Engine code is shared**: a fix in `build_notebook.py` must not break `build_story_notebook.py` (rebuild and test both).
- The user wants generated docs/reports as **local files**, not web artifacts.
- **Plan copy on Drive wins:** the notebook uses `MyDrive/ai-video/<project>/plan.json` if its `plan_version` is equal or newer than the built-in plan. After editing a plan locally, **bump `plan_version`** or the user's Colab keeps the old one (the old copy is kept as `plan.vN.backup.json`).
- **Project folder** comes from the plan (`project`, or `project_id` in older plans). Changing it starts a fresh project (nothing cached).
- **Scripts must be concrete, not generic** (user requirement). Follow **one real example from start to finish** with named characters and a named thing, and say plainly *who does what*: "Ananya opens Zomato and orders a chicken biryani", not "You tap 'Place Order' and a chain of people springs into action". No abstract words ("an algorithm", "the partner", "a chain of systems") without saying who or what it is. Step labels use the same simple words ("Step 3 · Zomato finds a rider"). Details in section 6, "Writing a script viewers understand".
- **Names and brand words must be pronounced and heard correctly.** The voice (Kokoro) guesses unknown words. Check its pronunciation with `kokoro_onnx.tokenizer.Tokenizer().phonemize(text, "en-us")` and fix it with the line's `tts` field: subtitles show `text`, the voice reads `tts` (e.g. `Zomato` → `"Zo-maato"`). List every name in the plan's `vocabulary`, so the speech check (Whisper) is told them up front and doesn't mishear them ("Arjun" → "Our gun").

**How to test (no GPU here).** Rebuild → compile every cell → run the notebook on CPU with a small runner that executes cells in order, skips `install`, points `DRIVE_ROOT` to a local folder, uses Whisper `tiny.en`, and injects fake clips (FFmpeg test patterns + valid sidecars). GPU-only code is tested on a 2-layer Wan transformer with real layer sizes on CPU. Details in section 15.

**Current state.** Narrator notebook v15, story notebook v5. Everything runs end to end on CPU tests. **Not yet verified on a real GPU:** actual Wan output quality with the latest changes, Kokoro/Whisper on CUDA, NVENC, the CLIP continuity threshold (0.2) on real cartoon frames, real time per clip with frames on demand. The previous Pip story notebook (`colab_story_v3.ipynb`) is on the older engine; rebuilding it needs its 8 long shots split (the plan check rejects them).

**Videos so far.**

| Video | Plan → notebook | State |
|---|---|---|
| How black holes are formed (narrator) | `black_holes_plan.json` → `colab_make_video_v15.ipynb` | Engine v15; CPU-tested |
| Pip and the Fallen Star (story, 4 characters) | `story_plan.json` → `colab_story_v3.ipynb` | Old engine; needs long shots split before rebuilding |
| What happens when you order food on Zomato (explainer, 3 characters + narrator, 16 shots, ~71 s) | `zomato_order_plan.json` (**plan_version 3**) → `colab_zomato_order_v1.ipynb` | Script rewritten to be concrete (Ananya orders a chicken biryani → Chef Raju at Raju's Kitchen → rider Arjun); all lines pass the speech check (0% wrong words); CPU-tested end to end |
| Food-order backend (characters + diagrams) | `food_order_backend_plan.json` → `colab_food_backend_v1.ipynb` | Story v5 engine |

**Typical tasks and where to do them.**

| Task | Where |
|---|---|
| New video | Write a plan JSON (section 6), build with the right builder |
| New sound effect | `make_sfx()` in the `sfx` cell + `SFX_TYPES` in both plan cells |
| New shot recipe | Plan check (both builders), `segment_recipe()` branch, QC counting, prefetch (skip if no GPU needed) |
| New voice effect / tone effect | `VOICE_FX` / `TONE_FX` in the story `narration` cell + allowed sets in the story plan cell |
| Faster video | `make_clip()` / `frames_for()` / `speed_up()` in the `clips` cell; measure with the "about N min left" line |
| Something broke in Colab | Ask for `MyDrive/ai-video/<project>/logs/run_*.log` |

---

## Contents

0. [Context for AI assistants](#0-context-for-ai-assistants-read-this-first)
1. [What the pipeline does](#1-what-the-pipeline-does)
2. [Quick start: making a new video](#2-quick-start-making-a-new-video)
3. [The architecture in one picture](#3-the-architecture-in-one-picture)
4. [Why this architecture (and not the alternatives)](#4-why-this-architecture-and-not-the-alternatives)
5. [Files and folders](#5-files-and-folders)
6. [The plan: the only thing that changes per video](#6-the-plan-the-only-thing-that-changes-per-video)
7. [The workflow, step by step (every notebook cell)](#7-the-workflow-step-by-step-every-notebook-cell)
8. [Models and libraries](#8-models-and-libraries)
9. [Running on a free Colab T4: GPU, RAM and disk](#9-running-on-a-free-colab-t4-gpu-ram-and-disk)
10. [Caching: what is saved and where](#10-caching-what-is-saved-and-where)
11. [Speed: where the time goes and what makes it fast](#11-speed-where-the-time-goes-and-what-makes-it-fast)
12. [Quality guards: how bad output is caught](#12-quality-guards-how-bad-output-is-caught)
13. [Logging](#13-logging)
14. [Every problem we hit, and how it was solved](#14-every-problem-we-hit-and-how-it-was-solved)
15. [How the code is tested without a GPU](#15-how-the-code-is-tested-without-a-gpu)
16. [Known limits](#16-known-limits)
17. [Troubleshooting](#17-troubleshooting)
18. [Rules that must not be broken](#18-rules-that-must-not-be-broken)

---

## 1. What the pipeline does

One Google Colab notebook turns a **plan** (a JSON file describing the video) into a finished, upload-ready video, end to end:

```
plan.json ─► voices ─► timeline ─► visuals (AI video / drawn diagrams) ─► music + sound effects ─► edit ─► checks ─► final.mp4 + YouTube text
```

Output, in `MyDrive/ai-video/<project>/render/`:

| File | What it is |
|---|---|
| `final.mp4` | 1920×1080, 24 fps, H.264, AAC audio at −14 LUFS (YouTube loudness), subtitles burned in |
| `subtitles.srt` | Word-timed subtitles (coloured per speaker in story videos) |
| `metadata.json` | Title, description (with sources and AI disclosure), tags, "Private", "altered or synthetic content: Yes" |
| `qc_report.json` | Every automatic check and its result |
| `timeline.json`, `script.txt` | Exact timing of every shot and line; the script in screenplay form |

Everything runs **on a free Colab T4** (or faster on L4/A100). Nothing is uploaded automatically; you upload by hand as **Private**.

---

## 2. Quick start: making a new video

1. **Write a plan** (`colab/<name>_plan.json`) with its own `"project"` name (the Drive folder). Ask Claude to write it, or copy an existing one: `black_holes_plan.json` (narrator only), `story_plan.json` (characters), `food_order_backend_plan.json` (characters + drawn diagrams). The format is in [section 6](#6-the-plan-the-only-thing-that-changes-per-video).
2. **Build the notebook** on your computer:
   ```bash
   cd colab
   python3 build_notebook.py my_plan.json colab_my_video.ipynb          # narrator-only plans
   python3 build_story_notebook.py my_plan.json colab_my_video.ipynb    # plans with characters / diagrams
   ```
3. **Open it in Colab**: upload the `.ipynb`, **Runtime → Change runtime type → T4 GPU** (or L4/A100), then **Runtime → Run all**, allow Google Drive.
4. **Watch the first lines**: cell 1 prints the version, cell 3 prints the RAM / disk / Drive budget and what is already cached, the plan cell prints the continuity check.
5. **Get the result** from `MyDrive/ai-video/<project>/render/`. If anything failed, send the log file from `MyDrive/ai-video/<project>/logs/`.
6. **Re-run any time**: everything already made is reused; only what changed is redone. Edit `plan.json` in the Drive project folder to change lines, prompts or voices.

---

## 3. The architecture in one picture

```
            ┌──────────────────────── written per video (data) ────────────────────────┐
            │  plan.json: shots, lines, voices, tones, moods, prompts, diagrams, sources │
            └───────────────────────────────────┬───────────────────────────────────────┘
                                                │  (validated before any GPU work)
┌───────────────────────────────── fixed, tested engine (code) ─────────────────────────────────┐
│                                                                                                 │
│  settings → run log → install → GPU/Drive/budget → plan + checks → helpers                      │
│        │                                                                                        │
│        ├─► background downloads (text encoder, video model) ──────────────┐                     │
│        │                                                                  ▼                     │
│        ├─► voices (Kokoro) + speech check (Whisper) ─► timeline ─► diagrams (drawn, CPU, bg)    │
│        │                                                     │                                  │
│        │                                                     ▼                                  │
│        │              prompts → text encoder (GPU, once) → Wan video (GPU) → continuity check  │
│        │                                                     │   (CLIP)  → remake if needed    │
│        │                                                     ▼                                  │
│        │                                   edit each shot in background (FFmpeg, CPU)          │
│        ├─► music (generated) + sound effects (generated)                                       │
│        ▼                                                                                        │
│   final edit: join → mix (ducking, loudness) → subtitles → final.mp4 → quality checks → text    │
│                                                                                                 │
│  every step: cached on Drive · logged · falls back instead of stopping                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Three ideas hold the whole thing together:**

1. **Plan = data, engine = fixed code.** The AI (Claude) only writes the plan. The code that runs it is the same, tested code for every video.
2. **A small menu of "recipes".** A shot is one of: AI video (`video`), drawn diagram (`diagram`); fallback is a still image with a camera move. Unknown things are rejected by the plan check, not improvised.
3. **Never block, never lose work.** Every finished piece is saved to Drive with a "sidecar" (`.json`) describing how it was made. Re-runs skip what is done. A failure in one shot uses a fallback and is reported, the run continues.

---

## 4. Why this architecture (and not the alternatives)

| Decision | What we chose | Alternatives considered | Why |
|---|---|---|---|
| Who writes what | Claude writes **only the plan (JSON)**; a fixed engine runs it | Claude writes a new Python program per video | New code per video = new bugs per video, no testing, no reuse. A plan can be validated in milliseconds; the engine is tested once and improves for every video. |
| Where it runs | **Google Colab** notebook | Own GPU server, cloud GPU APIs (Runway, Pika, Sora…), Kubernetes/workers | Free/cheap GPUs, nothing to install locally, no paid per-second video API. Colab bans distributed worker setups on the free tier, so one notebook it is. |
| State / storage | **Files on Google Drive + JSON sidecars** | A database (Postgres, DBOS, SQLite) | A single person's pipeline doesn't need a database. Files on Drive survive Colab disconnects, are visible and editable by hand, and a sidecar next to each file is enough to know "is this done and was it made with these settings?". |
| Resume after a crash | Content-based sidecars (re-make only if inputs changed) | Workflow engine, checkpoints | Colab sessions die (RAM, time limits). With sidecars, "Run all" again is the resume button. |
| Video model | **Wan 2.1 T2V 1.3B** on T4, **Wan 2.2 TI2V 5B** on L4/A100 | HunyuanVideo, LTX-Video, CogVideoX, Mochi, paid APIs | Apache-2.0 licence (commercial use OK), runs on a 15 GB T4 (1.3B), good quality, first-class diffusers support. |
| Speed-up on T4 | **Self-Forcing DMD LoRA**, 4 steps, no CFG | TeaCache/First Block Cache, SageAttention, TurboDiffusion, torch.compile | ~10x less work per clip and Apache-2.0 upstream. Caches barely help at 4 steps; SageAttention on T4 only exists as unofficial builds not faster than the default; TurboDiffusion needs 4090/40 GB+ GPUs; torch.compile recompiles when clip length changes. (First Block Cache is kept as the `"fast"` mode.) |
| Voice | **Kokoro-82M via kokoro-onnx** | Coqui XTTS, Bark, Piper, ElevenLabs | Apache-2.0, natural, fast on CPU, many stock voices, runs on Python 3.13. No voice cloning (safer, and not allowed on Colab free tier). |
| Speech check / subtitle timing | **faster-whisper** | OpenAI Whisper, WhisperX | Fast (CTranslate2), word timestamps, MIT. |
| Editing | **FFmpeg** with fixed filter graphs | MoviePy, Remotion, After Effects, OpenTimelineIO tools | FFmpeg is fast, already on Colab, scriptable, exact. Remotion needs a paid licence for some commercial use and a Node toolchain. |
| Music | **Generated in code** (numpy + FFmpeg synths) | Stock music, AI music models (MusicGen) | No licensing risk, follows the mood shot by shot, seconds to make. MusicGen is CC-BY-NC (non-commercial). You can still drop your own `music.mp3` in the project folder. |
| Diagrams / technical shots | **Drawn by code** (Pillow → FFmpeg) | AI video of "servers and databases" | AI video can't draw readable text or correct diagrams. Drawing is exact, sharp and takes ~5 s per shot with no GPU. |
| Text on screen | **Drawn by code** (titles, step labels, subtitles) | Asking the AI model to draw text | AI models garble text. |
| Upload | **By hand, Private** | Auto-upload via YouTube API | Safer while the pipeline is new; disclosure ("altered or synthetic content") must be set correctly; API upload quota is limited. |

---

## 5. Files and folders

**On your computer (`colab/`):**

| File | Role |
|---|---|
| `build_notebook.py` | **The engine.** Builds the narrated-explainer notebook and holds every shared cell (log, install, setup, helpers, downloads, diagrams, keyframes, editing, video, music, sfx, render, checks). |
| `build_story_notebook.py` | Builds notebooks for plans with **characters** (own voices, tones, subtitles colours) and/or **diagram shots**. Imports all shared cells from `build_notebook.py`, adds its own settings / plan / voices / timeline / YouTube-text cells. |
| `*_plan.json` | One plan per video. |
| `colab_*.ipynb` | Built notebooks (generated; rebuild after changing a builder or plan). |
| `tests/` | Test runner and targeted tests for a computer without a GPU (see `tests/README.md`). |

**On Google Drive (`MyDrive/ai-video/`):**

```
ai-video/
├── _model_cache/                 shared by all projects
│   ├── wan21-1.3b-float16/       video model, half precision (3.3 GB)
│   ├── kokoro/                   voice model (~350 MB)
│   ├── whisper/                  speech-check model
│   ├── hf/                       CLIP (continuity check), turbo LoRA
│   └── prompt_cache/             encoded prompts (so the 11 GB text encoder is rarely needed)
└── <project>/
    ├── plan.json                 your editable copy of the plan
    ├── narration/                one .wav per line + .json sidecar (text, voice, speed, speech-check result, word timings)
    ├── keyframes/                still images (only when needed) + sidecars
    ├── clips/                    AI video clips + sidecar per shot (prompts, seed, frames, continuity scores)
    ├── diagrams/                 drawn diagram animations (file name = content hash)
    ├── segments/                 each shot fully edited (file name = content hash)
    ├── render/                   final.mp4, subtitles.srt, metadata.json, qc_report.json, timeline.json, script.txt
    └── logs/                     run_<date>.log — everything that happened
```

---

## 6. The plan: the only thing that changes per video

A plan is JSON. The plan cell checks it before any GPU time is spent and stops with a clear list of problems.

**Common parts**

```jsonc
{
  "plan_version": 1,                       // bump it to replace the copy on Drive
  "project": "my_video_v1",                // Drive folder name
  "title": "...",
  "style":  { "suffix": "...visual style added to every AI prompt...", "negative": "...what to avoid..." },
  "moods":  { "calm": { "pause_after": 0.6, "grade": "neutral", "move_amount": 0.06,
                        "music": { "base": 0.5, "curious": 0.7, "tension": 0.0, "shimmer": 0.2 } } },
  "shots":  [ ... ],
  "claims": [ { "id": "c1", "text": "...", "sources": ["https://..."] } ],   // go into the description
  "vocabulary": ["Zomato", "Ananya", "Arjun", "biryani"],   // names/special words: given to the speech check
  "metadata": { "title": "...", "summary": "...", "disclaimer": "...", "category": "...", "audience": "...", "tags": [] }
}
```

- `summary` starts the YouTube description, then `claims` sources, then `disclaimer`; `category` defaults to Science & Technology. `audience` is used by the story builder.
- `plan_version`: see "Plan copy on Drive wins" in section 0.
- `vocabulary`: names and unusual words. They are passed to Whisper as `initial_prompt` (`ASR_PROMPT`) so the speech check spells them right. The story builder adds the character names automatically. Both builders support it.

**Narrator-only shot** (`build_notebook.py`). The voice is the `VOICE` setting (cell 1); each mood also needs a `"speed"` (voice speed for that mood):

```jsonc
{ "shot_id": "sh03", "recipe": "video", "mood": "tension", "seed": 103,
  "image_prompt": "what the scene looks like", "motion_prompt": "what moves",
  "sentences": [ { "id": "n03", "text": "What the narrator says.", "pause_after": 0.6 } ],
  "sfx": [ { "type": "boom", "at": 2.0, "dur": 1.5, "gain_db": -12 } ],
  "title": "optional big title", "fade_in": true, "flash_in": false, "move": "push_in" }
```

**Character plan** (`build_story_notebook.py`) adds characters and tones:

```jsonc
"characters": {
  "narrator": { "name": "Narrator", "voice": "bm_fable", "speed": 0.95, "pitch": 0, "fx": "none", "color": "#FFFFFF", "look": null },
  "pip": { "name": "Pip", "voice": "af_sky", "pitch": 1.5, "color": "#FFB347",
           "look": "a small red fox cub with big amber eyes ...",      // repeated in every prompt = consistency
           "tag": "a red fox cub" }                                   // what the continuity check looks for
},
"tones": { "excited": { "speed": 1.12, "pitch": 1.0, "gain_db": 1.5, "pause_after": 0.35 },
           "sad":     { "speed": 0.86, "pitch": -0.8, "gain_db": -3, "pause_after": 0.75, "fx": "shaky" } },
"shots": [ { "shot_id": "sh03", "mood": "joy", "seed": 103, "location": "clearing",
             "cast": ["pip"], "scene": "...", "action": "... talking ...",
             "label": "optional step badge",
             "lines": [ { "speaker": "pip", "tone": "excited", "text": "Whoa!", "tts": "optional pronunciation" } ] } ]
```

**Diagram shot** (needs an `architecture` block: fixed boxes and arrows shared by all diagram shots):

```jsonc
"architecture": { "nodes": { "api": { "label": "API Gateway", "sub": "auth", "kind": "service", "x": 475, "y": 230 } },
                  "edges": [ ["app", "api"] ] },
{ "shot_id": "sh04", "recipe": "diagram", "mood": "busy", "label": "Step 2 · One transaction",
  "diagram": { "show": "all", "reveal": false, "focus": ["order", "odb"],
               "flows": [ { "from": "order", "to": "odb", "label": "BEGIN ... COMMIT", "at": 1.0 } ],
               "table": { "name": "orders", "columns": ["id", "status"], "rows": [["1", "-"]],
                          "changes": [ { "row": 0, "col": "status", "value": "CREATED" } ] },
               "note": ["BEGIN;", "INSERT ...;", "COMMIT;"] },
  "lines": [ { "speaker": "narrator", "tone": "explainer", "text": "..." } ] }
```

Box kinds: `client`, `service`, `store`, `queue`, `cache`, `external` (each has its own colour). Coordinates are **box centres in a 1920×1080 frame**; boxes are 250×86 px. Keep boxes at x ≤ ~1100 (the table/code panel starts at x = 1270) and y between ~200 and ~780 (the step label is at the top, subtitles at the bottom). A 4×4 grid with centres x = 165/475/785/1095 and y = 230/400/570/740 works well; place connected boxes next to each other so arrows don't cross other boxes. `focus` boxes glow, the rest dim; `reveal: true` makes boxes appear one by one; flow and table-change times (`at`, seconds) are spread over the shot when omitted. Code-note lines: ≤ ~44 characters.

**What the plan check rejects (story builder):**

- unknown voice, tone, mood, effect, sound effect, recipe, character or diagram box;
- a character speaking who is not in the shot's `cast` ("speaks but is not on screen");
- a character who is in shots A and C at the same `location` but missing in B (would look like they vanished), unless B is marked `"cutaway": true`;
- an AI shot whose estimated speech is longer than one clip (5 s × `MAX_SLOWMO` ≈ 7.1 s): "split it into two shots";
- diagram flows/tables that don't match the architecture or table.

**Tips for writing plans:** one or two short sentences per AI shot (≤ ~14 words); repeat the same character `look` everywhere; never ask the AI model for text, logos or brand names (put text in `title`/`label`/diagram instead); use `tts` for words the voice mispronounces (`"Lumi"` → `"Loomee"`, `"Zomato"` → `"Zo-maato"`); avoid "Hmm" (the voice says "home"); put every name in `vocabulary`. Titles are wrapped onto two lines automatically when they are too wide.

**Writing a script viewers understand** (the user found the first Zomato script "generalized … hard to understand"):

1. **One concrete example, start to finish.** A named person orders a named dish from a named (fictional) place; the same people appear in every step. Viewers follow a story, not a list of abstractions.
2. **Say who does what, in plain words.** "Zomato sends her order to the restaurant, Raju's Kitchen, on their tablet", not "the order pops up on the partner app". "Now Zomato looks for a delivery rider near the restaurant", not "an algorithm looks for a nearby delivery partner".
3. **Name the topic the way people say it.** "When you order food on Zomato", not "when you tap 'Place Order'".
4. **One idea per shot**, ≤ ~14 words, so each line fits one continuous AI clip (the plan check enforces the length).
5. **Explain the "why" in everyday terms** ("He'll reach just as the food is ready"), never jargon.
6. **Step labels in the same simple words**: "Step 1 · You pay", "Step 2 · Restaurant gets the order", "Step 3 · Zomato finds a rider", "Step 4 · Food is cooked", "Step 5 · Rider picks up the food", "Step 6 · Live tracking", "Step 7 · Food delivered", "Step 8 · Rating and payment".
7. **End by naming the topic again** ("And that's what happens every time you order food on Zomato!").
8. **Real companies:** describe how such apps *generally* work (sources in `claims`), add a "not affiliated" `disclaimer`, and keep logos and brand names out of the AI prompts.

**Example (the Zomato video, final script):**

| Shot | Speaker | Line |
|---|---|---|
| sh01 (title) | Narrator | Ananya is hungry. She opens Zomato and orders a chicken biryani. |
| sh02 · Step 1 · You pay | Narrator | She pays online. The payment is checked and approved in a few seconds. |
| sh03 · Step 2 · Restaurant gets the order | Narrator | Zomato sends her order to the restaurant, Raju's Kitchen, on their tablet. |
| sh04 | Chef Raju | One chicken biryani! Accepted. Ready in fifteen minutes! |
| sh05 · Step 3 · Zomato finds a rider | Narrator | Now Zomato looks for a delivery rider near the restaurant. |
| sh06 | Narrator | It picks one based on distance, traffic and cooking time. |
| sh07 | Narrator | Arjun, a nearby rider, gets the order. He'll reach just as the food is ready. |
| sh08 | Arjun | Okay! Heading to the restaurant now! |
| sh09 · Step 4 · Food is cooked | Narrator | Meanwhile, Chef Raju cooks the biryani, packs it and seals the box. |
| sh10 · Step 5 · Rider picks up the food | Narrator | Arjun picks up the food and taps 'Picked up' in his app. |
| sh11 · Step 6 · Live tracking | Narrator | As he rides, his phone shares his location with Zomato. |
| sh12 | Narrator | So Ananya can watch him coming on a live map, with the arrival time. |
| sh13 · Step 7 · Food delivered | Arjun | Hello! Your biryani from Raju's Kitchen. |
| sh14 | Ananya | Perfect timing! Thank you so much! |
| sh15 · Step 8 · Rating and payment | Narrator | Then Ananya rates her food. Her money is shared between the restaurant, Arjun and Zomato. |
| sh16 | Narrator | And that's what happens every time you order food on Zomato! |

On-screen title: "What Happens When You Order Food on Zomato?". YouTube title: "What Happens When You Order Food on Zomato? (Explained Simply)", category Education, not made for kids, with the "not affiliated with Zomato" disclaimer. Sources for the facts: restaurant partner app ([registerkaro.in](https://www.registerkaro.in/guides/how-to-sell-food-on-zomato), [App Store listing](https://apps.apple.com/in/app/id1515073489)), rider assignment weighs distance, prep time and traffic ([restaurantindia.in](https://www.restaurantindia.in/article/zomato-delivery-model-explained-who-controls-what.16393)), riders timed to food readiness, as industry practice ([Grab](https://www.grab.com/inside-grab/stories/cutting-down-wait-times-delivery-drivers/)), live tracking ([order guide](https://partnerwithscphealth-jitsi.bluetext.com/blog/order-food-on-zomato-a)).

---

## 7. The workflow, step by step (every notebook cell)

| # | Cell | What it does | Problem it solves |
|---|---|---|---|
| 1 | **Settings** | Version print and every knob (voice, speed mode, slow-motion limit, sharpening, continuity check, caches). | One place to change behaviour; the version print proves the right notebook is running ("my changes were overwritten"). |
| 1b | **Run log** | Copies everything printed, every FFmpeg command, every error with full traceback, per-cell timings and RAM/GPU/disk every 30 s into `logs/run_<date>.log`. | Debugging from screenshots was slow; crashes left no trace. |
| 2 | **Install** | Removes Colab's old `torchao`, installs kokoro-onnx, faster-whisper, diffusers, accelerate, transformers, peft, onnxruntime-gpu. Runs **once per Colab machine** (marker file). | Version clashes (`FqnToConfig`), Python 3.13 incompatibilities, 1–2 min wasted on every re-run. |
| 3 | **Setup** | Mounts Drive, makes folders, detects the GPU and picks models (T4 → Wan 1.3B + SDXL; L4/A100 → Wan 2.2 5B + Z-Image-Turbo; no GPU → stills), prints the **Colab budget** (RAM, disk, GPU, Drive free, what is cached), defines disk-cleanup helpers. Moves the run log onto Drive. | Choosing models that fit the GPU; warning before RAM/disk run out. |
| 4 | **Plan** | Loads the plan (your Drive copy wins if its `plan_version` is equal or newer), validates it, builds AI prompts (scene + every character's look + action), sets what the continuity check must see. | Bad plans fail in a second, not after an hour of GPU work. |
| 5 | **Helpers** | `ff()` (FFmpeg with logging), sidecars, `free_gpu()`, speech-error scoring (word + letter level, numbers and apostrophes normalised), resource readout, video-encoder pick (NVENC if it works, else x264). | Shared, tested building blocks. |
| 5b | **Background downloads** | Works out what the video step will need and isn't cached (text encoder only if some prompt was never encoded; video model only if not in the Drive cache) and downloads it in a background thread. | First-session downloads (~17 GB) used to start only after the voices; now they overlap. |
| 6 | **Voices** | Kokoro speaks each line (story: each character's voice, speed and pitch from character × tone, voice effects via FFmpeg). Each line is transcribed by Whisper; if >15% wrong, it is re-made at a slightly different speed (best of 3 kept, flagged if still wrong). Words + timings saved. GPU first, clean CPU fallback. | Mispronounced or skipped words; subtitles need word timings. |
| 7 | **Timeline** | Places every line at its exact time with breathing pauses (from line → tone → mood), rounds shots to whole frames, builds the voice track, word-timed subtitles (≤ 42 characters per cue; coloured per speaker in stories), `timeline.json`, `script.txt`. Warns if an AI shot is longer than one clip. | Natural pacing; no robotic back-to-back sentences; subtitles in sync. |
| 7b | **Diagrams** | Draws every diagram shot frame by frame (boxes, arrows, travelling requests, table rows changing, code notes) and streams the frames straight into FFmpeg. Runs in a background thread; cached by content hash. | Technical topics need exact, readable visuals; no GPU needed. |
| 8 | **Keyframes** | Only when needed: on L4/A100 every AI shot starts from a still (image-to-video); on T4 stills are made **only as fallbacks** for shots whose video failed. | The T4 never used them: skipping saves a 7 GB download, RAM and ~10 min. |
| 8b | **Editing library** | Builds each shot's edit: speed-fit (≤ `MAX_SLOWMO`, gentle blend), scale to 1080p, light sharpening, colour grade by mood, fades/flash, title and step-label overlays. Records any frozen-frame time. Caches finished shots by content hash; edits shots **in the background** while the GPU works. | CPU editing used to start only after all video was done; unchanged shots were re-edited every run. |
| 9 | **AI video** | Encodes prompts (cache first; text encoder loaded straight onto the GPU once, then deleted from disk), loads Wan (from the Drive cache if possible) with the turbo LoRA, then for each shot: decides frames (only as many as the shot needs), generates, converts to 8-bit **on the GPU**, runs the **continuity check** (CLIP) and remakes once with a new seed if a character is missing, writes the clip with our own near-lossless writer (after a colour self-test). Prints time left. | Real video for every shot; T4 memory limits; missing characters; colour bugs; speed. |
| 10 | **Music** | Four generated layers (base drone, tension pulse, shimmer, "curious" plucked strings + music box) whose levels follow each shot's mood, smoothly. Or your own `music.mp3`. | "Same tone all the way is boring"; no licensing risk. |
| 11 | **Sound effects** | Generated boom, rumble, whoosh, shimmer, ding placed at plan times. | Story moments land with sound. |
| 12 | **Render** | Waits for background edits/diagrams, joins all shots, mixes voices + music (ducked under speech) + effects, normalises to −14 LUFS, burns subtitles, saves `final.mp4`. | One clean, YouTube-ready file. |
| 13 | **Quality checks** | Length, loudness, unplanned black frames, speech check, every video shot real AI video, no frozen frames, characters stayed on screen; writes `qc_report.json`. | Problems found by the notebook, not by viewers. |
| 14 | **YouTube text** | Title, description (summary, sources, disclaimer, voices used, AI disclosure), tags, Private, synthetic content = Yes, audience note. | Honest, ready-to-paste upload text. |
| 15 | **Preview** | Small preview inside the notebook. | Quick look without downloading. |

---

## 8. Models and libraries

**Models**

| Model | Used for | Runs on | Size | Licence |
|---|---|---|---|---|
| Wan 2.1 T2V 1.3B (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) | AI video on T4 (832×480, 16 fps) | GPU | 6.2 GB download, 3.3 GB cached | Apache-2.0 |
| Wan 2.2 TI2V 5B (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`) | AI video on L4/A100 (1280×704, 24 fps, image-to-video) | GPU (offload) | ~23 GB download (20 GB transformer + 2.8 GB VAE), not cached on Drive | Apache-2.0 |
| UMT5-XXL text encoder (bf16 copy from the Wan 2.2 repo) | Turns prompts into embeddings for Wan | GPU, once per new prompt | 11.4 GB | Apache-2.0 (Wan) |
| Self-Forcing DMD LoRA for Wan 1.3B (`Kijai/WanVideo_comfy`) | Turbo: 4 steps, no CFG | GPU | 91 MB | Upstream Self-Forcing: Apache-2.0 |
| Kokoro-82M (onnx v1.0) | All voices (54 stock voices) | GPU or CPU | ~350 MB | Apache-2.0 |
| Whisper `small.en` (faster-whisper) | Speech check + word timings | GPU (float16) or CPU | ~500 MB | MIT |
| CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) | Continuity check (is the character on screen?) | GPU | ~600 MB | MIT |
| SDXL base 1.0 | Fallback / keyframe stills on T4 | GPU | ~7 GB | CreativeML Open RAIL++-M |
| Z-Image-Turbo | Keyframe stills on L4/A100 | GPU | large | check the model card before commercial use |

**Libraries**

| Library | Why |
|---|---|
| `diffusers` (0.41) | Wan pipelines, schedulers, LoRA loading, First Block Cache |
| `transformers`, `accelerate`, `peft` | Text encoder, CLIP, loading straight onto the GPU, LoRA fuse |
| `torch` (Colab's) | Everything on the GPU |
| `huggingface_hub` (+ `hf_xet`) | Downloads (`HF_XET_HIGH_PERFORMANCE=1`), cache cleanup |
| `kokoro-onnx`, `onnxruntime-gpu` | Voices without PyTorch, works on Python 3.13 |
| `faster-whisper` (CTranslate2) | Fast transcription with word timestamps |
| `numpy`, `soundfile` | Audio tracks, music synthesis |
| `Pillow` | Diagrams, titles, labels, keyframe fallback images |
| `imageio`, `imageio-ffmpeg` | Dependency of diffusers' video utilities |
| `FFmpeg` (system) | All editing, encoding (NVENC or x264), audio mixing, loudness, subtitles, synth sounds |
| `psutil` | RAM readouts and budget |

---

## 9. Running on a free Colab T4: GPU, RAM and disk

A free T4 session has about **12.7 GB RAM, 15 GB GPU memory, ~70–110 GB disk**, and Drive has 15 GB on a free account (shared with Gmail/Photos).

| Resource | Rule we follow | How |
|---|---|---|
| **System RAM** | Stay under ~6 GB | Big models load **straight onto the GPU** (`device_map={"": 0}`), never through RAM. Text encoder runs once, then is freed. Frames become 8-bit on the GPU before reaching RAM (4× less). Diagram frames are streamed into FFmpeg, never held. Models are freed between steps. |
| **GPU memory** | One big model at a time | Text encoder (~11 GB) → freed → Wan 1.3B (~3.5 GB) + CLIP (~0.3 GB). Voices and speech check are freed before video starts. |
| **Disk** | Check before downloading, delete after use | `need_disk()` checks free space before big downloads (stops with a clear message instead of a half-written file). The 11 GB text encoder is deleted from disk once prompts are cached; the Wan download is deleted once it is in the Drive cache. |
| **Drive** | Model cache ~4 GB | Saved only if Drive has 4 GB free; a clear warning otherwise. Turn off with `MODEL_CACHE_ON_DRIVE = False`. |

What uses the **GPU**: text encoder, Wan video, CLIP check, Whisper, Kokoro (when its CUDA libraries work), video encoding (NVENC, when Colab's FFmpeg supports it). What stays on the **CPU** by nature: FFmpeg filters (slow motion, grading, scaling, subtitles), music synthesis, diagram drawing — all run in the background while the GPU works.

---

## 10. Caching: what is saved and where

| What | Where | Re-made when |
|---|---|---|
| Installed packages | Colab machine (marker file) | New Colab machine |
| Video model (Wan 1.3B, fp16; the T4 path only — the 5B model is too big for a free Drive) | `_model_cache/` on Drive | Never (until deleted) |
| Voice, speech-check, CLIP models, turbo LoRA | `_model_cache/` on Drive | Never |
| Encoded prompts | `_model_cache/prompt_cache/` (shared by all videos) | A prompt text is new |
| Voice lines | `<project>/narration/` + sidecar | Text, voice, speed or effects change |
| Keyframes | `<project>/keyframes/` + sidecar | Prompt or seed change |
| AI clips | `<project>/clips/` + sidecar | Prompts, seed or keyframe change, or clip made by an old buggy writer |
| Diagram animations | `<project>/diagrams/<shot>_<hash>.mp4` | Anything in the diagram or its length changes |
| Edited shots | `<project>/segments/<shot>_<hash>.mp4` | Anything that shapes the shot changes (clip, filters, length, title, label, encoder, editing code version) |

The 11 GB text encoder itself is **not** cached (too big for a free Drive); a brand-new video still downloads it once.

---

## 11. Speed: where the time goes and what makes it fast

On a T4, nearly all the time is the AI video model. Every speed-up so far:

| Change | Effect |
|---|---|
| **Turbo mode** (Self-Forcing LoRA, 4 steps, no negative-prompt pass) | ~28 min → ~3–5 min per clip |
| **Frames on demand** (only the frames a shot needs, ≤ 1.15× slow-down) | ~1.6× less video work on short shots |
| **Diagram shots** (drawn, no AI) | Technical shots cost ~5 s of CPU instead of minutes of GPU |
| **One continuous clip per character shot** | No wasted second takes |
| **Background downloads** during voices | ~3–5 min per first session |
| **Background editing** during video generation | Editing is mostly hidden |
| **Lazy keyframes** on T4 | Skips SDXL (7 GB) and ~10 min |
| **Model cache on Drive** | Next sessions skip ~6 GB of downloads |
| **Prompt cache** (shared) | Text encoder only for new prompts |
| **Edit cache** | Re-runs re-edit only changed shots (297 s → 64 s in a test) |
| **Install once per machine**, **veryfast** final encode, **NVENC** when available | Minutes per run |

Rough T4 times (**estimates** from model workload, not yet timed on a real T4 with all changes): a video with ~16 AI shots ≈ 45–60 min in the first session; a diagram-heavy video with ~4 AI shots ≈ 20–25 min. The "about N min left" line after the first clip gives the real number.

---

## 12. Quality guards: how bad output is caught

| Guard | Catches | Reaction |
|---|---|---|
| Plan check | Impossible or illogical plans (unknown names, speaker off screen, character vanishing, shot too long for one clip) | Stops before any GPU work, lists every problem |
| Speech check (Whisper) | Skipped / garbled words | Re-make the line (up to 3 tries), flag it in the report |
| Colour self-test | Clip writer changing colours | Stops before making any clip |
| NaN / black clip check | Broken model output | Shot falls back to a still, reported |
| Continuity check (CLIP) | Character missing from a clip (seen in < 6 of 8 sampled frames) | Remake with a new seed, keep the best, report |
| Frozen-frame check | A shot holding its last frame > 0.6 s | Reported in QC |
| Loudness, length, black-frame checks | Mix and render problems | Reported in QC |
| Old-clip detection | Clips from the inverted-colour writer | Remade automatically |

---

## 13. Logging

Every run writes `MyDrive/ai-video/<project>/logs/run_<date>_<time>.log`:

- everything printed (progress bars only in their final state);
- every FFmpeg command with time and exit code, and its error text;
- every error with the **full traceback**, including ones the notebook recovers from;
- start/end and duration of every cell, with RAM/GPU/disk;
- RAM, GPU memory, GPU load and free disk **every 30 seconds** (shows what happened right before a crash);
- settings, package versions, `nvidia-smi`, every prompt, every continuity score.

When something goes wrong, send this file.

---

## 14. Every problem we hit, and how it was solved

In the order they happened.

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | `pip install kokoro>=0.9.4` fails | Kokoro needs Python < 3.13; Colab moved to 3.13 | Use **kokoro-onnx** (same voices, no PyTorch); `--ignore-requires-python` fallback |
| 2 | "My changes were overwritten" | An older notebook copy was running | Version printed in cell 1; every rebuild gets a new file name |
| 3 | `cannot import name 'FqnToConfig'` | Colab's preinstalled `torchao` too old for new diffusers | Uninstall torchao in the install cell; warn to restart the session if diffusers was already loaded |
| 4 | faster-whisper crash (`metadata_errors`) | faster-whisper / PyAV version clash | Decode audio with FFmpeg to 16 kHz and pass a numpy array |
| 5 | FFmpeg `colorbalance` error | Invalid option `ms` | Correct option `bm` |
| 6 | `name 'true' is not defined` | Plan JSON pasted as Python | Embed the plan with `json.loads(r"""...""")` |
| 7 | Voice sounded flat, no breathing space | Same speed, no pauses | Per-mood speed and pauses, per-sentence `pause_after`, speech re-check |
| 8 | "Same tone all the way is boring" | One mood for the whole video | Mood per shot: voice speed, pauses, music layer levels, colour grade, camera move, sound effects |
| 9 | "Only moving images, not real video" | Shots were zooms over still pictures | Real AI video per shot (Wan), phrase-synced to the narration |
| 10 | Colab session crashed (RAM) on T4 | Wan's UMT5-XXL text encoder (22.7 GB fp32) loaded in system RAM (12.7 GB) | Encode all prompts **once on the GPU** (loaded straight to GPU), free it, load the video model **without** the text encoder, load everything with `device_map` to the GPU |
| 11 | "It only uses CPU RAM, use the GPU" | Voice and speech check were CPU-only, x264 encode | onnxruntime-gpu for Kokoro, Whisper on CUDA float16, NVENC encoder test; RAM and GPU readouts after every heavy step |
| 12 | Slow start: 22.7 GB download | Wan 2.1 repo stores the text encoder in fp32 | Use the identical bf16 copy from the Wan 2.2 repo (11.4 GB); cache encoded prompts on Drive |
| 13 | Voice error: `CUDNN_FE ... HEURISTIC_QUERY_FAILED` | cuDNN algorithm search fails for Kokoro on some T4s | Fixed algorithm (`cudnn_conv_algo_search=DEFAULT`), test a real sentence, clean CPU fallback with the error in the log |
| 14 | ~28 min per clip on T4 (6.5 h per video) | 25 steps × 2 passes (CFG) on a slow GPU | Turbo: Self-Forcing DMD LoRA, 4 steps, no CFG, its own scheduler; First Block Cache as `"fast"` mode |
| 15 | "Log everything" | Debugging from screenshots | Run log cell (section 13) |
| 16 | Too slow / too much RAM overall | SDXL stills made but unused on T4; models re-downloaded each session; editing after all video; float32 frames in RAM | Lazy keyframes; Drive model cache; background + cached editing; 8-bit frames on GPU; install once |
| 17 | **Video colours inverted (photo negative), blotchy** | diffusers `export_to_video` multiplies frames by 255, but ours were already 0–255 → overflow; low-quality writer | Own near-lossless FFmpeg writer, colour self-test, old clips auto-remade |
| 18 | Distorted / warped motion, soft picture | Motion-interpolated slow motion (`mci`) warps shapes; heavy slow-down; small clips upscaled | Gentle `blend`, `MAX_SLOWMO` 1.4, light sharpening after upscale |
| 19 | Speech check false alarms | "100" vs "a hundred", "restaurant's" vs "restaurants", "back end" vs "backend" | Numbers to words, apostrophes removed, letter-level comparison as a second score |
| 20 | Voice says "Hmm" as "home" | Kokoro pronunciation | Avoid "Hmm"; `tts` field for pronunciations |
| 21 | **Character vanishes and comes back** | Long lines were filled with a second, unrelated AI take; the model sometimes ignores a character; cutaways not checked | One continuous clip per character shot; plan check (length, speaker on screen, A-B-A vanishing); CLIP continuity check with remake |
| 22 | Continuity check said "missing" when visible | Compared "character" against an "empty scene" text that matched the background better | Texts differ only by the character; absence threshold 0.2 (tested: absent ≈ 0.01, present ≈ 0.5) |
| 23 | Long scenery shot froze 3.4 s | One-clip rule applied to shots without characters | Scenery-only shots may use extra takes (nothing can vanish); frozen-frame check in QC |
| 24 | Disk could fill up | 11 GB encoder + 6 GB model left on disk | `need_disk()` before downloads; delete them once cached |
| 25 | Model re-downloaded every session | Cache needed > 6 GB free on Drive; free Drives often have less | Threshold 4 GB, clear warning, Drive free space and cache contents shown at start |
| 26 | Prompts re-encoded for every new project; LoRA re-downloaded | Prompt cache per project; LoRA not in model cache | Shared prompt cache; LoRA in the Drive cache |
| 27 | Wasted video work | Every clip generated the full 5 s even for 3 s shots | Frames on demand (~1.6× faster) |
| 28 | Downloads after voices | Sequential steps | Background downloads during voices |
| 29 | Technical topics: AI can't draw diagrams or text | Model limitation | Diagram recipe drawn by code, in the background, cached |
| 30 | Crash on diagram-only shots in the prompt log | Code assumed every shot has an AI prompt | Log only AI shots (caught by the plan test before any GPU time) |
| 31 | Narrator notebook only worked for the first video | Project folder, YouTube description and category were hard-coded to the black-hole video | Taken from the plan (`project`/`project_id`, `metadata.summary`, `disclaimer`, `category`); black-hole text moved into its plan, unchanged |
| 32 | Tests could not be re-run later | Test scripts lived in a temporary scratch folder that gets wiped | Saved in `colab/tests/` with exact setup and run commands |
| 33 | **"The script is generalized, hard to understand"** (Zomato video) | Lines described the process in the abstract ("you tap 'Place Order'", "an algorithm", "the partner app") | Rewrote as one concrete story (Ananya → biryani → Raju's Kitchen → rider Arjun), plain "who does what" lines, simple step labels; rules in section 6, "Writing a script viewers understand" |
| 34 | Voice said "Zomato" as "zuh-MAH-doh"; speech check heard "the motto" / "some motto" | Kokoro guesses unknown words (weak first vowel, American flapped t) | `tts` spelling `"Zo-maato"` → clear "zoh-MAH-toh" (found with `Tokenizer().phonemize()`); subtitles still show "Zomato". All 6 lines then 0% wrong words |
| 35 | Speech check misheard names ("Arjun" → "Our gun") | Whisper doesn't know the names | Plan `vocabulary` + character names passed to Whisper as `initial_prompt` (`ASR_PROMPT`) |
| 36 | Long title would run off the screen | Title drawn on one line at 84 px | Title overlay wraps onto two centred lines when wider than the frame |
| 37 | Script edits didn't show up in a test run | The Drive `plan.json` had the same `plan_version`, and the Drive copy wins | Bumped the plan to `plan_version` 3. Always bump it after editing a plan |

---

## 15. How the code is tested without a GPU

The development machine has no NVIDIA GPU, so each change is tested in layers:

1. **Compile check** of every notebook cell after every build.
2. **Full end-to-end run on CPU** with a small runner (`colab/tests/run_nb.py`; setup and commands in `colab/tests/README.md`): it executes the notebook cells in order, skips the install cell, points Drive to a local folder, uses the tiny Whisper model, and injects **fake AI clips** (FFmpeg test patterns with valid sidecars). Real voices, speech checks, timeline, diagrams, music, editing, render and QC all run for real.
3. **GPU code on a small real model**: a 2-layer Wan transformer with the real layer sizes, the real VAE config and the real turbo LoRA, run on CPU to test LoRA loading/fusing, the turbo scheduler, frames-on-demand and the 8-bit frame path, and the model cache (save in session 1, load in session 2).
4. **Targeted tests**: the clip writer (old writer reproduces the negative colours, new keeps them), the continuity check with the real CLIP model on synthetic frames (object present / absent / partly present), the background-download job list (first session vs everything cached), the plan checker on good and bad plans, the speech-error score on real false alarms.

What still needs a real Colab run: actual Wan output quality, Kokoro/Whisper on CUDA, NVENC, real download/Drive speeds, the continuity threshold on real cartoon frames.

---

## 16. Known limits

- **No lip-sync**: mouths move but don't match the words.
- **T4 character consistency**: text-to-video can't see the previous shot, so a character may look slightly different between shots. L4/A100 (image-to-video from keyframes) is more consistent.
- **Resolution**: Wan 1.3B makes 832×480, upscaled to 1080p; it is softer than native HD. Use `VIDEO_SPEED = "fast"` or an L4 for more detail.
- **New videos download the 11 GB text encoder once** (it does not fit on a free Drive). On L4/A100 the 5B video model (~23 GB) is also downloaded every session.
- **Continuity threshold** is tuned on synthetic tests; check the scores in the log after the first real runs.
- Topics about real companies must be framed as "typical" designs unless official sources exist (with a disclaimer).

---

## 17. Troubleshooting

| You see | Do this |
|---|---|
| Cell 1 doesn't print the expected version | You opened an old notebook; upload the new one |
| "Old libraries are still loaded" / `FqnToConfig` | Runtime → Restart session → Run all |
| Session restarts (RAM) | Send the log: the 30-second RAM lines show which step |
| "Colab disk too full" | Runtime → Disconnect and delete runtime → Run all (work on Drive is safe) |
| "Only X GB free on Google Drive" | Free 4 GB on Drive, or set `MODEL_CACHE_ON_DRIVE = False` |
| "Voice model running on: CPU" | Fine: voices take a few minutes on CPU |
| "⚠️ CHECK THIS LINE" | Look at `heard` in `narration/<line>.wav.json`. A name misheard → add it to `vocabulary`; a word mispronounced → check `Tokenizer().phonemize()` and add a `tts` spelling (e.g. `Zo-maato`); otherwise reword |
| My script edits don't appear | Bump `plan_version` (the Drive copy wins when versions are equal) |
| Plan rejected | Read the listed problems; usually "split it into two shots" or "add to cast" |
| "not on screen enough … remaking" | Normal: the continuity check is remaking a clip |
| A shot shows "still image + camera move" | Its AI video failed: Run all again later (only that shot is retried) |
| Too slow | Check the "about N min left" line; L4 is 3–5× faster; keep AI shots short; use diagram shots for technical content |

---

## 18. Rules that must not be broken

- **No hard-coded credentials**, ever; nothing secret in plans or notebooks.
- **Stock voices only** — never clone a real person's voice (also banned on Colab's free tier).
- **No real people, logos or brand imagery** in AI prompts; real companies only as a topic, with a disclaimer.
- **Upload as Private**, answer **Yes** to "altered or synthetic content", set the audience honestly.
- **Don't bypass YouTube limits or disclosure rules.**
- **Sources for factual claims** go into the plan's `claims` and the description.
- **Claude writes plans, not engine code per video**: engine changes go into the builders, get tested, and benefit every video.

---
---

# Update 2026-10-15: engaging explainers, licensed music, validation safeguards, modular engine

> Appended; nothing above was removed. Where this part differs from the sections above, **this part wins**:
> notebooks are now built by `colab/build.py` from `colab/engine/` (sections 0, 5 and the "Build" line above describe
> the older layout), the default music is now a licensed library (section 21), and plans go through the scene
> validator (section 22). The rest of the document still applies.

## 19. Review of the previous video (what went wrong)

The food-delivery backend video (`food_order_backend_plan.json`) was criticised as "a technical presentation filled
with flowcharts". The plan shows why:

| Problem | Evidence in the plan | Fix now enforced |
|---|---|---|
| Flowchart overload | 13 of 17 shots were box-and-arrow diagrams; only 4 real scenes | Rule S2: max 25 % diagram shots, max 2 in a row; new `ui`, `journey`, `card` scenes |
| Jargon | idempotency key, outbox, ACID, Kafka, state machine, geospatial index, double-entry | Rule S6: jargon only via the plan's `glossary`, explained with an analogy where first used |
| Guesses stated as facts | "often kept in memory in Redis", internal services of a real company | Rules P4–P6: every factual line cites a sourced claim; internal-system words need a source |
| Process presented as fixed | Steps that vary by case were narrated as always happening | Rule P2: conditional steps must be worded conditionally |
| Long, dense sentences | 25+ word sentences with 3 concepts | Rule S4: ≤ 22 words per sentence |
| Weak music | synthesised drones/plucks | Licensed library tracks per chapter (section 21) |
| Engine hard to change | ~2,400 lines of Python inside strings in two builder scripts | Real modules + one builder (section 20) |

The old backend plan is now **rejected by the validator** (12 errors: S2 + S6). This is intended; rewrite it as a
story (one customer, one order, app screens and a journey) before rebuilding it.

## 20. Research: how to make explainers people understand and keep watching

**Evidence-based** (learning science):

| Finding | Source | What the engine/plans do |
|---|---|---|
| Mayer's multimedia principles: **coherence** (cut extra words/pictures/sounds), **signaling** (highlight what matters), **redundancy** (narration + graphics beats narration + graphics + lots of text), **spatial/temporal contiguity** (words next to and at the same time as the picture), **segmenting** (short chunks), **pre-training** (name key ideas first), **modality** (speak, don't print), **personalization** (conversational words) | [Dartmouth summary](https://services.dartmouth.edu/TDClient/1806/Portal/KB/PrintArticle?ID=171655), [NUS Medicine](https://medicine.nus.edu.sg/medixperience/scholarly-article/mayers-12-principles-of-multimedia-instruction/), [Adobe summary](https://elearning.adobe.com/2019/12/12-principles-of-multimedia-learning-relevant-to-video-from-richard-e-mayers-work) | One idea per shot (S4); short step labels and ≤ 9-word card lines (S5, `check_spec`); subtitles are the only long text; visuals change exactly when the narration does (timeline); glossary terms introduced before use (S6); conversational second person |
| 6.9 M edX viewing sessions (Guo, Kim & Rubin 2014, Table 1): **shorter videos are much more engaging** (keep < 6 min); a **personal feel beats high-budget studio** recordings; **Khan-style drawing with motion and continuous visual flow beats slides**; **speaking fairly fast with enthusiasm** is more engaging | [paper (MIT CSAIL PDF)](https://up.csail.mit.edu/other-pubs/las2014-pguo-engagement.pdf) | 1.5–2 min videos; drawn scenes animate continuously (no static slides); narration speed ≈ 1.0, warm/curious tones; no long intro |

**Practitioner advice** (creator/agency guides; useful but not research):

- Three acts: a short opening, a middle that raises a problem or question, an ending that resolves it; a relatable
  character turns an abstract process into a human story; analogies simplify abstract ideas
  ([Wyzowl](https://wyzowl.com/storytelling-in-explainer-videos/), [simpleshow](https://simpleshow.com/blog/storytelling-explainer-videos/)).
- Hook early: the first ~30 s decide retention; skip greetings and channel intros; confirm quickly that the video
  answers the title ([OutlierKit](https://outlierkit.com/resources/youtube-hooks-and-retention/),
  [GetResponse](https://www.getresponse.com/blog/youtube-hacks), [Soundstripe](https://soundstripe.com/blogs/youtube-intro-ideas)).
  YouTube's own help pages were not found in the search; treat the numbers as rules of thumb.

**Techniques adopted (and enforced where possible):**

1. **Open with the viewer's question** in the first shot (hook card), e.g. "Refund issued… so where is the money?" (S1).
2. **One protagonist, one concrete case, start to finish** (Maya, $59.99 earbuds, credit card) — see memory rule
   "concrete scripts" and section 6 "Writing a script viewers understand".
3. **Show processes the way people meet them**: the app screen they tap, the counter they visit, the bank statement
   they check; abstract steps as a **journey** (the money travelling between stations) instead of a flowchart.
4. **One idea per shot**, ≤ 22 words per sentence, ≤ ~14 words per AI shot (fits one clip).
5. **Explain jargon with an analogy the first time** ("card network: the road between banks that card payments
   travel on"), or avoid it.
6. **Say "if / some / usually"** for steps that don't always happen; never present a conditional step as fixed.
7. **End on the payoff**: answer the opening question in one memorable line ("'Issued' means sent; your bank sets the
   final day") (S7).
8. **Vary the picture**: no more than 3 shots in a row of the same kind/place (S3).

## 21. Background music and audio guidelines

**What research and practice say:**

- Music must support, not compete: simple, instrumental, mid-tempo beds; melody competes with narration
  ([Sonilo](https://sonilo.com/blog/how-to-add-background-music-to-a-video), [Wistia](https://wistia.com/blog/background-music-volume)).
- Level: voice-led videos keep music roughly **12–18 dB under the voice**; ducking (music dips while someone talks)
  of about **6–12 dB**, release ~250–700 ms ([CapCut guide](https://www.capcut.com/create/audio-ducking-for-clear-dialogue-in-video),
  Sonilo). Listener preferences for ducking depth vary a lot ([Torcoli et al., AES 2019](https://salford-repository.worktribe.com/output/1360952)).
- Change music only at **chapter boundaries**, with smooth crossfades; match energy to the scene.

**What the engine does now** (`engine/lib/musiclib.py`, music cell):

| Setting / part | Value | Why |
|---|---|---|
| `MUSIC_STYLE = "library"` (default) | Licensed tracks; `"curious"`/`"ambient"` = old generated score, kept as offline fallback | Real music sounds professional |
| Library | Kevin MacLeod, incompetech.com, **CC BY 4.0**. Official FAQ: "Can I use your music for YouTube videos? Yes, AND you can monetize the videos. Be sure to credit me." ([FAQ](https://incompetech.com/music/royalty-free/faq.html)) | Legal for monetised YouTube with credit |
| Credit | Added to the YouTube description automatically in incompetech's format: `"Title" Kevin MacLeod (incompetech.com) Licensed under Creative Commons: By Attribution 4.0 https://creativecommons.org/licenses/by/4.0/`; also `render/music_credits.txt`. QC check "music licensed and credited". | Licence requirement |
| Track choice | From incompetech's catalogue data (feel, bpm, description; `pieces.json`): **Easy Lemon** (82 bpm, "takes voice-over very well"), **Wallpaper** (92 bpm, "very clean mix… non-offensive"), **Investigations** (mysterious/humorous: hooks, "where is it?"), **Carefree** (uplifting: endings), Inspired, Life of Riley, Local Forecast – Elevator, Thinking Music | Matched to mood, all checked to download (HTTP 200) |
| Arrangement | Plan `"music": {"cues": [{"from": "sh01", "track": "Investigations"}, ...]}`; without cues: one track per `chapter`, or per mood with **≥ 12 s per track**; **2 s crossfades** at changes only | Changes feel intentional, never mid-scene |
| Level | Bed set from the measured voice loudness: `MUSIC_UNDER_VOICE_DB = 10` under the voice, then the existing sidechain ducking (threshold 0.02, ratio 8, attack 20 ms, release 500 ms) dips it further while someone speaks → ~16–20 dB under speech; final mix loudness-normalised to −14 LUFS | Matches the 12–18 dB guidance with headroom |
| Caching | Tracks download once to `_model_cache/music/` on Drive (≈ 3–9 MB each, 10–16 s first time), size-checked | Fast re-runs |
| Fallback | Download fails → generated score + warning; the run never stops | Never block |
| Unlicensed tracks | `arrange()` refuses titles not in the library | Licence safety |

Content ID: incompetech's music is CC BY; if YouTube's Content ID ever claims it, dispute with the licence
(incompetech's FAQ says the same). Your own music: put `music.mp3` in the project folder (you must own the rights).

**Sound effects** (generated, no licence needed): `boom`, `rumble`, `whoosh`, `shimmer`, `ding`, plus new **`click`**
(tap on a phone button), **`pop`** (notification), **`coins`** (money arrives). Use them where an action happens on
screen (signaling), at −12 to −20 dB.

## 22. Scene validation: rules and safeguards

All rules live in `engine/lib/validate.py` (pure Python, unit-tested in `tests/test_lib.py`). The plan cell runs them
**before any voice or GPU work**. **Errors stop the run** (nothing is generated; the list and fixes are printed and
saved to `render/validation_report.txt/.json`); **warnings** are reported and appear in the storyboard and QC.

| Rule | Checks | Level |
|---|---|---|
| **S1** hook | the first two shots pose a question (or are marked `hook`) | warn |
| **S2** flowcharts | ≤ 25 % diagram shots, ≤ 2 in a row (`rules` in the plan can change limits) | error |
| **S3** variety | ≤ 3 shots in a row of the same kind and place | warn |
| **S4** sentence length | ≤ 22 words | error |
| **S5** on-screen text | step label ≤ 6 words, title ≤ 9 words, card line ≤ 9 words | warn / error |
| **S6** jargon | technical words (acquirer, issuer, card network, settlement, API, database…) must be in `glossary`, explained (`explains`) where first used | error / warn |
| **S7** payoff | the last shot is the payoff/recap | warn |
| **P1** process order | shots follow the plan's `process` steps in order (`flashback` to go back) | error |
| **P2** conditional steps | steps marked `conditional` must be narrated with "if / some / may / usually / when / depending…" | error |
| **P3** completeness | every required (non-conditional) step is shown | error |
| **P4** sources | every claim has an https source; every factual narrator line in a process shot cites claims (`story: true` for pure story lines) | error |
| **P5** numbers | a timing the narrator says ("3 to 5 business days") must appear in a cited claim (character dialogue is fiction and exempt) | error |
| **P6** invented internals | for a `real_company`, words like algorithm, database, servers, system, backend need a cited claim | error |
| **P7** time | `day` values never go backwards (unless `flashback`) | error |
| **A1** narration ↔ visual | every plan `entity` the narration mentions (bank, earbuds, card network…) appears in what the shot shows (scene, action, UI text, journey stations, card lines), or is listed in `offscreen` | error |
| **A2** overlap | without entities: at least one narration word in the visual | warn |
| **V1** teleport | a character in two consecutive shots at different places needs a `transition` (or a `travel` route) | error |
| **V2/V3** props | objects (`props`) have one fixed look (added to every prompt) and move only forward along their `path` (home → store → returns center) | error |
| **V4** cast | only known characters | error |
| older checks | speaker on screen; A-B-A vanishing; AI shot fits one clip; drawn-scene specs (`drawkit.check_spec`: widget ids, event times 0–1, stations exist, card line length) | error |

**At render time** (clips cell, needs the GPU): for every AI clip, CLIP checks (1) each character and prop that must
be on screen is visible (seen in ≥ 6/8 frames), and (2) **the clip shows its own scene** better than the scene
descriptions of other places in the video (≥ half the frames). If not, the clip is remade with a new seed; if it
still fails, the shot's **`fallback`** drawn scene is used (when the plan gives one) and the shot is listed under
`review` in `qc_report.json` ("nothing flagged for review" check). A shot is never silently included when it failed.

**What automation can't do:** it can't know whether a fact is true. Accuracy comes from the research step: every
fact is a `claim` with a source URL, checked by a person while writing the plan; the validator then makes sure the
narration only says what the claims say, words conditional steps as conditional, and doesn't invent internals.

## 23. The improved architecture

```
colab/
  build.py                     ONE builder for every plan (narrator-only or with characters is decided by the plan)
  build_notebook.py, build_story_notebook.py   thin wrappers kept for old commands
  engine/
    cells/shared/*.py          cells every notebook uses: log, install, setup, lib, helpers, prefetch, diagrams
                               (= all drawn scenes), keyframes, editlib, clips, music, sfx, render, qc, preview
    cells/narrator/*.py        settings, plan, narration, timeline, metadata for narrator-only plans
    cells/cast/*.py            the same five for plans with characters
    lib/validate.py            scene validation rules (section 22)
    lib/drawkit.py             drawn scenes: ui (phone screens), journey (travel metaphor), card
    lib/musiclib.py            licensed music library, arrangement, credits
  <topic>_plan.json            everything about one video (the only file that changes per video)
  tests/test_lib.py            unit tests for engine/lib
```

- Cells are **real `.py` files** (lintable, readable, diffable). Placeholders filled by `build.py`:
  `__PLAN_JSON__`, `__PROJECT__`, `__VOICE__`, `__TTS_LIB__`, `__LIB_FILES__`.
- `engine/lib/*.py` are copied into the notebook's **lib cell** (still one file to upload) and imported; the same
  files are unit-tested locally.
- The move was proven safe: all four existing plans built byte-for-byte identical code cells before new features
  were added.
- **Shot recipes** (the menu): `video` (AI clip, GPU), `ui`, `journey`, `card`, `diagram` (drawn by code, no GPU).
  A new video = a new plan; a new kind of scene = a new renderer in `drawkit.py` + `check_spec` + `describe()`.

**Workflow per video:** research facts → write `claims` with sources → write `process` steps (mark conditional
ones) → write the plan (story, shots, visuals, music cues) → `python3 colab/tests/test_lib.py` and a quick local
`validate.validate(plan)` → `python3 colab/build.py plan.json out.ipynb` → Colab: Run all → read
`validation_report.txt`, `storyboard.md`, `qc_report.json` → watch → upload Private.

**Drawn scene spec (summary)** — all times are fractions 0–1 of the shot:

```jsonc
"ui":      { "screens": [[ {"type": "header", "text": "Your Orders"},
                           {"type": "product", "name": "...", "detail": "...", "price": "$59.99", "icon": "box"},
                           {"type": "choice", "id": "why", "options": ["...", "..."], "selected": 0},
                           {"type": "button", "id": "go", "text": "Return items"},
                           {"type": "status", "id": "st", "steps": ["...", "..."], "current": 0},
                           {"type": "statement", "id": "acct", "rows": [{"date": "Tue", "text": "Refund", "amount": "+$59.99", "highlight": true}]},
                           {"type": "notice", "text": "...", "icon": "clock"}, {"type": "text", "text": "..."} ]],
             "events": [ {"t": 0.3, "do": "select|tap|show|status|row|screen|notify", "target": "id", ...} ],
             "caption": {"text": "≤ 6 words", "t": 0.1} },
"journey": { "path": ["store", "network", "bank"], "focus": [...], "caption": "...",
             "calendar": {"start": "Thu", "days": 7, "label": "Business days"} },   // stations: plan-level "journey"
"card":    { "lines": ["Question?", "point", "point"], "icon": "question|clock|check|money", "bullets": true }
```

Icons: phone, store, warehouse, truck, bank, card, network, processor, person, box, money, clock, calendar, check,
question. Layout keeps clear of the step label (phone moves right, captions go left) and of the subtitles (phone
ends at y = 856). The theme comes from the plan's `look` colours, so every drawn scene in a video matches.

## 24. Performance

| Change | Effect (measured on the CPU test machine unless noted) |
|---|---|
| Drawn scenes instead of AI for screens, journeys, cards | Amazon video: 6 AI clips instead of ~17 → on a T4 ≈ 6 × 3–5 min instead of 17 × 3–5 min (estimate) |
| Drawn scenes render in the background (CPU) while voices/GPU work | 6–10 s per scene (113–190 frames) |
| Shadow blur on a small cached patch, static layers drawn once per shot | journey 464 → 32 ms/frame, ui 158 → 52 ms/frame |
| Drawn scenes cached by content (`diagrams/<shot>_<hash>.mp4`) | unchanged scenes are never redrawn (`DRAWKIT_VERSION` bump forces a redraw) |
| Music tracks cached on Drive | 10–16 s download once per track, then instant |
| Validation before anything else | a bad plan fails in < 1 s instead of after an hour of GPU work |
| Whole Amazon notebook on CPU (no GPU, fake clips) | 297 s first run (voices, music downloads), 135 s re-run; peak RAM ≈ 1 GB |

All earlier optimisations (sections 9–11) still apply: turbo AI video, model and prompt caches on Drive, edit cache,
background editing, frames on demand, disk/RAM guards.

## 25. Best practices for the next videos (checklist)

1. Pick a question viewers actually ask; make it the hook card.
2. Research first: official help pages > primary documentation > reputable secondary sources. Save exact quotes as
   `claims` with URLs; note what is **not** publicly documented and don't show it.
3. Write the `process` steps; mark every step that depends on country, seller, item or payment method as
   `conditional` with its condition.
4. One named protagonist, one concrete case (amount, item, payment method, place). Say where the story is set.
5. Choose the scene type per shot: people/places → `video`; anything on a screen → `ui`; invisible movement (money,
   data, parcels) → `journey`; question/recap/comparison → `card`. Diagrams only for genuinely technical audiences.
6. One idea per shot; ≤ 22 words per sentence; AI shots ≤ ~14 words; glossary + analogy for unavoidable terms.
7. Define `entities`, `props` (with `path`), `travel`, `transition`s so the validator can check alignment and
   continuity; give risky AI shots a drawn `fallback`.
8. Music: 3–5 cues at chapter changes (mystery for the hook, calm/bright for explaining, uplifting for the ending).
9. Validate locally, build, run; read `validation_report.txt`, `storyboard.md`, `qc_report.json` (`review` must be
   empty) before uploading. Bump `plan_version` after every plan edit.
10. Disclaimers for real companies ("simplified", "not affiliated", region/timeline caveats); stock voices only;
    upload Private with "altered or synthetic content: Yes".

## 26. The Amazon refund video

**Files:** plan `colab/amazon_refund_plan.json` → notebook `colab/colab_amazon_refund_v1.ipynb`; Drive folder
`MyDrive/ai-video/amazon_refund_v1/`. 17 scenes, ~90 s: 6 AI video, 5 app screens, 2 journeys, 4 cards; validation
0 errors, 0 warnings.

**Research (what the facts are based on):**

| Claim | Source (checked 2026-10-15) |
|---|---|
| Most items returnable within 30 days of delivery; start in Your Orders → Return Items | [Amazon Return Policy](https://www.amazon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7) |
| Refund once Amazon (or the third-party seller) has received the item and found you eligible; up to 30 days to receive and process | same |
| Some third-party sellers give a full refund without the item being returned | same |
| Most returns can be dropped off free; most need no box or label | same |
| "Advanced refunds" are issued when the carrier first scans the return | [Amazon Refunds](https://www.amazon.com/gp/help/customer/display.html?nodeId=GKQNFKFK5CF3C54B) |
| All returns are inspected against the expected item; partial refunds for use, damage, missing parts | same |
| Credit card 3–5 business days; debit card up to 10; Amazon gift card 2–3 hours; "Your bank might need extra time" | same |
| Once issued, "it may take additional time for your financial institution to make funds available" | Return Policy |
| If the original payment method can't be used, the refund goes to the Amazon account balance | Refunds page, [Check Your Refund Status](https://www.amazon.com/gp/help/customer/display.html?nodeId=GMP8PC8KBY5FCPM2) |
| Card refunds go to the customer's bank/card issuer and appear as a credit some business days later depending on the bank; a reference number (ARN) can trace a refund | [Stripe: Refunds](https://docs.stripe.com/refunds) (payment-industry documentation, not Amazon-specific) |

**Regional and case differences** (said in the video's description and kept out of absolute statements): the story
is set in the **US** because Amazon.com's help pages could be read directly; Amazon.in's help pages could not be
fetched (bot protection) and the third-party numbers found for India disagree, so none are used. Amazon UK's help
mentions refunds within 14 days and up to 7 business days for the bank. Seller-fulfilled items, item condition,
payment method and gift returns all change the path.

**Not shown because not publicly documented:** how Amazon's internal systems decide eligibility, which payment
processor Amazon uses, internal fraud checks. The money's route is shown only as store → card network → the
customer's bank, the general card-refund path.

**Storyboard** (the notebook also writes `render/storyboard.md` with exact times, music and checks per scene):

| # | Type | What the viewer sees | Narration / dialogue | Music · SFX | Claims |
|---|---|---|---|---|---|
| sh01 | card (hook) | "?" icon; "Refund issued." · "Bank account: nothing." · "So where is the money?" | Amazon says your refund is issued. Your bank says nothing. So where is the money? | Investigations · pop | (story) |
| sh02 | AI video | Maya at her kitchen table holds the left earbud to her ear, frowns | Meet Maya. Her new earbuds arrived, but the left earbud makes no sound. | Investigations | (story) |
| sh03 | app screen | Your Orders: earbuds $59.99, reason "Left earbud has no sound" selected, tap "Return items"; caption "Start in Your Orders"; label Step 1 | She opens Your Orders on Amazon, picks a reason, and taps Return Items. | Easy Lemon · click | c1 |
| sh04 | app screen | Return details: earbuds, "Eligible: returns accepted within 30 days of delivery", choose "Drop off: no box needed", tap Confirm; label Step 2 | Most items can be returned within 30 days of delivery. Her earbuds qualify. | click | c1 |
| sh05 | card | "Send it back?" · Most items: yes · Some sellers: refund, no return · Maya's earbuds: going back | Some sellers even refund you without asking for the item back. Maya's earbuds do go back. | | c3 |
| sh06 | AI video | Maya places the earbuds case on a shop return counter; label Step 3 | If a return is needed, she drops it off. Many returns need no box or label. | | c9 |
| sh07 | AI video | The clerk scans a code on Maya's phone | **Clerk:** All set! Your return is scanned. | ding | (dialogue) |
| sh08 | app screen | Return status: "Dropped off" ticks; notification "Return dropped off" | For some returns, the refund starts at that first scan. Most wait until the item arrives. | pop | c4, c2 |
| sh09 | AI video | A returns-center worker opens the earbuds case and checks it; label Step 4 | When the earbuds arrive, they're checked against what was ordered. | | c5 |
| sh10 | app screen | Status runs to "Refund issued"; notification "Refund issued: $59.99 — back to your credit card"; label Step 5 | Everything matches, so the refund is issued to her credit card, the card she paid with. | ding | c2, c10 |
| sh11 | AI video | Maya looks at her bank app, puzzled | **Maya:** Refund issued... so why is my bank app still empty? | Investigations | (dialogue) |
| sh12 | journey | The money travels Amazon → Card network → Maya's bank; caption "The money retraces the path of the payment"; label Step 6 | A card refund travels back the way the payment came: over the card network, to the bank behind the card. | Investigations · whoosh | c8 |
| sh13 | journey | Maya's bank highlighted; business-day calendar (Sat/Sun "closed"); label Step 7 | Then her bank has to post it. Banks work in business days, so a weekend adds a wait. | Wallpaper | c7, c8 |
| sh14 | card | "How long does it take?" · Credit card: 3–5 business days · Debit card: up to 10 · Amazon gift card: 2–3 hours | Amazon lists 3 to 5 business days for credit cards, and up to 10 for debit cards. | | c6 |
| sh15 | app screen | Card activity: "Refund: Amazon +$59.99" appears; label Step 8 | A few business days later, the refund shows up in her card account. | Carefree · coins | c7 |
| sh16 | AI video | Maya smiles with relief | **Maya:** There it is! It was just on its way. | | (dialogue) |
| sh17 | card (payoff) | ✓ "'Issued' means sent" · The money leaves the store · Your bank sets the final day | So 'refund issued' means the money has left the store. Your bank decides the day it lands. | | (story) |

**Voices:** narrator `af_heart`, Maya `af_jessica`, clerk `am_eric`; speech check vocabulary: Amazon, Maya, earbuds,
Your Orders, Return Items. All 17 lines passed (0–7 % wrong words) in the CPU test.

**How to generate it:**

1. (Optional, local) `python3 colab/tests/test_lib.py` → ALL PASSED; `python3 colab/build.py colab/amazon_refund_plan.json colab/colab_amazon_refund_v1.ipynb`.
2. Upload `colab_amazon_refund_v1.ipynb` to Colab → Runtime → Change runtime type → **T4 GPU** (L4 for more
   consistent characters) → **Run all** → allow Drive.
3. Watch for: cell 1 version; "Scene validation … 0 validation errors"; voices; "11 drawn scenes … in the
   background"; music cues; AI clips with "about N min left" (6 clips, ≈ 20–30 min on a T4 the first time,
   estimated); "✅ Saved".
4. Before uploading, read `render/qc_report.json` (`review` must be empty), `render/storyboard.md`, and watch
   `final.mp4`. Paste the text from `render/metadata.json` (description includes sources, disclaimer and music
   credits); upload **Private**, "altered or synthetic content: Yes", audience "not made for kids".
5. To change anything: edit `plan.json` in the Drive project folder (or the local plan and rebuild), **bump
   `plan_version`**, Run all — only changed parts are remade.

**Tested on the CPU machine:** the whole notebook ran end to end (voices, speech checks, validation, music
download and arrangement, drawn scenes, editing, render, QC, metadata). AI clips were stand-ins (no GPU); not yet
verified on a real GPU: the Wan clips for the 6 story scenes and the CLIP scene-relevance threshold (0.5) on real
frames. One unexplained event: two lines were misheard on the first attempt and passed on the retry; the same audio
re-transcribed correctly every time afterwards, so the failing transcript is now logged (`heard: … | expected: …`).

## 27. More problems hit (continuing section 14)

| # | Problem | Cause | Fix |
|---|---|---|---|
| 38 | "Flowcharts, too technical" (backend video) | 13/17 diagram shots, jargon, guesses as facts | Story + `ui`/`journey`/`card` scenes; validator rules S2, S6, P4–P6 |
| 39 | Synth music unsuitable | Drones/plucks generated in code | Licensed library (CC BY), per-chapter cues, voice-relative level, auto credits |
| 40 | Engine hard to maintain | Code in strings in two builders | `engine/` files + `build.py`, proven identical |
| 41 | Journey icons partly invisible | White shapes on white tiles | Icons switch to tinted shapes on light tiles (`on_light`) |
| 42 | Drawn scenes 10× too slow (464 ms/frame) | Every shadow blurred the whole 1080p frame | Blur a small cached patch; draw static parts once per shot |
| 43 | Journey scene broken at other sizes | Layout hard-coded for 1080p | Draw at 1080p, then scale |
| 44 | Step label overlapped phone and captions; subtitles over the phone | Fixed positions | Phone moves right when a label exists, captions go left, phone ends at y = 856 |
| 45 | Caption still on the wrong side | A loop variable `off` overwrote the phone offset | Renamed to `slide` |
| 46 | Chef's "ready in fifteen minutes" flagged as an invented fact | Number rule applied to dialogue | P5 applies to narrator lines only |
| 47 | Zomato: chef/rider changed place without a transition | New teleport rule (V1) found a real gap | `transition: "cut"` on sh10/sh11, `plan_version` 4 |
| 48 | Music for plans without chapters: one mysterious track for 71 s | Default grouped everything into one chapter | Follow scene moods, ≥ 12 s per track |
| 49 | Validator caught my own Amazon script: narration "her earbuds" while the screen didn't show them (sh04, sh05) | Alignment rule A1 | Added the earbuds to the screen and the card |

## 28. Current state (2026-10-15)

| Video | Plan → notebook | State |
|---|---|---|
| Amazon refund (new) | `amazon_refund_plan.json` → `colab_amazon_refund_v1.ipynb` | Validated (0/0), CPU-tested end to end; ready for a Colab run |
| Zomato order | `zomato_order_plan.json` (v4) → `colab_zomato_order_v1.ipynb` | Rebuilt on the new engine; validated (0 errors, 10 warnings); CPU-tested |
| Black holes | `black_holes_plan.json` → `colab_make_video_v15.ipynb` | Rebuilt on the new engine; CPU-tested |
| Food backend | `food_order_backend_plan.json` → `colab_food_backend_v1.ipynb` | **Rejected by validation by design** (flowcharts + jargon); needs a story rewrite |
| Pip story | `story_plan.json` → `colab_story_v3.ipynb` (old engine) | Not rebuilt: long shots must be split first |

## 29. Google Colab free tier: what was checked and changed (2026-10-15)

Goal: every notebook runs on a **free Colab T4** session (≈ 12.7 GB RAM, 15 GB GPU memory, ~70–110 GB disk,
15 GB free Google Drive, Python 3.13, Ubuntu 22.04's FFmpeg 4.4, limited GPU hours per day, idle disconnects).

| Risk | Check done | Result / change |
|---|---|---|
| Packages change under us (`pip install -U` pulled whatever was newest) | — | **Pinned the exact tested versions**: diffusers 0.41.0, transformers 5.19.0, accelerate 1.15.0, peft 0.21.2, huggingface_hub 1.33.0, safetensors 0.8.0, kokoro-onnx 0.6.1, faster-whisper 1.2.1, ctranslate2 4.8.3, soundfile 0.14.0, imageio 2.38.1, imageio-ffmpeg 0.6.0 (onnxruntime-gpu stays the newest build matching Colab's CUDA). Install marker bumped to `v3` so machines with older installs reinstall. |
| Python 3.13 | `pip download --python-version 3.13 --abi cp313` for every package; full notebook run in a real Python 3.13.16 venv | Every package has a cp313/py3 wheel; the whole notebook runs on 3.13 |
| **FFmpeg 4.4** (Colab's apt FFmpeg; local tests used 7.0) | Full notebook runs with the static FFmpeg 4.4.1 build | **Found a real bug:** the final mix failed ("filters could not choose their formats: sidechaincompress") — FFmpeg 4.4 can't negotiate formats for music ducking. **Fixed** with an explicit `aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo` on every mix input. All other filters (music, SFX, drawn scenes, per-shot edits, subtitles, QC) work on 4.4 and 7.0. |
| Fonts for drawn scenes and titles | Checked matplotlib's bundled fonts | System DejaVu first, then **matplotlib's DejaVu copies (matplotlib is preinstalled on Colab)**, then Liberation, then Pillow's default (logged). |
| GPU memory (15 GB) | Earlier real T4 logs: text encoder alone ≈ 12.7 GB, then freed; Wan 1.3B ≈ 3.5 GB; CLIP adds ≈ 0.3 GB | One big model at a time (unchanged); drawn scenes and music need no GPU |
| System RAM (≈ 12.7 GB) | Earlier real T4 logs ≈ 5.8 GB peak; CPU test of the full Amazon notebook ≈ 0.5–1.0 GB | Big models load straight to the GPU; frames become 8-bit on the GPU (unchanged) |
| Disk / Drive (15 GB free Drive) | Cache sizes | Drive model cache ≈ 4.9 GB (Wan 3.3 + voice 0.35 + Whisper 0.5 + CLIP 0.6 + LoRA 0.09 + music ≈ 0.03); disk guard before downloads; big downloads deleted after use (unchanged) |
| No GPU (free GPU quota used up / CPU runtime) | Ran the whole notebook without a GPU | Clear message at setup; the video still finishes (AI shots become stills + camera move); a later GPU session makes only the missing AI clips |
| Session limits (idle disconnect, ~12 h max, daily GPU time) | Time estimate printed after the plan check | e.g. Amazon: "~41 min the first time, ~29 min in a new session"; everything is saved on Drive, so reconnect → Run all continues |
| Scene-relevance check (new CLIP rule) | Real CLIP model on synthetic clips | own scene → seen 1.0 (passes); another place → 0.0 (flagged and remade) |

**Tested end to end (CPU, stand-in AI clips):** Amazon refund (fresh first session and re-run), Zomato order and
black holes on **Python 3.13 + FFmpeg 4.4**, and Amazon again on Python 3.12 + FFmpeg 7.0 — all produce
`final.mp4` (Amazon: 90.9 s, −14.2 LUFS) and pass every check except "every video shot is AI video" (no GPU here).
Unit tests: ALL PASSED. **Still only verifiable on a real Colab GPU:** Wan clip quality, Kokoro/Whisper on CUDA,
NVENC, download speeds.

**Free-tier tips for the user:** choose Runtime → Change runtime type → **T4 GPU**; keep the tab open while it runs;
if Colab says the GPU limit is reached, run anyway (drawn scenes, voices and music are made) and Run all again later
with a GPU; keep ≥ 6 GB free on Google Drive for the caches.

## 30. First real Colab run of the Amazon video: continuity warnings (2026-10-15)

What the user saw on a free T4: `sh02`, `sh06`, `sh09` → "not on screen enough … remaking" and then "still not
clearly visible after retries … flagged for review"; `sh07`, `sh11` passed. Clips took 1.9–10.8 min each.

| # | Problem | Cause | Fix |
|---|---|---|---|
| 50 | Earbuds "missing" in every shot that has them | CLIP sees each frame at 224×224 px; earbuds in a hand are a few pixels → it cannot recognise them (false alarm). Checking small props was a design mistake | Props stay in every prompt (consistency) but are only checked when the plan says `"check": true` |
| 51 | sh02 "scene: a cozy apartment kitchen" failed twice | The action (earbud held to the ear, frowning) is naturally a close-up: little kitchen is visible, so comparing scenes is unreliable | The scene check now only **flags** for review; it never triggers a remake |
| 52 | Remakes cost ~13 min of free GPU time in one run | Every doubt triggered a remake | A remake happens only when a **character** is missing (once, `CONTINUITY_RETRIES = 1`) |
| 53 | "Flagged for review" but nothing to look at | Only text scores | QC saves **`render/review/<shot>.jpg`** (4 frames across the clip) with the scores, also for clips made in earlier runs (no GPU) |

**What to do with flagged shots:** open `render/review/shNN.jpg` (or `clips/shNN_p1.mp4`). If the person/scene is
right, it's a false alarm: keep it. If not, delete `clips/shNN.json` on Drive and Run all: only that clip is
remade. Clips are never deleted automatically.

**Observation to follow up:** the log showed `GPU 11.6/15.0 GB` during the clip loop (earlier runs showed ~3.5 GB
at this stage). It did not cause a failure, but it leaves less headroom; the run log will show which step holds
the memory.

## 31. Logical & spatial consistency (2026-10-16) — summary; full design in `docs/09_LOGICAL_CONSISTENCY_ARCHITECTURE.md`

**Why:** generated shots looked fine alone but broke the story: people entering from the wrong side, reversing,
objects changing hands without an action, actions missing or cut off. The audit found the causes in the code:
pure text-to-video on the T4 with no spatial input, several actions per 5 s clip, CLIP-only appearance checks,
failed clips still rendered, seed-only retries picking the highest CLIP score, incomplete cache keys, and checks that
switched themselves off after an error (doc 09 §2, F1–F18).

**What changed (all reusable for future videos):**

| Part | What it does |
|---|---|
| Scene-state per AI shot (`state` in the plan; `engine/lib/scene_state.py`, pydantic) | camera setup/axis/side/move, start boxes and facing, who holds what, ordered small events with times and end boxes; simulated shot by shot before any GPU work (rules X1–X5, E1–E11: crossing the line, reversals, objects changing hands without an event, acting from afar or while walking, > 1 movement + 1 hand action per clip, actions at the cut) |
| Layout control (`layout_control.py`) | the same state gives every subject's expected box per frame and draws VACE's layout video (white background, coloured box per subject) |
| Controlled generation (`cells/shared/logic.py`) | scene-state shots use **Wan2.1-VACE-1.3B** (Apache-2.0, ~3.6 GB fp16, fits the T4, same text encoder/VAE as before, turbo LoRA fuses) steered by the layout video |
| Checks (`motion_check.py`, `trackers.py`, `vlm_check.py`) | OWLv2 detection + camera-motion estimate → start/end position, direction after camera compensation, stationary, camera really static, teleport, stop before acting, object goes from the hand to the planned spot, objects never move by themselves, event order, action finishes before the cut; Qwen3-VL-2B answers one strict question per hand action (time + evidence). Every check: PASS / FAIL / UNKNOWN with evidence |
| Decisions (`acceptance.py`) | ACCEPT only if all critical checks pass; FAIL → targeted repair (stronger layout, simpler layout, static camera, reseed, fewer frames) within `REPAIR_BUDGET`; UNKNOWN or exhausted → REVIEW. **A REVIEW clip is never used** unless approved in `approvals.json` for that exact generation; otherwise the shot's drawn `fallback` is used and listed in QC |
| Caching | generation key (model, recipe, steps/LoRA, frames, size, seed, prompt, control) separate from validation key (constraints, validator versions, thresholds): changed checks re-validate cached clips without regenerating |
| Amazon plan v4 | states for all 7 AI shots; sh06 split into sh06 (enter from screen-left, walk right, stop) + sh06b (place the case on the counter); sh07 clerk behind the counter on screen-right scans Maya's phone; sh09 earbuds "moved off-screen: shipped"; drawn fallback card for every AI shot |

**New settings (cell 1):** `VIDEO_RECIPE = "auto"`, `LOGIC_CHECK = True`, `VLM_CHECK = True`, `REPAIR_BUDGET = 2`.
**New outputs:** `render/review/<shot>.jpg` (4 frames), `<shot>_plan.png` (intended layout), `<shot>.txt` (every check
with evidence); `qc_report.json` → `logic` per shot and `not_logically_verified`.

**To approve a shot after looking at it:** create `MyDrive/ai-video/<project>/approvals.json`:
`{"sh06": {"approved": true, "gen_key": "<gen_key from clips/sh06.json>", "note": "direction fine"}}` → Run all.

**Tests:** `tests/test_consistency.py` (62 checks on synthetic videos), `tests/test_logic_loop.py` (15 checks: the
real notebook loop with stub models), `tests/test_lib.py` (31). OWLv2 checked on CPU with a real photo moved across
a frame (left→right PASS, right→left FAIL). **Unverified until a GPU run:** VACE + turbo quality and how well it
follows the boxes, OWLv2 on stylised frames, Qwen3-VL answers, GPU time and memory (also the earlier 11.6 GB reading).

| # | Problem | Cause | Fix |
|---|---|---|---|
| 54 | Wrong entry side, reversals, teleports, missing/wrong actions accepted | No spatial plan, no control, appearance-only checks, failed clips rendered | Scene-state + VACE layout control + geometric/VLM checks + acceptance gate (section 31, doc 09) |
| 55 | Handoff check passed a case still in her hand next to the counter | "Near the target" isn't proof | Object must reach the PLANNED spot and leave the hand; objects may only move while someone carries them |
| 56 | OWLv2 call failed with transformers 5.x | `post_process_object_detection` renamed | Use `post_process_grounded_object_detection` when present |
| 57 | Check evidence couldn't be saved | numpy numbers in JSON | Results converted to plain Python |
| 58 | A VLM-confirmed small-object action stayed "timing unknown" | Boxes can't time small objects | The VLM's timestamp is used as the completion time |
| 59 | Layout colours shifted when a repair left subjects out | Colours assigned after skipping | Colour by position in the full track list |
| 60 | Review reason overwritten by an old message | Two code paths set it | First (real) reason kept |

## 32. Video: "What Happens When You Book a Train Ticket?" (2026-10-16)

**Files:** plan `colab/train_ticket_plan.json` → notebook `colab/colab_train_ticket_v1.ipynb`; Drive folder
`MyDrive/ai-video/train_ticket_v1/`. India / Indian Railways (IRCTC). 14 scenes, ~82 s: 4 AI scenes with scene-states
(Riya at home; walks in from screen-left along the platform to coach B3 on screen-right and stops; the ticket checker
on screen-right checks her e-ticket and ID; Riya at the window), 6 cards, 4 app screens. Content validation 0 issues;
scene-state 0 issues; all 14 voice lines heard correctly (0–7 %); CPU end-to-end run OK (Python 3.13 + FFmpeg 4.4).

**Story:** hook "Ticket booked… but it says WL 12. Will you get a seat?" → search (up to 60 days ahead) → pay, PNR,
WL 12 → CNF / RAC / WL explained → Tatkal aside (10 AM AC, 11 AM non-AC) → the waiting list moves up as others
cancel (WL 12 → WL 5 → RAC 3) → the chart is prepared → confirmed: coach B3, berth 42 → if still fully waitlisted
after the chart, the e-ticket is cancelled and refunded automatically and the passenger can't board → platform →
e-ticket + original photo ID → on the train → payoff "WL isn't a no. It's a place in line, and the chart decides."

**Facts and sources (checked 2026-10-16):** 60-day advance booking from 1 Nov 2024 (Business Today, Outlook
Traveller); RAC/WL meaning and order of confirmation (Wikipedia, ixigo); chart moved to 8 h before departure in 2025
and later earlier for morning trains (Business Standard, The Hitavada) — **the video deliberately gives no exact chart
time** because it changed twice in 2025; fully waitlisted e-tickets cancelled and refunded automatically after the
chart, passengers can't board (ICICI Bank IRCTC terms, IRCTC e-ticket notice on crpf.gov.in); e-ticket SMS/VRM + one
original photo ID (same notice); Tatkal 10 AM AC / 11 AM non-AC, Aadhaar-authenticated accounts online from July
2025 (Deccan Herald, The South First). Official IRCTC pages couldn't be fetched (bot protection), so news/bank
sources are cited; the description says rules change and to check IRCTC.

**Engine changes made for this video:**
- Every required action that isn't walking/standing (e.g. "the ticket checker looks at her phone") is now asked to
  the VLM; before, such an action had no check at all and would have passed silently.
- Plan `look.currency` (here "₹") sets the symbol on the money icon (default "$"); DejaVu has the ₹ glyph.

## 33. "CUDA out of memory" on the T4 during VACE clips (2026-10-16)

**What the user saw:** `sh02: video failed (OutOfMemoryError: Tried to allocate 884 MiB … 14.30 GiB in use … 11.83 GiB
allocated by PyTorch, 2.33 GiB reserved)` with a `0/25` progress bar (turbo uses 4 steps).

| # | Cause (evidence) | Fix |
|---|---|---|
| 61 | **11.8 GB "allocated" before generation ≈ the 11.4 GB UMT5 text encoder still on the GPU** (also explains the earlier unexplained "GPU 11.6/15" reading). `del te; gc; empty_cache` isn't always enough for models loaded with `device_map`: a hidden reference (hooks/caches) keeps the weights alive ([transformers #31439](https://github.com/huggingface/transformers/issues/31439), [HF forum](https://discuss.huggingface.co/t/unable-to-free-whole-gpu-memory-even-after-del-var-gc-collect-empty-cache/108584), [PyTorch forum](https://discuss.pytorch.org/t/how-to-clear-gpu-memory-after-using-model/137408/3)) | `release_model()` (helpers): removes accelerate hooks and replaces every weight's storage with an empty CPU tensor, so memory is freed even if a reference survives; logs GPU memory before/after. Used for the text encoder, VACE, the T2V model, CLIP and the checkers |
| 62 | 2.33 GB "reserved but unallocated" = fragmentation ([PyTorch devlog](https://docs.pytorch.org/devlogs/eager/2026-06-01-cuda-caching-allocator/), [NVIDIA memory tuning](https://docs.nvidia.com/nemo/megatron-bridge/latest/skills/nemo-mbridge-perf-memory-tuning/SKILL.html)) | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set at the top of the setup cell, before CUDA starts (no speed cost) |
| 63 | 25 steps instead of 4: the turbo LoRA most likely failed to fuse because the GPU was already full, so the notebook fell back to the slower 25-step mode | Fixed by #61 (the log prints the speed mode; check it says "turbo") |
| 64 | Checkers (OWLv2 + Qwen3-VL ≈ 5 GB) would have shared the GPU with VACE | Scene-state shots run in **rounds**: (A) VACE makes every candidate → released; (B) checkers check all → released; repairs → next round. One model family on the GPU at a time |
| 65 | One OOM lost the shot | `make_vace_clip_safe`: logs the memory summary, frees memory, retries once with 25 % fewer frames |
| 66 | Video decode peak memory | VAE tiling (`p.vae.enable_tiling()`) on GPUs < 20 GB |

**Tests (CPU):** `tests/test_logic_loop.py` now 18 checks, including: OOM on the first try → retried with fewer
frames and accepted; VACE released before checking; `release_model` empties every weight although a reference
survives. All suites and end-to-end CPU runs pass. **To confirm on the next GPU run:** the log lines
`released text encoder: GPU 11.x allocated … → 0.x allocated`, `round 1: VACE loaded, GPU …`, and `Speed: turbo`.

## 34. System RAM running out (Colab session restarts) in the train-ticket notebook (2026-10-17)

**Symptom.** The free Colab tier has ~12.7 GB of system RAM, separate from the 15 GB GPU memory. When it runs out,
Colab restarts the whole session ("Your session crashed after using all available RAM") and the Python error is lost.

**What could fill the RAM.** Every model load was checked:

| Load | Before | Risk |
|---|---|---|
| Qwen3-VL-2B judge (~4.3 GB) | loaded fully into system RAM, then `.to("cuda")` | ~4.3 GB peak on top of the ~4–5 GB baseline (voice models, FFmpeg, notebook) |
| OWLv2 detector (~0.6 GB) | same pattern | small |
| CLIP continuity checker | same pattern | small |
| VACE / Wan video model (`load_to_gpu`) | direct GPU load, but if that failed: an unchecked full load into RAM (VACE is 7.15 GB in fp32) | the most likely crash |
| Background download | `HF_XET_HIGH_PERFORMANCE=1` always on, running while the voice models are loaded | larger download buffers (precaution: not measured) |
| Background download | the train video has only scene-state (VACE) shots, but the unused text-to-video model (5.7 GB) was still downloaded | wasted time and disk; the VACE model wasn't prefetched |
| Python memory after big loads | freed memory isn't always handed back to the OS | RAM use grows round after round |

**Fixes (other parts unchanged):**

1. `trackers.load_on()`, used by OWLv2 and Qwen3-VL. On the GPU it uses `device_map={"": 0}`, so the weights go straight
   to GPU memory. If that fails, it loads one weight at a time (`low_cpu_mem_usage=True`) and then moves the model.
2. `load_to_gpu()`, used for the video models.
   - It first tries the direct GPU load, as before.
   - The system-RAM fallback now runs only after `need_ram()` confirms there is room. The size comes from the
     weight files when they are on disk (the Drive cache); otherwise it uses a default.
   - The fallback uses `low_cpu_mem_usage=True` and moves the model to the GPU immediately.
   - If RAM is too small, it raises a clear `MemoryError`, which is logged. The shot then uses its fallback card and
     the session survives.
3. The CLIP checker uses the same direct GPU load and RAM guard.
4. `free_gpu()` also calls `malloc_trim(0)`, which hands freed system RAM back to the OS.
5. `HF_XET_HIGH_PERFORMANCE` is set only on machines with ≥ 24 GB RAM, so it is off on the free tier.
6. `prefetch_jobs()` downloads the VACE model for scene-state shots. It skips the text-to-video model when no shot
   uses it.
7. Logging: `ram()` lines (system RAM, GPU memory, disk) are written:
   - before and after every heavy load;
   - after the OWLv2 and Qwen3-VL loads;
   - in every RAM check ("RAM check for …: needs X GB, Y GB free").

   All of them go to `logs/run_*.log`.

**If it still happens:** open the newest `logs/run_*.log` on Drive and look at the last `RAM x/12.7 GB` and
`RAM check` lines before the restart. They show which step was running.

**Tests:** `colab/tests/test_logic_loop.py` now has 25 checks, all passing:
- `need_ram` refuses a step that doesn't fit and lets one through that does.
- `load_to_gpu` tries the GPU load first, and its fallback uses `low_cpu_mem_usage` and goes to the GPU.
- The fallback is refused before loading when RAM is short.
- `load_on` uses `device_map` and has a low-RAM fallback.
- The prefetch downloads VACE, not text-to-video, for scene-state-only plans.

`test_consistency.py` and `test_lib.py` still pass.

## 35. Architecture and memory-use review (2026-10-17)

**Measured on CPU.** The train-ticket notebook ran end to end on Python 3.13 with FFmpeg 4.4 and stand-in clips.
Peak RAM was 0.9 GB, in narration (Kokoro + Whisper `tiny.en`), and FFmpeg's peak was 0.9 GB. Colab uses `small.en`,
which adds about 0.5 GB. All the other cells stay under 0.6 GB. The CPU-side steps therefore need about 1.5 GB of the
12.7 GB. The RAM risk is only in model loads, which section 34 fixed.

**GPU budget on the T4 (15 GB).** These are estimates, not measured here:

| Step | Estimated GPU use |
|---|---|
| Text encoder | about 11.4 GB (only when new prompts) |
| VACE generation | about 4 GB of weights plus activations, about 7–10 GB peak (Wan2.1-1.3B is quoted at 8.19 GB) |
| Checkers (Qwen3-VL-2B + OWLv2) | about 5 GB |

One model family is on the GPU at a time, on purpose: VACE and Qwen together would be about 14–15 GB, which means OOM.

**New logging to measure the real numbers:**
- `gpu_mem()` now also prints the peak (`max_memory_reserved`) since the last model release, and `release_model` resets
  it, so each model's peak appears in `logs/run_*.log`.
- The monitor samples system RAM every 2 s and logs "RAM peak … in the last 30 s" with every RES line.

**Verdict.** The architecture fits the free tier:
- a JSON plan runs a fixed executor;
- drawn scenes for most shots;
- AI video only for a few layout-controlled shots, each checked automatically, with a fallback;
- one model family on the GPU at a time.

Wan2.2-5B and HunyuanVideo 1.5 need CPU offloading, so they need more RAM than the free tier has.

**Biggest open gap:** VACE gets no character reference picture, so the same character can look different from one
shot to the next. diffusers 0.41.0 `WanVACEPipeline` supports `reference_images`; this is the next improvement.

## 36. Speed: turbo failure, the 25-minute clip, VAE time, and QC false rejections (2026-10-17)

**What the T4 log showed:**
- `Turbo mode could not load … using fast mode instead`.
- 25 steps at 24–68 s per step.
- Total time per clip: sh02 36.5 min, sh11 26.6 min, sh12 23.7 min.
- sh11 needed repairs: `event(riya): walk (enter): geometry UNKNOWN, VLM FAIL`, which triggered a reseed. Then
  `camera_static: camera moves 0.25 frame widths`.

### A. Root causes

Each finding is marked by how it was established:
- **verified:** proven by a test or by reading the code;
- **measured:** taken from the Colab log;
- **suspected:** not yet proven.

| # | Finding | Status |
|---|---|---|
| 1 | Turbo is NOT incompatible with VACE. Wan2.1-VACE-1.3B's base blocks are bit-identical to Wan2.1-T2V-1.3B (4 sampled tensors read from both repos with HTTP range reads; relative difference 0). All 306 layers the LoRA targets exist in the full VACE model. diffusers drops only the norm "diff" entries (max \|Δ\| 0.004 on weights of about 1), and `diff_b` is kept as a LoRA bias. | verified |
| 2 | The notebook's own `speed_up()` code fuses turbo correctly on the same library versions (diffusers 0.41.0, peft 0.21.2, transformers 5.19.0, hub 1.33.0) on a reduced VACE model: base weights change and a 4-step generation runs. `device_map={"": 0}` adds no accelerate hooks on a single device (accelerate `dispatch_model`). | verified (CPU) |
| 3 | So the Colab failure comes from something only present on Colab. In both logs it comes right after a Hub request, which is the LoRA download into the **Google Drive** cache folder. Hugging Face's cache uses symlinks and lock files that a Drive mount doesn't reliably support. The exact exception was only in the log ("details in the log"). | suspected; the next run shows it on screen |
| 4 | Earlier row 63 (section 33) guessed that turbo failed because the GPU was full. That was **wrong**: this run failed with the GPU at only 4.6/15 GB. | measured |
| 5 | Why 25 steps take ~15–23 min: fast mode runs the transformer **twice per step** (guidance 5.0 with a negative prompt), so 25 steps means up to 50 passes over about 32,760 tokens (81 frames → 21 latent frames × 30 × 52), through 30 base blocks plus 15 VACE blocks. Turbo needs 4 single passes, about 12× fewer. | arithmetic from the code; time per pass measured from s/it |
| 6 | Where the other ~10–13 min per clip goes: outside the progress bar, VACE runs **two full VAE encodes** (the "inactive" and "reactive" videos) and **one VAE decode**. All run in fp32, and in **tiles** because VRAM < 20 GB. At 832×480, tiles of 256 with stride 192 cover about 2.5× the pixels. The time outside the bar was 13.3 / 10.9 / 10.1 min for sh02 / sh11 / sh12. | measured that the gap exists; the split between encode, decode and the rest is suspected → now timed |
| 7 | VACE's "inactive" video is **all zeros**, because our mask covers the whole frame. Its latent is therefore identical for every clip with the same frame count, but it was computed every time. | verified (code + bit-identical test) |
| 8 | "GPU 0–2 % busy": those RES lines are snapshots taken while models load or sit idle. Steady 24–68 s/it on 32k tokens is consistent with a compute-bound GPU. | suspected; now sampled every second during each clip |
| 9 | Case A: `enter`/`exit` events were sent to the VLM with **no geometry linked**. The code looked for an `object_transfer` result, which a walking event never has, so geometry was always UNKNOWN and the 2B VLM alone decided. A "no" then led to `split_action` (manual), so the automatic repair became a **blind reseed**. `walk_to` was already judged by geometry. | verified (code + test) |
| 10 | Case B: when the detector misses the walking subject, nothing is masked, and the walker alone measures as a camera move: 0.30 (scale 4) and 0.20 (scale 8, the notebook's) on a static synthetic shot, close to the 0.25 in the log. The same clip had "geometry UNKNOWN", which is consistent with OWLv2 missing Riya. | verified on synthetic clips; suspected for sh11 itself (its sidecar shows the detections) |
| 11 | A second camera bug found while calibrating: phase correlation used whole-pixel peaks between consecutive frames, so at scale 8 a real 0.15-width pan measured **0.000**. The static-camera check missed slow pans, and the direction check couldn't correct for them. | verified |
| 12 | Model reloads: VACE takes about 30 s from the Drive cache, and the checkers load once per round. In fast mode that is under 5 % of a round. With turbo it becomes a larger share, so it is now counted and timed per round. All models together don't fit: VACE (4.6 GB of weights plus activations) and Qwen3-VL (4.3 GB) exceed what a T4 can hold safely. | measured (load) / estimated (fit) |

### B. Fixes

| File | Change |
|---|---|
| `engine/cells/shared/clips.py` | `turbo_lora_file()` downloads the LoRA to the session disk (`/content/…/lora`), then keeps one plain copy on Drive (`MODEL_CACHE/lora/`, no symlinks or locks). |
| `engine/cells/shared/clips.py` | `fuse_turbo()` loads the LoRA from that file, fuses it, and **verifies** that the base blocks changed and the VACE blocks did not. Any failure names its stage (download, read, load, fuse, unload, verify, scheduler). |
| `engine/cells/shared/clips.py` | `speed_up()` now **prints** `⚠️ Turbo mode FAILED at step <stage>: <error>` and records `turbo_error`. Each clip's sidecar records the speed actually used (`speed`). |
| `engine/cells/shared/clips.py` | `vae_guard()` wraps every VAE encode and decode: GPU-synchronized timing; untiled first, with a retry of the same call in tiles after out-of-memory (`VAE_TILING="auto"`; `True` = old behaviour); the all-zero latent is saved and reused (`VAE_ZERO_CACHE`, bit-identical, one VAE pass fewer per clip, and it survives VACE reloads and sessions). |
| `engine/cells/shared/clips.py` | Rounds are instrumented: VACE and checker load counts and times, release time, write time, check time, regenerations. A **"Where the time went"** table is printed, and `render/perf.json` is written. Plan pre-checks run before generation. |
| `engine/cells/shared/logic.py` | `make_vace_clip` times each stage: layout video, VAE encode, every diffusion step, VAE decode, frame conversion. GPU utilisation is sampled every 1 s. The timings go into each attempt's sidecar. |
| `engine/cells/shared/logic.py` | `enter`/`walk_to`/`exit` events are judged by the actor's start, end and direction checks (`event_motion`). The VLM is asked only when geometry can't measure the walk. |
| `engine/cells/shared/logic.py` | Camera masks now combine detected boxes with the **planned** boxes of every person, prop and moving thing in each frame. Static scenery stays in, because it is what shows a real camera move. |
| `engine/cells/shared/logic.py` | `precheck()` warns before generation when a subject is too small to detect or is on screen too briefly. |
| `engine/lib/motion_check.py` | Union masking per compared pair; **sub-pixel peaks**; an anchor frame every 8 frames; `VERSION = 2`. The version bump means cached clips are re-checked with the new rules **without regenerating**. |
| `engine/lib/acceptance.py` | New `event_motion` failure → `layout_strength` → `layout_simplify` → fallback, instead of `split_action`/`reseed`. |
| `engine/cells/shared/helpers.py` | `GpuSampler` (1 s nvidia-smi sampling: average and busy share) and `cuda_now()` (synchronized clock). |
| `engine/cells/shared/setup.py` | Logs the compute capability and the attention kernels. The T4 is 7.5: no native bf16 or fp8, and no FlashAttention-2; PyTorch picks its memory-efficient attention kernel itself. |
| `engine/cells/shared/bench.py` (new cell) + `build.py` | `BENCHMARK = True` times `BENCH_SHOT` under `BENCH_CONFIGS` (`baseline` = old fast mode with tiled VAE; `turbo_tiled`; `optimized`). Same seed and frames for each, and every result goes through the same QC. Writes `render/bench/benchmark.md` and `.json` plus clips and frame strips, then stops on purpose. |
| settings (both variants) | `VAE_TILING`, `VAE_ZERO_CACHE`, `BENCHMARK`, `BENCH_SHOT`, `BENCH_CONFIGS`. No new packages. |

The acceptance gate is **not weakened**:
- A walking event still needs its start, end and direction checks to pass; those are critical, and `walk_to` already worked this way.
- A real camera move is now measured better: the 0.15 pan reads 0.137 where it used to read 0.000.
- Unverified or failing clips still go to REVIEW or their fallback.

### C. Before and after

There is no GPU here, so **every GPU number below waits for the T4 benchmark**. No speed-up is claimed until it is measured.

| Measurement | Original (from your log) | Optimized |
|---|---|---|
| Model initialisation | ~30 s (VACE from the Drive cache) | measure: `BENCHMARK` |
| Diffusion | 13.6–23.2 min (25 steps, 24–68 s/it) | measure (4 single passes instead of up to 50) |
| VAE encode + decode | ~10–13 min outside the bar (2 encodes + 1 decode, tiled) | measure (1 encode + 1 decode after the first clip; untiled unless out of memory) |
| Total clip | 23.7–36.5 min | measure |
| Peak GPU memory | not recorded | logged per model and per clip |
| Average GPU utilisation | not recorded (snapshots only) | sampled every 1 s per clip |
| Model reloads | VACE + checkers once per round | counted in the "Where the time went" table |
| QC regenerations for sh11 | 2 (reseed, then static camera) | both causes fixed in the validator; measure on the next run |

What is verified on CPU: the zero-latent reuse is bit-identical and saves one VAE encode per clip; turbo fuses and is verified on VACE; the camera estimate is calibrated (table in `tests/test_speed.py` output).

### D. Workflow now

1. Plan and pre-check.
2. Prompts come from the Drive cache.
3. Round 1: load VACE once (from the Drive cache), fuse turbo (verified; a failure is shown on screen and recorded), and generate every pending shot. The VAE runs untiled, with an automatic tile fallback, and the zero latent is reused from file.
4. Release VACE. Load the checkers, check every candidate (camera on the background with planned + detected masks; walking by geometry; other actions by geometry + VLM), then release them.
5. Targeted repairs for the next round, at most `REPAIR_BUDGET`.
6. Finish: write sidecars (with timing and the speed actually used), print the perf table, write `perf.json`.

Fallbacks:
- turbo → fast → quality;
- untiled VAE → tiled;
- out of memory during generation → 25 % fewer frames;
- checker missing → UNKNOWN → REVIEW;
- not accepted → drawn fallback card.

### E. Colab instructions

1. Open the rebuilt `colab/colab_train_ticket_v1.ipynb`. No new packages are needed.
2. Optional, recommended once: set `BENCHMARK = True` and `BENCH_CONFIGS = ["optimized", "turbo_tiled", "baseline"]`, then Run all.
   - Remove `"baseline"` to save about 25 min.
   - The table appears on screen and in `render/bench/benchmark.md`; the notebook then stops on purpose.
   - Set `BENCHMARK = False` afterwards.
3. Normal run: check for `Speed: turbo (4 steps, Self-Forcing distilled model, verified)`. If it says `Turbo mode FAILED at step …`, that line is the root cause to report.
4. Clips already made are **kept**: the generation key is unchanged. They are re-checked with the new camera and walking rules.
5. After the clips cell, read the "Where the time went" table.

### F. Limitations that remain on the free tier

- **T4 (sm 7.5):** fp16 only (no fast bf16 or fp8), no FlashAttention-2, and about 65 TFLOPS of fp16 at best.
- **Session limits:** the Colab session length limit and idle disconnects still apply.
- **Downloads each session:** the Qwen3-VL and OWLv2 checkers download once per session (the 15 GB Drive is kept for the video caches).
- **Not tested on a T4 yet:** whether the untiled VAE fits at 81 frames depends on free GPU memory, and fallback-to-tiles covers it if not. VACE + turbo **visual quality** has to be judged on real clips: compare the benchmark's frame strips.

## 37. Telugu videos (2026-10-17): train ticket in Telugu

**Files:**
- `colab/train_ticket_te_plan.json` is the plan (project `train_ticket_te_v1`, `"language": "te"`).
- `colab/colab_train_ticket_te_v1.ipynb` is the notebook.
- `colab/scripts/train_ticket_te_script.md` is the readable script: every line, how the voice says it, the on-screen
  text, and an English back-translation.

The visuals, scene states, layouts, claims and sources are the same as the English video. Only the language-facing
parts change.

### What Telugu needed, and why

| Part | English pipeline | Problem for Telugu | Change |
|---|---|---|---|
| Voice | Kokoro (no Telugu) | no Telugu voice | **Indic Parler-TTS** (ai4bharat, Apache-2.0, trained Telugu speakers Prakash / Lalitha / Kiran; Telugu native-speaker score 89.8 on its model card). Stock voices, no cloning. |
| Voice dependencies | — | parler-tts pins `transformers==4.46.1`; the notebook needs 5.19 (Qwen3-VL) | It runs in its **own virtual environment**, set up once per session (~3 min) with `uv` from a **pinned lock** (`parler_worker.REQUIREMENTS`, resolved for Python 3.13). Torch, torchaudio, numpy and the CUDA libraries are removed from the lock, so the environment uses Colab's own copies; it is installed `--no-deps`. One **worker process** (`engine/lib/parler_worker.py`) loads the model once and answers one JSON request per sentence. |
| Pronunciation | text = speech | "WL 12", "PNR" would be read as English letters, or badly | Each line has `text` (subtitles: Telugu + PNR, WL 12) and `tts` (everything spelled in Telugu script: డబ్ల్యూ ఎల్ పన్నెండు). |
| Bad takes | Kokoro is deterministic | Parler samples: a take can be cut off or ramble | Generated **sentence by sentence**. A take whose length per letter is implausible (`PARLER_SEC_PER_CHAR`) is remade with a new seed, and each speech-check retry uses a new seed. |
| Speech check | `small.en`, word error ≤ 15 % | English-only model; word error is unfair to long, inflected Telugu words | Telugu switches to `ASR_MODEL_MULTI = "large-v3-turbo"` with `language="te"`, scored by **letter error** with a retry above `SPEECH_RETRY_OTHER = 0.35`. **This threshold is unverified**: tune it from the logged values of the first run. |
| Word normalisation | `[a-z0-9]` only | Telugu was deleted entirely, so every line scored 100 % wrong | Letters, vowel signs and digits of any script are kept (`norm_words`). |
| Fonts | DejaVu (no Telugu) | missing letters | **Hind Guntur** (ITF, OFL): Telugu **and** Latin in one font, so PNR, WL 12 and ₹ render too. It is downloaded once to the model cache and used for every drawn text (`drawkit.set_script_font`) and the subtitles (`SUB_FONT`, FFmpeg `fontsdir`). Noto Sans Telugu was rejected because it has no Latin letters. |
| Shaping | — | without raqm, Telugu vowel signs are placed wrongly (కనీ instead of కానీ, checked on a render) | Pillow's raqm shaping (in the pip wheels) is required: the notebook checks for it and stops with a fix if it is missing. Subtitles go through libass/HarfBuzz. |
| Validator | English word rules | `\b` breaks on Telugu vowel signs; suffixes such as PNRలో; English hedge words only; English time units only | Script-aware `words()` and `has_phrase()`, Telugu hedges (ఒకవేళ, కొన్నిసార్లు, and the -తే/-టే/-చ్చు endings), and Telugu time units (రోజు, గంట…), so P2 and P5 still check facts. English plans give identical results. |
| Shot-length estimate | words ÷ 3 per second | far too short for Telugu, so an AI shot could exceed one clip unnoticed | Letters × 0.1 s (conservative) for non-English plans. |

### What you need to do once

**A Hugging Face token** (the model is gated, approved instantly). It is never written in code:
1. Log in at huggingface.co.
2. Open huggingface.co/ai4bharat/indic-parler-tts and **accept the terms**.
3. Go to huggingface.co/settings/tokens → **New token**, type **Read**, and copy it.
4. In Colab, open 🔑 **Secrets** (left bar) → **Add new secret**. Name it `HF_TOKEN`, paste the token, and turn on **Notebook access**.

Without it, the narration cell stops with these steps.

### Verified, and not yet verified

- **Verified on CPU:**
  - the worker protocol with the real parler-tts code (the ungated English checkpoint, same model class, `transformers 4.46.1` inside the environment and the system's torch);
  - the same seed gives the same take and a new seed a new take;
  - the corrected lock installs `--no-deps` on top of the system's torch and numpy;
  - every character on screen exists in Hind Guntur;
  - validator behaviour on both plans;
  - `tests/test_telugu.py` (24 checks).
- **Not verified:**
  - the Telugu model itself (gated, so it couldn't be downloaded here) and its voice quality;
  - Whisper `large-v3-turbo` accuracy on Telugu, and with it the 0.35 threshold;
  - the Telugu wording should get a **native speaker's read-through** before upload.
  Listen to every line the narration cell flags.

**Addendum: Telugu subtitles.** The end-to-end run showed that the burned-in subtitles from libass (in the tested
FFmpeg 4.4.1 build) **detached Telugu vowel signs**: మ ీరు instead of మీరు. Non-English videos therefore draw each
subtitle cue with Pillow + raqm, the same shaping as the cards, and overlay them as one transparent image stream
(`drawn_subtitles()` in the render cell). Position, size, outline and speaker colours match the English subtitles;
there are no italics. English videos keep libass. This was checked on frames of the rendered Telugu video: the cards,
the step label and the subtitles are all shaped correctly. The `.srt` file is still saved for YouTube's own captions.

## 38. Topic → video notebook, final QC gate, word-highlighted subtitles (2026-10-18)

The audit of `claude.md`, every decision with its reasons, and the measured results are in
`docs/10_CLAUDE_MD_AUDIT_AND_DECISIONS.md`.

**To use it:** open `colab/colab_topic_to_video.ipynb` and set three things in the first cell:
- `TOPIC`;
- `WIKI_CONTACT` (your e-mail or a web page; Wikimedia requires contact details);
- `TARGET_SECONDS`.

Then pick a T4 and Run all. The topic cell prints the script and saves these files:

| File | Contents |
|---|---|
| `render/script.txt` | the script |
| `render/factcheck.json` | every sentence: PASS/FAIL/UNKNOWN, why, scores |
| `assets/manifest.json` | every image considered: licence decision, scores, credit |
| `plan.json` | the plan |
| `stages/*.json` | stage checkpoints (resume) |

The rest is the tested cast pipeline.

**New in every notebook:**
- `qc_report.json` now has per-check `status`, `evidence`, `shots` and `fix`, plus `approved`, which is true only when every mandatory check is PASS. An UNKNOWN blocks approval like a FAIL.
- `render/contact_sheet.jpg`.
- `SUB_STYLE = "karaoke"` (ASS subtitles: words light up as spoken; numbers and key terms in the accent colour).
- `RUN_MODE`.
- Configurable `DUCK`.

## 39. Expressive, scene-aware narration (2026-10-18)

The full design, candidate comparison, measurements and limits are in `docs/11_EXPRESSIVE_NARRATION.md`.

**Settings** (cast notebooks, including the topic notebook):

| Setting | What it does |
|---|---|
| `EXPRESSIVE = True` | every line gets a delivery direction from the story |
| `VOICE_PROFILE` | `science_explainer` (default), `beginner_education`, `cinematic_documentary`, `mystery_documentary`, `historical_narration` or `inspirational_story` |
| `VOICE_DIRECTOR` | `"rules"` (plan tones + story roles) or `"llm"` (a local Qwen reads the whole story) |
| `VOICE_BACKEND` | `"kokoro"` (pace and pauses only; the default) or `"qwen3-tts"` (follows the whole direction as an instruction; narrator voice `NARRATOR_QWEN_SPEAKER = "aiden"` or `"ryan"`) |
| `EXPRESSIVE_MAX_INTENSITY = 0.5` | caps how strong the delivery can get on instruction-following voices |
| `QWEN_TTS_DTYPE` | precision for Qwen3-TTS (`float32` by default on the T4) |
| `VOICE_BENCH = True` | runs the neutral-vs-expressive comparison for each backend, writes the reports, shows a blind A/B, then stops |

**What you get in `render/`:**
- `voice_direction.json` (the direction per line);
- `prosody_report.json` (what each backend actually did with each field: native, instruction, silence or unsupported);
- `voice_qc_report.json` (words, pace, gaps, clipping, pitch as a diagnostic, narrator identity).

Lines that fail are retaken calmer, or listed to listen to.

**How to choose the backend:**
1. Run the notebook once with `VOICE_BENCH = True` and listen to samples A and B for each backend.
2. Then open `render/voice_bench/blind_test_key.json` and check `voice_qc_report.json`: time, real-time factor,
   peak VRAM, words and identity.
3. Set `VOICE_BACKEND` accordingly and set `VOICE_BENCH = False`.

## 40. Two friends explain it: Arjun and Meera (2026-10-18)

**What changed:** the video can now be a conversation between two recurring 2D friends instead of a narrator over flowcharts:
- Arjun asks; Meera explains with demonstrations.
- They are drawn by code on the CPU: no GPU and no image model, and they look the same in every video.

**Try it:**
1. Open `colab/colab_three_d_friends_v1.ipynb` ("Why do 3D movies look real?", 89 s).
2. Run all. A GPU is not needed.
3. `RUN_MODE = "draft"` renders at 720p in a few minutes.
4. Look at `render/dialogue_shot_timeline.md` and `render/contact_sheet.jpg`.

**Your own topic:**
- `colab_topic_to_video.ipynb` with `STORY_MODE = "two_friends_cinematic"` (the default). This is English only.
- Or write a story JSON (see `colab/stories/three_d_friends_story.json`) and build it:
  `colab/build_story_plan.py <story> <plan> --project <name> --ground <hf cache>`
  The `--ground` option fact-checks every claim against its Wikipedia source.

**Settings:**
| Setting | What it does |
|---|---|
| `STORY_MODE` | two friends, or the narrator |
| `LICENSE_STRICT` | True: no model with an unclear licence; AI clips are slower without turbo |
| `LIP_SYNC` | `"energy"`: the mouth opens with the speaker's voice; `"none"`: the mouth stays closed |
| `MAX_AI_VIDEO_SHOTS` | budget for AI video shots |
| `MAX_GPU_MINUTES` | budget for clip making |

**Honest limits:**
- The mouth movement is not phoneme lip sync.
- Not tested on a GPU yet.
- The Qwen story quality has not been read yet.

Full details: `docs/12_TWO_FRIEND_STORYTELLING.md`.

## 41. More natural conversations and better stories (2026-10-18)

- **New sample:** `colab/colab_three_d_friends_v2.ipynb`.
  - Arjun guesses that the glasses contain tiny screens. Meera asks him to close one eye, and the ball goes flat.
  - The explanation follows from that test, and the ending calls back to the opening.
  - 78 s, approved by QC on CPU.
- **Voices:**
  - Replies now come after ~0.1–0.6 s (before: 0.45–1.2 s), and the gaps vary like real talk.
  - Each line is directed by why the character says it.
  - With `VOICE_BENCH = True`, the notebook compares old and new delivery (and Qwen3-TTS on a T4) with blind A/B players and a rating sheet.
- **Topic notebook:** stories now go premise → outline → dialogue → critique → revision. Settings: `MAX_STORY_PREMISES`, `MAX_STORY_REVISIONS`, `STORY_BENCH`.
- **Honest limits:**
  - No human listening yet.
  - No GPU run yet.
  - Telugu voice benchmark needs `HF_TOKEN` (Colab).

## 42. Fix: "Speech check unavailable … Unsupported model binary version v774843950" (2026-10-18)

**Cause:** the speech-check model (faster-whisper) was cached on Google Drive in the normal Hugging Face layout, where every file is a symlink into `blobs/`. Drive does not keep symlinks. In a later session `model.bin` contained only the link text: the "version" 774843950 is the bytes `../.`. The speech check and word timing were then skipped, and the subtitles used estimated timing.

**Fix:**
- Models cached on Drive (the speech check, and the CLIP continuity check) are now downloaded as plain files into `MyDrive/ai-video/_model_cache/hf_files/<repo>` (helpers cell, `hf_files`), at the licence-verified revision.
- They are checked before loading. Broken or incomplete files are re-downloaded automatically.
- Reproduced locally with a link-text `model.bin` (same error); the new loader repairs it and the model loads.

**What to do:** download the newly built notebook and Run all. The old `_model_cache/whisper` folder is no longer used; you may delete it from Drive.

**Harmless:** `WARNING:phonemizer:words count mismatch` comes from Kokoro's espeak phonemizer on punctuation; the speech check verifies the words.

## 43. Fix: "Telugu voice failed: IndexError: tuple index out of range" (2026-10-18)

**What happened:** on Colab, one sampled take of Indic Parler-TTS failed inside the model during the second (optional) take of a hero line. The error stopped the whole voice cell.

**Fix:**
- A take that fails inside the voice model is retried with a new seed (up to 3 per sentence), like a cut-off take.
- An extra hero take that still fails is logged and skipped; the first take is kept and the run continues.

**Tested:** a stand-in worker returns the same error for the first take of each sentence (`STUB_FAIL_FIRST`). The line is still made and the retries are logged (`tests/test_telugu.py`, 26 checks).

**What to do:** download the rebuilt notebook and Run all. Finished voices are reused from Drive.

## 44. Speech check running on the CPU (slow Telugu runs) (2026-10-18)

**Symptom:** "Speech check running on: CPU" on a GPU session. The Telugu run is slow because every take is checked by Whisper large-v3-turbo on Colab's 2 CPU cores.

**Likely cause** (the exact reason was only in the log):
- faster-whisper's GPU backend (CTranslate2 4.8) needs `libcublas.so.12` and cuDNN 9.
- On Colab these live inside the pip packages `nvidia-cublas-cu12` / `nvidia-cudnn-cu12`, outside the library search path.
- So the GPU attempt failed and the code fell back to the CPU, logging the reason without showing it.

**Fix:**
- Those libraries are loaded by full path before the GPU attempt (`preload_cuda_libs`).
- If the GPU still fails, the reason is now printed next to "running on: CPU".
- While the check runs on the CPU, the optional second take of hero lines is skipped, which saves one slow check per hero line.

**Measured (CPU, small.en, 6 lines):** greedy search was only ~6 % faster than beam 5 (18.4 s vs 19.5 s), so the beam was kept at 5. On CPU the time is spent in the encoder and word timing, not the search.

**Not verified here:** whether the library preload fixes it on Colab (no GPU on this machine). The printed reason will show it.

## 45. Telugu 3D video with Gemini free-tier voices (2026-10-19)

**Notebook:** `colab/colab_three_d_friends_te_v2.ipynb`.
- New natural Telugu script (the one-eye test).
- Optional Google Gemini free-tier voices.
- Without a Gemini key it uses the usual Indic Parler voices.

**Setup:**
1. AI Studio → make an API key in a project **without billing**.
2. Check **Projects → Billing Tier = Free**.
3. Colab 🔑 Secrets → `GEMINI_API_KEY`.
4. In the settings cell, set `GEMINI_FREE_TIER_CONFIRMED = True`.
5. Keep `HF_TOKEN` too (Parler is the fallback).

**First run:**
1. `TELUGU_VOICE_AUDITION = True`: choose Arjun's and Meera's voices.
2. `TELUGU_VOICE_BENCH = True`: Parler vs Gemini, blind listening sheet for native speakers.
3. Set both back to False and Run all.

**Limits** (local, not Google's): 1 video/day, ≤ 6 Gemini calls per video. The ledger is `MyDrive/ai-video/_gemini_usage.json`.

**What is sent to Google:** only the Telugu dialogue lines and short style words. Free-tier data may be used to improve Google's products and reviewed by people.

Details: `docs/14_TELUGU_GEMINI_TTS.md`.

## 46. Telugu–English 3D glasses video, v3 (2026-10-20)

**Notebook:** `colab/colab_three_d_friends_te_v3.ipynb`. Drive folder: `MyDrive/ai-video/three_d_friends_te_v3/`.

### First run (free Colab)

1. **Hugging Face access** (one time):
   - Log in to huggingface.co with the account that owns your token.
   - Open https://huggingface.co/ai4bharat/indic-parler-tts and click **Agree and access repository**.
   - Create a **Read** token and add it to Colab 🔑 Secrets as `HF_TOKEN`, with notebook access ON.
2. **Runtime → Change runtime type → T4 GPU.** On CPU it still works, just slowly. For a quick preview set `RUN_MODE = "draft"` (1280×720).
3. **Runtime → Run all.**
   - The voice cell prints `Telugu voices: local Indic Parler-TTS (STRICT_OPEN_SOURCE: …)`.
   - If access is missing, it stops before any download with the steps above.
4. **Lines marked ⚠️ CHECK THIS LINE** are lines the speech recogniser was unsure about. Code-mixed speech confuses Whisper, so listen to them. QC stays *not approved* until they are checked.

### Optional

- **Pronunciation listening test** (before the full voice pass):
  - Set `TELUGU_PRON_BENCH = True` and run.
  - It makes the opening exchange with Parler respelled vs Latin input, plus Gemini display vs respelled if Gemini is enabled.
  - A native speaker fills `render/pron_bench/listening_sheet.md`, then opens `blind_key.json`.
  - Set the flag back to `False`.
- **Gemini voices (proprietary, free tier, opt-in):**
  - Settings: `TELUGU_TTS_BACKEND = "gemini_free"`, `GEMINI_FREE_TIER_CONFIRMED = True` (only after checking AI Studio shows Billing Tier "Free"), and the Colab secret `GEMINI_API_KEY`.
  - Voices stay Arjun = Achird, Meera = Sulafat.
  - At most 6 calls per video. A recogniser doubt never triggers a new call.

### Re-runs

Re-runs are restart-safe: lines already made with the same settings are reused from Drive, and only changed lines are re-synthesized.

## 47. Storybook visuals (illustrated, rigged Arjun and Meera) (2026-10-20)

- **Notebook:** `colab/colab_three_d_friends_te_v3_storybook.ipynb`. It is the same 3D-glasses story as v3, drawn with the storybook characters and the cinema/lobby environment packs.
- **Run it:** Runtime → Run all.
  - The drawn-scene cell installs the embedded character/environment versions into `MyDrive/ai-video/_assets/` once, then reuses them.
  - Drawing runs on the CPU: about 0.12 s per frame at 720p and about 0.17 s at 1080p (measured on this machine; Colab's CPU may be slower). That is roughly 5–10 minutes for a 90 s video, cached on Drive.
- **Other plans:** add `--visual storybook_v1` to `build_story_plan.py`, then build the notebook as usual.
- **Approval:** QC stays "not approved" until the character artwork is approved. v1 is provisional, code-drawn art. For final art, run `colab_character_studio.ipynb` on a T4, pick candidates, finish the layers, and approve with `approve_character.py`.

## 48. Making the cinematic characters (one time) (2026-10-20)

1. Open `colab/colab_character_studio.ipynb` on a **T4** and Run all. It stops at the approval gate after the proof-of-concept contact sheet (`MyDrive/ai-video/_assets/studio/poc/`).
2. Look at the sheet and set `POC_APPROVED` (all `True`) and `POC_PICK` in cell 1. Run all again: it makes the atlas images, cut-outs, review sheets, the v2 import and previews.
3. Approve each character with `python3 colab/approve_character.py <MyDrive>/ai-video/_assets <id> v2 --by "<you>"`.
4. Copy the v2 folders to `colab/assets/characters/` and rebuild the plan with `--visual storybook_v1 --char-version v2`.

Out of memory: set `SIZE = 640` and re-run the cell; finished images are skipped. After a disconnect, Run all continues where it stopped.

## 49. Storybook notebook: no embedded images (fixes the frozen Colab page) (2026-10-20)

- **What happened:** `colab_three_d_friends_te_v3_storybook.ipynb` was 5.7 MB. Its plan cell held the character and background pictures as one base64 line of 4.6 MB, and Colab's code editor froze trying to lay out that line.
- **Now:**
  - The notebook is about 1.1 MB and its longest line is about 77 KB, the same as the v3 notebook that runs fine.
  - The pictures are in a separate optional file next to it: `storybook_assets_<id>.tar.gz` (3.5 MB). The notebook only stores its name and sha256.
- **First run, either:**
  - **Upload the pack** once to `MyDrive/ai-video/_assets/`. Colab installs it after checking the sha256, so the art is byte-identical to what was checked.
  - **Do nothing.** Colab draws the two provisional characters from the same code once, in parallel, and keeps them on Drive. Later runs reuse them.
- **Approved AI characters (v2)** are never embedded: the Character Studio writes them on Drive, and plans pin their hashes.

# 06 — Example Video Workflow: "How the Pyramids of Egypt Were Built"

**Status:** Phase 1 design. This is a worked example of what the pipeline will do. It has not been executed yet. Runtimes are estimates **[est]** for the Strategy B (laptop + Colab) setup described in `04_EXECUTION_STRATEGY.md`.
**Date:** 2026-10-11

**Request:**

> "Create a 3-minute documentary about how the pyramids of Egypt were constructed, with cinematic visuals, a professional English narration, historical explanations, subtitles, and background music."

**Command:**

```bash
python main.py create --topic "How the pyramids of Egypt were constructed" \
  --duration 180 --style cinematic_documentary --voice english --format youtube
```

---

## 1. Creative brief (Director stage output)

The Director LLM call turns the one-line request into a brief. Output: `projects/pyramids_giza/01_plan/brief.json`.

```json
{
  "schema_version": "1.0",
  "project_id": "pyramids_giza",
  "working_title": "How the Great Pyramid Was Really Built",
  "audience": "General audience, 16+, curious about history; no prior knowledge assumed",
  "angle": "Follow one limestone block from quarry to pyramid, using the evidence archaeologists actually found (Merer's papyri, the workers' town, quarry ramps) and being honest about what is still debated (how the blocks went up).",
  "hook": "Around 4,500 years ago, someone kept a logbook of the stones for the Great Pyramid, and we can still read it.",
  "target_duration_s": 180,
  "format": "youtube_16x9",
  "fps": 24,
  "style_preset": "cinematic_documentary",
  "tone": ["curious", "grounded", "awe without exaggeration"],
  "structure": [
    {"section_id": "s01", "title": "The question",           "target_s": 20},
    {"section_id": "s02", "title": "Quarries and copper tools","target_s": 35},
    {"section_id": "s03", "title": "Moving the stones",       "target_s": 35},
    {"section_id": "s04", "title": "Raising the pyramid",     "target_s": 40},
    {"section_id": "s05", "title": "The workers",             "target_s": 30},
    {"section_id": "s06", "title": "What we still don't know","target_s": 20}
  ],
  "narration": {"words_target": 430, "wpm": 145, "voice": "english_male_documentary"},
  "visual_policy": {
    "people_on_screen": "anonymous workers in wide/medium shots; no close-up faces of named historical figures",
    "text_on_screen": "rendered by code only (dates, labels, map)",
    "reconstruction_label": true
  },
  "music": "slow orchestral ambient, low strings and frame drum, warm; builds in s04, resolves in s06",
  "disclosure": {"contains_synthetic_media": true, "reason": "AI-generated realistic historical reconstructions"}
}
```

---

## 2. Production stages and timeline

| # | Stage | Where | Output | Time [est] |
|---|---|---|---|---|
| 1 | Brief | Laptop → LLM API | `01_plan/brief.json` | 15–30 s |
| 2 | Research + fact check | Laptop → LLM API with web search | `01_plan/research.json` (claims, sources, confidence) | 2–5 min |
| 3 | Script | LLM | `01_plan/script.json` (~430 words, 6 sections) | 30–60 s |
| 4 | Storyboard (one call per section) | LLM | `01_plan/storyboard.json` (~34 shots) | 1–2 min |
| 5 | Validate + critic | Python + LLM | `plan_review.json`, `production_plan.json` | 30–60 s |
| 6 | Narration TTS (per sentence) | Laptop CPU (Kokoro) | `04_audio/narration/*.wav` | 2–6 min |
| 7 | ASR check + word alignment | Laptop CPU (faster-whisper `small`, int8) | `words.json` | 2–5 min |
| 8 | Lock shot durations | Python | `production_plan.json` (timed) | < 1 s |
| 9 | Route jobs | Python | `02_jobs/routing.json`, job files | < 1 s |
| 10 | Graphics (map, timeline, title cards) | Laptop CPU (Pillow/Matplotlib) | `03_visuals/graphics/*.png` | < 1 min |
| 11 | Keyframes (T2I) | Colab GPU | `03_visuals/keyframes/*.png` | 10–25 min on T4, 3–8 min on L4/A100 |
| 12 | I2V clips (~12 hero shots) | Colab GPU | `03_visuals/clips/*.mp4` | 1–3 h on T4 (if it fits), 25–60 min on L4/A100 |
| 13 | Music (3 cues) + SFX (10 cues) | Colab GPU, or curated CC0 library | `04_audio/music/*.wav`, `sfx/*.wav` | 2–10 min |
| 14 | Keyframe/clip QC | Laptop (+ optional VLM API) | `06_qc/assets_qc.json` | 2–5 min |
| 15 | Shot segments (camera moves, normalize) | Laptop CPU (FFmpeg) | `05_render/shots/*.mp4` | 5–15 min |
| 16 | Mix + loudness | Laptop CPU (FFmpeg) | `04_audio/mix.wav` | < 1 min |
| 17 | Subtitles | Python from `words.json` | `subtitles.srt/.ass` | < 1 s |
| 18 | Final render | Laptop CPU (libx264) | `05_render/final_16x9.mp4` | 5–15 min |
| 19 | Final QC | Laptop CPU | `06_qc/qc_report.json` | 2–4 min |
| 20 | Metadata + thumbnail | LLM + Pillow | `07_publish/metadata.json`, `thumbnail.png` | 1–2 min |
| 21 | Human review and approval | You | `07_publish/approval.json` | — |
| 22 | Upload (private) | Laptop → YouTube API | `publish_record.json` | 1–5 min (upload speed) |

**Total [est]:** about **1.5–4 hours** of wall clock. Most of that is Colab GPU time for keyframes and I2V, and it depends on which GPU Colab assigns. Laptop-only steps take about 30–50 minutes.

---

## 3. Research output (excerpt) and how uncertainty is handled

The Researcher stage writes claims with sources and a confidence label. The scriptwriter may state **high**-confidence claims plainly. It must hedge **medium** ones, and it may use **contested** ones only as "one theory is…".

```json
{
  "claims": [
    {"claim_id": "c01", "text": "The Great Pyramid was built for the pharaoh Khufu in the 4th Dynasty, about 2600–2560 BCE.",
     "confidence": "high", "sources": ["https://www.britannica.com/topic/Great-Pyramid-of-Giza"]},
    {"claim_id": "c02", "text": "Papyri found at Wadi al-Jarf (the 'Diary of Merer') record a crew transporting limestone from Tura to Giza by boat during Khufu's reign.",
     "confidence": "high", "sources": ["<peer-reviewed or museum source found at run time>"]},
    {"claim_id": "c03", "text": "A 2014 study showed that wetting sand in front of a sledge greatly reduces the force needed to pull it, consistent with a scene in the tomb of Djehutihotep.",
     "confidence": "high", "sources": ["<Physical Review Letters 2014, Fall et al.>"]},
    {"claim_id": "c04", "text": "Exactly how blocks were raised to the upper levels (straight, zigzag, spiral or internal ramps, or levers) is still debated.",
     "confidence": "contested", "sources": ["…"]},
    {"claim_id": "c05", "text": "The workforce were organized, paid and fed laborers living in a nearby town, not slaves.",
     "confidence": "high", "sources": ["<AERA / Lehner & Hawass excavations at Heit el-Ghurab>"]},
    {"claim_id": "c06", "text": "Around 2.3 million blocks were used.",
     "confidence": "medium", "note": "Widely repeated estimate; phrase as 'roughly' or 'an estimated'."}
  ],
  "uncertainties": ["internal ramp theory", "exact workforce size (estimates range widely)"]
}
```

In this example the sources marked `<…>` are placeholders. The real run fills them with URLs the research stage actually fetched. The validator rejects any claim whose source list is empty.

**Rules applied:**
- Dates, numbers and names on screen come from `research.json`, never from the image model.
- Contested topics get a **"theories" graphic**: three labelled diagrams side by side. The video never shows one reconstruction as fact.
- Every shot showing reconstructed ancient activity is tagged `factual_status: "reconstruction"`. The description says "Reconstructions are AI-generated illustrations."

---

## 4. Representative script segment (section s03, "Moving the stones")

```text
[s03_n01] Most of the pyramid's core is local limestone, cut just a few hundred metres away.
[s03_n02] But the fine white casing stone came from Tura, across the Nile.
[s03_n03] And we know how some of it travelled, because one of the men who moved it kept a diary.
[s03_n04] Fragments of papyrus found at Wadi al-Jarf record an inspector named Merer and his crew,
          shipping limestone blocks by boat along canals to Giza.
[s03_n05] On land, heavy blocks were dragged on wooden sledges.
[s03_n06] A painting in an Egyptian tomb shows a man pouring water in front of a sledge,
          and experiments in 2014 showed why: damp sand can roughly halve the force needed to pull it.
```

- ~95 words, about 39 s at 145 wpm. This is slightly over the 35 s budget, so the critic trims one clause.
- Each sentence has an ID. Shots reference these IDs, and alignment gives each one an exact start and end time.
- The script also carries per-sentence pronunciation hints for the TTS stage: `{"Djehutihotep": "jeh-HOO-tee-HOH-tep", "Wadi al-Jarf": "WAH-dee al-JARF"}`.

---

## 5. Scene-by-scene storyboard

Modes are defined in `02_ARCHITECTURE.md §4`:
- `SM` = still_motion (keyframe + FFmpeg camera move)
- `I2V` = image-to-video clip
- `GFX` = deterministic graphic

Durations are the planned values. They are re-locked to narration timing after TTS. 34 shots, 180 s.

| Shot | Dur (s) | Mode | Size / angle / lens | Camera move | Content | Transition | Audio cue |
|---|---|---|---|---|---|---|---|
| **s01 — The question (20 s)** ||||||||
| s01_sh01 | 4.0 | I2V | extreme wide / low / 24mm | slow push_in | Three Giza pyramids at dawn, mist, sun breaking behind | fade_from_black | wind, low drone |
| s01_sh02 | 5.0 | SM | wide / eye level / 35mm | slow pan_right | Great Pyramid's eroded stepped face, modern day, golden light | cut | — |
| s01_sh03 | 5.0 | SM | close_up / low / 50mm | slow tilt_up | Weathered limestone blocks, human hand for scale | cut | — |
| s01_sh04 | 6.0 | GFX | — | slow push_in | Title card: "Khufu's pyramid · c. 2560 BCE · ~146 m tall (originally)" | crossfade | music in |
| **s02 — Quarries and copper tools (35 s)** ||||||||
| s02_sh01 | 5.0 | GFX | map | slow push_in | Map: Giza, Tura, Aswan, Nile, Wadi al-Jarf (Natural Earth data) | dip_to_black | — |
| s02_sh02 | 5.0 | I2V | wide / high / 28mm | slow crane_down | Reconstruction: limestone quarry beside the plateau, workers cutting channels | cut | distant chisels |
| s02_sh03 | 4.0 | I2V | close_up / eye level / 85mm | static | Copper chisel struck with a wooden mallet, chips flying | cut | metal tap |
| s02_sh04 | 4.0 | SM | medium / eye level / 50mm | slow push_in | Dolerite pounding stones beside a granite block | cut | — |
| s02_sh05 | 5.0 | SM | wide / eye level / 35mm | slow pan_left | Wooden wedges in a cut channel, workers' tools on the ground | cut | — |
| s02_sh06 | 6.0 | I2V | medium / high / 35mm | slow truck_right | Team levering a freed block onto a sledge | cut | wood creak |
| s02_sh07 | 6.0 | SM | extreme wide / aerial / 24mm | slow pull_out | Quarry seen from above, scale of the operation | crossfade | — |
| **s03 — Moving the stones (35 s)** ||||||||
| s03_sh01 | 5.0 | SM | wide / eye level / 35mm | slow pan_right | Giza core-stone quarry next to the pyramid | cut | — |
| s03_sh02 | 5.0 | I2V | extreme wide / aerial / 24mm | drone_forward | Reconstruction: Nile flood plain, boats under sail toward Giza | cut | water, wind |
| s03_sh03 | 6.0 | SM | close_up / high / 85mm | slow push_in | Papyrus fragment with hieratic script (generic, illegible; not a copy of the real papyrus) | crossfade | paper rustle |
| s03_sh04 | 5.0 | GFX | — | static | Label: "Diary of Merer · found 2013 · Wadi al-Jarf" + simple route line on map | cut | — |
| s03_sh05 | 6.0 | I2V | medium / low / 35mm | slow truck_left | Wooden cargo boat with a limestone block, rowers | cut | oars |
| s03_sh06 | 4.0 | I2V | close_up / low / 50mm | static | Water poured on sand in front of a sledge runner | cut | water pour |
| s03_sh07 | 4.0 | I2V | wide / eye level / 35mm | slow pan_left | Team hauling a sledge with ropes across damp sand | cut | rope strain, chant |
| **s04 — Raising the pyramid (40 s)** ||||||||
| s04_sh01 | 5.0 | I2V | extreme wide / aerial / 24mm | slow orbit_right | Reconstruction: half-built pyramid, ramps, thousands of workers | dip_to_black | crowd bed |
| s04_sh02 | 8.0 | GFX | — | sequential reveal | "Theories" diagram: straight ramp / zigzag ramp / internal spiral ramp | cut | — |
| s04_sh03 | 5.0 | SM | wide / eye level / 35mm | slow push_in | Hatnub alabaster quarry ramp with post holes (illustrative) | cut | — |
| s04_sh04 | 5.0 | I2V | medium / eye level / 50mm | slow tilt_up | Block levered into place on a course, workers with ropes | cut | stone grind |
| s04_sh05 | 4.0 | SM | close_up / eye level / 85mm | static | Mason checking a joint with a plumb line | cut | — |
| s04_sh06 | 6.0 | SM | wide / low / 24mm | slow crane_up | Polished white Tura casing on the upper courses catching sunlight | cut | music swell |
| s04_sh07 | 7.0 | I2V | extreme wide / aerial / 24mm | slow pull_out | Completed pyramid gleaming white, river and fields | crossfade | — |
| **s05 — The workers (30 s)** ||||||||
| s05_sh01 | 5.0 | SM | wide / high / 28mm | slow pan_right | Reconstruction: workers' town, bakeries, smoke | cut | ambient town |
| s05_sh02 | 4.0 | I2V | medium / eye level / 50mm | static | Bread baked in clay moulds, steam | cut | fire crackle |
| s05_sh03 | 5.0 | SM | medium / eye level / 35mm | slow push_in | Work gang resting, sharing food (anonymous, no faces in focus) | cut | — |
| s05_sh04 | 5.0 | GFX | — | static | "Paid, fed, organized in named gangs" (with source-backed labels) | cut | — |
| s05_sh05 | 5.0 | SM | close_up / eye level / 85mm | slow push_in | Hieroglyphic graffiti of a gang name on a stone (generic, not legible text) | cut | — |
| s05_sh06 | 6.0 | SM | wide / eye level / 35mm | slow pull_out | Modern-day excavation trench at Giza (illustrative) | crossfade | — |
| **s06 — What we still don't know (20 s)** ||||||||
| s06_sh01 | 5.0 | SM | extreme wide / low / 24mm | slow push_in | Great Pyramid at dusk, first stars | cut | wind |
| s06_sh02 | 5.0 | SM | close_up / low / 50mm | slow tilt_up | Edge of the pyramid against the sky | cut | — |
| s06_sh03 | 5.0 | I2V | extreme wide / low / 24mm | slow pull_out | Night sky rotating above the pyramid | cut | music resolve |
| s06_sh04 | 5.0 | GFX | — | fade | End card: sources note + "Reconstructions are AI-generated illustrations" | fade_to_black | — |

- **Mode mix:** 12 I2V (35%), 16 SM (47%), 6 GFX (18%). This matches the default budget in `settings.yaml`.
- **Bible entries used for continuity:**
  - `loc_giza_plateau`: canonical description of the plateau, river and light.
  - `obj_sledge`: wooden sledge with curved runners and rope lashings.
  - `char_worker_gang`: linen kilts, shaved heads or short dark hair, no jewellery. A group description only; there are no named individuals.
  - `obj_casing_stone`: fine white limestone, polished.
- The plateau has one approved **reference keyframe**. Every later shot that shows the plateau is generated from a prompt containing the same canonical text, and its CLIP similarity to that reference is checked.

---

## 6. Generation prompts (representative)

The style preset `cinematic_documentary` appends a fixed suffix and negative prompt to every keyframe prompt:

- **Suffix:** `cinematic documentary still, photoreal, natural light, 35mm film look, subtle grain, high dynamic range, historically plausible`
- **Negative:** `text, letters, watermark, logo, modern objects, cars, power lines, tourists, extra limbs, deformed hands, blurry, lowres, cartoon, oversaturated`

| Shot | Keyframe prompt (T2I) | Motion prompt (I2V) |
|---|---|---|
| s01_sh01 | `Three pyramids of Giza at dawn seen from the desert to the south-west, low mist over the sand, the sun rising behind the Great Pyramid, long shadows, extreme wide shot, 24mm, low angle` + suffix | `slow forward dolly toward the pyramids, mist drifting gently, sunlight slowly intensifying, stable camera, no new objects` |
| s02_sh03 | `Close-up of an ancient Egyptian worker's hands holding a copper chisel against pale limestone, wooden mallet raised, stone dust in the air, warm sunlight, 85mm shallow depth of field` + suffix | `the mallet strikes the chisel once, small chips of limestone fly, dust puff, camera static` |
| s03_sh02 | `Aerial view of the Nile flood plain in ancient Egypt, wooden sailing boats carrying white limestone blocks on a canal, green fields, palm trees, the Giza plateau with an unfinished pyramid on the horizon, morning haze, 24mm` + suffix + `{loc_giza_plateau}` | `smooth aerial forward movement following the boats, sails filling with wind, gentle water ripples` |
| s03_sh06 | `Low close-up of the curved wooden runner of a sledge on desert sand, a worker's hand pouring water from a clay jar onto the sand in front, droplets catching sunlight` + suffix + `{obj_sledge}` | `water pours from the jar and darkens the sand, the sledge runner begins to slide forward, camera static` |
| s04_sh01 | `Extreme wide aerial view of a half-built pyramid in ancient Egypt, mud-brick construction ramps, thousands of small figures of workers, stone blocks on sledges, dust haze, late afternoon light` + suffix + `{loc_giza_plateau}` | `slow orbiting aerial movement to the right around the construction site, tiny figures moving, dust drifting` |

**An unsupported request, handled without stopping.** For s04_sh02 ("Theories"), the storyboard stage wanted an animated 3D cutaway of an internal spiral ramp. That recipe does not exist yet, so the plan uses the closest recipe and logs the wish:

```json
"visual": {"recipe": "comparison_panels",
           "params": {"panels": [{"label": "Straight ramp"}, {"label": "Zigzag ramp"}, {"label": "Internal spiral ramp"}]}},
"wish": {"description": "animated 3D cutaway of an internal spiral ramp rising through the pyramid",
         "closest_recipe": "comparison_panels", "reason": "no 3D recipe in the menu"}
```

The video renders normally. The wish goes to `recipe_backlog.jsonl`. If later videos keep asking for 3D cutaways, a Blender `cutaway_3d` recipe is built and tested, then added to the menu.

**Prompt rules the validator enforces:**
- No text, letters or numbers inside image prompts.
- Bible descriptions are inserted by code, not paraphrased by the LLM.
- Motion prompts describe **only motion and camera**, because the keyframe already defines the content.
- Every I2V shot has a still_motion fallback: same keyframe, same planned camera move.

---

## 7. Representative structured scene (from `production_plan.json`)

```json
{
  "shot_id": "s03_sh06",
  "beat_id": "s03_b03",
  "narration_ids": ["s03_n06"],
  "duration_s": 4.2,
  "purpose": "Show the physical trick (wetting the sand) that the narration explains",
  "render_mode": "i2v",
  "fallback_mode": "still_motion",
  "visual": {"recipe": "i2v_clip", "params": {"keyframe": "03_visuals/keyframes/s03_sh06.png", "seed": 90311, "trim": [0.0, 4.2]}},
  "overlays": [],
  "wish": null,
  "camera": {
    "shot_size": "close_up",
    "angle": "low",
    "lens_mm": 50,
    "move": "static",
    "move_speed": "very_slow",
    "depth_of_field": "shallow"
  },
  "subject": "Wooden sledge runner on sand, water being poured in front of it",
  "action": "Water darkens the sand; the runner slides forward",
  "environment": "Open desert ground near the Giza plateau",
  "lighting": "Hard late-morning sun, warm, high contrast",
  "time_of_day": "morning",
  "continuity_refs": ["obj_sledge", "char_worker_gang"],
  "image_prompt": "Low close-up of the curved wooden runner of a sledge on desert sand, a worker's hand pouring water from a clay jar onto the sand in front, droplets catching sunlight. Wooden sledge with two curved acacia-wood runners and rope lashings. cinematic documentary still, photoreal, natural light, 35mm film look, subtle grain, high dynamic range, historically plausible",
  "motion_prompt": "water pours from the jar and darkens the sand, the sledge runner begins to slide forward, camera static",
  "negative_prompt": "text, letters, watermark, logo, modern objects, extra limbs, deformed hands, blurry, lowres, cartoon",
  "on_screen_text": null,
  "transition_in": "cut",
  "audio_cues": [
    {"cue_id": "sfx_water_pour", "kind": "sfx", "description": "water poured from a clay jar onto sand", "offset_s": 0.2, "gain_db": -16}
  ],
  "factual_status": "reconstruction",
  "claim_ids": ["c03"],
  "generation": {
    "keyframe": {"adapter": "t2i.default", "width": 1344, "height": 768, "seed": 41872},
    "clip": {"adapter": "i2v.default", "width": 1280, "height": 704, "frames": 121, "fps_native": 24, "seed": 90311}
  },
  "timing": {"start_s": 108.34, "end_s": 112.54, "locked": true}
}
```

- `t2i.default` and `i2v.default` resolve through `config/models.yaml`, so the scene never names a model directly. See `03_TECHNOLOGY_COMPARISON.md` for the current defaults.
- With the default Wan 2.2 TI2V-5B adapter, a 121-frame clip at 24 fps is about 5 s. It is trimmed to the locked 4.2 s, so no frame-rate conversion is needed. A 16 fps adapter such as Wan 2.2 A14B is converted to 24 fps with RIFE, or with FFmpeg `minterpolate` as a fallback.

---

## 8. Model and tool selection per stage

These are the defaults from `03_TECHNOLOGY_COMPARISON.md` (Strategy B, prototype):

| Stage | Tool / model | Runs on | License note |
|---|---|---|---|
| Brief, research, script, storyboard, critic, metadata | Hosted LLM API with JSON-schema output (offline option: Qwen3-class model via Ollama on a GPU) | Laptop (API) | API terms, or Apache-2.0 for Qwen |
| Web research | LLM tool use: search + fetch, sources saved | Laptop | — |
| Narration | **Kokoro-82M** (CPU) | Laptop | Apache-2.0 |
| ASR check + alignment | **faster-whisper** (word timestamps) | Laptop | MIT |
| Graphics (map, title, theory diagrams) | **Pillow + Matplotlib + Natural Earth** data | Laptop | Open / public domain data |
| Keyframes | **Z-Image-Turbo** or **FLUX.2 klein 4B** (FLUX.1-schnell as a fallback) | Colab GPU | Apache-2.0 (all three) |
| Image-to-video | **Wan 2.2 TI2V-5B** (1280×704, 24 fps) on an L4 or A100 | Colab GPU | Apache-2.0 |
| Frame-rate conversion | **RIFE** (or FFmpeg `minterpolate`) | Colab / laptop | MIT / LGPL |
| Music | **ACE-Step 1.5** on Colab, or a curated CC0/licensed track library | Colab / laptop | MIT |
| SFX | **Stable Audio Open** (Community License) or **Freesound CC0** files | Colab / laptop | Check license per file |
| Camera moves on stills | **FFmpeg** `zoompan`/`crop` with supersampled source | Laptop | LGPL/GPL build |
| Edit, mix, ducking, loudness, encode | **FFmpeg** (`xfade`, `sidechaincompress`, `loudnorm`, libx264) | Laptop | — |
| Subtitles | Python → SRT/ASS from word timestamps | Laptop | — |
| QC | ffprobe, FFmpeg detect filters, faster-whisper WER, imagehash, optional VLM review | Laptop | — |
| Thumbnail | Best keyframe + **Pillow** text | Laptop | — |
| Upload | **google-api-python-client** + **google-auth-oauthlib** | Laptop | Apache-2.0 |

---

## 9. Files: inputs and outputs

```text
projects/pyramids_giza/
├── project.json                               ← CLI args + settings snapshot
├── 01_plan/  brief.json  research.json  script.json  storyboard.json
│             plan_review.json  production_plan.json
├── 02_jobs/  routing.json  remote/kf_*.json  remote/i2v_*.json  remote/music_*.json
├── 03_visuals/
│   ├── graphics/   s01_sh04.png s02_sh01.png s03_sh04.png s04_sh02.png s05_sh04.png s06_sh04.png
│   ├── keyframes/  28 × <shot_id>.png + .json         (1344×768 → upscaled 2× for camera moves)
│   ├── clips/      12 × <shot_id>.mp4 + .json         (~5 s, 1280×704, 24 fps native with TI2V-5B)
│   └── rejected/   failed attempts (auto-cleaned after the project is published)
├── 04_audio/
│   ├── narration/  ~32 × s0X_n0Y.wav                 (24 kHz mono)
│   ├── narration.wav  words.json
│   ├── music/      m01_open.wav m02_build.wav m03_resolve.wav
│   ├── sfx/        ~10 cue wavs
│   └── mix.wav                                     (48 kHz stereo, −14 LUFS integrated, −1.5 dBTP)
├── 05_render/
│   ├── timeline.json                               ← exact EDL
│   ├── shots/      34 × <shot_id>.mp4              (1920×1080, 24 fps, yuv420p, no audio)
│   ├── subtitles.srt  subtitles.ass
│   ├── final_16x9.mp4                              (~180 s, H.264 High CRF 18, AAC 192k)
│   └── contact_sheet.jpg
├── 06_qc/   assets_qc.json  qc_report.json
├── 07_publish/ metadata.json  thumbnail.png  approval.json  publish_record.json
└── logs/    pipeline.jsonl  llm/*.json
```

**Disk use [est]:**
- Keyframes ~150 MB, clips ~150 MB, shot segments ~400 MB, final file ~150–250 MB.
- Model weights are cached on Colab or Drive, **not** in the project folder. Total ~1 GB per project before cleanup.

---

## 10. How the video is assembled

1. **Timeline:** `rendering/timeline.py` reads `production_plan.json` + `words.json` and writes `timeline.json`. Each entry holds the shot ID, source file, in/out points, the camera move parameters, the transition and the grade.
2. **Per-shot segments** (all normalized to 1920×1080, 24 fps, yuv420p, SAR 1):

   ```bash
   # still_motion: slow push-in on a 2× supersampled keyframe (avoids zoompan jitter)
   ffmpeg -loop 1 -i kf/s01_sh02.png -t 5.0 -vf \
     "scale=3840:-1,zoompan=z='min(zoom+0.0006,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=120:s=1920x1080:fps=24,format=yuv420p" \
     -c:v libx264 -crf 16 -preset medium shots/s01_sh02.mp4

   # i2v clip: trim to the locked duration, scale to 1080p (TI2V-5B is already 24 fps; 16 fps models go through RIFE first)
   ffmpeg -i clips/s03_sh06.mp4 -t 4.2 -vf "scale=1920:1080:flags=lanczos,fps=24,format=yuv420p" \
     -an -c:v libx264 -crf 16 shots/s03_sh06.mp4
   ```

3. **Join:**
   - Runs of hard cuts use the concat demuxer.
   - Crossfades and dips use `xfade` (`fade`, `fadeblack`) with offset = cumulative start − transition length.
   - Section grades use `lut3d` with the preset `.cube` file.
4. **Audio:**
   - Narration sentences are placed at their aligned times. Music cues are placed per section.
   - Music is ducked under the voice with `sidechaincompress` and SFX are placed at their offsets.
   - Loudness uses two-pass `loudnorm` (I=−14, TP=−1.5, LRA=11).
5. **Subtitles:**
   - Lines of at most 42 characters, at most 2 lines and at most 17–20 characters per second, cut at sentence and phrase boundaries from word timestamps.
   - The SRT goes to YouTube as a caption track (preferred). Burned-in ASS is optional (`--burn-subtitles`).
6. **Encode:**

   ```bash
   ffmpeg -i video_joined.mp4 -i mix_norm.wav -map 0:v -map 1:a \
     -c:v libx264 -profile:v high -preset slow -crf 18 -pix_fmt yuv420p -g 48 -bf 2 \
     -c:a aac -b:a 192k -ar 48000 -movflags +faststart -shortest final_16x9.mp4
   ```

7. **Final QC** (hard gates): duration 180 ± 5 s, 1920×1080, 24 fps, H.264 + AAC, no black segment > 0.5 s except the planned fades, no freeze > 2 s, integrated loudness −14 ± 1 LUFS, final-mix WER vs script < 8%, all 34 shots present, no near-duplicate shots.

---

## 11. YouTube metadata and publishing

`07_publish/metadata.json` (generated by the LLM, then validated in code against YouTube's limits):

```json
{
  "title": "How the Great Pyramid Was Really Built (What the Evidence Shows)",
  "description": "Around 4,500 years ago, an inspector named Merer kept a logbook of limestone shipments for Khufu's pyramid...\n\nChapters\n0:00 The question\n0:20 Quarries and copper tools\n0:55 Moving the stones\n1:30 Raising the pyramid\n2:10 The workers\n2:40 What we still don't know\n\nSources\n- ...\n\nAbout this video: narration is synthetic (text-to-speech). Historical scenes are AI-generated reconstructions for illustration and are not photographs or footage of real events.\n\nref:pyramids_giza",
  "tags": ["pyramids", "ancient egypt", "great pyramid of giza", "khufu", "history documentary", "archaeology", "diary of merer"],
  "category_id": "27",
  "default_language": "en",
  "default_audio_language": "en",
  "privacy_status": "private",
  "self_declared_made_for_kids": false,
  "contains_synthetic_media": true,
  "publish_at": null
}
```

- Category 27 is Education.
- Chapters are derived from the real section start times in `timeline.json`. YouTube needs the first chapter at 0:00, at least three chapters, and each at least 10 s long.
- `ref:pyramids_giza` is the idempotency marker used to detect a previous upload (see `07_YOUTUBE_AUTOMATION.md`).

**Publishing sequence:**

```bash
python main.py review  --project projects/pyramids_giza      # opens final video, metadata, QC report
python main.py approve --project projects/pyramids_giza      # writes approval.json (who/when/privacy)
python main.py publish --project projects/pyramids_giza --privacy private
```

1. Pre-flight check:
   - `qc_report.json` passed.
   - `approval.json` exists and matches the final file's hash.
   - Metadata is within limits.
   - `contains_synthetic_media` is set.
2. `videos.insert` as a resumable upload with `privacyStatus=private`. The session URI is saved, so an interrupted upload resumes.
3. `thumbnails.set` (needs a verified channel), then `captions.insert` with the SRT file.
4. Write `publish_record.json` with `video_id`, `https://www.youtube.com/watch?v=<id>`, Studio URL, timestamps and quota used.
5. Changing to public or scheduled is a **separate, explicit command** (`publish --privacy public`, or `--publish-at`), after you watch the private upload.

---

## 12. Copyright, model limits and historical reconstruction

| Issue | How the pipeline handles it |
|---|---|
| **Factual uncertainty** | Every narrated claim maps to a sourced claim with a confidence label. Contested topics are presented as competing theories. The critic stage flags unsupported statements. The final review page lists every claim with its source. |
| **Historical reconstruction** | Reconstruction shots are labelled in the plan and disclosed in the description. Prompts say "historically plausible" and avoid named people's faces. The bible keeps costumes and tools consistent within the video, but they are illustrations, not evidence. |
| **Artefacts and real documents** | The real Merer papyrus, real tomb paintings and museum objects are **not** copied. Generic illustrative versions are generated instead. If you want the real objects, use public-domain or properly licensed photos (for example museum open-access collections), recorded in `licenses.json`. |
| **Copyright of generated media** | Only models whose licenses allow commercial use of outputs are enabled (see `01_RESEARCH_REPORT.md`). Music and SFX come from MIT/CC0/licensed sources and are logged per file. No reference images of copyrighted films or artworks are used. |
| **Model limits** | Image models garble text, so all text is drawn by code. Video models drift after ~5 s, so clips are short and trimmed. Crowds and hands are a common failure, so QC samples those shots and falls back to still_motion. Maps are generated from data, not from the image model. |
| **YouTube disclosure** | `containsSyntheticMedia=true` is set, plus a plain-language note in the description. The video remains private until you approve it. |

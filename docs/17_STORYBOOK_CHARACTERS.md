# 17. Storybook characters and cinematic environments (visual upgrade)

Written 2026-10-20 for the claude.md brief "Upgrade My AI Educational Video Automation System".

Code:
- `colab/engine/lib/storybook_art.py`, `storybook.py`, `storybook_scene.py`, `envfactory.py`, `studio.py`
- `colab/assets/`
- `colab/build_studio.py` → `colab_character_studio.ipynb`
- `colab/approve_character.py`
- `colab/storybook_previews.py`
- `colab/tests/test_storybook.py`

## 1. Phase 0: what existed and what was missing

| Area | Before this change | Reused as is? |
|---|---|---|
| Topic → research → facts → NLI check | `topic.py`, `research.py`, `factcheck.py` | yes |
| Story + dialogue (two friends) | `story.py` (premise → outline → dialogue → critic → grounding), `telugu_dialogue.py` | yes |
| Voices, voice direction, lip timing | `narration` cells, `voice_direction.py`, per-line WAV; mouth = loudness | voices yes; mouths extended (visemes) |
| Visual director | `story.direct()` → per-visual recipe (character / demo / card / photo / clip) | yes, plus 2 optional fields |
| Characters | `charanim.py`: circles, ellipses and lines drawn per frame from the Bible's colours | **replaced for new plans** (kept for old ones) |
| Backgrounds | `charanim.world_image`: one flat drawing per world | **replaced for new plans** by environment packs |
| Demonstrations | `demos.py` (eye views, projector + glasses, finger test), code-drawn | yes |
| Drawn-shot pipeline | `drawkit.render` → ffmpeg; content-hash cache on Drive (`diagrams.py`) | yes (key extended for storybook plans) |
| QC | `qc.py` PASS / FAIL / UNKNOWN, approval only if every mandatory check PASSes | yes, plus 3 checks |

Baseline before any change: all core suites passed (2026-10-20):
- test_gemini_telugu, test_story, test_story_v2, test_telugu, test_lib, test_qc, test_subtitles;
- test_registry, test_voice_backend, test_consistency, test_planner, test_assets;
- test_tenglish (93/93).

The missing pieces were:
- a versioned character asset library;
- animation-ready layered artwork with a rig;
- topic environment packs with camera framing;
- a way to produce approved artwork.

## 2. Architecture change: the smallest reliable version

```text
EXISTING research → facts → story/dialogue → Visual Story Director (story.direct)
                                                   │ visual spec (world, shot, subject, chars, changes, camera)
               plan["visual_style_version"] = "storybook_v1" ?  ── no ──► charanim (legacy, unchanged)
                                                   │ yes
            ┌──────────────────────────────────────┴───────────────────────────┐
   CHARACTER DIRECTOR (storybook_scene)                          ENVIRONMENT DIRECTOR (envfactory)
   pinned asset version (plan["character_assets"])              pinned pack version (plan["environments"])
   pose/gesture timeline + 0.45 s eased transitions              world + shot + subject → camera preset
   expression → eyes/brows/mouth, blink, gaze, ¾ turn            layered master → static plate (cached)
   visemes from the speaker's own audio                          restrained camera move + foreground parallax
            └──────────────────────────────────────┬───────────────────────────┘
                                        fixed compositor (Pillow, CPU) → drawkit → ffmpeg → EXISTING QC
```

- **Fixed executor stays:** the planner writes data; the tested renderer draws it. No per-video code generation.
- **One animation backend:** the Python cutout rig (Pillow).
  - Godot was evaluated and not adopted (§6).
  - OpenToonz was not added.
- **Character creation and environment creation are separate one-time workflows:**
  - the Character Studio notebook;
  - the environment builder.
  - Routine renders never load an image model.

## 3. The asset library

```text
colab/assets/  (in Colab: MyDrive/ai-video/_assets/, set by STORYBOOK_ASSETS)
  characters/<id>/<version>/manifest.json    status, provenance, palette, identity, rig (joints, limits, layer order),
                                             poses, expressions, visemes, every layer: file, offset, sha256
  characters/<id>/<version>/layers/*.png     75 cropped RGBA parts per character
  environments/<env>/<version>/manifest.json sets (layers + depth + lighting), camera presets (crop, move, character
                                             slots), world map, subtitle/diagram safe regions, provenance
  environments/<env>/<version>/layers/*.png  master-size (2400x1350) RGBA layers
```

- **Immutable:**
  - A version folder is written to `<version>.partial` and renamed when complete. Rebuilding an existing version is refused.
  - `verify()` re-hashes every layer, and a damaged folder refuses to load.
  - The only allowed edit is the approval (status, who, when).
- **Pinned:**
  - `build_story_plan.py --visual storybook_v1` writes `character_assets` and `environments` into the plan.
  - Approved artwork is pinned by its manifest sha256, so a plan can never be rendered with different art.
  - Provisional code-drawn art pins the version and art version, and may be rebuilt in Colab from code (once, then reused from Drive).
- **No silent degradation:**
  - Missing or damaged assets are errors.
  - An approved version missing on Drive stops the run with instructions.
  - Nothing falls back to simple shapes.

## 4. The characters (v1 = PROVISIONAL, code-drawn)

- **Design (Character Bible text from the brief):**
  - **Arjun:** warm medium-brown skin, big dark-brown eyes, short wavy hair with a **signature curl**, **teal hoodie** over a **yellow T-shirt**, dark trousers, orange-red sneakers.
  - **Meera:** warm brown skin, big eyes with lashes, wavy **half-up hair with a bun** and a **yellow star clip**, **coral jacket** over a **lavender top**, lavender sneakers.
- **How it is drawn (`storybook_art.py`):**
  - cubic-Bézier shapes, gradient form shading, rim light, soft coloured outlines;
  - two-highlight eyes; every part is drawn complete (shoulder caps, neck under the collar), so joints rotate without seams;
  - 2× supersampled.
- **Parts:** head, ears, neck, torso, legs, upper arms, forearms, 4 hand shapes per side, irises (normal and small), 6 eye states plus lids plus eye masks, 6 brow shapes, 13 mouth shapes, front and back hair, 3D glasses, back view, popcorn and glasses props.
- **Rig (`storybook.py`):**
  - Hierarchy: hips → torso (lean) → head group (tilt / nod) and arms (shoulder → elbow → wrist).
  - Joint limits.
  - 13 poses: idle, listen, explain, open_hands, point, think, wave, surprise, celebrate, duck, hold, show, nod.
  - 12 expressions: neutral, listening, smile, happy, curious, confused, surprised, excited, thinking, concerned, laughing, wink.
  - Blinking on a per-character schedule; irises move inside the eye whites (gaze); a ¾ turn by shifting the features toward the other person.
  - Small non-repeating head motion and breathing; listener nods; a real wave.
  - Hands move only when a gesture is scheduled.
- **Status:**
  - **Not approved.** The brief asks for AI-generated, human-approved artwork; that needs a GPU (§5).
  - v1 exists so that everything else can be built and tested now.
  - QC reports "character artwork approved by a person" as **UNKNOWN**, so no video made with v1 is approved.

## 5. Producing approved artwork: Character Studio (Colab T4)

`colab_character_studio.ipynb` (built by `colab/build_studio.py`, using `engine/lib/studio.py`):

1. **Preflight:** GPU name, VRAM, compute capability, native bf16 (never assumed on a T4), RAM, licence check (strict).
2. **Concepts:** FLUX.2 [klein] 4B, Apache-2.0, pinned revision `e7b7dc27…`.
   - Transformer fp16 on the GPU, text encoder in 4-bit, model offload.
   - N fixed seeds per character, from the Bible text plus the storybook style prompt.
   - Contact sheet. Already-made images are skipped on re-run.
3. **You pick** a candidate (`STUDIO_PICK`). Nothing is locked automatically.
4. **Reference-conditioned edits:** expressions and back view from the chosen sheet. Then the model is released.
5. **Cut-outs:** BiRefNet (MIT, pinned), plus `rig_template.png`.
   - The layer split is **semi-automatic**, and done once by hand in any editor.
   - Parts hidden behind other parts must be painted in. AI segmentation is never trusted blindly.
6. **Import:** `studio.import_layers()` writes a new provisional version with provenance (model, revision, seeds, prompts, pick).
7. **Preview, then approve:** `python3 colab/approve_character.py <root> arjun v2 --by "<name>"`. The plan then pins its hash.

**Not run here.** This environment has no GPU, so the Studio's real run on a T4 is its first benchmark.

**Memory:** the 4B model needs ~13 GB VRAM in bf16 according to its model card; the plan is fp16 + 4-bit text encoder + offload. No benchmark numbers are claimed.

## 6. Decisions after research

| Question | Decision | Evidence |
|---|---|---|
| Character model | FLUX.2 [klein] 4B (Apache-2.0); **9B refused** (FLUX non-commercial licence) | HF API licence tags, pinned revisions in `registry.py` |
| Background model | Z-Image-Turbo (Apache-2.0, already registered) for future packs; v1 packs are code-drawn | registry |
| Cut-out model | BiRefNet (MIT). **RMBG-2.0 refused** (CC BY-NC 4.0), **RMBG-1.4 refused** (needs a Bria agreement) | HF API card data |
| Animation backend | Python cutout rig (Pillow), not Godot | Godot's Skeleton2D / AnimationPlayer would work, but rendering frames needs a display + OpenGL (Xvfb + software GL) in Colab, plus a second toolchain. The brief allows the Python fallback with layered art. Godot was not tested in Colab (no access). |
| Lip sync | Built-in audio visemes (language-independent); Rhubarb optional | Rhubarb 1.14.0 (MIT) ran locally. Open/closed agreement with our track: 87.7% (English, PocketSphinx), 81.2% (Telugu, phonetic), 89.4% (Telugu audio through the English recogniser). Test speech was espeak-ng (robotic). |

## 7. Environments (v1 = provisional, code-drawn)

| Pack | Sets | Camera presets | Used by worlds |
|---|---|---|---|
| cinema | seats (facing the audience: back wall, 2 seat rows, the friends' row, screen light, foreground row) · screen (curtains, live movie, audience heads, foreground) | seats_two_shot, seats_wide, seats_closeup_{arjun,meera}, screen_back | cinema_seats, cinema_screen |
| lobby | main (posters, marquee bulbs, floor, foreground popcorn stand) | lobby_two_shot, lobby_wide, lobby_closeup_* | lobby |
| studio | main | studio_two_shot, studio_closeup_* | studio, living_room |
| space_observatory | main (stars, galaxy, window frame, hologram, console) | obs_two_shot, obs_wide | space_observatory |

- **Lighting:** each set has a light colour and strength applied to the characters, so they belong to the scene (cool screen light in the cinema, warm light in the lobby). Seated presets hide the legs behind the front row; standing presets add a soft contact shadow.
- **Camera:** static by default. `slow_push_in` (5% over the shot) / `slow_pull_back` / `pan_left` / `pan_right` only when the visual asks for it (`camera.movement`, or the director's push on close-ups). The foreground layer moves faster than the rest (parallax).
- **Re-use:** the same cinema serves every cinema shot of a video, and every cinema video.

## 8. Plan schema extensions (backward compatible)

| Field | Where | Meaning |
|---|---|---|
| `visual_style_version` | plan | `"storybook_v1"` turns the new renderer on; absent = legacy |
| `character_assets` | plan | `{id: {version, status, art_version, manifest_sha256}}` |
| `environments` | plan | `{env: version}` |
| `camera` | visual (story JSON) / character spec | `{"movement": ...}` |
| `environment_variant` | visual / spec | an explicit camera preset |
| `_visemes` | spec (runtime) | mouth shape per frame, from the speaker's audio |

Existing fields are reused, not duplicated:
- `world`, `shot`, `subject`, `chars` (expr, gesture, wear, hold), `changes`, `screen`, `push`, `label`.

Unknown characters, gestures, worlds, movements and missing versions give clear errors at plan-build time (`build_story_plan.py --visual`) and in the notebook's plan check.

## 9. How to …

- **Regenerate provisional assets:** delete nothing. Build a new version: `storybook.build_character(root, "arjun", "v1b")`, and rebuild the plan with `--char-version v1b`.
- **Approve and reuse characters:**
  1. Character Studio.
  2. `approve_character.py`.
  3. `build_story_plan.py ... --visual storybook_v1 --char-version v2`.
  - Every later video pins the same hash; nothing is regenerated.
- **Add a background:**
  - Add a pack to `envfactory.PACKS`: a draw function or generated layers, sets with layer depths, presets with crops and slots, and a world map.
  - Map the new world in `WORLD_ENV`.
  - `build_environment(root, env, "v1")`.
  - Generated packs must keep each depth layer separate.
- **Add a pose:** add joint angles to `storybook.POSES` (respecting `JOINT_LIMITS`), then check it on the pose sheet (`storybook_previews.py --only A`). A pose that needs new artwork (a new hand shape) needs a new asset version.

## 10. Measurements (actual, this machine: 12-core CPU, no GPU, 7.4 GB RAM — NOT a Colab benchmark)

| What | Measured |
|---|---|
| Building one character version (75 layers, 1.6 px/unit, 2× supersampled) | 342 s (Arjun), 353 s (Meera), one-time |
| Building the 4 environment packs | ~18 s |
| Size of the asset library | 2.9 MB characters + 1.7 MB environments; embedded notebook 5.8 MB |
| Storybook frame, 720p, steady state (two characters, push-in) | 0.11–0.12 s (first frames ~0.18 s while caches fill) |
| Storybook frame, 1080p | ~0.17 s |
| Full 3D-glasses video (2,581 frames at 720p, 6 scenes) | about 4.1 min of drawing on one core |
| Full notebook end to end (draft, espeak stand-in voices, no speech recogniser) | completed with 0 tracebacks |
| QC on the full video | 28/31 PASS; UNKNOWN: speech check (recogniser off for the stand-in), person review, artwork approval (provisional) |
| Resume on the same Drive folder | assets reused (0 re-installed), voices reused; drawn scenes redrawn only because their mouth data changed |
| Rhubarb vs built-in mouth timing (open/closed agreement) | 87.7% English · 81.2% Telugu (phonetic) · 89.4% Telugu (PocketSphinx) on espeak speech |

**Not measured:** anything on a Colab T4. That includes the Studio (FLUX.2 klein 4B / BiRefNet) memory and times, and Godot.

## 11. Cinematic characters: the pose atlas (`atlas_v1`) (appended 2026-10-20)

Report: `reports/character_poc/REPORT.md`.

- **The art:** v1's procedural drawing is replaced, for new versions, by FLUX.2 [klein] 4B artwork (Apache-2.0). It is generated once in the Character Studio and approved by a person.
- **Why not the rig:** a test on the real art showed that rotating cut-out limbs breaks it (`reports/character_poc/approach_A_cutout_arm_test.png`).
- **The format:** `asset_type: "atlas_v1"`.
  - whole pose images edited from the approved master;
  - region-scoped face patches: blink and glasses in the eye band, 6 mouth shapes in the mouth band, 11 expressions on the whole face;
  - an explicit `pose_map` for all 13 storybook poses;
  - every file sha256-verified, plus a `content_sha256` that approval never changes.
- **Rendering:** `storybook_scene` picks the atlas branch from the manifest. The plan schema, `_visemes`, blink schedule and 0.45 s transitions are unchanged.
  - Characters are placed by their head, so the camera presets frame them like v1.
  - Composition rules: glasses are a face patch on front poses and invisible from behind; a held prop uses the gesture's prop image, else the prop's hold pose; from behind every gesture uses the back image; anything else is a `check_spec` error.
- **Studio:** `studio_run.py` (hosted Space or local GPU; resumable; provenance sidecars; VRAM/RAM peaks) and `colab_character_studio.ipynb`:
  1. proof of concept;
  2. the `POC_APPROVED` gate;
  3. 35 atlas jobs;
  4. BiRefNet cut-outs;
  5. review sheets;
  6. import;
  7. previews;
  8. `approve_character.py` (refuses with its blockers).
- **Environments:** `envfactory.import_plates` makes a v2 pack from generated plates and keeps v1's camera presets, slots and safe regions.

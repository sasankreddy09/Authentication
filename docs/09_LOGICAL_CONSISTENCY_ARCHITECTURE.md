# Logical & Spatial Consistency Architecture for the AI Video Pipeline

*2026-10-16. Audit of `colab/colab_amazon_refund_v1.ipynb` (built from `colab/engine/`), research, redesign, and the
implementation now in the project. `ai_video_automation_research.md` is superseded; the current design documents are
`docs/08_COLAB_VIDEO_PIPELINE_GUIDE.md` (pipeline) and this file (consistency).*

**Honest scope.** Nothing here guarantees consistent video. The goal is: plan what must happen, constrain how it is
generated, check what actually happened, repair or reject what is wrong, and **never let a failed or unchecked shot
reach the final video silently**. Parts verified on this (GPU-less) machine are marked *tested*; GPU behaviour that
could not be run here is marked *unverified*.

---

## 1. How the existing workflow actually works

```
plan JSON ──► plan cell: content rules (validate.py: hooks, claims, jargon, entities) + continuity (cast, A-B-A, one-clip length)
          ──► narration (Kokoro + Whisper check) ──► timeline (shot lengths from the voice)
          ──► drawn scenes (ui / journey / card / diagram) in the background
          ──► keyframes (only for image-to-video on L4+; skipped on the T4)
          ──► clips: prompt = scene + character looks + action text + style
                     T4: Wan2.1-T2V-1.3B text-to-video (turbo LoRA, 4 steps); L4+: Wan2.2-5B image-to-video from a keyframe
                     after each clip: CLIP presence (characters/props in 8 frames) + CLIP scene relevance
                     on doubt: 1 remake with seed+1000; keep the take with the best CLIP score; flag
          ──► editing (stretch ≤ 1.4×, trim to the shot length, hold last frame) ──► music, SFX, mix ──► QC ──► metadata
```

Scene-to-scene logic existed only as **text**: `action` strings ("walks up to the return counter and places the white
earbuds case on it, smiling"), a `location` string per shot, `transition` flags, and a `prop_state` that only said
*where* a prop is (home → store → returns centre).

## 2. Flaw report (grounded in the code)

Locations refer to `colab/engine/` as uploaded (the notebook is built from these files).

| # | Location | Root cause | Example failure | Severity | Fix (status) | Confirmed? |
|---|---|---|---|---|---|---|
| F1 | `amazon_refund_plan.json` sh06/sh07 `action`; `cells/cast/plan.py` prompt build (`image_prompt = scene + looks`, `motion_prompt = action`) | No spatial representation: no start/end positions, facing, camera side, ownership | "walks up to the counter" → the model picks the direction; she can enter from the right | Critical | Typed scene-state per shot (`engine/lib/scene_state.py`) (**implemented, tested**) | Confirmed |
| F2 | `cells/shared/prefetch.py` `video_mode()` returns `"t2v"` for `wan21-1.3b`; `cells/shared/keyframes.py` skips keyframes in t2v | The free-tier path is pure text-to-video with no image, layout or trajectory conditioning | Each shot invents its own geography; nothing can enforce left/right | Critical | VACE layout-controlled generation from the scene-state (`cells/shared/logic.py`, `load_video_model("vace")`) (**implemented; tested on a reduced model; unverified on GPU**) | Confirmed |
| F3 | Amazon sh06 "walks up … places … smiling" in one ~5 s clip; sh07 "scans … and talks" | Several actions in one short clip | Models drop or merge actions: she arrives but never puts the case down | Critical | Rule E10 (≤ 1 movement + 1 hand action per AI clip); sh06 split into sh06 (enter, walk, stop) + sh06b (place) (**implemented, tested**) | Confirmed |
| F4 | `cells/shared/clips.py` `presence()` / `relevance()` | Only appearance is checked (CLIP text–image similarity on 8 frames): no tracking, direction, order, completion | A clip where she walks right-to-left, or never places the case, passes | Critical | Geometric checks with detection + camera compensation (`motion_check.py`) + VLM event checks (`vlm_check.py`) (**implemented; geometry tested on synthetic and real-photo videos; VLM unverified**) | Confirmed |
| F5 | `clips.py` loop: "kept the best take; flagged for review"; `use_fallback` only when the plan has a `fallback` (Amazon had none) | A failed check still puts the clip in the final video | The user's run: sh02/sh06/sh09 failed and were rendered anyway | Critical | Acceptance gate: only ACCEPTED or person-approved clips are used; otherwise the drawn fallback (`acceptance.usable`, `editlib.segment_recipe`) (**implemented, tested**) | Confirmed (user's log) |
| F6 | `clips.py` `make_clip(..., part + 1000 * attempt, ...)`; `best` chosen by `continuity_score` | Retry = new seed only; best = highest CLIP score | A wrong-direction clip is "remade" with the same uncontrolled prompt, and the higher-scoring invalid take wins | High | Failure → targeted repair table (`acceptance.REPAIRS`); no candidate is used because it scored highest (**implemented, tested**) | Confirmed |
| F7 | `clips.py` `clip_key()` = prompts + seed (+ keyframe prompt) | Generation cache key misses model, recipe, steps/LoRA, frame count, size, style, negative prompt, control signals | Switching `VIDEO_SPEED` reuses turbo clips; a longer narration reuses a too-short clip (then holds the last frame) | High | `acceptance.generation_key()` covers all of them (**implemented, tested**) for scene-state shots; legacy shots keep the old key | Confirmed |
| F8 | Sidecar `continuity` stored with the clip; no validator version | Validation results never re-checked when rules change | After the props rule changed, old "missing earbuds" flags and old passes stayed as they were | High | Separate `validation_key` (constraints + validator versions + thresholds); mismatch → re-validate the cached clip without regenerating (`logic_recheck`) (**implemented, tested**) | Confirmed |
| F9 | `clips.py`: `except: log_exc(...); tags = []` | A crashing checker is switched off for the rest of the run | Every later shot passes unchecked | High | `motion_check.run_checks` turns exceptions into UNKNOWN results; UNKNOWN → REVIEW (**implemented, tested**) | Confirmed |
| F10 | `editlib.py` `segment_recipe` / `render_segment` (`-frames:v row["frames"]`, stretch ≤ `MAX_SLOWMO`, `tpad` hold) | Trimming knows nothing about actions | An action finishing in the last 0.5 s is cut off | Medium | Plan rule E11 (actions end before `1 − tail`) + clip check `completes_before_cut` using the edit's real stretch and kept length (**implemented, tested**) | Confirmed (mechanism) |
| F11 | `validate.py` V1/V3 | Continuity by `location` string; props by place only, not holder | Case "in her hand" in the next shot after she put it down passes | Medium | Ownership simulation X2, hand-off preconditions E7/E9, declared `moved_offscreen` (**implemented, tested**) | Confirmed |
| F12 | Character `tag` used by CLIP ("a young woman in a teal hoodie") | Generic text identity; no reference image | Any woman in a teal top passes; identity drift undetected | Medium | Partly: OWLv2 query per entity + presence; real identity needs reference images (VACE `reference_images`, L4+) (**planned**) | Confirmed |
| F13 | `PRESENCE_FRAME_P`, `RELEVANCE_MIN` | Uncalibrated thresholds; tiny objects at 224 px | User's run: earbuds "missing" in every shot; close-ups failed "scene" | Medium | Small objects → UNKNOWN + VLM; scene check advisory for legacy shots (**implemented**) | Confirmed (user's log) |
| F14 | `clips.py` image-to-video multi-part: `img = Image.fromarray(arr[-1])` | The next part continues from an unchecked last frame | An error in part 1 propagates into part 2 | Medium | Scene-state shots are one clip; multi-part only for legacy L4 i2v (**kept, documented**) | Confirmed |
| F15 | `clips.py` `distractors()` excludes same-location shots | Relevance can't tell sh06 from sh07 (same counter) | The wrong action at the right place passes | Low | Replaced by event checks for scene-state shots | Confirmed |
| F16 | GPU log: "GPU 11.6/15.0 GB" during the clip loop (earlier runs ~3.5 GB) | Unknown: allocator cache, CLIP, or something not freed | Less headroom for VAE decode and checkers → possible OOM | Medium | Measure with the run log on the next GPU run; checkers loaded after generation only when needed | **Potential** |
| F17 | `editlib` camera moves (`move` push/pan) for fallback stills | Fallback camera direction unrelated to subject direction | A pan right while the story says she moves left | Low | Fallbacks are now drawn cards; stills keep slow push/pull | Potential |
| F18 | Storyboard written after the timeline, not used for checking | No narrative contract the clip is judged against | — | Medium | The scene-state IS the contract (same data steers generation and checks) (**implemented**) | Confirmed |

## 3. Research: techniques compared

**Verified** (official repos/docs, model cards, or the installed library source):

| Technology | What it gives | Hardware / size | Licence | Fit |
|---|---|---|---|---|
| **Wan2.1-VACE-1.3B** (`Wan-AI/Wan2.1-VACE-1.3B-diffusers`) | One model for reference-to-video, control-video-to-video and masked video-to-video. VACE's own task list includes **`layout_bbox`, `layout_track`** (moving boxes), `firstframe`, `lastframe`, `firstlastframe`, depth, pose, flow, scribble, inpainting, outpainting ([VACE repo](https://github.com/ali-vilab/VACE), `vace/configs`, `vace/annotators/layout.py`). Layout control = white video with a coloured 2-px box outline per subject, linearly moving from a start to an end box. Same 30-layer Wan2.1-1.3B + 15 VACE blocks; **same UMT5 text encoder and VAE as T2V-1.3B** (model_index/config). | transformer 7.15 GB fp32 ≈ 3.6 GB fp16 → fits a T4 | Apache-2.0 | **Chosen** for direction/position control on the free tier |
| diffusers 0.41 `WanVACEPipeline(video, mask, reference_images, conditioning_scale, …)` | Runs VACE; its docstring example does first/last-frame through the mask | — | Apache-2.0 | Used (*tested on a reduced model: layout call works, turbo LoRA fuses*) |
| `WanImageToVideoPipeline(image, last_image, …)` | First/last-frame generation (Wan2.1-FLF2V-14B, 90 GB) | A100-class | Apache-2.0 | Later, for exact start/end compositions on big GPUs |
| `LTXConditionPipeline(conditions, frame_index, …)` | Keyframes at chosen frames | 2B–13B | Custom LTXV licence: paid licence required at ≥ $10M annual revenue ([licence text](https://scancode-licensedb.aboutcode.org/ltxv-owl-2025-05-05.html)) | Alternative; licence restricts |
| Wan2.2 TI2V-5B / I2V-A14B | Better image-to-video quality | 5B: L4; A14B: A100 | Apache-2.0 | Keep for unconstrained shots on L4+ |
| ATI (ByteDance) | Trajectory control for Wan2.1 | 14B, 480p ([ComfyUI wiki](https://comfyui-wiki.com/en/models/wan/ati)) | not confirmed in search | Too big for T4; check licence before use |
| **OWLv2** (`google/owlv2-base-patch16-ensemble`) | Open-vocabulary detection by text query | ~0.6 GB | Apache-2.0 | **Chosen** detector (*tested on CPU with a real photo*) |
| **SAM 2.1** (`facebook/sam2.1-hiera-tiny/small`, `Sam2VideoModel` in transformers 5.19) | Mask tracking through a video from a box prompt | small | Apache-2.0 | Next step: replace frame-wise detection + linking for hands/objects |
| Grounding DINO tiny | Open-vocabulary detection | small | Apache-2.0 | Alternative to OWLv2 |
| CoTracker3 | Point tracking | — | **CC-BY-NC** (non-commercial) | **Excluded** (monetised channel) |
| **Qwen3-VL-2B/4B-Instruct** | Video/image QA; the technical report describes explicit text timestamps for temporal grounding ([arXiv 2511.21631](https://arxiv.org/pdf/2511.21631)) | 2B ≈ 4.3 GB fp16 | Apache-2.0 | **Chosen** event judge (*unverified on GPU*) |
| Qwen2.5-VL-7B | Same role, bigger | ~16 GB | Apache-2.0 (3B: no licence tag) | L4+ alternative |
| Phase correlation (numpy FFT) | Global camera translation between frames | CPU, ms | — | **Chosen** camera-motion estimate (*tested: 0.35 pan measured within 0.06*) |
| FFmpeg | Deterministic editing, mixing, QC | CPU | LGPL/GPL | Kept |

**Assumptions / engineering judgement** (not verified here): turbo LoRA + VACE quality on real weights; how strongly
VACE-1.3B follows layout boxes for people; OWLv2 accuracy on stylised 3D-animated frames; Qwen3-VL-2B reliability on
4–6 s clips; VBench-style automatic quality scores were not adopted (they measure smoothness/aesthetics, not story logic).

## 4. Recommended architecture

```
 1 Topic & research ─► claims with sources (validate P4–P6)
 2 Script & narrative plan ─► one protagonist, one idea per shot (S rules)
 3 References: characters/props/locations ─► scene_entities (query, small), looks, props paths
 4 Scene-state & continuity plan ─► per AI shot: camera(setup, axis, side, move), start boxes/facing, holders,
                                     ordered events (t, to_box), tail, transition, moved_offscreen
 5 Structured storyboard ─► render/storyboard.md (+ composition guides for review)
 6 Deterministic plan validation ─► validate.py + scene_state.check_sequence  ── errors ─► STOP (nothing generated)
 7 Narration & timing ─► Kokoro + Whisper; shot length fixes frame count
 8 Spatial keyframes ─► layout_control.expected_tracks ─► control video (+ first-frame keyframes on L4+, planned)
 9 Capability routing ─► scene-state shot → VACE layout | atmospheric → T2V | screens/numbers/maps → drawn
10 Controlled generation ─► WanVACEPipeline(video=layout, mask, conditioning_scale)
11 Validation ─► OWLv2 tracks + camera motion ─► motion_check (positions, direction, stationary, teleport, stop,
                  transfer, support, order, completes-before-cut) + Qwen3-VL events ─► PASS / FAIL / UNKNOWN + evidence
12 Diagnosis & targeted regeneration ─► acceptance.decide: ACCEPT | REPAIR(repair) | REVIEW   (budget per shot)
13 Shot-to-shot continuity ─► plan simulation (X1–X5) + per-clip checks of the planned start/end states
14 Deterministic edit & audio ─► FFmpeg; only ACCEPTED/approved clips, else the drawn fallback
15 Final QC ─► length, loudness, black/frozen frames, speech, licences, "every scene-state shot verified or approved"
16 Approval & publishing ─► approvals.json (per generation) ─► metadata ─► manual upload (Private)
```

Everything runs from one fixed executor driven by the JSON plan (no generated code). Colab notebook = thin cells;
logic in `colab/engine/lib/*.py` (unit-tested). Production later: the same modules behind a job queue with SQLite
state; no durable orchestrator until several videos run in parallel.

## 5. Scene-state design

Pydantic v2 models in `engine/lib/scene_state.py`: `Camera`, `Placement`, `Event`, `ShotState`, `Entity`, `World`.
Normalised screen coordinates (x: 0 = left edge … 1 = right edge); event times are fractions of the shot.

Amazon **sh06** (enter → walk → stop) and **sh06b** (place), as now in `colab/amazon_refund_plan.json`:

```jsonc
"scene_entities": {
  "maya":    {"kind": "character", "name": "Maya", "query": "a young woman in a teal hoodie"},
  "counter": {"kind": "anchor", "name": "the return counter", "query": "a shop counter"},
  "earbuds": {"kind": "prop", "name": "the white earbuds case", "query": "a small white case", "small": true}
},
{"shot_id": "sh06", "state": {
   "location": "store",
   "camera": {"setup": "store_wide", "axis": "store_line", "side": "front", "size": "wide", "move": "static"},
   "start": {"maya": {"box": [-0.04, 0.22, 0.14, 0.98], "facing": "right"},      // starts at the LEFT edge
             "counter": {"box": [0.62, 0.45, 1.0, 1.0]}},                        // counter stays screen-right
   "holders": {"earbuds": "maya", "phone": "maya"},
   "events": [
     {"id": "walk", "type": "enter", "actor": "maya", "target": "counter", "t": [0.05, 0.68],
      "to_box": [0.42, 0.22, 0.60, 0.98], "describe": "she walks from left to right"},
     {"id": "stop", "type": "stop", "actor": "maya", "t": [0.68, 0.76]}]}},     // ends with 24 % tail before the cut
{"shot_id": "sh06b", "state": {
   "location": "store",
   "camera": {"setup": "store_medium", "axis": "store_line", "side": "front", "size": "medium", "move": "static"},
   "start": {"maya": {"box": [0.14, 0.12, 0.50, 1.0], "facing": "right"}, "counter": {"box": [0.55, 0.50, 1.0, 1.0]}},
   "holders": {"earbuds": "maya", "phone": "maya"},
   "events": [{"id": "place", "type": "place", "actor": "maya", "obj": "earbuds", "target": "counter",
               "t": [0.15, 0.6], "to_box": [0.62, 0.42, 0.72, 0.50]}]}}
```

Simulation (`simulate` / `check_sequence`) carries a `World` (positions per camera axis/side, facing, holders,
travel direction) from shot to shot. Rules: X0 unknown entity · **X1 crossing the action line** without
`axis_change` · **X2 object changes holder without an event** (or declared `moved_offscreen` + reason, e.g. sh09
"shipped to the returns centre") · **X3 two subjects swap left/right** · **X4 travel direction reverses across a cut**
· E1 actor not present · E2 movement without `to_box` · E3 `enter` not from an edge · **E4 walking backwards** ·
E5 hand action without object · **E6 acting while still walking (no stop)** · **E7 giving/placing what you don't
hold** · **E8/E9 target absent / not within reach** · **E10 > 1 movement or > 1 hand action per clip** · **E11
action ends too close to the cut** · X5 AI shot with required actions but no drawn fallback (warning). Schema
validation also rejects events listed out of time order.

## 6. Spatial and temporal consistency strategy

- **Plan first**: positions, facing, camera side and ordered small events are data, simulated before any GPU work.
- **Same data steers and judges**: `layout_control.expected_tracks()` gives every subject's box per frame (people
  move with eased in/out timing; held objects ride in the hand on the side the person faces; a `place` moves the
  object from hand to its target box during the event). That drives (a) VACE's layout video and (b) the expected
  tracks the clip is checked against.
- **Camera vs subject motion**: the plan states the camera move; the checker measures camera translation by phase
  correlation with subjects masked out, adds it back to the subject's screen motion ("world motion"), and fails a
  "static" shot whose camera actually moved (so a pan can't fake a walk). Zoom-type moves are kept small (push/pull).
- **180° rule**: the camera side per action line is part of the state; crossing it requires `axis_change` with a
  reason (intentional reverse angles are allowed; accidental ones are rejected).
- **No silent flips**: footage is never mirrored to fix direction (it would swap every other spatial fact).
- **Short actions**: ≤ 1 movement + 1 hand action per AI clip; a stop between walking and placing; each action
  completes with ≥ 15 % of the shot left, and the clip check confirms it finishes before the edit's cut.

## 7. Validation architecture

Every check returns `PASS` / `FAIL` / `UNKNOWN` with evidence (`motion_check.CheckResult`).

| Layer | Checks | Method | Status |
|---|---|---|---|
| Technical | duration, loudness, black frames, frozen last frame, NaN/black clip, colour self-test, speech accuracy | FFmpeg / numpy / Whisper | existing |
| Spatial | presence, start position, end position, direction (camera-compensated), stationary, camera actually static, teleport jump | OWLv2 per sampled frame → continuity linking → `motion_check` | implemented; *synthetic + real-photo tested* |
| Temporal | stop before acting, event order, completes before the cut | tracks + VLM timestamps | implemented; *synthetic tested* |
| Interaction | object transfer (from giver to the PLANNED spot, leaving the hand), object only moves while carried | tracks | implemented; *synthetic tested* |
| Semantic events | "Does Maya put the case down on the counter?" per critical hand action/enter/exit, strict JSON with time + evidence | Qwen3-VL-2B | implemented; *parser tested; model unverified* |
| Identity | presence by the character's own query; full identity needs references | OWLv2 (now), reference embeddings (planned) | partial |
| Narrative | content rules + scene-state simulation (pre-generation) | validate.py, scene_state.py | implemented; *tested* |

Combination rules: geometry is primary for where/which way; the VLM confirms what boxes can't see (small objects,
scans). Geometry FAIL → FAIL. Geometry PASS + VLM FAIL → UNKNOWN (disagreement). Geometry UNKNOWN + VLM PASS with a
time and evidence → PASS (and the VLM time also serves as the completion time). Unsure/malformed VLM answers →
UNKNOWN. A crashing validator → UNKNOWN (`run_checks`). Small objects never produce FAIL from a missed detection,
only UNKNOWN. **UNKNOWN on a critical check never passes**: the shot goes to REVIEW.

## 8. Generation and retry strategy

| Shot type | Route |
|---|---|
| People/objects with required positions or actions (has `state`) | VACE layout control (free T4 and up) |
| Atmospheric, no required action | text-to-video (T4) / image-to-video (L4+) |
| Screens, numbers, dates, money flows, recaps | drawn scenes (`ui`, `journey`, `card`) — exact by construction |
| Exact start/end composition (planned) | first/last-frame (VACE masks; FLF2V-14B on A100) |
| Identity-critical recurring characters (planned) | VACE `reference_images` with prepared character sheets (L4+) |

Targeted repairs (`acceptance.REPAIRS`), applied in order within `REPAIR_BUDGET` (default 2):

| Failure | Repairs |
|---|---|
| direction / end position | layout_strength (conditioning +0.25) → layout_simplify (only the moving subject and its target) → fallback |
| start position | layout_strength → first_frame (L4+) → fallback |
| stationary / camera moved | layout_strength / static_camera (locked-off wording) → fallback |
| teleport | reseed → fewer_frames → fallback |
| missing person | reference_identity (L4+) → reseed → fallback |
| stop / transfer / order / missing action | split_action (plan change → REVIEW) → layout_strength → fallback |
| action cut off | earlier_action (plan change → REVIEW) → fallback |

The final decision uses the last/least-failing candidate only for the review pictures; it is **used** only if
ACCEPTED or approved (`approvals.json`: `{"sh06": {"approved": true, "gen_key": "<from the sidecar>", "note": "..."}}`,
valid only for that exact generation).

## 9. Code-level implementation

| Module | Role | Status |
|---|---|---|
| `engine/lib/scene_state.py` | Pydantic schema, simulation, continuity rules, prompt hints | implemented, tested |
| `engine/lib/layout_control.py` | expected tracks, VACE layout control video, composition guide | implemented, tested |
| `engine/lib/motion_check.py` | camera motion, geometric/temporal checks, `CheckResult`, `run_checks` | implemented, tested |
| `engine/lib/trackers.py` | `ColorTracker` (tests), `Owlv2Tracker` (GPU) | implemented; OWLv2 tested on CPU with a real photo |
| `engine/lib/vlm_check.py` | event questions, strict parsing, `QwenJudge` | implemented; judge unverified |
| `engine/lib/acceptance.py` | decide, repair table, generation/validation keys, `usable` | implemented, tested |
| `engine/cells/shared/logic.py` | glue: VACE generation, checks, repairs, re-validation, review assets | implemented; integration-tested with stubs |
| `cells/shared/clips.py` | routes scene-state shots to the logic loop; VACE loading via the Drive cache | implemented |
| `cells/shared/editlib.py`, `qc.py` | fallback + review for unaccepted shots; per-shot logic report | implemented |
| `cells/cast/plan.py` | parses and simulates scene-states; stops on errors | implemented |

Settings (cell 1): `VIDEO_RECIPE = "auto"`, `LOGIC_CHECK = True`, `VLM_CHECK = True`, `REPAIR_BUDGET = 2`.
New dependencies: pydantic ≥ 2.7 and scipy (both preinstalled on Colab; the install cell ensures them).
Migration: plans without `state` blocks behave as before (reported as "not logically verified"); adding `state`
to a shot switches it to the new route. Existing Amazon clips are regenerated (their generation key changed).

## 10. Updated implementation (what changed in the project)

- New modules above; `build.py` adds a `logic` cell before `clips`.
- `amazon_refund_plan.json` v4: `scene_entities`; `state` for all 7 AI shots; sh06 split into sh06 + sh06b; sh07 with
  clerk behind the counter on screen-right and Maya on screen-left; sh09 with the declared off-screen move; drawn
  fallback cards for every AI shot. Plan validation: 0 errors, 0 warnings; scene-state: 7 shots, no issues.
- Notebook rebuilt: `colab/colab_amazon_refund_v1.ipynb`.

## 11. Testing framework and acceptance criteria

| Suite | What | Result |
|---|---|---|
| `tests/test_consistency.py` (62 checks) | synthetic videos: left→right PASS, right→left FAIL, camera pan faking motion, stationary, intentional vs accidental axis change, missing action, wrong order, object teleport/jump, object moving by itself, broken hand-off, placing from afar / while walking / without holding, too many actions, action at the cut, trim cutting the action, ownership across shots and declared off-screen moves, reversal across a cut, walking backwards, identity drift, no evidence → UNKNOWN, small object → UNKNOWN, validator exception → UNKNOWN, acceptance decisions and budgets, cache invalidation (generation vs validation keys, approvals per generation), layout video, VLM answer parsing, JSON-safe evidence | ALL PASSED |
| `tests/test_logic_loop.py` (15 checks) | the notebook's real logic/clips code with stub generator/tracker/VLM: correct clips ACCEPTED without repairs; mirrored clip → direction FAIL → layout_strength → layout_simplify → REVIEW, not usable, review pictures saved, usable only after approval of that generation | ALL PASSED |
| `tests/test_lib.py` (31 checks) | content validation, drawn scenes, music | ALL PASSED |
| Real-detector check | OWLv2 on a real COCO photo moved across a frame: detected 8/8; left→right PASS, right→left FAIL | passed |
| End-to-end CPU runs | Amazon (Python 3.13 + FFmpeg 4.4): scene-state shots without a GPU use fallbacks and are listed; Zomato and black holes unchanged | passed |

**Acceptance targets for the GPU stage** (to measure, not claimed): build a labelled set of ≥ 50 generated clips
(direction correct/wrong, action done/missing, object transfer right/wrong), labelled by a person from the review
pictures.
- Wrong-direction clips detected (FAIL or UNKNOWN): ≥ 95 %; correct clips falsely failed: ≤ 15 %.
- Missing critical actions detected (FAIL/UNKNOWN): ≥ 90 %.
- Shots with a critical FAIL/UNKNOWN reaching `final.mp4` without approval: **0 %** (enforced by construction,
  covered by tests).
- Clips accepted after ≤ 2 repairs: tracked per video (target ≥ 60 % on T4; review the rest).

## 12. Performance and cost (free T4)

| Item | Cost | Notes |
|---|---|---|
| VACE-1.3B download | 7.15 GB once, cached on Drive at fp16 (~3.6 GB) | replaces the T2V-1.3B cache (3.3 GB) for scene-state videos |
| VACE turbo generation | similar to T2V turbo plus the 15 VACE blocks (estimate +20–40 %, *unverified*) | ~3–5 min per clip on T4 (estimate) |
| OWLv2 checks | ~0.6 GB, 12 frames per clip, seconds | session disk, not Drive |
| Qwen3-VL-2B | ~4.3 GB download per session; ~10–30 s per event question (estimate) | session disk; turn off with `VLM_CHECK = False` (those checks become UNKNOWN → review) |
| Repairs | ≤ 2 extra generations per failing shot | GPU minutes per shot are logged |
| VRAM | VACE ~3.6 GB + OWLv2 ~0.3 GB + Qwen3-VL-2B ~4.3 GB + activations | fits 15 GB; F16's 11.6 GB observation must be measured on the next run |
| Free Drive (15 GB) | Wan T2V 3.3 + VACE 3.6 + voice/Whisper/CLIP ~1.5 GB | checkers kept off Drive on purpose |

## 13. Remaining limitations (need a person)

- Identity across shots (same face/clothes) is only weakly checked until reference-image conditioning (L4+) and an
  identity embedding check are added.
- Small hand actions (scans, opening a case) rely on the VLM; its answers are cross-checked but not proof.
- Physical plausibility beyond position/order (grip, contact, gravity) is not measured.
- Detector accuracy on stylised 3D frames and VACE's adherence to boxes must be calibrated on real runs; thresholds
  in `motion_check.TH` will change (changing them re-validates cached clips automatically).
- Facial expressions, emotion and lip-sync are not checked.
- Plans can still describe something plausible but wrong in the real world (factual accuracy rests on the claims).

Sources: [VACE repo](https://github.com/ali-vilab/VACE) ·
[Wan2.1-VACE-1.3B](https://huggingface.co/wan-ai/wan2.1-vace-1.3b) ·
[ATI (ComfyUI wiki)](https://comfyui-wiki.com/en/models/wan/ati) ·
[ATI (Comfy docs)](https://docs.comfy.org/ja/tutorials/video/wan/wan-ati) ·
[LTXV licence text](https://scancode-licensedb.aboutcode.org/ltxv-owl-2025-05-05.html) ·
[LTX-2.3 licence](https://huggingface.co/Lightricks/LTX-2.3/blob/main/LICENSE) ·
[Qwen3-VL technical report](https://arxiv.org/pdf/2511.21631) ·
[Qwen3-VL report summary (Unite.AI)](https://www.unite.ai/alibaba-releases-qwen3-vl-technical-report-detailing-two-hour-video-analysis/) ·
[AMD ROCm VACE blog](https://rocm.blogs.amd.com/artificial-intelligence/video-editing-models/README.html) ·
Hugging Face model cards checked through the Hub API: Wan-AI Wan2.1/2.2, facebook sam2.1, google owlv2,
IDEA-Research grounding-dino, Qwen2.5-VL/Qwen3-VL, facebook cotracker3.

## Addendum (2026-10-17): validator corrections found while profiling (details: guide section 36)

- **Walking events:** `enter`/`exit` are judged by the actor's start, end and direction checks (`event_motion`), as
  `walk_to` always was. Before, they were judged by the VLM alone (geometry was never linked), and a "no" caused a
  blind reseed. A failure now maps to `layout_strength` → `layout_simplify`.
- **Camera motion:**
  - Planned boxes of people, props and moving things are blanked together with the detections, so a missed subject
    no longer fakes a camera move.
  - Each pair of frames is masked identically.
  - Peaks are sub-pixel, with an anchor frame every 8 frames, so slow real pans are measured: 0.15 → 0.137, which
    was 0.000 before.
  - `motion_check.VERSION = 2`: cached clips are re-checked, not regenerated.
- **Performance table correction (§12):** "VACE turbo ~3–5 min per clip" was never reached on Colab, because turbo
  failed to load in every run so far. Measured fast-mode clips took 23.7–36.5 min. Turbo was verified compatible
  (identical base weights). Its T4 timing awaits `BENCHMARK = True`.

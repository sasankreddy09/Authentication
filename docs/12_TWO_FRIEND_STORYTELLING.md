# 12. Two-friend cinematic storytelling (2026-10-18)

The brief (`claude.md`, "Two-Friend Cinematic AI Video Generator") asks for explainer videos told as a conversation
between two recurring friends. The visuals should be demonstrations rather than flowcharts. Everything must be free,
openly licensed and run on Google Colab Free.

This document covers what was checked, what was built, how it was tested, what was measured, and what is still
unverified.

## 1. Validation of the brief (what was checked before building)

| Claim / requirement in the brief | Checked how | Result → decision |
|---|---|---|
| Existing `review` registry items must not be silently accepted; inspect the turbo LoRA | Hugging Face API + model pages | `Kijai/WanVideo_comfy` has **no licence tag**. It repackages `gdhe17/Self-Forcing` (Apache-2.0, commit `2f8b779…`). **Strict mode refuses it.** AI clips then use "fast" mode (~4x slower). A licence-clean turbo needs the upstream weights converted locally (Phase E, not done). |
| Restricted visual checkpoints | registry | SDXL (OpenRAIL++, use restrictions) and `openai/clip-vit-base-patch32` (no Hub licence tag; "deployed use out of scope") are both `review`. **Strict mode refuses them**: fallback pictures are drawn, and the CLIP continuity check is turned off (scene-state shots keep OWLv2 + VLM, both Apache-2.0). |
| `LICENSE_STRICT = True` for storytelling | — | Default `True` for two-friend plans and the topic notebook. Old plans keep `False`, so their behaviour is unchanged. |
| Pin model revisions | Hugging Face API, 2026-10-18 | The commit of every registry model is recorded in `registry.REVISIONS`. `test_registry.py --online` reports when the Hub moves. **Not yet passed as `revision=` at every load site.** |
| Rhubarb Lip Sync optional | GitHub | MIT, v1.14.0 binary. **Not bundled.** The mouth follows the speaker's loudness instead (see §4). |
| Commons media for the 3D topic | Commons API | Images exist (PD/CC0, e.g. polarised stereo glasses). **Commons videos for this topic are mostly CC BY-SA**, which strict media policy rejects by default. Licensed media is wired in for images, and the sample uses none. |
| Telugu via Indic Parler, gated, never substitute | code review | Telugu always routes to Parler (the language branch comes before any Qwen choice). Preflight now records whether `HF_TOKEN` is present (never its value). An unknown language stops with an error. |
| "One turn may span cutaways" | — | Implemented as an **independent visual timeline per scene** (§3). |

## 2. Audit of the existing architecture (before this work)

- **Narration-first cast plans:**
  - Lines are placed on the audio track first.
  - One SHOT = one visual unit, so a cut could only happen between lines.
- **Visual recipes:** AI video (Wan 2.1 T2V / VACE scene-state), drawn `ui`/`journey`/`card`/`diagram`, and licensed `stock` images.
- **No recurring characters:**
  - "Characters" were text descriptions for the video model, so their identity drifted between AI clips.
  - Drawn scenes had no people at all.
- **Dialogue facts:**
  - Factual checks (P4/P5) applied only to narrator lines.
  - Dialogue was treated as fiction.
- **Already strong, kept as is:**
  - Stage caching on Drive.
  - Per-shot edit cache.
  - Scene-state checks.
  - QC gate.
  - Expressive voices.
  - The Telugu path.
  - The licence registry.

## 3. What was built (changed files and their purpose)

| File | Purpose |
|---|---|
| `colab/engine/lib/characters.py` (new) | **Character Bible** (Pydantic). Arjun (curious) and Meera (expert): stable ids, canonical look, personality, voices per language (en: Kokoro `am_michael` / `af_heart`; te: Parler `Prakash` / `Lalitha` with a style description), allowed expressions and gestures, content-hash **asset ids**. |
| `colab/engine/lib/charanim.py` (new) | **2D character animation on the CPU.** Original vector-style drawings made from the Bible. Cached sprites (2x supersampled). Breathing, deterministic blinks, gestures (explain / point / think / open hands / nod / show), 8 expressions, held props (3D glasses, popcorn), worn 3D glasses. 5 worlds (living room, cinema seats, cinema screen, lobby, studio) and 6 shot types (two-shot, wide, close-up, reaction, over-the-shoulder, establishing from behind). Cinema screen shows a "double" or "clean" picture. Reference sheets with asset ids. |
| `colab/engine/lib/demos.py` (new) | **Physical demonstrations:** the finger test, two eyes → two pictures → one view with depth, and projector → silver screen → polarised lenses → eyes (top view). Also licensed photos with a slow camera move and an on-screen credit. Physics checked: seen with the left eye, a near object appears to the right. |
| `colab/engine/lib/story.py` (new) | **Story Director + Visual Story Director:** schemas, staged LLM prompts, dialogue rules D0–D7, grounding (NLI), grounding enforcement (unsupported turns are dropped), visual selection with a fallback chain, the two statistics, plan builder, acting (speaker gesture/expression per turn, listener nods). |
| `colab/engine/lib/drawkit.py` | New drawn kinds `character`, `demo`, `photo`, plus `story` (a scene's sub-visuals drawn back to back, exactly the scene's frame count). |
| `colab/engine/lib/validate.py` | Story visuals count as "what is seen" (A1), including what each demo shows. With `factual_dialogue`, a friend's fact/correction line must cite a sourced claim (P4) and may only use numbers from it (P5). S3 looks at a scene's own cuts. |
| `colab/engine/lib/registry.py` | `gdhe17/Self-Forcing` entry; `REVISIONS` (commit per model); `MEDIA_SOURCES` (characters, demos, fonts, music, sfx, photos). |
| `colab/engine/lib/voice_qc.py` | Identity is not judged on clips < 3 s (`MIN_IDENTITY_S`). Measured: deterministic Kokoro lines of 2.0–2.5 s scored 0.79–0.80 against their own voice. They are UNKNOWN (logged), not a false FAIL. |
| `cells/cast/settings.py` (+ narrator, topic head) | `STORY_MODE`, `VISUAL_STYLE`, `LICENSE_STRICT`, `LIP_SYNC`, `MAX_AI_VIDEO_SHOTS`, `MAX_GPU_MINUTES`. Existing names are reused: TARGET_DURATION = `TARGET_SECONDS`, MAX_REPAIR_ATTEMPTS = `REPAIR_BUDGET`, `RUN_MODE`. |
| `cells/cast/plan.py` | Accepts story/character/demo, loads the plan's Bible, resolves photo paths, re-runs the dialogue rules on the plan as it will be made, and enforces the AI-shot budget (X6). |
| `cells/shared/helpers.py` | **Licence gate** (strict mode decides before any download), **preflight report** (`render/preflight.json`): CUDA, GPU, VRAM, compute capability, precision (bf16 only if native; T4 → fp16), RAM, disk, cache, HF token presence. |
| `cells/shared/diagrams.py` | Scene visual timeline from the placed lines (a visual starts at `{"line": k, "at": 0..1}`), per-frame speaker loudness, cache key = cut points + mouths + Bible + `CHARANIM_VERSION`. |
| `cells/shared/clips.py` | `MAX_GPU_MINUTES`: no new clip after the budget. The remaining shots use their fallback and are listed for review. |
| `cells/shared/qc.py` | Storytelling checks + two statistics + `render/dialogue_shot_timeline.md` + `render/story_stats.json`. |
| `cells/cast/timeline.py`, `cells/shared/lib.py` | Storyboard text for story scenes; new modules imported. |
| `cells/topic/topic.py` | Story mode for the topic notebook: facts → beats → dialogue → visuals → grounding → licensed media → plan. English only (see §8). |
| `colab/build.py`, `colab/build_story_plan.py` (new) | Placeholders for the new settings. Story JSON → checked plan, with `--ground` to run the NLI check. |
| `colab/stories/three_d_friends_story.json` (new) | The Phase D story: a hand-written stand-in for the LLM stages, grounded. |
| `colab/three_d_friends_plan.json`, `colab/colab_three_d_friends_v1.ipynb` (new) | The sample plan and notebook. |
| `colab/samples/` (new) | Playable draft sample + contact sheet + dialogue/shot timeline + statistics + QC report. |
| `colab/tests/test_story.py`, `run_topic_story.py` (new); `run_nb.py`, `plan_check.py`, `test_registry.py` | Tests (§6). |

## 4. Schemas (backward compatible)

- **Story (LLM output, `story.Story`):**
  - `title`, `topic`, `language`, `logline`.
  - `facts[]` = `{id, text, source_url, title, revid}`.
  - `scenes[]` = `{id, beat, world, transition?, turns[], visuals[]}`.
- **Turn** = `{speaker, text (≤30 words, sentences ≤22), kind (fact|question|analogy|reaction|misconception|correction|instruction|observation), fact_ids[], tone, tts?, expr?, gesture?}`.
- **Visual** = `{id, intent (13 intents), turn, at, subject?, template?, description, lines[], chars{}, changes[], screen?, shot?, world?}`.
- **Plan (cast format, unchanged keys):**
  - Story plans add `story_mode`, `license_strict`, `factual_dialogue` and `bible`.
  - A shot with `recipe: "story"` keeps `lines` (the audio units, each with `kind` and `claims`) and adds `story.visuals[]` = `{id, intent, medium, recipe, <recipe spec>, start: {line, at}, description, label?, fallback_trail?}`.
  - Old plans have none of these keys and behave exactly as before (tested).

## 5. Selection rules (Visual Story Director) and safeguards

- **Fallback chain** (each decision is recorded in `plan.director` with its trail):
  1. **Licensed media** for `real_event` / `licensed_media`, when an image passed the licence policy and the SigLIP 2 relevance gate. It is drawn with its credit.
  2. **Animated demonstration** for `object_demo` / `animated_illustration` / `analogy`, when a template fits.
  3. **AI video:** allowed only as whole shots within `MAX_AI_VIDEO_SHOTS`, and only for scenes without the 2D friends (Wan can't draw *these* characters). Inside a dialogue scene it is not built yet; the trail says so.
  4. **Minimal overlay** only for ≤ 6-word lines.
  5. Otherwise the **character scene**, which always works. A re-enactment of a real event is labelled **"Illustration"**, never presented as footage.
- **Graphics excess:**
  - Warning V5 when > 20 % of visuals are overlays.
  - QC FAIL when presentation graphics exceed 20 % of screen time.
  - Validator S2 still rejects flowchart-heavy plans.
- **Two statistics (QC, by screen time):**
  - *Narrative purpose* (two_friend, closeup, demonstration, …).
  - *Rendering medium* (character_scene, animated_demo, licensed_media, ai_video, presentation_graphic).
- **Mouth:**
  - `LIP_SYNC = "energy"` opens the mouth only while that speaker's own audio is loud (normalised per line, small hold), and keeps it closed in pauses and while the other friend talks.
  - This is **not phoneme lip sync**, and it is not claimed to be.
- **Colab Free:**
  - Preflight report.
  - Strict licence gate before downloads.
  - One heavy GPU family at a time (unchanged).
  - `MAX_GPU_MINUTES`.
  - Story scenes need **no GPU** (CPU drawing in a background thread).
  - Draft mode renders at 1280x720.

## 6. Tests (actual results, this machine, CPU only)

| Suite | Checks | Result |
|---|---|---|
| **test_story.py (new)** | 53 | ALL PASSED. Coverage: Bible validation, stable asset ids, EN/TE routing + no language substitution, D1–D7 rules, schema limits, grounding with a stand-in NLI, fallback chain (media / demo / overlay / AI trail), graphics excess, plan + validator P4/P5/V1/A1, backward compatibility, frame-exact visual timeline, cut inside a turn, idle life, loudness-driven mouth, determinism, speed, every world x shot, demo physics, scripted LLM stages with a retry, preflight precision (T4 fp16 / A100 bf16 / CPU fp32), HF token presence without its value, strict licence gate on/off, no silent lip-sync download, old notebooks build with `narrator` + `LICENSE_STRICT = False` |
| test_registry.py | 8 (+1 online) | ALL PASSED, incl. revisions recorded and media sources; online licences match the Hub |
| voice 24, voice_backend 6, qc 14, subtitles 7, planner 16, assets 8, telugu 25, speed 25, logic 27, consistency 62, lib 31 | 245 | ALL PASSED (re-run after the changes) |
| **Total** | **306 offline + 1 online** | **all passing** |

**End-to-end runs (whole notebooks on CPU, `RUN_MODE = "draft"`):**
1. **Two-friend 3D sample:** APPROVED (every mandatory QC check PASS), 89.0 s, 0 AI clips, 8 scenes, 18 visuals.
2. **Topic notebook in story mode** (black holes): APPROVED. It used real Wikipedia research, the real NLI, and one Commons image (NASA illustration, with its credit on screen). The planner was replaced by a scripted stand-in that writes from the real facts (`tests/run_topic_story.py`).
   - On its first run the plan validator caught 24-word sentences. That led to enforcing the 22-word limit in the turn schema, so the LLM gets the error during its retry.
   - It also led to the new repeat rule D7.
3. **Backward compatibility (old 3D narrator notebook):** identical to its pre-change run. The same 14 checks pass. The same 4 checks fail, as expected without a GPU (VACE scene-state shots).

**Grounding of the sample with the real NLI (`cross-encoder/nli-deberta-v3-base`):**
- The first draft **did not pass**. 3 lines failed the number gate ("one"/"two" not in the cited fact), and 3 were "neutral" (paraphrased too loosely).
- They were reworded closer to the sources.
- Now **10/10 claim-making turns PASS** (`colab/stories/three_d_friends_story_grounding.json`).
- Observations, reactions and instructions make no sourced claim and are not NLI-checked (by design).

## 7. Measured on the prototype (CPU laptop, draft 1280x720)

| Measure | Value |
|---|---|
| Peak memory | 1.39 GB (Python process incl. children) |
| Runtime per stage, first run | narration 72.9 s (Kokoro + Whisper tiny + ECAPA); 8 scenes drawn in the background in 117 s total (6.8–20.8 s per scene, 142–430 frames); render 114.8 s (incl. waiting for the drawings); music 15.2 s; QC 2.6 s; total ≈ 3.5 min |
| Second run (resume) | ≈ 25 s: 0 voices remade, 0 scenes redrawn, 0 shots re-edited |
| Drawing speed | 22–90 ms per 1080p frame on one core |
| AI video clips | 0 (none needed; strict licence mode) |
| Failed / retried shots | 0 / 0 |
| Licensing | characters, demos, sfx: original; music CC BY credited; fonts DejaVu; no review-status model loaded |
| Voice identity | lines ≥ 3 s pass (Kokoro, deterministic); 3 lines < 3 s UNKNOWN (too short to judge) |
| Visual relevance and story coherence | reviewed by eye on frame sheets. Fixed: label collision, caption under the subtitles, props out of frame in close-ups, fade of picture frames. Not yet judged by other viewers. |

## 8. Remaining limitations (explicit)

1. **Nothing here ran on a GPU.**
   - Not measured: T4 timings, AI cutaways and the strict-mode slowdown without turbo (~4x per clip, estimated).
   - Not measured: Qwen3-4B story quality. The topic e2e used a scripted stand-in; the 4B model's dialogue has not been read yet.
2. **Lip sync** is loudness-driven mouth opening, not phonemes. Rhubarb (MIT) is not bundled. Nobody has judged whether it looks convincing.
3. **Telugu two-friend videos:**
   - The plan path works (Bible voices, Parler routing, Telugu names).
   - The automatic topic story is English-only, because the NLI fact check is English-only. A Telugu story must be written as JSON and built with `build_story_plan.py`.
   - No Telugu two-friend video was rendered in this session.
4. **AI video inside a dialogue scene** (a cutaway mid-turn) is not built. Only whole AI shots, budgeted, for scenes without the friends.
5. **Commons video** is not used: mostly BY-SA for these topics. Only images are wired in.
6. **Revision pins** are recorded and checked online, but not yet enforced as `revision=` when loading.
7. **Art style:** one flat 2D style, two characters, five worlds and three demo templates. A new topic usually needs a new demo template (code) or falls back to character scenes.
8. **Visual quality** was judged by me on frame sheets, not by viewers. **Retention can only be measured on YouTube.**

## 9. How to run

**Sample (plan notebook):**
1. Upload `colab/colab_three_d_friends_v1.ipynb` to Colab.
2. Run all. A GPU is not needed.
3. `RUN_MODE = "draft"` for a quick 720p check, `"standard"` for 1080p.
4. Output: `MyDrive/ai-video/three_d_friends_v1/render/` → `final.mp4`, `dialogue_shot_timeline.md`, `story_stats.json`, `preflight.json`, `qc_report.json`, `contact_sheet.jpg`.

**Your own story:**
1. Write `colab/stories/<x>_story.json` (schema in §4).
2. Build the plan:
   `<python with pydantic> colab/build_story_plan.py colab/stories/<x>_story.json colab/<x>_plan.json --project <x>_v1 --ground <hf cache dir>`
3. Build the notebook: `python3 colab/build.py colab/<x>_plan.json colab/colab_<x>_v1.ipynb`.

**Topic notebook:**
1. In `colab_topic_to_video.ipynb`, set `TOPIC`, keep `STORY_MODE = "two_friends_cinematic"`, and choose a T4.
2. Run all.
3. The story, grounding and media stages are cached in `stages/`, so after a disconnect, Run all continues.
4. To get the old single-narrator video, use `STORY_MODE = "narrator"`.

**Example dialogue and storyboard:** `colab/scripts/three_d_friends_script.md`, `colab/samples/storyboard.md`, `colab/samples/dialogue_shot_timeline.md`.
**Playable sample:** `colab/samples/three_d_friends_draft_720p.mp4`.

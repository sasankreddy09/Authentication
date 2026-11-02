# MASTER IMPLEMENTATION PROMPT
## Cinematic 3D-Style Character Upgrade for My Existing AI Educational Video Automation System

You are acting as a senior AI image-generation researcher, character-design engineer, computer-graphics engineer, animation-pipeline architect, and Python developer.

Your task is to **research, validate, implement, test, and integrate a substantial visual-quality upgrade to my existing AI video automation system**.

I want two exceptionally cute, cinematic, 3D-style recurring characters, Arjun and Meera, based on the aesthetic qualities of the two reference portraits I have supplied.

My existing system already works. It generates stories, dialogue, narration, animations, educational demonstrations, and final videos. However, the currently generated Arjun and Meera characters look like flat vector cartoons, and their appearance is not good enough.

**Do not rebuild the video system. Improve the existing Storybook character-generation and rendering architecture.**

### Primary objectives

In priority order:

1. Exceptional, adorable, cinematic 3D-style character appearance.
2. Consistent character identities across expressions, poses, camera views and videos.
3. High-quality animation without destroying the visual appearance.
4. One-time AI asset creation, followed by unlimited reuse of approved assets.
5. Compatibility with Google Colab Free, preferably a T4 GPU.
6. Free, open-weight tools with licenses compatible with potential YouTube monetization.
7. Minimal changes to the current working architecture.
8. Reliable recovery from Colab disconnections, human approval, caching and automated QC.

The result should be suitable for recurring characters in hundreds of educational videos.

---

# PART 1 — UNDERSTAND THE EXISTING ARCHITECTURE

## 1.1 Read the actual project before writing code

Read the supplied documents completely:

- `17_STORYBOOK_CHARACTERS.md`
- `ai_video_automation_research.md`

Then inspect the actual repository.

Read the most recent architecture documents, including these if they exist:

- `docs/AI_VIDEO_AUTOMATION_MASTER_PLAN.md`
- `docs/02_ARCHITECTURE.md`
- `docs/08_COLAB_VIDEO_PIPELINE_GUIDE.md`
- `docs/09_LOGICAL_CONSISTENCY_ARCHITECTURE.md`
- `docs/12_TWO_FRIEND_STORYTELLING.md`
- `docs/15_ARCHITECTURE_VALIDATION.md`
- The latest Storybook character documentation.

Search for files if their paths differ. Do not assume that an uploaded document is present in the repository at exactly the same path.

Some sections of `ai_video_automation_research.md` are explicitly superseded.

**The actual implementation and most recent architecture decisions take precedence over historical research plans.**

## 1.2 Existing modules to inspect

At minimum, examine:

```text
colab/engine/lib/storybook_art.py
colab/engine/lib/storybook.py
colab/engine/lib/storybook_scene.py
colab/engine/lib/studio.py
colab/engine/lib/envfactory.py
colab/engine/lib/charanim.py
colab/engine/lib/registry.py

colab/build_studio.py
colab/approve_character.py
colab/storybook_previews.py
colab/build_story_plan.py
colab/build.py

colab/tests/test_storybook.py
colab/tests/test_assets.py
colab/tests/test_registry.py
```

Also inspect the relevant story-planning, character Bible, narration, voice-direction, viseme, rendering, QC and environment components.

Verify actual filenames and interfaces instead of assuming the list is exhaustive.

## 1.3 Architecture that must be preserved

The current architecture is approximately:

```text
Topic
  |
  v
Research + Fact Validation
  |
  v
Story Director + Dialogue
  |
  v
Visual Story Director
  |
  v
JSON Video Plan
  |
  +--------------------------------+
  |                                |
  v                                v
Character Director           Environment Director
storybook_scene.py           envfactory.py
  |                                |
  v                                v
Approved Character Assets   Approved Environment Assets
  |                                |
  +---------------+----------------+
                  |
                  v
         Pillow Compositor
                  |
                  v
       Existing Drawkit / FFmpeg
                  |
                  v
          Existing Video QC
                  |
                  v
             Final Video
```

Important existing design decisions:

- The planner produces structured data, not arbitrary new rendering code.
- The tested executor remains fixed.
- The existing rendering implementation uses Python, Pillow and FFmpeg.
- `visual_style_version="storybook_v1"` enables the Storybook renderer.
- Legacy plans still use the existing legacy path.
- Character versions are pinned in `character_assets`.
- Environment versions are pinned separately.
- Assets have manifests and SHA-256 verification.
- Human approval is required.
- Missing approved assets must produce an error, not a silent fallback.
- Google Drive holds persistent assets and cached work.
- The narration timeline controls dialogue timing.
- Audio-derived visemes control the characters' mouths.
- The existing QC system uses PASS, FAIL and UNKNOWN, with mandatory UNKNOWN results blocking approval.

Preserve these decisions unless a specific change is necessary, justified, backward-compatible and covered by tests.

Do not introduce:

- A new per-video LLM code-generation architecture.
- A new GPU broker.
- Paid orchestration services.
- An unnecessary database.
- A mandatory cloud rendering service.
- A completely separate animation engine.
- A new project replacing the current project.

## 1.4 Existing Storybook capabilities

The latest Storybook design already contains:

- Approximately 75 RGBA animation layers per character.
- 13 predefined poses.
- 12 predefined expressions.
- 13 mouth shapes.
- Blinking and eye movement.
- Facial-expression controls.
- Head tilt, nodding and small body motion.
- Audio-derived visemes.
- Versioned manifests.
- Character approval.
- Cached environment packs.
- CPU rendering.

The existing 13 poses are:

```text
idle
listen
explain
open_hands
point
think
wave
surprise
celebrate
duck
hold
show
nod
```

The existing 12 expressions are:

```text
neutral
listening
smile
happy
curious
confused
surprised
excited
thinking
concerned
laughing
wink
```

First map the new artwork to these existing interfaces.

Additional poses or expressions are optional extensions and must be implemented explicitly rather than assumed to exist.

## 1.5 Identify the actual problem

The existing visual output is produced using procedural Bézier shapes and geometric layers.

That system is useful for animation but has limited visual sophistication.

The main problem is the **source artwork quality**, not the overall story architecture.

Do not respond by simply increasing gradients, shadows or eye highlights in the same procedural drawing style.

The target requires artistically coherent, AI-generated, dimensional character designs.

The new character artwork must retain that dimensional quality throughout animation.

---

# PART 2 — VISUAL TARGET AND CHARACTER IDENTITY

## 2.1 Reference images

I have supplied four images:

1. Existing Arjun character sheet.
2. Existing Meera character sheet.
3. Desired boy character reference.
4. Desired girl character reference.

Images 1 and 2 show the current quality that needs replacing.

Images 3 and 4 represent the desired appearance.

Study these images carefully before selecting a model.

Analyze:

- Head-to-body proportions.
- Eye design.
- Hair modeling.
- Facial features.
- Skin shading.
- Lighting.
- Clothing materials.
- Color palette.
- Dimensionality.
- Expression appeal.
- Overall cinematic quality.

The reference images are artistic guidance, not characters to reproduce identically.

Create original, recognizable characters.

## 2.2 Arjun — Golden Hoodie Boy

Arjun is an adorable, curious, intelligent boy approximately 9–11 years old.

Permanent character features:

- Warm medium-brown skin.
- Large luminous chocolate-brown eyes.
- Soft rounded cheeks.
- Expressive natural eyebrows.
- Small rounded nose.
- Fluffy, slightly messy dark-chocolate curly hair.
- A recognizable signature forehead curl.
- Rich golden-yellow or amber hoodie.
- Burgundy or dark-maroon inner hood.
- Soft, believable hoodie fabric folds.
- Friendly, curious and emotionally expressive appearance.
- Appealing, stylized child proportions.

Arjun should look like a charming cinematic storybook character, not a flat vector cartoon or photorealistic human.

**The golden hoodie and signature curly hairstyle must remain consistent across every asset version unless I explicitly authorize a redesign.**

## 2.3 Meera — Butterfly-Hair Girl

Meera is an adorable, clever, confident girl approximately 9–11 years old.

Permanent character features:

- Warm brown skin.
- Large luminous expressive brown eyes.
- Soft rounded cheeks and gentle blush.
- Natural, delicate eyelashes.
- Small rounded nose.
- Long, flowing chestnut-brown wavy hair.
- A beautiful half-up hairstyle with a soft bun.
- Small butterfly-shaped ornaments nestled in her hair.
- Warm coral, peach or golden accents in the ornaments.
- Elegant strands framing her face.
- Soft ivory, sage-green or peach clothing with delicate fabric details.
- Intelligent, gentle, playful and confident personality.

**Her butterfly hair ornaments and recognizable half-up hairstyle are permanent identity features.**

The ornaments must not randomly disappear, change sides, become oversized or turn into unrelated decorations.

## 2.4 Shared artistic direction

Both characters must appear in the same visual universe.

Target style:

- Premium cinematic 3D-style storybook illustration.
- Softly sculpted facial features.
- Rich, volumetric hair.
- Large eyes with nuanced reflections.
- Gentle global-illumination-like lighting.
- Subsurface-scattering-like skin appearance.
- Soft facial shadows.
- Natural color gradients.
- Attractive, coherent fabric materials.
- Soft cinematic highlights.
- Warm, dreamy visual atmosphere.
- Appealing silhouettes.
- Childlike proportions without grotesque exaggeration.
- Cohesive character design at close-up, medium-shot and full-body scale.

Avoid:

- Flat vectors.
- Generic clipart.
- Emoji-like faces.
- Stiff geometric bodies.
- Plastic skin.
- Overly glossy surfaces.
- Poor facial symmetry.
- Distorted hands.
- Uncanny photorealism.
- Excessive eye enlargement.
- Unwanted character redesigns between poses.
- Inconsistent lighting and materials.

**Cinematic 3D-style appearance does not require creating actual 3D meshes.**

High-quality pre-rendered 2D assets with a consistent dimensional appearance are acceptable and may be more practical on Colab Free.

---

# PART 3 — RESEARCH AND MODEL SELECTION

## 3.1 Investigate the existing Character Studio first

The current Character Studio already proposes:

- FLUX.2 [klein] 4B for concept generation and reference-conditioned editing.
- BiRefNet for foreground segmentation.
- Human selection through contact sheets.
- Semi-automatic layered asset preparation.
- Immutable, versioned character asset packs.
- Existing Pillow animation integration.

Do not replace these tools merely because alternatives exist.

First establish whether the current implementation is sufficient for the desired appearance.

## 3.2 Primary model to evaluate

Start with:

**FLUX.2 [klein] 4B — distilled variant**

Official project:

https://github.com/black-forest-labs/flux2

Evaluate:

- Text-to-image.
- Single-reference editing.
- Multi-reference editing.
- Cinematic child character quality.
- Consistent expression editing.
- Consistent camera-view changes.
- Colab T4 memory requirements.
- FP16 inference.
- Available quantization.
- CPU/model offloading.
- Image-generation time.
- Host RAM consumption.
- Compatibility with the existing code.

The model is a promising starting point, but its successful execution in this project's actual Colab T4 environment must be measured.

Do not confuse official performance claims on newer consumer GPUs with measured results on a T4.

## 3.3 Alternative models

Research additional genuinely viable options, including:

- Z-Image-Turbo for character concepts or reusable backgrounds.
- Smaller open-weight reference-editing models where appropriate.
- Other permissively licensed models capable of the required visual quality.

Do not assume a text-to-image model supports image editing unless the official implementation confirms that capability.

Reject models that require paid APIs or licenses incompatible with the project's intended use.

In particular, do not silently substitute non-commercial FLUX variants.

## 3.4 Supporting models

Investigate BiRefNet for clean foreground extraction.

Determine whether its selected model weights and dependencies meet the project's license requirements.

Distinguish between:

- Foreground/background segmentation.
- Individual body-part segmentation.
- Hair masking.
- Rig-ready layer extraction.
- Inpainting of occluded body parts.

These are not the same task.

A foreground segmentation model alone cannot automatically produce a complete, beautiful 75-layer animation rig.

Evaluate the additional processing required.

## 3.5 Licensing

The system must support potentially monetized educational videos.

For each model and dependency, verify:

- Source-code license.
- Model-weight license.
- Commercial-use permissions.
- Any attribution obligations.
- Any restrictions on redistribution.
- Any restrictions on generated outputs.
- Access or gating requirements.

Record exact model identifiers and revisions.

Integrate the selected models into the existing license registry and strict-license checks.

Do not disable `LICENSE_STRICT` merely to make a new model work.

Do not use restricted models such as non-commercial FLUX variants or non-commercial background-removal checkpoints unless their use is separately authorized under appropriate terms.

## 3.6 Free hosted inference research

Research optional assistance from:

- Hugging Face Spaces / ZeroGPU.
- Hugging Face Inference Providers.
- Cloudflare Workers AI.
- Other real free-tier inference services.

Report verified:

- Available models.
- Free monthly or daily usage.
- Whether a credit card is required.
- Image-generation and editing capabilities.
- Commercial-use terms.
- Rate limits.
- Persistent storage behavior.
- Availability from the intended region.

These services are optional.

**The core architecture must remain usable without depending on a paid hosted image-generation service.**

Do not invent free-tier quotas or assume they will remain unchanged.

---

# PART 4 — GOOGLE COLAB FREE COMPATIBILITY

## 4.1 Hardware assumptions

The target is Google Colab Free, preferably with an NVIDIA T4 when available.

Do not assume:

- GPU access is always available.
- A T4 is always assigned.
- BF16 is natively supported.
- System RAM is sufficient for arbitrary CPU offloading.
- Sessions last a fixed number of hours.
- Multiple models can remain loaded simultaneously.

Before inference, detect:

- Actual GPU.
- Available VRAM.
- CUDA capabilities.
- Available RAM.
- Available disk space.
- Torch/CUDA compatibility.
- Required dependency versions.
- Model download and cache size.

## 4.2 Memory optimization

Investigate and benchmark:

- FP16 inference.
- Supported text-encoder quantization.
- Sequential CPU offloading.
- Model-component unloading.
- Reduced initial image resolution.
- Controlled reference-image count.
- Explicit garbage collection.
- CUDA cache cleanup.
- Existing project model-release functions.

Use the current project's memory-management conventions where possible.

Do not blindly copy BF16 examples into a T4 workflow.

Do not allow an out-of-memory failure to silently produce lower-quality vector artwork.

When the system cannot complete a model operation, it should report the reason, preserve completed results, and offer a compatible retry setting.

## 4.3 GPU scheduling

Create and maintain two separate workflows.

### Workflow A — One-time Character Studio

GPU-intensive tasks:

```text
Preflight
   ↓
Load image model
   ↓
Generate concepts
   ↓
Select canonical masters
   ↓
Reference-conditioned variants
   ↓
Unload image model
   ↓
Foreground segmentation
   ↓
Asset preparation
   ↓
Preview
   ↓
Human approval
   ↓
Save permanent asset version
```

The stages should be independently resumable.

Do not load the large image model and segmentation model simultaneously when unnecessary.

### Workflow B — Normal Video Generation

```text
Load approved character assets
   ↓
Read existing story plan
   ↓
Apply poses and expressions
   ↓
Animate eyes and mouth
   ↓
Composite with environments
   ↓
Render with Pillow / FFmpeg
   ↓
Existing QC
```

**Normal character rendering must not load, download or call the character image-generation model.**

Other existing AI video, speech or QC stages may still use the GPU according to the existing pipeline. Do not mistakenly remove them.

---

# PART 5 — MANDATORY VISUAL PROOF OF CONCEPT

This is a hard approval gate.

Do not generate an entire character library immediately.

First demonstrate the artistic quality.

## 5.1 Generate initial concepts

Produce a small set of candidates for each character.

Use controlled seeds and consistent settings.

Store:

- Prompt.
- Model revision.
- Seed.
- Sampling configuration.
- Generation dimensions.
- Reference images.
- Output images.
- Execution metadata.

Create clear, numbered contact sheets.

The candidates must reflect the target reference aesthetic.

## 5.2 Generate canonical candidates

Create for each character:

1. One high-quality facial portrait.
2. One full-body neutral character image.
3. One alternate facial expression.
4. One three-quarter view.

Use reference-conditioned editing where technically supported.

Do not assume independently generated images show the same character.

Prioritize a consistent original identity over getting every view in a single generation.

## 5.3 Separate beauty portraits from animation masters

This distinction is essential.

A beautiful portrait with cinematic bokeh is not automatically a suitable animation source.

Maintain two related kinds of artwork:

**Presentation images**

High-quality, beautifully lit cinematic portraits used for visual evaluation, thumbnails and character showcases.

**Production masters**

Neutral, consistently framed character assets with controlled lighting, clean backgrounds, complete visible anatomy and sufficient resolution for extraction and animation.

Both must preserve the same character identity and art direction.

Avoid using a cropped portrait as the only source for a full-body rig.

## 5.4 Visual approval

Present the generated results side by side with the user-provided references.

Ask for explicit approval of:

- Arjun's face.
- Arjun's hair.
- Arjun's golden hoodie.
- Meera's face.
- Meera's butterfly hairstyle.
- Shared artistic style.
- Visual consistency.

Do not automatically approve a candidate using only similarity metrics.

Do not continue to full asset production until the selected designs are approved.

If the selected model cannot reach the target quality, investigate prompt changes, reference-conditioning settings and alternative eligible models.

Do not compensate for poor character artwork by merely adding more procedural shading.

## 5.5 If no GPU is available

If the coding environment lacks a real GPU:

- Implement the runnable proof-of-concept notebook.
- Run CPU-based configuration and import tests.
- Validate model identifiers, APIs and dependencies.
- Provide exact Colab run instructions.
- Mark actual image-quality and performance results as pending.

Do not fabricate images, timings, screenshots or successful GPU execution.

---

# PART 6 — CHARACTER IDENTITY AND CONSISTENT ASSETS

After the visual proof of concept is approved, prepare an animation asset version for each character.

## 6.1 Canonical identity

Store a canonical character description and approved master images.

Maintain immutable identity features:

**Arjun**

- Golden hoodie.
- Burgundy hood interior.
- Signature curly hair.
- Brown eyes.
- Warm medium-brown skin.

**Meera**

- Half-up chestnut hair.
- Butterfly ornaments.
- Brown eyes.
- Warm brown skin.
- Approved outfit.

Do not change these features between shots unless the plan explicitly requests an allowed costume or state variation.

## 6.2 Required camera views

Support the views needed by existing Storybook shots first:

- Front.
- Left three-quarter.
- Right three-quarter.
- Back.
- Medium-shot front.
- Full-body neutral.
- Facial close-up.

Profile views can be added when they materially improve supported scenes.

Do not attempt unlimited arbitrary 3D rotation from 2D artwork.

For unsupported views, use approved pre-rendered assets or report an explicit limitation.

## 6.3 Expressions

Support the existing 12 expression IDs before adding anything new.

Produce expressions that preserve:

- Eye shape and color.
- Face proportions.
- Hair silhouette.
- Clothing.
- Lighting direction.
- Character identity.

Expressions must look like emotions of the same character, not like newly generated people.

Optional additional expressions such as sadness or mild frustration should be introduced through backward-compatible additions to the expression registry and tested.

## 6.4 Poses

Support the existing 13 Storybook poses.

Start with those most important for conversations:

- Idle.
- Listen.
- Explain.
- Think.
- Point.
- Open hands.
- Surprise.
- Celebrate.

Then complete the remaining existing poses.

Use a consistent full-body reference and approved character proportions.

Additional walking or sitting assets may be developed when the project requires them, but do not silently expand the implementation scope.

## 6.5 Avoid combinatorial asset explosion

Do not independently generate every possible:

```text
Pose × Expression × Camera Angle × Mouth Shape
```

This would be expensive and unnecessary.

Instead, determine which features can safely be composed from reusable assets.

Use pre-rendered combinations only where needed to preserve visual fidelity.

Prioritize common storytelling poses and facial close-ups.

---

# PART 7 — CHOOSE THE BEST ANIMATION STRATEGY

A major technical risk is that converting cinematic character artwork into rotating image pieces may destroy the visual quality.

Investigate this before mass-producing layers.

## 7.1 Compare two approaches

### Approach A — High-quality cutout rig

Reuse the current Pillow rig and replace the procedural shapes with high-quality AI-generated artwork.

Evaluate:

- Pivot positioning.
- Joint appearance.
- Rotated fabric shading.
- Hair overlap.
- Hand transitions.
- Shoulder continuity.
- Facial-expression overlays.
- Layer ordering.

### Approach B — Pre-rendered pose atlas with lightweight animation

Generate and approve a curated library of high-quality poses and expressions.

Animate using:

- Small translations.
- Gentle scaling.
- Minimal head tilt.
- Blinking.
- Eye movement.
- Mouth overlays.
- Breathing.
- Subtle pose changes.
- Controlled transitions.

This approach may retain the cinematic artwork more effectively.

## 7.2 Selection criteria

Compare both strategies for:

- Visual quality.
- Character identity.
- Natural movement.
- Joint artifacts.
- Mouth animation.
- Implementation complexity.
- CPU rendering cost.
- Google Drive storage.
- Reuse flexibility.
- Existing-code compatibility.

Choose the simplest option that demonstrably preserves the target visual quality.

A hybrid approach is acceptable.

Do not introduce Godot, Blender rendering or heavy video-generation dependencies unless the existing Pillow architecture demonstrably cannot satisfy a specific mandatory requirement and the replacement is independently justified.

## 7.3 Preserve existing animation timing

Do not break:

- Per-speaker audio alignment.
- Existing `_visemes`.
- Mouth timing.
- Blinking schedule.
- Gaze changes.
- Listener nodding.
- Gesture timeline.
- Approximately 0.45-second eased pose transitions.
- Existing story-plan expression and gesture fields.

If a new asset format requires an adapter, implement it behind the existing interfaces where practical.

## 7.4 Mouth animation

The characters must speak without destroying the dimensional appearance of the face.

Test:

- Closed mouth.
- Small open mouth.
- Wide mouth.
- Rounded mouth.
- Smile while speaking.
- Expressions during speech.

Verify that the mouth overlays align with the chin, cheeks and facial shading.

Avoid a flat pasted-on cartoon mouth on an otherwise detailed 3D-style face.

If the existing 13 mouth shapes need improved artwork, preserve their IDs and timing semantics.

## 7.5 Hair and accessories

Pay special attention to:

- Arjun's signature curls.
- Hair overlapping the forehead.
- Meera's long hair.
- Meera's butterfly ornaments.
- Hair movement during head tilt.
- Proper foreground/background ordering.

These identity features must remain attached and look natural while the characters animate.

---

# PART 8 — ASSET EXTRACTION AND RIG PREPARATION

## 8.1 Foreground removal

Evaluate BiRefNet using an approved checkpoint.

Generate clean transparent PNG images.

Check:

- Hair strands.
- Transparency edges.
- Skin edges.
- Clothing.
- Fingers.
- Butterfly accessories.
- Edge halos.

## 8.2 Rig-ready extraction

Do not assume that background removal creates independently movable body parts.

Determine how to separate:

- Head.
- Front and back hair.
- Torso.
- Upper arms.
- Forearms.
- Hands.
- Legs.
- Eyes.
- Eyebrows.
- Mouth.
- Accessories.

Where parts are hidden behind the visible character, they may require one-time inpainting or manual painting.

Provide editable intermediate assets and templates for such corrections.

Do not claim that AI segmentation is fully reliable.

## 8.3 Compatibility with the existing asset schema

Preserve the current animation hierarchy wherever practical:

```text
hips
  └─ torso
       ├─ head
       │    ├─ eyes
       │    ├─ brows
       │    ├─ mouth
       │    └─ hair
       ├─ left arm
       └─ right arm
```

Retain existing joint limits and layer ordering unless specifically updated and tested.

If a hybrid pose atlas is selected, introduce an explicit, versioned asset representation rather than pretending that complete character renders are individual joint layers.

Do not create false or blank layers merely to satisfy an expected layer count.

---

# PART 9 — PERMANENT REUSABLE ASSET STORAGE

## 9.1 Preserve the existing Google Drive structure

Use the existing `STORYBOOK_ASSETS` root and versioned character directories.

An expanded version may resemble:

```text
MyDrive/ai-video/_assets/
  characters/
    arjun/
      v1/                      # Existing provisional version
      v2/
        manifest.json
        canonical/
        views/
        expressions/
        poses/
        layers/
        previews/
        provenance/
    meera/
      v1/
      v2/
        manifest.json
        canonical/
        views/
        expressions/
        poses/
        layers/
        previews/
        provenance/

  environments/
    cinema/
    lobby/
    studio/
    space_observatory/
```

Adapt this structure to the actual manifest implementation.

Do not invent a completely separate incompatible storage system.

## 9.2 Manifest requirements

Preserve existing fields and add new optional metadata where required.

Record:

- Character identifier.
- Version.
- Approval status.
- Canonical identity.
- Palette.
- Source references.
- Generation prompts.
- Random seeds.
- Model identifier and revision.
- License information.
- Asset type.
- Rig metadata.
- Pose and expression definitions.
- Camera-view definitions.
- Relative file paths.
- SHA-256 for each asset.
- Immutable content fingerprint.
- Human approver and approval time.

Validate the interaction between existing `manifest_sha256` plan pinning and approval-state updates.

Do not allow approval metadata changes to silently invalidate existing content hashes.

If the current design hashes the entire manifest, follow and test its intended approval/freeze sequence. If a content-only fingerprint is needed, add it in a backward-compatible way.

## 9.3 Immutable asset versions

Never overwrite an approved character version.

Create a new version for actual redesigns.

Use incomplete staging directories during generation, validate all assets, then publish a complete version.

Ensure an interrupted Colab session cannot leave a partial asset directory that appears approved.

If an asset fails SHA verification, stop with a clear error.

## 9.4 Plan integration

Preserve:

```json
{
  "visual_style_version": "storybook_v1",
  "character_assets": {
    "arjun": {
      "version": "v2",
      "status": "approved",
      "manifest_sha256": "<computed-approved-manifest-hash>"
    },
    "meera": {
      "version": "v2",
      "status": "approved",
      "manifest_sha256": "<computed-approved-manifest-hash>"
    }
  }
}
```

This is an illustrative shape.

Use the actual schema and preserve any existing required `art_version` fields.

Do not hardcode placeholder hash strings into generated plans.

Use the existing plan-building interface, including `build_story_plan.py --visual storybook_v1 --char-version v2`, after checking the current CLI.

## 9.5 No repeated generation

This requirement is mandatory.

If I create 100 videos using the approved characters:

- Their canonical faces must remain the same.
- Their permanent outfits must remain the same.
- Their asset versions must remain pinned.
- The image-generation model must not run again.
- Existing assets must load from Drive or a verified local cache.

Only a deliberate asset redesign or approved asset extension should trigger new character-image generation.

Prove this behavior with automated tests.

---

# PART 10 — BACKGROUND AND ENVIRONMENT COMPATIBILITY

The current system includes reusable environment packs for:

- Cinema.
- Lobby.
- Studio.
- Space observatory.

Do not replace these automatically.

First render the new characters inside the existing environments.

Evaluate whether the backgrounds appear too flat compared with the new cinematic characters.

If the visual mismatch is significant, implement a narrowly scoped environment upgrade.

Use:

- Consistent warm/cool lighting.
- Compatible camera perspective.
- Soft contact shadows.
- Correct character scale.
- Existing camera presets.
- Existing foreground parallax.
- Existing safe regions for subtitles and diagrams.

Optional AI-generated backgrounds should also be one-time, versioned, approved and reused.

Keep exact diagrams and educational demonstrations code-generated.

Do not replace mathematically accurate diagrams with generated images.

Do not change environment layouts or scene-state logic without regression tests.

---

# PART 11 — QUALITY CONTROL

Extend the existing PASS / FAIL / UNKNOWN QC framework.

Do not create a second incompatible approval system.

## 11.1 Character identity QC

Verify:

- Same facial identity across views.
- Consistent eyes.
- Consistent skin tone.
- Consistent hairstyle.
- Consistent outfit.
- Correct signature accessories.

Use supported identity or image-similarity checks as diagnostics where useful.

Do not assume that a generic image-embedding score proves identity.

Calibrate any automated thresholds using actual approved and rejected examples.

## 11.2 Artwork quality QC

Inspect:

- Eye anatomy.
- Facial symmetry.
- Hair detail.
- Hands.
- Fingers.
- Clothing folds.
- Blurriness.
- Lighting.
- Unwanted text.
- Watermarks.
- Artifacting.

Automated checks may flag issues, but final character-design approval remains human.

## 11.3 Rig QC

Check:

- Layer alignment.
- Correct pivots.
- Natural arm movement.
- No disconnected body parts.
- No transparent seams.
- No distorted face.
- No floating mouth.
- No detached butterflies.
- No clipped hair.
- Correct clothing overlap.

## 11.4 Animation QC

Generate previews for:

- Idle talking.
- Listening.
- Explaining.
- Thinking.
- Surprised reaction.
- Two-character conversation.

Include examples where characters blink, speak and gesture simultaneously.

The preview should demonstrate real reuse of the approved assets.

## 11.5 Approval gate

A character pack is approved only when:

- The required assets exist.
- Asset hashes verify.
- Mandatory rig checks pass.
- Required previews are generated.
- License checks pass.
- Human approval is explicitly recorded.

Do not silently mark an untested pack as approved.

Mandatory UNKNOWN results must continue to block final approval.

---

# PART 12 — PERFORMANCE BENCHMARKING

## 12.1 Character generation benchmark

Measure on actual Colab hardware:

- GPU name.
- VRAM capacity.
- Peak allocated VRAM.
- Peak reserved VRAM.
- Host RAM usage.
- Model download time.
- Model load time.
- Generation time per image.
- Reference-editing time.
- Segmentation time.
- Resolution.
- Output storage size.
- Model-unload effectiveness.
- Failure or OOM conditions.

Record the precise inference configuration.

Do not report timings from a different GPU as T4 results.

## 12.2 Rendering benchmark

Measure:

- 720p frame rendering.
- 1080p frame rendering.
- One-character shot.
- Two-character shot.
- Talking and blinking.
- Pose transitions.
- Memory usage.
- Cache behavior.

Compare the upgraded implementation against the existing Storybook baseline.

The earlier project documentation reports approximately 0.11–0.12 seconds per 720p frame and approximately 0.17 seconds per 1080p frame on its CPU test machine.

Treat those as historical CPU measurements, not guaranteed targets or Colab GPU benchmarks.

Document any speed or memory regression.

## 12.3 Resume benchmark

Run a test with an interrupted asset-generation stage.

Verify that:

- Completed images remain saved.
- Their checksums remain valid.
- Existing candidates are skipped.
- Failed work can resume.
- Approved assets are never overwritten.

---

# PART 13 — BACKWARD COMPATIBILITY

Do not break the working video automation system.

Preserve:

- Legacy `charanim` rendering.
- Existing Storybook plan schema.
- Existing story and dialogue generation.
- Fact checking.
- Telugu language support.
- Existing voice backends.
- Optional Gemini free-tier TTS behavior already present in the project.
- Voice timing.
- Subtitles.
- Demonstrations.
- Video generation recipes.
- Scene-state validation.
- Existing environment packs.
- Content-hash caching.
- Final QC.

This task is primarily a character-art and animation-quality upgrade.

Do not modify unrelated working features.

If a migration is necessary, provide a compatibility layer and regression tests.

---

# PART 14 — IMPLEMENTATION PHASES AND APPROVAL GATES

Follow these phases in order.

## Phase 1 — Existing-code audit

Inspect the actual implementation.

Deliver:

- Current architecture map.
- Why the characters look flat.
- Current Storybook capabilities.
- Current limitations.
- Exact code changes required.
- Risks to legacy compatibility.

Do not redesign the entire architecture.

## Phase 2 — Model feasibility

Research and compare suitable open-weight models.

Select:

- Primary image model.
- Backup image model if needed.
- Segmentation model.
- Preferred animation-asset strategy.

Verify licensing and documented resource requirements.

Implement the preflight and benchmark harness.

## Phase 3 — Visual proof of concept

Generate or prepare the notebook to generate:

- Arjun canonical portrait.
- Meera canonical portrait.
- Arjun full-body master.
- Meera full-body master.
- At least one reference-conditioned expression.
- At least one alternate camera view.

Save prompts, seeds, model versions and outputs.

Create a side-by-side visual comparison.

**Do not start full asset production until the character appearance is approved.**

## Phase 4 — Animation proof of concept

Using the approved sample artwork:

- Test the current layered rig.
- Test the pre-rendered pose strategy.
- Compare quality and performance.
- Validate mouth and eye animation.
- Render a short proof-of-concept conversation.

Decide which animation strategy preserves the visual appearance best.

Obtain approval of the animation appearance before mass-producing the asset library.

## Phase 5 — Complete reusable character packs

Build the existing required poses, expressions, mouth variants and views.

Prepare assets, metadata and previews.

Verify files and hashes.

Integrate with the current Character Studio and approval workflow.

## Phase 6 — Existing pipeline integration

Connect the approved assets to:

- Storybook Character Director.
- Existing visual plans.
- Existing environments.
- Pillow compositor.
- Existing video renderer.
- Existing QC.

Keep legacy plans unchanged.

## Phase 7 — End-to-end verification

Produce representative scenes or a short sample story using Arjun and Meera.

Test:

- A conversation.
- An explanation.
- A surprised reaction.
- A hand gesture.
- Camera changes.
- Two-character compositing.
- Audio-derived mouth animation.

Then render a second independent scene using the exact same approved asset versions.

Verify that no character-image generation occurred during the second scene.

Run all relevant existing tests.

## Phase 8 — Documentation and handoff

Document:

- One-time setup.
- Running the Character Studio.
- Supplying reference images.
- Selecting a candidate.
- Creating a version.
- Fixing any manual layer issues.
- Previewing animations.
- Approving assets.
- Generating videos with the approved characters.
- Handling a Colab disconnect.
- Adding an expression.
- Adding a pose.
- Creating a new character version.
- Troubleshooting GPU memory failures.

---

# PART 15 — EXAMPLE GENERATION PROMPTS

These are starting points, not mandatory literal strings.

Adapt them to the selected model's actual prompting format.

## Arjun master concept

A remarkably adorable original ten-year-old storybook boy with warm medium-brown skin, luminous chocolate-brown eyes, beautifully rounded cheeks, soft youthful facial structure, an expressive curious gaze, a small natural nose and a gentle charming smile.

He has fluffy dark-chocolate curly hair with beautifully sculpted individual curls and a distinctive curl falling across his forehead.

He wears a luxurious golden-amber hoodie with a deep burgundy inner hood, soft realistic knitted fabric folds and a comfortable childlike fit.

Premium cinematic 3D-style storybook character art, dimensional face sculpting, gently subsurface-shaded skin, layered natural eye reflections, carefully modeled curls, warm cinematic bounce lighting, soft golden rim light, nuanced shadows, physically believable materials, sophisticated color harmony and extremely appealing original character design.

Beautiful stylized child proportions, whimsical but refined, high-end animated-feature visual quality.

## Meera master concept

A remarkably adorable original ten-year-old storybook girl with warm brown skin, large luminous chocolate-brown eyes with delicate layered highlights, naturally rounded cheeks, a gentle rosy blush, soft youthful facial structure, delicate eyelashes and an intelligent, playful expression.

She has long flowing chestnut-brown wavy hair with an elegant half-up bun. Small warm coral and golden butterfly-shaped hair ornaments are nestled naturally in her hair, with delicate loose strands softly framing her face.

She wears a refined ivory and pastel-sage outfit with soft fabric textures, elegant details and gentle folds.

Premium cinematic 3D-style storybook character art, softly sculpted dimensional features, delicate skin shading, beautiful expressive eyes, coherent individual hair strands, realistic soft lighting, elegant golden-hour illumination, subtle rim light, premium material quality, harmonious colors and exceptionally charming original character design.

The result is whimsical, cinematic and emotionally appealing, not photorealistic or flat.

## Reference-conditioned expression prompt

Using the approved canonical character image as the identity reference, create the exact same original character with a new facial expression.

Preserve the facial structure, skin tone, eye color, hair design, permanent accessories, clothing and lighting.

Only change the expression to a natural, emotionally convincing look of joyful surprise.

Maintain the same artistic style and camera framing.

Do not redesign or replace the character.

## Reference-conditioned three-quarter view prompt

Using the approved canonical character reference, create the same original character in a natural three-quarter view.

Preserve the same face, hairstyle, clothing, color palette, body proportions and permanent accessories.

Maintain consistent sculpted three-dimensional appearance and compatible lighting.

Do not introduce a different face or hairstyle.

Use clean framing suitable for a reusable animation asset.

## Negative qualities

Where the selected model supports negative prompts, discourage:

Flat vector graphics, clipart, simple geometric cartoon shapes, emoji faces, stiff paper puppets, plastic skin, uncanny realism, distorted faces, mismatched eyes, malformed hands, extra fingers, duplicated limbs, broken hair, misplaced accessories, blurry facial details, exaggerated gloss, inconsistent clothing, low-quality shading, watermarks, unwanted text and logos.

Do not assume that negative prompts work identically across model families.

---

# PART 16 — DELIVERABLES

Produce the following:

### A. Research and architecture

- Clear audit of the existing implementation.
- Comparison of the most relevant eligible image models.
- Official source links and license verification.
- Colab T4 feasibility assessment.
- Animation-strategy comparison.
- Recommended minimal architecture changes.

### B. Working code

- Updated Character Studio source code.
- Rebuilt Character Studio notebook.
- Working reference-image inputs.
- Candidate/contact-sheet generation.
- Reference-based variant generation.
- Background/asset extraction.
- Versioned asset importing.
- Character preview generation.
- Approval workflow.
- Existing renderer integration.

### C. Visual assets

When GPU execution and human approval are available:

- Arjun portrait and master.
- Meera portrait and master.
- Reference-conditioned variants.
- Transparent assets.
- Pose and expression previews.
- Short animation preview.
- Sample two-character scene.

### D. Tests

- Model registry/license tests.
- Asset schema tests.
- Manifest-hash tests.
- Approval tests.
- Resume/idempotency tests.
- Character reuse tests.
- Legacy rendering regression tests.
- Mouth/animation integration tests.
- Normal-render-without-image-model test.

### E. Benchmark report

Clearly distinguish:

- Verified measurements.
- Official documented requirements.
- Engineering estimates.
- Untested assumptions.
- Pending Colab executions.

### F. Documentation

Provide explicit instructions for creating the characters once and using them in future videos.

---

# PART 17 — ACCEPTANCE CRITERIA

The project is successful only when all applicable criteria are met.

## Visual appearance

- Arjun has the approved cute, cinematic golden-hoodie appearance.
- Meera has the approved butterfly-haired appearance.
- Both have cohesive dimensional storybook styling.
- Their faces are substantially more refined and visually appealing than the current flat characters.
- Hair, eyes, clothing and skin shading retain the approved design.

## Identity consistency

- Expressions look like the same character.
- Alternate views preserve recognizable identity.
- Permanent accessories remain consistent.
- Character proportions remain stable.

## Animation

- Characters can speak, blink and gesture.
- Existing audio-driven visemes remain functional.
- Mouth shapes do not look pasted on.
- Hair and accessories do not become detached.
- Pose transitions do not visibly break the artwork.
- Visual quality remains acceptable inside an actual video, not just in still images.

## Architecture

- The existing JSON-plan renderer remains functional.
- `storybook_v1` works with approved assets.
- Legacy paths remain intact.
- Existing voice, story, diagram and QC systems remain operational.
- Character image generation is not required during normal video rendering.

## Storage

- Approved assets persist on Google Drive.
- Asset manifests and hashes verify correctly.
- Plans pin the intended version.
- Colab restarts can resume incomplete work.
- Repeated videos reuse the same characters.

## Cost and licensing

- No mandatory paid API.
- No mandatory paid GPU service.
- No mandatory subscription.
- No incompatible commercial license.
- No false assertion of unlimited Colab resources.

## Approval

- Human character approval is explicit.
- Unapproved artwork cannot accidentally enter an approved video.
- Missing assets do not silently fall back to unattractive cartoons.
- Mandatory QC failures block approval.

---

# PART 18 — IMPORTANT FAILURE CONDITIONS

Do not report success when:

- Only pretty portraits were generated, without testing animation compatibility.
- The generated characters are inconsistent between views.
- The model requires more memory than the available environment without a working mitigation.
- Images are saved but cannot be loaded by the existing renderer.
- The system regenerates the same character for every new video.
- The new character format breaks existing JSON plans.
- Animation quality is substantially worse than the static portrait quality.
- Non-commercial or unverified model dependencies are silently introduced.
- Human approval is skipped.
- GPU benchmarks are invented.
- A Colab failure is hidden behind a silent procedural-art fallback.
- The implementation only works by using a paid GPU or hosted service.

If a limitation remains, document it explicitly and provide the best viable solution under the constraints.

---

# PART 19 — FINAL REPORT FORMAT

At the end, provide a concise engineering report containing:

1. **What existed:** architecture and limitations found.
2. **What was changed:** exact modules and behavior.
3. **Why the chosen model:** quality, license, hardware suitability.
4. **Visual evidence:** actual generated samples and comparisons, when available.
5. **Animation evidence:** actual pose, expression and video previews.
6. **Benchmark results:** measured GPU, RAM, time and storage.
7. **Tests:** commands and pass/fail results.
8. **Reuse proof:** evidence that normal videos do not call the character image generator.
9. **Remaining limitations:** honest technical gaps.
10. **How I run it:** exact commands, notebooks and approval steps.

Do not replace evidence with statements such as "should work" or "production-ready."

If the coding environment cannot run the Colab GPU workflow, complete the implementation and available tests, provide the notebook, and explicitly report the remaining GPU verification steps.

---

# FINAL INSTRUCTION

Your job is not to build an entirely new AI video platform.

Your job is to transform the existing provisional Arjun and Meera cartoon assets into **exceptionally cute, cinematic-quality, consistent, reusable character artwork**, while retaining the reliable automation system already built.

First audit the code.

Then research and select the most practical free/open-weight model.

Next build a **small, real visual proof of concept**.

Require my approval of the character design before expanding the asset library.

Validate that the artwork can be animated without losing its beauty.

Then implement reusable versioned assets, integrate with the existing Storybook renderer, and test the complete pipeline.

Preserve current code and functionality wherever possible.

**Generate once. Approve once per version. Reuse across future videos.**

The highest priority is that viewers should immediately find Arjun and Meera adorable, expressive, visually polished and memorable—not just that the implementation passes technical tests.

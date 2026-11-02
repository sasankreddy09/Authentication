# MASTER IMPLEMENTATION PROMPT
## Cinematic 3D-Style Character Upgrade — ChatGPT Codex Cloud + GitHub + Google Colab Free

### YOUR ROLE

Act as a senior AI research engineer, computer graphics engineer, character-animation developer, Python architect, Google Colab optimization specialist, and GitHub software engineer.

You are working inside **ChatGPT Codex Cloud**, connected to my existing GitHub repository.

Your assignment is to upgrade ONLY the existing cinematic 3D-style character-generation, asset-preparation, animation and rendering components of my AI educational video automation project.

The final result must produce Arjun and Meera with the same visual style and closely matching appearance as the two reference portraits attached to this task.

**Do not create an entirely new project. Do not redesign unrelated systems. Do not modify working functionality outside the cinematic character pipeline.**

Your final code changes must be committed and pushed to a dedicated GitHub feature branch, with a pull request for review whenever the connected environment supports it.

---

# 1. ABSOLUTE PRIORITIES

Follow these priorities in order:

1. Closely reproduce the artistic quality, character proportions, facial appeal, hairstyles, clothing, materials and lighting of my two supplied reference images.
2. Preserve character identity across poses, expressions, camera views and video scenes.
3. Animate these characters inside my actual video-generation pipeline.
4. Preserve the cinematic, dimensional appearance during speech and movement.
5. Use free, open-source libraries and commercially compatible open-weight models.
6. Make Google Colab Free the primary GPU environment, with an NVIDIA T4 as the preferred target when available.
7. Use ChatGPT Codex Cloud for source-code analysis, implementation, testing and GitHub integration.
8. Generate approved characters once and reuse their assets across future videos.
9. Update ONLY code related to cinematic character creation, rigging, rendering, asset loading and directly related tests.
10. Push the implementation to GitHub without modifying the main branch directly.

Visual quality in the actual MP4 is more important than producing hundreds of image assets or passing superficial tests.

---

# 2. REFERENCE IMAGES — HIGHEST-PRIORITY REQUIREMENT

I have attached TWO new visual references.

**These two images override the older Arjun and Meera character-design descriptions in `claude.md`.**

The earlier document described a golden-hoodie boy and a butterfly-haired girl.

Those older costumes and hairstyles are no longer the desired default design.

The new images are authoritative.

Do not silently continue using the old wardrobe or character designs.

## 2.1 Arjun — exact reference direction

Use the attached BOY portrait as the primary appearance reference.

Target features:

- An exceptionally adorable stylized animated boy.
- Approximately 8–11 years old in appearance.
- Large, warm, expressive brown eyes.
- Beautifully sculpted, rounded facial features.
- Soft naturally blushed cheeks.
- Small rounded nose.
- Gentle, charming smile.
- Short, fluffy, slightly messy medium-brown hair.
- Natural individual hair tufts with beautiful lighting.
- Expressive eyebrows.
- Rich mustard-orange short-sleeved T-shirt.
- Deep blue denim dungarees/overalls.
- Visible denim stitching, straps, buckles and front pocket.
- Rounded, appealing, stylized child proportions.
- Warm cinematic sunlight.
- Soft realistic shadows and detailed material shading.

Preserve the hairstyle, face proportions, eye appearance, clothing combination and color palette.

Do not replace the overalls with the old golden hoodie.

Do not produce generic anime, flat-vector or emoji-like characters.

The reference has a premium animated-feature appearance with naturally dimensional facial modeling.

Match that visual language as closely as the selected tools can achieve.

## 2.2 Meera — exact reference direction

Use the attached GIRL portrait as the primary appearance reference.

Target features:

- An exceptionally adorable stylized animated girl.
- Approximately 8–11 years old in appearance.
- Large luminous warm-brown eyes.
- Rounded, softly sculpted face.
- Naturally rosy cheeks.
- Small delicate nose.
- Gentle, expressive smile.
- Beautiful natural eyelashes.
- Long, voluminous auburn/chestnut-brown hair.
- A high or half-up ponytail secured with a small turquoise/teal hair tie.
- Soft bangs and gently curved loose strands around the face.
- Flowing, naturally shaded hair.
- Ivory or pale-cream short-sleeved T-shirt.
- Beautiful small pink and coral floral print on the front.
- Soft natural fabric folds.
- Warm indoor sunlight.
- Premium dimensional skin and hair shading.

The reference girl has an especially appealing large-eyed animated-feature appearance.

Preserve her recognizable face, ponytail, hair color and floral T-shirt.

Do not substitute the older butterfly-hair character design.

## 2.3 Shared visual quality requirements

Both characters must look as though they belong to the same premium cinematic animated film.

Required artistic properties:

- High-quality stylized 3D-rendered appearance
- Dimensional head and body modeling
- Soft subsurface-scattering-like skin
- Natural skin gradients
- Detailed hair strands and volume
- Soft global-illumination-like lighting
- Beautiful eyes with multiple subtle reflections
- Physically believable fabric textures
- Warm natural highlights
- Soft contact shadows
- Cohesive color grading
- Expressive eyebrows and facial features
- Attractive childlike proportions
- Refined, non-plastic material appearance
- Soft depth of field in portrait scenes
- Consistent character identity

Avoid:

- Flat vector cartoons
- Basic circles, lines and rectangles used to construct the character appearance
- SVG-like facial features
- Stiff puppets
- Paper-cutout appearances
- Oversimplified clothing
- Generic emoji expressions
- Excessively glossy plastic skin
- Inconsistent eyes
- Inconsistent hairstyles
- Different faces between images
- Unnatural anatomy
- Poor hands or extra fingers
- Broken shadows
- Inconsistent lighting
- Blurry faces
- Low-resolution output
- Unwanted accessories

The attached reference images are the visual source of truth.

If the text instructions conflict with the visible reference appearance, prioritize the reference images and report the difference.

Use the supplied images directly as conditioning inputs when the chosen model supports that capability.

Do not assume that files attached to one ChatGPT conversation automatically exist in the Codex Cloud working directory. Verify their availability. If needed, expose a documented reference-image input path in the Colab notebook and use the actual attached images there.

Do not silently replace missing reference images with unrelated sample images.

---

# 3. CHATGPT CODEX CLOUD ARCHITECTURE

The system must use the following division of responsibilities.

## 3.1 ChatGPT Codex Cloud — development environment

Use ChatGPT Codex Cloud to:

- Access the GitHub repository selected in the cloud environment.
- Inspect the actual Python implementation.
- Understand the current rendering pipeline.
- Diagnose the existing flat character-generation problem.
- Research appropriate open-source libraries.
- Implement focused code changes.
- Add asset-generation and animation adapters.
- Prepare Google Colab notebooks.
- Run available CPU-based tests.
- Validate compatibility with the existing project.
- Produce technical documentation.
- Commit and push code to GitHub.
- Prepare a pull request.

Codex Cloud is primarily the control and development environment.

It must not assume that it has access to an NVIDIA GPU.

If the cloud environment has only CPU resources, continue with code implementation and CPU tests.

Do not falsely report successful GPU image generation.

## 3.2 Google Colab Free — GPU image-generation environment

Google Colab Free must handle the computationally expensive character-artwork stages when compatible GPU hardware is allocated.

These include:

- Cinematic character generation
- Reference-conditioned image editing
- Character-view generation
- Complex foreground extraction when needed
- GPU-dependent asset-preparation steps

Do not assume that Colab Free resources are guaranteed.

The workflow must remain usable when GPU availability changes.

## 3.3 Existing CPU video-rendering pipeline

Continue using the project's existing Pillow and FFmpeg rendering infrastructure.

After a cinematic character has been approved and its animation assets have been generated, ordinary video rendering should not require the character image-generation model.

## 3.4 GitHub — source control

GitHub stores:

- Character-generation implementation
- Character-animation implementation
- Character-asset loader
- Compatible Storybook renderer updates
- Related tests
- Colab notebooks
- Minimal technical documentation

Do not commit large model weights, Hugging Face caches, temporary diffusion files, sensitive credentials or unnecessary generated media.

## 3.5 Google Drive — persistent asset storage

Use the existing project asset library and Drive checkpointing conventions.

Store:

- Approved canonical portraits
- Production-ready character masters
- Approved animation poses
- Facial expression assets
- Mouth and blinking assets
- Character manifests
- Model revisions and seeds
- Asset hashes
- Validation outputs
- Resumable generation checkpoints

Reuse the same approved character versions across videos.

---

# 4. STRICT CODE MODIFICATION BOUNDARY

This requirement is extremely important.

**Only update the cinematic 3D-style character-related components.**

Inspect these existing modules first, if present:

- `colab/engine/lib/storybook_art.py`
- `colab/engine/lib/storybook.py`
- `colab/engine/lib/storybook_scene.py`
- `colab/engine/lib/studio.py`
- `colab/engine/lib/charanim.py`
- `colab/engine/lib/envfactory.py`
- `colab/engine/lib/registry.py`
- `colab/build_studio.py`
- `colab/approve_character.py`
- `colab/storybook_previews.py`
- Character-specific renderer interfaces
- Character-specific tests

These paths are inspection targets, not permission to rewrite each file.

Modify only the files actually necessary for the character upgrade.

A minimal integration change in `drawkit.py` or another renderer entry point is allowed ONLY if required to connect the new approved-character assets to the existing pipeline.

Do not make unrelated changes to:

- Story-generation logic
- Research agents
- Factual validation
- General planning
- Narration generation
- Existing TTS engines
- Telugu or English language routing
- General subtitles
- Music generation
- YouTube publishing
- Authentication
- Database or storage systems
- Unrelated educational demonstrations
- General UI features
- Existing unrelated AI video workflows

Preserve old functionality.

Add new character-specific tests and documentation only where necessary.

Before committing, produce an exact changed-file list.

Inspect the Git diff and remove unrelated formatting changes, dependency upgrades and refactoring.

**Do not modify files merely because they could be improved. This task has a strict scope.**

If a required fix genuinely cannot be isolated to the character pipeline, explain the dependency and implement only the smallest compatible change.

---

# 5. ROOT-CAUSE ANALYSIS BEFORE IMPLEMENTATION

Inspect the actual repository and compare it with the supplied architecture documents.

Trace:

1. Story JSON and production plan
2. Character selection
3. Storybook rendering mode
4. Asset-version resolution
5. Character master loading
6. Pose and expression selection
7. Facial animation
8. Character compositing
9. FFmpeg input frames
10. Final MP4 output

Determine exactly why the current video shows flat characters.

Potential causes include:

- The old `charanim.py` renderer remains active.
- The Storybook renderer is using provisional geometric art.
- AI-generated character portraits are never converted into animation assets.
- The final renderer never loads the AI-generated images.
- Approved character versions are not selected.
- Cached video frames contain old characters.
- Image segmentation produces unusable layers.
- Character pose assets are missing.
- A GPU failure triggers a procedural fallback.
- Existing manifests refer to older artwork.
- Notebook generation embeds outdated assets.

Do not assume any single cause without verifying the source code.

For each confirmed problem, identify the exact file, function, current behavior, required fix and test.

The diagnosis must distinguish the image-generation stage from the animation-rendering stage.

A beautiful PNG existing on Google Drive does not prove that the final video uses it.

---

# 6. OPEN-SOURCE IMAGE GENERATION

Investigate the most practical high-quality open-weight model that can reproduce the desired visual aesthetic while remaining compatible with Colab Free.

## Primary model candidate

**FLUX.2 Klein 4B — distilled model**

Official references:

- https://github.com/black-forest-labs/flux2
- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B

Reasons to evaluate it:

- Apache-2.0 licensed model weights
- Text-to-image support
- Reference-conditioned image editing
- Multi-reference editing
- Compact relative to larger image models
- Suitable starting point for original cinematic character creation

Its published VRAM figures are not a guarantee that every workflow will fit on an NVIDIA T4.

Test the actual memory consumption in the allocated Colab environment.

## Alternative candidates

Evaluate suitable permissively licensed alternatives only when necessary.

Z-Image-Turbo may be evaluated for image generation:

https://huggingface.co/Tongyi-MAI/Z-Image-Turbo

However, distinguish standalone text-to-image generation from identity-preserving reference editing.

Do not assume all models support equivalent conditioning.

Do not introduce commercially restricted models.

Do not use the non-commercial FLUX 9B or dev variants.

## Model-selection criteria

Evaluate:

- Match to the reference images
- Facial quality
- Eye and hair detail
- Fabric quality
- Consistent identity
- Reference-editing capability
- Ease of generating alternate poses
- Compatibility with Colab T4
- Peak VRAM
- Peak system RAM
- Model download size
- Generation time
- License
- Commercial-use compatibility
- Compatibility with the existing Python project

Select the model based on actual proof-of-concept results.

Do not optimize only for speed.

If a model generates attractive but inconsistent characters, it is not suitable as the final production asset generator.

---

# 7. GOOGLE COLAB FREE OPTIMIZATION

The implementation must work within the limitations of Google Colab Free wherever technically feasible.

## Runtime preflight

Detect:

- GPU availability
- GPU name
- CUDA compute capability
- Available GPU memory
- System RAM
- Temporary disk
- PyTorch version
- CUDA version
- Image-model dependencies
- Existing generated assets
- Existing Drive checkpoints

The target GPU is an NVIDIA T4 with approximately 16 GB VRAM when assigned.

Do not assume native BF16 support on a T4.

Evaluate validated FP16, supported quantization and offloading strategies.

Do not blindly enable newer GPU kernels or FP8 compute paths that are unsupported on T4.

## Memory optimization

Research and test:

- Supported FP16 execution
- Quantized model-weight loading
- Attention memory optimizations supported by the current GPU
- CPU offloading
- Sequential model-component unloading
- Modest initial image resolutions
- Limited reference-image counts
- Separate image-generation and segmentation phases
- Explicit cleanup of unused GPU resources
- Efficient model caching

Use the existing project's dependency versions when possible.

Do not upgrade the entire environment unnecessarily.

Start with a small image-generation test.

Only increase resolution after recording actual memory usage.

## Out-of-memory handling

When an OOM error occurs:

1. Preserve completed assets.
2. Record the failure and memory statistics.
3. Unload the model.
4. Retry only with validated lower-memory settings.
5. Reduce dimensions if necessary.
6. Report failure if the model cannot be run reliably.

Never silently replace the desired cinematic artwork with flat cartoons.

## Colab policy compliance

Use an interactive Colab notebook.

Do not use artificial keep-alive scripts, prohibited unattended workers, SSH access or workarounds intended to bypass Colab limits.

Codex Cloud must prepare the notebook and code, but it must not pretend that a separate Colab Free GPU session is automatically available.

If interactive Colab execution is required, provide an exact run sequence and preserve all other completed work.

---

# 8. GENERATE THE CANONICAL CHARACTER ARTWORK

This phase has the highest visual-quality priority.

Use the two uploaded portraits as identity and artistic references.

The objective is to create production-ready cinematic assets that closely resemble the supplied characters.

Do not create completely different people merely because they also look like 3D cartoons.

## Generation sequence

For each character:

1. Load the supplied reference image.
2. Establish a stable canonical identity.
3. Produce a high-quality reference-conditioned master.
4. Produce a complete full-body production master.
5. Generate one alternate expression.
6. Generate one additional camera view.
7. Compare all results for consistency.
8. Save prompts, seeds and settings.
9. Present the output for visual approval.

Separate beautiful portrait artwork from animation-ready character masters.

A portrait with cinematic depth of field is useful as an aesthetic target, but a complete animation master needs visible anatomy and controlled framing.

Do not build a full library until a usable animation prototype proves the artwork can retain its quality during motion.

## Character consistency

Preserve:

- Facial structure
- Eye color and shape
- Hair color and silhouette
- Head-to-body proportions
- Skin appearance
- Permanent outfit
- Clothing material
- Light direction
- Artistic style

Do not generate every pose from independent text prompts.

Use approved reference-conditioned editing whenever supported.

If an alternate view does not retain identity, reject it.

## Human approval

Present both characters in a contact sheet alongside their original supplied references.

Require approval of facial appearance, clothing, hairstyles, dimensionality and artistic consistency.

Do not automatically approve using image similarity scores alone.

If image generation is unavailable, build and validate the notebook but mark the visual generation result as untested.

---

# 9. ANIMATION — DO NOT DESTROY THE ARTWORK

This is the most important engineering challenge.

A cinematic-looking image is not automatically an animatable 3D model.

Do not assume that cropping an arm or head and rotating it produces natural animation.

Your goal is to preserve the original rendered appearance through the actual video frames.

Compare these strategies.

## Strategy A — Layered 2.5D rig

Use high-resolution character artwork with separated body, head, facial and accessory layers.

Advantages:

- Reusable motion
- Low CPU rendering cost
- Compatibility with existing animation controls

Risks:

- Broken shoulders
- Disconnected joints
- Incorrect shadows
- Flattened hair
- Visible paper-puppet artifacts

## Strategy B — Pre-rendered cinematic pose atlas

Generate a small approved set of complete character poses and expressions.

Animate using:

- Pose switching
- Short natural transitions
- Breathing
- Subtle head movement
- Eye blinking
- Eye direction
- Facial expressions
- Lip synchronization
- Light body movement

This approach may preserve dimensional appearance better because complete poses retain their original shading.

## Strategy C — Hybrid character rig

Combine approved complete body poses with independently animatable facial assets.

Use pre-rendered body poses and carefully controlled mouth, eye and expression changes.

**Prefer evaluating this hybrid approach first for the actual two-friend conversational scenes.**

Do not select it merely by assumption; compare visual test results.

If the existing renderer cannot support the required animation without damaging the artwork, introduce the smallest compatible adapter.

Do not implement an entirely separate video engine.

## Expected animation

The characters should:

- Blink naturally
- Speak
- Move their mouths in sync with their voices
- Smile
- Look surprised
- Think
- React to each other
- Point
- Open their hands
- Make small natural head movements
- Move their bodies subtly
- Maintain realistic-looking shadows and clothing
- Keep hair and accessories connected

Do not promise unrestricted three-dimensional body rotation from flat image layers.

Use approved alternate-view assets for camera-angle changes.

If full 3D mesh animation becomes genuinely necessary, report it as a separately scoped architectural option rather than silently adding Blender or a large 3D reconstruction system.

---

# 10. OPEN-SOURCE CHARACTER SEGMENTATION AND LIP SYNC

## Character segmentation

Evaluate BiRefNet:

- https://github.com/ZhengPeng7/BiRefNet

Use a checkpoint whose weight license and dependencies are verified.

Purpose:

- Character foreground extraction
- Hair-edge preservation
- Transparent production assets

Optionally evaluate SAM 2 for specific part segmentation:

- https://github.com/facebookresearch/sam2

Do not assume that foreground extraction provides a complete animation rig.

Correct occluded anatomy and damaged boundaries with appropriate asset preparation or reference-conditioned inpainting.

Keep segmentation optional when manually prepared or already approved assets are sufficient.

## Lip synchronization

Evaluate Rhubarb Lip Sync:

- https://github.com/DanielSWolf/rhubarb-lip-sync

Use it as an optional CPU-based lip-cue generator.

Map its output to the existing Storybook mouth-control interface.

Preserve the current audio-derived viseme system.

For English dialogue, test alignment and mouth appearance.

For Telugu, evaluate Rhubarb's language-independent phonetic recognizer but verify the output carefully.

Do not assume English-specific recognizers work for Telugu.

Do not introduce commercially restricted facial-animation model weights.

The new lip synchronization must not look like flat geometric mouths pasted onto beautifully rendered faces.

Use approved mouth-artwork variations with consistent lighting, color and facial contours.

---

# 11. PRESERVE THE EXISTING STORYBOOK SYSTEM

Your current implementation already includes:

- Character asset versions
- Existing poses
- Existing expressions
- Mouth shapes
- Blink controls
- Voice-driven animation timing
- Storybook environments
- Pillow compositing
- FFmpeg rendering
- JSON production plans
- Human approval
- QC
- Drive caching

Do not replace these systems unnecessarily.

Implement a backward-compatible character-renderer improvement.

If appropriate, introduce an explicit cinematic asset/backend version so that the existing flat renderer remains available for legacy plans.

Do not change the semantics of existing plan fields without migration tests.

Preserve:

- `visual_style_version`
- `character_assets`
- Manifest hashes
- Character IDs
- Pose and expression IDs
- Existing narration timing
- Existing environment references
- Existing scene and shot schemas

Use the real implementation rather than assuming illustrative schemas in the documents are exact.

New cinematic character plans must select the approved AI-artwork renderer.

Legacy plans must remain unchanged.

If cinematic artwork is unavailable, report an error.

Do not fall back silently.

---

# 12. GOOGLE DRIVE CHARACTER ASSET LIBRARY

Maintain the existing `STORYBOOK_ASSETS` storage conventions.

Use immutable versions, for example:

- Arjun: current provisional version → new approved cinematic version
- Meera: current provisional version → new approved cinematic version

Each character asset version should contain the approved canonical design, production masters, necessary views, poses, facial variants, animation metadata and previews.

The manifest should record:

- Character identity
- Asset version
- Approval status
- Source reference
- Model identifier
- Model revision
- Generation prompt
- Seed
- Image dimensions
- Animation compatibility
- File paths
- SHA-256 hashes
- Character fingerprint
- License and provenance
- Approval metadata

Preserve existing manifest semantics.

Use staging directories during generation.

Do not publish incomplete assets as approved.

Avoid re-generating existing approved assets.

A normal video build must resolve a specific approved character version and verify its content hashes.

## Reuse requirement

If I generate 100 educational videos, the pipeline must reuse the same approved Arjun and Meera character identities without running the image-generation model 100 times.

Normal rendering should only load, animate and composite the approved character assets.

---

# 13. REAL END-TO-END VIDEO TEST

Do not declare success after generating images.

Generate a short real video through the existing video-rendering system.

The sample must include:

- Arjun and Meera in the same scene
- The newly approved cinematic character artwork
- Arjun speaking
- Meera listening
- Natural blinking
- Mouth movement
- A facial reaction
- One pointing or hand gesture
- A camera change or close-up
- Existing environment artwork
- Existing narration audio
- Existing subtitles

Use the current production pipeline to encode the MP4.

Extract representative frames from the final encoded MP4.

Compare the frames with the approved reference designs.

Check:

- Does Arjun retain the fluffy brown hair and denim overalls?
- Does Meera retain the auburn ponytail and floral shirt?
- Do both retain the desired cinematic facial appearance?
- Are the eyes still dimensional?
- Does animation preserve the hair and clothing shading?
- Does lip synchronization look natural?
- Are arm movements visually convincing?
- Is the scene genuinely using the approved character assets?
- Did any old geometric character asset accidentally enter the video?

If the MP4 contains old flat characters, mark the implementation FAILED.

## Reuse verification

Render a second independent short scene.

Verify that it uses the same approved character versions and does not call the image-generation model.

---

# 14. REQUIRED AUTOMATED TESTS

Implement focused tests only for the modified cinematic character components.

Required coverage:

- Character asset loading
- Approval validation
- Manifest hash validation
- Character identity and version selection
- Correct cinematic renderer selection
- Legacy-renderer isolation
- No-silent-fallback behavior
- Pose selection
- Expression selection
- Mouth-cue mapping
- Blink animation
- Character compositing
- Correct video-frame inputs
- Cache invalidation after asset updates
- Asset reuse without image regeneration
- Backward compatibility

Run existing related tests before and after the code changes.

Do not modify unrelated test suites.

Document actual results.

Distinguish CPU-tested behavior from GPU-dependent tasks not executed.

---

# 15. STRICT GITHUB IMPLEMENTATION WORKFLOW

Use the connected GitHub repository selected for this Codex Cloud environment.

Do not guess a repository name and do not change unrelated repositories.

## Before changes

1. Confirm the repository and default branch.
2. Read repository-specific development instructions.
3. Inspect the current Git status.
4. Record the starting commit SHA.
5. Create a dedicated feature branch.

Suggested branch:

`feat/cinematic-storybook-characters-v2`

Do not commit directly to `main` or `master`.

## During implementation

Make minimal focused commits.

Suggested commit categories:

- Character-renderer audit and targeted fix
- Reference-conditioned Character Studio support
- Cinematic animation adapter
- Character-asset and manifest integration
- Focused tests and notebook documentation

Do not commit model weights.

Do not commit private Google Drive files, access tokens, generated caches or large binary model files.

Use `.gitignore` rules for character-related temporary outputs only if necessary.

Avoid modifying unrelated repository configuration.

## Before pushing

Run:

- Relevant unit tests
- Available integration tests
- Static/import checks
- Git diff inspection
- Changed-file scope validation
- Secret detection where available

Confirm that only cinematic character-related files were changed.

Restore unrelated modifications.

Preserve pre-existing user modifications rather than overwriting them.

## Push changes

After reviewing the focused diff and committing successfully, push the feature branch to the connected GitHub repository using the credentials provided by the cloud environment.

Do not force-push.

Do not merge into the default branch automatically.

Open a pull request if repository permissions and the Codex Cloud workflow permit it.

Suggested PR title:

**Upgrade Storybook Characters to Cinematic 3D-Style Reusable Animation Assets**

Include:

- Root-cause findings
- Exact changed files
- Implemented architecture
- Open-source dependencies and licenses
- Colab Free execution requirements
- CPU test results
- Real GPU test status
- Animation preview status
- Known limitations
- Steps required for human visual approval

If pushing fails because GitHub permissions or authentication are unavailable, preserve the changes and provide the exact failure, branch name and commands required to finish the push.

Never claim that code has been pushed if no remote push succeeded.

---

# 16. IMPLEMENTATION PHASES

Execute in this sequence.

### Phase 1 — Inspect and diagnose

Read the repository, previous Markdown documentation and current character-rendering implementation.

Find the exact source of the flat character appearance.

### Phase 2 — Minimal rendering integration

Demonstrate that the existing scene renderer can load a high-quality character image and preserve it in actual video frames.

Do not expand the complete asset library yet.

### Phase 3 — Character Studio upgrade

Implement reference-image inputs, model preflight, low-memory generation, consistency checks, Drive checkpointing and human approval.

Prepare the Google Colab Free notebook.

### Phase 4 — Cinematic animation prototype

Implement and test a minimal 2.5D or hybrid animation approach.

Preserve image quality.

### Phase 5 — Visual and animation approval

Present the generated masters, animation preview and frames from the actual renderer.

Do not assume approval.

### Phase 6 — Complete approved reusable assets

After visual approval, build the necessary poses, expressions, mouth variants and camera views.

Avoid unnecessary combinatorial image generation.

### Phase 7 — End-to-end pipeline validation

Render a complete test conversation through the existing production workflow.

Verify asset use, animations, audio and video output.

### Phase 8 — GitHub delivery

Commit and push the scoped implementation to a feature branch.

Open the pull request.

Clearly identify any unverified Colab GPU or visual-approval steps.

---

# 17. DEFINITION OF SUCCESS

The implementation is successful only when the following are genuinely verified:

**Visual appearance**

Arjun closely matches the attached boy reference in face, hairstyle, eyes, clothing, proportions and cinematic appearance.

Meera closely matches the attached girl reference in face, hairstyle, eyes, clothing, proportions and cinematic appearance.

**Animation**

The characters blink, talk, react and gesture while retaining the beauty of their approved designs.

**Video integration**

The real final MP4 contains the approved cinematic characters rather than the previous procedural cartoons.

**Consistency**

Faces, hair, clothing and character proportions remain recognizable across shots.

**Resource efficiency**

One-time character generation runs through a free/open-weight model on compatible Colab Free resources when available.

Normal cinematic character rendering does not need a character-generation GPU.

**Architecture**

Existing story, dialogue, narration, educational demonstrations, subtitles, QC and publishing remain functional.

**Scope**

Only 3D-style character-related implementation, essential integration points, tests and documentation have changed.

**GitHub**

Code is committed and pushed to a dedicated feature branch, with a reviewable PR when permissions allow.

---

# 18. REQUIRED FINAL ENGINEERING REPORT

When finished, report:

1. The actual root cause of the flat characters.
2. The files and functions responsible.
3. Which open-source image model was selected and why.
4. Verified model and dependency licenses.
5. Measured Colab GPU and memory requirements, if tested.
6. The animation strategy selected and why.
7. Exactly which files were changed.
8. Evidence that the production renderer uses the new artwork.
9. Actual screenshots from the final MP4, if rendered.
10. Tests run and their results.
11. Which requirements remain unverified.
12. How to run the Character Studio on Colab Free.
13. How to generate a normal video using approved character assets.
14. Git branch name, commit SHA and push status.
15. Pull request URL, if successfully created.

Do not fabricate benchmark measurements or successful test results.

Do not claim the project is visually complete if the actual animation has not been reviewed.

Do not claim a GitHub push succeeded without confirmation.

---

# FINAL MANDATORY INSTRUCTION

**I do not want simple cartoons. I do not want another procedural Bézier character system. I do not want beautiful static images that cannot be used in the final video.**

I want the characters in my generated educational videos to look as close as technically possible to my two attached reference portraits.

Their faces, eyes, hair, clothes, proportions, materials and cinematic lighting must remain attractive while they speak, blink and move.

Use:

- ChatGPT Codex Cloud for development and GitHub work.
- Free, open-source libraries and commercially compatible open-weight models.
- Google Colab Free for the computationally expensive artwork-generation stages.
- The existing Python/Pillow/FFmpeg pipeline for efficient reusable character animation.
- Google Drive for persistent approved assets.
- GitHub for version-controlled code delivery.

**Modify only the cinematic 3D-style character-related portions of my existing project. Push the focused changes to a dedicated GitHub branch. Preserve every unrelated working feature.**

The ultimate measure of success is not generated image count, model complexity or test count.

**It is whether Arjun and Meera look like the supplied premium cinematic animated characters in the actual final educational MP4.**

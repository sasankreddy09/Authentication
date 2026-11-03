# Cinematic character upgrade v2

The character tools now use the canonical Python engine in `colab/engine/lib/`. The repository has been reorganized for video automation and the website removed. Only the latest Telugu 3D video notebook uses the new cinematic assets; other notebooks retain their previous contents. See [Colab setup](../docs/GOOGLE_COLAB.md) and [current architecture](../docs/02_ARCHITECTURE.md).

## Status and evidence

CPU behavior is tested with explicitly synthetic texture fixtures. GPU image generation, T4 memory/timing, reference fidelity, natural facial animation and human approval are **pending**. Test movies are engineering fixtures, not finished Arjun/Meera artwork. No model weights or user reference images are committed.

The attached images were visible in chat but were not exposed as local image files. Upload the original images into Drive at the paths below. The boy image shows a red jacket, yellow striped shirt, teal trousers and sneakers; the written overalls description conflicts with that image. The supplied image takes precedence. The girl's auburn ponytail, teal tie and cream floral shirt replace the old butterfly design. Use the girl's dimensional materials/lighting as the shared cinematic target; retain the boy's visible identity and outfit.

## Root causes and targeted changes

| Evidence | Cause | Change / test |
|---|---|---|
| Sample Storybook plan pins procedural v1; archive manifests are provisional | New cinematic images were not selected by that plan | `cinematic_workflow.bind_plan` makes a separate pinned plan; new `cinematic_characters` flag requires approved `atlas_v1` assets |
| `storybook.load` returned cached objects before checking expected hashes | A different plan's pins could be bypassed after a load | Cache keys include pins; manifest and file stat changes cause verification/reload; tampering and wrong-pin tests |
| `studio.generate` used one size for both axes | Requested portrait dimensions were ignored | Forward actual width/height; tested with a fake pipeline |
| `studio_run.run_job` checked only saved output hash | Changed prompt/reference/settings could reuse stale candidates | Request fingerprint includes references, prompt, dimensions, seed, revision and backend; changed requests require a new candidate folder |
| Portrait and full-body jobs had independent prompts | Identity could drift between canonical views | Actual reference → portrait → conditioned full-body → conditioned alternate view; no missing-reference fallback |
| Automatic three-quarter atlas selection bypassed front-only patches | Speaking/blinking could disappear | Front view for conversation; explicit back view remains; per-view facial patches are a future extension |
| Source `engine/` and test tree are missing from this checkout | Original builders cannot reproduce notebooks from these paths | `bootstrap.py` extracts unchanged embedded dependencies and overlays only character modules |

The character sources and recovered production modules are reviewable under `colab/engine/lib/`. Studio-only code is excluded from normal render exports. The existing Character Studio notebook is preserved as a self-contained snapshot; rebuilding it is an explicit separate action.

## Models and licensing

- **Primary candidate:** `black-forest-labs/FLUX.2-klein-4B`, distilled, 4 steps. The [official FLUX repository](https://github.com/black-forest-labs/flux2) lists Apache-2.0 weights and single/multiple-reference editing for 4B. Keep the repository's pinned revision. The official README's consumer-GPU figures are not a measured T4 result. Model selection remains provisional until actual reference-conditioned output is reviewed.
- **Segmentation:** `ZhengPeng7/BiRefNet`, existing pinned revision. [Official source](https://github.com/ZhengPeng7/BiRefNet) is MIT; the existing model registry records MIT for the checkpoint. Direct Hugging Face model-card retrieval was blocked in this development environment, so the weight license was not independently re-fetched here. Check the pinned model card in Colab before use. Manually prepared transparent PNGs may replace segmentation. Do not substitute BRIA RMBG weights.
- **Runtime:** existing Pillow/numpy/FFmpeg composition. FFmpeg licensing depends on its build; these tools do not imply a new license grant for user references.
- **Optional lip cues:** `rhubarb_visemes` maps Rhubarb JSON to the existing mouth IDs. Run Rhubarb separately if wanted. Mapping is CPU-tested; real English/Telugu listening/visual calibration has not been repeated. The normal audio-derived visemes remain in use.
- **Memory strategy:** GPU preflight, T4 FP16 rather than unsupported native BF16, required 4-bit text encoder support, CPU offload, initial 640px-width candidates, and separate generation/segmentation. OOM unloads the generator and preserves completed files. No automatic retry with unmeasured settings and no procedural-art substitute.

The notebook installs only image-stage dependencies and preserves Colab's torch. Those package combinations and quantization APIs require execution on the allocated Colab runtime; CPU tests do not establish CUDA compatibility.

## CPU validation

From the repository root (Python with Pillow, numpy and Pydantic 2; FFmpeg/ffprobe on PATH):

```bash
python3 -m unittest discover -s cinematic/tests -v
python3 cinematic/build_notebook.py
```

Tests cover reference/approval contracts, dimensions, resume, hashes, identity/version selection, missing-asset rejection, pose/facial patches, prototype isolation, legacy opt-out, Rhubarb mapping, notebook structure, and two actual encoded scenes without image-model imports. The previously documented historical suites were absent; the new `colab/tests` suite checks the current Telugu target. Tests do not infer image quality from synthetic fixtures.

## Colab: generate a small proof of concept

1. Open `colab/notebooks/colab_cinematic_characters_v2.ipynb` from this branch in Colab, select an allocated T4 GPU and mount Drive.
2. Upload the actual attached images to:
   - `MyDrive/ai-video/references/arjun.png`
   - `MyDrive/ai-video/references/meera.png`
   Adjust `REFERENCES` if using different names or extensions. There is no downloaded sample replacement.
3. Run preflight and the small generation cell. It creates a reference/portrait/full-body/expression/view comparison under `_assets/studio/reference-v2-candidate-01/`.
4. Inspect identity, outfit, proportions, hands and shared material style. Girl lower-body details are not visible in the supplied portrait and require design review. Set `REVIEWER` and `DESIGN_APPROVED` only after actual review.
5. Run the minimal animation jobs and segmentation. Manually correct bad transparent edges and the estimated head/face boxes in each `atlas_src/spec.json`. Segmentation is not automatic rig creation.
6. Render a small prototype as below. Do not proceed to full generation until the actual MP4 has been reviewed.

## Render a prototype with existing speech and environments

In an interactive Colab cell, obtain the focused source (or upload a ZIP of this branch):

```python
!git clone --branch feat/cinematic-storybook-characters-v2 --single-branch https://github.com/sasankreddy09/Authentication.git /content/Authentication
```

Run the following command, replacing the two WAVs and their exact spoken text with existing narration outputs. WAVs must be mono PCM16 at the same sample rate. Convert copies with FFmpeg if necessary; do not alter existing originals.

```bash
python3 /content/Authentication/cinematic/preview.py \
  --candidate /content/drive/MyDrive/ai-video/_assets/studio/reference-v2-candidate-01 \
  --assets /content/drive/MyDrive/ai-video/_assets \
  --version cinematic-v2-prototype-01 \
  --audio /path/to/arjun-line.wav /path/to/meera-line.wav \
  --text 'Exact Arjun dialogue' 'Exact Meera dialogue' \
  --output /content/drive/MyDrive/ai-video/_assets/studio/reference-v2-candidate-01/preview
```

The prototype explicitly substitutes idle for ungenerated poses. It is provisional and cannot be approved as a production pack. Review `animation.mp4`, `frame-*.png`, subtitle text and `render-evidence.json`. Compare with the old layered-rig preview in `colab/reports/storybook/`. This uses the existing drawkit → character dispatch → Storybook/Pillow path and FFmpeg, with supplied existing audio; it is a character integration preview, not a rerun of all research/TTS/QC stages.

Copy the reviewed preview to `reference-v2-candidate-01/animation-review.mp4`. Only then set `ANIMATION_APPROVED` and run the full-atlas cells. Approval records bind master and preview hashes. A changed/deleted reviewed input invalidates approval. Keep that candidate directory on Drive.

## Import and approve the full pack

After full generation, segmentation and spec review, the notebook imports a NEW provisional `cinematic-v2` version. It never overwrites an existing version. Use a new version name for corrections.

Run `preview.py` again without `--candidate`, using `--version cinematic-v2`, to render the imported full pack. Also run it with `--closeup` into a separate output directory. Inspect mouth alignment, blinking, expression change, hand gesture, hair, clothing and shadows. Inspect all remaining poses/expressions in `atlas_src` before approval. Copy the actual reviewed `animation.mp4` into each imported version's `previews/animation.mp4`.

Set `PACK_APPROVED` only after that review and run the final notebook cell. It records the approver in each manifest. Approved plans must be created AFTER approval because the full manifest hash changes; the content fingerprint does not.

## Use approved characters in the existing complete production pipeline

From the checkout, with Drive assets available, build a NEW notebook:

```bash
python3 cinematic/deliver.py \
  --template colab/notebooks/colab_three_d_friends_te_v3_storybook.ipynb \
  --plan colab/plans/three_d_friends_te_v3_storybook_plan.json \
  --assets /content/drive/MyDrive/ai-video/_assets \
  --version cinematic-v2 \
  --output /content/colab_cinematic_video.ipynb
```

Only character modules and artwork pins are replaced. Story, voice, language, subtitles, music and other cell sources are preserved. Use a new project folder in the output notebook settings to avoid an older `plan.json` on Drive. A retained plan selecting different artwork stops with an explicit error. The existing drawn-shot cache includes `character_assets`, so new approved hashes produce new segments.

Run the resulting notebook interactively. Review final QC and extract frames from its final MP4. Render a second scene/video with the same pins. Do not declare the project visually complete until these actual outputs meet the reference and animation criteria. No full production run with approved new artwork has been performed during this change.

## Known limits

- Reference fidelity and real T4 feasibility are pending; no GPU timings are claimed.
- The atlas reuses whole poses and front-view face patches. It does not provide arbitrary 3D rotation, per-view talking patches, or separately animated eye-gaze patches. Close-ups change framing without pretending new geometry exists.
- Atlas pose crossfades can ghost; head/face estimates and patch alignment require visual correction. Human animation comparison decides whether this strategy is acceptable.
- The checked-in sample assets remain provisional. This upgrade cannot turn them into approved cinematic artwork without the generation and review steps.
- No full library is generated before design and prototype-animation approvals.

## Delivery verification (2026-11-01)

- `python3 -m unittest discover -s cinematic/tests -v`: **20 passed**, including two encoded fixture scenes with audio and subtitles, decoded frame extraction, and image-model imports disabled during rendering.
- Baseline versus overlay rendering: legacy character frames and provisional Storybook v1 frames were **byte-identical** for the checked two-character scene.
- All new Python sources and generated code cells parse/compile. Notebook rebuild is deterministic. Generated notebook contains no execution output, model data or attached images.
- Git whitespace check and a targeted credential-pattern scan passed. No dedicated secret-scanner executable was available; the pattern scan is not a comprehensive security audit.
- Existing historical test suites, GPU inference, real reference images, complete production-video QC and human visual acceptance remain unrun/pending as described above.

Exact changed-file list (all additions; existing tracked files preserved):

```text
cinematic/.gitignore
cinematic/README.md
cinematic/bootstrap.py
cinematic/build_notebook.py
cinematic/deliver.py
colab/engine/lib/cinematic_workflow.py
colab/engine/lib/storybook.py
colab/engine/lib/storybook_scene.py
colab/engine/lib/studio.py
colab/engine/lib/studio_run.py
cinematic/preview.py
cinematic/tests/test_characters.py
colab/notebooks/colab_cinematic_characters_v2.ipynb
```

Feature branch: `feat/cinematic-storybook-characters-v2`, based on `4f25ceb579b3a584362ba1d28dbc8c6f7558472e`.

## Current Telugu target

The updated `colab/notebooks/colab_three_d_friends_te_v3_storybook.ipynb` binds approved Drive packs at runtime. You do not need to run `deliver.py` for this target. Set `CHARACTER_VERSIONS` to the approved versions (default `cinematic-v2` for both), supply the cinema/lobby folders and run it as described in [the Drive guide](../docs/GOOGLE_COLAB.md). The full atlas must support front/back views, worn 3D glasses, held glasses, popcorn and the gestures the story uses; plan validation checks this before speech. The two-pose prototype is insufficient.

`deliver.py` remains an advanced export tool for making a separately pinned notebook from a template. The current video notebook, its default settings and the canonical source tree are maintained with `python colab/build_telugu.py`. Do not rebuild the unchanged Studio or historical video notebooks as part of that command.

## Inline image-to-video demo

The cinematic v2 notebook now generates a reference-conditioned `view_back.png`
for each character during the initial PoC. Rerun the image cell in the same candidate
to retain matching completed jobs and generate the new rear views. Changed inputs
still require a new candidate directory. Rear-view details remain inferred artwork.

Run **Quick video demo** immediately after image generation to encode front,
three-quarter and rear views into `image-motion-demo.mp4` on Drive and play it inline.
It uses Pillow and FFmpeg on CPU, H.264/yuv420p and faststart, without speech or a
second generative model. Optional download has its own cell. After reconnecting,
Drive settings and the demo cell suffice when the images and FFmpeg are available.
This is a camera-motion preview of still images, not articulated character animation;
it does not satisfy the separate production rig animation review.

The implementation follows FFmpeg rawvideo input and IPython Video embedding APIs.
CPU tests encode and decode synthetic inputs, verify frame count/codec and preserve
existing output on missing input. GPU inference and visual quality require Colab
validation with real references. Existing design/animation/pack review stops are
intentional; Run all still pauses at those production gates.

## Colab host-RAM recovery

The current cinematic notebook quantizes both the FLUX transformer and text encoder
with bitsandbytes NF4/double quantization, requests low-CPU-memory loading and enables
VAE tiling/slicing. LocalGen loads weights lazily so cached jobs do not load a model.
Segmentation cleanup now runs on both success and failure. These reduce memory
pressure; CPU offloading still retains weights in host RAM and no Colab RAM benchmark
has been performed here.

After a RAM crash, delete/reconnect the runtime and choose a T4 GPU. The new default
candidate is `reference-v2-lowram-01`; the backend fingerprint changed to prevent
silently mixing quantization configurations. Existing Drive candidates are preserved.
Run the PoC cell for `POC_CHARACTERS = ['arjun']`, then change it to `['meera']` and
rerun. Processing one character per run limits work, but the model's peak footprint
is reduced by quantization, not by the character count. Matching jobs resume.
If loading still exceeds available host RAM, use a High-RAM GPU allocation if available.
For already generated images, skip model cells and run the CPU video demo against
the original candidate path.

## Early-cell runtime disconnect investigation

Cell numbers vary when Colab counts the title; code cells now identify installation,
engine unpacking and lightweight preflight explicitly. Earlier setup launched a
second Python process importing the whole Diffusers/Transformers/bitsandbytes stack.
Preflight also used `hasattr(diffusers, 'Flux2KleinPipeline')`, which resolves a lazy
pipeline import rather than checking metadata. Both early imports are removed.
The old installer said it preserved torch but did not constrain transitive dependencies;
it now preserves installed torch, torchvision, torchaudio, NumPy, NVIDIA distributions
and Triton with pip constraints, and accepts binary wheels only. A resolver conflict
is reported rather than replacing the native stack.

These are confirmed code paths, not a confirmed explanation of the user's VM death.
The allocated Colab runtime and its termination logs are unavailable here. A killed
kernel, platform eviction, native crash and browser disconnect cannot be distinguished
from the reported symptom alone. Startup writes timestamped stage/memory snapshots
(including available host memory and cgroup counters when exposed) every five seconds
to `MyDrive/ai-video/_diagnostics/cinematic-v2/runtime-*.jsonl`. The final sample can be
missing on abrupt termination; absence of an OOM counter does not rule out OOM.
Send the latest log and the last cell output after a recurrence. Logs contain package
metadata/stages/memory, not credentials or image contents. Colab UI restart instructions
are manual; the notebook does not kill or restart its own kernel.

Validation: execute the generated engine cell in a fresh Python process with imports
of torch/Diffusers/Transformers/bitsandbytes explicitly forbidden; test lightweight
preflight, durable diagnostics and native-version constraints. This validates startup
isolation, not Colab CUDA compatibility or model memory limits.

Research: pip's official constraints documentation explains that constraints limit
which dependency versions are eligible without forcing installation:
https://pip.pypa.io/en/stable/user_guide/#constraints-files
The official Colab FAQ returned HTTP 403 in this development environment and could
not be consulted live. No claim of a platform-specific root cause follows from it.

## Uploaded preview investigation

The supplied 22-second H.264 clip (640x360, 24 fps, 528 frames) matches the former
preview helper: six seconds of independent expression/mouth/blink cycling followed
by one second per atlas pose. This intentionally enumerated assets but was not labeled
as a catalogue. It produced unexplained prop changes and silent mouth movement.
Pose-dependent head estimates also change scale; generated pose proportions are not
registered geometry. Contact-sheet inspection confirmed these transitions.

Default pack previews now hold one idle pose for eight seconds with subtle breathing;
no speech and no unverified blink overlay. This avoids pose-scale jumps rather than
claiming to repair independent pose geometry. Optional `use_blink=True` requires a
usable blink patch. `diagnostics=True` creates a separate labeled diagnostics.mp4,
with one effect at a time, and leaves animation.mp4 intact. Neither file demonstrates
natural locomotion. Whole-face expressions are exclusive with independent facial
patches; eye and mouth patches cannot stack when their declared boxes overlap.

Pillow's alpha compositing applies supplied pixels; it does not register faces or
resolve semantic conflicts: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.alpha_composite
Review of the local renderer confirms that explicit registration/selection is the
application's responsibility. Tests cover expression exclusivity, preview encoding,
and preservation of the default movie when rendering diagnostics. Actual user atlas
files were not supplied, so corrected rendering of that precise pack remains untested.

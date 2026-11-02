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

## Runtime reconnects: low-memory Studio revision

The original `cinematic-v2-05` path quantized only the text encoder; the approximately 4B-parameter transformer was still loaded at 16-bit precision (roughly 8 GB of weight data before other components/overheads), and CPU offload was enabled only after loading. This is a credible host-RAM/VRAM pressure risk on a free Colab machine, not a confirmed diagnosis of an individual disconnected session.

The updated Studio notebook requests NF4/double quantization for both `transformer` and `text_encoder`, `low_cpu_mem_usage=True`, model CPU offload and VAE tiling. It checks available host RAM and VRAM before loading and records stage/resource readings to `MyDrive/ai-video/_assets/studio/reference-v2-lowmem-01/memory-events.jsonl`. The 5 GiB host / 10 GiB GPU admission thresholds are conservative heuristics, not a measured guarantee. No full-precision fallback is attempted when quantization configuration fails. GPU loading/quality remain unverified on an actual T4; CPU contract tests cannot certify CUDA compatibility.

Open a fresh copy of the updated GitHub notebook: an existing Drive copy does not automatically receive repository changes. Save current outputs, disconnect/delete the old runtime, choose an available T4 GPU, then run cells individually. Use the new candidate directory so older generated images are not treated as outputs of the new quantization settings. Reviewed/approved packs remain immutable.

- Failure at **cell 01**: only Drive mounting/configuration runs there. Check GPU allocation, Drive authorization and browser/session connectivity; model memory is not yet involved.
- Last event **before_model_load**, followed by kernel reset: loading or a session interruption is implicated; host OOM is possible, not proven. Retain the last console lines, RAM graph and Colab runtime logs.
- Last event **weights_loaded/model_ready**: weights loaded; investigate offload setup or inference next. A CUDA OOM traceback identifies a different failure mode from a browser reconnect.
- Last event **before_poc_inference**: investigate inference peaks and reference-image workload. Smaller image dimensions may reduce inference memory but cannot solve an oversized model-load peak.
- No traceback with runtime variables/files lost: possible kernel termination or runtime reclamation. If variables survive reconnection, investigate the client connection rather than assuming a model crash.

The event file is a stage snapshot, not a continuous peak-memory profiler. Missing later entries cannot establish OOM by themselves. Do not repeatedly retry a hard crash; supply the runtime GPU type, last event and Colab system message. If this preset still cannot fit, prepare approved assets in a separately suitable runtime or import already-prepared packs; the ordinary Telugu video renderer does not load FLUX.

"""Build the character-only Colab workflow from preserved engine sources and overlays."""
import json
from pathlib import Path
from bootstrap import ROOT, sources


def build():
    cells = []
    def add(text, kind="code"):
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add("""# Reference-conditioned cinematic characters v2
Run interactively in Colab. Upload the actual boy and girl images to the paths below.
Boy reference: red jacket, yellow striped shirt, teal trousers. The overalls description conflicts with the image.
Girl reference: auburn ponytail, teal tie, cream floral shirt. Older hoodie/butterfly defaults are superseded.
This notebook stops for design and animation reviews. GPU inference and visual quality remain unverified until you run it.
Keep new candidate folders for changed inputs; approved assets are immutable.

### Startup diagnostics and RAM recovery
Code cells are labeled STEP 2 (installation), STEP 3 (CPU engine), STEP 4 (lightweight preflight).
None of these loads image-model weights. Persistent stage/memory logs start in the Drive cell under
`MyDrive/ai-video/_diagnostics/cinematic-v2/`. After a disconnect, keep the latest JSONL log.
It records stages and memory samples, not reference pixels or tokens. A last log entry alone does
not prove OOM: abrupt platform eviction, a native crash or lost connectivity can look similar.
Installation preserves the allocated Colab torch/torchvision/torchaudio/NumPy/CUDA versions and
uses binary wheels only. A dependency conflict is reported rather than replacing that stack.
The old subprocess import probe and preflight lazy-pipeline import have been removed.

### RAM recovery
Start with **Runtime > Disconnect and delete runtime**, reconnect to a T4 GPU, then run setup.
The low-memory loader quantizes both the transformer and text encoder to NF4, loads weights only
for uncached jobs, and uses VAE tiling/slicing. CPU offload still uses host RAM; this is not a
promise that every free Colab allocation can fit the model. If loading still exhausts RAM,
use a High-RAM GPU runtime if available. Keep the last cell output to identify the failing stage.
This version uses a new `reference-v2-lowram-01` candidate because quantization changed;
your previous Drive images remain untouched. For a video from already completed images only,
set STUDIO to your old candidate and run Quick video demo directly after Drive settings.
Do not rerun generation into that old candidate: its request fingerprints differ.

""", "markdown")
    add('''from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/ai-video')
ASSETS = ROOT / '_assets'
STUDIO = ASSETS / 'studio/reference-v2-lowram-01'
STUDIO.mkdir(parents=True, exist_ok=True)
REFERENCES = {'arjun': str(ROOT / 'references/arjun.png'), 'meera': str(ROOT / 'references/meera.png')}
REVIEWER = ''
DESIGN_APPROVED = False
ANIMATION_APPROVED = False
PACK_APPROVED = False
NEW_VERSION = 'cinematic-v2'
''' + '\n' + (ROOT / 'cinematic/runtime_support.py').read_text() + '''
if 'runtime_diagnostics' in globals():
    runtime_diagnostics.stop()
runtime_diagnostics = RuntimeDiagnostics(ROOT / '_diagnostics/cinematic-v2')
print('Persistent diagnostics:', runtime_diagnostics.path)
''')
    add('''# STEP 2: installation only; no model imports or weights are loaded here.
# On a fresh runtime run this once. Never modify packages under a loaded model.
if any(name in sys.modules for name in ('diffusers', 'transformers', 'torchao', 'torch')):
    raise RuntimeError('Restart the session before installation; model libraries are already loaded.')
runtime_diagnostics.note('installation: collecting installed package versions')
print(installed_versions())
constraints = Path('/content/cinematic-colab-constraints.txt')
print('Preserving Colab packages:', write_constraints(constraints))
runtime_diagnostics.note('installation: removing unused optional torchao')
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
packages = ['diffusers>=0.36,<0.42', 'transformers>=4.57,<6', 'accelerate>=1.2,<2',
            'bitsandbytes>=0.45,<1', 'safetensors', 'Pillow', 'psutil',
            'pydantic>=2.7,<3', 'einops', 'timm']
runtime_diagnostics.note('installation: resolving binary wheels with Colab constraints')
# Binary-only avoids large source builds. Dependency conflicts stop explicitly;
# do not remove constraints or replace Colab torch to force installation.
subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
                '--upgrade-strategy', 'only-if-needed', '--constraint', str(constraints),
                *packages], check=True)
if not shutil.which('ffmpeg'):
    runtime_diagnostics.note('installation: ffmpeg')
    subprocess.run(['apt-get', 'update', '-qq'], check=True)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'ffmpeg'], check=True)
# Metadata lookup does not import native libraries or duplicate the model stack.
print(installed_versions())
runtime_diagnostics.note('installation complete; no image pipeline imported')
''')
    add('''# STEP 3: unpack Python source and import CPU utilities only.
runtime_diagnostics.note('engine: unpacking source files')
import json, sys, os
WORK = Path('/content/cinematic-engine-v2'); WORK.mkdir(exist_ok=True)
MODULES = json.loads(''' + repr(json.dumps(sources(studio=True))) + ''')
for name, source in MODULES.items():
    (WORK / name).write_text(source)
del MODULES, source
runtime_diagnostics.note('engine: importing CPU utilities')
sys.path.insert(0, str(WORK))
os.environ['STORYBOOK_ASSETS'] = str(ASSETS)
import cinematic_workflow as CW, studio, studio_run, registry, storybook as SB, storybook_atlas as SA
from PIL import Image
runtime_diagnostics.note('engine ready; no model weights loaded')
''')
    add('''# STEP 4: metadata and nvidia-smi only; no torch/Diffusers import.
runtime_diagnostics.note('preflight: checking reference images and GPU metadata')
report = lightweight_preflight(REFERENCES, STUDIO / 'preflight.json')
print(report)
if not report.get('gpu'):
    raise RuntimeError('Select an allocated T4 GPU; generation has not run.')
for model in (studio.FLUX_MODEL, studio.SEG_MODEL):
    ok, why = registry.model_ok(model, strict=True)
    if not ok: raise RuntimeError(why)
runtime_diagnostics.note('preflight complete; next cell is the first model-loading stage')
''')
    add('''# Small PoC only. Process one character at a time to bound work per run.
# Change this to ['meera'] and rerun after Arjun finishes (or use both).
POC_CHARACTERS = ['arjun']
POC = STUDIO / 'poc'
runtime_diagnostics.note('generation: importing torch and constructing lazy generator')
gen = studio_run.LocalGen(registry.REVISIONS[studio.FLUX_MODEL], log=runtime_diagnostics.note)
try:
    results = studio_run.run_poc(gen, POC, characters=POC_CHARACTERS, style_refs={k: [v] for k,v in REFERENCES.items()})
finally:
    gen.unload()
    runtime_diagnostics.note('generation finished; weights released')
print(results)
if any(v not in ('made', 'skipped') for v in results.values()):
    raise RuntimeError('Some PoC jobs failed; review logs before continuing.')
items = []
for cid in REFERENCES:
    paths = [(cid + ' supplied reference', Path(REFERENCES[cid]))]
    paths += [(cid + ' ' + p.stem, p) for p in sorted((POC / cid).glob('*.png'))]
    for label, path in paths:
        with Image.open(path) as source:
            thumb = source.convert('RGB')
            thumb.thumbnail((320, 320))
            items.append((label, thumb))
try:
    studio.contact_sheet(items, STUDIO / 'reference-comparison.png', cols=5, cell=320)
finally:
    for _, thumb in items:
        thumb.close()
    del items
with Image.open(STUDIO / 'reference-comparison.png') as sheet:
    display(sheet)
print('Before the demo/design cells, run this cell for both arjun and meera.')
''')
    add("""## Quick video demo — run after the image cell
The initial image set now includes `view_back.png`, conditioned on each full-body master.
Unseen rear details are inferred and need visual review; a front image cannot specify them exactly.
The next cell makes a silent 12-second MP4 from front, three-quarter and rear images for both characters,
with gentle camera zoom, and plays it here. It needs no extra GPU model, audio, segmentation or approval.
This is a still-image motion demo, not walking, talking or continuous 3D rotation. It does not replace
`animation-review.mp4` or approve the production rig. To reuse existing images after reconnecting,
run the Drive settings cell and this demo cell (plus installation only if FFmpeg/Pillow are missing).

Implementation references: [FLUX.2 single-reference editing](https://github.com/black-forest-labs/flux2),
[FFmpeg rawvideo input](https://ffmpeg.org/ffmpeg-formats.html#rawvideo),
[IPython embedded video](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.Video).
The CPU path keeps video generation separate from the memory-heavy image model.
""", "markdown")
    add((ROOT / 'cinematic/image_video_demo.py').read_text() + '\n' + '''
from IPython.display import Video, display
POC = STUDIO / 'poc'
demo_images = [(f'{cid} - {view}', POC / cid / f'{view}.png')
               for cid in REFERENCES
               for view in ('full_body_master', 'view_three_quarter', 'view_back')]
missing = [str(path) for _, path in demo_images if not path.is_file()]
if missing:
    print('Generate the missing views with the image cell, then rerun this demo:', *missing, sep='\\n')
else:
    DEMO_VIDEO = render_image_demo(demo_images, STUDIO / 'image-motion-demo.mp4')
    print('Saved to Drive:', DEMO_VIDEO)
    display(Video(str(DEMO_VIDEO), embed=True, mimetype='video/mp4'))
''')
    add('''# Optional download: set True and run this cell to download the saved MP4.
DOWNLOAD_DEMO = False
if DOWNLOAD_DEMO:
    from google.colab import files
    video_path = STUDIO / 'image-motion-demo.mp4'
    if video_path.is_file():
        files.download(str(video_path))
    else:
        print('Run the quick video demo first.')
''')
    add('''# STOP: review faces, clothing, hair, shared style, full-body anatomy and alternate-view identity.
import shutil
if not DESIGN_APPROVED or not REVIEWER.strip():
    raise RuntimeError('Human design approval is pending. Inspect the comparison before setting the controls.')
for cid in REFERENCES:
    d = STUDIO / cid / 'atlas'; d.mkdir(parents=True, exist_ok=True)
    master = d / 'idle_front.png'
    original = POC / cid / 'full_body_master.png'
    if master.exists() and CW.sha(master) != CW.sha(original):
        raise FileExistsError('Use a new candidate directory for a new master.')
    if not master.exists(): shutil.copy2(original, master)
    CW.record_review(STUDIO / cid / 'design_approval.json',
        [master, Path(REFERENCES[cid]), STUDIO / 'reference-comparison.png'], 'design', REVIEWER, True)
''')
    add('''# Minimal animation assets first; no full pose library yet.
PROTOTYPE_JOBS = {'point_front', 'blink', 'mouth_closed', 'mouth_small', 'mouth_wide',
                  'mouth_round', 'mouth_teeth', 'mouth_smile_talk', 'expr_surprised'}
runtime_diagnostics.note('generation: importing torch and constructing lazy generator')
gen = studio_run.LocalGen(registry.REVISIONS[studio.FLUX_MODEL], log=runtime_diagnostics.note)
try:
    for cid in REFERENCES:
        master = STUDIO / cid / 'atlas/idle_front.png'
        CW.require_review(STUDIO / cid / 'design_approval.json', [master], 'design')
        for name, kind, prompt, extra in studio.ATLAS_JOBS:
            if name in PROTOTYPE_JOBS:
                status = studio_run.run_job(gen, master.parent / (name + '.png'), prompt, [master],
                    (640, 832), 2000, {'character': cid, 'stage': 'animation-prototype'})
                if status not in ('made', 'skipped'): raise RuntimeError(name + ': ' + status)
finally:
    gen.unload()
    runtime_diagnostics.note('generation finished; weights released')
''')
    add('''# Segmentation is separate from image generation. Manually prepared RGBA cut-outs are accepted.
for cid in REFERENCES:
    cut = STUDIO / cid / 'cutouts'; cut.mkdir(exist_ok=True)
    for name in ('idle_front', 'point_front'):
        target = cut / (name + '.png')
        if not target.exists():
            image = Image.open(STUDIO / cid / 'atlas' / (name + '.png'))
            rgba = studio.remove_background([image], registry.REVISIONS[studio.SEG_MODEL])[0]
            rgba.save(target)
    src, spec = studio.build_atlas_source(STUDIO, cid)
    print(cid, src / 'spec.json', 'Review the face/head boxes and cut-out edges before importing.')
''')
    add('''## Animation review before full production
Use `cinematic/preview.py` from the feature branch to import a **provisional prototype**, render both characters
through the existing drawkit/Pillow/FFmpeg path, and extract MP4 frames. Supply real existing per-speaker WAVs
and subtitle text. See `cinematic/README.md` for the exact command. Review joint/shadow quality against the
old layered-rig preview, as well as blink/mouth/expression patches. The atlas is a candidate, not visually approved.
Copy the resulting animation MP4 and frames into this candidate directory. Do not set ANIMATION_APPROVED
until you have watched it. The synthetic test movie produced by unit tests is not approval evidence.
''', 'markdown')
    add('''ANIMATION_VIDEO = STUDIO / 'animation-review.mp4'
if not ANIMATION_APPROVED or not REVIEWER.strip():
    raise RuntimeError('Human animation approval is pending; full atlas production is blocked.')
for cid in REFERENCES:
    master = STUDIO / cid / 'atlas/idle_front.png'
    CW.record_review(STUDIO / cid / 'animation_approval.json', [master, ANIMATION_VIDEO],
                     'animation', REVIEWER, True)
# New output names already made at prototype dimensions/seeds must retain identical requests.
runtime_diagnostics.note('generation: importing torch and constructing lazy generator')
gen = studio_run.LocalGen(registry.REVISIONS[studio.FLUX_MODEL], log=runtime_diagnostics.note)
try:
    for cid in REFERENCES:
        remaining = [j[0] for j in studio.ATLAS_JOBS if j[0] not in PROTOTYPE_JOBS]
        result = studio_run.run_atlas(gen, STUDIO, cid, size=(640,832), only=remaining)
        if any(v not in ('made','skipped') for v in result.values()): raise RuntimeError(result)
finally:
    gen.unload()
    runtime_diagnostics.note('generation finished; weights released')
''')
    add('''# Finish cut-outs and prepare reviewed source specs. Reusing a candidate keeps completed images.
for cid in REFERENCES:
    cut = STUDIO / cid / 'cutouts'
    for name, kind, _, _ in studio.ATLAS_JOBS:
        target = cut / (name + '.png')
        if kind in ('pose','view') and not target.exists():
            image = Image.open(STUDIO / cid / 'atlas' / (name + '.png'))
            studio.remove_background([image], registry.REVISIONS[studio.SEG_MODEL])[0].save(target)
    studio.build_atlas_source(STUDIO, cid)
print('Review spec.json face/head boxes, patch alignment and every output before importing a new version.')
''')
    add('''# Import remains provisional; importing does not mean approved.
for cid in REFERENCES:
    src = STUDIO / cid / 'atlas_src'
    spec = json.loads((src / 'spec.json').read_text())
    spec['identity'] = studio_run.IDENTITY[cid]
    prov = {'model': studio.FLUX_MODEL, 'revision': registry.REVISIONS[studio.FLUX_MODEL],
            'reference_sha256': CW.sha(REFERENCES[cid]), 'reference_file': Path(REFERENCES[cid]).name,
            'candidate': str(STUDIO), 'license': 'Apache-2.0 model; supplied reference rights remain with their owner'}
    SA.import_atlas(src, ASSETS, cid, NEW_VERSION, spec, prov)
print('Render production previews with preview.py, inspect every pose/expression, then approve below.')
''')
    add('''# A person must review the actual imported pack and its rendered animation.
if not PACK_APPROVED or not REVIEWER.strip(): raise RuntimeError('Final pack approval pending')
for cid in REFERENCES:
    vdir = ASSETS / 'characters' / cid / NEW_VERSION
    if not (vdir / 'previews/animation.mp4').is_file():
        raise RuntimeError('Render the imported pack preview first')
    print(cid, SB.approve(vdir, REVIEWER, 'Reviewed reference comparison and rendered animation'))
print('Next: bind approved versions and build a new video notebook with cinematic/deliver.py.')
''')
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "cells": cells}
    for i, c in enumerate(cells): c["id"] = f"cinematic-v2-{i:02}"
    return nb


if __name__ == "__main__":
    path = ROOT / "colab/notebooks/colab_cinematic_characters_v2.ipynb"
    path.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(path)

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
Low-memory revision: both transformer and text encoder use 4-bit weights, with low-memory loading and VAE tiling.
This reduces memory pressure but is not a verified Colab Free fit. Run cells individually from a fresh T4 runtime.
If the runtime fails during the Drive-mount cell (01), no image model has loaded: check session allocation,
Drive authorization and connectivity first. If it fails in cell 05, inspect the last memory-events.jsonl entry.
Do not repeatedly rerun a crashing cell. Preserve the logs and completed outputs on Drive.
""", "markdown")
    add('''from pathlib import Path
import os
from google.colab import drive
drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/ai-video')
ASSETS = ROOT / '_assets'
STUDIO = ASSETS / 'studio/reference-v2-lowmem-01'
STUDIO.mkdir(parents=True, exist_ok=True)
os.environ['STUDIO_MEMORY_LOG'] = str(STUDIO / 'memory-events.jsonl')
REFERENCES = {'arjun': str(ROOT / 'references/arjun.png'), 'meera': str(ROOT / 'references/meera.png')}
REVIEWER = ''
DESIGN_APPROVED = False
ANIMATION_APPROVED = False
PACK_APPROVED = False
NEW_VERSION = 'cinematic-v2'
''')
    add('''# Stable, idempotent Colab setup. Do not downgrade the environment's Diffusers on every run.
# Keep Colab's GPU-enabled torch installation unchanged.
import subprocess, sys, importlib.metadata as metadata
requirements = ['diffusers>=0.42,<0.43', 'transformers>=4.57,<6',
                'accelerate>=1.2,<2', 'bitsandbytes>=0.45,<1',
                'safetensors', 'Pillow', 'numpy', 'psutil',
                'pydantic>=2.7,<3', 'einops', 'timm']
subprocess.run([sys.executable, '-m', 'pip', 'install', *requirements], check=True)
# Diffusers/torchao imports must agree. Reinstall only when the required public API is missing.
probe = subprocess.run([sys.executable, '-c',
    'from torchao.quantization import FqnToConfig; from diffusers import Flux2KleinPipeline'],
    capture_output=True, text=True)
if probe.returncode:
    print('Import check failed:', (probe.stderr or '')[-2500:])
    print('Installing a matching torchao version without replacing torch/CUDA ...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                    'torchao==0.17.0'], check=True)
    probe = subprocess.run([sys.executable, '-c',
        'from torchao.quantization import FqnToConfig; from diffusers import Flux2KleinPipeline'],
        capture_output=True, text=True)
    if probe.returncode:
        raise RuntimeError('FLUX import still fails. Restart the Colab session once and re-run setup. '
                           + (probe.stderr or '')[-1600:])
for package in ('torch', 'torchao', 'diffusers', 'transformers'):
    print(package, metadata.version(package))
print('FLUX imports verified. Continue to preflight.')
''')
    add('''# Existing engine, with character-only overlays. No repository checkout needed.
import json, sys, os
WORK = Path('/content/cinematic-engine-v2'); WORK.mkdir(exist_ok=True)
MODULES = json.loads(''' + repr(json.dumps(sources(studio=True))) + ''')
for name, source in MODULES.items():
    (WORK / name).write_text(source)
sys.path.insert(0, str(WORK))
os.environ['STORYBOOK_ASSETS'] = str(ASSETS)
import cinematic_workflow as CW, studio, studio_run, registry, storybook as SB, storybook_atlas as SA
from PIL import Image
''')
    add('''report = CW.preflight(REFERENCES, STUDIO / 'preflight.json')
print(report)
if not report.get('gpu'):
    raise RuntimeError('Select an allocated T4 GPU; generation has not run.')
for model in (studio.FLUX_MODEL, studio.SEG_MODEL):
    ok, why = registry.model_ok(model, strict=True)
    if not ok: raise RuntimeError(why)
import diffusers
if not hasattr(diffusers, 'Flux2KleinPipeline'):
    raise RuntimeError('Installed Diffusers lacks Flux2KleinPipeline; see cinematic/README.md.')
''')
    add('''# Small PoC only. Source image -> portrait -> full-body master -> alternate view.
POC = STUDIO / 'poc'
gen = None
try:
    gen = studio_run.LocalGen(registry.REVISIONS[studio.FLUX_MODEL])
    studio.memory_snapshot('before_poc_inference')
    results = studio_run.run_poc(gen, POC, style_refs={k: [v] for k,v in REFERENCES.items()})
    studio.memory_snapshot('poc_complete')
finally:
    if gen is not None: gen.unload()
print(results)
if any(v not in ('made', 'skipped') for v in results.values()):
    raise RuntimeError('Some PoC jobs failed; review logs before continuing.')
items = []
for cid in REFERENCES:
    items.append((cid + ' supplied reference', Image.open(REFERENCES[cid])))
    items.extend((cid + ' ' + p.stem, Image.open(p)) for p in sorted((POC / cid).glob('*.png')))
studio.contact_sheet(items, STUDIO / 'reference-comparison.png', cols=5, cell=320)
display(Image.open(STUDIO / 'reference-comparison.png'))
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
gen = None
try:
    gen = studio_run.LocalGen(registry.REVISIONS[studio.FLUX_MODEL])
    for cid in REFERENCES:
        master = STUDIO / cid / 'atlas/idle_front.png'
        CW.require_review(STUDIO / cid / 'design_approval.json', [master], 'design')
        for name, kind, prompt, extra in studio.ATLAS_JOBS:
            if name in PROTOTYPE_JOBS:
                status = studio_run.run_job(gen, master.parent / (name + '.png'), prompt, [master],
                    (640, 832), 2000, {'character': cid, 'stage': 'animation-prototype'})
                if status not in ('made', 'skipped'): raise RuntimeError(name + ': ' + status)
finally:
    if gen is not None: gen.unload()
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
gen = None
try:
    gen = studio_run.LocalGen(registry.REVISIONS[studio.FLUX_MODEL])
    for cid in REFERENCES:
        remaining = [j[0] for j in studio.ATLAS_JOBS if j[0] not in PROTOTYPE_JOBS]
        result = studio_run.run_atlas(gen, STUDIO, cid, size=(640,832), only=remaining)
        if any(v not in ('made','skipped') for v in result.values()): raise RuntimeError(result)
finally:
    if gen is not None: gen.unload()
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

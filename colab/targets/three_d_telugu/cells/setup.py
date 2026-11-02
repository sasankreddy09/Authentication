# Persistent results on Drive; disposable scratch on Colab's local disk.
import os, json, subprocess, shutil, math, random, gc, time, re, difflib, hashlib
from pathlib import Path
PROJ = Path(DRIVE_ROOT) / PROJECT
WORK = Path(os.environ.get('WORK_DIR', '/content/work')) / PROJECT
for d in ('narration', 'keyframes', 'clips', 'render'):
    (PROJ / d).mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
# Keep large model weights on local scratch by default to protect free Drive space.
MODEL_CACHE = (Path(DRIVE_ROOT) if MODEL_CACHE_ON_DRIVE else WORK.parent) / '_model_cache'
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
if not LOG.path.startswith(str(PROJ)):
    LOG.move_to(PROJ / 'logs' / Path(LOG.path).name)
GPU = {'name': None, 'vram_gb': 0.0, 'bf16': False}
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    GPU = {'name': p.name, 'vram_gb': round(p.total_memory / 2**30, 1),
           'bf16': torch.cuda.get_device_capability(0)[0] >= 8}
BIG_GPU = False
VIDEO_ENGINE, IMAGE_MODEL = None, 'procedural'  # All visuals use approved assets and drawn scenes.

def free_disk_gb():
    return shutil.disk_usage(WORK).free / 2**30

def need_disk(gb, what, purge=()):
    if free_disk_gb() < gb:
        raise RuntimeError(f'Need {gb} GiB free scratch for {what}. Completed outputs remain on Drive.')

def free_tier_estimate(n_ai, n_drawn):
    print(f'{n_drawn} drawn scenes; {n_ai} AI clips. No diffusion model downloads. '
          'Runtime depends on cached speech and the hardware Colab allocates; quota is not guaranteed.')

print('Project:', PROJ, '| GPU:', GPU['name'] or 'none', '| local model cache:', MODEL_CACHE)

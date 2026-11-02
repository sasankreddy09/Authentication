# Check inputs before installing packages or downloading voice models.
import os, json, shutil, time
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
ASSETS = Path(DRIVE_ROOT) / '_assets'
for cid, version in CHARACTER_VERSIONS.items():
    if not version or Path(version).name != version or version in ('.', '..'):
        raise ValueError('Character versions must be simple folder names.')
    manifest = ASSETS / 'characters' / cid / version / 'manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Missing {manifest}. Prepare and approve the cinematic character pack first; '
                                'see docs/GOOGLE_COLAB.md. Reference pictures alone are not animation packs.')
    info = json.loads(manifest.read_text())
    if info.get('status') != 'approved' or info.get('asset_type') != 'atlas_v1':
        raise ValueError(f'{cid}/{version} is not an approved cinematic atlas.')
for env in ('cinema',):
    if not (ASSETS / 'environments' / env / 'v1/manifest.json').is_file():
        raise FileNotFoundError(f'Copy colab/assets/environments/{env}/v1 to {ASSETS}/environments/{env}/v1.')
if shutil.disk_usage('/content').free < 15 * 2**30:
    raise RuntimeError('Need at least 15 GiB free local scratch space before this run.')
import psutil, torch
if psutil.virtual_memory().available < 5 * 2**30:
    raise RuntimeError('Need at least 5 GiB available system RAM. Restart the runtime to release other models.')
if not RENDER_CACHED_AUDIO_ONLY:
    if not torch.cuda.is_available():
        raise RuntimeError('New Telugu speech needs an allocated GPU. Select T4 GPU; if unavailable, resume later. '
                           'Set RENDER_CACHED_AUDIO_ONLY=True only when this project already has matching narration.')
    if torch.cuda.get_device_properties(0).total_memory < 12 * 2**30:
        raise RuntimeError('This speech preset requires at least 12 GiB VRAM.')
# CUDA allocator choice must precede model loading; T4 uses float32 for Parler's T5 stability.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['OMP_NUM_THREADS'] = str(CPU_THREADS)
os.environ['MKL_NUM_THREADS'] = str(CPU_THREADS)
torch.set_num_threads(CPU_THREADS)
print('Inputs found. Full manifest/content verification follows before speech generation.')
print('CPU: animation, Telugu text, small int8 speech check. GPU: Parler speech and NVENC if available.')
print('Colab Free availability and session length vary; the local budget does not grant extra quota.')

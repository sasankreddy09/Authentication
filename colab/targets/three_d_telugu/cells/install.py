# Install only dependencies used by this drawn Telugu target; keep Colab's matching torch/CUDA.
import subprocess, sys
packages = ['numpy>=1.26,<3', 'Pillow>=10,<14', 'soundfile>=0.12,<1',
            'faster-whisper>=1.1,<2', 'pydantic>=2.7,<3', 'scipy>=1.11,<2',
            'librosa>=0.10,<2', 'huggingface_hub>=0.34,<1', 'psutil>=5.9,<8',
            'imageio>=2.34,<3', 'imageio-ffmpeg>=0.5,<1', 'ftfy>=6,<7']
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages], check=True)
# Indic Parler's isolated environment is installed only if uncached speech is needed.
# No diffusion, FLUX, Wan, Qwen, or hosted service is installed for rendering this video.
subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.DEVNULL)
from PIL import features
if not features.check('raqm'):
    raise RuntimeError('Pillow needs RAQM for Telugu text. Install libraqm0/libfribidi0 and restart the runtime.')

"""CPU-only still-image camera-motion demo, embedded verbatim in the Colab notebook."""
from pathlib import Path
import math
import shutil
import subprocess
import tempfile
from PIL import Image, ImageOps, ImageDraw


def render_image_demo(images, output, fps=24, seconds=2.0, size=(640, 360)):
    """Encode labeled stills with a gentle zoom; no invented body motion or speech."""
    images = [(str(label), Path(path)) for label, path in images]
    w, h = size
    if not images or fps <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Provide images and positive finite timing.')
    if w < 64 or h < 64 or w % 2 or h % 2:
        raise ValueError('H.264 output needs even dimensions of at least 64 pixels.')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('FFmpeg is missing. Run the notebook installation cell first.')
    # Validate every image before creating output; report the exact missing/corrupt input.
    for _, path in images:
        with Image.open(path) as im:
            im.verify()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, round(fps * seconds))
    # Unique staging directory keeps an existing successful demo intact on failure.
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        movie = Path(tmp) / 'demo.mp4'
        with tempfile.TemporaryFile() as log:
            proc = subprocess.Popen([ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo',
                '-pixel_format', 'rgb24', '-video_size', f'{w}x{h}', '-framerate', str(fps),
                '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(movie)],
                stdin=subprocess.PIPE, stderr=log)
            try:
                for label, path in images:
                    with Image.open(path) as source:
                        rgba = ImageOps.exif_transpose(source).convert('RGBA')
                        background = Image.new('RGBA', rgba.size, (240, 237, 230, 255))
                        background.alpha_composite(rgba)
                        base = background.convert('RGB')
                    for i in range(frames):
                        scale = 0.90 + 0.06 * i / max(1, frames - 1)
                        image = ImageOps.contain(base, (int(w * scale), int((h-36) * scale)),
                                                Image.Resampling.LANCZOS)
                        canvas = Image.new('RGB', size, (24, 27, 35))
                        canvas.paste(image, ((w-image.width)//2, (h-36-image.height)//2))
                        ImageDraw.Draw(canvas).text((12, h-26), label, fill='white')
                        proc.stdin.write(canvas.tobytes())
                proc.stdin.close()
                code = proc.wait(timeout=120)
                if code:
                    log.seek(0)
                    raise RuntimeError('FFmpeg failed: ' + log.read().decode(errors='replace')[-2000:])
            except BaseException:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                raise
            finally:
                if not proc.stdin.closed:
                    proc.stdin.close()
        if movie.stat().st_size == 0:
            raise RuntimeError('FFmpeg produced an empty video.')
        movie.replace(output)
    return output

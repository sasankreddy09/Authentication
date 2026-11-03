"""Render an imported character atlas, including its actual facial patches, on CPU."""
from pathlib import Path
import subprocess
import tempfile
from PIL import Image


def render_pack_preview(vdir, fps=24):
    import storybook as SB
    import storybook_atlas as SA
    vdir = Path(vdir)
    errors = SB.verify(vdir)
    if errors:
        raise ValueError(errors)
    ch = SA.AtlasCharacter(vdir)
    painter = SA.AtlasPainter(ch, 0.17)
    poses = list(ch.man['poses'])
    expressions = list(ch.man.get('face', {}).get('expressions', {}))
    mouths = list(ch.man.get('face', {}).get('visemes', {}))
    anchor = ch.man['face']['anchor_pose']
    output = vdir / 'previews'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        movie = Path(tmp) / 'animation.mp4'
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                '-s','640x360','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264',
                '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)], stdin=subprocess.PIPE, stderr=err)
            try:
                # First six seconds exercise facial animation; remaining shots exercise poses.
                for i in range((6 + len(poses))*fps):
                    t = i/fps
                    pid = anchor if t < 6 else poses[min(int(t)-6, len(poses)-1)]
                    face = {'blink': i % fps < 3, 'speaking': t < 6,
                            'viseme': mouths[(i//6)%len(mouths)] if mouths else 'rest',
                            'expr': expressions[int(t)%len(expressions)] if expressions else 'neutral'}
                    frame = Image.new('RGBA',(640,360),(35,38,45,255))
                    painter.draw(frame, (230,10), pid, face)
                    proc.stdin.write(frame.convert('RGB').tobytes())
                proc.stdin.close()
                if proc.wait(timeout=120):
                    err.seek(0)
                    raise RuntimeError(err.read().decode(errors='replace'))
            finally:
                if proc.poll() is None: proc.kill()
                proc.wait()
                if not proc.stdin.closed: proc.stdin.close()
        movie.replace(output / 'animation.mp4')
    return output / 'animation.mp4'

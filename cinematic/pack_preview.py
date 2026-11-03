"""Render an imported character atlas, including its actual facial patches, on CPU."""
from pathlib import Path
import subprocess
import tempfile
import math
from PIL import Image, ImageDraw


def render_pack_preview(vdir, fps=24, diagnostics=False, use_blink=False):
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
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
                # Stable idle by default. A separately named catalogue is explicitly diagnostic.
                shots = [(anchor, {}, 'Idle: no speech; gentle breathing')]
                if diagnostics:
                    shots += [(anchor, {'expr': e}, 'Expression: ' + e) for e in expressions]
                    shots += [(anchor, {'speaking': True, 'viseme': m}, 'Mouth test (silent): ' + m) for m in mouths]
                    shots += [(anchor, {'blink': True}, 'Blink patch test')]
                    shots += [(p, {}, 'Pose test: ' + p) for p in poses]
                seconds = 2 if diagnostics else 8
                for pid, face, label in shots:
                    for i in range(seconds * fps):
                        t = i / fps
                        state = dict(face)
                        if not diagnostics and use_blink:
                            state['blink'] = 2.0 <= t % 3.5 < 2.15
                        frame = Image.new('RGBA',(640,360),(35,38,45,255))
                        painter.draw(frame, (230,10), pid, state,
                                     motion=(0.0, 0.003*math.sin(2*math.pi*t/4)))
                        ImageDraw.Draw(frame).text((12,335), label, fill='white')
                        proc.stdin.write(frame.convert('RGB').tobytes())
                proc.stdin.close()
                if proc.wait(timeout=120):
                    err.seek(0)
                    raise RuntimeError(err.read().decode(errors='replace'))
            finally:
                if proc.poll() is None: proc.kill()
                proc.wait()
                if not proc.stdin.closed: proc.stdin.close()
        destination = output / ('diagnostics.mp4' if diagnostics else 'animation.mp4')
        movie.replace(destination)
    return destination

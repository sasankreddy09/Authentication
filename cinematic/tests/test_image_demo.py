"""Encoding checks use synthetic images, not artwork quality evidence."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_video_demo import render_image_demo
from build_notebook import build


class DemoTests(unittest.TestCase):
    def test_encode_and_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = []
            for index, color in enumerate(('red', 'blue')):
                path = root / f'{index}.png'
                Image.new('RGBA', (90, 130), color).save(path)
                images.append((color, path))
            out = render_image_demo(images, root/'demo.mp4', fps=10, seconds=.5, size=(160, 120))
            info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(out)]))
            stream = info['streams'][0]
            self.assertEqual(stream['codec_name'], 'h264')
            self.assertEqual(stream['pix_fmt'], 'yuv420p')
            self.assertEqual(int(stream['nb_frames']), 10)
            raw = subprocess.check_output(['ffmpeg','-v','error','-i',str(out),'-f','rawvideo','-pix_fmt','rgb24','-'])
            self.assertEqual(len(raw), 10*160*120*3)
            self.assertNotEqual(raw[:160*120*3], raw[-160*120*3:])
            old = out.read_bytes()
            with self.assertRaises(FileNotFoundError):
                render_image_demo([('missing',root/'missing.png')], out)
            self.assertEqual(old,out.read_bytes())

    def test_notebook_cells_compile(self):
        notebook = build()
        for c in notebook['cells']:
            if c['cell_type'] == 'code':
                compile(''.join(c['source']),c['id'],'exec')
        sources = [''.join(c['source']) for c in notebook['cells']]
        self.assertLess(next(i for i,s in enumerate(sources) if 'def render_image_demo' in s),
                        next(i for i,s in enumerate(sources) if '# STOP: review faces' in s))

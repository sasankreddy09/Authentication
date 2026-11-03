import sys
from pathlib import Path
import tempfile
import unittest
import subprocess
import json
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_characters import fixture
from pack_preview import render_pack_preview
import studio


class PatchBoundsTests(unittest.TestCase):
    def test_reject_resized_edit(self):
        with self.assertRaisesRegex(ValueError, 'canvas size'):
            studio.align_patch(Image.new('RGB',(100,100)),Image.new('RGB',(200,200)),[30,30,60,60])

    def test_reject_large_surround_drift(self):
        with self.assertRaisesRegex(ValueError, 'drift'):
            studio.align_patch(Image.new('RGB',(100,100),'black'),Image.new('RGB',(100,100),'white'),[30,30,60,60])

    def test_render_imported_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = fixture(root, 'arjun', approved=False)
            output = render_pack_preview(folder, fps=6)
            info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(output)]))
            self.assertEqual(info['streams'][0]['codec_name'], 'h264')
            self.assertGreater(int(info['streams'][0]['nb_frames']),36)

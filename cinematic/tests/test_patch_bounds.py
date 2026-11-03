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

    def test_expression_not_stacked_with_independent_patches(self):
        import numpy as np
        import storybook_atlas as SA
        with tempfile.TemporaryDirectory() as tmp:
            folder = fixture(Path(tmp), 'arjun', approved=False)
            painter = SA.AtlasPainter(SA.AtlasCharacter(folder), .17)
            expression = painter._pose_image('idle', {'expr': 'surprised'})
            combined = painter._pose_image('idle', {'expr': 'surprised', 'blink': True,
                                                   'speaking': True, 'viseme': 'wide'})
            self.assertTrue(np.array_equal(np.asarray(expression), np.asarray(combined)))

    def test_diagnostics_do_not_replace_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = fixture(Path(tmp), 'arjun', approved=False)
            normal = render_pack_preview(folder, fps=6)
            original = normal.read_bytes()
            diagnostic = render_pack_preview(folder, fps=6, diagnostics=True)
            self.assertEqual(diagnostic.name, 'diagnostics.mp4')
            self.assertEqual(normal.read_bytes(), original)

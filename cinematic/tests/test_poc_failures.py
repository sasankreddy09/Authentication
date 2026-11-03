import sys
from pathlib import Path
import tempfile
import unittest
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'colab/engine/lib'))
import studio_run


class FailingGenerator:
    revision = 'test'
    name = 'test'
    def __init__(self): self.calls = 0
    def __call__(self, *args):
        self.calls += 1
        raise MemoryError('original loading failure')


class PocFailureTests(unittest.TestCase):
    def test_failed_portrait_stops_chain_and_retains_cause(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / 'reference.png'
            Image.new('RGB', (8, 8)).save(ref)
            gen = FailingGenerator()
            with self.assertRaisesRegex(RuntimeError, 'original loading failure') as caught:
                studio_run.run_poc(gen, root / 'poc', characters=('arjun',),
                                   style_refs={'arjun': [ref]}, log=lambda _: None)
            self.assertIsInstance(caught.exception.__cause__, MemoryError)
            self.assertEqual(gen.calls, 1)
            self.assertFalse((root / 'poc/arjun/full_body_master.png').exists())

    def test_missing_dependency_identifies_exact_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / 'portrait.png'
            gen = FailingGenerator()
            with self.assertRaises(FileNotFoundError) as caught:
                studio_run.run_job(gen, root / 'full_body_master.png', 'prompt',
                                   [missing], (32, 32), 1, {})
            self.assertIn(str(missing), str(caught.exception))
            self.assertEqual(gen.calls, 0)

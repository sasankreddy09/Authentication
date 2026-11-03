"""Exercise startup diagnostics and early notebook cells without model imports."""
import builtins
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_support import RuntimeDiagnostics, lightweight_preflight, write_constraints
from build_notebook import build


class StartupTests(unittest.TestCase):
    def test_preflight_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'reference.png'
            Image.new('RGB', (8, 8), 'red').save(image)
            original = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.split('.')[0] in ('torch', 'diffusers', 'transformers', 'bitsandbytes'):
                    raise AssertionError('Unexpected heavyweight import: ' + name)
                return original(name, *args, **kwargs)
            with patch('builtins.__import__', side_effect=guarded), \
                 patch('runtime_support.shutil.which', return_value=None):
                result = lightweight_preflight({'arjun': image, 'meera': image}, root / 'report.json')
            self.assertIsNone(result['gpu'])
            self.assertEqual(result['references']['arjun']['size'], [8, 8])
            self.assertEqual(result, json.loads((root / 'report.json').read_text()))

    def test_logs_survive_monitor_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = RuntimeDiagnostics(tmp)
            try:
                monitor.note('test: before import')
            finally:
                monitor.stop()
            records = [json.loads(line) for line in monitor.path.read_text().splitlines()]
            self.assertEqual(records[-1]['stage'], 'test: before import')
            self.assertIn('memory', records[-1])
            self.assertFalse(monitor.thread.is_alive())

    def test_constraints_preserve_installed_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            versions = dict(torch='2.8.0+cu126', torchvision='0.23.0+cu126', torchaudio=None, numpy='2.0.2')
            path = Path(tmp) / 'constraints.txt'
            with patch('runtime_support.installed_versions', return_value=versions), \
                 patch('runtime_support.importlib.metadata.distributions', return_value=[]):
                write_constraints(path)
            self.assertIn('torch==2.8.0+cu126\n', path.read_text())
            self.assertNotIn('torchaudio', path.read_text())

    def test_engine_cell_in_clean_process(self):
        notebook = build()
        engine = next(''.join(c['source']) for c in notebook['cells'] if '# STEP 3:' in ''.join(c['source']))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = engine.replace('/content/cinematic-engine-v2', str(root / 'engine'))
            script = '''from pathlib import Path
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'diffusers', 'transformers', 'bitsandbytes'):
        raise AssertionError('Unexpected model import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
class Log:
    def note(self, text): print(text)
runtime_diagnostics = Log()
'''
            script += 'ASSETS = Path(' + repr(str(root / 'assets')) + ')\n' + engine
            path = root / 'startup.py'
            path.write_text(script)
            result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('engine ready', result.stdout)

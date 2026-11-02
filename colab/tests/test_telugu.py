"""CPU tests for the current target. Synthetic media does not establish speech/art quality."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'colab'))
sys.path.insert(0, str(ROOT / 'colab/engine/lib'))
sys.path.insert(0, str(ROOT / 'cinematic/tests'))
import build_telugu
import colab_budget
from test_characters import fixture
import numpy as np


class TargetTests(unittest.TestCase):
    def test_reproducible_valid_notebook(self):
        import nbformat
        self.assertEqual(build_telugu.OUTPUT.read_text(), build_telugu.serialized())
        nbformat.validate(nbformat.reads(build_telugu.serialized(), as_version=4))

    def test_other_notebooks_unchanged(self):
        manifest = json.loads((Path(__file__).parent / 'unchanged_notebooks.json').read_text())
        for name, expected in manifest.items():
            with self.subTest(notebook=name):
                self.assertEqual(hashlib.sha256((ROOT / 'colab/notebooks' / name).read_bytes()).hexdigest(), expected)

    def test_budget_and_plan_limits(self):
        clock = [0]
        b = colab_budget.SessionBudget(1, lambda: clock[0])
        self.assertEqual(b.timeout(120), 60)
        clock[0] = 60
        with self.assertRaises(colab_budget.BudgetExceeded): b.check('speech')
        with self.assertRaises(ValueError): colab_budget.SessionBudget(91)
        plan = json.loads((ROOT / 'colab/plans/three_d_friends_te_v3_storybook_plan.json').read_text())
        colab_budget.validate_target(plan)
        plan['shots'][0]['recipe'] = 'video'
        with self.assertRaises(ValueError): colab_budget.validate_target(plan)

    def test_v4_rejects_unexplained_location_and_props(self):
        plan = json.loads((ROOT / 'colab/plans/three_d_friends_te_v3_storybook_plan.json').read_text())
        colab_budget.validate_v4_continuity(plan)
        broken = copy.deepcopy(plan)
        broken['shots'][0]['location'] = 'lobby'
        with self.assertRaises(ValueError): colab_budget.validate_v4_continuity(broken)
        broken = copy.deepcopy(plan)
        visual = next(v for v in broken['shots'][0]['story']['visuals'] if v['recipe'] == 'character')
        visual['character']['chars']['arjun']['hold'] = 'popcorn'
        with self.assertRaises(ValueError): colab_budget.validate_v4_continuity(broken)

    def test_revised_dialogue_matches_source(self):
        plan = json.loads((ROOT / 'colab/plans/three_d_friends_te_v3_storybook_plan.json').read_text())
        source = json.loads((ROOT / 'colab/stories/three_d_friends_te_v4/story.json').read_text())
        self.assertEqual(plan['story_revision'], 'three_d_friends_te_v4')
        self.assertEqual([[l['text'] for l in sh['lines']] for sh in plan['shots']],
                         [[t['text'] for t in sc['turns']] for sc in source['scenes']])
        self.assertEqual(plan['project'], 'three_d_friends_te_v4_cinematic')
        self.assertEqual(plan['script_review_status'], 'pending')

    def test_offline_cached_audio_render(self):
        """Execute actual target cells with fixture audio/art, no GPU or model downloads."""
        import soundfile as sf
        cells = {c['metadata']['tags'][0]: ''.join(c['source']) for c in build_telugu.build()['cells']}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / '_assets'
            assets.mkdir()
            for cid in ('arjun', 'meera'): fixture(assets, cid, story=True)
            shutil.copytree(ROOT / 'colab/assets/environments', assets / 'environments')
            log_path = root / 'test.log'; log_path.touch()
            logs = []
            ns = {'json': json, 'Path': Path, 'ASSETS': assets,
                  'log': lambda *a: logs.append(str(a)), 'log_exc': lambda *a: logs.append(str(a)),
                  'ram': lambda: None, 'resources': lambda: 'CPU fixture', 'DEVICE_REPORT': {}, 'RUN_PEAKS': {'ram_gb': 0, 'gpu_gb': 0},
                  'LOG': types.SimpleNamespace(path=str(log_path), move_to=lambda p: None)}
            def run(tag): exec(compile(cells[tag], tag, 'exec'), ns)
            fake_torch = types.ModuleType('torch')
            fake_torch.__version__ = 'cpu-test-stub'
            fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
            def font_fixture(url, destination):
                # Placeholder glyphs are intentional; this test verifies pipeline plumbing, not Telugu shaping.
                shutil.copy('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', destination)
            with patch.dict(os.environ, {'WORK_DIR': str(root / 'work'), 'STORYBOOK_ASSETS': str(assets)}), \
                 patch.dict(sys.modules, {'torch': fake_torch, 'diffusers': None, 'transformers': None}), \
                 patch('urllib.request.urlretrieve', side_effect=font_fixture):
                run('settings')
                ns.update(DRIVE_ROOT=str(root), CHARACTER_VERSIONS={c:'test-v2' for c in ('arjun','meera')},
                          RENDER_CACHED_AUDIO_ONLY=True, W=320, H=180, FPS=8, MUSIC_STYLE='curious')
                run('setup'); run('lib'); run('budget'); run('plan')
                saved = (ns['PROJ'] / 'plan.json').read_bytes()
                run('plan')  # resume accepts the same bound plan
                self.assertEqual(saved, (ns['PROJ'] / 'plan.json').read_bytes())
                with patch.dict(ns['CHARACTER_VERSIONS'], {'arjun':'missing'}):
                    with self.assertRaises(FileNotFoundError): run('plan')
                run('plan'); run('helpers'); run('prefetch')
                self.assertEqual(ns['prefetch_jobs'](), [])
                src = cells['narration']; split = src.index('FLAGGED, _voices_checked, TAKES =')
                exec(compile(src[:split], 'narration definitions', 'exec'), ns)
                with self.assertRaisesRegex(RuntimeError, 'missing/stale'): ns['get_tts']()
                for shot in ns['SHOTS']:
                    for line in shot['lines']:
                        out = ns['PROJ'] / 'narration' / (line['id']+'.wav')
                        sf.write(out, np.sin(np.arange(7200)*.1).astype(np.float32)*.1, 24000)
                        ns['sidecar'](out, words=[], flagged=False, **ns['line_settings'](line))
                with patch.object(ns['voice_qc'], 'SpeakerCheck', side_effect=RuntimeError('No models in CPU fixture')):
                    exec(compile(src[split:], 'cached narration', 'exec'), ns)
                for tag in ['timeline', 'diagrams', 'keyframes', 'editlib', 'logic', 'bench', 'clips', 'music', 'sfx', 'render', 'qc', 'metadata']:
                    run(tag)
                movie = ns['PROJ'] / 'render/final.mp4'
                self.assertTrue(movie.is_file())
                info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(movie)]))
                self.assertEqual({s['codec_type'] for s in info['streams']}, {'video','audio'})
                self.assertTrue((ns['PROJ'] / 'render/subtitles.srt').is_file())
                self.assertFalse(json.loads((ns['PROJ'] / 'render/qc_report.json').read_text())['approved'])
                self.assertIn('Indic Parler-TTS', (ns['PROJ'] / 'render/metadata.json').read_text())
                self.assertIsNone(ns['VIDEO_ENGINE'])
                self.assertFalse(ns['GPU']['name'])
                ns['DIAGRAM_POOL'].shutdown(wait=True)
                ns['EDITOR'].shutdown(wait=True)


if __name__ == '__main__': unittest.main()

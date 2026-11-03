import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

class ASRDeviceTests(unittest.TestCase):
    def setup_asr(self, free_gb=4, mode='auto', failure=False):
        path = Path(__file__).resolve().parents[1] / 'targets/three_d_telugu/cells/narration.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_asr')
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.cuda.mem_get_info.return_value = (free_gb*2**30, 16*2**30)
        ct = MagicMock()
        ct.get_cuda_device_count.return_value = 1
        ct.get_supported_compute_types.return_value = {'int8_float16'}
        factory = MagicMock()
        if failure: factory.side_effect = [RuntimeError('CUDA init failed'), object()]
        modules = {'torch': torch, 'ctranslate2': ct, 'faster_whisper': types.SimpleNamespace(WhisperModel=factory)}
        ns = dict(_asr=None, _asr_failed=False, ASR_ON_GPU=False, ASR_DEVICE=mode,
                  LANGUAGE='te', ASR_MODEL='small.en', ASR_MODEL_MULTI='small', WHISPER_REPOS={},
                  BUDGET=MagicMock(), hf_files=lambda *args: '/cached/model', CPU_THREADS=2,
                  sys=sys, DEVICE_REPORT={}, preload_cuda_libs=MagicMock(), log=MagicMock(), log_exc=MagicMock())
        exec(compile(ast.Module(body=[function], type_ignores=[]),str(path),'exec'),ns)
        return ns, modules, factory

    def test_gpu_and_reuse(self):
        ns, mods, factory = self.setup_asr()
        with patch.dict(sys.modules,mods):
            first = ns['get_asr']()
            self.assertIs(first, ns['get_asr']())
        self.assertEqual(factory.call_count,1)
        self.assertEqual(factory.call_args.kwargs['device'],'cuda')
        self.assertTrue(ns['ASR_ON_GPU'])

    def test_low_memory_cpu(self):
        ns, mods, factory = self.setup_asr(free_gb=.5)
        with patch.dict(sys.modules,mods): ns['get_asr']()
        self.assertEqual(factory.call_args.kwargs['device'],'cpu')
        self.assertIn('only 0.50',ns['DEVICE_REPORT']['speech check (Whisper)'])

    def test_gpu_failure_fallback(self):
        ns, mods, factory = self.setup_asr(failure=True)
        with patch.dict(sys.modules,mods): ns['get_asr']()
        self.assertEqual(factory.call_count,2)
        self.assertFalse(ns['ASR_ON_GPU'])

    def test_forced_gpu_does_not_silently_use_cpu(self):
        ns, mods, factory = self.setup_asr(free_gb=.5,mode='cuda')
        with patch.dict(sys.modules,mods):
            with self.assertRaisesRegex(RuntimeError, 'GPU speech check required'):
                ns['get_asr']()
        factory.assert_not_called()
        self.assertTrue(ns['_asr_failed'])

    def test_honors_configured_multilingual_model(self):
        ns, mods, factory = self.setup_asr(free_gb=5)
        ns['ASR_MODEL_MULTI'] = 'large-v3-turbo'
        ns['WHISPER_REPOS'] = {'large-v3-turbo': 'test/turbo'}
        ns['hf_files'] = MagicMock(return_value='/cached/turbo')
        with patch.dict(sys.modules,mods): ns['get_asr']()
        self.assertEqual(ns['hf_files'].call_args.args[0], 'test/turbo')
        self.assertIn('large-v3-turbo',ns['DEVICE_REPORT']['speech check (Whisper)'])

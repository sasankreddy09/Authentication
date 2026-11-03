"""CPU configuration/lifecycle tests; these do not measure CUDA or Colab RAM."""
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'colab/engine/lib'))
import studio
import studio_run


class MemorySetupTests(unittest.TestCase):
    def test_quantizes_both_large_components(self):
        fake_torch = MagicMock()
        fake_diffusers = MagicMock()
        # Restrict the interface: a generic MagicMock silently accepted nonexistent APIs.
        pipe = MagicMock(spec_set=['vae', 'enable_model_cpu_offload'])
        pipe.vae = MagicMock(spec_set=['enable_tiling', 'enable_slicing'])
        fake_diffusers.Flux2KleinPipeline.from_pretrained.return_value = pipe
        with patch.dict(sys.modules, {'torch': fake_torch, 'diffusers': fake_diffusers}), \
             patch.object(studio, 'gpu_report', return_value={'cuda': True, 'native_bf16': False}):
            actual, dtype = studio.load_flux('pinned', log=lambda _: None)
        self.assertIs(actual, pipe)
        config = fake_diffusers.PipelineQuantizationConfig.call_args.kwargs
        self.assertEqual(config['components_to_quantize'], ['transformer', 'text_encoder'])
        self.assertTrue(config['quant_kwargs']['bnb_4bit_use_double_quant'])
        self.assertTrue(fake_diffusers.Flux2KleinPipeline.from_pretrained.call_args.kwargs['low_cpu_mem_usage'])
        pipe.vae.enable_tiling.assert_called_once()
        pipe.vae.enable_slicing.assert_called_once()
        pipe.enable_model_cpu_offload.assert_called_once()

    def test_constructor_does_not_load_weights(self):
        with patch.dict(sys.modules, {'torch': MagicMock()}), patch.object(studio, 'load_flux') as load:
            gen = studio_run.LocalGen('pinned', log=lambda _: None)
            load.assert_not_called()
            gen.unload()
            load.assert_not_called()

    def test_load_once_and_release_hooks(self):
        pipe = MagicMock()
        with patch.dict(sys.modules, {'torch': MagicMock()}), \
             patch.object(studio, 'load_flux', return_value=(pipe, 'fp16')) as load:
            gen = studio_run.LocalGen('pinned', log=lambda _: None)
            gen._ensure_loaded()
            gen._ensure_loaded()
            load.assert_called_once()
            gen.unload()
            pipe.maybe_free_model_hooks.assert_called_once()
            self.assertFalse(hasattr(gen, 'pipe'))

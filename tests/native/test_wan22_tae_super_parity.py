"""Native Super decoder against the pinned upstream architecture.

The synthetic test generates its own weights. The optional real-weight test
requires the exact upstream Git blob and a captured raw Wan2.2 latent.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROBE = Path(os.environ.get("TURBOCIDER_WAN22_VAE_BENCH_PROBE", "/nonexistent")).resolve()
SOURCE = Path(os.environ.get("TURBOCIDER_TAEHV_SOURCE", "/nonexistent")).resolve()
COMFY = Path(os.environ.get("TURBOCIDER_COMFY_ROOT", "/nonexistent")).resolve()
SOURCE_BLOB = "8514281c20318c71ccdf916e4383d9a65f825009"
WEIGHTS = Path(os.environ.get("TURBOCIDER_TAE_SUPER_WEIGHTS", "/nonexistent")).resolve()
LATENTS = Path(os.environ.get("TURBOCIDER_WAN22_VAE_LATENTS", "/nonexistent")).resolve()
WEIGHTS_BLOB = "2021b7ee3c9996cb2248fe36b044f18c6fefcb97"


@unittest.skipUnless(PROBE.is_file() and SOURCE.is_file() and (COMFY / "comfy/latent_formats.py").is_file(),
                     "native probe, pinned upstream source and Comfy latent format required")
class Wan22TAESuperParityTests(unittest.TestCase):
    def test_super_decoder_against_pinned_upstream(self):
        import torch
        from safetensors.torch import load_file, save_file

        content = SOURCE.read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        self.assertEqual(digest, SOURCE_BLOB)
        self.assertTrue(torch.backends.mps.is_available(), "MPS required for this fixture")
        spec = importlib.util.spec_from_file_location("taehv_test_pinned", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.path.insert(0, str(COMFY))
        from comfy.latent_formats import Wan22

        torch.manual_seed(122)
        model = module.TAEHV(checkpoint_path=None, arch_name="taew2_2_super").half().eval()
        raw = torch.randn((1, 48, 3, 4, 4), dtype=torch.float32).to(torch.bfloat16)
        normalized = Wan22().process_in(raw.float()).transpose(1, 2).to(device="mps", dtype=torch.float16)
        with tempfile.TemporaryDirectory(prefix="tc-taew22-super-") as temporary:
            root = Path(temporary)
            save_file({key: value.contiguous() for key, value in model.state_dict().items()}, str(root / "model.safetensors"))
            save_file({"latents": raw}, str(root / "input.safetensors"))
            model = model.to("mps")
            with torch.inference_mode():
                expected = model.decode_video(normalized, parallel=True, show_progress_bar=False).transpose(1, 2).float().cpu()
            result = subprocess.run([str(PROBE), "super", str(root / "model.safetensors"),
                                     str(root / "input.safetensors"), str(root / "output.safetensors"), "0", "1"],
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = (load_file(str(root / "output.safetensors"))["output"].float() + 1) / 2
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.isfinite(actual).all())
        error = (actual - expected).abs()
        self.assertLess(error.mean().item(), .005)
        self.assertLess(error.max().item(), .05)

    @unittest.skipUnless(WEIGHTS.is_file() and LATENTS.is_file(), "real Super weights and captured latent required")
    def test_real_super_weights_against_pinned_upstream(self):
        import torch
        from safetensors.torch import load_file

        weights = WEIGHTS.read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(weights)).encode() + b"\0" + weights).hexdigest()
        self.assertEqual(digest, WEIGHTS_BLOB)
        source = SOURCE.read_bytes()
        self.assertEqual(hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest(), SOURCE_BLOB)
        self.assertTrue(torch.backends.mps.is_available(), "MPS required for this fixture")
        spec = importlib.util.spec_from_file_location("taehv_test_pinned_real", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.path.insert(0, str(COMFY))
        from comfy.latent_formats import Wan22
        raw = load_file(str(LATENTS))["latents"]
        normalized = Wan22().process_in(raw.float()).transpose(1, 2).to(device="mps", dtype=torch.float16)
        model = module.TAEHV(str(WEIGHTS), arch_name="taew2_2_super").to(device="mps", dtype=torch.float16).eval()
        with tempfile.TemporaryDirectory(prefix="tc-taew22-super-real-") as temporary, torch.inference_mode():
            expected = model.decode_video(normalized, parallel=True, show_progress_bar=False).transpose(1, 2).float().cpu()
            output = Path(temporary) / "output.safetensors"
            result = subprocess.run([str(PROBE), "super", str(WEIGHTS), str(LATENTS), str(output), "0", "1"],
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = (load_file(str(output))["output"].float() + 1) / 2
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.isfinite(actual).all())
        error = (actual - expected).abs()
        self.assertLess(error.mean().item(), .002)
        self.assertLess(error.max().item(), .03)


if __name__ == "__main__":
    unittest.main()

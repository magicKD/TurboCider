"""Request-contract regression checks for the optional pruned LTX decoder."""
import ctypes as C
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class PrunaRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native"))
        cls.lib = C.CDLL(str(directory / "libturbocider.dylib"))
        cls.lib.tc_plan_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        cls.lib.tc_string_free.argtypes = [C.c_void_p]

    def plan(self, request):
        out, error = C.c_void_p(), C.c_void_p()
        code = self.lib.tc_plan_json(json.dumps(request).encode(), C.byref(out), C.byref(error))
        result = json.loads(C.string_at(out).decode()) if out.value else None
        message = C.string_at(error).decode() if error.value else ""
        self.lib.tc_string_free(out)
        self.lib.tc_string_free(error)
        return code, result, message

    def request(self, **changes):
        return dict(model="ltx-2.5-distilled", operation="video.generate",
                    prompt="A red fox", output="/tmp/pruna-test.mp4",
                    width=704, height=448, frames=97, fps=24, steps=11,
                    audio=False, execution="gpu", residency="component_staged", **changes)

    def test_default_and_explicit_decoder(self):
        code, result, error = self.plan(self.request())
        self.assertEqual(code, 0, error)
        self.assertEqual(result["ltx_video_vae"], "conv")
        code, result, error = self.plan(self.request(ltx_video_vae="prunavaed", allow_approximation=True))
        self.assertEqual(code, 0, error)
        self.assertEqual(result["ltx_video_vae"], "prunavaed")
        self.assertIn("ltx_prunavaed_pruned_decoder", result["algorithm_approximations"])

    def test_invalid_and_missing_opt_in(self):
        for decoder in ("diffvae", "", True, 1):
            code, _, error = self.plan(self.request(ltx_video_vae=decoder, allow_approximation=True))
            self.assertNotEqual(code, 0, (decoder, error))
        code, _, error = self.plan(self.request(ltx_video_vae="prunavaed"))
        self.assertNotEqual(code, 0)
        self.assertIn("allow_approximation=true", error)

    def test_v2_selects_same_decoder(self):
        request = json.loads((ROOT / "examples/requests/ltx-video.json").read_text())
        request["execution"].update(ltx_video_vae="prunavaed", allow_approximation=True)
        code, result, error = self.plan(request)
        self.assertEqual(code, 0, error)
        self.assertEqual(result["ltx_video_vae"], "prunavaed")
        request["execution"]["memory_constrained"] = dict(enabled=True, limit_bytes=20 << 30)
        self.assertNotEqual(self.plan(request)[0], 0)

    def test_non_ltx_rejects_decoder(self):
        code, _, error = self.plan(dict(model="flux2-klein-4b", operation="image.generate",
                                       prompt="fox", ltx_video_vae="prunavaed", allow_approximation=True))
        self.assertNotEqual(code, 0)
        self.assertIn("only supported by LTX", error)

    def test_external_decoder_path_contract(self):
        request = self.request(ltx_video_vae="prunavaed", allow_approximation=True,
                               ltx_video_vae_path="/library/prunavaed.safetensors")
        code, result, error = self.plan(request)
        self.assertEqual(code, 0, error)
        self.assertEqual(result["ltx_video_vae_path"], request["ltx_video_vae_path"])
        self.assertNotEqual(self.plan({**request, "ltx_video_vae": "conv"})[0], 0)
        self.assertNotEqual(self.plan({**request, "ltx_video_vae_path": "relative.safetensors"})[0], 0)


class PrunaWeightTests(unittest.TestCase):
    def test_wrong_latent_statistics_rejected_before_inference(self):
        binary = Path(os.environ.get("TURBOCIDER_VAE_PROBE_BINARY", ROOT / "build/prunavaed/ltx-video-vae-decode"))
        if not binary.is_file():
            self.skipTest("Build tools/native/build_ltx_video_vae_probe.sh first")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Well-formed safetensors container, unsupported model geometry.
            header = json.dumps({
                "latents_mean": dict(dtype="BF16", shape=[128], data_offsets=[0, 256]),
                "latents_std": dict(dtype="BF16", shape=[127], data_offsets=[256, 510]),
            }).encode()
            header += b" " * (-len(header) % 8)
            checkpoint = root / "bad.safetensors"
            checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 510)
            latent = root / "latent.bf16"
            latent.write_bytes(b"\0" * 256)
            env = os.environ.copy()
            env.pop("TURBOCIDER_LTX_VIDEO_VAE_CHECKPOINT_FD", None)
            result = subprocess.run([str(binary), str(checkpoint), str(latent), "1", "1", "1",
                                     str(root / "output.bf16")], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid PrunaVAED tensor: latents_std", result.stderr)
            self.assertFalse((root / "output.bf16").exists())


if __name__ == "__main__":
    unittest.main()

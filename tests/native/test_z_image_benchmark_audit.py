"""CPU-only adversarial fixtures for the strict completed-run auditor."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "audit", ROOT / "tools/validation/z_image_benchmark_audit.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.paths = []
        for i, variant in enumerate(("baseline", "fused", "fused", "baseline")):
            folder = root / str(i)
            folder.mkdir()
            for name in ("z_latent_initial", "z_latent_final", "z_decoded", "z_latent_step_1"):
                (folder / (name + ".safetensors")).write_bytes(b"identical fixture, not a tensor")
            (folder / "image.png").write_bytes(b"identical fixture, not a PNG")
            request = dict(width=512, height=512, steps=1, seed=123, prompt="fixture",
                           execution="gpu", output=str(folder / "image.png"))
            metrics = dict(actual_denoise_steps=1, hybrid={}, plan=dict(
                requested_execution="gpu", precision="bf16", requested_shape=[512,512,1]),
                timings_seconds=dict(denoise=2 if variant == "baseline" else 1,
                                     vae_decode=.2))
            runs = [dict(warmup=j == 0, parity=j == 2, request=copy.deepcopy(request),
                         error=None, metrics=copy.deepcopy(metrics), wall_seconds=3)
                    for j in range(3)]
            runs[-1]["request"]["dump_tensors"] = str(folder)
            report = dict(variant=variant, library_sha256="native", environment={},
                          hardware={"gpu":"fixture"}, runtime_libraries={
                              "libturbocider.dylib":{"sha256":"native"},
                              "libmlx.dylib":{"sha256":"mlx"}}, runs=runs)
            if variant == "baseline":
                report["environment"] = {k:"1" for k in (
                    "TURBOCIDER_Z_DISABLE_FUSED_QKV", "TURBOCIDER_Z_DISABLE_FUSED_MOD",
                    "TURBOCIDER_Z_DISABLE_MPP_SWIGLU",
                    "TURBOCIDER_Z_DISABLE_MPP_PROJECTIONS",
                    "TURBOCIDER_Z_DISABLE_MPP_QKV_PREPARE",
                    "TURBOCIDER_Z_DISABLE_GATE_NORM",
                    "TURBOCIDER_Z_DISABLE_CACHE_CONTEXT")}
            path = folder / "report.json"
            path.write_text(json.dumps(report))
            self.paths.append(path)
        self.summary = root / "summary.json"
        self.summary.write_text(json.dumps(dict(
            reports=[str(p) for p in self.paths], library_sha256="native",
            conditions=dict(size=512, steps=1, seed=123, prompt="fixture", control="unfused"),
            samples=dict(baseline=[2,2], fused=[1,1]))))

    def mutate_report(self, fn):
        value = json.loads(self.paths[1].read_text())
        fn(value)
        self.paths[1].write_text(json.dumps(value))

    def test_valid_and_exclude_cold_quality(self):
        result = AUDIT.audit(self.summary)
        self.assertEqual(result["diffusion_speedup"], 2)
        self.assertEqual(result["warm_samples"], {"baseline":2,"fused":2})

    def test_partial_schedule_rejected(self):
        data = json.loads(self.summary.read_text())
        data["reports"].pop()
        self.summary.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "schedule"):
            AUDIT.audit(self.summary)

    def test_actual_step_mismatch(self):
        self.mutate_report(lambda r: r["runs"][1]["metrics"].update(actual_denoise_steps=0))
        with self.assertRaisesRegex(ValueError, "step mismatch"):
            AUDIT.audit(self.summary)

    def test_runtime_mismatch(self):
        self.mutate_report(lambda r: r["runtime_libraries"]["libmlx.dylib"].update(sha256="other"))
        with self.assertRaisesRegex(ValueError, "runtimes differ"):
            AUDIT.audit(self.summary)

    def test_parity_missing(self):
        (self.paths[1].parent / "z_latent_step_1.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "tensor files"):
            AUDIT.audit(self.summary)

    def test_parity_differs(self):
        (self.paths[1].parent / "image.png").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "parity files differ"):
            AUDIT.audit(self.summary)

    def test_same_path_control_rejected(self):
        report = json.loads(self.paths[0].read_text())
        report["environment"] = {}
        self.paths[0].write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "control switches missing"):
            AUDIT.audit(self.summary)

    def test_legacy_report_in_degraded_m4_state_rejected(self):
        for path in self.paths:
            report = json.loads(path.read_text())
            report["hardware"]["gpu"] = "Apple M4 Max"
            for step in range(2, 9):
                (path.parent / f"z_latent_step_{step}.safetensors").write_bytes(
                    b"identical fixture, not a tensor")
            for row in report["runs"]:
                row["request"]["steps"] = 8
                row["metrics"]["actual_denoise_steps"] = 8
            if report["variant"] == "baseline":
                for row in report["runs"]:
                    row["metrics"]["timings_seconds"].update(denoise=14, vae_decode=5)
                    row["wall_seconds"] = 20
            path.write_text(json.dumps(report))
        summary = json.loads(self.summary.read_text())
        summary["conditions"]["steps"] = 8
        summary["samples"]["baseline"] = [14,14]
        self.summary.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "system state"):
            AUDIT.audit(self.summary)


if __name__ == "__main__":
    unittest.main()

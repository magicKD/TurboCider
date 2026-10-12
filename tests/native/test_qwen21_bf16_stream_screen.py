import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen21_bf16_stream_screen import validate_bf16_rows, validate_bf16_registration


def rows():
    result = []
    for index in range(3):
        row = dict(model="qwen-image-2.1", operation="image.generate", width=512, height=512, steps=40,
            actual_denoise_steps=40, seed=29, reference_tokens=0, prompt_cache_hit=False, hybrid=None,
            runtime_backend="mlx_cpp_metal_qwen21_bf16_streaming_experimental",
            timings_seconds=dict(request_wall=60, text_encode=3, denoise=56, vae_decode=.6),
            qwen_ffn_phases=dict(policy="gpu", prefill=dict(steps_this_request=1), decode=dict(steps_this_request=39)))
        metrics = dict(weight_precision="original_bf16", dynamic_layer_weight_arguments=True,
            components_released_before_vae=True, public_memory_qualification=False, physical_overlap_proved=False)
        for name, layers, passes in (("encoder", 36, 1), ("denoiser", 32, 40)):
            fills = (layers - 24) * passes
            metrics[name] = dict(layers=layers, prefix=24, slots=2, completed_passes=passes,
                completed_layers=layers * passes, completed_prefix_layers=24 * passes, completed_streamed_layers=fills,
                fills=fills, reader_fences=fills, completed_reader_fences=fills, managed_weight_budget_bytes=11 << 30,
                managed_weight_capacity_bytes=10 << 30, resident_source_bytes=9 << 30, streamed_source_bytes=fills << 28,
                source_file_bytes=14 << 30, verification_bytes=0 if index else 14 << 30,
                verification_cache_hits=1 if index else 0, source_sha256="a" * 64, drained=True)
        row["qwen_bf16_streaming"] = metrics
        result.append(row)
    return result


class Bf16StreamReceiptTests(unittest.TestCase):
    def test_registered_new_process_and_real_first_registration(self):
        hashes = dict(encoder="a" * 64, denoiser="a" * 64)
        sizes = {"encoder.safetensors": 14 << 30, "dit.safetensors": 14 << 30}
        for phase in ("first", "reuse"):
            data = rows()
            if phase == "reuse":
                for stage in data[0]["qwen_bf16_streaming"].values():
                    if isinstance(stage, dict):
                        stage.update(verification_bytes=0, verification_cache_hits=1)
            validate_bf16_rows(data, 40, True, (24, 24), 11 << 30, hashes, phase)
            records = []
            for index in range(6):
                initial = index < 2 and phase == "first"
                records.append(dict(component=list(sizes)[index % 2], enabled=True,
                    verification_bytes=(14 << 30) if initial else 0, verification_cache_hits=int(not initial),
                    registration_cache_hits=int(index < 2 and phase == "reuse"),
                    registration_records_written=int(initial), first_registration_seconds=.5 if initial else 0,
                    verification_seconds=.5 if initial else .001))
            validate_bf16_registration(records, phase, sizes)
            for field, value in (("component", "wrong"), ("enabled", False),
                ("registration_cache_hits", True), ("verification_bytes", 1),
                ("verification_cache_hits", -1), ("registration_records_written", 2),
                ("first_registration_seconds", float("nan")), ("verification_seconds", 0)):
                broken = copy.deepcopy(records); broken[0][field] = value
                with self.subTest(phase=phase, field=field), self.assertRaises(ValueError):
                    validate_bf16_registration(broken, phase, sizes)
            broken = copy.deepcopy(records)
            broken[2]["registration_cache_hits"] = 1
            with self.assertRaises(ValueError):
                validate_bf16_registration(broken, phase, sizes)
            with self.assertRaises(ValueError):
                validate_bf16_registration(records[:-1], phase, sizes)
        with self.assertRaises(ValueError):
            validate_bf16_rows(rows(), 40, True, (24, 24), 11 << 30, hashes, "unknown")

    def test_complete_actual_work_and_rejected_shortcuts(self):
        data = rows()
        hashes = dict(encoder="a" * 64, denoiser="a" * 64)
        validate_bf16_rows(data, 40, True, (24, 24), 11 << 30, hashes)
        for field, value in (("fills", 0), ("completed_reader_fences", 1), ("completed_passes", 1),
            ("completed_layers", 0), ("slots", True), ("managed_weight_capacity_bytes", 12 << 30),
            ("source_sha256", "b" * 64), ("verification_bytes", 0), ("drained", False)):
            invalid = copy.deepcopy(data)
            invalid[0]["qwen_bf16_streaming"]["denoiser"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_bf16_rows(invalid, 40, True, (24, 24), 11 << 30, hashes)
        invalid = copy.deepcopy(data)
        invalid[1]["qwen_bf16_streaming"]["encoder"]["verification_cache_hits"] = 0
        with self.assertRaises(ValueError):
            validate_bf16_rows(invalid, 40, True, (24, 24), 11 << 30, hashes)


if __name__ == "__main__":
    unittest.main()

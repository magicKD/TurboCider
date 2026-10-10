import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen21_bf16_stream_screen import validate_bf16_rows


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

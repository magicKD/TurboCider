import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen21_gguf_hybrid_screen import validate_receipts


def receipts(channels=4096, steps=40):
    result = []
    for index in range(3):
        row = dict(model="qwen-image-2.1", operation="image.generate", width=512, height=512,
                   seed=29, steps=steps, actual_denoise_steps=steps, reference_tokens=0,
                   prompt_cache_hit=False, encoder_execution="gpu", encoder_hybrid={},
                   encoder_runtime_precision="q4_k_m_affine_fp16_io",
                   encoder_weight_residency=dict(enabled=True, weights_retained=True,
                                                weights_reused=index > 0, loads_session_total=1),
                   timings_seconds=dict(request_wall=45, text_encode=.3, denoise=44),
                   runtime_backend="mlx_cpp_metal_gguf+private_ane_runtime_weight_experimental" if channels else "mlx_cpp_metal_qwen21_gguf",
                   qwen_ffn_phases=dict(policy="all" if channels else "gpu"), hybrid=None)
        for phase, count in (("prefill", 1), ("decode", steps - 1)):
            row["qwen_ffn_phases"][phase] = dict(steps_this_request=count,
                completed_channel_blocks_this_request=32 * count if channels else 0,
                runtime_calls_this_request=32 * count if channels else 0)
        if channels:
            row["hybrid"] = dict(runtime_failed=False, runtime_failures_session_total=0,
                runtime_calls_session_total=steps * 32 * (index + 1),
                runtime_weight=dict(executor_backend="private_ane", data_path="w8a8_hadamard",
                    partition_axis="intermediate_channels", ane_channels=channels,
                    channel_blocks_session_total=steps * 32 * (index + 1),
                    async_hybrid_blocks_session_total=steps * 32 * (index + 1),
                    fp32_channel_join_enabled=False, io_path="gpu_iosurface",
                    fallback_blocks_session_total=0, overflow_retries_session_total=0, headroom_scale=1))
        result.append(row)
    return result


class MixedKHybridScreenTests(unittest.TestCase):
    def test_actual_complete_gpu_and_hybrid_work(self):
        validate_receipts(receipts(0), 0)
        for channels in (4096, 5120):
            validate_receipts(receipts(channels), channels)

    def test_forged_source_phase_channels_fallback_or_lifecycle_rejected(self):
        faults = (
            lambda r: r.update(prompt_cache_hit=True),
            lambda r: r.update(encoder_runtime_precision="bf16"),
            lambda r: r["encoder_weight_residency"].update(weights_reused=True),
            lambda r: r["encoder_weight_residency"].update(loads_session_total=2),
            lambda r: r["hybrid"].update(runtime_failed=True),
            lambda r: r["hybrid"]["runtime_weight"].update(ane_channels=5120),
            lambda r: r["hybrid"]["runtime_weight"].update(fallback_blocks_session_total=1),
            lambda r: r["hybrid"]["runtime_weight"].update(overflow_retries_session_total=1),
            lambda r: r["hybrid"].update(runtime_failures_session_total=False),
            lambda r: r["hybrid"]["runtime_weight"].update(fp32_channel_join_enabled=True),
            lambda r: r["qwen_ffn_phases"]["decode"].update(runtime_calls_this_request=0),
            lambda r: r["timings_seconds"].update(denoise=float("nan")),
        )
        for fault in faults:
            rows = copy.deepcopy(receipts())
            fault(rows[0])
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                validate_receipts(rows, 4096)


if __name__ == "__main__":
    unittest.main()

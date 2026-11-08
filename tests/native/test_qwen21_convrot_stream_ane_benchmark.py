"""The ANE timing harness must reject toggles, fallback and incomplete execution."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/benchmarks"))
from qwen21_convrot_stream_ane_benchmark import validate_route


def completed_route():
    return {"hybrid": {"runtime_failed": False, "runtime_calls_session_total": 224, "runtime_weight": {
        "executor_backend": "private_ane", "data_path": "w8a8_convrot", "io_path": "gpu_iosurface",
        "partition_axis": "intermediate_channels", "ane_channels": 4096, "gpu_channels": 8192,
        "fp32_channel_join_enabled": True, "fixed_program_enabled": True, "program_disabled": False,
        "program_instances_session_total": 1, "fallback_blocks_session_total": 0,
        "prefetch_submissions_session_total": 0, "channel_blocks_session_total": 224,
        "untimed_hybrid_blocks_session_total": 224, "convrot_stage_submissions_session_total": 224,
        "device_io_calls_session_total": 224, "capability_calls_session_total": 3,
        "estimated_bytes": 500 << 20, "source_recipe": "original-i8+fp32-partial-join-v1",
    }}, "qwen_ffn_phases": {"prefill": {"steps_this_request": 1},
        "decode": {"steps_this_request": 7, "actual_rows": 1024}}}


class StreamAneEvidenceTests(unittest.TestCase):
    def test_complete_original_int8_split_is_accepted(self):
        validate_route(completed_route(), True)
        validate_route({"hybrid": None}, False)

    def test_toggle_intent_and_gpu_fallback_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_route({"execution": "gpu_ane", "hybrid": {}}, True)
        for key, value in (("fallback_blocks_session_total", 1), ("data_path", "fp16"),
                           ("channel_blocks_session_total", 223), ("convrot_stage_submissions_session_total", 223),
                           ("program_disabled", True), ("device_io_calls_session_total", 0),
                           ("capability_calls_session_total", 0), ("ane_channels", 6144)):
            with self.subTest(key=key):
                record = completed_route()
                record["hybrid"]["runtime_weight"][key] = value
                with self.assertRaises(ValueError):
                    validate_route(record, True)

    def test_control_cannot_hide_private_work_and_decode_must_complete(self):
        with self.assertRaises(ValueError):
            validate_route(completed_route(), False)
        record = copy.deepcopy(completed_route())
        record["qwen_ffn_phases"]["decode"]["steps_this_request"] = 6
        with self.assertRaises(ValueError):
            validate_route(record, True)


if __name__ == "__main__":
    unittest.main()

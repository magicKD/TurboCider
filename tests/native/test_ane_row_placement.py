import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests.native.test_runtime_ane_quality_compare import bind_execution
from runtime_ane_common import validate_row_placement

ROOT=Path(__file__).resolve().parents[2]


class RowPlacementTests(unittest.TestCase):
    def test_host_window_bounds_and_inverse(self):
        with tempfile.TemporaryDirectory() as raw:
            exe=Path(raw)/"row-window"
            built=subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "tests/native/ane_row_window_test.cpp","-o",str(exe)],cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10)
            self.assertEqual(run.returncode,0,run.stderr)

    def test_requested_placement_requires_actual_counters_and_protected_rows(self):
        row=dict(model="z-image-turbo",lora_strategy="none",hybrid=dict(runtime_calls_session_total=256,
            runtime_weight=dict(partition_axis="rows",data_path="fp16",row_placement="image_prefix",
                hybrid_blocks_session_total=256,row_suffix_blocks_session_total=16,
                row_prefix_blocks_session_total=240,row_image_tail_blocks_session_total=0,row_protected_rows_session_total=7680)))
        validate_row_placement([row],"image_prefix")
        with self.assertRaises(ValueError):validate_row_placement([row])
        for change in ({"row_placement":"suffix"},{"row_prefix_blocks_session_total":0},
                       {"row_prefix_blocks_session_total":True},{"row_image_tail_blocks_session_total":1},
                       {"row_protected_rows_session_total":0},{"data_path":"w8a8_hadamard"},
                       {"partition_axis":"intermediate_channels"},{"hybrid_blocks_session_total":255}):
            candidate=copy.deepcopy(row);candidate["hybrid"]["runtime_weight"].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):validate_row_placement([candidate],"image_prefix")
        for key in list(row["hybrid"]["runtime_weight"]):
            if key.startswith("row_"):
                candidate=copy.deepcopy(row);candidate["hybrid"]["runtime_weight"].pop(key)
                with self.subTest(key=key),self.assertRaises(ValueError):validate_row_placement([candidate],"image_prefix")
        validate_row_placement([dict(hybrid={})]) # legacy suffix receipt compatibility

    def test_execution_binding_does_not_infer_image_placement_from_flags(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);gpu,ane=root/"gpu.json",root/"ane.json"
            common=dict(model="z-image-turbo",width=512,height=512,seed=42,steps=8,actual_denoise_steps=8,
                lora_strategy="none",lora_applied_projections=0,timings_seconds=dict(request_wall=1.,denoise=.8))
            baseline=dict(**common,runtime_backend="mlx_cpp_metal")
            runtime=dict(executor_backend="private_ane",data_path="fp16",io_path="gpu_iosurface",partition_axis="rows",
                device_io_calls_session_total=256,fallback_blocks_session_total=0,failure_reason="",row_placement="image_tail",
                hybrid_blocks_session_total=256,row_suffix_blocks_session_total=16,row_prefix_blocks_session_total=0,
                row_image_tail_blocks_session_total=240,row_protected_rows_session_total=7680)
            candidate=dict(**common,runtime_backend="mlx_cpp_metal+private_ane_runtime_weight_experimental",
                hybrid=dict(runtime_failed=False,runtime_failures_session_total=0,runtime_calls_session_total=256,runtime_weight=runtime))
            for path,row in ((gpu,baseline),(ane,candidate)):
                path.write_text(json.dumps(dict(exit_code=0,binary_sha256="a"*64,adjacent_library_sha256="b"*64,
                    artifacts_unchanged=True,stdout=json.dumps(row))))
            policy=dict(data_path="fp16",channel_auto=False,device_io=True)
            with self.assertRaises(ValueError):bind_execution(gpu,ane,"z-image-turbo",**policy)
            bound=bind_execution(gpu,ane,"z-image-turbo",row_placement="image_tail",**policy)
            self.assertTrue(bound["candidate_ane_executed"])
            self.assertEqual(bound["row_placement"],"image_tail")


if __name__=="__main__":unittest.main(verbosity=2)

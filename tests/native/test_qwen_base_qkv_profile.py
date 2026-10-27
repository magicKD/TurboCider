"""Portable QKV component denominators and fixed-program observation guards."""
from collections import Counter
import copy
from pathlib import Path
import sys
import json
import struct
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/profiling/qwen_base"))
import qkv_collect as screen
import qkv_audit as audit
from test_qwen_base_profiling import fixture as observer_fixture


def report_fixture():
    samples = []
    for rep in range(-2, 10):
        for j in range(4):
            arm = screen.ARMS[(j + rep + 4) % 4]
            samples.append(dict(arm=arm, rep=rep, warmup=rep < 0, seconds=.03,
                                weight_stage_seconds=.003 if arm == "private-restaged" else 0,
                                activation_seconds=0, submit_seconds=0, join_seconds=0))
    return dict(status="complete", complete_request=False, image_export=False, chunk=512, repeats=10, warmups=2,
                fresh_inputs_weights_outputs_verified=True, borrowed_values_unchanged=True, predictions=29, weight_stages=15,
                actual_source_request=dict(model="qwen-image-2.1", width=512, height=512, steps=40, frames=1, audio=False,
                    lora_count=0, execution="gpu", residency="resident", allow_approximation=False,
                    operation="image.generate", reference_count=0), samples=samples)


class BaseQkvProfilingTests(unittest.TestCase):
    def test_capture_reserves_space_before_starting_the_model(self):
        screen.require_output_space((2 << 30) + (256 << 20))
        with self.assertRaises(ValueError):
            screen.require_output_space((2 << 30) + (256 << 20) - 1)

    def test_warmup_and_honest_staging_denominators(self):
        result = screen.validate(report_fixture(), 10)
        self.assertTrue(all(len(x["seconds"]) == 10 for x in result.values()))
        self.assertEqual(result["gpu-original"]["median_seconds"], .03)

    def test_narrower_source_or_incomplete_work_cannot_claim_a_base40_screen(self):
        for mutate in (lambda r: r["actual_source_request"].update(steps=20),
                       lambda r: r["actual_source_request"].update(audio=True),
                       lambda r: r["actual_source_request"].update(lora_count=1),
                       lambda r: r.update(complete_request=True), lambda r: r.update(predictions=28),
                       lambda r: r["samples"].pop(),
                       lambda r: next(x for x in r["samples"] if x["arm"] == "private-restaged").update(weight_stage_seconds=0)):
            value = report_fixture()
            mutate(value)
            with self.assertRaises(ValueError):
                screen.validate(value, 10)

    def test_actual_model_identity_and_hot_no_load_are_required(self):
        value, _ = observer_fixture()
        # Keep valid raw calibration/clock records, put setup outside hot work.
        for row in value["events"]:
            if row["name"] in ("ane.compile_api", "ane.load_api"):
                row["request"] = 0
            elif row["name"] == "ane.evaluate_api":
                row["request"] += 1
        self.assertTrue(screen.observer_contract(value, 2)["hot_no_compile"])
        value = copy.deepcopy(value)
        next(r for r in value["events"] if r["name"] == "ane.load_api")["request"] = 1
        with self.assertRaises(ValueError):
            screen.observer_contract(value, 2)

    def test_weight_switch_must_not_create_a_second_program(self):
        value, _ = observer_fixture()
        for row in value["events"]:
            if row["name"] in ("ane.compile_api", "ane.load_api"):
                row["request"] = 0
            elif row["name"] == "ane.evaluate_api":
                row["request"] += 1
        next(r for r in value["events"] if r["name"] == "ane.evaluate_api")["model"] = 2
        value["counts"] = dict(Counter(r["name"] for r in value["events"]))
        with self.assertRaises(ValueError):
            screen.observer_contract(value, 2)

    def test_full_array_reader_checks_dtype_extents_and_finiteness(self):
        with tempfile.TemporaryDirectory(prefix="tc-qkv-audit-") as directory:
            path = Path(directory) / "array.safetensors"
            def save(dtype="BF16", end=4):
                header=json.dumps(dict(x=dict(dtype=dtype, shape=[1,2], data_offsets=[0,end]))).encode()
                path.write_bytes(struct.pack("<Q",len(header))+header+struct.pack("<HH",0x3f80,0x4000))
            save()
            np.testing.assert_array_equal(audit.arrays(path)["x"], [[1.,2.]])
            for dtype,end in (("F16",4),("BF16",6)):
                save(dtype,end)
                with self.assertRaises(ValueError):
                    audit.arrays(path)
            with self.assertRaises(ValueError):
                audit.quality(np.array([1.]), np.array([np.nan]))

    def test_bad_calibration_is_not_reclassified_as_success(self):
        value, _ = observer_fixture()
        value["events"].append(copy.deepcopy(value["events"][0]))
        value["counts"] = dict(Counter(r["name"] for r in value["events"]))
        with self.assertRaises(ValueError):
            screen.observer_contract(value, 2)


if __name__ == "__main__":
    unittest.main()

"""Mechanism and rejection tests; never starts Metal/CoreML from this suite."""
import copy
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

WIKI = Path(__file__).resolve().parents[1]
LABS = WIKI / "labs"
sys.path.insert(0, str(LABS))
import common
import run as lab_run
import caches
import operators
import policies
import systems
import inspect_model
import summarize
import upscaling
import check_catalog


class NumericalReferences(unittest.TestCase):
    def test_every_case_runs_and_labels_scope(self):
        for name in lab_run.CASES:
            with self.subTest(case=name):
                value = lab_run.run_case(name)
                self.assertEqual(value["lab_id"], name)
                self.assertFalse(value["production_qualified"])
                self.assertNotEqual(value["physical_ane_placement"], "verified")
                json.dumps(value, allow_nan=False)

    def test_multiple_w8_shapes_and_seeds(self):
        for seed in (3, 19):
            for shape in ((1, 8, 16), (3, 16, 32)):
                value = lab_run.run_case("w8a8-reference", seed, dict(zip(("rows", "hidden", "width"), shape)))
                self.assertLess(value["details"]["unquantized_rotation_witness"]["max_abs"], 1e-12)

    def test_rotation_inverse_norm_and_invalid_block(self):
        row = [float(index - 4) for index in range(16)]
        transformed = common.rotate(row, 8)
        self.assertLess(common.metrics(row, common.rotate(transformed, 8, inverse=True))["max_abs"], 1e-12)
        self.assertAlmostEqual(sum(x*x for x in row), sum(x*x for x in transformed), places=10)
        with self.assertRaises(ValueError): common.rotate(row, 3)

    def test_quantizer_zero_ties_ranges_nonfinite_and_overflow(self):
        codes, scales = common.quantize_rows([[0., 0.]], normalize=True, half_scales=True)
        self.assertEqual(codes, [[0, 0]])
        self.assertGreater(scales[0], 0)
        self.assertEqual([common.round_even_int8(x) for x in (.5, 1.5, 2.5, -.5, -1.5)], [0, 2, 2, 0, -2])
        self.assertEqual(common.round_even_int8(-129, -128), -128)
        for value in (float("inf"), float("nan")):
            with self.assertRaises(ValueError): common.quantize_rows([[value]])
        with self.assertRaises(ValueError): common.fp16(70000)

    def test_packed_tail_and_invalid_types(self):
        q = lab_run.quantization
        for bits in (4, 8):
            for count in (1, 3, 7, 9, 17):
                codes = [index % (1 << bits) for index in range(count)]
                self.assertEqual(q.unpack_codes(q.pack_codes(codes, bits), bits, count), codes)
        for codes, bits in (([True], 4), ([-1], 4), ([16], 4), ([1], 3)):
            with self.assertRaises(ValueError): q.pack_codes(codes, bits)
        with self.assertRaises(ValueError): q.unpack_codes(b"\0", 4, 1)

    def test_online_softmax_extremes_masks_and_tail(self):
        q, k, v = [[1000., -1000.], [-3., 5.]], [[10., -10.], [9., -8.], [-2., 1.]], [[1., 2.], [-3., 1.], [0., 4.]]
        for mask in (None, lambda i, j: i == 1):
            expected = operators.dense_attention(q, k, v, mask)
            actual = operators.online_attention(q, k, v, 2, mask)
            self.assertLess(common.metrics(expected, actual)["max_abs"], 1e-12)
        with self.assertRaises(ValueError): operators.online_attention(q, k, v, 0)

    def test_negative_norm_crop_and_upscale_controls(self):
        self.assertGreater(operators.global_stats()["details"]["local_stat_counterexample"]["max_abs"], .1)
        self.assertGreater(operators.causal_convolution()["details"]["independent_crop_counterexample"]["max_abs"], .01)
        self.assertGreater(upscaling.tiled_upscale()["details"]["local_reflect_feather_approximation"]["max_abs"], .01)

    def test_bad_options_rejected(self):
        for options in ({"hidden":3}, {"rows":10000}, {"rows":True}, {"unknown":1}):
            with self.assertRaises(ValueError): lab_run.run_case("w8a8-reference", options=options)

    def test_hidden_sequence_axes_and_shared_bandwidth_floor(self):
        value = lab_run.run_case("partition-axes")["details"]
        for key in ("hidden_input_reduce_before_activation", "hidden_output_concat", "sequence_ffn_concat", "kv_sequence_global_softmax_merge"):
            self.assertLess(value[key]["max_abs"], 1e-12)
        for key in ("wrong_sum_of_branch_nonlinearities", "wrong_per_output_shard_rms", "wrong_average_of_local_softmax_outputs"):
            self.assertGreater(value[key]["max_abs"], 1e-4)
        self.assertAlmostEqual(value["bandwidth_scenario"]["joint_lower_bound_seconds"], .006)


class SchedulingAndCache(unittest.TestCase):
    def test_cycle_unknown_dependency_and_bad_span(self):
        for tasks in (
            [{"id":"a","resource":"gpu","duration":1.,"deps":["a"]}],
            [{"id":"a","resource":"gpu","duration":1.,"deps":["missing"]}],
            [{"id":"a","resource":"gpu","duration":float("nan"),"deps":[]}]):
            with self.assertRaises(ValueError): systems.schedule(tasks)

    def test_slot_old_producer_and_live_consumer(self):
        slot = systems.Slot(); generation = slot.fill("A")
        with self.assertRaises(ValueError): slot.borrow("A", "consumer")
        slot.ready(generation); slot.borrow("A", "consumer"); slot.timeout()
        with self.assertRaises(ValueError): slot.fill("B")
        slot.complete(generation, "consumer"); next_generation = slot.fill("B")
        with self.assertRaises(ValueError): slot.ready(generation)
        slot.ready(next_generation, success=False)
        with self.assertRaises(ValueError): slot.borrow("B", "consumer")

    def test_calibration_decline_and_extrapolation(self):
        points = [{"share":.25,"gpu":.008,"ane":.003,"both":.009}, {"share":.75,"gpu":.003,"ane":.008,"both":.0095}]
        self.assertIsNone(policies.select_share(points, .006, [(1024,4096)], 2048)["selected"])
        with self.assertRaises(ValueError): policies.interpolate(*points, .1, "gpu")
        invalid = copy.deepcopy(points); invalid[0]["both"] = .0001
        with self.assertRaises(ValueError): policies.select_share(invalid, .02, [(1024,4096)], 2048)

    def test_step_condition_reset_and_no_alias(self):
        controller = caches.ResidualReplay(1., warmup=0, max_skips=2)
        x = [1.]; controller.run(0, 10, "A", x, lambda row: [1.1]); x[0] = 99
        self.assertEqual(controller.previous, [1.])
        _, reused = controller.run(1, 10, "A", [1.], lambda row: [1.1]); self.assertTrue(reused)
        _, reused = controller.run(4, 10, "A", [1.], lambda row: [1.2]); self.assertFalse(reused)
        _, reused = controller.run(5, 10, "B", [1.], lambda row: [1.3]); self.assertFalse(reused)

    def test_schedule_off_and_capability_rejection(self):
        item = dict(id="x",requires=["ffn"],phase=1,reads=[],writes=["ffn_precision"],steps=[])
        self.assertEqual(policies.compose([item], ["ffn"]), [])
        with self.assertRaises(ValueError): policies.compose([item], [])
        with self.assertRaises(ValueError): policies.compose([{**item, "steps":[-1]}], ["ffn"])

    def test_residual_shape_reset_and_callback_failure_retirement(self):
        controller = caches.ResidualReplay(1., warmup=0)
        controller.run(0, 10, "A", [1.], lambda row: [1.1])
        _, reused = controller.run(1, 10, "A", [1.,2.], lambda row: [1.1,2.2])
        self.assertFalse(reused)
        def failure(row): raise RuntimeError("original callback")
        with self.assertRaisesRegex(RuntimeError, "original callback"):
            controller.run(4, 10, "A", [1.,2.], failure)
        self.assertIsNone(controller.residual)
        _, reused = controller.run(5, 10, "A", [1.,2.], lambda row: [1.1,2.2])
        self.assertFalse(reused)

    def test_residual_invalid_numeric_policy_and_output(self):
        with self.assertRaises(ValueError): caches.ResidualReplay(float("nan"))
        controller = caches.ResidualReplay(1.)
        for value in ([float("nan")], []):
            with self.assertRaises(ValueError): controller.run(0, 3, "A", value, lambda row: row)
        with self.assertRaises(ValueError): controller.run(0, 3, "A", [1.], lambda row: [1.,2.])
        self.assertIsNone(controller.previous)
        with self.assertRaisesRegex(ValueError, "subtraction overflow"):
            controller.run(0, 3, "A", [-1e308], lambda row: [1e308])
        self.assertIsNone(controller.residual)

    def test_private_file_read_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"; path.write_bytes(b"1234")
            with path.open("rb") as stream:
                self.assertEqual(systems.read_exact(stream.fileno(), 1, 2), b"23")
                with self.assertRaises(IOError): systems.read_exact(stream.fileno(), 3, 2)


class MetadataAndEvidence(unittest.TestCase):
    def write_fixture(self, path, header):
        raw = json.dumps(header).encode(); path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0"*24)

    def fixture_header(self):
        return {"blocks.0.w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
                "blocks.1.w":{"dtype":"F32","shape":[2],"data_offsets":[8,16]},
                "norm":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}

    def test_header_only_actual_bounds_and_floor_components(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.safetensors"; self.write_fixture(path, self.fixture_header())
            details = inspect_model.inspect(path)["details"]
            self.assertEqual((details["tensor_count"], details["widest_block_bytes"], details["nonblock_payload_bytes"]), (3,8,8))
            self.assertFalse(details["payload_content_verified"])
            self.assertLess(details["header_bytes_read"], path.stat().st_size)

    def test_bad_header_keys_offsets_shapes_caps(self):
        with self.assertRaises(ValueError): inspect_model.unique_object([("x",1),("x",2)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.safetensors"
            for change in ({"data_offsets":[1,9]}, {"shape":[3]}, {"shape":[True]}, {"dtype":"mystery"}):
                header = self.fixture_header(); header["blocks.0.w"].update(change); self.write_fixture(path, header)
                with self.subTest(change=change), self.assertRaises(ValueError): inspect_model.inspect(path)
            self.write_fixture(path, self.fixture_header())
            with self.assertRaises(ValueError): inspect_model.inspect(path, max_header=8)

    def device_report(self):
        return {"schema":"turbocider-metal-teaching-lab-v1", "actual_metal_dispatch":True,
                "all_outputs_finite":True,"tail_guards_intact":True,
                "samples":[{"variant":"baseline","pair":0,"input_generation":2,"wall_seconds":.01},
                           {"variant":"candidate","pair":0,"input_generation":2,"wall_seconds":.008}]}

    def test_summary_scope_and_mismatched_pairs(self):
        result = summarize.summarize(self.device_report()); self.assertFalse(result["production_qualified"])
        self.assertEqual(result["matched_pairs"], 1)
        value = self.device_report(); value["samples"][1]["input_generation"] = 3
        with self.assertRaises(ValueError): summarize.summarize(value)
        with self.assertRaises(ValueError): summarize.summarize({"schema":"integration", "wall_seconds":1})
        value = self.device_report(); value["tail_guards_intact"] = False
        with self.assertRaises(ValueError): summarize.summarize(value)

    def test_summary_keeps_diagnostic_mode_and_refuses_changed_sources(self):
        for mode in ("none", "counter-stage", "capture", "unknown"):
            value = self.device_report()
            value.update(diagnostic_mode=mode, formal_performance_score=True, source_inputs_unchanged=True)
            result = summarize.summarize(value)
            self.assertEqual(result["context"]["diagnostic_mode"], mode)
            self.assertEqual(result["formal_performance_score"], mode == "none")
        value["source_inputs_unchanged"] = False
        with self.assertRaises(ValueError):
            summarize.summarize(value)

    def test_emit_no_overwrite_and_no_nonfinite_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with redirect_stdout(io.StringIO()): common.emit({"ok":True}, path)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()): common.emit({"other":1}, path)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()): common.emit({"x":float("nan")}, Path(directory)/"bad.json")
            self.assertFalse((Path(directory)/"bad.json").exists())

    def test_catalog_covers_technical_chapters(self):
        self.assertEqual(check_catalog.audit()["errors"], [])

    def test_catalog_rejects_missing_card_case_and_controls(self):
        original = json.loads((LABS/"catalog.json").read_text())
        value = copy.deepcopy(original); value["entries"].pop()
        self.assertTrue(any("lacks implementation" in row for row in check_catalog.audit(value)["errors"]))
        value = copy.deepcopy(original); value["entries"][0]["cases"] = ["not-a-case"]
        self.assertTrue(any("unknown case" in row for row in check_catalog.audit(value)["errors"]))
        value = copy.deepcopy(original); value["entries"][0]["controls"] = []
        self.assertTrue(any("incomplete executable" in row for row in check_catalog.audit(value)["errors"]))


if __name__ == "__main__": unittest.main()

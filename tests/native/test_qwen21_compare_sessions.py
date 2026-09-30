"""Offline guards against mismatched resident prefix-hit comparisons."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools/validation/qwen21_compare_sessions.py"
SPEC = importlib.util.spec_from_file_location("qwen21_compare_sessions", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrefixHitSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.gpu, self.hybrid = root / "gpu", root / "hybrid"
        self.gpu.mkdir()
        self.hybrid.mkdir()

    def record(self, root, index, hit, *, references=12288):
        data = dict(model="qwen-image-2.1", operation="image.edit", width=512,
                    height=512, steps=5, text_tokens=130, reference_tokens=references,
                    prompt_cache_hit=True,
                    acceleration_selection=f"experimental resident prefix KV {'hit' if hit else 'miss'}")
        (root / f"run-{index}.json").write_text(json.dumps(data))

    def test_three_matching_hits_follow_miss(self):
        for root in (self.gpu, self.hybrid):
            for index in range(4):
                self.record(root, index, index > 0)
        self.assertEqual(MODULE.select_runs(self.gpu, self.hybrid, prefix_hits_only=True),
                         ["run-1.json", "run-2.json", "run-3.json"])

    def test_rejects_miss_among_hits(self):
        for root in (self.gpu, self.hybrid):
            self.record(root, 0, False)
            self.record(root, 1, True)
            self.record(root, 2, True)
        self.record(self.hybrid, 2, False)
        with self.assertRaisesRegex(AssertionError, "mixed prefix miss/hit"):
            MODULE.select_runs(self.gpu, self.hybrid, prefix_hits_only=True)

    def test_rejects_mismatched_or_gapped_sequences(self):
        for root in (self.gpu, self.hybrid):
            self.record(root, 0, False)
            self.record(root, 1, True)
        self.record(self.hybrid, 2, True)
        with self.assertRaisesRegex(AssertionError, "unmatched runs"):
            MODULE.select_runs(self.gpu, self.hybrid, prefix_hits_only=True)
        self.record(self.gpu, 2, True)
        for root in (self.gpu, self.hybrid):
            (root / "run-1.json").unlink()
        with self.assertRaisesRegex(AssertionError, "contiguous"):
            MODULE.select_runs(self.gpu, self.hybrid, prefix_hits_only=True)

    def test_rejects_changed_geometry_and_missing_initial_miss(self):
        for root in (self.gpu, self.hybrid):
            self.record(root, 0, False)
            self.record(root, 1, True)
        self.record(self.gpu, 1, True, references=8192)
        with self.assertRaisesRegex(AssertionError, "changed request geometry"):
            MODULE.select_runs(self.gpu, self.hybrid, prefix_hits_only=True)
        self.record(self.gpu, 1, True)
        self.record(self.hybrid, 0, True)
        with self.assertRaisesRegex(AssertionError, "preceding resident prefix miss"):
            MODULE.select_runs(self.gpu, self.hybrid, prefix_hits_only=True)


class DBCacheAccountingTest(unittest.TestCase):
    def record(self, *, calls, cached=None, rectangle=False, front=8, warmup=8):
        item = dict(steps=40, hybrid=dict(runtime_calls_session_total=calls,
                                        runtime_failures_session_total=0,
                                        checkpoint_sha256_verified=True),
                    width=768 if rectangle else 512, height=512,
                    plan=dict(algorithm_approximations=[
                        "qwen21_rectangular_decode_w8a8_tiled_diagnostic"] if rectangle else []))
        if cached is not None:
            item["qwen21_dbcache"] = dict(front_blocks=front, back_blocks=0,
                threshold=0.24 if front == 1 else 0.25, warmup_steps=warmup,
                max_consecutive=3 if front == 1 else 2, cached_steps=cached,
                saved_middle_blocks=cached * (32 - front))
        return item

    def test_request_delta_matches_saved_blocks(self):
        baseline = self.record(calls=2560)
        candidate = self.record(calls=2056, cached=21)
        self.assertIsNone(MODULE.validate_db_cache_pair(
            baseline, candidate, self.record(calls=1280),
            self.record(calls=1280)))
        with self.assertRaisesRegex(AssertionError, "per-request Core ML"):
            MODULE.validate_db_cache_pair(baseline, self.record(calls=2080, cached=21),
                                          self.record(calls=1280), self.record(calls=1280))

    def test_rejects_cached_baseline_and_wrong_bookkeeping(self):
        baseline = self.record(calls=1280)
        candidate = self.record(calls=776, cached=21)
        with self.assertRaisesRegex(AssertionError, "already cached"):
            MODULE.validate_db_cache_pair(self.record(calls=1280, cached=3), candidate)
        candidate["qwen21_dbcache"]["saved_middle_blocks"] = 480
        with self.assertRaisesRegex(AssertionError, "saved-block count"):
            MODULE.validate_db_cache_pair(baseline, candidate)

    def test_rectangular_cache_saves_two_calls_per_skipped_layer(self):
        baseline = self.record(calls=2496, rectangle=True)
        candidate = self.record(calls=1488, cached=21, rectangle=True)
        self.assertIsNone(MODULE.validate_db_cache_pair(
            baseline, candidate, self.record(calls=0, rectangle=True),
            self.record(calls=0, rectangle=True)))
        with self.assertRaisesRegex(AssertionError, "changed rectangular hybrid route"):
            MODULE.validate_db_cache_pair(baseline, self.record(calls=1488, cached=21),
                                          self.record(calls=0, rectangle=True), self.record(calls=0))

    def test_f1_w4_saves_31_blocks_per_cached_step(self):
        # Different pre-request totals ensure this is per-request accounting,
        # including the rectangular route's two predictions per skipped layer.
        for rectangle, full_calls, tiles in ((False, 1280, 1), (True, 2496, 2)):
            with self.subTest(rectangle=rectangle):
                baseline = self.record(calls=1000 + full_calls, rectangle=rectangle)
                candidate = self.record(calls=2000 + full_calls - 27 * 31 * tiles,
                                        cached=27, rectangle=rectangle, front=1, warmup=4)
                before_baseline = self.record(calls=1000, rectangle=rectangle)
                before_candidate = self.record(calls=2000, rectangle=rectangle)
                self.assertEqual(candidate["qwen21_dbcache"]["saved_middle_blocks"], 837)
                self.assertIsNone(MODULE.validate_db_cache_pair(
                    baseline, candidate, before_baseline, before_candidate))
                candidate["hybrid"]["runtime_calls_session_total"] += tiles
                with self.assertRaisesRegex(AssertionError, "per-request Core ML"):
                    MODULE.validate_db_cache_pair(
                        baseline, candidate, before_baseline, before_candidate)
                candidate["qwen21_dbcache"]["saved_middle_blocks"] = 27 * 24
                with self.assertRaisesRegex(AssertionError, "saved-block count"):
                    MODULE.validate_db_cache_pair(baseline, candidate)

    def test_target_only_prefill_call_budget_depends_on_route(self):
        full_ref = dict(text_tokens=130, reference_tokens=12288, width=512, height=512,
                        plan=dict(qwen21_reference_size=1024, algorithm_approximations=[]))
        self.assertEqual(MODULE.expected_target_only_saved_hybrid_calls(full_ref, full_ref), 0)
        tiled = dict(full_ref, plan=dict(qwen21_reference_size=512,
                         algorithm_approximations=["qwen21_tiled_prefill_last16_w8a8_diagnostic"]))
        self.assertEqual(MODULE.expected_target_only_saved_hybrid_calls(tiled, tiled), 13)
        with self.assertRaisesRegex(AssertionError, "changed hybrid prefill route"):
            MODULE.expected_target_only_saved_hybrid_calls(full_ref, tiled)

    def test_full_reference_gpu_to_hybrid_checks_cache_policy_and_calls(self):
        gpu = dict(operation="image.edit", steps=40, reference_tokens=8192,
                   plan=dict(execution="gpu", qwen21_reference_size=1024,
                             lora_count=0, qwen21_w8a8=False,
                             algorithm_approximations=[]))
        hybrid = dict(operation="image.edit", steps=40, reference_tokens=8192,
                      plan=dict(execution="gpu_ane_experimental", qwen21_reference_size=1024,
                                lora_count=0, qwen21_w8a8=True,
                                algorithm_approximations=["qwen21_w8a8_full_reference_diagnostic"]),
                      hybrid=dict(checkpoint_sha256_verified=True, runtime_failed=False,
                                  runtime_failures_session_total=0, block_count=32,
                                  bucket=1024, ane_mlp_range=[0, 6144],
                                  runtime_calls_session_total=2528))
        previous = dict(hybrid=dict(runtime_calls_session_total=1280))
        MODULE.validate_full_reference_hybrid_pair(gpu, hybrid, previous)
        cached = dict(front_blocks=8, back_blocks=0, warmup_steps=8,
                      threshold=0.25, max_consecutive=8, cached_steps=28,
                      saved_middle_blocks=672)
        gpu["qwen21_dbcache"] = cached
        hybrid["qwen21_dbcache"] = cached
        gpu["plan"]["algorithm_approximations"].append("qwen21_decode_dbcache_diagnostic")
        hybrid["plan"]["algorithm_approximations"].append("qwen21_decode_dbcache_diagnostic")
        hybrid["hybrid"]["runtime_calls_session_total"] = 1856
        MODULE.validate_full_reference_hybrid_pair(gpu, hybrid, previous)
        hybrid["hybrid"]["runtime_calls_session_total"] += 1
        with self.assertRaisesRegex(AssertionError, "per-request Core ML"):
            MODULE.validate_full_reference_hybrid_pair(gpu, hybrid, previous)
        hybrid["hybrid"]["runtime_calls_session_total"] -= 1
        hybrid["qwen21_dbcache"] = dict(cached, max_consecutive=4)
        with self.assertRaisesRegex(AssertionError, "changed DBCache policy"):
            MODULE.validate_full_reference_hybrid_pair(gpu, hybrid, previous)


if __name__ == "__main__":
    unittest.main()

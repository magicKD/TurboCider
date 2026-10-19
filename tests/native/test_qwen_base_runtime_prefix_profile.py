"""Host negative controls for actual runtime-prefix receipts and step counts."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/profiling/qwen_base"))
import collect


def metrics():
    rows, calls = [], 0
    for i in range(3):
        hit = i > 0
        calls += 1280 if hit else 1440
        rows.append(dict(operation="image.edit", reference_tokens=4096, text_tokens=35,
            qwen_runtime_prefix_snapshot=dict(admitted=True, hit=hit, published=not hit, decline_reason="", bytes=(4096+35)*524288,
                                             recipe_sha256="a"*64),
            qwen_ffn_phases=dict(policy="all", prefill=dict(steps_this_request=0 if hit else 1,
                completed_channel_blocks_this_request=0 if hit else 32), decode=dict(steps_this_request=40 if hit else 39,
                actual_rows=1024, completed_channel_blocks_this_request=1280 if hit else 1248)),
            hybrid=dict(runtime_calls_session_total=calls)))
    return rows


class RuntimePrefixProfileTests(unittest.TestCase):
    def test_explicit_recipe_and_real_steps(self):
        collect.validate_runtime_prefix(metrics())
        on = collect.environment("runtime", "stock", 6144, True)
        off = collect.environment("runtime", "stock", 6144, False)
        self.assertEqual({k for k in on if on[k] != off[k]}, {"TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT"})

    def test_false_hits_fallback_stale_recipe_and_partial_steps_are_rejected(self):
        for change in (lambda r: r[1]["qwen_runtime_prefix_snapshot"].update(hit=False),
                       lambda r: r[0]["qwen_runtime_prefix_snapshot"].update(published=False),
                       lambda r: r[2]["qwen_runtime_prefix_snapshot"].update(recipe_sha256="b"*64),
                       lambda r: r[1]["qwen_runtime_prefix_snapshot"].update(decline_reason="GPU fallback"),
                       lambda r: r[1]["qwen_ffn_phases"]["decode"].update(steps_this_request=39),
                       lambda r: r[1]["qwen_ffn_phases"]["prefill"].update(steps_this_request=1),
                       lambda r: r[1]["hybrid"].update(runtime_calls_session_total=2880)):
            rows = copy.deepcopy(metrics())
            change(rows)
            with self.assertRaises(ValueError):
                collect.validate_runtime_prefix(rows)

    def test_conflicting_profile_routes_are_rejected(self):
        for route, mode in (("gpu", "stock"), ("cache", "stock"), ("runtime", "runtime-layers")):
            with self.assertRaises(ValueError):
                collect.environment(route, mode, 4096, True)
        with self.assertRaises(ValueError):
            collect.environment("runtime", "stock", 6144, True, True)

    def test_existing_retained_prefix_has_separate_receipts(self):
        rows = metrics()
        for i, row in enumerate(rows):
            del row["qwen_runtime_prefix_snapshot"]
            row["acceleration_selection"] = "; experimental resident prefix KV " + ("hit" if i else "miss")
        collect.validate_runtime_prefix(rows, True)
        rows[1]["acceleration_selection"] = "; experimental resident prefix KV miss"
        with self.assertRaises(ValueError):
            collect.validate_runtime_prefix(rows, True)


if __name__ == "__main__":
    unittest.main()

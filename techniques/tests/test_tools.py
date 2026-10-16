"""CPU-only tests for wiki helpers, independent of installed ML frameworks."""
import copy
import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


compare = module("compare_runs", "scripts/compare_runs.py")
memory = module("interval_peak", "scripts/interval_peak.py")
history = module("history_index", "scripts/history_index.py")
wiki = module("check_wiki", "scripts/check_wiki.py")
standalone = module("check_standalone", "scripts/check_standalone.py")
split = module("split_ffn", "examples/split_ffn_reference.py")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "examples/paired-runs.json").read_text())

    def test_paired_stats_and_synthetic_scope(self):
        result = compare.summarize(self.data)
        self.assertEqual(result["warm_pairs"], 2)
        self.assertAlmostEqual(result["pooled_median_speedup"], 10.1 / 8.1)
        self.assertFalse(result["qualification_passed"])
        self.assertEqual(result["recorded_order"], "ABBA")

    def test_incomplete_pair_rejected(self):
        self.data["samples"].pop()
        with self.assertRaises(ValueError):
            compare.summarize(self.data)

    def test_duplicate_rejected(self):
        self.data["samples"].append(copy.deepcopy(self.data["samples"][0]))
        with self.assertRaises(ValueError):
            compare.summarize(self.data)

    def test_mixed_build_rejected(self):
        self.data["samples"][-1]["build"] = "unexpected"
        with self.assertRaises(ValueError):
            compare.summarize(self.data)

    def test_sample_identity_override_rejected(self):
        self.data["samples"][0]["identity"] = {"workload": "different"}
        with self.assertRaises(ValueError):
            compare.summarize(self.data)

    def test_unqualified_samples_withhold_speedup(self):
        for field, value in (("quality", "unknown"), ("quality", "fail"), ("status", "error"), ("status", "cancelled")):
            with self.subTest(field=field, value=value):
                document = copy.deepcopy(self.data)
                document["samples"][0][field] = value
                result = compare.summarize(document)
                self.assertNotIn("pooled_median_speedup", result)
                self.assertEqual(result["unsuccessful_or_unqualified_samples"], 1)

    def test_invalid_duration(self):
        for value in (True, 0, -1, float("nan"), float("inf"), "1"):
            with self.subTest(value=value):
                self.data["samples"][0]["wall_seconds"] = value
                with self.assertRaises(ValueError):
                    compare.summarize(self.data)

    def test_incomplete_cold_does_not_disappear(self):
        cold = copy.deepcopy(self.data["samples"][0])
        cold["phase"] = "cold"
        self.data["samples"].append(cold)
        with self.assertRaises(ValueError):
            compare.summarize(self.data)

    def test_missing_context_rejected(self):
        del self.data["identity"]["weights"]
        with self.assertRaises(ValueError):
            compare.summarize(self.data)

    def test_scope_is_not_promoted(self):
        self.data["scope"] = "operator"
        self.assertEqual(compare.summarize(self.data)["scope"], "operator")

    def test_schema_boolean_and_synthetic_string_rejected(self):
        for key, value in (("schema_version", True), ("synthetic", "false")):
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare.summarize(data)

    def test_extreme_units_rejected_not_infinite_speedup(self):
        for sample in self.data["samples"]:
            sample["wall_seconds"] = 1e307 if sample["variant"] == "baseline" else 1e-307
        with self.assertRaises(ValueError):
            compare.summarize(self.data)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "examples/memory-plan.json").read_text())

    def test_alias_once_and_live_intervals(self):
        result = memory.analyze(self.data)
        self.assertEqual(result["peak_phase"], "decode")
        self.assertEqual(result["declared_peak_bytes"], 134217728 + 4294967296 + 33554432)
        self.assertFalse(result["hard_memory_bound"])
        self.assertTrue(result["declared_fits"])

    def test_alias_full_allocation_must_agree(self):
        self.data["claims"][4]["bytes"] = 1
        with self.assertRaises(ValueError):
            memory.analyze(self.data)

    def test_reverse_interval_rejected(self):
        self.data["claims"][0].update(first="decode", last="condition")
        with self.assertRaises(ValueError):
            memory.analyze(self.data)

    def test_invalid_bytes(self):
        for value in (True, -1, 0.5, "123"):
            with self.subTest(value=value):
                self.data["claims"][0]["bytes"] = value
                with self.assertRaises(ValueError):
                    memory.analyze(self.data)

    def test_too_small_target(self):
        self.data["target_bytes"] = 1
        self.assertFalse(memory.analyze(self.data)["declared_fits"])

    def test_unknown_phase(self):
        self.data["claims"][0]["first"] = "unknown"
        with self.assertRaises((KeyError, ValueError)):
            memory.analyze(self.data)

    def test_duplicate_name_and_phase_rejected(self):
        self.data["phases"].append("decode")
        with self.assertRaises(ValueError):
            memory.analyze(self.data)
        self.data["phases"].pop()
        self.data["claims"].append(copy.deepcopy(self.data["claims"][0]))
        with self.assertRaises(ValueError):
            memory.analyze(self.data)

    def test_synthetic_marker_survives(self):
        self.assertTrue(memory.analyze(self.data)["synthetic"])


class HistoryTests(unittest.TestCase):
    def test_only_prose_not_tool_or_reasoning(self):
        for role in ("system", "tool", "developer"):
            self.assertEqual(history.messages({"type": "response_item", "payload": {"type": "message", "role": role, "content": [{"type": "output_text", "text": "secret"}]}}), [])
        self.assertEqual(history.messages({"type": "response_item", "payload": {"type": "function_call", "arguments": "secret"}}), [])
        self.assertEqual(history.messages({"type": "response_item", "payload": {"type": "message", "role": "assistant", "channel": "analysis", "content": [{"type": "output_text", "text": "secret"}]}}), [])

    def test_scope_and_no_default_text_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            records = [{"type": "session_meta", "payload": {"cwd": "/synthetic/TurboCider", "id": "demo", "timestamp": "2026-01-01"}},
                       {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "IOSurface channel split W8A8"}]}}]
            path.write_text("\n".join(json.dumps(record) for record in records))
            self.assertIsNone(history.inspect(path, ["unrelated"]))
            result = history.inspect(path, ["TurboCider"])
            self.assertEqual(result["hits"]["handoff"], 1)
            self.assertNotIn("private_excerpts", result)
            self.assertNotIn("IOSurface", json.dumps(result))

    def test_redaction(self):
        self.assertNotIn("sk-example", history.SECRET.sub("[REDACTED]", "key sk-example123456789"))

    def test_filtered_tail_excerpts_do_not_include_tool_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            records = [{"type":"session_meta","payload":{"cwd":"/synthetic/TurboCider"}}]
            for index in range(3):
                records.append({"type":"response_item","payload":{"type":"message","role":"assistant",
                    "content":[{"type":"output_text","text":"W8A8 lesson " + str(index)}]}})
            records.append({"type":"response_item","payload":{"type":"function_call","arguments":"W8A8 tool secret"}})
            path.write_text("\n".join(json.dumps(record) for record in records))
            result = history.inspect(path, ["TurboCider"], excerpts=True, limit=1, match=re.compile("W8A8"), tail_excerpts=True)
            self.assertEqual(result["private_excerpts"][0]["text"], "W8A8 lesson 2")
            self.assertNotIn("tool secret", json.dumps(result))

    def test_reviewer_and_context_are_not_technical_prose(self):
        for content in ("Guardian review handoff W8A8", '<codex_internal_context> W8A8', '{"risk_level":"low","outcome":"allow"}'):
            record = {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}}
            self.assertEqual(history.messages(record), [])


class AlgebraTests(unittest.TestCase):
    def test_uneven_channels_and_full_down_lora(self):
        for strength in (0, .7, -1.2):
            for rows in (1, 5, 9):
                with self.subTest(strength=strength, rows=rows):
                    args = split.fixture(rows=rows, strength=strength)
                    self.assertLess(split.max_error(split.full(*args), split.channels(*args, cuts=[2, 5])), 1e-12)

    def test_row_split(self):
        args = split.fixture()
        actual = split.full(args[0][:2], *args[1:]) + split.full(args[0][2:], *args[1:])
        self.assertEqual(actual, split.full(*args))

    def test_invalid_partition(self):
        for cuts in ([], [0], [7], [2, 2], [5, 2], [2.5], [True]):
            with self.subTest(cuts=cuts):
                with self.assertRaises(ValueError):
                    split.channels(*split.fixture(), cuts=cuts)


class WikiTests(unittest.TestCase):
    def test_local_asset_hash_is_not_external_checkout_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.small_wiki(directory, sources=[{"id":"local","scope":"local","path":"README.md",
                "sha256":hashlib.sha256(b"# Example\n").hexdigest()}])
            result = wiki.audit(root)
            self.assertEqual(result["errors"],[])
            self.assertEqual(result["source_content_hashes_checked"],1)
            self.assertEqual(result["external_snapshots_checked"],0)

    def test_real_wiki_links_and_symbols(self):
        self.assertEqual(wiki.audit(ROOT)["errors"], [])

    def test_fenced_links_are_not_navigation(self):
        self.assertNotIn("missing.md", wiki.without_fences("```md\n[x](missing.md)\n```\nreal"))

    def test_product_links_are_optional_not_missing_local_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.small_wiki(directory, "# Example\n[optional product trace](../native/missing.hpp)\n")
            self.assertEqual(wiki.audit(root)["errors"], [])
            self.assertEqual(wiki.audit(root)["optional_integration_links"], 1)
            self.assertTrue(wiki.audit(root, include_integrations=True)["errors"])

    def test_required_local_source_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.small_wiki(directory, sources=[{"id": "escape", "path": "../source.py", "scope": "local"}])
            self.assertTrue(any("escapes techniques" in value for value in wiki.audit(root)["errors"]))

    def small_wiki(self, directory, readme="# Example\n", sources=None):
        root = Path(directory) / "wiki"
        (root / "references").mkdir(parents=True)
        (root / "README.md").write_text(readme)
        (root / "references/sources.json").write_text(json.dumps({"sources": sources or []}))
        return root

    def test_broken_link_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.small_wiki(directory, "# Example\n[broken](missing.md)\n")
            self.assertTrue(any("missing link" in error for error in wiki.audit(root)["errors"]))

    def test_orphan_page_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.small_wiki(directory)
            (root / "orphan.md").write_text("# Orphan\n")
            self.assertTrue(any("no incoming" in error for error in wiki.audit(root)["errors"]))

    def test_optional_external_missing_warns_required_fails(self):
        for optional in (True, False):
            with self.subTest(optional=optional), tempfile.TemporaryDirectory() as directory:
                root = self.small_wiki(directory, sources=[{"id": "missing", "path": "missing-source", "optional": optional}])
                result = wiki.audit(root)
                self.assertTrue(result["warnings"] if optional else result["errors"])
                if optional:
                    self.assertFalse(result["errors"])

    def test_source_symbols_and_snapshot_drift_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.cpp"
            source.write_text("class Current {};\n")
            root = self.small_wiki(directory, sources=[{"id": "source", "path": "source.cpp", "symbols": ["DeletedSymbol"], "optional": True, "sha256": "0" * 64}])
            result = wiki.audit(root)
            self.assertTrue(any("symbol missing" in error for error in result["errors"]))
            self.assertTrue(any("snapshot changed" in warning for warning in result["warnings"]))


class StandaloneTests(unittest.TestCase):
    def test_real_learning_assets_and_imports_are_local(self):
        self.assertEqual(standalone.audit(ROOT)["errors"], [])

    def test_isolated_copy_excludes_caches_and_raw_captures_but_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            artifacts = (
                ".cache/compiled.bin", "labs/.pytest_cache/state.json",
                "labs/ane/.aneforge_cache/weights.bin", "integrations/.aneforge-cache/tuning.json",
                "integrations/.build/release/probe", "integrations/.swiftpm/state.json",
                "labs/DerivedData/Build/temp.o", "labs/CMakeFiles/cmake.check_cache",
                "labs/metal/raw.gputrace/metadata.json", "labs/metal/raw.trace/data",
                "labs/metal/raw.gputrace.zip", "labs/metal/raw.trace.zip",
                "integrations/ffn-bf16.tcbf16z", "review/raw-pair.tcbf16z",
                "outputs/report.json", "labs/private/session.json",
            )
            portable = ("labs/ane/probe.py", "review/evidence.json", "examples/tiny.npy")
            for relative in artifacts + portable:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n")

            def child(command, **kwargs):
                destination = Path(kwargs["cwd"]) / "techniques"
                for relative in artifacts:
                    self.assertFalse((destination / relative).exists(), relative)
                for relative in portable:
                    self.assertEqual((destination / relative).read_text(), "fixture\n", relative)
                return standalone.subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

            with patch.object(standalone, "audit", return_value={"errors": []}), \
                    patch.object(standalone.subprocess, "run", side_effect=child) as run:
                self.assertEqual(standalone.isolate(root)["errors"], [])
                self.assertEqual(run.call_count, 5)

    def copy_wiki(self, directory):
        target = Path(directory) / "techniques"
        shutil.copytree(ROOT, target)
        return target

    def test_escaping_required_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.copy_wiki(directory)
            path = root / "labs/catalog.json"
            value = json.loads(path.read_text())
            value["entries"][0]["code"].append("../native/secret.hpp")
            path.write_text(json.dumps(value))
            self.assertTrue(any("non-local code" in value for value in standalone.audit(root)["errors"]))

    def test_dynamic_external_import_and_native_include_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.copy_wiki(directory)
            (root / "labs/forbidden.py").write_text("import sibling_runtime\nimportlib.util.spec_from_file_location('x', '../x.py')\n")
            (root / "labs/forbidden.mm").write_text('#include "../../../native/runtime.hpp"\n')
            errors = standalone.audit(root)["errors"]
            self.assertTrue(any("undeclared import" in value for value in errors))
            self.assertTrue(any("dynamic source loading" in value for value in errors))
            self.assertTrue(any("native include escapes" in value for value in errors))

    def test_missing_named_stage_and_design_fail_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.copy_wiki(directory)
            path = root / "references/absorbed-designs.json"
            value = json.loads(path.read_text())
            value["stages"].pop()
            value["designs"].pop()
            path.write_text(json.dumps(value))
            errors = standalone.audit(root)["errors"]
            self.assertTrue(any("model-stage coverage" in value for value in errors))
            self.assertTrue(any("absorbed-design coverage" in value for value in errors))


if __name__ == "__main__":
    unittest.main()

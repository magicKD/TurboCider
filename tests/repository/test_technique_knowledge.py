"""Documentation/retrieval maintenance only; no saved experiment validation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "techniques"
sys.path.insert(0, str(ROOT / "scripts"))
import knowledge
from check_wiki import audit
from generate_query_index import render


class KnowledgeOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = knowledge.load(ROOT)

    def test_no_data_payloads_or_experiment_packages(self):
        self.assertEqual(list(ROOT.rglob("*.json")), [])
        self.assertEqual(list(ROOT.rglob("*.jsonl")), [])
        self.assertEqual(list(ROOT.rglob("*.ndjson")), [])
        for folder in ("data", "labs", "integrations", "tests", "review", "candidates",
                       "algorithms/archive", "metal/evidence", "gpu_ane/evidence", "ane/records"):
            self.assertFalse((ROOT / folder).exists(), folder)
        # Guard against simply renaming experiment data into another format.
        for path in ROOT.rglob("*"):
            if path.is_file():
                self.assertNotIn(path.suffix, (".yaml", ".toml", ".csv", ".npy", ".npz", ".gz", ".zip"))
                self.assertLess(path.stat().st_size, 300000, str(path))
        self.assertLess(sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()), 10000000)

    def test_all_notes_index_without_a_catalog(self):
        self.assertEqual(len(self.index["nodes"]), len(list(ROOT.rglob("*.md"))))
        self.assertTrue(all(n["kind"] in ("technique", "source") for n in self.index["nodes"]))

    def test_deepcache_learn_finds_method_before_case(self):
        result = knowledge.query(self.index, "DeepCache", limit=3)
        self.assertEqual(result["results"][0]["path"], "algorithms/methods/deepcache.md")
        self.assertTrue(any(n["page_type"] == ["case-study"] for n in result["results"][1:]))

    def test_small_m_implementation_finds_the_relevant_case(self):
        result = knowledge.query(self.index, "small-M", intent="implement")
        self.assertEqual(result["results"][0]["path"],
                         "metal/case-studies/mlx-small-m-wide-gemv-and-head-attribution.md")

    def test_source_reading_is_not_experiment_qualification(self):
        result = knowledge.query(self.index, "MLX", intent="sources")
        self.assertEqual(result["results"][0]["kind"], "source")
        self.assertFalse(result["retrieval_grants_qualification"])
        self.assertFalse(any(n["path"].endswith(".json") for n in result["results"]))

    def test_legacy_evidence_intent_explicitly_reports_no_payloads(self):
        result = knowledge.query(self.index, "MLX", intent="evidence")
        self.assertEqual(result["intent"], "sources")
        self.assertTrue(result["warnings"])

    def test_controlled_topic_filters_and_aliases(self):
        result = knowledge.query(self.index, "共享", {"section": ["gpu_ane"], "page_type": ["method"]})
        self.assertEqual(result["results"][0]["path"], "gpu_ane/memory-and-handoff.md")
        self.assertEqual(knowledge.keyword_groups("前馈网络 ffn feed forward network")[0]["canonical"], "ffn")
        self.assertEqual(len(knowledge.keyword_groups("前馈网络 ffn feed forward network")), 1)
        self.assertNotIn("ane", knowledge.keyword_groups("NAX")[0]["variants"])
        for filters in ({"device": ["NAX"]}, {"representation": ["made-up"]}, {"kind": ["evidence"]}):
            with self.subTest(filters=filters), self.assertRaises(ValueError):
                knowledge.query(self.index, filters=filters)

    def test_moved_case_keeps_legacy_id_as_alias(self):
        node = next(n for n in self.index["nodes"] if n["aliases"])
        result = knowledge.get_page(self.index, node["aliases"][0])
        self.assertEqual(result["pages"][0]["id"], node["id"])
        self.assertEqual(node["page_type"], ["case-study"])

    def test_links_and_generated_index(self):
        report = audit(ROOT)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["warnings"], [])
        self.assertEqual(render(self.index), (ROOT / "queries/README.md").read_text())

    def test_get_page_is_bounded_and_only_expands_small_linked_code(self):
        result = knowledge.get_page(self.index, "designs/hybrid-ffn.md", include_code=True, maximum_bytes=24000)
        self.assertTrue(result["code"])
        used = sum(len(p["body"].encode()) for p in result["pages"] + result["code"])
        self.assertEqual(used, result["body_bytes_used"])
        self.assertLessEqual(used, 24000)
        self.assertTrue(all(p["path"].startswith("examples/") for p in result["code"]))

    def test_cli_does_not_depend_on_json_files(self):
        child = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/query.py"),
                                "DeepCache", "--limit", "1"], capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout)["results"][0]["path"], "algorithms/methods/deepcache.md")

    def test_examples_are_readable_stdlib_code_not_device_clients(self):
        for path in (ROOT / "examples").rglob("*.py"):
            compile(path.read_text(), str(path), "exec")
        sys.path.insert(0, str(ROOT / "examples"))
        from reference.contracts import OwnedGraph
        from reference.operators import online_attention
        self.assertTrue(callable(OwnedGraph) and callable(online_attention))


class MarkdownFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "wiki"
        self.root.mkdir()
        self.page = self.root / "before.md"
        self.page.write_text("# Fixture\n\n<!-- knowledge\nid: technique-fixture\ntype: method\n-->\n\n" + "中文" * 600)

    def test_explicit_identity_survives_rename(self):
        before = knowledge.load(self.root)
        self.page.rename(self.root / "after.md")
        after = knowledge.load(self.root)
        self.assertEqual(set(before["by_id"]), set(after["by_id"]))

    def test_utf8_pagination_and_invalid_offsets(self):
        index = knowledge.load(self.root)
        first = knowledge.get_page(index, "technique-fixture", maximum_bytes=256)["pages"][0]
        self.assertTrue(first["truncated"])
        rest = knowledge.get_page(index, "technique-fixture", offset_bytes=first["next_byte_offset"])["pages"][0]
        self.assertEqual(first["body"] + rest["body"], self.page.read_text())
        for lookup in ("../private.md", str(self.page), "missing.md"):
            with self.subTest(lookup=lookup), self.assertRaises(ValueError):
                knowledge.get_page(index, lookup)
        offset = len(self.page.read_text().split("中文")[0].encode()) + 1
        with self.assertRaises(ValueError):
            knowledge.get_page(index, "technique-fixture", offset_bytes=offset)

    def test_duplicate_metadata_and_escaping_symlinks_rejected(self):
        other = self.root / "other.md"
        other.write_text(self.page.read_text())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            knowledge.load(self.root)
        other.unlink()
        outside = Path(self.temp.name) / "outside.md"
        outside.write_text("# Private outside knowledge\n")
        other.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escaping"):
            knowledge.load(self.root)

    def test_reading_note_accepts_no_code_or_experiment_metadata(self):
        node = knowledge.load(self.root)["nodes"][0]
        self.assertEqual(node["code"], [])
        self.assertNotIn("version_review", node)
        self.assertNotIn("executable", node)


if __name__ == "__main__":
    unittest.main()

"""Offline retrieval tests: registered assets only, bounded and deterministic."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("knowledge", ROOT / "scripts/knowledge.py")
knowledge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(knowledge)


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "wiki"
        for folder in ("labs", "references", "metal"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        (self.root / "metal/example.md").write_text("# Example GPU layout 布局\n" + "body " * 100, encoding="utf-8")
        (self.root / "labs/example.py").write_text("# source witness\n", encoding="utf-8")
        (self.root / "labs/other.py").write_text("# another local asset\n", encoding="utf-8")
        self.catalog = {"entries": [{"chapter": "metal/example.md", "code": ["labs/example.py", "labs/other.py"],
                                     "cases": ["example-case"], "controls": ["independent oracle"],
                                     "revisit_when": "new layout"}]}
        self.sources = {"sources": [{"id": "local-example", "scope": "local", "path": "labs/example.py",
                                      "role": "source witness", "symbols": ["witness"]},
                                     {"id": "external", "scope": "integration", "path": "../missing.py"}]}
        self.save()

    def save(self):
        (self.root / "labs/catalog.json").write_text(json.dumps(self.catalog), encoding="utf-8")
        (self.root / "references/sources.json").write_text(json.dumps(self.sources), encoding="utf-8")

    def test_real_catalog_loads_without_uncreated_metadata_files(self):
        index = knowledge.load(ROOT)
        catalog = json.loads((ROOT / "labs/catalog.json").read_text())
        self.assertEqual(sum(node["kind"] == "technique" for node in index["nodes"]), len(catalog["entries"]))
        self.assertTrue(knowledge.query(index, "LoRA")["results"])

    def test_only_local_registered_sources_are_loaded(self):
        index = knowledge.load(self.root)
        self.assertEqual(len(index["nodes"]), 2)
        self.assertNotIn("source-external", index["by_id"])
        self.assertEqual(index["by_id"]["technique-metal-example"]["sources"], ["source-local-example"])

    def test_declared_source_cards_are_resolved_without_external_access(self):
        (self.root / "references/card.md").write_text("# Upstream mechanism\nsource facts\n")
        self.sources["sources"].append({"id": "source-card", "scope": "local", "path": "references/card.md",
                                        "role": "upstream source", "upstream": {"repository": "owner/project"}})
        self.catalog["entries"][0]["source_ids"] = ["source-card", "local-example"]
        self.save()
        index = knowledge.load(self.root)
        self.assertEqual(index["by_id"]["technique-metal-example"]["sources"], ["source-source-card", "source-local-example"])
        result = knowledge.get_page(index, "metal/example.md", follow_sources=True)
        self.assertEqual(len(result["pages"]), 3)
        self.assertEqual(result["pages"][1]["metadata"]["upstream"]["repository"], "owner/project")

    def test_unknown_or_malformed_source_card_links_are_rejected(self):
        for value in (["missing"], [{}], "local-example"):
            self.catalog["entries"][0]["source_ids"] = value
            self.save()
            with self.subTest(value=value), self.assertRaises(ValueError):
                knowledge.load(self.root)

    def test_official_sdk_metadata_is_retained_without_a_fake_git_commit(self):
        folder = self.root / "sources/official"
        folder.mkdir(parents=True)
        (folder / "example.md").write_text("# Official example\nAPI contract\n", encoding="utf-8")
        self.sources["sources"].append({"id": "official-example", "scope": "local",
            "path": "sources/official/example.md", "role": "official SDK contract",
            "official": {"publisher": "API Publisher", "snapshot_identity": "SDK fixture"},
            "evidence": {"runtime": "unknown"}})
        self.catalog["entries"][0]["source_ids"] = ["official-example"]
        self.save()
        index = knowledge.load(self.root)
        node = index["by_id"]["source-official-example"]
        self.assertEqual(node["official"]["snapshot_identity"], "SDK fixture")
        self.assertNotIn("upstream", node)
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            from generate_source_index import render
            text = render(index)
        finally:
            sys.path.pop(0)
        self.assertIn("| [API Publisher](official/example.md) | SDK fixture | unknown |", text)

    def test_local_evidence_has_a_distinct_kind_and_opt_in_expansion(self):
        (self.root / "references/evidence.json").write_text('{"status":"failed"}\n')
        self.sources["sources"].append({"id": "local-run", "kind": "evidence", "scope": "local",
                                        "path": "references/evidence.json", "role": "negative experiment"})
        self.catalog["entries"][0]["evidence_ids"] = ["local-run"]
        self.save()
        index = knowledge.load(self.root)
        result = knowledge.query(index, filters={"kind": ["evidence"]})
        self.assertEqual([node["id"] for node in result["results"]], ["evidence-local-run"])
        self.assertEqual(len(knowledge.get_page(index, "metal/example.md")["pages"]), 1)
        expanded = knowledge.get_page(index, "metal/example.md", follow_evidence=True)
        self.assertEqual(expanded["pages"][-1]["id"], "evidence-local-run")
        self.assertIn('"failed"', expanded["pages"][-1]["body"])

    def test_unknown_evidence_reference_is_rejected(self):
        self.catalog["entries"][0]["evidence_ids"] = ["missing-run"]
        self.save()
        with self.assertRaisesRegex(ValueError, "evidence"):
            knowledge.load(self.root)

    def test_keyword_filters_and_order_are_deterministic(self):
        index = knowledge.load(self.root)
        first = knowledge.query(index, "布局", {"section": ["METAL"], "kind": ["technique"]})
        self.assertEqual(first["results"][0]["id"], "technique-metal-example")
        self.assertEqual(first, knowledge.query(knowledge.load(self.root), "布局", {"section": ["METAL"], "kind": ["technique"]}))
        self.assertEqual(knowledge.query(index, "missing-unrelated")["results"], [])

    def test_invalid_queries_fail_cleanly(self):
        index = knowledge.load(self.root)
        for filters in ({"device": ["ANE"]}, {42: ["ANE"]}, {"kind": ["missing"]}, {"section": "metal"},
                        {"kind": [["technique"]]}, {"kind": ["source", "source"]}):
            with self.subTest(filters=filters), self.assertRaises(ValueError):
                knowledge.query(index, filters=filters)
        for limit in (True, 0, 101, "1"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                knowledge.query(index, limit=limit)

    def test_malformed_catalog_rejected_without_type_error(self):
        for value in (None, "code.py", [{}], [True], ["same", "same"]):
            with self.subTest(value=value):
                self.catalog["entries"][0]["code"] = value
                self.save()
                with self.assertRaises(ValueError):
                    knowledge.load(self.root)

    def test_duplicate_chapter_or_source_is_rejected(self):
        self.catalog["entries"].append(copy.deepcopy(self.catalog["entries"][0]))
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            knowledge.load(self.root)
        self.catalog["entries"].pop()
        self.sources["sources"].append(copy.deepcopy(self.sources["sources"][0]))
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            knowledge.load(self.root)

    def test_unregistered_paths_and_bad_budgets_are_rejected(self):
        index = knowledge.load(self.root)
        for lookup in ("../private.txt", str(self.root / "metal/example.md"), "labs/other.py", []):
            with self.subTest(lookup=lookup), self.assertRaises(ValueError):
                knowledge.get_page(index, lookup)
        for budget in (True, 0, 255, 256001, "256"):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                knowledge.get_page(index, "metal/example.md", maximum_bytes=budget)

    def test_symlink_escape_is_rechecked_during_expansion(self):
        index = knowledge.load(self.root)
        outside = Path(self.temporary.name) / "private.txt"
        outside.write_text("not a registered asset")
        (self.root / "metal/escape.md").symlink_to(outside)
        self.catalog["entries"][0]["chapter"] = "metal/escape.md"
        self.save()
        with self.assertRaises(ValueError):
            knowledge.load(self.root)
        (self.root / "labs/escape.py").symlink_to(outside)
        index["by_id"]["technique-metal-example"]["code"] = ["labs/escape.py"]
        with self.assertRaises(ValueError):
            knowledge.get_page(index, "metal/example.md", include_code=True)

    def test_manifest_symlink_escape_is_rejected_before_reading_json(self):
        outside = Path(self.temporary.name) / "private.txt"
        outside.write_text("not JSON; must not be opened")
        for relative in ("labs/catalog.json", "references/sources.json"):
            with self.subTest(relative=relative):
                manifest = self.root / relative
                manifest.unlink()
                manifest.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "escaping"):
                    knowledge.load(self.root)
                manifest.unlink()
                self.save()

    def test_budget_discloses_truncation_and_omitted_sources_and_code(self):
        result = knowledge.get_page(knowledge.load(self.root), "metal/example.md", True, True, 256)
        self.assertEqual(result["body_bytes_used"], 256)
        self.assertTrue(result["pages"][0]["truncated"])
        self.assertEqual(result["omitted_pages"], ["source-local-example"])
        self.assertEqual(result["omitted_code"], ["labs/example.py", "labs/other.py"])

    def test_source_expansion_does_not_duplicate_code(self):
        result = knowledge.get_page(knowledge.load(self.root), "metal/example.md", True, True)
        self.assertEqual([page["id"] for page in result["pages"]], ["technique-metal-example", "source-local-example"])
        self.assertEqual([item["path"] for item in result["code"]], ["labs/other.py"])
        self.assertEqual(result["omitted_pages"], [])

    def test_paging_keeps_utf8_characters_and_discloses_next_offset(self):
        original = "# Example\n" + "汉" * 150
        (self.root / "metal/example.md").write_text(original, encoding="utf-8")
        index = knowledge.load(self.root)
        first = knowledge.get_page(index, "metal/example.md", maximum_bytes=256)["pages"][0]
        self.assertTrue(first["truncated"])
        second = knowledge.get_page(index, "metal/example.md", maximum_bytes=256,
                                    offset_bytes=first["next_byte_offset"])["pages"][0]
        self.assertEqual(first["body"] + second["body"], original)
        self.assertIsNone(second["next_byte_offset"])
        for offset in (True, -1, 10000, len("# Example\n".encode()) + 1):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                knowledge.get_page(index, "metal/example.md", offset_bytes=offset)

    def test_cli_outputs_json_and_invalid_lookup_fails_cleanly(self):
        for script, arguments in (("query.py", ["布局"]), ("get_page.py", ["metal/example.md"])):
            child = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / script),
                                    "--root", str(self.root), *arguments], capture_output=True, text=True, timeout=10)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertIsInstance(json.loads(child.stdout), dict)
        child = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/get_page.py"),
                                "--root", str(self.root), "../private.txt"], capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 2)
        self.assertIn("error", json.loads(child.stderr))


if __name__ == "__main__":
    unittest.main()

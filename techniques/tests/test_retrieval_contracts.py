"""No-device retrieval tests: controlled topics, safe aliases and historical review."""
import copy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import knowledge
import retrieval_contracts as contracts
from check_knowledge import audit
from generate_query_index import render


class RetrievalContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for folder in ("labs", "references", "metal", "review", "data"):
            (self.root / folder).mkdir()
        (self.root / "metal/ffn.md").write_text("# GPU FFN example\nA synthetic topic, not a hardware result.\n")
        (self.root / "labs/probe.py").write_text('raise RuntimeError("retrieval must never execute source")\n')
        self.evidence = self.root / "review/failure.json"
        self.evidence.write_text('{"status":"failed","comparisons":null}\n')
        self.vocabulary = json.loads((ROOT / "data/vocabulary.json").read_text())
        self.facets = {field: [] for field in contracts.FACETS}
        self.facets.update(stage=["dit", "ffn"], operator=["swiglu"], backend=["metal"], device=["gpu"],
                           symptom=["wrong-layout"], semantics=["numerically-equivalent"], api=["public"])
        self.catalog = {"retrieval_schema_version": 1, "entries": [{"chapter": "metal/ffn.md", "facets": self.facets,
            "code": ["labs/probe.py"], "cases": ["synthetic"], "controls": ["data only", "not executed"],
            "source_ids": ["origin"], "evidence_ids": ["failed"], "revisit_when": "new source/context"}]}
        self.sources = {"sources": [{"id": "origin", "scope": "local", "path": "labs/probe.py", "role": "source, not a measured result"},
                                    {"id": "failed", "kind": "evidence", "scope": "local", "path": "review/failure.json", "role": "failed synthetic record"}]}
        self.claim = {"id": "claim-synthetic-only", "statement": "Synthetic review contract; not a device result.",
                      "applies_to": ["technique-metal-ffn", "evidence-failed"],
                      "bindings": {"evidence-failed": hashlib.sha256(self.evidence.read_bytes()).hexdigest()},
                      "observed_context": {"mlx": "0.32.0", "os_build": "25G83"},
                      "reviewed_at": "2026-10-16", "review_after_days": 90}
        self.claims = {"schema": "tc-version-review-v1", "claims": [self.claim]}
        self.save()

    def save(self):
        for relative, value in (("labs/catalog.json", self.catalog), ("references/sources.json", self.sources),
                                ("data/vocabulary.json", self.vocabulary), ("data/version-claims.json", self.claims)):
            (self.root / relative).write_text(json.dumps(value))

    def query(self, **kwargs):
        return knowledge.query(knowledge.load(self.root), as_of="2026-10-16", **kwargs)

    def test_all_real_chapters_have_explicit_canonical_topic_metadata(self):
        index = knowledge.load(ROOT)
        cards = json.loads((ROOT / "labs/catalog.json").read_text())["entries"]
        nodes = [n for n in index["nodes"] if n["kind"] == "technique"]
        self.assertEqual(len(nodes), len(cards))
        self.assertTrue(all(n["topic_metadata_declared"] for n in nodes))
        for node in nodes:
            self.assertEqual(set(contracts.FACETS) - set(node), set())
        self.assertEqual(render(index), (ROOT / "queries/README.md").read_text())

    def test_filters_are_conjunctions_and_typed_aliases_are_disclosed(self):
        result = self.query(filters={"stage": ["dit", "前馈网络"], "backend": ["metal"], "api": ["公开"]})
        self.assertEqual(result["filters"]["stage"], ["dit", "ffn"])
        self.assertEqual([n["id"] for n in result["results"]], ["technique-metal-ffn"])
        self.assertEqual(self.query(filters={"backend": ["metal", "coreml"]})["results"], [])
        self.assertFalse(result["retrieval_grants_qualification"])

    def test_aliases_cannot_change_source_or_evidence_state(self):
        result = self.query(text="前馈网络")
        self.assertEqual(result["results"][0]["id"], "technique-metal-ffn")
        failed = knowledge.get_page(knowledge.load(self.root), "review/failure.json", as_of="2026-10-16")
        self.assertEqual(json.loads(failed["pages"][0]["body"])["status"], "failed")
        self.assertFalse(failed["pages"][0]["metadata"]["retrieval_grants_qualification"])

    def test_sources_do_not_inherit_topics_from_associated_techniques(self):
        self.assertEqual(self.query(filters={"kind": ["source"], "device": ["gpu"]})["results"], [])
        result = self.query(filters={"kind": ["source"]})
        self.assertEqual(result["results"][0]["device"], [])
        self.assertFalse(result["results"][0]["topic_metadata_declared"])
        self.assertGreater(result["metadata_coverage"]["unclassified_by_field"]["device"], 0)

    def test_alias_groups_score_once_and_find_embedded_chinese_phrases(self):
        aliases = contracts.vocabulary(self.vocabulary)["keywords"]
        groups = contracts.keyword_groups("前馈网络 ffn feed forward network", aliases)
        self.assertEqual([g["canonical"] for g in groups], ["ffn"])
        self.assertIn("ffn", [g["canonical"] for g in contracts.keyword_groups("量化前馈网络慢", aliases)])
        self.assertEqual([g["canonical"] for g in contracts.keyword_groups("NAX", aliases)], ["nax"])
        self.assertNotIn("ane", contracts.keyword_groups("NAX", aliases)[0]["variants"])
        self.assertEqual([g["canonical"] for g in contracts.keyword_groups("Straße 前馈网络", aliases)], ["strasse", "ffn"])

    def test_nax_ane_formats_containers_and_skus_stay_distinct(self):
        for target, alias, field in (("ane", "NAX", "device"), ("q8", "fp8", "representation"),
                                     ("q4", "gguf", "representation"), ("gpu", "M4 Max", "device"),
                                     ("gpu", "Apple M5 Pro", "device")):
            document = copy.deepcopy(self.vocabulary)
            document["aliases"].setdefault(field, {}).setdefault(target, []).append(alias)
            with self.subTest(alias=alias), self.assertRaises(ValueError):
                contracts.vocabulary(document)
        for value in ("NAX", "M4", "M4 Max"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.query(filters={"device": [value]})

    def test_alias_chains_ambiguity_unknown_targets_and_bad_types_fail(self):
        cases = [{"ffn": ["attention"]}, {"ffn": "not-list"}, {"ffn": [True]},
                 {"ffn": ["alias", "ALIAS"]}]
        for rows in cases:
            value = copy.deepcopy(self.vocabulary)
            value["keywords"] = {"ffn": [], "attention": [], **rows}
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                contracts.vocabulary(value)
        value = copy.deepcopy(self.vocabulary)
        value["aliases"]["stage"] = {"missing": ["alias"]}
        with self.assertRaisesRegex(ValueError, "unknown alias target"):
            contracts.vocabulary(value)

    def test_unknown_facets_and_alias_duplicate_filters_are_rejected(self):
        for filters in ({"unknown": ["gpu"]}, {"stage": ["ffn", "前馈网络"]}, {"backend": ["fake"]}):
            with self.subTest(filters=filters), self.assertRaises(ValueError):
                self.query(filters=filters)
        for kind in ("missing-field", "unknown-value", "unknown-field", "boolean"):
            value = copy.deepcopy(self.facets)
            if kind == "missing-field":
                value.pop("device")
            elif kind == "unknown-value":
                value["backend"] = ["fake"]
            elif kind == "unknown-field":
                value["qualifies"] = True
            else:
                value["device"] = [True]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                contracts.facets(value, contracts.vocabulary(self.vocabulary), required=True)

    def test_recorded_context_match_is_not_fresh_compatible_or_qualified(self):
        result = self.query(observed=self.claim["observed_context"])
        row = result["results"][0]["version_review"][0]
        self.assertEqual(row["state"], "recorded-context-only")
        self.assertFalse(row["retrieval_grants_qualification"])
        self.assertNotIn("compatible", row)
        self.assertNotIn("performance_qualified", row)

    def test_missing_and_uncovered_context_cannot_be_silently_matched(self):
        missing = self.query()["results"][0]["version_review"][0]
        self.assertEqual(missing["state"], "context-incomplete")
        self.assertEqual(missing["missing_context_fields"], ["mlx", "os_build"])
        extra = dict(self.claim["observed_context"], soc="unrecorded synthetic SKU")
        row = self.query(observed=extra)["results"][0]["version_review"][0]
        self.assertEqual(row["state"], "context-incomplete")
        self.assertEqual(row["uncovered_context_fields"], ["soc"])

    def test_environment_change_warns_but_does_not_delete_historical_evidence(self):
        result = self.query(observed={"mlx": "0.33.0", "os_build": "different"})
        row = result["results"][0]["version_review"][0]
        self.assertEqual(row["state"], "needs-review")
        self.assertEqual(row["changed_context_fields"], ["mlx", "os_build"])
        self.assertEqual(len(result["results"]), 3)

    def test_content_change_is_checked_again_after_index_loading(self):
        index = knowledge.load(self.root)
        self.evidence.write_text('{"status":"changed-but-still-not-qualified"}\n')
        result = knowledge.get_page(index, "review/failure.json", as_of="2026-10-16")
        row = result["pages"][0]["metadata"]["version_review"][0]
        self.assertEqual(row["state"], "needs-review")
        self.assertEqual(row["changed_bindings"], ["evidence-failed"])
        self.assertEqual(index["claims"][self.claim["id"]]["changed_bindings"], [])

    def test_expiry_and_future_review_dates_have_exact_boundaries(self):
        at = contracts.iso_day(self.claim["reviewed_at"])
        for delta, reason in ((90, "review-window-expired"), (-1, "review-date-is-in-the-future")):
            row = contracts.review(self.claim, self.claim["observed_context"], (at + timedelta(days=delta)).isoformat())
            self.assertEqual(row["state"], "needs-review")
            self.assertIn(reason, row["reasons"])
        self.assertEqual(contracts.review(self.claim, self.claim["observed_context"], (at + timedelta(days=89)).isoformat())["state"], "recorded-context-only")

    def test_context_is_explicit_typed_data_not_a_probe_or_version_range(self):
        for value in ([], {"os": "latest"}, {"mlx": True}, {"mlx": ""}, {"mlx": ["0.32.0"]},
                      {"mlx": "latest"}, {"mlx": ">=0.32.0"}, {"soc": "any"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.query(observed=value)
        for day in (True, "2026/10/16", "2026-02-30"):
            with self.subTest(day=day), self.assertRaises(ValueError):
                knowledge.query(knowledge.load(self.root), as_of=day)

    def test_claims_require_real_ids_source_bindings_and_valid_dates(self):
        for key, value in (("applies_to", ["technique-missing"]), ("bindings", {"technique-metal-ffn": "0" * 64}),
                           ("bindings", {"evidence-failed": True}), ("reviewed_at", "2026-02-30"),
                           ("review_after_days", True), ("observed_context", {})):
            row = copy.deepcopy(self.claim)
            row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                contracts.claim_rows({"schema": "tc-version-review-v1", "claims": [row]}, knowledge.load(self.root)["by_id"])
        self.claims["claims"].append(copy.deepcopy(self.claim))
        self.save()
        with self.assertRaisesRegex(ValueError, "unique"):
            knowledge.load(self.root)

    def test_new_manifests_cannot_escape_root_through_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "outside.json"
            outside.write_text("not JSON; must never be parsed")
            for relative in ("data/vocabulary.json", "data/version-claims.json"):
                path = self.root / relative
                path.unlink()
                path.symlink_to(outside)
                with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "escaping"):
                    knowledge.load(self.root)
                path.unlink()
                self.save()

    def test_generated_index_uses_existing_page_inventory_without_mutation(self):
        index = knowledge.load(self.root)
        before = copy.deepcopy(index["nodes"])
        text = render(index)
        self.assertIn("../metal/ffn.md", text)
        self.assertNotIn("review/failure.json", text)
        self.assertEqual(before, index["nodes"])
        self.assertEqual(text, render(knowledge.load(self.root)))

    def test_source_cards_expand_only_declared_local_code_with_shared_budget(self):
        self.sources["sources"][0]["code"] = ["labs/probe.py"]
        self.save()
        index = knowledge.load(self.root)
        expanded = knowledge.get_page(index, "source-origin", follow_sources=True, include_code=True, as_of="2026-10-16")
        # A page that is itself code is not duplicated or executed.
        self.assertEqual(expanded["code"], [])
        self.assertEqual(len(expanded["pages"]), 1)
        self.assertEqual(index["by_id"]["source-origin"]["sources"], [])
        self.assertIn("must never execute", expanded["pages"][0]["body"])
        (self.root / "labs/helper.py").write_text("# separate local helper\n" * 100)
        self.sources["sources"][0]["code"].append("labs/helper.py")
        self.save()
        bounded = knowledge.get_page(knowledge.load(self.root), "source-origin", include_code=True, maximum_bytes=256, as_of="2026-10-16")
        self.assertEqual(bounded["body_bytes_used"], 256)
        self.assertTrue(bounded["code"][0]["truncated"])
        self.sources["sources"][0]["code"] = ["../outside.py"]
        self.save()
        with self.assertRaises(ValueError):
            knowledge.load(self.root)

    def test_cli_filters_context_json_and_strict_review_status(self):
        def run(script, *args):
            return subprocess.run([sys.executable, "-E", "-S", "-B", str(ROOT / "scripts" / script),
                                   "--root", str(self.root), "--as-of", "2026-10-16", *args],
                                  capture_output=True, text=True, timeout=10)
        result = run("query.py", "前馈网络", "--stage", "dit", "--backend", "metal")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["results"][0]["id"], "technique-metal-ffn")
        changed = run("check_knowledge.py", "--strict", "--context", '{"mlx":"different"}')
        self.assertEqual(changed.returncode, 1, changed.stderr)
        self.assertEqual(json.loads(changed.stdout)["needs_review"], ["claim-synthetic-only"])
        for args in (("--context", "[]"), ("--stage", "fake"), ("--context", "not-json")):
            child = run("query.py", *args)
            self.assertEqual(child.returncode, 2, child.stderr)
            self.assertIn("error", json.loads(child.stderr))
        self.assertEqual(audit(self.root, as_of="2026-10-16")["needs_review"], [])

    def test_declared_null_metadata_and_malformed_top_level_are_not_unclassified(self):
        self.sources["sources"][0]["facets"] = None
        self.save()
        with self.assertRaisesRegex(ValueError, "facets"):
            knowledge.load(self.root)
        for value in (None, [], True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                contracts.vocabulary(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                contracts.claim_rows(value, {})

    def test_duplicate_json_keys_cannot_silently_override_controlled_rules(self):
        path = self.root / "data/vocabulary.json"
        path.write_text('{"schema":"tc-retrieval-vocabulary-v1","schema":"silently-override"}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            knowledge.load(self.root)


if __name__ == "__main__":
    unittest.main()

"""Portable audit source/evidence/denominator gates; not runtime qualification."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/validation"))
import qwen_async_step_pair_audit as audit


class AsyncStepPairAuditTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="tc-async-audit-host-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source_name = "owned-source.py"
        source = "# synthetic host-only source\n"
        (self.root / self.source_name).write_text(source)
        (self.root / "source.json").write_text(json.dumps({self.source_name: source}))
        memory = dict(verified=dict(complete=True, command_exit_code=0))
        for name in ("evidence", "report"):
            path = self.root / f"memory-{name}.json"
            path.write_text(json.dumps(dict(synthetic=name)))
            memory[name] = path.name
            memory[name + "_sha256"] = audit.sha256_file(path)
        self.summary = dict(source_sha256={self.source_name: audit.sha256_file(self.root / self.source_name)},
                            memory=memory)
        root_patch = patch.object(audit, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_current_source_and_saved_snapshot_are_both_required(self):
        audit.verify_sources(self.root, self.summary)
        path = self.root / self.source_name
        original = path.read_text()
        path.write_text("# changed current source\n")
        with self.assertRaises(ValueError):
            audit.verify_sources(self.root, self.summary)
        path.write_text(original)
        (self.root / "source.json").write_text(json.dumps({self.source_name: "# changed snapshot\n"}))
        with self.assertRaises(ValueError):
            audit.verify_sources(self.root, self.summary)

    def test_changed_evidence_or_memory_report_is_rejected(self):
        for name in ("evidence", "report"):
            summary = copy.deepcopy(self.summary)
            summary["memory"][name + "_sha256"] = "0" * 64
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.verify_sources(self.root, summary)

    def test_timeout_incomplete_sampling_and_nonzero_exit_are_not_completion(self):
        for changes in (dict(complete=False), dict(command_exit_code=1),
                        dict(command_exit_code=None)):
            summary = copy.deepcopy(self.summary)
            summary["memory"]["verified"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                audit.verify_sources(self.root, summary)

    def test_normalization_retains_sampler_seed_and_reference_content(self):
        request = dict(output="a.png", dump_tensors="raw", steps=5, seed=73,
                       prompt="same", inputs=[dict(path="ref.png")],
                       loras=[dict(path="pinned.safetensors", strength=1.0)])
        path = self.root / "request.json"
        path.write_text(json.dumps(request))
        expected = {key: value for key, value in request.items()
                    if key not in ("output", "dump_tensors")}
        self.assertEqual(audit.normalized(path), expected)

    def test_window_checks_actual_library_runtime_sequence_and_reanalysis(self):
        seq = [dict(synthetic=True)]
        analysis = dict(median_seconds=6.0)
        summary = dict(self.summary, status="complete", asset_sha256=dict(library="library-sha"),
                       mode="timing", case="generate", steps=5, async_pair="on", prefix=False,
                       frozen_sequence=seq, analysis=analysis)
        report = dict(loaded_runtime_id="runtime-id")
        (self.root / "trial").mkdir()
        (self.root / "trial/report.json").write_text(json.dumps(report))
        with patch.object(audit.screen, "points", return_value=seq), \
                patch.object(audit.screen, "analyze", return_value=analysis), \
                patch.object(audit.screen.base.prefix_screen, "images", return_value=[]):
            (self.root / "summary.json").write_text(json.dumps(summary))
            self.assertEqual(audit.window(self.root, "library-sha", "runtime-id")[2], seq)
            for changes in (dict(status="running"), dict(asset_sha256=dict(library="other")),
                            dict(frozen_sequence=[]), dict(analysis=dict(median_seconds=0.1))):
                (self.root / "summary.json").write_text(json.dumps({**summary, **changes}))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    audit.window(self.root, "library-sha", "runtime-id")
            (self.root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                audit.window(self.root, "library-sha", "other-runtime")


if __name__ == "__main__":
    unittest.main()

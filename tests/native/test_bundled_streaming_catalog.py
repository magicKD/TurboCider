#!/usr/bin/env python3
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
import generate_bundled_streaming_catalog as bundled
from test_streaming_test_catalog import record


class BundledCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.inventory = self.root / "catalog.json"
        self.runtime = self.root / "runtime.json"
        self.runtime.write_text(json.dumps({"runtime_build_id": "test-build"}))
        self.catalog = {"schema": bundled.SCHEMA, "revision": record()["catalog_revision"], "records": []}

    def generate(self):
        self.inventory.write_text(json.dumps(self.catalog))
        return bundled.generate(self.inventory, self.runtime)

    def test_empty_catalog_and_independent_revision(self):
        header, manifest = self.generate()
        self.assertEqual(manifest["record_digests"], [])
        self.catalog["revision"] = "separate-data-revision"
        updated, other = self.generate()
        self.assertNotEqual(header, updated)
        self.assertNotEqual(manifest["input_sha256"], other["input_sha256"])
        self.assertEqual(manifest["runtime_build_id"], other["runtime_build_id"])

    def test_entries_require_original_evidence_and_reverification(self):
        self.catalog["records"] = [{"status": "verified", "record": record()}]
        with self.assertRaisesRegex(ValueError, "original builder inputs"):
            self.generate()
        entry = {key: key + ".json" for key in bundled.INPUT_KEYS}
        entry["expected_record_digest"] = record()["canonical_record_digest"]
        self.catalog["records"] = [entry]
        with patch.object(bundled, "build_record", return_value={"record": record()}) as builder:
            _, manifest = self.generate()
            self.assertEqual(manifest["record_digests"], [entry["expected_record_digest"]])
            self.assertEqual(builder.call_args.args, tuple(self.root / (key + ".json") for key in
                ("bundle", "record_input", "review", "performance_bundle", "default_bundle", "swap_bundle")))
        # With no verifier stub, the missing evidence must fail, not silently
        # accept the expected digest or a caller-provided verified label.
        with self.assertRaises((OSError, ValueError, bundled.CatalogBuildError, bundled.EvidenceError)):
            self.generate()

    def test_record_digest_runtime_channel_and_duplicates_rejected(self):
        original = record()
        for key, value in [("runtime", {**original["runtime"], "turbocider_build_id": "old-build"}),
                           ("canonical_record_digest", "0" * 64),
                           ("release", {**original["release"], "channel": "staging"})]:
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed[key] = value
                with self.assertRaises((ValueError, bundled.CatalogBuildError)):
                    bundled.render_catalog(self.catalog["revision"], [changed], "test-build")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            bundled.render_catalog(self.catalog["revision"], [original, original], "test-build")

    def test_cli_checks_generated_header_and_inventory_after_build(self):
        self.generate()
        output = self.root / "generated"
        command = [sys.executable, str(ROOT / "tools/native/generate_bundled_streaming_catalog.py"),
                   "--inventory", str(self.inventory), "--runtime-manifest", str(self.runtime),
                   "--output", str(output)]
        subprocess.run(command, check=True, capture_output=True)
        subprocess.run(command + ["--verify"], check=True, capture_output=True)
        header = output / "turbocider_bundled_catalog_generated.hpp"
        header.write_text(header.read_text() + "// changed\n")
        failure = subprocess.run(command + ["--verify"], capture_output=True, text=True)
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("inputs changed", failure.stderr)
        subprocess.run(command, check=True, capture_output=True)
        self.catalog["revision"] = "changed-after-compilation"
        self.inventory.write_text(json.dumps(self.catalog))
        failure = subprocess.run(command + ["--verify"], capture_output=True, text=True)
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("inputs changed", failure.stderr)

    def test_emission_rejects_native_integer_overflow_and_code_keys(self):
        with self.assertRaisesRegex(ValueError, "integer range"):
            bundled.assignments("r", {"width": 1 << 32})
        with self.assertRaisesRegex(ValueError, "invalid catalog member"):
            bundled.assignments("r", {"x; abort()": 1})
        self.assertEqual(bundled.cpp_string('"\\\n'), '"\\042\\134\\012"')
        with self.assertRaises(ValueError):
            bundled.cpp_string("a\0b")

    def test_generated_v1_v2_data_pass_native_canonical_validation(self):
        native = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()
        if not (native / "libturbocider.dylib").is_file():
            self.skipTest("native library is required for generated C++ roundtrip")
        legacy = record()
        portable = copy.deepcopy(legacy)
        portable["id"] += "-portable"
        portable["source"]["identity_version"] = 2
        portable["release"]["channel"] = "public-stable"
        del portable["source"]["source_snapshot_digest"]
        portable["canonical_record_digest"] = bundled.canonical_record_digest(portable)
        (self.root / "turbocider_bundled_catalog_generated.hpp").write_text(
            bundled.render_catalog(self.catalog["revision"], [legacy, portable], "test-build"))
        driver = self.root / "roundtrip.cpp"
        driver.write_text('''#include "runtime/streaming/preset_catalog.hpp"
#include <cassert>
int main() {
const auto &c = tc::streaming::production_streaming_preset_catalog();
assert(c.records.size() == 2);
assert(c.records[0].source.identity_version == 1);
assert(c.records[1].source.identity_version == 2);
assert(c.records[1].source.source_snapshot_digest.empty());
for (const auto &r : c.records) tc::streaming::validate_streaming_preset_record(r, c.revision);
}
''')
        compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
        binary = self.root / "roundtrip"
        subprocess.run([compiler, "-std=c++20", "-DTURBOCIDER_HAS_BUNDLED_CATALOG=1",
                        "-I", str(ROOT / "native"), "-I", str(ROOT / "native/core"),
                        "-isysroot", subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip(),
                        "-I", str(self.root), str(driver),
                        str(ROOT / "native/runtime/streaming/preset_catalog.cpp"),
                        "-L", str(native), "-lturbocider", "-Wl,-rpath," + str(native),
                        "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

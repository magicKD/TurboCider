#!/usr/bin/env python3
"""Independent native processes: durable first-use SHA and invalidation."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class SourceRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="tc-source-registration-build-")
        cls.binary = Path(cls.build.name) / "registration-test"
        compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
        sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
        flags = ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread", "-isysroot", sdk]
        sanitizer = os.environ.get("TC_STREAMING_SANITIZER", "")
        if sanitizer:
            if sanitizer not in ("address,undefined", "thread"):
                raise ValueError("unsupported sanitizer")
            flags += ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=" + sanitizer]
        sources = ["tests/native/source_registration_test.cpp", "native/runtime/streaming/source_lease.cpp",
                   "native/runtime/streaming/canonical_encoding.cpp", "native/runtime/memory_manifest.cpp",
                   "native/core/common.cpp", "native/core/json_keys.cpp"]
        subprocess.run([compiler, *flags, *(str(ROOT / p) for p in sources), "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix="tc-source-registration-")
        self.addCleanup(self.fixture.cleanup)
        self.root = Path(self.fixture.name)
        self.cache = self.root / "cache"
        self.source = self.root / "model.bin"
        self.source.write_bytes(b"original source bytes")
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.env = dict(os.environ, TURBOCIDER_SOURCE_REGISTRATION_CACHE=str(self.cache))

    def run_native(self, mode="registered", source=None, expected=None, second=None, error=None):
        command = [str(self.binary), mode, str(source or self.source)]
        if expected is not None or second is not None:
            command.append(expected or "")
        if second is not None:
            command.append(str(second))
        result = subprocess.run(command, env=self.env, text=True, capture_output=True, timeout=30)
        if error:
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn(error, result.stderr)
            return None
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def receipt(self):
        return next(self.cache.glob("*.receipt"))

    def assert_rehashed(self, source=None, mode="registered"):
        path = source or self.source
        record = self.run_native(mode, path)
        self.assertEqual(record["verification_bytes"], path.stat().st_size)
        self.assertEqual(record["verification_cache_hits"], 0)
        self.assertEqual(record["registration_cache_hits"], 0)
        self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        return record

    def assert_reused(self, mode="registered", source=None):
        record = self.run_native(mode, source)
        self.assertEqual(record["verification_bytes"], 0)
        self.assertEqual(record["verification_cache_hits"], 1)
        self.assertEqual(record["registration_cache_hits"], 1)
        self.assertEqual(record["registration_records_written"], 0)
        return record

    def test_first_registration_and_fresh_process_reuse(self):
        first = self.assert_rehashed()
        self.assertEqual(first["registration_records_written"], 1)
        initial = self.receipt().stat()
        second = self.assert_reused()
        self.assertEqual(first["artifact_digest"], second["artifact_digest"])
        self.assertEqual(initial.st_mtime_ns, self.receipt().stat().st_mtime_ns)
        self.assertEqual(initial.st_ino, self.receipt().stat().st_ino)
        self.assertGreater(first["verification_seconds"], 0)

    def test_independent_hmac_and_private_modes(self):
        self.assert_rehashed()
        record = self.receipt().read_bytes()
        body, mac = record.rsplit(b"\n", 2)[:2]
        self.assertEqual(hmac.new((self.cache / "key-v1").read_bytes(), body + b"\n", hashlib.sha256).hexdigest(), mac.decode())
        self.assertEqual(self.cache.stat().st_mode & 0o777, 0o700)
        for path in (self.receipt(), self.cache / "key-v1", self.cache / ".lock-v1"):
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_same_size_mutation_with_restored_mtime(self):
        first = self.assert_rehashed()
        original = self.source.stat()
        self.source.write_bytes(b"different source byte")
        self.assertEqual(self.source.stat().st_size, original.st_size)
        os.utime(self.source, ns=(original.st_atime_ns, original.st_mtime_ns))
        changed = self.assert_rehashed()
        self.assertNotEqual(first["sha256"], changed["sha256"])
        self.assert_reused()

    def test_replaced_file_same_size_same_mtime(self):
        self.assert_rehashed()
        original = self.source.stat()
        other = self.root / "replacement.bin"
        other.write_bytes(self.source.read_bytes())
        os.utime(other, ns=(original.st_atime_ns, original.st_mtime_ns))
        other.replace(self.source)
        self.assertNotEqual(self.source.stat().st_ino, original.st_ino)
        self.assert_rehashed()

    def test_symlink_retarget(self):
        alias = self.root / "alias.bin"
        alias.symlink_to(self.source)
        self.assert_rehashed(alias)
        self.assert_reused(source=alias)
        other = self.root / "other.bin"
        other.write_bytes(self.source.read_bytes())
        alias.unlink()
        alias.symlink_to(other)
        self.assert_rehashed(alias)

    def test_size_change(self):
        self.assert_rehashed()
        self.source.write_bytes(b"larger source" * 20)
        self.assert_rehashed()

    def test_missing_receipt(self):
        self.assert_rehashed()
        self.receipt().unlink()
        self.assert_rehashed()

    def test_corrupt_receipt(self):
        self.assert_rehashed()
        self.receipt().write_bytes(b"truncated receipt")
        self.assert_rehashed()
        self.assert_reused()

    def test_forged_digest_without_valid_hmac(self):
        self.assert_rehashed()
        record = self.receipt().read_bytes().replace(self.digest.encode(), b"0" * 64)
        self.receipt().write_bytes(record)
        self.assert_rehashed()

    def test_receipt_bound_to_path_and_generation(self):
        self.assert_rehashed()
        forged = self.receipt().read_bytes()
        other = self.root / "other.bin"
        other.write_bytes(self.source.read_bytes())
        self.assert_rehashed(other)
        target = self.cache / (hashlib.sha256(str(other).encode()).hexdigest() + ".receipt")
        target.write_bytes(forged)
        self.assert_rehashed(other)

    def test_public_metadata_and_process_only_capture_not_upgraded(self):
        self.assert_rehashed()
        self.run_native("metadata", expected=self.digest, error="not verified")
        self.run_native("preverified", error="artifact_verification_required")
        self.assert_rehashed(mode="verified")

    def test_expected_digest_still_checked_on_reuse(self):
        self.assert_rehashed()
        self.run_native(expected="0" * 64, error="digest mismatch")
        self.assert_reused()

    def test_mismatched_initial_expected_digest_not_published(self):
        self.run_native(expected="0" * 64, error="digest mismatch")
        self.assertEqual(list(self.cache.glob("*.receipt")), [])
        self.assert_rehashed()

    def test_multi_file_failure_does_not_publish_partial_set(self):
        other = self.root / "second.bin"
        other.write_bytes(b"second source")
        self.run_native(second=other, error="digest mismatch")
        self.assertEqual(list(self.cache.glob("*.receipt")), [])

    def test_cancel_before_capture_no_registration(self):
        self.run_native("cancel", error="cancelled")
        self.assertFalse(self.cache.exists())

    def test_pending_initial_and_persistent_reuse(self):
        self.assert_rehashed(mode="pending")
        self.assert_reused(mode="pending")
        self.run_native("pending-stop", error="cancelled")

    def test_held_fd_guard_after_persistent_hit(self):
        self.assert_rehashed()
        self.run_native("held-mutate", error="changed")

    def test_untrusted_cache_permissions_fail_safe(self):
        self.assert_rehashed()
        self.cache.chmod(0o777)
        self.addCleanup(self.cache.chmod, 0o700)
        record = self.assert_rehashed()
        self.assertEqual(record["registration_records_written"], 0)

    def test_untrusted_receipt_permissions_fail_safe(self):
        self.assert_rehashed()
        self.receipt().chmod(0o644)
        self.assert_rehashed()

    def test_symlink_receipt_not_followed(self):
        self.assert_rehashed()
        record = self.receipt()
        target = self.root / "external-receipt"
        record.replace(target)
        record.symlink_to(target)
        original = target.read_bytes()
        self.assert_rehashed()
        self.assertEqual(target.read_bytes(), original)
        self.assertFalse(record.is_symlink())

    def test_invalid_key_never_grants_reuse(self):
        self.assert_rehashed()
        (self.cache / "key-v1").write_bytes(b"invalid")
        record = self.assert_rehashed()
        self.assertEqual(record["registration_records_written"], 0)

    def test_missing_key_reverifies(self):
        self.assert_rehashed()
        (self.cache / "key-v1").unlink()
        self.assert_rehashed()
        self.assert_reused()

    def test_symlink_cache_directory_not_used(self):
        external = self.root / "external"
        external.mkdir(mode=0o700)
        self.cache.symlink_to(external, target_is_directory=True)
        record = self.assert_rehashed()
        self.assertEqual(record["registration_records_written"], 0)
        self.assertEqual(list(external.iterdir()), [])

    def test_concurrent_initial_processes_keep_complete_receipts(self):
        self.source.write_bytes(b"s" * (2 * 1024 * 1024 + 17))
        for round_index in range(4):
            self.cache = self.root / f"concurrent-cache-{round_index}"
            self.env["TURBOCIDER_SOURCE_REGISTRATION_CACHE"] = str(self.cache)
            workers = [subprocess.Popen([str(self.binary), "registered", str(self.source)], env=self.env,
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(4)]
            rows = []
            results = [worker.communicate(timeout=30) for worker in workers]
            for worker, (stdout, stderr) in zip(workers, results):
                self.assertEqual(worker.returncode, 0, stderr)
                rows.append(json.loads(stdout))
            self.assertEqual(len({row["sha256"] for row in rows}), 1)
            self.assertGreater(sum(row["verification_bytes"] for row in rows), 0)
            self.assert_reused()
            self.assertEqual(list(self.cache.glob(".pending-*")), [])


if __name__ == "__main__":
    unittest.main()

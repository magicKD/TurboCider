from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen21_gguf_registered_buffer_screen import validate_record


class RegisteredBufferScreenTests(unittest.TestCase):
    def test_prefetched_capacity_and_actual_source_queue(self):
        data=dict(schema="tc-qwen21-encoder-import-screen-v1",source_sha256="a"*64,embedding_mode="packed",
            qualification_passed=False,verification_overlap=False,source_verified_before_publication=True,
            workers=8,verification_bytes=0,registration_cache_hits=1,read_buffer_capacity_bytes=16<<20,
            source_read_bytes=100,packed_capacity_bytes=100,managed_peak_bytes=100+(16<<20),
            final_weight_logical_bytes=100,mlx_logical_peak_bytes=100,native_verification_seconds=.001,
            bank_load_seconds=1,bank_read_seconds=.6,bank_decode_seconds=.4,embedding_setup_seconds=0,
            token_gather_seconds=.001,total_observed_seconds=1.1,read_buffer_slots=2,read_ahead_enabled=True,
            read_ahead_drained=True,read_ahead_jobs=4,read_ahead_bytes=90,read_wait_seconds=.1)
        validate_record(data,8,"a"*64,True)
        for key,value in (("read_buffer_capacity_bytes",8<<20),("read_buffer_slots",1),
            ("read_ahead_drained",False),("read_ahead_jobs",0),("read_ahead_bytes",101),
            ("read_wait_seconds",float("nan"))):
            bad=dict(data);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_record(bad,8,"a"*64,True)

    def test_bounded_buffers_native_proof_and_actual_read_counters(self):
        for mib in (1, 4, 8):
            record = dict(schema="tc-qwen21-encoder-import-screen-v1", source_sha256="a" * 64,
                embedding_mode="packed", qualification_passed=False, verification_overlap=False,
                source_verified_before_publication=True, workers=8, verification_bytes=0, registration_cache_hits=1,
                read_buffer_capacity_bytes=mib << 20, source_read_bytes=100, packed_capacity_bytes=100,
                managed_peak_bytes=100 + (mib << 20), final_weight_logical_bytes=100, mlx_logical_peak_bytes=100,
                native_verification_seconds=.001, bank_load_seconds=1, bank_read_seconds=.6, bank_decode_seconds=.4,
                embedding_setup_seconds=0, token_gather_seconds=.001, total_observed_seconds=1.1)
            validate_record(record, mib, "a" * 64)
            for field, value in (("registration_cache_hits", True), ("verification_bytes", 1),
                ("read_buffer_capacity_bytes", 0), ("managed_peak_bytes", 101 + (mib << 20)),
                ("bank_read_seconds", float("nan")), ("source_read_bytes", 0),
                ("source_sha256", "b" * 64), ("verification_overlap", True)):
                broken = dict(record); broken[field] = value
                with self.subTest(mib=mib, field=field), self.assertRaises(ValueError):
                    validate_record(broken, mib, "a" * 64)


if __name__ == "__main__":
    unittest.main()

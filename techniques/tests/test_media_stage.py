"""Independent PNG fixtures and full recorded public pixel-stage regressions."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
import media_stage as stage


def chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def fixture_png(width, height, values, channels=3, method=0, compressed=None, interlace=0):
    pitch = width * channels
    filtered = bytearray()
    previous = bytes(pitch)
    for y in range(height):
        row = values[y * pitch:(y + 1) * pitch]
        filtered.append(method)
        for i, value in enumerate(row):
            left = row[i - channels] if i >= channels else 0
            above = previous[i]
            corner = previous[i - channels] if i >= channels else 0
            predictors = [0, left, above, (left + above) // 2]
            p = left + above - corner
            pa, pb, pc = abs(p - left), abs(p - above), abs(p - corner)
            predictors.append(left if pa <= pb and pa <= pc else above if pb <= pc else corner)
            filtered.append((value - predictors[method]) % 256)
        previous = row
    header = struct.pack(">IIBBBBB", width, height, 8, 2 if channels == 3 else 6, 0, 0, interlace)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(filtered) if compressed is None else compressed) + chunk(b"IEND", b"")


class PNGContracts(unittest.TestCase):
    def decode(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.png"
            path.write_bytes(data)
            return stage.png_rgb(path)

    def test_all_five_png_filters_match_complete_known_rgb_bytes(self):
        rgb = bytes((i * 31 + i // 7) % 256 for i in range(7 * 5 * 3))
        for method in range(5):
            data = fixture_png(7, 5, rgb, method=method)
            metadata, actual = self.decode(data)
            self.assertEqual(actual, rgb)
            self.assertEqual(metadata["rgb_sha256"], hashlib.sha256(rgb).hexdigest())
            self.assertEqual(metadata["file_sha256"], hashlib.sha256(data).hexdigest())

    def test_opaque_rgba_drops_only_alpha_without_compositing(self):
        rgba = bytes([3, 7, 11, 255, 253, 2, 89, 255])
        metadata, rgb = self.decode(fixture_png(2, 1, rgba, channels=4, method=4))
        self.assertEqual(rgb, bytes([3, 7, 11, 253, 2, 89]))
        self.assertEqual((metadata["width"], metadata["height"]), (2, 1))

    def test_partial_alpha_is_refused_with_exact_summary_not_silently_ignored(self):
        with self.assertRaises(stage.SourceRefused) as error:
            self.decode(fixture_png(2, 1, bytes([1, 2, 3, 255, 4, 5, 6, 253]), channels=4))
        self.assertEqual(error.exception.summary["opaque_pixels"], 1)
        self.assertEqual(error.exception.summary["alpha_min"], 253)
        self.assertEqual(error.exception.summary["total_pixels"], 2)

    def test_crc_truncation_trailing_png_and_unknown_critical_chunk_are_rejected(self):
        data = fixture_png(2, 1, bytes(range(6)))
        corrupt = bytearray(data)
        corrupt[20] ^= 1
        for value in (bytes(corrupt), data[:-1], data + b"trailing", data[:33] + chunk(b"ABCD", b"unknown") + data[33:]):
            with self.subTest(size=len(value)), self.assertRaises(ValueError):
                self.decode(value)

    def test_zlib_trailing_concatenation_and_decompression_cap_are_rejected(self):
        filtered = b"\0" + bytes(range(6))
        for payload in (zlib.compress(filtered) + b"trailing", zlib.compress(filtered) + zlib.compress(b"second"),
                        zlib.compress(b"\0" * 100000)):
            with self.assertRaises(ValueError):
                self.decode(fixture_png(2, 1, bytes(range(6)), compressed=payload))

    def test_interlace_animation_and_bad_filter_are_explicitly_unsupported(self):
        ordinary = fixture_png(2, 1, bytes(range(6)))
        for value in (fixture_png(2, 1, bytes(range(6)), interlace=1),
                      ordinary[:33] + chunk(b"acTL", b"animated") + ordinary[33:],
                      fixture_png(2, 1, bytes(range(6)), compressed=zlib.compress(b"\x05" + bytes(range(6))))):
            with self.assertRaises(ValueError):
                self.decode(value)


class AddressContracts(unittest.TestCase):
    def test_forward_scatter_equals_an_independent_inverse_gather_for_all_directions(self):
        width, height = 5, 3
        rgb = bytes(range(width * height * 3))
        for code in range(8):
            ow, oh, actual = stage.oriented_rgb(rgb, width, height, code, 1)
            expected = bytearray(len(rgb))
            for y in range(oh):
                for x in range(ow):
                    tx, ty = ow - 1 - x if code & 1 else x, oh - 1 - y if code & 2 else y
                    sx, sy = (ty, tx) if code & 4 else (tx, ty)
                    for c in range(3):
                        expected[(y * ow + x) * 3 + c] = rgb[(sy * width + sx) * 3 + c] ^ 1
            self.assertEqual(actual, expected)

    def test_source_padding_alpha_and_generations_do_not_change_rgb_channel_contract(self):
        rgb = stage.synthetic_rgb(7, 5, 29)
        for generation in (0, 1):
            pitch, bgra = stage.source_bgra(rgb, 7, 5, generation)
            self.assertGreater(pitch, 7 * 4)
            for y in range(5):
                self.assertEqual(bgra[y * pitch + 28:(y + 1) * pitch], b"\xa5" * (pitch - 28))
                for x in range(7):
                    p = bgra[y * pitch + x * 4:y * pitch + x * 4 + 4]
                    self.assertEqual(bytes(v ^ generation for v in (p[2], p[1], p[0])), rgb[(y * 7 + x) * 3:(y * 7 + x + 1) * 3])
                    self.assertEqual(p[3], (x * 11 + y * 3) % 256)

    def test_malformed_geometry_seed_generation_and_repeat_types_are_rejected(self):
        for args in ((0, 3, 17), (5, True, 17), (2049, 3, 17), (5, 3, False)):
            with self.assertRaises(ValueError):
                stage.synthetic_rgb(*args)
        for width, height, generation in ((True, 1, 0), (1, 1, True), (0, 1, 0)):
            with self.assertRaises(ValueError):
                stage.source_bgra(b"\0\0\0", width, height, generation)
        for value in (True, 0, 11, 100):
            with self.assertRaises(ValueError):
                stage.order(value)

    def test_default_case_and_help_do_not_execute_a_device_or_import_optional_packages(self):
        self.assertFalse(stage.stage_contract()["details"]["actual_device_executed"])
        child = subprocess.run([sys.executable, "-E", "-S", "-B", str(ROOT / "integrations/media_stage_probe.py"), "--help"],
                               capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr)


class RecordedStageEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "review/media-stage-evidence-20261016.json").read_text())

    def native(self):
        return copy.deepcopy(self.data["runs"][-1]["report"]["native"])

    def test_all_calls_controls_and_first_build_failure_remain_in_the_record(self):
        calls = gpu_negatives = cached = 0
        self.assertEqual(len(self.data["runs"]), 2)
        for row in self.data["runs"]:
            proof = stage.verify(row["report"]["native"])
            self.assertEqual(proof, row["independent_validation"])
            self.assertTrue(row["owned_wrapper_exit_confirmed"])
            self.assertTrue(row["full_source_bytes_independently_verified"])
            calls += proof["completed_stage_calls"]
            gpu_negatives += proof["actual_gpu_negative_calls"]
            cached += proof["cached_negative_observations"]
        self.assertEqual((calls, gpu_negatives, cached), (2208, 64, 32))
        self.assertEqual(len(self.data["build_failures"]), 1)
        failed = self.data["build_failures"][0]
        self.assertIsNone(failed["report"]["native"])
        self.assertFalse(failed["independent_validation"]["device_work_started"])
        for key, relative in (("contract_sha256", "labs/media_stage.py"), ("curator_sha256", "scripts/curate_media_stage.py")):
            self.assertEqual(self.data[key], hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())

    def test_source_opacity_refusal_is_not_a_quality_pass_or_device_failure(self):
        for row in self.data["runs"] + self.data["build_failures"]:
            value = row["report"]["source_admission_rejections"][0]
            self.assertEqual((value["alpha_min"], value["alpha_max"], value["opaque_pixels"], value["total_pixels"]), (253, 255, 239847, 262144))
            self.assertFalse(value["native_work_started_for_this_source"])
        self.assertFalse(self.data["gpu_fp32_outputs_reexecuted_by_verifier"])
        self.assertFalse(self.data["raw_pixels_or_paths_exported"])

    def test_fixed_quality_full_shape_and_all_generations_are_required(self):
        for kind in ("budget", "nonfinite", "roundtrip", "elements", "dtype", "generation", "orientation"):
            value = self.native()
            sample = value["scenarios"][0]["samples"][0]
            if kind == "budget":
                value["gate"]["max_abs"] = .01
            elif kind == "nonfinite":
                sample["metrics"]["finite"] = False
            elif kind == "roundtrip":
                sample["metrics"]["recovered_rgb_mismatches"] = 1
            elif kind == "elements":
                sample["output_elements"] -= 1
            elif kind == "dtype":
                sample["output_dtype"] = "float16"
            elif kind == "generation":
                sample["generation"] = True
            else:
                value["scenarios"][0]["orientation"] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                stage.verify(value)

    def test_omitted_reordered_failed_or_physically_promoted_protocols_cannot_score(self):
        for kind in ("call", "order", "scenario", "failure", "request", "physical"):
            value = self.native()
            if kind == "call":
                value["scenarios"][0]["samples"].pop()
            elif kind == "order":
                value["scenarios"][0]["samples"][:3] = value["scenarios"][0]["samples"][:3][::-1]
            elif kind == "scenario":
                value["scenarios"].pop()
            elif kind == "failure":
                value["status"] = "failed"
            elif kind == "request":
                value["whole_request_qualified"] = True
            else:
                value["physical_overlap_qualified"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                stage.verify(value)

    def test_real_api_copy_bytes_source_hashes_guards_and_completion_are_required(self):
        for kind in ("calls", "bytes", "hash", "guard", "source", "drain", "owner"):
            value = self.native()
            sample = next(s for s in value["scenarios"][0]["samples"] if s["arm"] == "vimage")
            if kind == "calls":
                sample["vimage_api_calls"] = 0
            elif kind == "bytes":
                sample["logical_import_bytes"] += 1
            elif kind == "hash":
                sample["rgb_sha256"] = "0" * 64
            elif kind == "guard":
                sample["host_guards_pass"] = False
            elif kind == "source":
                sample["source_unchanged"] = False
            elif kind == "drain":
                sample["mlx_stream_drained"] = False
            else:
                value["all_owned_gpu_work_completed"] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                stage.verify(value)

    def test_complete_host_intervals_and_actual_negative_evaluations_stay_distinct(self):
        for kind in ("clock", "duration", "import", "stale", "channels"):
            value = self.native()
            scenario = value["scenarios"][0]
            event = scenario["samples"][0]["events"][0]
            if kind == "clock":
                event["clock_domain"] = "GPU-clock"
            elif kind == "duration":
                event["duration_ns"] = True
            elif kind == "import":
                scenario["samples"][0]["events"][1]["start_ns"] = 0
            elif kind == "stale":
                scenario["stale_actual_output_control"]["new_gpu_dispatch"] = True
            else:
                scenario["wrong_channels_control"]["rejected"] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                stage.verify(value)

    def test_fixed_gpu_denominator_and_counter_direction_are_not_hidden_by_oracle_min(self):
        proof = stage.verify(self.native())
        for comparison in proof["component_comparisons"]:
            self.assertTrue(comparison["best_cpu_is_per_trial_oracle_not_a_deployable_policy"])
            self.assertGreater(comparison["paired_vimage_over_gpu"], 1)
            self.assertFalse(comparison["production_qualified"])
        counter = next(c for c in proof["component_comparisons"] if c["id"] == "synthetic-o3")
        self.assertLess(counter["paired_loop_over_vimage"], 1)
        self.assertGreater(counter["median_wall_ns"]["product-loop"] / counter["median_wall_ns"]["vimage"], 1)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Synthetic contract fixtures; never product qualification evidence."""
import copy
import json
import hashlib
import struct
import zlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/native'))
import verify_streaming_text_capacity as verifier
import build_streaming_catalog as builder
from test_streaming_catalog_builder import record as base_record


def fixture():
    record = base_record()
    record['source'] = dict(identity_version=2, model_variant='z-image-turbo-comfy-bf16',
        weight_format='comfy-bf16-single-file', artifact_manifest_digest='a' * 64)
    record['runtime']['adapter_revision'] = 'z-image-public-adapter-v6-text-capacity'
    record['runtime']['kernel_revision'] = 'kernel-test'
    record['workload']['execution_container'] = 'cli_worker'
    record['calibration']['execution_container'] = 'cli_worker'
    record['workload']['token_shapes'] = [dict(encoder='qwen3', tokenizer_revision='qwen3-z-image-v1',
        template_revision='z-image-template-v1', valid_rows=1024, padded_rows=1024, compute_rows=1024)]
    record['text_capacity'] = dict(policy_revision='z-image-dynamic-text-capacity-v1', minimum_rows=7, maximum_rows=1024)
    record['canonical_record_digest'] = builder.canonical_record_digest(record)
    actual = copy.deepcopy(record['workload'])
    for key in ('valid_rows', 'padded_rows', 'compute_rows'):
        actual['token_shapes'][0][key] = 32
    workload = verifier.workload_digest(actual)
    request = dict(schema_version=2, model='z-image-turbo', operation='image.generate',
        inputs=[dict(kind='text', role='prompt', text='fixture prompt')],
        outputs=[dict(kind='image', path='/not-a-real-output.png', width=512, height=512)],
        sampling=dict(seed=42, steps=9), parameters=dict(dynamic_text=True),
        execution=dict(policy='gpu', streaming=dict(schema_version=2, enabled=True, selection='memory_tier',
            retention='request', target_request_memory_bytes=16 << 30)))
    public_plan = dict(request=request, library_sha256='b' * 64,
        config=dict(constructor='public', execution_container='cli_worker', verify_streaming_sources=True))
    manual_plan = copy.deepcopy(public_plan)
    manual_plan['request']['execution']['streaming'] = copy.deepcopy(record['plan']['canonical_config'])
    manual_plan['config']['constructor'] = 'candidate'
    layout = 'c' * 64
    selected = dict(actual_layout_digest=layout, authorized_layout_digest=layout, actual_plan_verified=True,
        record_digest=record['canonical_record_digest'], workload_digest=workload,
        resolution_digest='d' * 64, device_digest='e' * 64, execution_container='cli_worker',
        memory_scope='execution_process_tree_v1', target_request_memory_bytes=16 << 30,
        component_policy_revision=record['plan']['component_policy_revision'],
        receipt_digest='f' * 64, receipt_schema_version=2, receipt_source_generation=1,
        **{k+'_digest': verifier.identity_digest(record[k], k) for k in ('source', 'runtime')})
    stage = dict(digest=layout, drained=True, source_lease_verified=True, pass_count=9,
        kernel_revision='kernel-test', weight_format='comfy-bf16-single-file',
        receipt=dict(canonical_digest='f' * 64, schema_version=2, source_generation=1,
                     reader_fences_issued=144, reader_fences_completed=144),
        **record['plan']['canonical_config']['stages']['denoiser'])
    result = dict(model='z-image-turbo', operation='image.generate', width=512, height=512, steps=9,
        seed=42, valid_text_tokens=32, text_tokens=32, actual_denoise_steps=9)
    public = dict(status=0, error=None, result=dict(**result, public_streaming=selected,
                  streaming_stages=[dict(stage_index=0, runtime=stage)]))
    private_stage = copy.deepcopy(stage)
    private_stage.update(digest='9' * 64, source_lease_verified=False)
    manual = dict(status=0, error=None, result=dict(**result, block_streaming=dict(actual_layout=private_stage)))
    process = dict(pid=123, start_seconds=456, start_microseconds=789)
    documents = dict(public_plan=public_plan, manual_plan=manual_plan,
        public_result=public, manual_result=manual,
        public_source=dict(report=dict(status='verified', artifact_manifest_digest='a' * 64)),
        manual_source=dict(report=dict(status='verified', artifact_manifest_digest='a' * 64)),
        public_resolution=dict(status=0, error=None, result=dict(request_digest=workload,
            resolution_digest='d' * 64, exact_selector=dict(expected_resolution_digest='d' * 64),
            selection=dict(layout_digest=layout, record_digest=record['canonical_record_digest']),
            identity={k: selected[k] for k in ('source_digest', 'runtime_digest', 'device_digest')})),
        public_observation=dict(status='succeeded', cleanup_returned=True, cancellation_sent=False, process_identity=process),
        memory_summary=dict(complete=True, command_exit_code=0, tree_peak_phys_footprint_bytes=9 << 30,
            max_gap_ns=20_000_000, allowed_max_gap_ns=100_000_000, swap_in_bytes=4096, swap_out_bytes=0,
            root_identity=copy.deepcopy(process)))
    return record, documents


class CapacityEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.record, self.documents = fixture()

    def validate(self):
        return verifier.validate_case(self.record, 32, self.documents,
            library_sha256='b' * 64, target=16 << 30, memory_limit=15 << 30)

    def test_boundaries_include_both_sides_and_minimum(self):
        self.assertEqual(verifier.required_rows(self.record['text_capacity']),
            [7, 31, 32, 33, 511, 512, 513, 1023, 1024])
        for invalid in (0, True, 1025):
            with self.assertRaises(ValueError):
                verifier.required_rows(dict(minimum_rows=invalid, maximum_rows=1024))

    def test_actual_shape_and_private_reference_policy_are_explicit(self):
        checked = self.validate()
        self.assertEqual(checked['rows'], 32)
        self.assertEqual(checked['swap_in_bytes'], 4096)  # Not a zero-swap claim.
        self.assertNotEqual(checked['workload_digest'], verifier.workload_digest(self.record['workload']))

    def test_false_success_partial_sampling_and_different_process_are_rejected(self):
        mutations = [
            (('public_observation', 'cleanup_returned'), False),
            (('public_observation', 'process_identity', 'pid'), 124),
            (('memory_summary', 'complete'), False),
            (('memory_summary', 'tree_peak_phys_footprint_bytes'), 17 << 30),
            (('memory_summary', 'max_gap_ns'), 101_000_000),
            (('memory_summary', 'swap_out_bytes'), 4096),
            (('public_result', 'result', 'valid_text_tokens'), 31),
            (('public_result', 'result', 'actual_denoise_steps'), 8),
            (('public_result', 'result', 'public_streaming', 'actual_plan_verified'), False),
            (('public_result', 'result', 'public_streaming', 'authorized_layout_digest'), '9' * 64),
            (('public_result', 'result', 'public_streaming', 'workload_digest'), verifier.workload_digest(self.record['workload'])),
            (('public_result', 'result', 'public_streaming', 'receipt_source_generation'), 2),
            (('public_source', 'report', 'artifact_manifest_digest'), '9' * 64),
            (('manual_plan', 'request', 'sampling', 'seed'), 43),
            (('manual_plan', 'request', 'parameters', 'compile_gpu'), True),
            (('public_plan', 'library_sha256'), '9' * 64),
        ]
        for path, value in mutations:
            with self.subTest(path=path):
                self.record, self.documents = fixture()
                target = self.documents
                for key in path[:-1]: target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError): self.validate()

    def test_missing_inventory_cannot_be_replaced_with_pass_label(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / 'range.json').write_text(json.dumps(dict(passed=True)))
            with self.assertRaises(ValueError): verifier.verify(root, {}, 'a' * 40)

    def test_boundary_peak_and_gap_are_included_in_calibration(self):
        memory = dict(peak_p95_bytes=dict(candidate=8 << 30), maximum_sample_gap_ns=20_000_000)
        self.record['calibration']['estimator_revision'] = 'tree-phys-footprint-p95-with-text-boundaries-v1'
        boundary = dict(target_bytes=16 << 30, maximum_peak_bytes=9 << 30,
            maximum_sample_gap_ns=30_000_000, range_file=dict(sha256='a' * 64))
        peak, gap, digest = builder.expected_memory_calibration(self.record, memory, 16 << 30, 'b' * 64, boundary)
        self.assertEqual((peak, gap), (9 << 30, 30_000_000))
        self.assertNotEqual(digest, 'b' * 64)
        changed = copy.deepcopy(boundary); changed['range_file']['sha256'] = 'c' * 64
        self.assertNotEqual(digest, builder.expected_memory_calibration(self.record, memory, 16 << 30, 'b' * 64, changed)[2])
        for invalid in (None, dict(boundary, target_bytes=12 << 30)):
            with self.assertRaises(builder.CatalogBuildError):
                builder.expected_memory_calibration(self.record, memory, 16 << 30, 'b' * 64, invalid)
        self.record.pop('text_capacity')
        self.assertEqual(builder.expected_memory_calibration(self.record, memory, 16 << 30, 'b' * 64),
                         (8 << 30, 20_000_000, 'b' * 64))

    def test_boundary_calibration_survives_record_validation_and_catalog_rendering(self):
        from generate_bundled_streaming_catalog import render_catalog
        record = self.record
        record['release'].update(channel='public-calibrated', policy_revision=verifier.CALIBRATED)
        record['calibration']['estimator_revision'] = builder.TEXT_CAPACITY_CALIBRATION_ESTIMATOR
        memory = dict(peak_p95_bytes=dict(candidate=8 << 30), maximum_sample_gap_ns=20_000_000)
        boundary = dict(target_bytes=16 << 30, maximum_peak_bytes=9 << 30,
            maximum_sample_gap_ns=30_000_000, range_file=dict(sha256='a' * 64))
        peak, gap, digest = builder.expected_memory_calibration(record, memory, 16 << 30, 'b' * 64, boundary)
        record['calibration'].update(calibrated_request_bytes=peak, maximum_sample_gap_ns=gap,
                                     evidence_digest=digest)
        builder.validate_record_shape(record)
        record['canonical_record_digest'] = builder.canonical_record_digest(record)
        header = render_catalog(record['catalog_revision'], [record], record['runtime']['turbocider_build_id'])
        self.assertIn(builder.TEXT_CAPACITY_CALIBRATION_ESTIMATOR, header)
        self.assertIn(str(9 << 30), header)
        exact = copy.deepcopy(record)
        exact.pop('text_capacity')
        with self.assertRaisesRegex(builder.CatalogBuildError, 'estimator_revision'):
            builder.validate_record_shape(exact)
        # Shape compatibility must not let a release omit its boundary proof.
        for invalid in (None, dict(boundary, target_bytes=12 << 30)):
            with self.assertRaises(builder.CatalogBuildError):
                builder.expected_memory_calibration(record, memory, 16 << 30, 'b' * 64, invalid)
        record['calibration']['estimator_revision'] = builder.CALIBRATION_ESTIMATOR
        with self.assertRaisesRegex(builder.CatalogBuildError, 'boundary peaks'):
            builder.expected_memory_calibration(record, memory, 16 << 30, 'b' * 64, boundary)

    def test_full_inventory_walk_and_missing_boundary_or_mutated_evidence(self):
        # Only the low-level OS sampler is mocked here; its own tests validate
        # real hash-chained logs. All range inventory/identity checks run.
        from generate_runtime_build_identity import seal
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            def save(name, value):
                data = json.dumps(value).encode()
                (root / name).write_bytes(data)
                return dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            build = seal(dict(policy=dict(test_hooks='1', audit_counters='1', common_flags=[]), sources=[
                dict(path='tools/native/run_public_streaming_smoke.py', sha256='1' * 64),
                dict(path='tools/native/run_image_streaming_smoke.py', sha256='2' * 64)]))
            r, d = self.record, self.documents
            r['runtime']['turbocider_build_id'] = build['catalog_runtime_id']
            r['text_capacity'].update(minimum_rows=32, maximum_rows=32)
            for key in ('valid_rows', 'padded_rows', 'compute_rows'): r['workload']['token_shapes'][0][key] = 32
            r['plan']['layout_digest'] = 'c' * 64
            r['release'].update(channel='public-calibrated', policy_revision=verifier.CALIBRATED)
            r['canonical_record_digest'] = builder.canonical_record_digest(r)
            selected = d['public_result']['result']['public_streaming']
            selected['record_digest'] = r['canonical_record_digest']
            selected['runtime_digest'] = verifier.identity_digest(r['runtime'], 'runtime')
            resolved = d['public_resolution']['result']
            resolved['selection']['record_digest'] = r['canonical_record_digest']
            resolved['identity']['runtime_digest'] = selected['runtime_digest']
            catalog = save('catalog.json', dict(records=[r]))
            d['public_plan'].update(catalog_sha256=catalog['sha256'], script_sha256='1' * 64)
            d['manual_plan']['script_sha256'] = '2' * 64
            def chunk(kind, payload):
                return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
            png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 512, 512, 8, 2, 0, 0, 0))
            png += chunk(b'IDAT', zlib.compress((b'\0' * (1 + 512 * 3)) * 512)) + chunk(b'IEND', b'')
            (root / 'image.png').write_bytes(png)
            image = dict(path='image.png', bytes=len(png), sha256=hashlib.sha256(png).hexdigest())
            d['public_observation']['image_sha256'] = image['sha256']
            d['memory_summary'].update(correlation_id='fixture-32', evidence_digest='9' * 64)
            parsed_memory = dict(d['memory_summary'], final_evidence_digest='9' * 64)
            case = dict(rows=32, **{key: save(key + '.json', value) for key, value in d.items()})
            case.update(public_image=image, manual_image=image, memory=save('memory.jsonl', dict(fixture=True)))
            binding = builder.catalog_binding(r)
            evidence = dict(schema=verifier.SCHEMA, binding=binding, reviewed_commit='a' * 40,
                source_identity=save('source.json', dict(commit='a' * 40, source_manifest_sha256='b' * 64, clean=True)),
                build_manifest=save('build.json', build), library_sha256='b' * 64, catalog=catalog,
                target_bytes=16 << 30, cases=[case])
            save('range.json', evidence)
            with patch.object(verifier, 'verify_memory', return_value=parsed_memory):
                result = verifier.verify(root, binding, 'a' * 40)
                self.assertTrue(result['passed'])
                self.assertEqual(result['maximum_peak_bytes'], 9 << 30)
                self.assertFalse(result['production_authorized'])
                evidence['cases'] = []
                save('range.json', evidence)
                with self.assertRaisesRegex(ValueError, 'boundary cases'): verifier.verify(root, binding, 'a' * 40)
                evidence['cases'] = [case, case]
                save('range.json', evidence)
                with self.assertRaisesRegex(ValueError, 'duplicate'): verifier.verify(root, binding, 'a' * 40)
                evidence['cases'] = [case]
                save('range.json', evidence)
                (root / 'public_result.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'digest or size'): verifier.verify(root, binding, 'a' * 40)


if __name__ == '__main__':
    unittest.main(verbosity=2)

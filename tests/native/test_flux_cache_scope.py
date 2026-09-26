#!/usr/bin/env python3
"""Verify real C API failure paths restore Flux's process-wide MLX cache limit."""
import os
import json
from pathlib import Path
import tempfile

os.environ['TURBOCIDER_TEST_FLUX_MODEL'] = 'flux2-klein-4b'
import mlx.core as mx
from test_flux_candidate_streaming_gate import LIB, create, fixture, generate, prepare, request
from test_flux_public_streaming import byte_vocab


def main():
    sentinel = 19 << 20
    original = mx.set_cache_limit(sentinel)
    try:
        with tempfile.TemporaryDirectory(prefix='tc-flux-cache-') as raw:
            root = Path(raw)
            fixture(root)
            tokenizer = root / "tokenizer/tokenizer.json"
            vocabulary = json.loads(tokenizer.read_text())
            vocabulary["model"]["vocab"] = byte_vocab()
            tokenizer.write_text(json.dumps(vocabulary))
            engine = create(LIB.tc_engine_create_model_candidate, root)
            try:
                # Exact generation reaches the absent encoder weights after
                # installing its cache limit. Preparation may be rejected earlier
                # by the conservative resident admission on a 16 GiB machine.
                value = request()
                for operation in ('generate', 'prepare'):
                    if operation == 'prepare':
                        value['execution'].pop('streaming')
                        value['execution']['residency'] = 'component_staged'
                    status, error = (generate if operation == 'generate' else prepare)(engine, value)
                    observed = mx.set_cache_limit(sentinel)
                    assert status != 0, 'Missing encoder weights unexpectedly succeeded'
                    if operation == 'generate':
                        assert 'safetensors' in error.lower(), (operation, error)
                    else:
                        assert 'safetensors' in error.lower() or 'insufficient physical memory' in error, error
                        print('Preparation failure:', error)
                    assert observed == sentinel, (operation, observed, sentinel, error)
            finally:
                LIB.tc_engine_free(engine)
    finally:
        mx.set_cache_limit(original)
    print('PASS Flux cache scope: exact generation failure restores the embedding cache limit; preparation rejection preserves it')


if __name__ == '__main__':
    main()

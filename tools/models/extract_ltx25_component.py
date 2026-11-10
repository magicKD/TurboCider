#!/usr/bin/env python3
"""Extract an isolated real FFN from pinned, already received model ranges.

This creates a component fixture, never a complete model installation. A
provenance manifest records original extents and tensor hashes so the fixture
can be rechecked against the final SHA256-verified checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


class NotReceivedYet(RuntimeError):
    pass


def received_bytes(source, begin, size):
    """Read only byte extents known present; never read sparse/unreceived data."""
    partial = source.with_name(source.name + '.part')
    ranges = source.with_name(source.name + '.ranges')
    candidates = []
    for path in (source, partial):
        if path.is_file():
            candidates.append((0, path.stat().st_size, path))
    if ranges.is_dir():
        for path in ranges.glob('*.range'):
            match = re.fullmatch(r'(\d+)-(\d+)\.range', path.name)
            if match and path.is_file():
                lo, hi = map(int, match.groups())
                candidates.append((lo, lo + min(path.stat().st_size, hi - lo + 1), path))
    result = bytearray()
    end = begin + size
    while begin < end:
        matching = [item for item in candidates if item[0] <= begin < item[1]]
        if not matching:
            raise NotReceivedYet(f'checkpoint range {begin}:{end} has not arrived yet')
        lo, hi, path = max(matching, key=lambda item: item[1])
        count = min(hi, end) - begin
        with path.open('rb') as stream:
            stream.seek(begin - lo)
            block = stream.read(count)
        if len(block) != count:
            raise NotReceivedYet('checkpoint transfer changed during component extraction')
        result.extend(block)
        begin += count
    return bytes(result)


def extract(asset, source, kind, block_index, output, metadata_network=False):
    if output.exists():
        raise ValueError('fresh component output directory required')
    length = struct.unpack('<Q', received_bytes(source, 0, 8))[0]
    if not 0 < length < min(asset['bytes'] - 8, 64 << 20):
        raise ValueError('invalid pinned checkpoint header size')
    header_bytes = received_bytes(source, 8, length)
    header = json.loads(header_bytes)
    if kind == 'ltx-ffn':
        base = f'model.diffusion_model.transformer_blocks.{block_index}.ff.'
        stems = [('fc1', base + 'net.0.proj'), ('fc2', base + 'net.2')]
    else:
        base = f'model.layers.{block_index}.mlp.'
        stems = [('fc1', base + 'gate_proj'), ('up', base + 'up_proj'), ('fc2', base + 'down_proj')]
    payloads, tensors, provenance = [], {}, []
    offset = 0
    for renamed, stem in stems:
        for suffix in ('.weight', '.weight_scale', '.bias', '.comfy_quant'):
            original = stem + suffix
            if original not in header and suffix == '.bias':
                continue
            entry = header[original]
            lo, hi = entry['data_offsets']
            start = length + 8 + lo
            count = hi - lo
            if lo < 0 or count <= 0 or start + count > asset['bytes']:
                raise ValueError('tensor extent exceeds pinned checkpoint')
            try:
                data = received_bytes(source, start, count)
            except NotReceivedYet:
                if not metadata_network or suffix != '.comfy_quant' or count > 4096:
                    raise
                end = start + count - 1
                url = f"https://modelscope.cn/models/{asset['repository']}/resolve/{asset['revision']}/{asset['filename']}"
                request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f"bytes {start}-{end}/{asset['bytes']}":
                        raise ValueError('source did not honor exact quantization-metadata extent')
                    data = response.read(count + 1)
                    if len(data) != count:
                        raise ValueError('quantization-metadata extent has wrong length')
            if suffix == '.comfy_quant':
                quant = json.loads(data)
                if not quant.get('convrot') or quant.get('convrot_groupsize') != 256 or quant.get('format') != 'int8_tensorwise':
                    raise ValueError('unexpected ConvRot source format')
            renamed_key = renamed + suffix
            tensors[renamed_key] = {**entry, 'data_offsets': [offset, offset + count]}
            provenance.append(dict(source_name=original, fixture_name=renamed_key, source_offset=start,
                                   bytes=count, sha256=hashlib.sha256(data).hexdigest()))
            payloads.append(data)
            offset += count
    tensors['__metadata__'] = dict(scope='isolated FFN component; not a complete model',
                                  source_checkpoint=asset['relative_path'], source_revision=asset['revision'])
    encoded = json.dumps(tensors, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    output.mkdir(parents=True)
    weights = output / 'weights.safetensors'
    with weights.open('xb') as stream:
        stream.write(struct.pack('<Q', len(encoded)))
        stream.write(encoded)
        for data in payloads:
            stream.write(data)
    with weights.open('rb') as stream:
        fixture_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = dict(schema='tc-ltx25-real-ffn-component-v1', kind=kind, block_index=block_index,
                    source=asset, source_header_sha256=hashlib.sha256(header_bytes).hexdigest(),
                    full_source_sha256_verified=False, fixture_sha256=fixture_sha, tensors=provenance,
                    scope='real weight slices and synthetic inputs; full checkpoint verification still required')
    (output / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'models/LTX-2.5-INT8')
    parser.add_argument('--kind', choices=('ltx-ffn', 'gemma-mlp'), required=True)
    parser.add_argument('--block', type=int, choices=range(48), default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata-network', action='store_true', help='fetch only missing small quantization tags')
    args = parser.parse_args()
    assets = json.loads(Path(__file__).with_name('ltx25_modelscope_assets.json').read_text())
    asset = assets[0 if args.kind == 'ltx-ffn' else 1]
    try:
        result = extract(asset, args.root / asset['relative_path'], args.kind, args.block,
                         args.output, args.metadata_network)
    except NotReceivedYet as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(dict(fixture=str(args.output), fixture_sha256=result['fixture_sha256'],
                          full_source_sha256_verified=False)), flush=True)


if __name__ == '__main__':
    main()

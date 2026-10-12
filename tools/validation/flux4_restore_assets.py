#!/usr/bin/env python3
"""Restore pinned FLUX4B DiT; prove exact reuse of local encoder/VAE, no inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import urllib.request

from flux9_bf16_framework_screen import generation, registered_hashes, write_json
from mflux_original_layout import read_header
from runtime_ane_common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
REPO = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
ENCODER = ROOT / "models/Comfy-Org-z_image_turbo/split_files/text_encoders/qwen_3_4b.safetensors"
VAE = ROOT / "models/FLUX.2-klein-9B/vae/diffusion_pytorch_model.safetensors"
RESERVE = 2 << 30
SHARDS = (
    ("text_encoder/model-00001-of-00002.safetensors", 4967215360, "8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea"),
    ("text_encoder/model-00002-of-00002.safetensors", 3077766632, "82f2bd839378541b0557bfabaf37c7d3d637071fdcb73302dedd7cf61162ce07"),
)


def ranged_prefix(name, count):
    if not 0 < count <= (1 << 20):
        raise ValueError("bounded remote header required")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true&header_bytes={count}"
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{count - 1}"})
    with urllib.request.urlopen(request, timeout=45) as response:
        match = re.fullmatch(r"bytes 0-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or not match or int(match[1]) != count - 1:
            raise ValueError("server must honor bounded Range; never read whole remote encoder")
        data = response.read(count + 1)
        if len(data) != count:
            raise ValueError("exact remote safetensors header extent required")
        return data, int(match[2])


def reconstruct_shard_digest(local_header, local_prefix, source_fd, remote_header, remote_prefix, total):
    digest = hashlib.sha256(remote_prefix)
    cursor = 0
    names = []
    for name, tensor in sorted(((name, tensor) for name, tensor in remote_header.items() if name != "__metadata__"),
                               key=lambda item: item[1]["data_offsets"][0]):
        start, end = tensor["data_offsets"]
        local = local_header.get(name)
        if (start != cursor or end <= start or not isinstance(local, dict) or
                local["dtype"] != tensor["dtype"] or local["shape"] != tensor["shape"] or
                local["data_offsets"][1] - local["data_offsets"][0] != end - start):
            raise ValueError("exact complete original HF tensor descriptors required: " + name)
        offset = len(local_prefix) + local["data_offsets"][0]
        remaining = end - start
        while remaining:
            block = os.pread(source_fd, min(4 << 20, remaining), offset)
            if not block:
                raise ValueError("local original encoder truncated")
            digest.update(block)
            offset += len(block)
            remaining -= len(block)
        cursor = end
        names.append(name)
    if len(remote_prefix) + cursor != total:
        raise ValueError("complete canonical remote shard bytes required")
    return digest.hexdigest(), names


def verify_encoder():
    identity = generation(ENCODER)
    header, prefix = read_header(ENCODER)
    verified, all_names = [], []
    with ENCODER.open("rb") as source:
        for name, size, expected in SHARDS:
            first, total = ranged_prefix(name, 8)
            length = struct.unpack("<Q", first)[0]
            raw, actual_size = ranged_prefix(name, 8 + length)
            if total != size or actual_size != size or raw[:8] != first:
                raise ValueError("pinned original encoder extent differs")
            actual, names = reconstruct_shard_digest(header, prefix, source.fileno(), json.loads(raw[8:]), raw, size)
            if actual != expected:
                raise ValueError("local Z encoder does not byte-exactly reconstruct original FLUX4B encoder")
            all_names.extend(names)
            verified.append(dict(file=name, bytes=size, sha256=actual, tensors=len(names),
                scope="exact original HF whole-shard SHA reconstructed from local tensor bytes; remote header only"))
    if len(all_names) != len(set(all_names)) or set(all_names) != set(header) - {"__metadata__"} or generation(ENCODER) != identity:
        raise ValueError("complete unchanged original encoder union required")
    return dict(local_source=str(ENCODER.relative_to(ROOT)), generation=identity, shards=verified,
                tensor_count=len(all_names), whole_shard_payload_identity_proven=True)


def restore(model, output):
    from huggingface_hub import HfApi, snapshot_download
    if model.exists() or output.exists() or model.parent != ROOT / "models":
        raise ValueError("fresh workspace model and receipt required")
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    selected = [row for row in info.siblings if row.rfilename.endswith((".json", ".jinja", ".txt")) or
                row.rfilename == "transformer/diffusion_pytorch_model.safetensors"]
    selected = [row for row in selected if row.rfilename != "text_encoder/model.safetensors.index.json"]
    required = sum(row.size for row in selected)
    free = shutil.disk_usage(ROOT).free
    if free - required < RESERVE:
        raise ValueError("restore exceeds two-GiB reserve; originals never deleted")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(schema="tc-flux4-asset-restore-v1", status="running", repo=REPO, revision=REVISION,
                  runtime_verified=False, performance_verified=False, model_root=str(model.relative_to(ROOT)),
                  download_bytes=required, initial_free_bytes=free, reserve_bytes=RESERVE)
    write_json(output, result)
    try:
        result["encoder_reuse"] = verify_encoder()
        hashes = registered_hashes([VAE], ROOT / "outputs/benchmark-source-registrations/flux9-bf16.json")
        remote_vae = next(row for row in info.siblings if row.rfilename == "vae/diffusion_pytorch_model.safetensors")
        if hashes[str(VAE)] != remote_vae.lfs.sha256:
            raise ValueError("shared VAE differs from original FLUX4B checksum")
        result["vae_reuse"] = dict(source=str(VAE.relative_to(ROOT)), sha256=hashes[str(VAE)],
                                   scope="same original4B VAE payload; directory name does not mean9B DiT reuse")
        write_json(output, result)
        snapshot_download(REPO, revision=REVISION, token=False, local_dir=model,
                          allow_patterns=[row.rfilename for row in selected], max_workers=2)
        dit = model / "transformer/diffusion_pytorch_model.safetensors"
        expected_dit = next(row for row in selected if row.rfilename == "transformer/diffusion_pytorch_model.safetensors")
        if dit.stat().st_size != expected_dit.size or sha256_file(dit) != expected_dit.lfs.sha256:
            raise ValueError("downloaded original4B DiT extent/SHA mismatch")
        (model / "text_encoder/model.safetensors").symlink_to(ENCODER)
        (model / "vae/diffusion_pytorch_model.safetensors").symlink_to(VAE)
        result["dit"] = dict(path=str(dit.relative_to(ROOT)), bytes=expected_dit.size, sha256=expected_dit.lfs.sha256)
        result.update(status="complete_assets_only", final_free_bytes=shutil.disk_usage(ROOT).free)
        if result["final_free_bytes"] < RESERVE:
            raise ValueError("post-restore reserve violated; retain assets and failure receipt")
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output, result)
    print(json.dumps(dict(status=result["status"], model_root=result["model_root"], download_bytes=required)))


def restore_quantized(model, output):
    # A separately labelled official mflux pack; never silently substitute
    # local Z weights after the original-BF16 identity check failed.
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    from huggingface_hub import HfApi, snapshot_download
    repo = "mflux-community/flux2-klein-4b-mflux-q4"
    revision = "794cd159538149ad9830848508c31f0ea7088e58"
    if model.exists() or output.exists() or model.parent != ROOT / "models":
        raise ValueError("fresh explicitly quantized model and receipt required")
    info = HfApi(token=False).model_info(repo, revision=revision, files_metadata=True)
    selected = [row for row in info.siblings if row.rfilename.endswith((".safetensors", ".json", ".jinja"))]
    required = sum(row.size for row in selected)
    free = shutil.disk_usage(ROOT).free
    if free - required < RESERVE:
        raise ValueError("quantized pack exceeds two-GiB reserve")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(schema="tc-flux4-asset-restore-v1", status="running", repo=repo, revision=revision,
        runtime_verified=False, performance_verified=False, model_root=str(model.relative_to(ROOT)),
        representation="upstream mflux Q4 pack; own recipe, not originalBF16 or local Z encoder reuse",
        download_bytes=required, initial_free_bytes=free, reserve_bytes=RESERVE, files=[])
    write_json(output, result)
    try:
        snapshot_download(repo, revision=revision, token=False, local_dir=model,
                          allow_patterns=[row.rfilename for row in selected], max_workers=2)
        for row in selected:
            path = model / row.rfilename
            digest = sha256_file(path)
            if path.stat().st_size != row.size or (row.lfs and digest != row.lfs.sha256):
                raise ValueError("pinned downloaded pack identity differs: " + row.rfilename)
            result["files"].append(dict(path=row.rfilename, bytes=row.size, sha256=digest))
        result.update(status="complete_assets_only", final_free_bytes=shutil.disk_usage(ROOT).free)
        if result["final_free_bytes"] < RESERVE:
            raise ValueError("post-download reserve violated, retain failure receipt")
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output, result)
    print(json.dumps(dict(status=result["status"], model_root=result["model_root"], download_bytes=required)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/FLUX.2-klein-4B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=("bf16", "mflux-q4"), default="bf16")
    args = parser.parse_args()
    (restore_quantized if args.representation == "mflux-q4" else restore)(args.model.absolute(), args.output.absolute())


if __name__ == "__main__":
    main()

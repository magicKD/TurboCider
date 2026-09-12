#!/usr/bin/env python3
"""Prepare the ModelScope FastH3 Preview v1 INT6 MLX artifact.

Downloading and conversion are intentionally separate.  Fetch the published
checkpoint with ``download_fasth3_modelscope.py`` first, then pass its local
``transformer/`` directory to this tool.  This tool performs no remote model
access and never deletes the source checkpoint automatically: callers should
verify the converted manifest before removing the 66.25 GB (61.70 GiB) BF16
transformer.

Examples::

    python tools/h3/prepare_fasth3_mlx.py \
      --source /models/FastH3-ModelScope/transformer \
      --out /models/FastH3-MLX --formats int6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_REPO = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
DEFAULT_REVISION = "master"
TRANSFORMER_INDEX = "diffusion_pytorch_model.safetensors.index.json"
REQUIRED_FILES = ("config.json",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def _safetensors_header(path: Path) -> dict:
    """Read only one safetensors header (no tensor payload)."""
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        header_bytes = struct.unpack("<Q", raw)[0]
        if header_bytes <= 0 or header_bytes > (1 << 30):
            raise ValueError(f"invalid safetensors header size {header_bytes}: {path}")
        payload = stream.read(header_bytes)
        if len(payload) != header_bytes:
            raise ValueError(f"truncated safetensors metadata: {path}")
    return json.loads(payload)


def ensure_transformer_index(root: Path) -> Path:
    """Use the published index or synthesize one for ModelScope shard layouts.

    The ModelScope repository currently publishes numbered transformer shards
    without the diffusers index file. FastVideo already supports globbing those
    shards, but TurboCider's validation and provenance need a deterministic
    index. Building it from headers is cheap and never reads tensor payloads.
    """
    index_path = root / TRANSFORMER_INDEX
    if index_path.is_file():
        return index_path
    shards = sorted(root.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if not shards:
        shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"transformer has no safetensors shards under {root}")
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard in shards:
        header = _safetensors_header(shard)
        for key, record in header.items():
            if key == "__metadata__":
                continue
            if key in weight_map:
                raise ValueError(f"duplicate transformer tensor across shards: {key}")
            weight_map[key] = shard.name
            offsets = record.get("data_offsets")
            if isinstance(offsets, list) and len(offsets) == 2:
                total_size += int(offsets[1]) - int(offsets[0])
    index_path.write_text(json.dumps({
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }, indent=2, sort_keys=True) + "\n")
    return index_path


def available_bytes(path: Path) -> int:
    return shutil.disk_usage(path if path.exists() else path.parent).free


def source_checkpoint_sha256(source: Path) -> str:
    content_path = source.parent / "checkpoint_content.json"
    if not content_path.is_file():
        return ""
    value = json.loads(content_path.read_text()).get("aggregate_sha256", "")
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid aggregate_sha256 in {content_path}")
    return value


def download_file(url: str, target: Path, *, retries: int = 3) -> None:
    """Resume one large file directly into the final directory.

    A content-addressed snapshot cache would keep a second local copy, which
    is unsafe for a 65 GiB transformer on a nearly full SSD.  This downloader
    uses only a ``.part`` sibling and atomically renames it after a successful
    response; no second shard copy is created.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        print(f"[skip] {target}", flush=True)
        return
    part = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    offset = 0
                    part.unlink(missing_ok=True)
                mode = "ab" if offset else "wb"
                content_length = int(response.headers.get("Content-Length", "0"))
                total = offset + content_length if content_length else 0
                downloaded = offset
                last_report = time.monotonic()
                with part.open(mode) as stream:
                    while True:
                        chunk = response.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report > 10:
                            suffix = f"/{total / 2**30:.2f} GiB" if total else ""
                            print(f"  {target.name}: {downloaded / 2**30:.2f} GiB{suffix}", flush=True)
                            last_report = now
                if total and downloaded != total:
                    raise IOError(f"short response ({downloaded} != {total} bytes)")
            part.replace(target)
            print(f"[done] {target}", flush=True)
            return
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            print(f"[retry {attempt}/{retries}] {target.name}: {error}", flush=True)
            if attempt == retries:
                raise


def validate_transformer(root: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"transformer snapshot is incomplete under {root}: missing {missing}")
    index = json.loads(ensure_transformer_index(root).read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("transformer index has no non-empty weight_map")
    missing_shards = sorted({name for name in weight_map.values() if not (root / name).is_file()})
    if missing_shards:
        raise FileNotFoundError(f"transformer snapshot is missing shards: {missing_shards[:8]}")
    config = json.loads((root / "config.json").read_text())
    # The student provenance is intentionally checked before spending hours on
    # a conversion.  These fields are present in the released Preview config.
    if int(config.get("num_layers", config.get("num_hidden_layers", 50))) != 50:
        raise ValueError("source transformer does not look like the 50-block FastH3 student")


def run_conversion(source: Path, out: Path, formats: str, include_vsa: bool) -> None:
    reference = Path(__file__).resolve().parents[2] / "../references/FastVideo"
    converter = reference / "scripts/checkpoint_conversion/convert_minimax_h3_mlx.py"
    runner = Path(__file__).resolve().with_name("run_fastvideo_mlx_converter.py")
    if not converter.is_file():
        raise FileNotFoundError(f"FastVideo converter not found: {converter}")
    if not runner.is_file():
        raise FileNotFoundError(f"FastVideo MLX converter bootstrap not found: {runner}")
    command = [
        sys.executable,
        str(runner),
        "--model-root",
        str(source),
        "--out",
        str(out),
        "--formats",
        formats,
    ]
    if include_vsa:
        command.append("--include-vsa")
    subprocess.run(command, cwd=reference, check=True)


def validate_output(out: Path, formats: list[str], source: Path,
                    repository: str, revision: str) -> None:
    for fmt in formats:
        artifact = out / fmt
        manifest_path = artifact / "mlx_h3_dit.json"
        weights_path = artifact / "mlx_h3_dit.safetensors"
        if not manifest_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"conversion did not produce a complete {fmt} artifact")
        manifest = json.loads(manifest_path.read_text())
        quant = manifest.get("quantization") or {}
        if quant.get("mode") != "affine" or quant.get("bits") != 6 or quant.get("group_size") != 64:
            raise ValueError(f"{artifact} is not affine INT6/g64: {quant}")
        if manifest.get("num_blocks") != 50 or manifest.get("num_refiner_blocks") != 2:
            raise ValueError(f"{artifact} is not a FastH3 50+2 checkpoint")
        manifest.setdefault("source", {})
        manifest["source"].update({
            "repository": repository,
            "revision": revision,
            "transformer_index_sha256": sha256_file(source / TRANSFORMER_INDEX),
        })
        checkpoint_sha256 = source_checkpoint_sha256(source)
        if checkpoint_sha256:
            manifest["source"]["checkpoint_sha256"] = checkpoint_sha256
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"[ready] {artifact} ({directory_bytes(artifact) / 2**30:.2f} GiB)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="local transformer/ directory downloaded from ModelScope")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--repository", default=DEFAULT_REPO,
                        help="source ModelScope repository recorded in the manifest")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--formats", default="int6")
    parser.add_argument("--include-vsa", action="store_true")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate the local ModelScope transformer but do not convert it")
    parser.add_argument("--min-free-gib", type=float, default=35.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = args.formats.split()
    if formats != ["int6"]:
        raise ValueError("TurboCider's first product profile accepts only --formats int6")
    source = args.source.expanduser().resolve()
    validate_transformer(source)
    if args.validate_only:
        print(f"[ready] transformer snapshot: {source} ({directory_bytes(source) / 2**30:.2f} GiB)")
        return
    free_gib = available_bytes(args.out.expanduser().resolve().parent) / 2**30
    if free_gib < args.min_free_gib:
        raise RuntimeError(f"only {free_gib:.1f} GiB free; refusing conversion below {args.min_free_gib:.1f} GiB")
    run_conversion(source, args.out.expanduser().resolve(), args.formats, args.include_vsa)
    validate_output(args.out.expanduser().resolve(), formats, source,
                    args.repository, args.revision)


if __name__ == "__main__":
    main()

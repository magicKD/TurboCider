from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

MAC_LOCAL_AI = Path(__file__).resolve().parents[2] / "mac_local_ai"
sys.path.insert(0, str(MAC_LOCAL_AI))

from research.scripts.safetensor_io import SafeTensorCollection, SafeTensorFile  # noqa: E402
from scripts.flux2_checkpoint import inspect_model_spec  # noqa: E402


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header = {}
    chunks = []
    offset = 0
    for name, value in tensors.items():
        array = np.asarray(value, dtype=np.float16)
        payload = array.tobytes()
        header[name] = {
            "dtype": "F16",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        chunks.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def test_collection_reads_diffusers_indexed_shards(tmp_path: Path) -> None:
    transformer = tmp_path / "model" / "transformer"
    transformer.mkdir(parents=True)
    first = transformer / "diffusion_pytorch_model-00001-of-00002.safetensors"
    second = transformer / "diffusion_pytorch_model-00002-of-00002.safetensors"
    _write_safetensors(first, {"first": np.array([1.0, 2.0])})
    _write_safetensors(second, {"second": np.array([3.0, 4.0])})
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({
            "weight_map": {
                "first": first.name,
                "second": second.name,
            }
        })
    )

    with SafeTensorCollection(tmp_path / "model") as checkpoint:
        assert checkpoint.path == transformer.resolve()
        np.testing.assert_array_equal(checkpoint.tensor("first"), [1.0, 2.0])
        np.testing.assert_array_equal(checkpoint.tensor("second"), [3.0, 4.0])
        assert checkpoint.total_bytes == first.stat().st_size + second.stat().st_size


def test_collection_rejects_missing_indexed_shard(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"missing": "missing.safetensors"}})
    )
    with pytest.raises(FileNotFoundError, match="missing/escaped shard"):
        SafeTensorCollection(tmp_path)


def test_collection_rejects_index_tensor_mapped_to_wrong_shard(tmp_path: Path) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    _write_safetensors(first, {"first": np.array([1.0])})
    _write_safetensors(second, {"second": np.array([2.0])})
    index = tmp_path / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(json.dumps({
        "weight_map": {"first": second.name, "second": first.name}
    }))
    with pytest.raises(ValueError, match="wrong shard"):
        SafeTensorCollection(index)


def test_collection_honors_an_explicit_index_when_two_are_present(tmp_path: Path) -> None:
    model_shard = tmp_path / "model-shard.safetensors"
    diffusion_shard = tmp_path / "diffusion-shard.safetensors"
    _write_safetensors(model_shard, {"model_tensor": np.array([1.0])})
    _write_safetensors(diffusion_shard, {"diffusion_tensor": np.array([2.0])})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model_tensor": model_shard.name}
    }))
    diffusion_index = tmp_path / "diffusion_pytorch_model.safetensors.index.json"
    diffusion_index.write_text(json.dumps({
        "weight_map": {"diffusion_tensor": diffusion_shard.name}
    }))

    with SafeTensorCollection(diffusion_index) as checkpoint:
        assert checkpoint.reference_path == diffusion_index.resolve()
        assert set(checkpoint.header) == {"diffusion_tensor"}
        np.testing.assert_array_equal(checkpoint.tensor("diffusion_tensor"), [2.0])


def test_collection_rejects_tensor_absent_from_index(tmp_path: Path) -> None:
    shard = tmp_path / "shard.safetensors"
    _write_safetensors(shard, {
        "indexed": np.array([1.0]),
        "extra": np.array([2.0]),
    })
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"indexed": shard.name}}))

    with pytest.raises(ValueError, match="absent from the index"):
        SafeTensorCollection(index)


def test_collection_closes_open_shards_when_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    _write_safetensors(first, {"duplicate": np.array([1.0])})
    _write_safetensors(second, {"duplicate": np.array([2.0])})
    closed: list[Path] = []
    original_close = SafeTensorFile.close

    def recording_close(handle: SafeTensorFile) -> None:
        closed.append(handle.path)
        original_close(handle)

    monkeypatch.setattr(SafeTensorFile, "close", recording_close)
    with pytest.raises(ValueError, match="duplicate tensor"):
        SafeTensorCollection(tmp_path)
    assert set(closed) == {first, second}


def test_safetensor_file_closes_stream_after_invalid_header(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.safetensors"
    malformed.write_bytes(b"short")
    with pytest.raises(ValueError, match="truncated safetensors header prefix"):
        SafeTensorFile(malformed)


def test_model_spec_is_derived_from_sharded_headers(tmp_path: Path) -> None:
    hidden = 4
    mlp_width = 12
    wide_output = 3 * hidden + 2 * mlp_width
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    _write_safetensors(first, {
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight": np.zeros(
            (wide_output, hidden), dtype=np.float16
        ),
        "single_transformer_blocks.0.attn.to_out.weight": np.zeros(
            (hidden, hidden + mlp_width), dtype=np.float16
        ),
    })
    _write_safetensors(second, {
        "single_transformer_blocks.1.attn.to_qkv_mlp_proj.weight": np.zeros(
            (wide_output, hidden), dtype=np.float16
        ),
        "single_transformer_blocks.1.attn.to_out.weight": np.zeros(
            (hidden, hidden + mlp_width), dtype=np.float16
        ),
    })

    with SafeTensorCollection(tmp_path) as checkpoint:
        spec = inspect_model_spec(checkpoint)
    assert spec.hidden == hidden
    assert spec.mlp_width == mlp_width
    assert spec.wide_output == wide_output
    assert spec.block_count == 2

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/native/benchmark_h3_mlx.py"
SPEC = importlib.util.spec_from_file_location("benchmark_h3_mlx", SCRIPT)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def arguments(tmp_path: Path) -> argparse.Namespace:
    binary = tmp_path / "turbocider"
    python = tmp_path / "python"
    runner = tmp_path / "runner.py"
    for path in (binary, python, runner):
        path.write_text("fixture")
    model = tmp_path / "model"
    checkpoint = tmp_path / "checkpoint"
    model.mkdir()
    checkpoint.mkdir()
    (model / "modelscope_download.json").write_text(json.dumps({
        "model_id": benchmark.MODEL_ID,
        "endpoint": benchmark.MODEL_ENDPOINT,
    }))
    (checkpoint / "mlx_h3_dit.json").write_text(json.dumps({
        "quantization": {"bits": 6, "group_size": 64, "mode": "affine"},
        "source": {"repository": benchmark.MODEL_ID},
    }))
    return argparse.Namespace(
        turbocider_bin=binary,
        python=python,
        fastvideo_runner=runner,
        model_root=model,
        checkpoint=checkpoint,
        profile="dense",
        output_dir=tmp_path / "output",
        prompts=["A red fox runs through fresh snow."],
        seeds=[2026],
        modes=["warm"],
        blocks=1,
        warmups=1,
        width=832,
        height=480,
        frames=124,
        fps=24,
        steps=4,
        max_overhead_ratio=1.15,
        resume=False,
        keep_media=False,
        vsa=False,
        vsa_sparsity=0.9,
        vsa_tile_size=64,
        vsa_prefix_mode="exempt",
        vsa_dense_first_n_steps=0,
        vsa_dense_layers=[],
        vsa_impl="reference",
    )


def test_abba_schedule_is_balanced_and_ordered() -> None:
    assert benchmark.abba_schedule(2) == [
        "fastvideo", "turbocider", "turbocider", "fastvideo",
        "fastvideo", "turbocider", "turbocider", "fastvideo",
    ]


def test_benchmark_requires_modelscope_int6_contract(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    benchmark.validate_args(args)
    manifest = json.loads((args.model_root / "modelscope_download.json").read_text())
    manifest["endpoint"] = "https://huggingface.co"
    (args.model_root / "modelscope_download.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ModelScope"):
        benchmark.validate_args(args)


def test_commands_use_same_geometry_and_local_assets(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    output = tmp_path / "out.mp4"
    command = benchmark.fastvideo_command(args, args.prompts[0], 2026, output, tmp_path / "cache")
    assert "--num-frames" in command and command[command.index("--num-frames") + 1] == "124"
    assert "--video-decode-backend" in command
    assert "h3-vae" in command
    request = benchmark.turbocider_request(args, args.prompts[0], 2026, output)
    assert request["model"] == "minimax-h3-fasth3-mlx-int6"
    assert request["frames"] == 124 and request["steps"] == 4
    assert request["audio"] is True


def test_vsa_commands_and_provenance_are_explicit(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    args.profile = "vsa"
    args.vsa = True
    args.vsa_dense_layers = [3, 7]
    args.model_root.joinpath("modelscope_download.json").write_text(json.dumps({
        "model_id": benchmark.VSA_MODEL_ID,
        "endpoint": benchmark.MODEL_ENDPOINT,
    }))
    args.checkpoint.joinpath("mlx_h3_dit.json").write_text(json.dumps({
        "quantization": {"bits": 6, "group_size": 64, "mode": "affine"},
        "source": {"repository": benchmark.VSA_MODEL_ID},
        "vsa": {"capable": True, "num_gate_matrices": 50},
    }))
    benchmark.validate_args(args)
    request = benchmark.turbocider_request(args, args.prompts[0], 2026, tmp_path / "vsa.mp4")
    assert request["model"] == "minimax-h3-fasth3-mlx-int6-vsa"
    assert request["vsa"] is True and request["vsa_dense_layers"] == [3, 7]
    command = benchmark.fastvideo_command(args, args.prompts[0], 2026,
                                          tmp_path / "vsa.mp4", tmp_path / "cache")
    assert "--vsa" in command and "--vsa-dense-layers" in command


def test_json_parser_ignores_progress_objects() -> None:
    text = '{"phase":"progress"}\nnoise\n{\n  "timings_s": {"generate_s": 1.0}\n}\n'
    assert benchmark.extract_last_json(text)["timings_s"]["generate_s"] == 1.0


def test_process_tree_rss_includes_descendants_only() -> None:
    output = """
      10   1 100
      11  10 200
      12  11 300
      20   1 900
    """
    assert benchmark.process_tree_rss_bytes(output, 10) == (100 + 200 + 300) * 1024


def test_absolute_path_preserves_managed_python_symlink(tmp_path: Path,
                                                        monkeypatch) -> None:
    target = tmp_path / "external-python"
    target.write_text("fixture")
    managed = tmp_path / "Python" / "bin" / "python"
    managed.parent.mkdir(parents=True)
    managed.symlink_to(target)
    monkeypatch.chdir(tmp_path)
    normalized = benchmark.absolute_without_resolving_symlinks(
        Path("Python/bin/python"))
    assert normalized == managed
    assert normalized.resolve() == target

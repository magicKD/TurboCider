import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/native/benchmark_h3.py"
SPEC = importlib.util.spec_from_file_location("benchmark_h3", SCRIPT)
assert SPEC and SPEC.loader
benchmark_h3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_h3)


def arguments(tmp_path: Path, **overrides) -> argparse.Namespace:
    h3 = tmp_path / "h3"
    turbocider = tmp_path / "turbocider"
    model = tmp_path / "model"
    h3.write_text("fixture")
    turbocider.write_text("fixture")
    model.mkdir()
    values = {
        "h3_bin": h3,
        "turbocider_bin": turbocider,
        "model": model,
        "output_dir": tmp_path / "output",
        "prompt": "prompt",
        "width": 512,
        "height": 512,
        "frames": 22,
        "fps": 24,
        "steps": 4,
        "seed": 42,
        "residency": "streamed",
        "memory_budget_bytes": 0,
        "rounds": 2,
        "max_overhead_percent": 5.0,
        "allow_approximation": False,
        "allow_output_difference": False,
        "lora": None,
        "lora_strength": 0.0625,
        "overwrite": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_exact_direct_command_matches_turbocider_h3_flags(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    benchmark_h3.validate_args(args)
    command = benchmark_h3.build_direct_command(args, tmp_path / "direct.mp4")
    assert "--ssd-streaming" in command
    assert "--use-reference-rope" in command
    assert "--use-slower-bf16-mlp" in command
    assert "--use-slower-bf16-qkv" in command
    assert "--use-slower-bf16-attention-output" in command


def test_lora_request_is_manifest_bound(tmp_path: Path) -> None:
    lora = tmp_path / "adapter.safetensors"
    lora.write_bytes(b"fixture")
    args = arguments(tmp_path, lora=lora)
    benchmark_h3.validate_args(args)
    request = benchmark_h3.build_turbocider_request(args, tmp_path / "out.mp4")
    assert request["loras"] == [
        {
            "path": str(lora.resolve()),
            "role": "transformer",
            "strength": 0.0625,
        }
    ]
    assert request["execution"]["residency"] == "streamed"


def test_memory_budget_is_forwarded_for_streamed_requests(tmp_path: Path) -> None:
    args = arguments(tmp_path, memory_budget_bytes=16 << 30)
    benchmark_h3.validate_args(args)
    request = benchmark_h3.build_turbocider_request(args, tmp_path / "out.mp4")
    assert request["execution"]["memory_budget_bytes"] == 16 << 30


def test_memory_budget_requires_streamed_residency(tmp_path: Path) -> None:
    args = arguments(tmp_path, residency="resident", memory_budget_bytes=16 << 30)
    with pytest.raises(ValueError, match="streamed residency"):
        benchmark_h3.validate_args(args)


def test_retained_probe_reports_request_deltas_and_resets_the_dit() -> None:
    source = (ROOT / "tools/native/h3_dit_streaming_probe.c").read_text()
    assert "h3_dit_reset_run(dit" in source
    assert '\\"bytes_read_delta\\"' in source
    assert '\\"wait_seconds_delta\\"' in source
    assert '\\"outputs_equal\\"' in source


def test_approximation_cannot_be_mislabeled_exact(tmp_path: Path) -> None:
    args = arguments(tmp_path, allow_approximation=True)
    with pytest.raises(ValueError, match="allow-output-difference"):
        benchmark_h3.validate_args(args)


def test_invalid_h3_geometry_is_rejected(tmp_path: Path) -> None:
    args = arguments(tmp_path, width=500)
    with pytest.raises(ValueError, match="multiples of 32"):
        benchmark_h3.validate_args(args)


def test_report_records_the_actual_peer_hash_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = arguments(tmp_path, rounds=1)
    benchmark_h3.validate_args(args)

    def fake_run(command: list[str], _cwd: Path):
        if command[1] == "doctor":
            return 0.01, '{"gpu_available":true,"gpu":"fixture"}\n', ""
        if command[0] == str(args.h3_bin):
            output = Path(command[command.index("-o") + 1])
            output.write_bytes(b"same media")
            return 1.0, "", ""
        request = json.loads(Path(command[-1]).read_text())
        output = Path(request["outputs"][0]["path"])
        output.write_bytes(b"same media")
        return 1.02, '{"seconds":1.0}\n', ""

    monkeypatch.setattr(benchmark_h3, "run_process", fake_run)
    monkeypatch.setattr(benchmark_h3, "probe_media", lambda _path: {})
    report, passed = benchmark_h3.benchmark(args)
    assert passed
    assert report["comparison"]["byte_exact_output"] is True
    assert all(run["output_matches_peer"] is True for run in report["runs"])

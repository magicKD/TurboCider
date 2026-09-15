import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/native/benchmark_ltx_gemma_encoder_sweep.py"
SPEC = importlib.util.spec_from_file_location("benchmark_ltx_gemma_encoder_sweep", SCRIPT)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def gate_args() -> argparse.Namespace:
    return argparse.Namespace(
        max_relative_l2=0.025,
        min_cosine=0.999,
        max_relative_abs=0.1,
    )


def test_quality_gate_covers_video_audio_and_mask(tmp_path: Path) -> None:
    import numpy as np

    staged = tmp_path / "staged"
    fused = tmp_path / "fused"
    staged.mkdir()
    fused.mkdir()
    arrays = {
        "video": np.arange(8, dtype=np.float32),
        "audio": np.arange(6, dtype=np.float32) + 1,
        "mask": np.zeros(2, dtype=np.float32),
    }
    for name, values in arrays.items():
        words = (values.view(np.uint32) >> np.uint32(16)).astype("<u2")
        words.tofile(staged / f"{name}.bf16")
        words.tofile(fused / f"{name}.bf16")
    qualities = {
        name: benchmark.tensor_quality(
            fused / f"{name}.bf16", staged / f"{name}.bf16"
        ) for name in arrays
    }
    assert benchmark.quality_gate(qualities, gate_args())

    changed = np.array([1.0, 0.0], dtype=np.float32)
    (changed.view(np.uint32) >> np.uint32(16)).astype("<u2").tofile(
        fused / "mask.bf16"
    )
    qualities["mask"] = benchmark.tensor_quality(
        fused / "mask.bf16", staged / "mask.bf16"
    )
    assert not benchmark.quality_gate(qualities, gate_args())


def test_conditioning_quality_rejects_row_mismatch() -> None:
    result = benchmark.conditioning_quality(
        {"rows": 2}, {"rows": 1}
    )
    assert set(result) == {"video", "audio", "mask"}
    assert all(not quality["shape_match"] for quality in result.values())


def test_run_case_parses_resident_first_and_warm_telemetry(
    tmp_path: Path, monkeypatch
) -> None:
    import numpy as np

    payload = {
        "format": "turbocider-native-gemma-raw-candidate-v1",
        "rows": 2,
        "video_dim": benchmark.VIDEO_DIM,
        "audio_dim": benchmark.AUDIO_DIM,
        "fused_mlp": False,
        "gpu_taps": False,
        "runs": 3,
        "warm_runs": 2,
        "create_seconds": 0.25,
        "first_seconds": 1.5,
        "warm_seconds": [0.9, 1.0],
        "resident_weights_enabled": True,
        "resident_weight_cache_hits": 292,
        "resident_weight_cache_misses": 0,
        "resident_weight_bytes": 12_000_000_000,
        "ane_plan_reason": "no_ane_manifest",
    }

    def fake_run(command, **_kwargs):
        folder = Path(command[-2])
        folder.mkdir(parents=True, exist_ok=True)
        np.zeros(payload["rows"] * benchmark.VIDEO_DIM,
                 dtype="<u2").tofile(folder / "raw_video_context.bf16")
        np.zeros(payload["rows"] * benchmark.AUDIO_DIM,
                 dtype="<u2").tofile(folder / "raw_audio_context.bf16")
        np.zeros(payload["rows"], dtype="<u2").tofile(folder / "raw_text_mask.bf16")
        diagnostic = (
            benchmark.COREML_E5_DIAGNOSTIC_START +
            "/tmp/test.bundle/main_ane/model.anehash" +
            benchmark.COREML_E5_DIAGNOSTIC_END
        )
        return SimpleNamespace(
            returncode=0,
            stdout=diagnostic + json.dumps(payload, separators=(",", ":")) +
                   "\n",
            stderr="123 maximum resident set size\n",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setenv("TURBOCIDER_LTX_GEMMA_RESIDENT_WEIGHTS", "1")
    result = benchmark.run_case(
        ["probe", "checkpoint", "tokenizer", "shader", "prompt",
         str(tmp_path / "case"), "2"],
        tmp_path / "case", 60, False,
    )
    assert result["runs"] == 3
    assert result["warm_runs"] == 2
    assert result["warm_median_seconds"] == 0.95
    assert result["warm_min_seconds"] == 0.9
    assert result["warm_max_seconds"] == 1.0
    assert 0 < result["warm_cv"] < 0.1
    assert result["process_peak_rss_bytes"] == 123
    assert result["resident_weights_enabled"]
    assert result["resident_weight_cache_hits"] == 292
    assert result["resident_weight_cache_misses"] == 0
    assert result["resident_weight_bytes"] == 12_000_000_000
    assert result["coreml_stdout_diagnostics"] == 1
    assert result["video"].stat().st_size == 2 * benchmark.VIDEO_DIM * 2


def test_probe_stdout_rejects_unknown_preamble_or_multiple_payloads(
    tmp_path: Path,
) -> None:
    payload = json.dumps(
        {"format": benchmark.PROBE_FORMAT}, separators=(",", ":"),
    )
    try:
        benchmark.parse_probe_stdout("unknown diagnostic" + payload, tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown probe stdout preamble was accepted")

    try:
        benchmark.parse_probe_stdout(payload + payload, tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("multiple probe payloads were accepted")


def test_run_case_selects_fused_and_gpu_tap_environment(
    tmp_path: Path, monkeypatch
) -> None:
    import numpy as np

    payload = {
        "format": "turbocider-native-gemma-raw-candidate-v1",
        "rows": 1,
        "video_dim": benchmark.VIDEO_DIM,
        "audio_dim": benchmark.AUDIO_DIM,
        "fused_mlp": False,
        "gpu_taps": False,
        "runs": 2,
        "warm_runs": 1,
        "create_seconds": 0.25,
        "first_seconds": 1.5,
        "warm_seconds": [0.9],
        "ane_plan_reason": "no_ane_manifest",
    }
    environments = {}

    def fake_run(command, **kwargs):
        folder = Path(command[-2])
        folder.mkdir(parents=True, exist_ok=True)
        np.zeros(benchmark.VIDEO_DIM, dtype="<u2").tofile(
            folder / "raw_video_context.bf16"
        )
        np.zeros(benchmark.AUDIO_DIM, dtype="<u2").tofile(
            folder / "raw_audio_context.bf16"
        )
        np.zeros(1, dtype="<u2").tofile(folder / "raw_text_mask.bf16")
        environments[folder.name] = {
            name: kwargs["env"].get(name)
            for name in (
                "TURBOCIDER_LTX_GEMMA_FUSED_MLP",
                "TURBOCIDER_LTX_GEMMA_GPU_TAPS",
            )
        }
        case_payload = {
            **payload,
            "fused_mlp": environments[folder.name][
                "TURBOCIDER_LTX_GEMMA_FUSED_MLP"
            ] == "1",
            "gpu_taps": environments[folder.name][
                "TURBOCIDER_LTX_GEMMA_GPU_TAPS"
            ] == "1",
        }
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(case_payload) + "\n",
            stderr="123 maximum resident set size\n",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setenv("TURBOCIDER_LTX_GEMMA_FUSED_MLP", "inherited")
    monkeypatch.setenv("TURBOCIDER_LTX_GEMMA_GPU_TAPS", "inherited")
    for variant in benchmark.VARIANTS:
        folder = tmp_path / variant
        command = [
            "probe", "checkpoint", "tokenizer", "shader", "prompt",
            str(folder), "1",
        ]
        benchmark.run_case(command, folder, 60, variant)

    assert environments == {
        "staged": {
            "TURBOCIDER_LTX_GEMMA_FUSED_MLP": None,
            "TURBOCIDER_LTX_GEMMA_GPU_TAPS": None,
        },
        "fused": {
            "TURBOCIDER_LTX_GEMMA_FUSED_MLP": "1",
            "TURBOCIDER_LTX_GEMMA_GPU_TAPS": None,
        },
        "fused_gpu_taps": {
            "TURBOCIDER_LTX_GEMMA_FUSED_MLP": "1",
            "TURBOCIDER_LTX_GEMMA_GPU_TAPS": "1",
        },
    }
    assert json.loads(
        (tmp_path / "fused_gpu_taps" / "process.json").read_text()
    )["variant"] == "fused_gpu_taps"


def test_resident_weight_telemetry_is_fail_closed(tmp_path: Path) -> None:
    enabled = {
        "resident_weights_enabled": True,
        "resident_weight_cache_hits": 292,
        "resident_weight_cache_misses": 0,
        "resident_weight_bytes": 12_000_000_000,
    }
    assert benchmark.resident_weight_telemetry(
        enabled, True, tmp_path,
    )["resident_weight_cache_hits"] == 292

    for malformed in (
        {**enabled, "resident_weight_cache_hits": 0},
        {**enabled, "resident_weight_cache_misses": 1},
        {**enabled, "resident_weight_bytes": 0},
        {**enabled, "resident_weights_enabled": False},
    ):
        try:
            benchmark.resident_weight_telemetry(malformed, True, tmp_path)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid resident telemetry was accepted")

    assert benchmark.resident_weight_telemetry(
        {}, False, tmp_path,
    ) == {
        "resident_weights_enabled": False,
        "resident_weight_cache_hits": 0,
        "resident_weight_cache_misses": 0,
        "resident_weight_bytes": 0,
    }


def test_variant_order_rotates_all_requested_modes() -> None:
    variants = list(benchmark.VARIANTS)
    assert benchmark.variant_order(variants, 0) == variants
    assert benchmark.variant_order(variants, 1) == variants[1:] + variants[:1]
    assert benchmark.variant_order(variants, 3) == variants


def test_ane_telemetry_is_fail_closed_and_internally_consistent(
    tmp_path: Path,
) -> None:
    payload = {
        "rows": 128,
        "ane_requested": True,
        "ane_used": True,
        "ane_layers_available": 8,
        "ane_layers_attempted": 8,
        "ane_layers_succeeded": 8,
        "ane_layers_fallback": 0,
        "ane_cache_hits": 8,
        "ane_cache_misses": 0,
        "ane_preload_models_session_total": 8,
        "ane_preload_workers": 4,
        "ane_preload_seconds_session_total": 5.75,
        "ane_selected_bucket": 128,
        "ane_padding_rows": 0,
        "ane_minimum_profitable_rows": 128,
        "ane_execution_rows": 1024,
        "ane_total_seconds": 0.05,
        "ane_output_backing_used": True,
        "ane_plan_reason": "exact_bucket",
    }
    telemetry = benchmark.ane_telemetry(payload, True, tmp_path)
    assert telemetry["ane_layers_succeeded"] == 8
    assert telemetry["ane_cache_hits"] == 8
    assert telemetry["ane_preload_models_session_total"] == 8
    assert telemetry["ane_preload_workers"] == 4

    malformed = {**payload, "ane_cache_hits": 7}
    try:
        benchmark.ane_telemetry(malformed, True, tmp_path)
    except ValueError as error:
        assert "inconsistent ANE execution telemetry" in str(error)
    else:
        raise AssertionError("inconsistent cache telemetry was accepted")

    malformed_preload = {**payload, "ane_preload_workers": 0}
    try:
        benchmark.ane_telemetry(malformed_preload, True, tmp_path)
    except ValueError as error:
        assert "inconsistent ANE execution telemetry" in str(error)
    else:
        raise AssertionError("inconsistent preload telemetry was accepted")

    gpu_only = {
        "rows": 129, "ane_requested": False,
        "ane_plan_reason": "no_ane_manifest",
    }
    assert not benchmark.ane_telemetry(
        gpu_only, False, tmp_path
    )["ane_used"]


def test_warm_qualification_rejects_too_few_or_unstable_samples() -> None:
    stable = {"warm_runs": 3, "warm_cv": 0.05}
    unstable = {"warm_runs": 3, "warm_cv": 0.5}
    too_short = {"warm_runs": 1, "warm_cv": 0.0}
    assert benchmark.warm_qualification(stable, stable, 0.25) == {
        "warm_samples_passed": True,
        "stability_passed": True,
    }
    assert not benchmark.warm_qualification(
        stable, unstable, 0.25
    )["stability_passed"]
    assert not benchmark.warm_qualification(
        stable, too_short, 0.25
    )["warm_samples_passed"]


def test_dry_run_plans_all_three_gpu_variants(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT),
        "--checkpoint", str(tmp_path / "checkpoint.safetensors"),
        "--tokenizer", str(tmp_path / "tokenizer.json"),
        "--prompt", "test prompt",
        "--output", str(tmp_path / "evidence"),
        "--dry-run",
    ])
    assert benchmark.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["format"] == "turbocider-ltx-gemma-encoder-sweep-plan-v2"
    assert plan["variants"] == list(benchmark.VARIANTS)
    assert plan["min_qualifying_warm_runs"] == 3
    assert plan["max_warm_cv"] == 0.25
    assert plan["planned"][0]["order"] == list(benchmark.VARIANTS)
    assert set(plan["planned"][0]["commands"]) == set(benchmark.VARIANTS)

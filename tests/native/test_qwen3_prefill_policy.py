import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "build/native/qwen3-prefill-plan-probe"


def run_plan(tmp_path: Path, shape: dict, *tokens: int) -> list[dict]:
    if not PROBE.is_file() or not os.access(PROBE, os.X_OK):
        pytest.skip("native Qwen3 prefill planner probe is not built")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 2, "shape": shape}))
    environment = os.environ.copy()
    environment.pop("TURBOCIDER_QWEN3_DISABLE_MLP_TAIL_PADDING", None)
    environment.pop("TURBOCIDER_QWEN3_MLP_MAX_PADDING_PERCENT", None)
    result = subprocess.run(
        [str(PROBE), str(manifest), *(str(token) for token in tokens)],
        text=True, capture_output=True, env=environment, check=True,
    )
    return json.loads(result.stdout)["cases"]


def test_unqualified_tail_padding_fails_closed(tmp_path: Path) -> None:
    [plan] = run_plan(tmp_path, {"buckets": [256]}, 129)
    assert plan == {
        "actual_tokens": 129,
        "selected_bucket": 256,
        "compute_tokens": 129,
        "padding_tokens": 0,
        "minimum_profitable_rows": 256,
        "use_hybrid": False,
        "fixed_shape": False,
        "reason": "below_min_profitable_rows",
    }


def test_measured_bucket_crossover_enables_padding(tmp_path: Path) -> None:
    [plan] = run_plan(
        tmp_path,
        {"buckets": [256], "minimum_profitable_rows": {"256": 129}},
        129,
    )
    assert plan["selected_bucket"] == 256
    assert plan["minimum_profitable_rows"] == 129
    assert plan["compute_tokens"] == 256
    assert plan["padding_tokens"] == 127
    assert plan["use_hybrid"] is True
    assert plan["reason"] == "fixed_bucket"


def test_exact_bucket_bypasses_tail_profitability_floor(tmp_path: Path) -> None:
    [plan] = run_plan(tmp_path, {"buckets": [64]}, 64)
    assert plan["use_hybrid"] is True
    assert plan["reason"] == "exact_bucket"


def test_unknown_profitability_bucket_is_rejected(tmp_path: Path) -> None:
    if not PROBE.is_file() or not os.access(PROBE, os.X_OK):
        pytest.skip("native Qwen3 prefill planner probe is not built")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 2,
        "shape": {
            "buckets": [256],
            "minimum_profitable_rows": {"512": 300},
        },
    }))
    result = subprocess.run(
        [str(PROBE), str(manifest), "256"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "unknown bucket" in result.stderr

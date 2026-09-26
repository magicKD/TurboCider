import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "build/native/ltx-gemma-ane-mlp-probe"


def run_plan(tmp_path: Path, shape: dict, rows: int) -> dict:
    if not PROBE.is_file() or not os.access(PROBE, os.X_OK):
        pytest.skip("native LTX Gemma ANE planner probe is not built")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "ltx-gemma-ane-mlp-v1",
        "shape": shape,
    }))
    result = subprocess.run(
        [str(PROBE), "--plan", str(manifest), str(rows)],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_legacy_manifest_keeps_exact_bucket_eligible(tmp_path: Path) -> None:
    plan = run_plan(tmp_path, {
        "rows": 64,
        "supported_rows": [64, 128, 256, 512, 1024],
    }, 128)
    assert plan == {
        "requested_rows": 128,
        "execution_rows": 128,
        "minimum_profitable_rows": 128,
    }


def test_legacy_manifest_defaults_padding_to_exact_only(tmp_path: Path) -> None:
    plan = run_plan(tmp_path, {
        "rows": 64,
        "supported_rows": [64, 128, 256, 512, 1024],
    }, 129)
    assert plan["execution_rows"] == 256
    assert plan["minimum_profitable_rows"] == 256


def test_measured_crossover_is_reported_for_selected_bucket(
    tmp_path: Path,
) -> None:
    plan = run_plan(tmp_path, {
        "rows": 64,
        "supported_rows": [64, 128, 256, 512, 1024],
        "minimum_profitable_rows": {"256": 129},
    }, 129)
    assert plan["execution_rows"] == 256
    assert plan["minimum_profitable_rows"] == 129


def test_unknown_profitability_bucket_is_rejected(tmp_path: Path) -> None:
    if not PROBE.is_file() or not os.access(PROBE, os.X_OK):
        pytest.skip("native LTX Gemma ANE planner probe is not built")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "ltx-gemma-ane-mlp-v1",
        "shape": {
            "rows": 64,
            "supported_rows": [64, 128, 256, 512, 1024],
            "minimum_profitable_rows": {"2048": 1024},
        },
    }))
    result = subprocess.run(
        [str(PROBE), "--plan", str(manifest), "1024"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "threshold is invalid" in result.stderr

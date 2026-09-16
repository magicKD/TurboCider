"""CPU-only validation of research timing/accuracy evidence admission."""
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[2] / "experimental/transformer/scripts/campaign.py"
SPEC = importlib.util.spec_from_file_location("transformer_campaign", PATH)
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


def records():
    return [dict(record_type="samples", gpu_ms=[2., 2.], hetero_ms=[1., 1.]),
            dict(benchmark="prefill_transformer_block", nrmse=.001, cosine=1.,
                 gpu_reference_nrmse=.001, gpu_reference_cosine=1.,
                 hetero_reference_nrmse=.002, hetero_reference_cosine=.9999,
                 gpu_wall_p50_ms=2., hetero_wall_p50_ms=1., speedup=2.,
                 up_ane_output_backing_rate=1., reference_enabled=True)]


def test_accepts_complete_evidence():
    assert CAMPAIGN.validate_records(records(), 2) == dict(
        finite_metrics=True, reference_gate_passed=True, output_backing_all_used=True)


@pytest.mark.parametrize("change", ["missing", "short", "nan", "boolean", "ratio", "reference"])
def test_rejects_invalid_evidence(change):
    rows = records()
    if change == "missing":
        rows.pop()
    elif change == "short":
        rows[0]["gpu_ms"].pop()
    elif change == "nan":
        rows[1]["nrmse"] = float("nan")
    elif change == "boolean":
        rows[0]["gpu_ms"][0] = True
    elif change == "ratio":
        rows[1]["speedup"] = 3.
    else:
        rows[1]["reference_enabled"] = False
    with pytest.raises(ValueError):
        CAMPAIGN.validate_records(rows, 2)


def test_preserves_completed_but_inaccurate_result():
    rows = records()
    rows[1]["hetero_reference_nrmse"] = .2
    rows[1]["up_ane_output_backing_rate"] = .5
    result = CAMPAIGN.validate_records(rows, 2)
    assert not result["reference_gate_passed"]
    assert not result["output_backing_all_used"]

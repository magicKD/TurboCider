from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def help_text(script: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/native" / script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_native_benchmark_selects_registered_model_id() -> None:
    text = help_text("benchmark_native.py")
    assert "--model-id" in text
    source = (ROOT / "tools/native/benchmark_native.py").read_text()
    assert "tc_engine_create_model" in source
    assert "r['model']=a.model_id" in source


def test_comparison_allows_gpu_only_without_ane_arguments() -> None:
    text = help_text("benchmark_comparison.py")
    assert "--model-id" in text
    assert "--manifest MANIFEST" in text
    assert "--bridge BRIDGE" in text
    source = (ROOT / "tools/native/benchmark_comparison.py").read_text()
    assert "required=True);p.add_argument('--bridge'" not in source
    assert "FLUX.2 Klein 9B GPU+ANE is not validated" in source

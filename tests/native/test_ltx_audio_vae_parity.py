import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "models/LTX-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors"
LATENT = (
    ROOT
    / "outputs/benchmark-ltx-turbocider-gpu-ane-run1.ltx-artifacts/audio_latent.bf16"
)
LTX_MAC = ROOT.parent / "ltx-mac"
NATIVE = ROOT / "build/native/ltx-audio-vae-decode"
ORACLE = ROOT / "tools/native/ltx_audio_vae_oracle.py"


def test_ltx_audio_vae_native_matches_python_oracle(tmp_path):
    missing = [
        path
        for path in (WEIGHTS, LATENT, LTX_MAC / "python", NATIVE, ORACLE)
        if not path.exists()
    ]
    if missing:
        pytest.skip("local LTX audio parity fixtures are unavailable")

    native_dir = tmp_path / "native"
    oracle_dir = tmp_path / "oracle"
    native_mel = tmp_path / "native-mel.bf16"
    command = [
        str(NATIVE),
        str(WEIGHTS),
        str(LATENT),
        "1",
        "101",
        str(native_mel),
        str(native_dir),
    ]
    native = subprocess.run(command, capture_output=True, text=True)
    if native.returncode != 0 and "Metal GPU is not available" in native.stderr:
        pytest.skip("Metal device is unavailable")
    assert native.returncode == 0, native.stderr

    oracle = subprocess.run(
        [
            sys.executable,
            str(ORACLE),
            "--ltx-mac",
            str(LTX_MAC),
            "--weights",
            str(WEIGHTS),
            "--latent",
            str(LATENT),
            "--batch",
            "1",
            "--tokens",
            "101",
            "--dump-dir",
            str(oracle_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert oracle.returncode == 0, oracle.stderr

    expected = [
        "00_input_blc.f32",
        "01_denormalized_blc.f32",
        "02_nhwc.f32",
        "03_conv_in.f32",
        "04_mid_block_1.f32",
        "05_mid_block_2.f32",
        "06_up_2.f32",
        "07_up_1.f32",
        "08_up_0.f32",
        "09_pre_out.f32",
        "10_conv_out_nhwc.f32",
        "11_output_bctf.f32",
    ]
    for name in expected:
        native_values = np.fromfile(native_dir / name, dtype=np.float32)
        oracle_values = np.fromfile(oracle_dir / name, dtype=np.float32)
        assert native_values.size > 0
        np.testing.assert_array_equal(native_values, oracle_values, err_msg=name)

    assert native_mel.stat().st_size == 1 * 2 * 401 * 64 * 2

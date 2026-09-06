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
AUDIO_TOOL = ROOT / "build/native/ltx-audio-vae-decode"
VOCODER_TOOL = ROOT / "build/native/ltx-vocoder-decode"
BWE_TOOL = ROOT / "build/native/ltx-bwe-decode"
AUDIO_ORACLE = ROOT / "tools/native/ltx_audio_vae_oracle.py"
VOCODER_ORACLE = ROOT / "tools/native/ltx_vocoder_oracle.py"
BWE_ORACLE = ROOT / "tools/native/ltx_bwe_oracle.py"


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 and "Metal GPU is not available" in result.stderr:
        pytest.skip("Metal device is unavailable")
    assert result.returncode == 0, result.stderr


def test_ltx_bwe_matches_python_oracle(tmp_path):
    required = [
        WEIGHTS,
        LATENT,
        LTX_MAC / "python",
        AUDIO_TOOL,
        VOCODER_TOOL,
        BWE_TOOL,
        AUDIO_ORACLE,
        VOCODER_ORACLE,
        BWE_ORACLE,
    ]
    if any(not path.exists() for path in required):
        pytest.skip("local LTX BWE parity fixtures are unavailable")

    mel = tmp_path / "mel.bf16"
    run(
        [
            str(AUDIO_TOOL),
            str(WEIGHTS),
            str(LATENT),
            "1",
            "101",
            str(mel),
        ]
    )
    waveform_16k = tmp_path / "waveform-16k.f32"
    run(
        [
            str(VOCODER_TOOL),
            str(WEIGHTS),
            str(mel),
            "1",
            "401",
            str(waveform_16k),
        ]
    )

    native_dir = tmp_path / "native"
    waveform_48k = tmp_path / "waveform-48k.f32"
    run(
        [
            str(BWE_TOOL),
            str(WEIGHTS),
            str(waveform_16k),
            "1",
            "64160",
            str(waveform_48k),
            str(native_dir),
        ]
    )
    oracle_dir = tmp_path / "oracle"
    run(
        [
            sys.executable,
            str(BWE_ORACLE),
            "--ltx-mac",
            str(LTX_MAC),
            "--weights",
            str(WEIGHTS),
            "--waveform",
            str(waveform_16k),
            "--batch",
            "1",
            "--samples",
            "64160",
            "--dump-dir",
            str(oracle_dir),
        ]
    )

    exact_stages = [
        "00_waveform_16k_btc.f32",
        "01_bwe_mel_bc.f32",
        "02_bwe_mel_btm.f32",
        "03_bwe_conv_pre.f32",
        "09_skip_bct.f32",
    ]
    for name in exact_stages:
        native = np.fromfile(native_dir / name, dtype=np.float32)
        oracle = np.fromfile(oracle_dir / name, dtype=np.float32)
        np.testing.assert_array_equal(native, oracle)

    generator_stages = [f"04_bwe_stage_{stage}.f32" for stage in range(5)]
    for name in generator_stages:
        native = np.fromfile(native_dir / name, dtype=np.float32)
        oracle = np.fromfile(oracle_dir / name, dtype=np.float32)
        np.testing.assert_allclose(native, oracle, rtol=0, atol=1.5e-5)

    native_residual = np.fromfile(
        native_dir / "08_residual_bct.f32", dtype=np.float32
    )
    oracle_residual = np.fromfile(
        oracle_dir / "08_residual_bct.f32", dtype=np.float32
    )
    np.testing.assert_allclose(native_residual, oracle_residual, rtol=0, atol=7e-8)

    native_waveform = np.fromfile(waveform_48k, dtype=np.float32)
    oracle_waveform = np.fromfile(
        oracle_dir / "10_waveform_48k_btc.f32", dtype=np.float32
    )
    np.testing.assert_allclose(native_waveform, oracle_waveform, rtol=0, atol=7e-8)

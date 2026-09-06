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
AUDIO_ORACLE = ROOT / "tools/native/ltx_audio_vae_oracle.py"
VOCODER_ORACLE = ROOT / "tools/native/ltx_vocoder_oracle.py"


def test_ltx_base_vocoder_matches_python_oracle(tmp_path):
    required = [
        WEIGHTS,
        LATENT,
        LTX_MAC / "python",
        AUDIO_TOOL,
        VOCODER_TOOL,
        AUDIO_ORACLE,
        VOCODER_ORACLE,
    ]
    if any(not path.exists() for path in required):
        pytest.skip("local LTX audio parity fixtures are unavailable")

    audio_dir = tmp_path / "audio-native"
    mel = tmp_path / "mel.bf16"
    native = subprocess.run(
        [
            str(AUDIO_TOOL),
            str(WEIGHTS),
            str(LATENT),
            "1",
            "101",
            str(mel),
            str(audio_dir),
        ],
        capture_output=True,
        text=True,
    )
    if native.returncode != 0 and "Metal GPU is not available" in native.stderr:
        pytest.skip("Metal device is unavailable")
    assert native.returncode == 0, native.stderr

    audio_oracle_dir = tmp_path / "audio-oracle"
    oracle = subprocess.run(
        [
            sys.executable,
            str(AUDIO_ORACLE),
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
            str(audio_oracle_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert oracle.returncode == 0, oracle.stderr

    waveform = tmp_path / "waveform.f32"
    native = subprocess.run(
        [
            str(VOCODER_TOOL),
            str(WEIGHTS),
            str(mel),
            "1",
            "401",
            str(waveform),
        ],
        capture_output=True,
        text=True,
    )
    assert native.returncode == 0, native.stderr
    oracle_dir = tmp_path / "vocoder-oracle"
    oracle = subprocess.run(
        [
            sys.executable,
            str(VOCODER_ORACLE),
            "--ltx-mac",
            str(LTX_MAC),
            "--weights",
            str(WEIGHTS),
            "--mel",
            str(mel),
            "--batch",
            "1",
            "--mel-time",
            "401",
            "--dump-dir",
            str(oracle_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert oracle.returncode == 0, oracle.stderr

    native_wave = np.fromfile(waveform, dtype=np.float32)
    oracle_wave = np.fromfile(oracle_dir / "09_waveform_16k.f32", dtype=np.float32)
    np.testing.assert_allclose(native_wave, oracle_wave, rtol=0, atol=4e-7)

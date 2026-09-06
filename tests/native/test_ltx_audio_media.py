import json
import shutil
import subprocess
import wave
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "models/LTX-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors"
LATENT = (
    ROOT
    / "outputs/benchmark-ltx-turbocider-gpu-ane-run1.ltx-artifacts/audio_latent.bf16"
)
VIDEO = ROOT / "outputs/benchmark-ltx-turbocider-gpu-ane-run1.mp4"
AUDIO_TOOL = ROOT / "build/native/ltx-audio-vae-decode"
VOCODER_TOOL = ROOT / "build/native/ltx-vocoder-decode"
BWE_TOOL = ROOT / "build/native/ltx-bwe-decode"
MUX_TOOL = ROOT / "build/native/ltx-audio-mux"


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 and "Metal GPU is not available" in result.stderr:
        pytest.skip("Metal device is unavailable")
    assert result.returncode == 0, result.stderr
    return result


def test_native_audio_wav_aac_mux_preserves_video_duration(tmp_path):
    required = [WEIGHTS, LATENT, VIDEO, AUDIO_TOOL, VOCODER_TOOL, BWE_TOOL, MUX_TOOL]
    if any(not path.exists() for path in required) or shutil.which("ffprobe") is None:
        pytest.skip("local LTX audio media fixtures are unavailable")

    mel = tmp_path / "mel.bf16"
    run([str(AUDIO_TOOL), str(WEIGHTS), str(LATENT), "1", "101", str(mel)])
    waveform16 = tmp_path / "waveform16.f32"
    run([str(VOCODER_TOOL), str(WEIGHTS), str(mel), "1", "401", str(waveform16)])
    waveform48 = tmp_path / "waveform48.f32"
    run([
        str(BWE_TOOL), str(WEIGHTS), str(waveform16), "1", "64160",
        str(waveform48),
    ])

    output = tmp_path / "muxed.mp4"
    wav = tmp_path / "audio.wav"
    mux = run([
        str(MUX_TOOL), str(VIDEO), str(waveform48), "192480", str(output), str(wav)
    ])
    payload = json.loads(mux.stdout)
    assert payload["sample_rate"] == 48000
    assert payload["channels"] == 2
    assert payload["muxed_samples"] == 194000
    assert payload["clipped_samples"] == 0

    with wave.open(str(wav), "rb") as stream:
        assert stream.getframerate() == 48000
        assert stream.getnchannels() == 2
        assert stream.getsampwidth() == 2
        assert stream.getnframes() == 192480

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,duration,start_time,sample_rate,channels",
            "-of", "json", str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")
    assert video["duration"] == audio["duration"]
    assert audio["sample_rate"] == "48000"
    assert audio["channels"] == 2
    assert not list(tmp_path.glob("muxed.*.mp4"))

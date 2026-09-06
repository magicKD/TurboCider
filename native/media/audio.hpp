#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace tc {

struct AudioMediaInfo {
    int sample_rate = 0;
    int channels = 0;
    int samples = 0;
    double duration_seconds = 0.0;
    int clipped_samples = 0;
};

/* Write finite normalized BTC floating-point PCM as a stereo/interleaved
 * 16-bit WAV.  The input is not retained after this call. */
AudioMediaInfo write_audio_pcm16_wav(
    const std::filesystem::path& output,
    const float* waveform,
    size_t samples,
    int sample_rate,
    int channels);

/* Mux normalized BTC floating-point PCM into an AAC track alongside the
 * existing H.264 video.  The final audio track is padded/truncated to the
 * exact video duration and the output is committed atomically. */
AudioMediaInfo mux_video_with_audio(
    const std::filesystem::path& video,
    const std::filesystem::path& output,
    const float* waveform,
    size_t samples,
    int sample_rate,
    int channels);

}  // namespace tc

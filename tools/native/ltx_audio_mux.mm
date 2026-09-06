#include "audio.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint32_t parse_samples(const char* value) {
    char* end = nullptr;
    errno = 0;
    unsigned long long result = std::strtoull(value, &end, 10);
    if (errno || !end || *end || result == 0 || result > UINT32_MAX) {
        throw std::runtime_error("invalid sample count");
    }
    return static_cast<uint32_t>(result);
}

std::vector<float> read_waveform(const std::filesystem::path& path,
                                 uint32_t samples) {
    const size_t elements = static_cast<size_t>(samples) * 2u;
    const size_t expected_bytes = elements * sizeof(float);
    std::error_code error;
    if (std::filesystem::file_size(path, error) != expected_bytes || error) {
        throw std::runtime_error("waveform byte count does not match stereo shape");
    }
    FILE* stream = std::fopen(path.c_str(), "rb");
    if (!stream) throw std::runtime_error(std::strerror(errno));
    std::vector<float> result(elements);
    const size_t read = std::fread(result.data(), sizeof(float), elements, stream);
    const int closed = std::fclose(stream);
    if (read != elements || closed != 0) {
        throw std::runtime_error("cannot read waveform input");
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr,
                     "usage: %s VIDEO WAVE48_F32 SAMPLES OUTPUT_MP4 [OUTPUT_WAV]\n",
                     argv[0]);
        return 2;
    }
    try {
        const uint32_t samples = parse_samples(argv[3]);
        auto waveform = read_waveform(argv[2], samples);
        tc::AudioMediaInfo wav_info;
        if (argc == 6) {
            wav_info = tc::write_audio_pcm16_wav(
                argv[5], waveform.data(), samples, 48000, 2);
        }
        auto mux_info = tc::mux_video_with_audio(
            argv[1], argv[4], waveform.data(), samples, 48000, 2);
        std::printf(
            "{\"sample_rate\":%d,\"channels\":%d,\"input_samples\":%u,"
            "\"muxed_samples\":%d,\"duration_seconds\":%.9f,"
            "\"clipped_samples\":%d,\"wav_written\":%s}\n",
            mux_info.sample_rate, mux_info.channels, samples,
            mux_info.samples, mux_info.duration_seconds,
            mux_info.clipped_samples, argc == 6 ? "true" : "false");
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "error: %s\n", exception.what());
        return 1;
    }
}

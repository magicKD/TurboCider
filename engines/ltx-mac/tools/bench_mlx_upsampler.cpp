#include "ltx_mlx_upsampler.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

namespace {

uint16_t f32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>(bits >> 16u);
}

uint32_t parse_u32(const char *text, const char *label) {
    char *end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno || !text[0] || !end || *end || value == 0 ||
        value > std::numeric_limits<uint32_t>::max()) {
        std::fprintf(stderr, "bench_mlx_upsampler: invalid %s: %s\n",
                     label, text);
        std::exit(2);
    }
    return static_cast<uint32_t>(value);
}

bool read_exact(const char *path, void *data, size_t bytes) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    stream.read(static_cast<char *>(data), static_cast<std::streamsize>(bytes));
    return stream.good() && stream.peek() == std::char_traits<char>::eof();
}

bool write_exact(const char *path, const void *data, size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) return false;
    stream.write(static_cast<const char *>(data),
                 static_cast<std::streamsize>(bytes));
    return stream.good();
}

void bcfhw_to_bfhwc(const std::vector<uint16_t> &source,
                    std::vector<uint16_t> &target,
                    uint32_t frames,
                    uint32_t height,
                    uint32_t width) {
    constexpr uint32_t channels = 128;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                for (uint32_t channel = 0; channel < channels; ++channel) {
                    const size_t source_index =
                        ((static_cast<size_t>(channel) * frames + frame) *
                         height + y) * width + x;
                    const size_t target_index =
                        ((static_cast<size_t>(frame) * height + y) * width + x) *
                        channels + channel;
                    target[target_index] = source[source_index];
                }
            }
        }
    }
}

void bfhwc_to_bcfhw(const std::vector<uint16_t> &source,
                    std::vector<uint16_t> &target,
                    uint32_t frames,
                    uint32_t height,
                    uint32_t width) {
    constexpr uint32_t channels = 128;
    for (uint32_t channel = 0; channel < channels; ++channel) {
        for (uint32_t frame = 0; frame < frames; ++frame) {
            for (uint32_t y = 0; y < height; ++y) {
                for (uint32_t x = 0; x < width; ++x) {
                    const size_t source_index =
                        ((static_cast<size_t>(frame) * height + y) * width + x) *
                        channels + channel;
                    const size_t target_index =
                        ((static_cast<size_t>(channel) * frames + frame) *
                         height + y) * width + x;
                    target[target_index] = source[source_index];
                }
            }
        }
    }
}

double seconds_since(const std::chrono::steady_clock::time_point &started) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 8) {
        std::fprintf(
            stderr,
            "Usage: %s CHECKPOINT [frames height width iterations "
            "[input_bcfhw.bf16 output_bcfhw.bf16]]\n",
            argv[0]);
        return 2;
    }
    const uint32_t frames = argc > 2 ? parse_u32(argv[2], "frames") : 13;
    const uint32_t height = argc > 3 ? parse_u32(argv[3], "height") : 7;
    const uint32_t width = argc > 4 ? parse_u32(argv[4], "width") : 11;
    const uint32_t iterations =
        argc > 5 ? parse_u32(argv[5], "iterations") : 3;
    const char *input_path = argc > 6 ? argv[6] : nullptr;
    const char *output_path = argc > 7 ? argv[7] : nullptr;
    if ((input_path == nullptr) != (output_path == nullptr)) {
        std::fputs("input and output paths must be paired\n", stderr);
        return 2;
    }

    const size_t input_elements =
        static_cast<size_t>(frames) * height * width * 128u;
    const size_t output_elements =
        static_cast<size_t>(frames) * (height * 2u) * (width * 2u) * 128u;
    std::vector<uint16_t> input_bcfhw(input_elements);
    std::vector<uint16_t> input_tokens(input_elements);
    std::vector<uint16_t> output_tokens(output_elements);
    if (input_path) {
        if (!read_exact(input_path, input_bcfhw.data(),
                        input_bcfhw.size() * sizeof(uint16_t))) {
            std::fprintf(stderr, "cannot read %s\n", input_path);
            return 1;
        }
    } else {
        for (size_t index = 0; index < input_bcfhw.size(); ++index) {
            const float value =
                static_cast<float>(static_cast<int>(index % 257u) - 128) /
                64.0f;
            input_bcfhw[index] = f32_to_bf16(value);
        }
    }
    bcfhw_to_bfhwc(input_bcfhw, input_tokens, frames, height, width);

    char error[2048] = {};
    auto started = std::chrono::steady_clock::now();
    ltx_mlx_upsampler *upsampler =
        ltx_mlx_upsampler_create(argv[1], error, sizeof(error));
    const double load_seconds = seconds_since(started);
    if (!upsampler) {
        std::fprintf(stderr, "bench_mlx_upsampler: %s\n", error);
        return 1;
    }

    started = std::chrono::steady_clock::now();
    int ok = ltx_mlx_upsampler_run_tokens_bf16(
        upsampler, output_tokens.data(), output_tokens.size(),
        input_tokens.data(), input_tokens.size(),
        1, frames, height, width, error, sizeof(error));
    const double first_seconds = seconds_since(started);
    started = std::chrono::steady_clock::now();
    for (uint32_t iteration = 0; ok && iteration < iterations; ++iteration) {
        ok = ltx_mlx_upsampler_run_tokens_bf16(
            upsampler, output_tokens.data(), output_tokens.size(),
            input_tokens.data(), input_tokens.size(),
            1, frames, height, width, error, sizeof(error));
    }
    const double warm_seconds = seconds_since(started);

    ltx_mlx_upsampler_info info = {};
    ltx_mlx_upsampler_get_info(upsampler, &info);
    if (ok && output_path) {
        std::vector<uint16_t> output_bcfhw(output_elements);
        bfhwc_to_bcfhw(
            output_tokens, output_bcfhw, frames, height * 2u, width * 2u);
        ok = write_exact(output_path, output_bcfhw.data(),
                         output_bcfhw.size() * sizeof(uint16_t));
        if (!ok) std::snprintf(error, sizeof(error), "cannot write %s", output_path);
    }
    if (ok) {
        std::printf(
            "MLX Upsampler: C=%u hidden=%u weights=%u (%.3f GiB)\n",
            info.input_channels, info.hidden_channels, info.weight_tensors,
            static_cast<double>(info.weight_bytes) /
                (1024.0 * 1024.0 * 1024.0));
        std::printf(
            "Shape: [1,128,%u,%u,%u] -> [1,128,%u,%u,%u]\n",
            frames, height, width, frames, height * 2u, width * 2u);
        std::printf(
            "Load: %.3f s | first: %.3f s | warm mean: %.3f s "
            "(%u iterations)\n",
            load_seconds, first_seconds,
            warm_seconds / static_cast<double>(iterations), iterations);
        if (output_path) std::printf("Output: %s\n", output_path);
    } else {
        std::fprintf(stderr, "bench_mlx_upsampler: %s\n", error);
    }

    ltx_mlx_upsampler_free(upsampler);
    return ok ? 0 : 1;
}

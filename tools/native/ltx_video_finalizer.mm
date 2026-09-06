#include "ltx_mlx_video_vae.h"
#include "ltx_video_convert.h"
#include "video.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double environment_seconds(const char* name) {
    const char* text = std::getenv(name);
    if (!text || !text[0]) return 0.0;
    char* end = nullptr;
    errno = 0;
    double value = std::strtod(text, &end);
    return errno == 0 && end && *end == '\0' && std::isfinite(value) &&
        value >= 0.0 ? value : 0.0;
}

uint32_t parse_u32(const char* text, const char* label) {
    char* end = nullptr;
    errno = 0;
    unsigned long value = std::strtoul(text, &end, 10);
    if (errno || !text[0] || !end || *end || value == 0 ||
        value > UINT32_MAX) {
        throw std::runtime_error(std::string("invalid ") + label);
    }
    return static_cast<uint32_t>(value);
}

std::vector<uint16_t> read_exact(const std::filesystem::path& path,
                                 size_t elements) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.good()) throw std::runtime_error("cannot open latent input");
    std::vector<uint16_t> values(elements);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(elements * sizeof(uint16_t)));
    if (!stream.good() || static_cast<size_t>(stream.gcount()) !=
            elements * sizeof(uint16_t))
        throw std::runtime_error("cannot read latent input");
    char trailing = 0;
    if (stream.get(trailing)) throw std::runtime_error("latent input has trailing bytes");
    return values;
}

std::string json_result(const std::filesystem::path& output,
                        const std::string& operation,
                        const std::string& execution,
                        const tc::VideoMediaInfo& info,
                        double pre_finalizer_seconds,
                        double vae_decode_seconds,
                        double rgb_convert_seconds,
                        double export_seconds,
                        double finalizer_wall_seconds) {
    @autoreleasepool {
        const char* cache_value = std::getenv(
            "TURBOCIDER_LTX_CONDITIONING_CACHE_HIT");
        const bool cache_hit = cache_value && std::strcmp(cache_value, "1") == 0;
        const char* mode_value = std::getenv("TURBOCIDER_LTX_CONDITIONING_MODE");
        const std::string conditioning_mode = mode_value && mode_value[0] ?
            mode_value : "unknown";
        NSDictionary* value = @{
            @"schema_version": @1,
            @"model": @"ltx-2.5-distilled",
            @"operation": @(operation.c_str()),
            @"output": @(output.c_str()),
            @"width": @(info.width),
            @"height": @(info.height),
            @"frames": @(info.frames),
            @"fps": @(info.fps),
            @"audio": @NO,
            @"execution": @(execution.c_str()),
            @"conditioning_cache_hit": @(cache_hit),
            @"conditioning_mode": @(conditioning_mode.c_str()),
            @"video_vae_isolation": @"exec",
            @"timings_seconds": @{
                @"pre_finalizer": @(pre_finalizer_seconds),
                @"video_vae_decode": @(vae_decode_seconds),
                @"rgb_convert": @(rgb_convert_seconds),
                @"export": @(export_seconds),
                @"finalizer_wall": @(finalizer_wall_seconds),
                @"request_wall_estimate": @(
                    pre_finalizer_seconds + finalizer_wall_seconds),
            },
            @"validation": @"native_gpu_video_only_exec_finalizer",
        };
        NSError* error = nil;
        NSData* data = [NSJSONSerialization dataWithJSONObject:value
                                                        options:0
                                                          error:&error];
        if (!data)
            throw std::runtime_error(error.localizedDescription.UTF8String);
        return std::string(static_cast<const char*>(data.bytes), data.length);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::fprintf(stderr,
            "usage: %s CHECKPOINT LATENT_BF16 LATENT_FRAMES LATENT_HEIGHT "
            "LATENT_WIDTH OUTPUT_MP4 OPERATION EXECUTION\n", argv[0]);
        return 2;
    }
    try {
        const auto finalizer_started = Clock::now();
        const double pre_finalizer_seconds = environment_seconds(
            "TURBOCIDER_LTX_PRE_FINALIZER_SECONDS");
        const uint32_t latent_frames = parse_u32(argv[3], "latent frames");
        const uint32_t latent_height = parse_u32(argv[4], "latent height");
        const uint32_t latent_width = parse_u32(argv[5], "latent width");
        const uint64_t output_frames64 = static_cast<uint64_t>(latent_frames) * 8u - 7u;
        const uint64_t output_height64 = static_cast<uint64_t>(latent_height) * 32u;
        const uint64_t output_width64 = static_cast<uint64_t>(latent_width) * 32u;
        if (output_frames64 > UINT32_MAX || output_height64 > UINT32_MAX ||
            output_width64 > UINT32_MAX)
            throw std::runtime_error("Video VAE output geometry overflows host size");
        const size_t latent_elements = static_cast<size_t>(latent_frames) *
            latent_height * latent_width * 128u;
        const size_t pixel_elements = static_cast<size_t>(3u) *
            static_cast<size_t>(output_frames64) * output_height64 * output_width64;
        auto latent = read_exact(argv[2], latent_elements);
        std::vector<uint16_t> pixels(pixel_elements);
        char error[2048] = {};
        ltx_mlx_video_vae* vae = ltx_mlx_video_vae_create(
            argv[1], error, sizeof(error));
        if (!vae) throw std::runtime_error(error[0] ? error : "cannot load Video VAE");
        const int ok = ltx_mlx_video_vae_decode_tokens_bf16(
            vae, pixels.data(), pixels.size(), latent.data(), latent.size(),
            1u, latent_frames, latent_height, latent_width,
            error, sizeof(error));
        ltx_mlx_video_vae_free(vae);
        if (!ok) throw std::runtime_error(error[0] ? error : "Video VAE decode failed");
        const auto vae_decoded = Clock::now();
        std::vector<uint8_t> rgb(static_cast<size_t>(output_frames64) *
                                  output_height64 * output_width64 * 3u);
        if (!ltx_video_bf16_planar_to_rgb24(
                rgb.data(), rgb.size(), pixels.data(), pixels.size(),
                static_cast<uint32_t>(output_frames64),
                static_cast<uint32_t>(output_height64),
                static_cast<uint32_t>(output_width64),
                error, sizeof(error)))
            throw std::runtime_error(error[0] ? error : "RGB conversion failed");
        const auto rgb_converted = Clock::now();
        const std::filesystem::path output = argv[6];
        tc::write_video_rgb24(output, rgb.data(),
                              static_cast<int>(output_frames64),
                              static_cast<int>(output_width64),
                              static_cast<int>(output_height64), 24);
        auto info = tc::probe_video(output);
        const auto exported = Clock::now();
        const double vae_decode_seconds = std::chrono::duration<double>(
            vae_decoded - finalizer_started).count();
        const double rgb_convert_seconds = std::chrono::duration<double>(
            rgb_converted - vae_decoded).count();
        const double export_seconds = std::chrono::duration<double>(
            exported - rgb_converted).count();
        const double finalizer_wall_seconds = std::chrono::duration<double>(
            exported - finalizer_started).count();
        std::puts(json_result(
            output, argv[7], argv[8], info, pre_finalizer_seconds,
            vae_decode_seconds, rgb_convert_seconds, export_seconds,
            finalizer_wall_seconds).c_str());
        std::fflush(stdout);
        std::error_code ignored;
        const std::filesystem::path latent_path = argv[2];
        std::filesystem::remove(latent_path, ignored);
        ignored.clear();
        std::filesystem::remove(latent_path.parent_path(), ignored);
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "ltx-video-finalizer: %s\n", exception.what());
        return 1;
    }
}

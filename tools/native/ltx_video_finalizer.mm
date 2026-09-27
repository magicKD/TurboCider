#include "ltx_mlx_video_vae.h"
#include "ltx_mlx_audio_vae.h"
#include "ltx_mlx_vocoder.h"
#include "ltx_mlx_bwe.h"
#include "ltx_video_convert.h"
#include "audio.hpp"
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
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
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

int environment_fd(const char* name) {
    const char* text = std::getenv(name);
    if (!text || !text[0]) return -1;
    char* end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < 0 ||
        value > std::numeric_limits<int>::max())
        throw std::runtime_error(std::string("invalid ") + name);
    return static_cast<int>(value);
}

NSDictionary* read_public_streaming_envelope() {
    const int fd = environment_fd("TURBOCIDER_LTX_PUBLIC_ENVELOPE_FD");
    if (fd < 0) return nil;
    struct stat metadata {};
    if (::fstat(fd, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
        metadata.st_size <= 0 || metadata.st_size > (8ll << 20)) {
        ::close(fd);
        throw std::runtime_error(
            "invalid LTX public streaming finalizer envelope file");
    }
    if (::lseek(fd, 0, SEEK_SET) < 0) {
        ::close(fd);
        throw std::runtime_error(
            "cannot seek LTX public streaming finalizer envelope");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(metadata.st_size));
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::read(fd, bytes.data() + offset,
                                     bytes.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            ::close(fd);
            throw std::runtime_error(
                "cannot read LTX public streaming finalizer envelope");
        }
        offset += static_cast<size_t>(count);
    }
    ::close(fd);
    NSData* data = [NSData dataWithBytes:bytes.data() length:bytes.size()];
    NSError* error = nil;
    id parsed = [NSJSONSerialization JSONObjectWithData:data
                                                options:0
                                                  error:&error];
    if (![parsed isKindOfClass:NSDictionary.class])
        throw std::runtime_error(error ? error.localizedDescription.UTF8String :
                                 "invalid LTX public streaming finalizer envelope JSON");
    NSDictionary* value = parsed;
    id schema = value[@"schema_version"];
    id public_result = value[@"public_streaming"];
    id verified = [public_result isKindOfClass:NSDictionary.class] ?
        public_result[@"actual_plan_verified"] : nil;
    if (![value[@"format"] isEqual:
            @"turbocider-ltx-public-finalizer-envelope-v1"] ||
        ![schema isKindOfClass:NSNumber.class] ||
        CFGetTypeID((__bridge CFTypeRef)schema) == CFBooleanGetTypeID() ||
        ![schema isEqualToNumber:@1] ||
        ![public_result isKindOfClass:NSDictionary.class] ||
        ![value[@"streaming_stages"] isKindOfClass:NSArray.class] ||
        ![value[@"streaming_boundaries"] isKindOfClass:NSArray.class] ||
        ![value[@"streaming_receipt"] isKindOfClass:NSDictionary.class] ||
        ![value[@"block_streaming"] isKindOfClass:NSDictionary.class] ||
        ![value[@"block_residency"] isKindOfClass:NSDictionary.class] ||
        ![verified isKindOfClass:NSNumber.class] ||
        CFGetTypeID((__bridge CFTypeRef)verified) != CFBooleanGetTypeID() ||
        ![verified boolValue])
        throw std::runtime_error(
            "incomplete LTX public streaming finalizer envelope");
    return value;
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

void remove_managed_staging_directory(
        const std::filesystem::path& latent_path) noexcept {
    const auto directory = latent_path.parent_path().lexically_normal();
    std::error_code root_error;
    const auto temporary_root =
        std::filesystem::temp_directory_path(root_error).lexically_normal();
    if (root_error) return;
    std::error_code equivalent_error;
    const bool same_root = std::filesystem::equivalent(
        directory.parent_path(), temporary_root, equivalent_error);
    const auto name = directory.filename().string();
    if (latent_path.filename() != "video_latent.bf16" ||
        equivalent_error || !same_root ||
        !name.starts_with("turbocider-ltx-exec-finalizer-"))
        return;
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

std::string json_result(const std::filesystem::path& output,
                        const std::string& operation,
                        const std::string& execution,
                        const tc::VideoMediaInfo& info,
                        bool audio,
                        const tc::AudioMediaInfo& audio_info,
                        double pre_finalizer_seconds,
                        double video_input_read_seconds,
                        double video_weight_load_seconds,
                        double video_decode_compute_seconds,
                        double vae_decode_seconds,
                        double rgb_convert_seconds,
                        double export_seconds,
                        double audio_decode_seconds,
                        double audio_mux_seconds,
                        double finalizer_wall_seconds,
                        NSDictionary* public_streaming_envelope) {
    @autoreleasepool {
        const char* cache_value = std::getenv(
            "TURBOCIDER_LTX_CONDITIONING_CACHE_HIT");
        const bool cache_hit = cache_value && std::strcmp(cache_value, "1") == 0;
        const char* mode_value = std::getenv("TURBOCIDER_LTX_CONDITIONING_MODE");
        const std::string conditioning_mode = mode_value && mode_value[0] ?
            mode_value : "unknown";
        NSMutableDictionary* value = [@{
            @"schema_version": @1,
            @"model": @"ltx-2.5-distilled",
            @"operation": @(operation.c_str()),
            @"output": @(output.c_str()),
            @"width": @(info.width),
            @"height": @(info.height),
            @"frames": @(info.frames),
            @"fps": @(info.fps),
            @"audio": @(audio),
            @"audio_channels": audio ? @(audio_info.channels) : (id)[NSNull null],
            @"audio_sample_rate": audio ? @(audio_info.sample_rate) : (id)[NSNull null],
            @"audio_samples": audio ? @(audio_info.samples) : (id)[NSNull null],
            @"audio_duration_seconds": audio ? @(audio_info.duration_seconds) : (id)[NSNull null],
            @"audio_clipped_samples": audio ? @(audio_info.clipped_samples) : (id)[NSNull null],
            @"execution": @(execution.c_str()),
            @"conditioning_cache_hit": @(cache_hit),
            @"conditioning_mode": @(conditioning_mode.c_str()),
            @"video_vae_isolation": @"exec",
            @"timings_seconds": @{
                @"pre_finalizer": @(pre_finalizer_seconds),
                @"video_input_read": @(video_input_read_seconds),
                @"video_vae_weight_load": @(video_weight_load_seconds),
                @"video_vae_compute": @(video_decode_compute_seconds),
                @"video_vae_decode": @(vae_decode_seconds),
                @"rgb_convert": @(rgb_convert_seconds),
                @"export": @(export_seconds),
                @"audio_decode": @(audio_decode_seconds),
                @"audio_mux": @(audio_mux_seconds),
                @"finalizer_wall": @(finalizer_wall_seconds),
                @"request_wall_estimate": @(
                    pre_finalizer_seconds + finalizer_wall_seconds),
            },
            @"validation": audio ?
                @"native_gpu_audio_video_exec_finalizer" :
                @"native_gpu_video_only_exec_finalizer",
        } mutableCopy];
        if (public_streaming_envelope) {
            for (NSString* key in @[
                    @"public_streaming", @"block_residency",
                    @"block_streaming", @"streaming_stages",
                    @"streaming_boundaries", @"streaming_receipt"])
                value[key] = public_streaming_envelope[key];
            value[@"validation"] =
                @"native_gpu_video_only_exec_finalizer_public_verified";
        }
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
    if (argc != 9 && argc != 12) {
        std::fprintf(stderr,
            "usage: %s CHECKPOINT LATENT_BF16 LATENT_FRAMES LATENT_HEIGHT "
            "LATENT_WIDTH OUTPUT_MP4 OPERATION EXECUTION "
            "[AUDIO_CHECKPOINT AUDIO_LATENT_BF16 AUDIO_TOKENS]\n", argv[0]);
        return 2;
    }
    std::filesystem::path transient_video;
    try {
        const auto finalizer_started = Clock::now();
        const bool audio = argc == 12;
        NSDictionary* public_streaming_envelope =
            read_public_streaming_envelope();
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
        const auto video_input_read = Clock::now();
        std::vector<uint16_t> pixels(pixel_elements);
        char error[2048] = {};
        const int video_vae_fd = environment_fd(
            "TURBOCIDER_LTX_VIDEO_VAE_CHECKPOINT_FD");
        ltx_mlx_video_vae* vae = video_vae_fd >= 0 ?
            ltx_mlx_video_vae_create_fd(
                video_vae_fd, argv[1], error, sizeof(error)) :
            ltx_mlx_video_vae_create(argv[1], error, sizeof(error));
        if (!vae) throw std::runtime_error(error[0] ? error : "cannot load Video VAE");
        const auto video_weight_loaded = Clock::now();
        const int ok = ltx_mlx_video_vae_decode_tokens_bf16(
            vae, pixels.data(), pixels.size(), latent.data(), latent.size(),
            1u, latent_frames, latent_height, latent_width,
            error, sizeof(error));
        ltx_mlx_video_vae_free(vae);
        ltx_mlx_video_vae_clear_cache();
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
        std::filesystem::path video_only = output;
        if (audio) {
            video_only += "." + std::string(NSUUID.UUID.UUIDString.UTF8String) +
                ".mp4";
            transient_video = video_only;
        }
        tc::write_video_rgb24(video_only, rgb.data(),
                              static_cast<int>(output_frames64),
                              static_cast<int>(output_width64),
                              static_cast<int>(output_height64), 24);
        const auto exported = Clock::now();
        tc::AudioMediaInfo audio_info{};
        auto audio_decoded = exported;
        auto audio_muxed = exported;
        if (audio) {
            const uint32_t audio_tokens = parse_u32(argv[11], "audio tokens");
            if (audio_tokens > UINT32_MAX / 4u)
                throw std::runtime_error("audio token geometry overflows host size");
            const size_t audio_elements =
                static_cast<size_t>(audio_tokens) * 128u;
            auto audio_latent = read_exact(argv[10], audio_elements);
            const uint32_t mel_time = audio_tokens * 4u - 3u;
            const size_t mel_elements =
                static_cast<size_t>(2u) * mel_time * 64u;
            const size_t waveform_samples =
                static_cast<size_t>(mel_time) * 160u;
            std::vector<uint16_t> mel(mel_elements);
            std::vector<float> waveform16(waveform_samples * 2u);
            std::vector<float> waveform48(waveform_samples * 3u * 2u);

            ltx_mlx_audio_vae* audio_vae = ltx_mlx_audio_vae_create(
                argv[9], error, sizeof(error));
            if (!audio_vae)
                throw std::runtime_error(error[0] ? error :
                                         "cannot load Audio VAE");
            int audio_ok = ltx_mlx_audio_vae_decode_bf16(
                audio_vae, mel.data(), mel.size(), audio_latent.data(),
                audio_latent.size(), 1u, audio_tokens, error, sizeof(error));
            ltx_mlx_audio_vae_free(audio_vae);
            ltx_mlx_audio_vae_clear_cache();
            if (!audio_ok)
                throw std::runtime_error(error[0] ? error :
                                         "Audio VAE decode failed");

            ltx_mlx_vocoder* vocoder = ltx_mlx_vocoder_create_base(
                argv[9], error, sizeof(error));
            if (!vocoder)
                throw std::runtime_error(error[0] ? error :
                                         "cannot load base vocoder");
            audio_ok = ltx_mlx_vocoder_decode_base_bf16(
                vocoder, waveform16.data(), waveform16.size(), mel.data(),
                mel.size(), 1u, mel_time, error, sizeof(error));
            ltx_mlx_vocoder_free(vocoder);
            ltx_mlx_vocoder_clear_cache();
            if (!audio_ok)
                throw std::runtime_error(error[0] ? error :
                                         "base vocoder decode failed");

            ltx_mlx_bwe* bwe = ltx_mlx_bwe_create(
                argv[9], error, sizeof(error));
            if (!bwe)
                throw std::runtime_error(error[0] ? error :
                                         "cannot load BWE vocoder");
            audio_ok = ltx_mlx_bwe_extend_f32(
                bwe, waveform48.data(), waveform48.size(), waveform16.data(),
                waveform16.size(), 1u,
                static_cast<uint32_t>(waveform_samples),
                error, sizeof(error));
            ltx_mlx_bwe_free(bwe);
            ltx_mlx_bwe_clear_cache();
            if (!audio_ok)
                throw std::runtime_error(error[0] ? error :
                                         "BWE vocoder decode failed");
            audio_decoded = Clock::now();
            audio_info = tc::mux_video_with_audio(
                video_only, output, waveform48.data(), waveform48.size() / 2u,
                48000, 2);
            audio_muxed = Clock::now();
            std::error_code ignored;
            std::filesystem::remove(transient_video, ignored);
            transient_video.clear();
        }
        const tc::VideoMediaInfo info{
            static_cast<int>(output_width64),
            static_cast<int>(output_height64),
            static_cast<int>(output_frames64), 24};
        const double vae_decode_seconds = std::chrono::duration<double>(
            vae_decoded - finalizer_started).count();
        const double video_input_read_seconds = std::chrono::duration<double>(
            video_input_read - finalizer_started).count();
        const double video_weight_load_seconds = std::chrono::duration<double>(
            video_weight_loaded - video_input_read).count();
        const double video_decode_compute_seconds = std::chrono::duration<double>(
            vae_decoded - video_weight_loaded).count();
        const double rgb_convert_seconds = std::chrono::duration<double>(
            rgb_converted - vae_decoded).count();
        const double export_seconds = std::chrono::duration<double>(
            exported - rgb_converted).count();
        const double audio_decode_seconds = std::chrono::duration<double>(
            audio_decoded - exported).count();
        const double audio_mux_seconds = std::chrono::duration<double>(
            audio_muxed - audio_decoded).count();
        const double finalizer_wall_seconds = std::chrono::duration<double>(
            audio_muxed - finalizer_started).count();
        std::puts(json_result(
            output, argv[7], argv[8], info, audio, audio_info,
            pre_finalizer_seconds, video_input_read_seconds,
            video_weight_load_seconds, video_decode_compute_seconds,
            vae_decode_seconds, rgb_convert_seconds,
            export_seconds, audio_decode_seconds, audio_mux_seconds,
            finalizer_wall_seconds, public_streaming_envelope).c_str());
        std::fflush(stdout);
        const std::filesystem::path latent_path = argv[2];
        remove_managed_staging_directory(latent_path);
        return 0;
    } catch (const std::exception& exception) {
        if (!transient_video.empty()) {
            std::error_code ignored;
            std::filesystem::remove(transient_video, ignored);
        }
        remove_managed_staging_directory(argv[2]);
        std::fprintf(stderr, "ltx-video-finalizer: %s\n", exception.what());
        return 1;
    }
}

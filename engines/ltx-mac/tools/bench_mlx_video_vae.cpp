#include "ltx_mlx_video_vae.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unistd.h>
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
        std::fprintf(stderr, "bench_mlx_video_vae: invalid %s: %s\n",
                     label, text);
        std::exit(2);
    }
    return static_cast<uint32_t>(value);
}

uint32_t parse_count(const char *text, const char *label) {
    char *end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno || !text[0] || !end || *end ||
        value > std::numeric_limits<uint32_t>::max()) {
        std::fprintf(stderr, "bench_mlx_video_vae: invalid %s: %s\n",
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

double seconds_since(const std::chrono::steady_clock::time_point &started) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
}

bool write_generation_metadata(const char *directory,
                               const char *seed,
                               const char *text_rows,
                               uint32_t latent_frames,
                               uint32_t latent_height,
                               uint32_t latent_width,
                               double decode_seconds) {
    if (!directory || !directory[0] || !seed || !seed[0] ||
        !text_rows || !text_rows[0]) return false;
    const std::filesystem::path path =
        std::filesystem::path(directory) / "generation.json";
    const uint32_t output_frames = latent_frames * 8u - 7u;
    const uint32_t output_height = latent_height * 32u;
    const uint32_t output_width = latent_width * 32u;
    const char *requested_width_text =
        std::getenv("LTX_MLX_FINALIZE_REQUESTED_WIDTH");
    const char *requested_height_text =
        std::getenv("LTX_MLX_FINALIZE_REQUESTED_HEIGHT");
    const uint32_t requested_width =
        requested_width_text && requested_width_text[0] ?
        parse_u32(requested_width_text, "requested width") : output_width;
    const uint32_t requested_height =
        requested_height_text && requested_height_text[0] ?
        parse_u32(requested_height_text, "requested height") : output_height;
    const uint64_t video_rows =
        static_cast<uint64_t>(latent_frames) * latent_height * latent_width;
    std::ofstream metadata(path);
    if (!metadata) return false;
    metadata
        << "{\n"
        << "  \"format\": \"ltx-mac-generation-v2\",\n"
        << "  \"dtype\": \"bfloat16\",\n"
        << "  \"seed\": " << seed << ",\n"
        << "  \"text_rows\": " << text_rows << ",\n"
        << "  \"requested_video\": {\"width\": " << requested_width
        << ", \"height\": " << requested_height
        << ", \"frames\": " << output_frames << ", \"fps\": 24},\n"
        << "  \"decoded_video\": {\"width\": " << output_width
        << ", \"height\": " << output_height
        << ", \"frames\": " << output_frames << ", \"fps\": 24},\n"
        << "  \"video\": {\"file\": \"video_latent.bf16\", "
           "\"layout\": \"BFHWC-token-major\", "
           "\"shape\": [1, " << latent_frames << ", " << latent_height
        << ", " << latent_width << ", 128], \"rows\": " << video_rows
        << "},\n"
        << "  \"audio\": {\"file\": \"audio_latent.bf16\", "
           "\"layout\": \"BLC-token-major\", "
           "\"shape\": [1, 101, 128]},\n"
        << "  \"native_video\": {\"file\": \"video_pixels.bf16\", "
           "\"layout\": \"BCFHW\", \"dtype\": \"bfloat16\", "
           "\"shape\": [1, 3, " << output_frames << ", " << output_height
        << ", " << output_width << "], "
           "\"decode_seconds\": " << decode_seconds << "}\n"
        << "}\n";
    return metadata.good();
}

int run_resident_worker(int argc, char **argv) {
    if (argc != 8) {
        std::fprintf(
            stderr,
            "Usage: %s --resident-worker CHECKPOINT frames height width "
            "token_input.bf16 output.bf16\n",
            argv[0]);
        return 2;
    }
    const char *checkpoint = argv[2];
    const uint32_t frames = parse_u32(argv[3], "frames");
    const uint32_t height = parse_u32(argv[4], "height");
    const uint32_t width = parse_u32(argv[5], "width");
    const char *input_path = argv[6];
    const char *output_path = argv[7];
    const uint32_t output_frames = frames * 8u - 7u;
    const uint32_t output_height = height * 32u;
    const uint32_t output_width = width * 32u;
    const size_t input_elements =
        static_cast<size_t>(frames) * height * width * 128u;
    const size_t output_elements =
        static_cast<size_t>(3) * output_frames * output_height * output_width;
    std::vector<uint16_t> input(input_elements, 0u);
    std::vector<uint16_t> output(output_elements);
    char error[2048] = {};

    auto started = std::chrono::steady_clock::now();
    ltx_mlx_video_vae *vae =
        ltx_mlx_video_vae_create(checkpoint, error, sizeof(error));
    const double load_seconds = seconds_since(started);
    if (!vae) {
        std::fprintf(stderr, "bench_mlx_video_vae worker: %s\n", error);
        return 1;
    }

    started = std::chrono::steady_clock::now();
    int ok = ltx_mlx_video_vae_decode_tokens_bf16(
        vae, output.data(), output.size(), input.data(), input.size(),
        1u, frames, height, width, error, sizeof(error));
    const double warmup_seconds = seconds_since(started);
    if (!ok) {
        std::fprintf(stderr, "bench_mlx_video_vae worker warmup: %s\n", error);
        ltx_mlx_video_vae_free(vae);
        return 1;
    }
    ltx_mlx_video_vae_clear_cache();
    std::printf(
        "mlx_vae_worker_ready pid=%d load_seconds=%.6f "
        "warmup_seconds=%.6f shape=1x%ux%ux%ux128\n",
        static_cast<int>(getpid()), load_seconds, warmup_seconds,
        frames, height, width);
    std::fflush(nullptr);

    if (std::raise(SIGSTOP) != 0) {
        std::fprintf(stderr, "bench_mlx_video_vae worker: SIGSTOP failed\n");
        ltx_mlx_video_vae_free(vae);
        return 1;
    }

    if (!read_exact(input_path, input.data(),
                    input.size() * sizeof(uint16_t))) {
        std::fprintf(stderr, "bench_mlx_video_vae worker: cannot read %s\n",
                     input_path);
        ltx_mlx_video_vae_free(vae);
        return 1;
    }
    started = std::chrono::steady_clock::now();
    ok = ltx_mlx_video_vae_decode_tokens_bf16(
        vae, output.data(), output.size(), input.data(), input.size(),
        1u, frames, height, width, error, sizeof(error));
    const double decode_seconds = seconds_since(started);
    if (ok) {
        ok = write_exact(output_path, output.data(),
                         output.size() * sizeof(uint16_t));
        if (!ok)
            std::snprintf(error, sizeof(error), "cannot write %s", output_path);
    }
    if (ok) {
        std::printf(
            "video_decode_seconds=%.6f decoder_load_seconds=%.6f "
            "worker_warmup_seconds=%.6f shape=1x3x%ux%ux%u dtype=bf16 "
            "media_backend=mlx-cpp isolation=resident-worker\n",
            decode_seconds, load_seconds, warmup_seconds,
            output_frames, output_height, output_width);
        std::fflush(nullptr);
    } else {
        std::fprintf(stderr, "bench_mlx_video_vae worker decode: %s\n", error);
    }
    ltx_mlx_video_vae_free(vae);
    return ok ? 0 : 1;
}

int run_encode(int argc, char **argv) {
    if (argc != 8) {
        std::fprintf(
            stderr,
            "Usage: %s --encode CHECKPOINT frames height width "
            "pixels.bf16 latent.bf16\n",
            argv[0]);
        return 2;
    }
    const char *checkpoint = argv[2];
    const uint32_t frames = parse_u32(argv[3], "frames");
    const uint32_t height = parse_u32(argv[4], "height");
    const uint32_t width = parse_u32(argv[5], "width");
    const char *input_path = argv[6];
    const char *output_path = argv[7];
    if (height % 32u || width % 32u) {
        std::fputs(
            "bench_mlx_video_vae: encode height and width must be divisible "
            "by 32\n", stderr);
        return 2;
    }
    const uint32_t latent_frames = (frames + 7u) / 8u;
    const uint32_t latent_height = height / 32u;
    const uint32_t latent_width = width / 32u;
    const size_t input_elements =
        static_cast<size_t>(3) * frames * height * width;
    const size_t output_elements =
        static_cast<size_t>(latent_frames) * latent_height *
        latent_width * 128u;
    std::vector<uint16_t> input(input_elements);
    std::vector<uint16_t> output(output_elements);
    if (!read_exact(input_path, input.data(), input.size() * sizeof(uint16_t))) {
        std::fprintf(stderr, "cannot read %s\n", input_path);
        return 1;
    }

    char error[2048] = {};
    auto started = std::chrono::steady_clock::now();
    ltx_mlx_video_vae *vae =
        ltx_mlx_video_vae_create_encoder(checkpoint, error, sizeof(error));
    const double load_seconds = seconds_since(started);
    if (!vae) {
        std::fprintf(stderr, "bench_mlx_video_vae encode: %s\n", error);
        return 1;
    }
    started = std::chrono::steady_clock::now();
    int ok = ltx_mlx_video_vae_encode_pixels_bf16(
        vae, output.data(), output.size(), input.data(), input.size(),
        1u, frames, height, width, error, sizeof(error));
    const double encode_seconds = seconds_since(started);
    if (ok) {
        ok = write_exact(output_path, output.data(),
                         output.size() * sizeof(uint16_t));
        if (!ok)
            std::snprintf(error, sizeof(error), "cannot write %s", output_path);
    }
    ltx_mlx_video_vae_info info = {};
    if (ok) ok = ltx_mlx_video_vae_get_info(vae, &info);
    if (ok) {
        std::printf(
            "video_encode_seconds=%.6f encoder_load_seconds=%.6f "
            "input_shape=1x3x%ux%ux%u "
            "latent_shape=1x%ux%ux%ux128 dtype=bf16 "
            "weight_tensors=%u weight_bytes=%llu\n",
            encode_seconds, load_seconds, frames, height, width,
            latent_frames, latent_height, latent_width,
            info.weight_tensors,
            static_cast<unsigned long long>(info.weight_bytes));
    } else {
        std::fprintf(stderr, "bench_mlx_video_vae encode: %s\n", error);
    }
    ltx_mlx_video_vae_free(vae);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--resident-worker") == 0)
        return run_resident_worker(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "--encode") == 0)
        return run_encode(argc, argv);
    if (argc < 2 || argc > 8) {
        std::fprintf(
            stderr,
            "Usage: %s CHECKPOINT [frames height width iterations "
            "[token_input.bf16 output.bf16]]\n",
            argv[0]);
        return 2;
    }
    const uint32_t frames = argc > 2 ? parse_u32(argv[2], "frames") : 2;
    const uint32_t height = argc > 3 ? parse_u32(argv[3], "height") : 2;
    const uint32_t width = argc > 4 ? parse_u32(argv[4], "width") : 3;
    const uint32_t iterations =
        argc > 5 ? parse_count(argv[5], "iterations") : 1;
    const char *input_path = argc > 6 ? argv[6] : nullptr;
    const char *output_path = argc > 7 ? argv[7] : nullptr;
    if ((input_path == nullptr) != (output_path == nullptr)) {
        std::fputs("input and output paths must be paired\n", stderr);
        return 2;
    }

    const uint32_t output_frames = frames * 8u - 7u;
    const uint32_t output_height = height * 32u;
    const uint32_t output_width = width * 32u;
    const size_t input_elements =
        static_cast<size_t>(frames) * height * width * 128u;
    const size_t output_elements =
        static_cast<size_t>(3) * output_frames * output_height * output_width;
    std::vector<uint16_t> input(input_elements);
    std::vector<uint16_t> output(output_elements);
    if (input_path) {
        if (!read_exact(input_path, input.data(),
                        input.size() * sizeof(uint16_t))) {
            std::fprintf(stderr, "cannot read %s\n", input_path);
            return 1;
        }
    } else {
        for (size_t index = 0; index < input.size(); ++index) {
            const float value =
                static_cast<float>(static_cast<int>(index % 257u) - 128) /
                256.0f;
            input[index] = f32_to_bf16(value);
        }
    }

    const auto process_started = std::chrono::steady_clock::now();
    char error[2048] = {};
    auto started = std::chrono::steady_clock::now();
    ltx_mlx_video_vae *vae =
        ltx_mlx_video_vae_create(argv[1], error, sizeof(error));
    const double load_seconds = seconds_since(started);
    if (!vae) {
        std::fprintf(stderr, "bench_mlx_video_vae: %s\n", error);
        return 1;
    }

    started = std::chrono::steady_clock::now();
    int ok = ltx_mlx_video_vae_decode_tokens_bf16(
        vae, output.data(), output.size(), input.data(), input.size(),
        1, frames, height, width, error, sizeof(error));
    const double first_seconds = seconds_since(started);
    started = std::chrono::steady_clock::now();
    for (uint32_t iteration = 0; ok && iteration < iterations; ++iteration) {
        ok = ltx_mlx_video_vae_decode_tokens_bf16(
            vae, output.data(), output.size(), input.data(), input.size(),
            1, frames, height, width, error, sizeof(error));
    }
    const double warm_seconds = seconds_since(started);

    ltx_mlx_video_vae_info info = {};
    ltx_mlx_video_vae_get_info(vae, &info);
    if (ok && output_path) {
        ok = write_exact(output_path, output.data(),
                         output.size() * sizeof(uint16_t));
        if (!ok) std::snprintf(error, sizeof(error), "cannot write %s", output_path);
    }
    const char *finalize_directory = std::getenv("LTX_MLX_FINALIZE_DIR");
    const char *finalize_seed = std::getenv("LTX_MLX_FINALIZE_SEED");
    const char *finalize_text_rows =
        std::getenv("LTX_MLX_FINALIZE_TEXT_ROWS");
    const char *predecode_text =
        std::getenv("LTX_MLX_FINALIZE_PREDECODE_SECONDS");
    const char *boundary_text =
        std::getenv("LTX_MLX_FINALIZE_BOUNDARY_SECONDS");
    const bool finalize = finalize_directory && finalize_directory[0];
    if (ok && finalize) {
        ok = write_generation_metadata(
            finalize_directory, finalize_seed, finalize_text_rows,
            frames, height, width,
            first_seconds);
        if (!ok) std::snprintf(error, sizeof(error),
                               "cannot write generation metadata");
    }
    if (ok) {
        std::printf(
            "MLX Video VAE: weights=%u (%.3f GiB)\n",
            info.weight_tensors,
            static_cast<double>(info.weight_bytes) /
                (1024.0 * 1024.0 * 1024.0));
        std::printf(
            "Shape: [1,%u,%u,%u,128] -> [1,3,%u,%u,%u]\n",
            frames, height, width,
            output_frames, output_height, output_width);
        if (iterations > 0) {
            std::printf(
                "Load: %.3f s | first: %.3f s | warm mean: %.3f s "
                "(%u iterations)\n",
                load_seconds, first_seconds,
                warm_seconds / static_cast<double>(iterations), iterations);
        } else {
            std::printf("Load: %.3f s | first: %.3f s\n",
                        load_seconds, first_seconds);
        }
        if (output_path) std::printf("Output: %s\n", output_path);
        if (finalize) {
            const double helper_seconds = seconds_since(process_started);
            const double predecode_seconds = predecode_text ?
                std::strtod(predecode_text, nullptr) : 0.0;
            const double boundary_seconds = boundary_text ?
                std::strtod(boundary_text, nullptr) : 0.0;
            std::printf(
                "video_decode_seconds=%.6f decoder_load_seconds=%.6f "
                "finalize_seconds=%.6f "
                "shape=1x3x%ux%ux%u dtype=bf16 "
                "media_backend=mlx-cpp isolation=exec\n",
                first_seconds, load_seconds, helper_seconds,
                output_frames, output_height, output_width);
            std::printf(
                "two_stage_elapsed_seconds=%.6f blocks=48 "
                "boundary_seconds=%.6f noise=seeded-pcg32-gaussian "
                "quality=mlx-conv-vae-decoded\n",
                predecode_seconds + helper_seconds, boundary_seconds);
            std::printf(
                "generation_artifacts=%s format=ltx-mac-generation-v2\n",
                finalize_directory);
        }
    } else {
        std::fprintf(stderr, "bench_mlx_video_vae: %s\n", error);
    }

    ltx_mlx_video_vae_free(vae);
    return ok ? 0 : 1;
}

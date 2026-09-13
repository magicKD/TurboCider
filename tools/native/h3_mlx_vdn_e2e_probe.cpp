#include "../../native/models/h3_mlx/pipeline.hpp"

#include <cstdlib>
#include <iostream>
#include <unordered_map>

namespace {

int parse_dimension(const char *name, const char *value) {
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    tc::require(end != value && *end == '\0' && parsed > 0 &&
                    parsed <= 1'000'000,
                std::string("invalid ") + name);
    return static_cast<int>(parsed);
}

} // namespace

int main(int argc, char **argv) {
    try {
        tc::require(argc == 6,
                    "usage: h3-mlx-vdn-e2e-probe CHECKPOINT OUTPUT WIDTH HEIGHT FRAMES");
        const auto root = std::filesystem::absolute(argv[1]);
        const auto output = std::filesystem::absolute(argv[2]);
        const int width = parse_dimension("width", argv[3]);
        const int height = parse_dimension("height", argv[4]);
        const int frames = parse_dimension("frames", argv[5]);
        tc::require(width % 32 == 0 && height % 32 == 0,
                    "VDN probe dimensions must be multiples of 32");

        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::h3_mlx::Pipeline pipeline;
        pipeline.load(root, event, cancelled);
        tc::require(pipeline.checkpoint().identity().vdn.enabled,
                    "VDN probe requires a minimax-h3-vdn checkpoint");

        constexpr int text_tokens = 8;
        auto text = tc::mx::zeros({text_tokens, 5120}, tc::mx::float32);
        std::vector<int32_t> tags(text_tokens, tc::h3_mlx::text_tag);
        tc::h3_mlx::DenoiseOptions options;
        options.width = width;
        options.height = height;
        options.frames = frames;
        options.fps = 24;
        options.steps = pipeline.checkpoint().identity().steps;
        options.seed = 42;
        options.affine_dq_gemm_min_rows = 768;
        options.capture_first_velocity = true;
        auto result = pipeline.denoise(text, tags, options, event, cancelled);
        tc::mx::eval(result.video_rows, result.audio_rows,
                     *result.first_video_velocity, *result.first_audio_velocity);

        std::filesystem::create_directories(output.parent_path());
        tc::mx::save_safetensors(output.string(), {
            {"video_rows", tc::mx::astype(result.video_rows, tc::mx::float32)},
            {"audio_rows", tc::mx::astype(result.audio_rows, tc::mx::float32)},
            {"first_video_velocity",
             tc::mx::astype(*result.first_video_velocity, tc::mx::float32)},
            {"first_audio_velocity",
             tc::mx::astype(*result.first_audio_velocity, tc::mx::float32)},
        });

        std::cout << "{\"width\":" << width
                  << ",\"height\":" << height
                  << ",\"frames\":" << frames
                  << ",\"sequence\":" << result.layout.sequence_length
                  << ",\"denoise_seconds\":" << result.metrics.denoise_seconds
                  << ",\"peak_bytes\":" << result.metrics.peak_bytes
                  << ",\"qmm_calls\":" << result.metrics.quantized_matmul_calls
                  << ",\"dq_gemm_calls\":" << result.metrics.dequantized_gemm_calls
                  << ",\"experimental_fused_qkv\":"
                  << (result.metrics.experimental_fused_qkv ? "true" : "false")
                  << "}\n";
        pipeline.unload();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

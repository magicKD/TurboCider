#include "../../native/models/h3_mlx/pipeline.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 2 || argc == 3,
                    "usage: h3-mlx-vsa-pipeline-probe VSA_INT6_CHECKPOINT_DIR [OUTPUT]");
        const auto root = std::filesystem::absolute(argv[1]);
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::h3_mlx::Pipeline pipeline;
        pipeline.load(root, event, cancelled);

        constexpr int text_tokens = 4;
        auto text = tc::mx::zeros({text_tokens, 5120}, tc::mx::float32);
        std::vector<int32_t> tags(text_tokens, tc::h3_mlx::text_tag);
        tc::h3_mlx::DenoiseOptions options;
        options.width = 32;
        options.height = 32;
        options.frames = 22;
        options.fps = 24;
        options.steps = 4;
        options.seed = 2026;
        options.affine_dq_gemm_min_rows = 768;
        options.video_noise = tc::mx::zeros({7, 96}, tc::mx::float32);
        options.audio_noise = tc::mx::zeros({74, 32}, tc::mx::float32);
        options.vsa.enabled = true;
        options.vsa.sparsity = 0.9f;
        options.vsa.tile_size = 64;
        options.vsa.prefix_mode = tc::h3_mlx::VSAPrefixMode::exempt;
        options.vsa.implementation = tc::h3_mlx::VSAImplementation::reference;
        options.capture_steps = argc == 3;
        auto result = pipeline.denoise(text, tags, options, event, cancelled);
        tc::mx::eval(result.video_rows, result.audio_rows);
        tc::require(result.video_rows.shape(1) == 96,
                    "H3 VSA probe video patch geometry mismatch");
        tc::require(result.audio_rows.shape(1) == 32,
                    "H3 VSA probe audio row geometry mismatch");
        tc::require(result.metrics.vsa.has_value(),
                    "H3 VSA probe did not return VSA metrics");
        const auto &vsa = *result.metrics.vsa;
        tc::require(vsa.sparse_calls > 0 && vsa.video_keep > 0,
                    "H3 VSA probe did not execute sparse attention");
        if (argc == 3) {
            tc::mx::save_safetensors(
                std::filesystem::absolute(argv[2]).string(),
                {{"video_rows", result.video_rows},
                 {"audio_rows", result.audio_rows}});
        }
        std::cout << "{\"video_rows\":" << result.video_rows.shape(0)
                  << ",\"audio_rows\":" << result.audio_rows.shape(0)
                  << ",\"denoise_seconds\":" << result.metrics.denoise_seconds
                  << ",\"peak_bytes\":" << result.metrics.peak_bytes
                  << ",\"attention_calls\":" << vsa.attention_calls
                  << ",\"sparse_calls\":" << vsa.sparse_calls
                  << ",\"video_keep\":" << vsa.video_keep
                  << ",\"achieved_sparsity\":" << vsa.achieved_sparsity
                  << ",\"implementation\":\"" << vsa.implementation
                  << "\"}\n";
        pipeline.unload();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

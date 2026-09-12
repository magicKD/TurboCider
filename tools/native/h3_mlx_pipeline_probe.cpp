#include "../../native/models/h3_mlx/pipeline.hpp"

#include <iostream>
#include <unordered_map>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 2 || argc == 3,
                    "usage: h3-mlx-pipeline-probe INT6_CHECKPOINT_DIR [OUTPUT]");
        const auto root = std::filesystem::absolute(argv[1]);
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::h3_mlx::Pipeline pipeline;
        pipeline.load(root, event, cancelled);

        // A deliberately tiny valid H3 geometry exercises every real DiT
        // block while keeping the packed sequence short. This is a loader and
        // numerical smoke test, not a quality benchmark.
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
        options.capture_first_velocity = argc == 3;
        options.capture_steps = argc == 3;
        options.capture_debug = argc == 3;
        auto result = pipeline.denoise(text, tags, options, event, cancelled);
        tc::mx::eval(result.video_rows, result.audio_rows);
        tc::require(result.video_rows.shape(1) == 96,
                    "H3 MLX probe video patch geometry mismatch");
        tc::require(result.audio_rows.shape(1) == 32,
                    "H3 MLX probe audio row geometry mismatch");
        if (argc == 3) {
            std::vector<float> positions(result.layout.position_ids.begin(),
                                          result.layout.position_ids.end());
            auto position_tensor = tc::Tensor(positions.data(),
                {result.layout.sequence_length, 3}, tc::mx::float32);
            auto tags_tensor = tc::Tensor(result.layout.token_tags.data(),
                {result.layout.sequence_length}, tc::mx::int32);
            std::unordered_map<std::string, tc::Tensor> tensors{
                 {"video_rows", result.video_rows}, {"audio_rows", result.audio_rows},
                 {"first_video_velocity", *result.first_video_velocity},
                 {"first_audio_velocity", *result.first_audio_velocity},
                 {"first_video_sample", *result.first_video_sample},
                 {"first_audio_sample", *result.first_audio_sample},
                 {"debug_video_embed", *result.debug.video_embed},
                 {"debug_audio_embed", *result.debug.audio_embed},
                 {"debug_text_embed", *result.debug.text_embed},
                 {"debug_packed_embed", *result.debug.packed_embed},
                 {"debug_first_norm1", *result.debug.first_norm1},
                 {"debug_first_modulated1", *result.debug.first_modulated1},
                 {"debug_first_query", *result.debug.first_query},
                 {"debug_first_key", *result.debug.first_key},
                 {"debug_first_value", *result.debug.first_value},
                 {"debug_first_attention", *result.debug.first_attention},
                 {"debug_first_after_attention", *result.debug.first_after_attention},
                 {"debug_first_norm2", *result.debug.first_norm2},
                 {"debug_first_modulated2", *result.debug.first_modulated2},
                 {"debug_first_feed_forward", *result.debug.first_feed_forward},
                 {"debug_first_block", *result.debug.first_block},
                 {"position_ids", position_tensor}, {"token_tags", tags_tensor}};
            for (size_t step = 0; step < result.steps.size(); ++step) {
                const auto suffix = std::to_string(step);
                tensors.emplace("step" + suffix + "_video_velocity",
                                result.steps[step].video_velocity);
                tensors.emplace("step" + suffix + "_audio_velocity",
                                result.steps[step].audio_velocity);
                tensors.emplace("step" + suffix + "_video_sample",
                                result.steps[step].video_sample);
                tensors.emplace("step" + suffix + "_audio_sample",
                                result.steps[step].audio_sample);
                tc::Tensor unique(result.steps[step].unique_timesteps.data(),
                                   {int(result.steps[step].unique_timesteps.size())},
                                   tc::mx::float32);
                tc::Tensor inverse(result.steps[step].inverse_timesteps.data(),
                                    {int(result.steps[step].inverse_timesteps.size())},
                                    tc::mx::int32);
                tensors.emplace("step" + suffix + "_unique_timesteps", unique);
                tensors.emplace("step" + suffix + "_inverse_timesteps", inverse);
            }
            tc::mx::save_safetensors(
                std::filesystem::absolute(argv[2]).string(), tensors);
        }
        std::cout << "{\"video_rows\":" << result.video_rows.shape(0)
                  << ",\"audio_rows\":" << result.audio_rows.shape(0)
                  << ",\"denoise_seconds\":" << result.metrics.denoise_seconds
                  << ",\"peak_bytes\":" << result.metrics.peak_bytes
                  << ",\"qmm_calls\":" << result.metrics.quantized_matmul_calls
                  << ",\"dq_gemm_calls\":" << result.metrics.dequantized_gemm_calls
                  << "}\n";
        pipeline.unload();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

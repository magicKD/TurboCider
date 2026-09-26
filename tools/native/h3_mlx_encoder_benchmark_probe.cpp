// Full 50-layer H3 Qwen3-VL encoder benchmark with deterministic token IDs.
#include "../../native/models/h3_mlx/conditioner.hpp"
#include "../../native/backends/coreml.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sys/resource.h>

int main(int argc, char **argv) {
    try {
        tc::require(
            argc == 6 || argc == 7,
            "usage: h3-mlx-encoder-benchmark-probe TEXT_ENCODER TOKENIZER "
            "TOKENS RUNS OUTPUT [ENCODER_MANIFEST]");
        tc::configure_streams();
        char *token_end = nullptr;
        char *run_end = nullptr;
        const long token_count = std::strtol(argv[3], &token_end, 10);
        const long runs = std::strtol(argv[4], &run_end, 10);
        tc::require(token_end != argv[3] && *token_end == '\0' &&
                        token_count >= 1 && token_count <= 512 &&
                        run_end != argv[4] && *run_end == '\0' &&
                        runs >= 1 && runs <= 50,
                    "H3 encoder benchmark requires 1...512 tokens and 1...50 runs");
        const auto component = std::filesystem::absolute(argv[1]);
        const auto tokenizer = std::filesystem::absolute(argv[2]);
        const auto output = std::filesystem::absolute(argv[5]);
        const auto manifest = argc == 7
            ? std::filesystem::absolute(argv[6])
            : std::filesystem::path{};
        tc::require(!std::filesystem::exists(output),
                    "H3 encoder benchmark output already exists");

        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        const auto load_started = tc::Clock::now();
        tc::h3_mlx::Conditioner conditioner(component, tokenizer, manifest, 0);
        const double load_seconds = std::chrono::duration<double>(
            tc::Clock::now() - load_started).count();
        std::vector<int> tokens;
        tokens.reserve(static_cast<size_t>(token_count));
        for (int index = 0; index < token_count; ++index)
            tokens.push_back(100 + (index * 7919) %
                (conditioner.config().vocabulary_size - 100));

        std::cout << "{\"tokens\":" << token_count
                  << ",\"runs\":" << runs
                  << ",\"fused_sdpa\":"
                  << (std::getenv("TURBOCIDER_H3_FUSED_SDPA") ? "true" : "false")
                  << ",\"hybrid\":" << (manifest.empty() ? "false" : "true")
                  << ",\"load_seconds\":" << load_seconds
                  << ",\"samples\":[";
        std::optional<tc::h3_mlx::ConditioningResult> result;
        for (int iteration = 0; iteration <= runs; ++iteration) {
            tc::mx::reset_peak_memory();
            const auto started = tc::Clock::now();
            result = conditioner.encode_tokens(tokens, event, cancelled);
            tc::mx::eval(result->hidden_states);
            const double seconds = std::chrono::duration<double>(
                tc::Clock::now() - started).count();
            rusage usage{};
            getrusage(RUSAGE_SELF, &usage);
            if (iteration)
                std::cout << ',';
            std::cout << "{\"warmup\":"
                      << (iteration == 0 ? "true" : "false")
                      << ",\"seconds\":" << seconds
                      << ",\"mlx_peak_bytes\":" << tc::mx::get_peak_memory()
                      << ",\"process_peak_rss_bytes\":" << usage.ru_maxrss;
            if (auto *hybrid = conditioner.hybrid_session()) {
                const auto metrics = hybrid->metrics();
                std::cout << ",\"ane_calls_session_total\":" << metrics.calls
                          << ",\"ane_seconds_session_total\":"
                          << metrics.prediction_seconds
                          << ",\"runtime_failures_session_total\":"
                          << metrics.runtime_failures
                          << ",\"runtime_failed\":"
                          << (metrics.runtime_failed ? "true" : "false")
                          << ",\"runtime_failure_block\":"
                          << metrics.runtime_failure_block
                          << ",\"output_copy_bytes_session_total\":"
                          << metrics.copied_bytes
                          << ",\"coreml_bucket\":" << metrics.bucket
                          << ",\"coreml_block_count\":" << metrics.block_count
                          << ",\"minimum_profitable_rows\":"
                          << metrics.minimum_profitable_rows
                          << ",\"coreml_load_seconds\":" << metrics.load_seconds
                          << ",\"prefill_actual_tokens\":"
                          << metrics.prefill_actual_tokens
                          << ",\"prefill_selected_bucket\":"
                          << metrics.prefill_selected_bucket
                          << ",\"prefill_compute_tokens\":"
                          << metrics.prefill_compute_tokens
                          << ",\"prefill_padding_tokens\":"
                          << metrics.prefill_padding_tokens
                          << ",\"prefill_fixed_shape\":"
                          << (metrics.prefill_fixed_shape ? "true" : "false")
                          << ",\"prefill_plan_reason\":\""
                          << metrics.prefill_plan_reason << "\"";
            }
            std::cout << '}';
        }
        std::vector<int32_t> tags(tokens.size(), tc::h3_mlx::text_tag);
        tc::Tensor tag_tensor(tags.data(), {static_cast<int>(tags.size())},
                              tc::mx::int32);
        tc::mx::save_safetensors(
            output.string(),
            {{"hidden_states", result->hidden_states}, {"token_tags", tag_tensor}});
        std::cout << "]}" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

// Encoder-only benchmark; deterministic IDs isolate conditioning from tokenization.
#include "../../native/components/text/qwen3.hpp"
#include "../../native/backends/coreml.hpp"
#include <iostream>
#include <sys/resource.h>

int main(int argc, char **argv) {
    try {
        tc::require(argc >= 5 && argc <= 7,
                    "usage: qwen3-quant-probe WEIGHTS TOKENS RUNS OUTPUT "
                    "[z_image|flux_klein] [ENCODER_MANIFEST]");
        tc::configure_streams();
        const int count = std::stoi(argv[2]), runs = std::stoi(argv[3]);
        tc::require(count > 0 && count <= 2048 && runs > 0, "invalid benchmark size");
        std::string mode = "z_image";
        std::optional<std::filesystem::path> manifest;
        if (argc >= 6) {
            const std::string argument = argv[5];
            if (argument == "z_image" || argument == "flux_klein")
                mode = argument;
            else
                manifest = argument; // Preserve the original optional-manifest form.
        }
        if (argc == 7) {
            tc::require(!manifest.has_value(),
                        "select the encoder mode before the manifest path");
            manifest = argv[6];
        }
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::Weights weights;
        const auto weight_path = std::filesystem::absolute(argv[1]);
        const auto checkpoint = tc::components::qwen3_checkpoint_path(weight_path);
        auto start = tc::Clock::now();
        if (std::filesystem::is_directory(weight_path) ||
            weight_path.filename().string().ends_with(".safetensors.index.json"))
            weights.load(std::filesystem::is_directory(weight_path)
                             ? weight_path : weight_path.parent_path(),
                         event, cancelled);
        else
            weights.load_file(weight_path);
        weights.materialize();
        std::unique_ptr<tc::HybridSession> hybrid;
        if (manifest)
            hybrid = std::make_unique<tc::HybridSession>(
                std::filesystem::absolute(*manifest),
                weight_path.parent_path(), count, event,
                cancelled, 0, checkpoint,
                std::vector<tc::LoRAAsset>{}, 0,
                mode == "flux_klein" ? 27 : 35, true);
        const double load = std::chrono::duration<double>(tc::Clock::now() - start).count();
        tc::Tokens tokens;
        tokens.valid = count;
        for (int i = 0; i < count; ++i) tokens.ids.push_back(100 + (i * 7919) % 100000);
        std::cout << "{\"mode\":\"" << mode
                  << "\",\"tokens\":" << count
                  << ",\"runs\":" << runs
                  << ",\"hybrid\":" << (hybrid ? "true" : "false")
                  << ",\"weight_bytes\":" << weights.bytes()
                  << ",\"load_seconds\":" << load << ",\"samples\":[";
        auto config = mode == "flux_klein"
                          ? tc::components::Qwen3Conditioning::flux_klein()
                          : tc::components::Qwen3Conditioning::z_image();
        for (int i = 0; i <= runs; ++i) {
            tc::mx::reset_peak_memory();
            start = tc::Clock::now();
            auto result = tc::components::qwen3_conditioning(
                tokens, weights, config, event, cancelled, hybrid.get());
            tc::mx::eval(result);
            const double seconds = std::chrono::duration<double>(tc::Clock::now() - start).count();
            rusage usage{};
            getrusage(RUSAGE_SELF, &usage);
            if (i) std::cout << ',';
            std::cout << "{\"warmup\":" << (i == 0 ? "true" : "false") << ",\"seconds\":" << seconds
                      << ",\"mlx_peak_bytes\":" << tc::mx::get_peak_memory()
                      << ",\"process_peak_rss_bytes\":" << usage.ru_maxrss;
            if (hybrid) {
                auto metrics = hybrid->metrics();
                std::cout << ",\"ane_calls_session_total\":" << metrics.calls
                          << ",\"ane_seconds_session_total\":"
                          << metrics.prediction_seconds
                          << ",\"ane_first_runtime_calls_session_total\":"
                          << metrics.first_runtime_prediction_calls
                          << ",\"ane_first_runtime_seconds_session_total\":"
                          << metrics.first_runtime_prediction_seconds
                          << ",\"ane_subsequent_runtime_calls_session_total\":"
                          << metrics.subsequent_runtime_prediction_calls
                          << ",\"ane_subsequent_runtime_seconds_session_total\":"
                          << metrics.subsequent_runtime_prediction_seconds
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
                          << ",\"qualified_flexible_backing\":"
                          << (metrics.qualified_flexible_backing ? "true" : "false")
                          << ",\"coreml_load_seconds\":" << metrics.load_seconds
                          << ",\"coreml_model_load_seconds\":"
                          << metrics.model_load_seconds
                          << ",\"coreml_interface_setup_seconds\":"
                          << metrics.model_interface_setup_seconds
                          << ",\"coreml_output_backing_setup_seconds\":"
                          << metrics.output_backing_setup_seconds
                          << ",\"quality_validation_calls_session_total\":"
                          << metrics.quality_validation_calls
                          << ",\"quality_max_relative_l2_session\":"
                          << metrics.quality_max_relative_l2
                          << ",\"quality_min_cosine_session\":"
                          << metrics.quality_min_cosine
                          << ",\"quality_max_abs_session\":"
                          << metrics.quality_max_abs
                          << ",\"quality_max_relative_abs_session\":"
                          << metrics.quality_max_relative_abs
                          << ",\"quality_validation_passed\":"
                          << (metrics.quality_validation_passed ? "true" : "false")
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
            std::cout << "}";
            if (i == runs) tc::mx::save_safetensors(argv[4], {{"conditioning", result}});
        }
        std::cout << "]}" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}

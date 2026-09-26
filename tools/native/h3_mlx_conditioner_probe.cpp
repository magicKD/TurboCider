#include "../../native/models/h3_mlx/conditioner.hpp"
#include "../../native/backends/coreml.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

int main(int argc, char **argv) {
    try {
        tc::require(argc >= 5,
                    "usage: h3-mlx-conditioner-probe TEXT_ENCODER TOKENIZER OUTPUT PROMPT "
                    "[--encoder-ane-manifest PATH] [--warmup-iterations 0...8] "
                    "[--layers] [--intermediates]");
        bool save_layers = false;
        bool save_intermediates = false;
        std::filesystem::path encoder_manifest;
        int warmup_iterations = 0;
        bool warmup_seen = false;
        for (int index = 5; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--layers") {
                save_layers = true;
            } else if (option == "--intermediates") {
                save_intermediates = true;
            } else if (option == "--encoder-ane-manifest") {
                tc::require(++index < argc && encoder_manifest.empty(),
                            "--encoder-ane-manifest requires one path");
                encoder_manifest = std::filesystem::absolute(argv[index]);
            } else if (option == "--warmup-iterations") {
                tc::require(++index < argc && !warmup_seen,
                            "--warmup-iterations requires one integer");
                char *end = nullptr;
                const long parsed = std::strtol(argv[index], &end, 10);
                tc::require(end != argv[index] && *end == '\0' &&
                                parsed >= 0 && parsed <= 8,
                            "--warmup-iterations must be 0...8");
                warmup_iterations = static_cast<int>(parsed);
                warmup_seen = true;
            } else {
                tc::require(false,
                            "unknown H3 conditioner probe option: " + option);
            }
        }
        tc::require(encoder_manifest.empty() || !save_intermediates,
                    "hybrid intermediate capture is a diagnostic full-MLP path; "
                    "run it separately from timing measurements");
        const auto component = std::filesystem::absolute(argv[1]);
        const auto tokenizer = std::filesystem::absolute(argv[2]);
        const auto output = std::filesystem::absolute(argv[3]);
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &phase, int current, int total) {
            if (current == total || current % 5 == 0)
                std::cerr << phase << ' ' << current << '/' << total << '\n';
        };
        tc::mx::reset_peak_memory();
        const auto started = tc::Clock::now();
        tc::h3_mlx::Conditioner conditioner(
            component, tokenizer, encoder_manifest, warmup_iterations);
        std::vector<tc::Tensor> debug_layers;
        tc::h3_mlx::ConditionerDebugTensors debug_tensors;
        auto result = conditioner.encode_prompt(
            argv[4], event, cancelled, save_layers ? &debug_layers : nullptr,
            save_intermediates ? &debug_tensors : nullptr);
        auto tags = tc::Tensor(result.token_tags.data(),
                               {static_cast<int>(result.token_tags.size())},
                               tc::mx::int32);
        tc::mx::eval(result.hidden_states, tags);
        std::unordered_map<std::string, tc::Tensor> tensors{
            {"hidden_states", result.hidden_states}, {"token_tags", tags}};
        if (save_layers) {
            for (size_t index = 0; index < debug_layers.size(); ++index)
                tensors.emplace("layer" + std::to_string(index), debug_layers[index]);
        }
        for (auto &[name, value] : debug_tensors)
            tensors.emplace("debug_" + name, value);
        tc::mx::save_safetensors(output.string(), tensors);
        const double seconds = std::chrono::duration<double>(
            tc::Clock::now() - started).count();
        std::cout << "{\"tokens\":" << result.tokens
                  << ",\"seconds\":" << seconds
                  << ",\"peak_bytes\":" << tc::mx::get_peak_memory();
        if (auto *hybrid = conditioner.hybrid_session()) {
            const auto metrics = hybrid->metrics();
            std::cout << ",\"encoder_hybrid\":{"
                      << "\"load_seconds\":" << metrics.load_seconds
                      << ",\"prediction_seconds\":" << metrics.prediction_seconds
                      << ",\"runtime_calls\":" << metrics.runtime_calls
                      << ",\"warmup_calls\":" << metrics.warmup_calls
                      << ",\"bucket\":" << metrics.bucket
                      << ",\"ane_mlp_start\":" << metrics.ane_mlp_start
                      << ",\"ane_mlp_end\":" << metrics.ane_mlp_end
                      << ",\"quality_validation_calls\":"
                      << metrics.quality_validation_calls
                      << ",\"quality_validation_passed\":"
                      << (metrics.quality_validation_passed ? "true" : "false")
                      << "}";
        }
        std::cout
                  << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

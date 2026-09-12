#include "../../native/models/h3_mlx/conditioner.hpp"

#include <chrono>
#include <iostream>
#include <unordered_map>

int main(int argc, char **argv) {
    try {
        tc::require(argc >= 5 && argc <= 7,
                    "usage: h3-mlx-conditioner-probe TEXT_ENCODER TOKENIZER OUTPUT PROMPT [--layers] [--intermediates]");
        bool save_layers = false;
        bool save_intermediates = false;
        for (int index = 5; index < argc; ++index) {
            const std::string option = argv[index];
            tc::require(option == "--layers" || option == "--intermediates",
                        "unknown H3 conditioner probe option: " + option);
            save_layers = save_layers || option == "--layers";
            save_intermediates = save_intermediates || option == "--intermediates";
        }
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
        tc::h3_mlx::Conditioner conditioner(component, tokenizer);
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
                  << ",\"peak_bytes\":" << tc::mx::get_peak_memory()
                  << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

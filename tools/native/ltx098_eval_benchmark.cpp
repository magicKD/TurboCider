// Probe only: compare MLX block materialization intervals on identical input.
#include "../../native/models/ltx098/transformer.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 7, "usage: ltx098-eval-benchmark CHECKPOINT INPUT OUTPUT EVAL_EVERY WARMUPS REPEATS");
        const std::filesystem::path output(argv[3]);
        const int interval = std::stoi(argv[4]), warmups = std::stoi(argv[5]), repeats = std::stoi(argv[6]);
        tc::require(!std::filesystem::exists(output) && interval >= 0 && interval <= 28 &&
                    warmups >= 1 && warmups <= 10 && repeats >= 2 && repeats <= 20,
                    "invalid output, interval, warmup or repeat count");
        auto data = tc::mx::load_safetensors(argv[2]).first;
        tc::ltx098::Transformer model;
        model.load(argv[1], 28, interval);
        auto text = model.prepare_text(data.at("text"));
        tc::mx::eval(text);
        std::vector<double> samples;
        std::optional<tc::Tensor> last;
        for (int i = 0; i < warmups + repeats; ++i) {
            const auto start = std::chrono::steady_clock::now();
            auto prediction = model.forward(data.at("latents"), text, .7f, 24.f, true, 1.f, data.at("text_mask"));
            tc::mx::eval(prediction);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (i >= warmups) samples.push_back(seconds);
            last = std::move(prediction);
        }
        tc::mx::save_safetensors(output.string(), {{"output", *last}});
        std::cout << std::setprecision(12) << "{\"interval\":" << interval << ",\"warmups\":" << warmups
                  << ",\"seconds\":[";
        for (size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
        std::cout << "]}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

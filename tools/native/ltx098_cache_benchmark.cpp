#include "../../native/models/ltx098/transformer.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 6, "usage: ltx098-cache-benchmark CHECKPOINT INPUT NEW_OUTPUT WARMUPS REPEATS");
        const std::filesystem::path destination(argv[3]);
        const int warmups = std::stoi(argv[4]), repeats = std::stoi(argv[5]);
        tc::require(!std::filesystem::exists(destination) && warmups >= 1 && repeats >= 2,
                    "invalid output or iteration count");
        auto input = tc::mx::load_safetensors(argv[2]).first;
        tc::ltx098::Transformer model;
        model.load(argv[1]);
        auto text = model.prepare_text(input.at("text"));
        tc::mx::eval(text);
        const auto start_cache = std::chrono::steady_clock::now();
        auto cache = model.prepare_cross_attention(text);
        const double cache_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_cache).count();
        std::vector<std::vector<double>> samples(4);
        std::vector<bool> exact;
        std::optional<tc::Tensor> reference, last;
        for (int run = 0; run < 4; ++run) {
            const bool cached = run == 1 || run == 2;
            for (int iteration = 0; iteration < warmups + repeats; ++iteration) {
                const auto started = std::chrono::steady_clock::now();
                auto value = cached ? model.forward_cached(input.at("latents"), text, .7f, 24.f, true, 1.f,
                                                            input.at("text_mask"), cache)
                                    : model.forward(input.at("latents"), text, .7f, 24.f, true, 1.f,
                                                    input.at("text_mask"));
                tc::mx::eval(value);
                if (iteration >= warmups)
                    samples[run].push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
                last = std::move(value);
            }
            if (!reference) reference = *last;
            exact.push_back(tc::mx::all(*reference == *last).item<bool>());
        }
        tc::mx::save_safetensors(destination.string(), {{"baseline", *reference}, {"last", *last}});
        std::cout << std::setprecision(12) << "{\"cache_prepare_seconds\":" << cache_seconds << ",\"runs\":[";
        for (int run = 0; run < 4; ++run) {
            std::cout << (run ? "," : "") << "{\"cached\":" << ((run == 1 || run == 2) ? "true" : "false")
                      << ",\"exact\":" << (exact[run] ? "true" : "false") << ",\"seconds\":[";
            for (size_t i = 0; i < samples[run].size(); ++i) std::cout << (i ? "," : "") << samples[run][i];
            std::cout << "]}";
        }
        std::cout << "]}\n";
        tc::require(std::all_of(exact.begin(), exact.end(), [](bool value) { return value; }),
                    "cached and uncached transformer outputs differ");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

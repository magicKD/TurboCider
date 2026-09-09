// Development-only comparison; baseline.hpp is extracted from pinned Git history.
#include "../../native/components/text/qwen3.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include "baseline.hpp"

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4, "usage: qwen3-refactor-probe WEIGHTS TOKENS VALID");
        const int count = std::stoi(argv[2]), valid = std::stoi(argv[3]);
        tc::require(count > 0 && count <= 512 && valid > 0 && valid <= count,
                    "invalid token geometry");
        tc::configure_streams();
        tc::Tokens tokens;
        tokens.valid = valid;
        // Deterministic vocabulary IDs, including masked padding. This tests
        // the encoder, not tokenizer or prompt semantics.
        for (int i = 0; i < count; ++i)
            tokens.ids.push_back(i < valid ? 100 + (i * 7919) % 100000 : 0);
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::Weights weights;
        weights.load(std::filesystem::absolute(argv[1]), event, cancelled);
        tc::require(weights.at("model.embed_tokens.weight").shape(1) == 2560 &&
                    weights.has("model.layers.35.input_layernorm.weight"),
                    "probe requires the 36-layer Qwen3-4B checkpoint");
        weights.materialize();
        for (bool z_image : {false, true}) {
            auto config = z_image ? tc::components::Qwen3Conditioning::z_image()
                                  : tc::components::Qwen3Conditioning::flux_klein();
            auto run = [&](bool shared) {
                if (shared)
                    return tc::components::qwen3_conditioning(tokens, weights, config,
                                                             event, cancelled);
                if (!z_image) return tc::baseline::flux(tokens, weights, event, cancelled);
                auto ids = tc::Tensor(tokens.ids.data(), {1, count}, tc::mx::int32);
                return tc::baseline::qwen_encode(ids, weights, valid, event, cancelled);
            };
            auto reference = run(false);
            auto candidate = run(true);
            tc::mx::eval(reference, candidate);
            tc::require(reference.shape() == candidate.shape() &&
                        reference.dtype() == candidate.dtype(), "output contract differs");
            auto exact = tc::mx::all(reference == candidate);
            auto finite = tc::mx::all(tc::mx::isfinite(candidate));
            tc::mx::eval(exact, finite);
            tc::require(exact.item<bool>() && finite.item<bool>(),
                        "encoder output is nonfinite or differs from baseline");
            std::vector<double> old_times, new_times;
            // Three ABBA blocks, both implementations warmed above. Always
            // evaluate inside the timer; correctness checks are outside it.
            for (int block = 0; block < 3; ++block) {
                for (bool shared : {false, true, true, false}) {
                    const auto start = tc::Clock::now();
                    auto output = run(shared);
                    tc::mx::eval(output);
                    const double seconds = std::chrono::duration<double>(tc::Clock::now() - start).count();
                    (shared ? new_times : old_times).push_back(seconds);
                    auto equal = tc::mx::all(reference == output);
                    tc::mx::eval(equal);
                    tc::require(equal.item<bool>(), "repeated encoder output differs");
                    std::cout << "sample mode=" << (z_image ? "z_image" : "flux_klein")
                              << " shared=" << shared << " seconds=" << seconds << std::endl;
                }
            }
            auto median = [](std::vector<double> values) {
                std::sort(values.begin(), values.end());
                return (values[2] + values[3]) / 2;
            };
            const double before = median(old_times), after = median(new_times);
            std::cout << "PASS mode=" << (z_image ? "z_image" : "flux_klein")
                      << " tokens=" << count << " valid=" << valid
                      << " exact=true baseline_median=" << before
                      << " shared_median=" << after << " ratio=" << after / before << std::endl;
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

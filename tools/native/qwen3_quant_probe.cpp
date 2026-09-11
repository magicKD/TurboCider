// Encoder-only benchmark; deterministic IDs isolate conditioning from tokenization.
#include "../../native/components/text/qwen3.hpp"
#include <iostream>
#include <sys/resource.h>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 5, "usage: qwen3-quant-probe WEIGHTS TOKENS RUNS OUTPUT");
        tc::configure_streams();
        const int count = std::stoi(argv[2]), runs = std::stoi(argv[3]);
        tc::require(count > 0 && count <= 512 && runs > 0, "invalid benchmark size");
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::Weights weights;
        auto start = tc::Clock::now();
        weights.load_file(argv[1]);
        weights.materialize();
        const double load = std::chrono::duration<double>(tc::Clock::now() - start).count();
        tc::Tokens tokens;
        tokens.valid = count;
        for (int i = 0; i < count; ++i) tokens.ids.push_back(100 + (i * 7919) % 100000);
        std::cout << "{\"weight_bytes\":" << weights.bytes() << ",\"load_seconds\":" << load << ",\"samples\":[";
        auto config = tc::components::Qwen3Conditioning::z_image();
        for (int i = 0; i <= runs; ++i) {
            tc::mx::reset_peak_memory();
            start = tc::Clock::now();
            auto result = tc::components::qwen3_conditioning(tokens, weights, config, event, cancelled);
            tc::mx::eval(result);
            const double seconds = std::chrono::duration<double>(tc::Clock::now() - start).count();
            rusage usage{};
            getrusage(RUSAGE_SELF, &usage);
            if (i) std::cout << ',';
            std::cout << "{\"warmup\":" << (i == 0 ? "true" : "false") << ",\"seconds\":" << seconds
                      << ",\"mlx_peak_bytes\":" << tc::mx::get_peak_memory()
                      << ",\"process_peak_rss_bytes\":" << usage.ru_maxrss << "}";
            if (i == runs) tc::mx::save_safetensors(argv[4], {{"conditioning", result}});
        }
        std::cout << "]}" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}

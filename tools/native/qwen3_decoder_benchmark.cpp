// Matched whole-encoder screen, same checkpoint/math/budget. Not qualification.
#include "../../native/components/text/qwen3_gguf.hpp"
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    try {
        using namespace tc;
        require(argc == 7, "usage: qwen3-decoder-benchmark GGUF CONFIG TOKENIZER OUTDIR PREFETCH ITERATIONS");
        const std::filesystem::path output(argv[4]);
        require(!std::filesystem::exists(output), "output already exists");
        const auto prefetch = uint32_t(std::stoul(argv[5])); const int iterations = std::stoi(argv[6]);
        require(iterations >= 8 && iterations <= 100, "iterations must be 8..100");
        std::filesystem::create_directories(output); configure_streams(); mx::set_cache_limit(0);
        std::atomic<bool> cancel{false}; Event event = [](const std::string &, int, int) {};
        std::optional<Tensor> reference;
        std::cout << std::setprecision(17) << "{\"schema\":\"tc-qwen3-decoder-screen-v1\",\"prefetch\":"
                  << prefetch << ",\"warmups_per_arm\":3,\"managed_limit_bytes\":8589934592,\"samples\":[";
        bool first = true;
        for (int i = 0; i < iterations + 3; ++i) {
            for (bool simd : {bool(i % 2), !bool(i % 2)}) {
                const auto start = Clock::now();
                components::Qwen3GgufEncoder encoder(argv[1], argv[2], argv[3], prefetch, 8ull << 30, event, cancel, {simd});
                const auto tokens = encoder.tokenize("A studio photograph of an adult ceramic artist holding a blue cup.", true);
                auto value = encoder.encode(tokens); mx::eval(value);
                const double wall = std::chrono::duration<double>(Clock::now() - start).count();
                const auto metrics = encoder.metrics();
                if (!reference) reference = value;
                require(reference->shape() == value.shape() && reference->dtype() == value.dtype() &&
                        mx::all(*reference == value).item<bool>(), "scalar/SIMD encoder output differs");
                if (i == iterations + 2) mx::save_safetensors((output / (simd ? "simd.safetensors" : "scalar.safetensors")).string(), {{"tensor", value}});
                require(encoder.drain_safely(), "unproven encoder drain");
                if (!first) std::cout << ','; first = false;
                std::cout << "{\"iteration\":" << i << ",\"warmup\":" << (i < 3 ? "true" : "false")
                          << ",\"decoder\":\"" << (simd ? "cpu_simd" : "cpu_scalar") << "\",\"wall_seconds\":" << wall
                          << ",\"decode_seconds\":" << metrics.decode_seconds << ",\"wait_seconds\":" << metrics.exposed_wait_seconds
                          << ",\"managed_peak_bytes\":" << metrics.managed_peak_bytes << ",\"fills\":" << metrics.fills
                          << ",\"slots\":" << metrics.slots << ",\"token_rows\":" << tokens.ids.size() << '}';
                std::cout.flush();
            }
        }
        std::cout << "],\"correctness\":\"exact\",\"production_qualified\":false}\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

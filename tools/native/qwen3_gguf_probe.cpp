#include "../../native/components/text/qwen3_gguf.hpp"
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    try {
        using namespace tc;
        require(argc >= 6 && argc <= 11, "usage: qwen3-gguf-probe GGUF CONFIG TOKENIZER_JSON OUTDIR PREFETCH [REFERENCE_WEIGHTS|-] [PROMPT] [BUDGET] [simd|scalar] [packed_resident|packed_streamed]");
        configure_streams(); mx::set_cache_limit(0);
        const std::filesystem::path output(argv[4]);
        require(!std::filesystem::exists(output), "probe output already exists");
        std::filesystem::create_directories(output);
        const uint32_t prefetch = uint32_t(std::stoul(argv[5]));
        const uint64_t budget = argc > 8 ? std::stoull(argv[8]) : 8ull << 30;
        require(argc <= 9 || std::string(argv[9]) == "simd" || std::string(argv[9]) == "scalar", "invalid decoder mode");
        const gguf::DecodeOptions options{argc <= 9 || std::string(argv[9]) == "simd"};
        std::atomic<bool> cancelled{false};
        Event event = [](const std::string &, int, int) {};
        const auto start = Clock::now();
        components::Qwen3GgufEncoder encoder(argv[1], argv[2], argv[3], prefetch, budget, event, cancelled, options,
            argc > 10 ? argv[10] : "packed_resident");
        const auto tokens = encoder.tokenize(argc > 7 ? argv[7] : "A studio photograph of an adult ceramic artist holding a blue cup.", true);
        if (argc > 6 && std::string(argv[6]) != "-") {
            Weights reference; reference.load(argv[6], event, cancelled);
            auto hidden = components::qwen3_conditioning(tokens, reference, components::Qwen3Conditioning::z_image(), event, cancelled);
            mx::save_safetensors((output / "reference.safetensors").string(), {{"tensor", hidden}});
            reference.clear(); mx::synchronize(); mx::clear_cache();
        }
        auto value = encoder.encode(tokens);
        mx::save_safetensors((output / "conditioning.safetensors").string(), {{"tensor", value}});
        const auto m = encoder.metrics();
        std::cout << std::setprecision(17) << "{\"status\":\"pass\",\"source_sha256\":\"" << m.source_sha256
                  << "\",\"layout_digest\":\"" << m.layout_digest << "\",\"identity\":\"" << encoder.identity()
                  << "\",\"prefetch\":" << prefetch << ",\"slots\":" << m.slots << ",\"fills\":" << m.fills
                  << ",\"packed_capacity_bytes\":" << m.packed_capacity_bytes << ",\"dense_capacity_bytes\":" << m.dense_capacity_bytes
                  << ",\"managed_peak_bytes\":" << m.managed_peak_bytes << ",\"decode_seconds\":" << m.decode_seconds
                  << ",\"decode_backend\":\"" << m.decode_backend << "\""
                  << ",\"source_residency\":\"" << m.source_residency << "\",\"read_buffer_bytes\":" << m.read_buffer_bytes
                  << ",\"source_logical_bytes\":" << m.source_logical_bytes << ",\"source_read_bytes\":" << m.source_read_bytes
                  << ",\"streamed_read_seconds\":" << m.streamed_read_seconds
                  << ",\"wait_seconds\":" << m.exposed_wait_seconds << ",\"valid_rows\":" << tokens.valid
                  << ",\"elapsed_seconds\":" << std::chrono::duration<double>(Clock::now() - start).count()
                  << ",\"whole_request_certified\":false,\"token_ids\":[";
        for (size_t i = 0; i < tokens.ids.size(); ++i) { if (i) std::cout << ','; std::cout << tokens.ids[i]; }
        std::cout << "]}\n";
        require(encoder.drain_safely(), "unproven encoder drain"); return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

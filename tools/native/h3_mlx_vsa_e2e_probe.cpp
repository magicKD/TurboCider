#include "../../native/models/h3_mlx/pipeline.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>

namespace {

struct ProbeOptions {
    double sparsity = 0.9;
    int width = 832;
    int height = 480;
    int frames = 124;
    int tile_size = 64;
    bool gate_compress = true;
    bool capture_debug = false;
    tc::h3_mlx::VSAPrefixMode prefix_mode = tc::h3_mlx::VSAPrefixMode::exempt;
    tc::h3_mlx::VSAImplementation implementation =
        tc::h3_mlx::VSAImplementation::reference;
};

int parse_int(const char *name, const std::string &value) {
    char *end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    tc::require(end != value.c_str() && *end == '\0',
                std::string("invalid ") + name + ": " + value);
    tc::require(parsed > 0 && parsed <= 1'000'000,
                std::string(name) + " must be a positive integer");
    return static_cast<int>(parsed);
}

double parse_float(const char *name, const std::string &value) {
    char *end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    tc::require(end != value.c_str() && *end == '\0',
                std::string("invalid ") + name + ": " + value);
    tc::require(parsed >= 0.0 && parsed < 1.0,
                std::string(name) + " must be in [0, 1)");
    return parsed;
}

ProbeOptions parse_options(int argc, char **argv) {
    ProbeOptions options;
    // Keep the historical positional sparsity argument working while exposing
    // the dimensions and all FastVideo VSA routing knobs for reproducible
    // 22/124-frame probes.
    int index = 5;
    if (index < argc && argv[index][0] != '-') {
        options.sparsity = parse_float("sparsity", argv[index++]);
    }
    while (index < argc) {
        const std::string flag = argv[index++];
        tc::require(index < argc, "missing value for " + flag);
        const std::string value = argv[index++];
        if (flag == "--sparsity") {
            options.sparsity = parse_float("--sparsity", value);
        } else if (flag == "--width") {
            options.width = parse_int("--width", value);
        } else if (flag == "--height") {
            options.height = parse_int("--height", value);
        } else if (flag == "--frames") {
            options.frames = parse_int("--frames", value);
        } else if (flag == "--tile-size") {
            options.tile_size = parse_int("--tile-size", value);
            tc::require(options.tile_size == 64 || options.tile_size == 256,
                        "--tile-size must be 64 or 256");
        } else if (flag == "--prefix-mode") {
            options.prefix_mode = tc::h3_mlx::vsa_prefix_mode(value);
        } else if (flag == "--impl") {
            options.implementation = tc::h3_mlx::vsa_implementation(value);
        } else if (flag == "--disable-gate") {
            tc::require(value == "true" || value == "false",
                        "--disable-gate must be true or false");
            options.gate_compress = value != "true";
        } else if (flag == "--capture-debug") {
            tc::require(value == "true" || value == "false",
                        "--capture-debug must be true or false");
            options.capture_debug = value == "true";
        } else {
            tc::require(false, "unknown VSA probe option: " + flag);
        }
    }
    return options;
}

} // namespace

int main(int argc, char **argv) {
    try {
        tc::require(argc >= 5,
                    "usage: h3-mlx-vsa-e2e-probe CHECKPOINT CONDITIONING NOISE OUTPUT "
                    "[SPARSITY] [--width N --height N --frames N --tile-size 64|256 "
                    "--prefix-mode exempt|compete --impl auto|reference|simd "
                    "--disable-gate true|false --capture-debug true|false]");
        const auto probe = parse_options(argc, argv);
        const auto checkpoint_root = std::filesystem::absolute(argv[1]);
        auto conditioning_arrays = tc::mx::load_safetensors(
            std::filesystem::absolute(argv[2]).string()).first;
        auto noise_arrays = tc::mx::load_safetensors(
            std::filesystem::absolute(argv[3]).string()).first;
        auto conditioning = conditioning_arrays.find("conditioning");
        auto token_tags = conditioning_arrays.find("token_tags");
        auto video_noise = noise_arrays.find("video_noise");
        auto audio_noise = noise_arrays.find("audio_noise");
        tc::require(conditioning != conditioning_arrays.end() &&
                        token_tags != conditioning_arrays.end(),
                    "conditioning fixture must contain conditioning and token_tags");
        tc::require(video_noise != noise_arrays.end() &&
                        audio_noise != noise_arrays.end(),
                    "noise fixture must contain video_noise and audio_noise");

        auto tags_tensor = tc::mx::contiguous(
            tc::mx::astype(token_tags->second, tc::mx::int32));
        tc::mx::eval(tags_tensor);
        std::vector<int32_t> tags(tags_tensor.data<int32_t>(),
                                  tags_tensor.data<int32_t>() + tags_tensor.size());

        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::h3_mlx::Pipeline pipeline;
        pipeline.load(checkpoint_root, event, cancelled);
        tc::h3_mlx::DenoiseOptions options;
        options.width = probe.width;
        options.height = probe.height;
        options.frames = probe.frames;
        options.fps = 24;
        options.steps = 4;
        options.seed = 2026;
        options.affine_dq_gemm_min_rows = 768;
        options.video_noise = tc::mx::astype(video_noise->second, tc::mx::float32);
        options.audio_noise = tc::mx::astype(audio_noise->second, tc::mx::float32);
        options.vsa.enabled = true;
        options.vsa.gate_compress = probe.gate_compress;
        options.capture_debug = probe.capture_debug;
        options.vsa.sparsity = probe.sparsity;
        options.vsa.tile_size = probe.tile_size;
        options.vsa.prefix_mode = probe.prefix_mode;
        options.vsa.implementation = probe.implementation;
        auto result = pipeline.denoise(
            tc::mx::astype(conditioning->second, tc::mx::float32), tags,
            options, event, cancelled);
        tc::mx::eval(result.video_rows, result.audio_rows);
        tc::require(result.metrics.vsa.has_value(),
                    "H3 VSA E2E probe did not return VSA metrics");
        std::unordered_map<std::string, tc::Tensor> tensors{
            {"conditioning", tc::mx::astype(conditioning->second, tc::mx::float32)},
            {"token_tags", tags_tensor},
            {"video_rows", tc::mx::astype(result.video_rows, tc::mx::float32)},
            {"audio_rows", tc::mx::astype(result.audio_rows, tc::mx::float32)}};
        if (probe.capture_debug) {
            tc::require(result.debug.first_query.has_value() &&
                            result.debug.first_key.has_value() &&
                            result.debug.first_value.has_value() &&
                            result.debug.first_gate.has_value(),
                        "VSA debug capture did not produce Q/K/V/gate");
            tensors.emplace("debug_first_query", *result.debug.first_query);
            tensors.emplace("debug_first_key", *result.debug.first_key);
            tensors.emplace("debug_first_value", *result.debug.first_value);
            tensors.emplace("debug_first_gate", *result.debug.first_gate);
            tensors.emplace("debug_first_attention", *result.debug.first_attention);
            tensors.emplace("debug_first_attended", *result.debug.first_attended);
            tensors.emplace("debug_first_modulated1", *result.debug.first_modulated1);
            const auto &vsa_debug = *result.metrics.vsa;
            tc::require(vsa_debug.debug_captured && vsa_debug.debug_scores &&
                            vsa_debug.debug_block_indices && vsa_debug.debug_q_pool &&
                            vsa_debug.debug_k_pool,
                        "H3 VSA debug routing tensors were not captured");
            tensors.emplace("debug_vsa_q_pool", *vsa_debug.debug_q_pool);
            tensors.emplace("debug_vsa_k_pool", *vsa_debug.debug_k_pool);
            tensors.emplace("debug_vsa_scores", *vsa_debug.debug_scores);
            tensors.emplace("debug_vsa_block_indices", *vsa_debug.debug_block_indices);
        }
        tc::mx::save_safetensors(std::filesystem::absolute(argv[4]).string(), tensors);
        const auto &vsa = *result.metrics.vsa;
        std::cout << "{\"denoise_seconds\":" << result.metrics.denoise_seconds
                  << ",\"peak_bytes\":" << result.metrics.peak_bytes
                  << ",\"qmm_calls\":" << result.metrics.quantized_matmul_calls
                  << ",\"dq_gemm_calls\":" << result.metrics.dequantized_gemm_calls
                  << ",\"attention_calls\":" << vsa.attention_calls
                  << ",\"sparse_calls\":" << vsa.sparse_calls
                  << ",\"video_keep\":" << vsa.video_keep
                  << ",\"achieved_sparsity\":" << vsa.achieved_sparsity
                  << ",\"tile_size\":" << vsa.tile_size
                  << ",\"prefix_mode\":\"" << vsa.prefix_mode << "\""
                  << ",\"implementation\":\"" << vsa.implementation
                  << "\"}\n";
        pipeline.unload();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

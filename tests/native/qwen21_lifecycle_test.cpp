#include "../../native/models/qwen21/transformer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <unistd.h>

namespace {
using namespace tc;

constexpr int layers = 3, hidden = 64, channels = 16, context = 32, ffn = 96;
constexpr int variants = 8, repetitions = 24;
constexpr size_t idle_tolerance = 4096;

struct Fixture {
    std::filesystem::path directory;
    Fixture() {
        char pattern[] = "/tmp/turbocider-qwen21-lifecycle-XXXXXX";
        const auto *created = mkdtemp(pattern);
        require(created != nullptr, "cannot create Qwen21 lifecycle fixture");
        directory = created;
    }
    ~Fixture() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
};

qwen21::TransformerConfig config() {
    qwen21::TransformerConfig result;
    result.layers = layers;
    result.heads = 2;
    result.head_dim = 32;
    result.channels = channels;
    result.context_dim = context;
    result.rope_axes = {4, 12, 16};
    return result;
}

Tensor matrix(int out, int in, int seed) {
    return mx::astype(mx::random::normal({out, in}, mx::float32,
                     mx::random::key(seed)) / std::sqrt(float(in)), mx::bfloat16);
}

void bind_new_weights(Weights &weights) {
    std::vector<std::string> names;
    std::vector<Tensor> values;
    int seed = 1900;
    auto add_matrix = [&](const std::string &name, int out, int in) {
        names.push_back(name + ".weight");
        values.push_back(matrix(out, in, seed++));
    };
    auto add_vector = [&](const std::string &name, int width, float value) {
        names.push_back(name + ".weight");
        values.push_back(mx::full({width}, value, mx::bfloat16));
    };
    add_matrix("time_text_embed.timestep_embedder.linear_1", hidden, 256);
    add_matrix("time_text_embed.timestep_embedder.linear_2", hidden, hidden);
    add_matrix("modulation.1", 4 * hidden, hidden);
    add_matrix("img_in", hidden, channels);
    add_vector("txt_in.text_norm", context, 0.f);
    add_matrix("txt_in.in_layer", hidden, context);
    add_matrix("txt_in.out_layer", hidden, hidden);
    add_matrix("norm_out.linear", hidden, hidden);
    add_matrix("proj_out", channels, hidden);
    for (int i = 0; i < layers; ++i) {
        const auto p = "transformer_blocks." + std::to_string(i);
        for (const auto *projection : {"to_q", "to_k", "to_v", "to_out.0"})
            add_matrix(p + ".attn." + projection, hidden, hidden);
        add_vector(p + ".attn.norm_q", 32, 1.f);
        add_vector(p + ".attn.norm_k", 32, 1.f);
        add_matrix(p + ".img_mlp.gate_up", 2 * ffn, hidden);
        add_matrix(p + ".img_mlp.out", hidden, ffn);
    }
    weights.bind_arrays(names, values);
    // Materializing the captured weights before constructing the compiled
    // Transformer is essential to reproduce MLX's multi-output graph leak.
    weights.materialize();
}

void write_adapter(const std::filesystem::path &path) {
    std::unordered_map<std::string, Tensor> arrays;
    int seed = 2900;
    auto add = [&](const std::string &stem, int out, int in) {
        arrays.emplace(stem + ".lora_A.weight", matrix(4, in, seed++) * Tensor(.1f, mx::bfloat16));
        arrays.emplace(stem + ".lora_B.weight", matrix(out, 4, seed++) * Tensor(.1f, mx::bfloat16));
    };
    for (int i = 0; i < layers; ++i) {
        const auto p = "transformer_blocks." + std::to_string(i);
        add(p + ".attn.to_q", hidden, hidden);
        // The real loader maps these independent branches into gate_up.
        add(p + ".img_mlp.gate_layer", ffn, hidden);
        add(p + ".img_mlp.proj", ffn, hidden);
    }
    mx::save_safetensors(path.string(), arrays);
}

std::vector<float> host_values(const Tensor &value) {
    auto floats = mx::astype(value, mx::float32);
    mx::eval(floats);
    require(mx::all(mx::isfinite(floats)).item<bool>(),
            "tiny Qwen21 Transformer returned nonfinite output");
    const float *data = floats.data<float>();
    return {data, data + floats.size()};
}

struct Sample {
    std::vector<float> prefill, decode;
    size_t weight_bytes;
};

Sample run_cycle(int variant, const std::filesystem::path &adapter) {
    Weights weights;
    bind_new_weights(weights);
    const bool lora = variant % 2;
    if (lora) {
        std::atomic<bool> cancelled{false};
        const auto bound = weights.apply_loras(
            {{adapter.string(), 1.f, "transformer"}}, "transformer",
            [](const std::string &, int, int) {}, cancelled, true);
        require(bound == 3 * layers, "tiny runtime LoRA failed to bind");
    }
    // This is the production Transformer: full compiled blocks, ordinary
    // FP32 paired RoPE, BF16 base projections and the fused gate/up FFN.
    qwen21::Transformer transformer(weights, config());
    auto latents = mx::astype(mx::random::normal({1, 4, channels}, mx::float32,
                             mx::random::key(3900)), mx::bfloat16);
    auto text = mx::astype(mx::random::normal({1, 4, context}, mx::float32,
                          mx::random::key(3901)), mx::bfloat16);
    std::vector<qwen21::ReferenceLatents> references;
    for (int i = 0; i < variant / 2; ++i)
        references.push_back({mx::astype(mx::random::normal({1, 4, channels},
            mx::float32, mx::random::key(4000 + i)), mx::bfloat16), {2, 2, i + 1}});
    mx::eval(latents, text);
    auto prefill = transformer.forward(latents, text, 1.f, 2, 2, true, nullptr,
                                       references);
    auto first = host_values(prefill);
    require(transformer.cached_layers() == size_t(layers),
            "tiny prefill did not populate every prefix layer");
    auto decode = transformer.forward(latents, text, .5f, 2, 2, true, nullptr,
                                      references);
    auto second = host_values(decode);
    require(transformer.cached_layers() == size_t(layers),
            "tiny decode lost its prefix layer bank");
    require(prefill.shape() == mx::Shape{1, 4, channels} &&
                decode.shape() == prefill.shape(), "tiny output geometry changed");
    return {std::move(first), std::move(second), weights.bytes()};
}

size_t idle_active() {
    // Read only after all request objects and compiled wrappers have died.
    // Clear the allocator's idle cache, never the compiler cache: a regression
    // must fail even when its unreachable graph buffers cannot be reclaimed.
    mx::synchronize();
    mx::clear_cache();
    require(mx::get_cache_memory() == 0, "allocator idle cache was not cleared");
    return mx::get_active_memory();
}
} // namespace

int main() {
    try {
        require(std::getenv("MLX_DISABLE_COMPILE") == nullptr,
                "lifecycle regression must run with MLX compilation enabled");
        tc::configure_streams();
        Fixture fixture;
        const auto adapter = fixture.directory / "tiny-lora.safetensors";
        write_adapter(adapter);
        std::array<Sample, variants> expected;
        // Prime static activation kernels and all 0/1/2/3-reference + LoRA
        // shapes before recording the idle baseline. Every visit owns a new
        // weight allocation and a new compiled Transformer.
        for (int variant = 0; variant < variants; ++variant)
            expected[variant] = run_cycle(variant, adapter);
        const auto baseline = idle_active();
        size_t largest = baseline, first_weight_bytes = expected[0].weight_bytes;
        std::vector<size_t> retained;
        for (int i = 0; i < repetitions; ++i) {
            const int variant = i % variants;
            auto sample = run_cycle(variant, adapter);
            require(sample.prefill == expected[variant].prefill &&
                        sample.decode == expected[variant].decode,
                    "fresh compiled Transformer changed deterministic output");
            const auto active = idle_active();
            retained.push_back(active);
            largest = std::max(largest, active);
            require(active <= baseline + idle_tolerance,
                    "compiled Transformer retained live buffers after destruction: baseline=" +
                    std::to_string(baseline) + " active=" + std::to_string(active) +
                    " cycle=" + std::to_string(i));
        }
        const auto minimum = *std::min_element(retained.begin(), retained.end());
        require(largest - minimum <= idle_tolerance,
                "compiled Transformer idle active memory accumulates");
        std::cout << "{\"cycles\":" << repetitions
                  << ",\"variants\":" << variants
                  << ",\"baseline_active_bytes\":" << baseline
                  << ",\"maximum_active_bytes\":" << largest
                  << ",\"final_active_bytes\":" << retained.back()
                  << ",\"weight_bytes_per_base_cycle\":" << first_weight_bytes
                  << ",\"idle_tolerance_bytes\":" << idle_tolerance
                  << ",\"deterministic_prefill_decode\":true}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#include "../../native/models/qwen21/transformer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
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

void bind_variant(Weights &weights, int variant, const std::filesystem::path &adapter) {
    bind_new_weights(weights);
    if (variant % 2) {
        std::atomic<bool> cancelled{false};
        const auto bound = weights.apply_loras(
            {{adapter.string(), 1.f, "transformer"}}, "transformer",
            [](const std::string &, int, int) {}, cancelled, true);
        require(bound == 3 * layers, "tiny runtime LoRA failed to bind");
    }
}

Sample run_cycle(int variant, const std::filesystem::path &adapter) {
    Weights weights;
    bind_variant(weights, variant, adapter);
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

void run_snapshot_cycle(int variant, const std::filesystem::path &adapter, size_t baseline) {
    auto latents = mx::astype(mx::random::normal({1, 4, channels}, mx::float32,
                             mx::random::key(3900)), mx::bfloat16);
    auto text = mx::astype(mx::random::normal({1, 4, context}, mx::float32,
                          mx::random::key(3901)), mx::bfloat16);
    std::vector<qwen21::ReferenceLatents> references;
    for (int i = 0; i < variant / 2; ++i)
        references.push_back({mx::astype(mx::random::normal({1, 4, channels},
            mx::float32, mx::random::key(4000 + i)), mx::bfloat16), {2, 2, i + 1}});
    mx::eval(latents, text);
    std::optional<qwen21::Transformer::PrefixSnapshot> snapshot;
    std::vector<float> first, second;
    {
        Weights weights;
        bind_variant(weights, variant, adapter);
        qwen21::Transformer source(weights, config());
        require(!source.export_prefix_snapshot(std::numeric_limits<uint64_t>::max()),
                "empty Transformer exported a prefix");
        first = host_values(source.forward(latents, text, 1.f, 2, 2, true, nullptr, references));
        snapshot = source.export_prefix_snapshot(std::numeric_limits<uint64_t>::max());
        require(snapshot.has_value(), "completed tiny prefill did not export a snapshot");
        require(!source.export_prefix_snapshot(snapshot->bytes - 1),
                "snapshot exceeded its byte budget");
        for (const auto *bank : {&snapshot->keys, &snapshot->values})
            for (const auto &value : *bank)
                require(value.is_available() && !value.has_primitive() &&
                            value.inputs().empty() && value.siblings().empty() &&
                            value.flags().row_contiguous && value.data_size() == value.size(),
                        "snapshot contains lazy graphs or noncompact K/V buffers");
        second = host_values(source.forward(latents, text, .5f, 2, 2, true, nullptr, references));
    }
    // Source Transformer and materialized weights are gone; a snapshot may
    // retain only the compact K/V and the small immutable condition owners.
    size_t retained_bytes = snapshot->bytes + text.nbytes() + latents.nbytes();
    for (const auto &reference : references) retained_bytes += reference.latents.nbytes();
    require(idle_active() <= baseline + retained_bytes + idle_tolerance,
            "prefix snapshot retained source weights or full target backing");
    {
        Weights weights;
        bind_variant(weights, variant, adapter);
        qwen21::Transformer restored(weights, config());
        auto rejected = [&](const auto &candidate, const Tensor &condition,
                            int height, int width, const auto &refs) {
            require(!restored.import_prefix_snapshot(candidate, condition, height, width, refs),
                    "invalid prefix snapshot was accepted");
            require(restored.cached_layers() == 0,
                    "failed snapshot import changed the receiver");
        };
        rejected(*snapshot, mx::copy(text), 2, 2, references);
        rejected(*snapshot, text, 1, 4, references);
        auto bad = *snapshot;
        bad.bytes += 1;
        rejected(bad, text, 2, 2, references);
        bad = *snapshot; bad.config.epsilon *= 2.f;
        rejected(bad, text, 2, 2, references);
        bad = *snapshot; bad.keys.pop_back();
        rejected(bad, text, 2, 2, references);
        bad = *snapshot; bad.keys[0] = mx::copy(snapshot->keys[0]);
        rejected(bad, text, 2, 2, references);
        bad = *snapshot;
        bad.keys[0] = mx::astype(snapshot->keys[0], mx::float32);
        rejected(bad, text, 2, 2, references);
        if (!references.empty()) {
            auto changed = references;
            changed[0].latents = mx::copy(changed[0].latents);
            rejected(*snapshot, text, 2, 2, changed);
            changed = references; changed[0].geometry.text_slot += 1;
            rejected(*snapshot, text, 2, 2, changed);
        }
        if (references.size() >= 2) {
            auto reordered = references;
            std::reverse(reordered.begin(), reordered.end());
            rejected(*snapshot, text, 2, 2, reordered);
        }
        require(restored.import_prefix_snapshot(*snapshot, text, 2, 2, references),
                "new Transformer rejected matching immutable conditions");
        require(restored.cached_layers() == size_t(layers), "snapshot import lost layers");
        require(host_values(restored.forward(latents, text, 1.f, 2, 2, true, nullptr, references)) == first,
                "snapshot first-step target-only output differs from uncached prefill");
        require(host_values(restored.forward(latents, text, .5f, 2, 2, true, nullptr, references)) == second,
                "snapshot decode differs after source Transformer destruction");
        auto changed_seed = mx::astype(mx::random::normal({1, 4, channels}, mx::float32,
                                      mx::random::key(4900)), mx::bfloat16);
        qwen21::Transformer oracle(weights, config());
        auto expected = host_values(oracle.forward(changed_seed, text, 1.f, 2, 2,
                                                  true, nullptr, references));
        auto actual = host_values(restored.forward(changed_seed, text, 1.f, 2, 2,
                                                  true, nullptr, references));
        require(actual == expected, "prefix snapshot reused target noise from an earlier seed");
    }
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
            run_snapshot_cycle(variant, adapter, baseline);
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
                  << ",\"deterministic_prefill_decode\":true"
                  << ",\"snapshot_cycles\":" << repetitions
                  << ",\"snapshot_cross_transformer_parity\":true}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

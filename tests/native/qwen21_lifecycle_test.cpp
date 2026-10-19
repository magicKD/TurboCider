#include "../../native/models/qwen21/transformer.hpp"
#include "../../native/models/qwen21/diagnostic_options.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <utility>
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

qwen21::TransformerConfig config(int depth = layers) {
    qwen21::TransformerConfig result;
    result.layers = depth;
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

void bind_new_weights(Weights &weights, int depth = layers) {
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
    for (int i = 0; i < depth; ++i) {
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
    require(transformer.block_graph_builds() == 0 && transformer.block_graph_traces() == 0,
            "ordinary Transformer unexpectedly enabled graph diagnostic counters");
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
        if(variant==0) {
            const std::string recipe="explicit-runtime-arithmetic-fixture-v1";
            auto tagged=source.export_prefix_snapshot(std::numeric_limits<uint64_t>::max(),false,false,false,recipe);
            require(tagged && tagged->runtime_recipe==recipe,"explicit runtime snapshot lost its recipe");
            qwen21::Transformer receiver(weights,config());
            require(!receiver.import_prefix_snapshot(*tagged,text,2,2,references) && receiver.cached_layers()==0,
                    "ordinary GPU imported runtime-tagged K/V");
            require(!receiver.import_prefix_snapshot(*tagged,text,2,2,references,false,false,"different") && receiver.cached_layers()==0,
                    "wrong runtime recipe imported K/V or mutated the receiver");
            require(!receiver.import_prefix_snapshot(*snapshot,text,2,2,references,false,false,recipe),
                    "runtime route imported an untagged GPU bank");
            require(receiver.import_prefix_snapshot(*tagged,text,2,2,references,false,false,recipe),
                    "matching explicit runtime recipe rejected complete K/V");
            require(host_values(receiver.forward(latents,text,1.f,2,2,true,nullptr,references))==first,
                    "recipe metadata altered a same-prefix GPU fixture");
            auto other_noise=mx::astype(mx::random::normal({1,4,channels},mx::float32,mx::random::key(5900)),mx::bfloat16);
            qwen21::Transformer independent(weights,config());
            (void)host_values(independent.forward(other_noise,text,.375f,2,2,true,nullptr,{}));
            auto independent_prefix=independent.export_prefix_snapshot(std::numeric_limits<uint64_t>::max(),false,false,false,recipe);
            require(independent_prefix && independent_prefix->references.empty() && independent_prefix->prefix_length==text.shape(1),
                "text-only snapshot geometry or ownership differs");
            for(size_t layer=0;layer<tagged->keys.size();++layer)
                require(host_values(independent_prefix->keys[layer])==host_values(tagged->keys[layer]) &&
                        host_values(independent_prefix->values[layer])==host_values(tagged->values[layer]),
                    "text-prefix K/V changed with target noise or target timestep");
            require(!receiver.import_prefix_snapshot(*tagged,mx::copy(text),2,2,{},false,false,recipe),
                "same-shaped different text identity imported stale K/V");
            source.set_decode_mlp([](int,const Tensor &x){return x;});
            require(!source.export_prefix_snapshot(std::numeric_limits<uint64_t>::max(),false,false,false,recipe),
                    "runtime tag bypassed a live callback ownership guard");
            source.set_decode_mlp({});
        }
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

void run_db_snapshot_composition() {
    constexpr int depth = 10, steps = 40;
    Weights weights;
    bind_new_weights(weights, depth);
    weights.materialize();
    auto text = mx::reshape(matrix(4, context, 6401), {1, 4, context});
    auto latents = mx::reshape(matrix(4, channels, 6402), {1, 4, channels});
    std::vector<qwen21::ReferenceLatents> references{
        {mx::reshape(matrix(4, channels, 6403), {1, 4, channels}), {2, 2, 1}},
    };
    mx::eval(text, latents, references[0].latents);
    qwen21::Transformer source(weights, config(depth));
    source.configure_db_cache(true, .15f, steps, 1);
    for (int step = 0; step < steps; ++step) {
        source.set_db_cache_step(step);
        if (step == 1)
            source.set_decode_mlp([](int, const Tensor &input) { return mx::zeros_like(input); });
        (void)host_values(source.forward(latents, text, .5f, 2, 2, true, nullptr, references));
    }
    require(source.db_cached_steps() > 0, "snapshot source did not populate temporal residuals");
    require(!source.export_prefix_snapshot(std::numeric_limits<uint64_t>::max()),
            "DBCache snapshot composition became implicit");
    require(!source.export_prefix_snapshot(std::numeric_limits<uint64_t>::max(), true),
            "snapshot bypassed a live decode callback guard");
    source.set_decode_mlp({});
    auto snapshot = source.export_prefix_snapshot(std::numeric_limits<uint64_t>::max(), true);
    require(snapshot.has_value(), "explicit DB/snapshot composition did not export static K/V");
    qwen21::Transformer restored(weights, config(depth));
    require(restored.import_prefix_snapshot(*snapshot, text, 2, 2, references),
            "fresh request rejected DB-independent prefix snapshot");
    // Two new noise identities and new request-local DB configurations must
    // match fresh full-prefill oracles, including skip decisions at every step.
    for (int seed = 0; seed < 2; ++seed) {
        auto noise = mx::reshape(matrix(4, channels, 6500 + seed), {1, 4, channels});
        mx::eval(noise);
        qwen21::Transformer oracle(weights, config(depth));
        oracle.configure_db_cache(true, .15f, steps, 1);
        restored.configure_db_cache(true, .15f, steps, 1);
        require(restored.db_cached_steps() == 0, "new request inherited temporal skip count");
        for (int step = 0; step < steps; ++step) {
            oracle.set_db_cache_step(step);
            restored.set_db_cache_step(step);
            const auto expected = host_values(oracle.forward(noise, text, .5f, 2, 2, true, nullptr, references));
            const auto actual = host_values(restored.forward(noise, text, .5f, 2, 2, true, nullptr, references));
            require(actual == expected, "prefix snapshot imported an old target/residual at step " + std::to_string(step));
            require(restored.db_cached_steps() == oracle.db_cached_steps(),
                    "snapshot hit changed request-local DB skip policy");
        }
        require(restored.db_cached_steps() > 0, "restored request performed no temporal reuse");
    }
    restored.configure_db_cache(false);
    qwen21::Transformer ordinary(weights, config(depth));
    require(host_values(restored.forward(latents, text, .5f, 2, 2, true, nullptr, references)) ==
            host_values(ordinary.forward(latents, text, .5f, 2, 2, true, nullptr, references)),
            "turning DB off retained target residual from a previous request");
}

void run_graph_reuse_regression(const std::filesystem::path &adapter) {
    Weights weights;
    bind_variant(weights, 1, adapter);
    auto text = mx::reshape(matrix(4, context, 7100), {1, 4, context});
    auto next_text = mx::reshape(matrix(4, context, 7101), {1, 4, context});
    auto noise = mx::reshape(matrix(4, channels, 7102), {1, 4, channels});
    std::vector<qwen21::ReferenceLatents> refs{
        {mx::reshape(matrix(4, channels, 7103), {1, 4, channels}), {2, 2, 1}},
    };
    mx::eval(text, next_text, noise, refs[0].latents);
    qwen21::Transformer retained(weights, config());
    retained.enable_graph_counters();
    (void)host_values(retained.forward(noise, text, 1.f, 2, 2, true, nullptr, refs));
    (void)host_values(retained.forward(noise, text, .5f, 2, 2, true, nullptr, refs));
    const auto builds = retained.block_graph_builds(), traces = retained.block_graph_traces();
    require(builds > 0 && traces > 0, "graph instrumentation did not observe actual block tracing");
    retained.reset();
    require(retained.cached_layers() == 0 && retained.graph_geometry_matches(next_text, 2, 2, refs),
            "graph reset lost geometry or retained prefix values");
    qwen21::Transformer fresh(weights, config());
    for (float sigma : {1.f, .5f}) {
        const auto expected = host_values(fresh.forward(noise, next_text, sigma, 2, 2, true, nullptr, refs));
        const auto actual = host_values(retained.forward(noise, next_text, sigma, 2, 2, true, nullptr, refs));
        require(actual == expected, "retained graph captured previous text/input values");
    }
    require(retained.block_graph_builds() == builds && retained.block_graph_traces() == traces,
            "same-geometry fresh conditions retraced retained blocks");
    retained.reset();
    auto longer_text = mx::reshape(matrix(5, context, 7104), {1, 5, context});
    mx::eval(longer_text);
    require(!retained.graph_geometry_matches(longer_text, 2, 2, refs), "changed geometry reported graph hit");
    qwen21::Transformer longer_oracle(weights, config());
    require(host_values(retained.forward(noise, longer_text, 1.f, 2, 2, true, nullptr, refs)) ==
            host_values(longer_oracle.forward(noise, longer_text, 1.f, 2, 2, true, nullptr, refs)),
            "geometry change reused an invalid block graph");
    require(retained.block_graph_builds() > builds && retained.block_graph_traces() > traces,
            "geometry change failed to rebuild/retrace");
}

void run_db_cache_regression(const std::filesystem::path &adapter) {
    constexpr int depth = 10;
    Weights weights;
    bind_new_weights(weights, depth);
    std::atomic<bool> cancelled{false};
    require(weights.apply_loras({{adapter.string(), .75f, "transformer"}},
                "transformer", [](const std::string &, int, int) {}, cancelled, true) == 3 * layers,
            "DBCache tiny ordinary runtime adapter failed to bind");
    weights.materialize();
    auto latents = matrix(4, channels, 5900);
    latents = mx::reshape(latents, {1, 4, channels});
    auto text = mx::reshape(matrix(4, context, 5901), {1, 4, context});
    std::vector<qwen21::ReferenceLatents> references{
        {mx::reshape(matrix(4, channels, 5902), {1, 4, channels}), {2, 2, 1}},
    };
    mx::eval(latents, text, references[0].latents);
    qwen21::Transformer disabled(weights, config(depth)), named_off(weights, config(depth));
    Request off;
    off.qwen21_dit_cache_explicit = true;
    const auto options = qwen21::db_cache_options(off);
    require(!options.enabled, "explicit cache Off failed to disable diagnostics");
    named_off.configure_db_cache(options.enabled, options.threshold, 25, options.max_consecutive);
    const auto prefill = host_values(disabled.forward(latents, text, .5f, 2, 2, true, nullptr, references));
    require(host_values(named_off.forward(latents, text, .5f, 2, 2, true, nullptr, references)) == prefill,
            "named cache off changed the prefill output");
    const auto oracle = host_values(disabled.forward(latents, text, .5f, 2, 2, true, nullptr, references));
    require(host_values(named_off.forward(latents, text, .5f, 2, 2, true, nullptr, references)) == oracle,
            "named cache off changed the decode output");
    // A fixed input/sigma makes every front residual identical. Exercise real
    // 25/40-step forwards, checking both the numerical full steps and policy.
    for (const auto &[front, warmup] : {std::pair{8, 8}, std::pair{1, 4}}) {
        for (int steps : {25, 40}) {
            for (int maximum : {1, 2, 3, 4}) {
                qwen21::Transformer cached(weights, config(depth));
                cached.configure_db_cache(true, front == 1 ? .24f : .25f, steps, maximum, front, warmup);
                require(cached.db_cache_front_blocks() == front &&
                            cached.db_cache_warmup_steps() == warmup,
                        "DBCache did not retain the selected front/warmup policy");
                int consecutive = 0, total = 0;
                for (int step = 0; step < steps; ++step) {
                    cached.set_db_cache_step(step);
                    const int before = cached.db_cached_steps();
                    const auto actual = host_values(cached.forward(latents, text, .5f,
                                                                   2, 2, true, nullptr, references));
                    const bool skipped = cached.db_cached_steps() > before;
                    const bool eligible = step >= warmup && step < steps - 1 && consecutive < maximum;
                    require(skipped == eligible, "DBCache warmup/final/consecutive policy failed at step " +
                            std::to_string(step));
                    if (skipped) { ++consecutive; ++total; }
                    else {
                        consecutive = 0;
                        require(actual == (step == 0 ? prefill : oracle),
                                "DBCache full step differs from uncached ordinary runtime LoRA");
                    }
                }
                require(cached.db_cached_steps() == total && total > 0,
                        "DBCache request-local skip accounting failed");
                require(!cached.export_prefix_snapshot(std::numeric_limits<uint64_t>::max()),
                        "DBCache exported a prefix bank on its excluded route");
            }
        }
    }
    // Count the executed blocks through the production split callback. This
    // proves that a skip bypasses middle computation, rather than only raising
    // a receipt counter. No Core ML runtime or large model is involved.
    auto check_invalidation = [&](int front, int warmup) {
        qwen21::Transformer counted(weights, config(depth));
        int calls = 0;
        bool fail_middle = false;
        counted.set_decode_mlp([&](int block, const Tensor &input) {
            ++calls;
            if (fail_middle && block == front) throw Cancelled();
            return mx::zeros(input.shape(), input.dtype());
        });
        counted.configure_db_cache(true, .25f, 25, 2, front, warmup);
        auto forward = [&](int step, const Tensor &condition,
                           std::unordered_map<std::string, Tensor> *trace = nullptr) {
            counted.set_db_cache_step(step);
            calls = 0;
            return host_values(counted.forward(latents, condition, .5f, 2, 2, true, trace, references));
        };
        forward(0, text);
        for (int step = 1; step <= warmup; ++step) {
            forward(step, text);
            require(calls == (step < warmup ? depth : front), "DBCache did not skip the actual middle callback");
        }
        forward(warmup + 1, text);
        require(calls == front && counted.db_cached_steps() == 2, "DBCache consecutive reuse lost prior residual");
        forward(12, text); // Gap: a stale middle residual must not be reused.
        require(calls == depth && counted.db_cached_steps() == 2, "DBCache reused a noncontiguous step");
        forward(13, text);
        require(calls == front && counted.db_cached_steps() == 3, "DBCache did not resume after a fresh full step");
        forward(13, text); // Repeated step is another discontinuity.
        require(calls == depth && counted.db_cached_steps() == 3, "DBCache reused a repeated step");
        forward(0, text); // New request, matching prefix may remain but residuals cannot.
        require(calls == depth && counted.db_cached_steps() == 0, "DBCache crossed a request restart");
        for (int step = 1; step <= warmup; ++step) forward(step, text);
        require(counted.db_cached_steps() == 1, "DBCache restart did not reset warmup/counts");
        auto new_text = mx::copy(text);
        mx::eval(new_text);
        forward(warmup + 1, new_text); // Same values/shape, distinct conditioning identity.
        require(calls == 0 && counted.db_cached_steps() == 0,
                "DBCache reused residuals after conditioning identity changed");
        forward(warmup + 2, new_text);
        require(calls == depth && counted.db_cached_steps() == 0,
                "DBCache used first-step prefix output as a middle residual");
        fail_middle = true;
        bool caught = false;
        try { forward(14, new_text); } // Gap forces full execution and callback failure.
        catch (const Cancelled &) { caught = true; }
        require(caught && counted.db_cached_steps() == 0,
                "DBCache callback cancellation retained request counters");
        fail_middle = false;
        forward(14, new_text);
        require(calls == depth && counted.db_cached_steps() == 0,
                "DBCache cancellation retry reused partial front state");

        // Tracing runs the ordinary GPU graph and must invalidate split-path
        // residuals before returning to the callback route.
        counted.set_decode_mlp({});
        std::unordered_map<std::string, Tensor> trace;
        forward(15, new_text, &trace);
        require(!trace.empty() && calls == 0, "DBCache tiny trace did not use the ordinary graph");
        counted.set_decode_mlp([&](int, const Tensor &input) {
            ++calls; return mx::zeros(input.shape(), input.dtype());
        });
        forward(16, new_text);
        require(calls == depth && counted.db_cached_steps() == 0,
                "DBCache reused residuals across a traced forward");
    };
    check_invalidation(8, 8);
    check_invalidation(1, 4);
}

void run_db_cache_reference_shapes(const std::filesystem::path &adapter) {
    constexpr int depth = 10;
    for (bool use_lora : {false, true}) {
        Weights weights;
        bind_new_weights(weights, depth);
        if (use_lora) {
            std::atomic<bool> cancelled{false};
            require(weights.apply_loras({{adapter.string(), .75f, "transformer"}},
                        "transformer", [](const std::string &, int, int) {}, cancelled, true) == 3 * layers,
                    "resized-reference ordinary runtime adapter failed to bind");
        }
        weights.materialize();
        auto latents = mx::reshape(matrix(4, channels, 6900), {1, 4, channels});
        auto text = mx::reshape(matrix(4, context, 6901), {1, 4, context});
        for (int count : {1, 2, 3}) {
            // A scaled-down 512/1024 pair preserves the fourfold reference
            // token ratio while exercising the actual compiled Transformer.
            std::vector<qwen21::ReferenceLatents> small, large;
            for (int i = 0; i < count; ++i) {
                small.push_back({mx::reshape(matrix(4, channels, 6910 + i),
                                             {1, 4, channels}), {2, 2, i + 1}});
                large.push_back({mx::reshape(matrix(16, channels, 6920 + i),
                                             {1, 16, channels}), {4, 4, i + 1}});
                mx::eval(small.back().latents, large.back().latents);
            }
            for (const char *mode : {"conservative", "balanced", "fast"}) {
                Request request;
                request.qwen21_dit_cache = mode;
                const auto options = qwen21::db_cache_options(request);
                qwen21::Transformer cached(weights, config(depth)), oracle(weights, config(depth));
                cached.configure_db_cache(options.enabled, options.threshold, 25,
                                          options.max_consecutive, options.front_blocks, options.warmup_steps);
                for (int step = 0; step <= options.warmup_steps; ++step) {
                    cached.set_db_cache_step(step);
                    const auto expected = host_values(oracle.forward(latents, text, .5f,
                                                                      2, 2, true, nullptr, small));
                    const auto actual = host_values(cached.forward(latents, text, .5f,
                                                                   2, 2, true, nullptr, small));
                    require(cached.db_cached_steps() == (step == options.warmup_steps ? 1 : 0),
                            "resized-reference DiT preset warmup/skip policy changed");
                    require(actual.size() == size_t(4 * channels), "reference resize changed target output shape");
                    if (step < options.warmup_steps)
                        require(actual == expected, "resized-reference DiT full step differs from exact base/LoRA");
                }
                auto verify_reset = [&](int step, const auto &references) {
                    cached.set_db_cache_step(step);
                    const auto actual = host_values(cached.forward(latents, text, .5f,
                                                                   2, 2, true, nullptr, references));
                    qwen21::Transformer fresh(weights, config(depth));
                    const auto expected = host_values(fresh.forward(latents, text, .5f,
                                                                    2, 2, true, nullptr, references));
                    require(cached.db_cached_steps() == 0 && actual == expected,
                            "DiT residual/prefix survived changed reference geometry, content or order");
                };
                verify_reset(options.warmup_steps + 1, large);
                verify_reset(options.warmup_steps + 2, small);
                auto changed = small;
                changed[0].latents = mx::copy(small[0].latents);
                mx::eval(changed[0].latents);
                verify_reset(options.warmup_steps + 3, changed);
                if (count > 1) {
                    std::reverse(changed.begin(), changed.end());
                    // Reordered image contents still occupy image1/image2/...
                    // in ascending prompt slots, as a real reordered request.
                    for (int i = 0; i < count; ++i) changed[i].geometry.text_slot = i + 1;
                    verify_reset(options.warmup_steps + 4, changed);
                }
            }
        }
    }
}
void run_joint_qkv_mlp_regression() {
    Weights weights;bind_new_weights(weights);
    auto latents=mx::astype(mx::random::normal({1,4,channels},mx::float32,mx::random::key(8700)),mx::bfloat16);
    auto text=mx::astype(mx::random::normal({1,4,context},mx::float32,mx::random::key(8701)),mx::bfloat16);
    mx::eval(latents,text);
    auto qkv=mx::compile([](const std::vector<Tensor>&a){return std::vector<Tensor>{mx::concatenate({
        mx::matmul(a[0],mx::transpose(a[1])),mx::matmul(a[0],mx::transpose(a[2])),
        mx::matmul(a[0],mx::transpose(a[3]))},-1)};});
    auto mlp=mx::compile([](const std::vector<Tensor>&a){
        auto pair=mx::matmul(a[0],mx::transpose(a[1]));
        auto feed=silu(slice_axis(pair,-1,0,ffn))*slice_axis(pair,-1,ffn,2*ffn);
        return std::vector<Tensor>{mx::matmul(feed,mx::transpose(a[2]))};});
    for(int refs=0;refs<=2;++refs)for(bool external_prefill:{false,true}){
        std::vector<qwen21::ReferenceLatents> references;
        for(int i=0;i<refs;++i)references.push_back({mx::copy(latents),{2,2,i+1}});
        qwen21::Transformer original(weights,config());
        const auto first=host_values(original.forward(latents,text,1.f,2,2,true,nullptr,references));
        const auto second=host_values(original.forward(latents,text,.5f,2,2,true,nullptr,references));
        qwen21::Transformer candidate(weights,config());
        int stages=0,projections=0,feeds=0;
        candidate.set_stage_qkv([&](int,int){++stages;});
        candidate.set_project_qkv([&](int layer,const Tensor&x){
            ++projections;const auto p="transformer_blocks."+std::to_string(layer)+".attn.to_";
            auto y=qkv({x,weights.at(p+"q.weight"),weights.at(p+"k.weight"),weights.at(p+"v.weight")})[0];
            mx::eval(y);return y;});
        candidate.set_plan_qkv([&](int,int rows){return external_prefill||rows==4?
            qwen21::Transformer::QKVPlan::Hybrid:qwen21::Transformer::QKVPlan::Gpu;});
        candidate.set_observe_qkv([](int,int,double){});
        auto feed=[&](int layer,const Tensor&x){++feeds;const auto p="transformer_blocks."+std::to_string(layer)+".img_mlp.";
            auto y=mlp({x,weights.at(p+"gate_up.weight"),weights.at(p+"out.weight")})[0];mx::eval(y);return y;};
        candidate.set_prefill_mlp(feed);candidate.set_decode_mlp(feed);
        candidate.set_plan_mlp([](int,int){return qwen21::Transformer::MLPPlan::SplitUntimed;});
        candidate.set_observe_mlp([](int,int,double){});
        bool rejected=false;try{(void)candidate.forward(latents,text,1.f,2,2,true,nullptr,references);}
            catch(const std::exception&){rejected=true;}
        require(rejected&&stages==0&&feeds==0,"joint callbacks bypassed default guard or submitted work before consent");
        candidate.set_joint_runtime_qkv_mlp(true);
        auto check=[&](const auto&actual,const auto&expected){
            require(actual.size()==expected.size(),"joint callback output geometry changed");
            double error=0,energy=0;for(size_t i=0;i<actual.size();++i){error+=std::pow(actual[i]-expected[i],2);energy+=std::pow(expected[i],2);}
            require(std::sqrt(error/std::max(energy,1e-30))<.01,"GPU-only joint callback algebra disagrees with ordinary block");};
        check(host_values(candidate.forward(latents,text,1.f,2,2,true,nullptr,references)),first);
        check(host_values(candidate.forward(latents,text,.5f,2,2,true,nullptr,references)),second);
        require(feeds==2*layers&&projections==(external_prefill?2:1)*layers&&stages==projections,
            "joint callback variant/prefill/decode counts changed");
        require(!candidate.export_prefix_snapshot(UINT64_MAX),"joint callback exported a live-owned prefix");
        candidate.set_project_qkv({});candidate.set_stage_qkv({});candidate.set_plan_qkv({});candidate.set_observe_qkv({});
        candidate.set_prefill_mlp({});candidate.set_decode_mlp({});candidate.set_plan_mlp({});candidate.set_observe_mlp({});
        require(!candidate.export_prefix_snapshot(UINT64_MAX),"joint consent bypassed prefix ownership guard");
        candidate.set_joint_runtime_qkv_mlp(false);
        require(bool(candidate.export_prefix_snapshot(UINT64_MAX)),"drained joint GPU fixture cannot export independent K/V");
    }
    {
        // A primed text prefix models an imported compact bank. Exercise
        // decode slot1 (GPU QKV + split FFN), then slot3 (external QKV +
        // split FFN) at the SAME target shape; plan changes must not reuse
        // a stale compiled-block route or omit successful consumers.
        qwen21::Transformer original(weights,config()),candidate(weights,config());
        (void)host_values(original.forward(latents,text,1.f,2,2,true));
        (void)host_values(candidate.forward(latents,text,1.f,2,2,true));
        int step=0,stages=0,projections=0,feeds=0;
        candidate.set_joint_runtime_qkv_mlp(true);
        candidate.set_plan_qkv([&](int,int){return step==0?qwen21::Transformer::QKVPlan::Gpu:qwen21::Transformer::QKVPlan::Hybrid;});
        candidate.set_stage_qkv([&](int,int){++stages;});candidate.set_observe_qkv([](int,int,double){});
        candidate.set_project_qkv([&](int layer,const Tensor &x){++projections;const auto p="transformer_blocks."+std::to_string(layer)+".attn.to_";
            auto y=qkv({x,weights.at(p+"q.weight"),weights.at(p+"k.weight"),weights.at(p+"v.weight")})[0];mx::eval(y);return y;});
        candidate.set_decode_mlp([&](int layer,const Tensor &x){++feeds;const auto p="transformer_blocks."+std::to_string(layer)+".img_mlp.";
            auto y=mlp({x,weights.at(p+"gate_up.weight"),weights.at(p+"out.weight")})[0];mx::eval(y);return y;});
        candidate.set_plan_mlp([](int,int){return qwen21::Transformer::MLPPlan::SplitUntimed;});candidate.set_observe_mlp([](int,int,double){});
        for(step=0;step<3;++step){const float sigma=1.f-.25f*step;
            auto expected=host_values(original.forward(latents,text,sigma,2,2,true));
            auto actual=host_values(candidate.forward(latents,text,sigma,2,2,true));
            double error=0,energy=0;for(size_t i=0;i<actual.size();++i){error+=std::pow(actual[i]-expected[i],2);energy+=std::pow(expected[i],2);}
            require(std::sqrt(error/std::max(energy,1e-30))<.01,"same-shape GPU-first QKV plan changed joint fixture algebra");
            require(stages==step*layers && projections==stages && feeds==(step+1)*layers,"GPU-first/remaining-private callback coverage differs");
        }
        candidate.set_project_qkv({});candidate.set_stage_qkv({});candidate.set_plan_qkv({});candidate.set_observe_qkv({});
        candidate.set_decode_mlp({});candidate.set_plan_mlp({});candidate.set_observe_mlp({});candidate.set_joint_runtime_qkv_mlp(false);
        require(bool(candidate.export_prefix_snapshot(UINT64_MAX)),"GPU-first joint route retained live callback ownership");
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
        run_joint_qkv_mlp_regression();
        run_db_cache_regression(adapter);
        run_db_snapshot_composition();
        run_graph_reuse_regression(adapter);
        run_db_cache_reference_shapes(adapter);
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
        run_db_cache_regression(adapter);
        run_graph_reuse_regression(adapter);
        run_db_cache_reference_shapes(adapter);
        run_joint_qkv_mlp_regression();
        require(idle_active() <= baseline + idle_tolerance,
                "Qwen21 diagnostics retained compiled weights or request buffers after destruction");
        std::cout << "{\"cycles\":" << repetitions
                  << ",\"variants\":" << variants
                  << ",\"baseline_active_bytes\":" << baseline
                  << ",\"maximum_active_bytes\":" << largest
                  << ",\"final_active_bytes\":" << retained.back()
                  << ",\"weight_bytes_per_base_cycle\":" << first_weight_bytes
                  << ",\"idle_tolerance_bytes\":" << idle_tolerance
                  << ",\"deterministic_prefill_decode\":true"
                  << ",\"snapshot_cycles\":" << repetitions
                  << ",\"snapshot_cross_transformer_parity\":true"
                  << ",\"text_prefix_seed_sigma_independent\":true"
                  << ",\"dbcache_prefix_snapshot_isolation\":true"
                  << ",\"retained_graph_fresh_values_and_geometry\":true"
                  << ",\"dbcache_policy_and_invalidation\":true"
                  << ",\"dbcache_reference_geometry_and_lora\":true"
                  << ",\"joint_qkv_mlp_explicit_consent_and_gpu_parity\":true}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

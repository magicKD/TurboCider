#include "../../native/models/qwen21/transformer.hpp"
#include <cassert>
#include <iostream>

// Exercise actual compiled Qwen blocks, not a mock cache: changing a runtime
// decision on a later visit must neither reuse the wrong output ABI nor omit
// an FFN. Tiny synthetic weights avoid full checkpoints and timing claims.
int main() {
    using namespace tc;
    qwen21::TransformerConfig config;
    config.layers = 3; config.heads = 2; config.head_dim = 32;
    config.channels = 16; config.context_dim = 32; config.rope_axes = {4, 12, 16};
    std::vector<std::string> names;
    std::vector<Tensor> values;
    int seed = 900;
    auto matrix = [&](const std::string &name, int out, int in) {
        names.push_back(name + ".weight");
        values.push_back(mx::random::normal({out, in}, mx::float32, mx::random::key(seed++)) /
                         std::sqrt(float(in)));
    };
    auto vector = [&](const std::string &name, int width, float value) {
        names.push_back(name + ".weight");
        values.push_back(mx::full({width}, value));
    };
    matrix("time_text_embed.timestep_embedder.linear_1", 64, 256);
    matrix("time_text_embed.timestep_embedder.linear_2", 64, 64);
    matrix("modulation.1", 256, 64);
    matrix("img_in", 64, 16);
    vector("txt_in.text_norm", 32, 0.f);
    matrix("txt_in.in_layer", 64, 32); matrix("txt_in.out_layer", 64, 64);
    matrix("norm_out.linear", 64, 64); matrix("proj_out", 16, 64);
    for (int i = 0; i < config.layers; ++i) {
        const auto p = "transformer_blocks." + std::to_string(i);
        for (const auto *name : {"to_q", "to_k", "to_v", "to_out.0"})
            matrix(p + ".attn." + name, 64, 64);
        vector(p + ".attn.norm_q", 32, 1.f); vector(p + ".attn.norm_k", 32, 1.f);
        matrix(p + ".img_mlp.gate_up", 192, 64); matrix(p + ".img_mlp.out", 64, 96);
    }
    Weights weights;
    weights.bind_arrays(names, values);
    weights.materialize();
    qwen21::Transformer ordinary(weights, config), routed(weights, config);
    auto latents = mx::random::normal({1, 4, 16}, mx::float32, mx::random::key(seed++));
    auto text = mx::random::normal({1, 4, 32}, mx::float32, mx::random::key(seed++));
    int visit = 0, planned = 0, untimed = 0, staged = 0, called = 0, probes = 0, observed = 0;
    int staged_layer = -1, staged_rows = 0;
    routed.set_plan_mlp([&](int layer, int rows) {
        assert(rows == 4 || rows == 8);
        switch ((visit + layer) % 4) {
            case 0: ++planned; return qwen21::Transformer::MLPPlan::Split;
            case 1: ++probes; return qwen21::Transformer::MLPPlan::GpuProbe;
            case 2: ++planned; ++untimed; return qwen21::Transformer::MLPPlan::SplitUntimed;
            default: return qwen21::Transformer::MLPPlan::Gpu;
        }
    });
    routed.set_observe_mlp([&](int layer, int rows, double seconds) {
        assert((visit + layer) % 4 < 2 && (rows == 4 || rows == 8));
        assert(staged_layer == -1 && std::isfinite(seconds) && seconds > 0);
        ++observed;
    });
    routed.set_stage_mlp([&](int layer, int rows) {
        assert(staged_layer == -1);
        staged_layer = layer; staged_rows = rows; ++staged;
    });
    auto ffn = [&](int layer, const Tensor &input) {
        assert(layer == staged_layer && input.shape(1) == staged_rows);
        staged_layer = -1; ++called;
        const auto p = "transformer_blocks." + std::to_string(layer) + ".img_mlp.";
        auto gu = mx::split(weights.project(input, p + "gate_up"), 2, -1);
        auto output = weights.project(silu(gu[0]) * gu[1], p + "out");
        mx::eval(output); // satisfy SplitUntimed's owned/evaluated contract
        return output;
    };
    routed.set_prefill_mlp(ffn); routed.set_decode_mlp(ffn);
    float worst = 0;
    // Six prefill visits populate both cache variants; then prefill+decode
    // exercise the prefix K/V output indexes and the separate decode caches.
    for (; visit < 13; ++visit) {
        const bool cache_prefix = visit >= 6;
        const float timestep = 1.f - float(visit) / 20.f;
        auto expected = ordinary.forward(latents, text, timestep, 2, 2, cache_prefix);
        auto actual = routed.forward(latents, text, timestep, 2, 2, cache_prefix);
        const float error = mx::sqrt(mx::sum(mx::square(actual - expected)) /
                                     mx::sum(mx::square(expected))).item<float>();
        assert(std::isfinite(error) && error < 2e-5f);
        worst = std::max(worst, error);
        assert(planned == staged && staged == called && staged_layer == -1);
        assert(observed == planned - untimed + probes);
        if (cache_prefix) assert(routed.cached_layers() == size_t(config.layers));
    }
    // Removing callbacks must recover ordinary full blocks on the same object.
    routed.set_plan_mlp({}); routed.set_stage_mlp({});
    routed.set_observe_mlp({});
    routed.set_prefill_mlp({}); routed.set_decode_mlp({});
    auto expected = ordinary.forward(latents, text, .1f, 2, 2, false);
    auto actual = routed.forward(latents, text, .1f, 2, 2, false);
    assert(mx::all(actual == expected).item<bool>());
    int qkv_staged = -1, qkv_calls = 0;
    auto enable_qkv = [&] {
        routed.set_stage_qkv([&](int layer, int rows) {
            assert(qkv_staged == -1 && (rows == 4 || rows == 8));
            qkv_staged = layer;
        });
        routed.set_project_qkv([&](int layer, const Tensor &input) {
            assert(layer == qkv_staged && (input.shape(1) == 4 || input.shape(1) == 8) &&
                   input.shape(2) == 64);
            qkv_staged = -1; ++qkv_calls;
            const auto p = "transformer_blocks." + std::to_string(layer) + ".attn.to_";
            auto output = mx::concatenate({weights.project(input, p + "q"),
                                           weights.project(input, p + "k"),
                                           weights.project(input, p + "v")}, -1);
            mx::eval(output); // no borrowed/stale GPU tensor across the next block
            return output;
        });
    };
    for (int qvisit = 0; qvisit < 8; ++qvisit) {
        if (qvisit % 3 == 2) {
            routed.set_stage_qkv({}); routed.set_project_qkv({});
        } else enable_qkv();
        const bool cache_prefix = qvisit >= 3;
        const float timestep = .9f - float(qvisit) / 20.f;
        expected = ordinary.forward(latents, text, timestep, 2, 2, cache_prefix);
        actual = routed.forward(latents, text, timestep, 2, 2, cache_prefix);
        const float error = mx::sqrt(mx::sum(mx::square(actual - expected)) /
                                     mx::sum(mx::square(expected))).item<float>();
        assert(std::isfinite(error) && error < 2e-5f);
        assert(qkv_staged == -1 && ordinary.cached_layers() == routed.cached_layers());
    }
    assert(qkv_calls == 18);
    routed.set_stage_qkv({}); routed.set_project_qkv({});
    int qkv_visit = 0, qkv_observed = 0, qkv_planned_hybrid = 0, qkv_planned_gpu = 0;
    enable_qkv();
    routed.set_plan_qkv([&](int, int rows) {
        assert(rows == 4 || rows == 8);
        if (qkv_visit % 4 == 0) { ++qkv_planned_hybrid; return qwen21::Transformer::QKVPlan::HybridTimed; }
        if (qkv_visit % 4 == 1) { ++qkv_planned_gpu; return qwen21::Transformer::QKVPlan::GpuProbe; }
        if (qkv_visit % 4 == 2) { ++qkv_planned_hybrid; return qwen21::Transformer::QKVPlan::Hybrid; }
        ++qkv_planned_gpu; return qwen21::Transformer::QKVPlan::Gpu;
    });
    routed.set_observe_qkv([&](int, int rows, double seconds) {
        assert((rows == 4 || rows == 8) && std::isfinite(seconds) && seconds > 0);
        ++qkv_observed;
    });
    const int qkv_before = qkv_calls;
    for (; qkv_visit < 8; ++qkv_visit) {
        const bool cache_prefix = qkv_visit >= 4;
        const float timestep = .8f - float(qkv_visit) / 20.f;
        expected = ordinary.forward(latents, text, timestep, 2, 2, cache_prefix);
        actual = routed.forward(latents, text, timestep, 2, 2, cache_prefix);
        const float error = mx::sqrt(mx::sum(mx::square(actual - expected)) /
                                     mx::sum(mx::square(expected))).item<float>();
        assert(std::isfinite(error) && error < 2e-5f);
        assert(qkv_staged == -1 && ordinary.cached_layers() == routed.cached_layers());
    }
    assert(qkv_planned_hybrid == 12 && qkv_planned_gpu == 12);
    assert(qkv_observed == 12 && qkv_calls == qkv_before + 12);
    routed.set_stage_qkv({}); routed.set_project_qkv({});
    routed.set_plan_qkv({}); routed.set_observe_qkv({});
    std::cout << "PASS Qwen full/split cache switching, prefill/decode and complete FFNs; relative L2="
              << worst << '\n';
}

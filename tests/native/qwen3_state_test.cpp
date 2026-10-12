#include "../../native/components/text/qwen3.hpp"
#include <iostream>

namespace tc::components {
Tensor qwen3_conditioning_reference(const Tokens &, const Weights &, const Qwen3Conditioning &,
                                    const Event &, std::atomic<bool> &, HybridSession *);
}
int main() {
    try {
        using namespace tc; configure_streams();
        Weights weights; std::vector<std::string> keys; std::vector<Tensor> values;
        int seed = 42;
        auto add = [&](const std::string &key, const mx::Shape &shape, bool norm = false) {
            keys.push_back(key);
            values.push_back(norm ? mx::ones(shape, mx::float32) :
                mx::astype(mx::random::normal(shape, mx::float32, mx::random::key(seed++)) * .05f, mx::bfloat16));
        };
        add("model.embed_tokens.weight", {32, 32});
        for (int layer = 0; layer < 3; ++layer) {
            const auto p = "model.layers." + std::to_string(layer) + ".";
            add(p + "input_layernorm.weight", {32}, true); add(p + "post_attention_layernorm.weight", {32}, true);
            add(p + "self_attn.q_proj.weight", {16, 32}); add(p + "self_attn.k_proj.weight", {8, 32});
            add(p + "self_attn.v_proj.weight", {8, 32}); add(p + "self_attn.o_proj.weight", {32, 16});
            add(p + "self_attn.q_norm.weight", {8}, true); add(p + "self_attn.k_norm.weight", {8}, true);
            add(p + "mlp.gate_proj.weight", {64, 32}); add(p + "mlp.up_proj.weight", {64, 32});
            add(p + "mlp.down_proj.weight", {32, 64});
        }
        mx::eval(values); weights.bind_arrays(keys, values);
        std::atomic<bool> cancel{false}; Event event = [](const auto &, int, int) {};
        for (bool fp32 : {false, true}) for (int count : {1, 33, 129}) for (bool padded : {false, true}) {
            Tokens tokens; tokens.valid = padded && count > 1 ? count - 3 : count;
            for (int i = 0; i < count; ++i) tokens.ids.push_back(i % 32);
            components::Qwen3Conditioning c; c.heads = 2; c.kv_heads = 1; c.head_dim = 8;
            c.output_layers = {0, 2}; c.float32_residual = fp32;
            const auto baseline = components::qwen3_conditioning_reference(tokens, weights, c, event, cancel, nullptr);
            const auto resident = components::qwen3_conditioning(tokens, weights, c, event, cancel);
            auto ids = Tensor(tokens.ids.data(), {1, count}, mx::int32);
            components::Qwen3ConditioningState state(tokens, mx::take(weights.at("model.embed_tokens.weight"), ids, 0), c);
            for (int layer = 0; layer < 3; ++layer) { state.advance(weights, event, cancel); state.materialize(); }
            auto submitted = state.finish(event); mx::eval(baseline, resident, submitted);
            require(baseline.shape() == resident.shape() && baseline.dtype() == resident.dtype(), "state changed resident output contract");
            require(mx::all(baseline == resident).item<bool>(), "state refactor differs from pinned pre-refactor resident math");
            require(mx::all(baseline == submitted).item<bool>(), "each-layer submission changed mathematical output");
            require(mx::all(mx::isfinite(submitted)).item<bool>(), "nonfinite state output");
        }
        std::cout << "PASS Qwen3 state: 12 pinned-reference, resident and submitted cases\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

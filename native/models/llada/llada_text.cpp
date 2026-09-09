#include "llada.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace tc {
namespace {

constexpr int kHidden = 2048;
constexpr int kAttentionHeads = 16;
constexpr int kKVHeads = 4;
constexpr int kHeadDim = 128;
constexpr int kRotaryDim = 64;
constexpr int kExperts = 256;
constexpr int kExpertsPerToken = 8;
constexpr int kExpertGroups = 8;
constexpr int kSelectedGroups = 4;
constexpr int kExpertsPerGroup = kExperts / kExpertGroups;

Tensor rms_plain(const Tensor &x, float eps) {
    return mx::astype(mx::fast::rms_norm(mx::astype(x, mx::float32), {}, eps),
                      x.dtype());
}

Tensor layer_norm_plain(const Tensor &x, float eps) {
    return mx::astype(mx::fast::layer_norm(mx::astype(x, mx::float32), {}, {}, eps),
                      x.dtype());
}

void dump_text_stage(const std::string &directory, const std::string &name,
                     const Tensor &value) {
    if (directory.empty())
        return;
    std::filesystem::create_directories(directory);
    mx::save_safetensors(
        (std::filesystem::path(directory) / ("llada_text_" + name + ".safetensors")).string(),
        {{"tensor", value}});
}

Tensor gelu_tanh(const Tensor &x) {
    constexpr float kScale = 0.7978845608028654f;
    auto f = mx::astype(x, mx::float32);
    auto y = Tensor(.5f, mx::float32) * f *
             (Tensor(1.f, mx::float32) +
              mx::tanh(Tensor(kScale, mx::float32) *
                       (f + Tensor(.044715f, mx::float32) * f * f * f)));
    return mx::astype(y, x.dtype());
}

Tensor linear_weight(const Tensor &x, const Tensor &weight,
                     const std::optional<Tensor> &bias = std::nullopt) {
    auto output = mx::matmul(x, mx::transpose(weight));
    if (bias)
        output = output + *bias;
    return output;
}

Tensor queryformer(const Tensor &inputs, const Weights &weights) {
    auto query = mx::expand_dims(weights.at("meta_queries"), 0);
    const auto prefix = std::string("query_blocks.0");
    auto normalized_query = layer_norm_plain(query, 1e-6f);
    auto normalized_inputs = layer_norm_plain(inputs, 1e-6f);
    const auto &projection = weights.at(prefix + ".cross_attn.in_proj_weight");
    const auto &projection_bias = weights.at(prefix + ".cross_attn.in_proj_bias");
    auto q = linear_weight(normalized_query,
                           slice_axis(projection, 0, 0, kHidden),
                           slice_axis(projection_bias, 0, 0, kHidden));
    auto k = linear_weight(normalized_inputs,
                           slice_axis(projection, 0, kHidden, 2 * kHidden),
                           slice_axis(projection_bias, 0, kHidden, 2 * kHidden));
    auto v = linear_weight(normalized_inputs,
                           slice_axis(projection, 0, 2 * kHidden, 3 * kHidden),
                           slice_axis(projection_bias, 0, 2 * kHidden, 3 * kHidden));
    // Match the released QueryFormer's BF16 Q/K/V API boundary.  MLX's fused
    // SDPA retains its own stable internal accumulation.
    auto attention = attend(heads(q, kAttentionHeads, kHeadDim),
                            heads(k, kAttentionHeads, kHeadDim),
                            heads(v, kAttentionHeads, kHeadDim), false);
    query = normalized_query + weights.project(attention, prefix + ".cross_attn.out_proj");
    query = layer_norm_plain(query, 1e-6f);
    auto feed = weights.project(gelu_tanh(weights.project(query, prefix + ".mlp.fc1")),
                                prefix + ".mlp.fc2");
    return query + feed;
}

std::vector<Tensor> text_rope(const Tensor &query, const Tensor &key,
                              const Tensor &cosine, const Tensor &sine) {
    auto rotate = [&](const Tensor &value) {
        auto rotary = slice_axis(value, -1, 0, kRotaryDim);
        auto passthrough = slice_axis(value, -1, kRotaryDim, kHeadDim);
        auto halves = mx::split(rotary, 2, -1);
        auto rotated_half = mx::concatenate({-halves[1], halves[0]}, -1);
        auto c = mx::expand_dims(cosine, 1);
        auto s = mx::expand_dims(sine, 1);
        auto rotated = rotary * c + rotated_half * s;
        return mx::concatenate({rotated, passthrough}, -1);
    };
    return {rotate(query), rotate(key)};
}

Tensor dense_mlp(const Tensor &x, const Weights &weights, const std::string &prefix) {
    return weights.project(
        silu(weights.project(x, prefix + ".gate_proj")) *
            weights.project(x, prefix + ".up_proj"),
        prefix + ".down_proj");
}

Tensor sparse_moe(const Tensor &x, const Weights &weights, const std::string &prefix,
                  const std::string &dump_directory, int layer) {
    const int tokens = int(x.size() / x.shape(-1));
    auto flat = mx::reshape(x, {tokens, kHidden});
    auto logits = mx::matmul(mx::astype(flat, mx::float32),
                             mx::transpose(mx::astype(
                                 weights.at(prefix + ".gate.weight"), mx::float32)));
    auto scores = mx::sigmoid(logits);
    auto routing_scores = scores + mx::astype(weights.at(prefix + ".gate.expert_bias"),
                                               mx::float32);

    auto grouped = mx::reshape(routing_scores,
                               {tokens, kExpertGroups, kExpertsPerGroup});
    auto group_scores = mx::sum(mx::topk(grouped, 2, -1), -1);
    auto group_order = mx::argsort(group_scores, -1);
    auto selected_groups = slice_axis(group_order, -1,
                                      kExpertGroups - kSelectedGroups, kExpertGroups);
    auto all_groups = mx::reshape(mx::arange(kExpertGroups, mx::int32),
                                  {1, kExpertGroups, 1});
    auto selected = mx::expand_dims(selected_groups, 1);
    auto group_mask = mx::any(all_groups == selected, -1);
    auto expert_mask = mx::repeat(group_mask, kExpertsPerGroup, -1);
    auto masked = mx::where(expert_mask, routing_scores,
                            Tensor(-INFINITY, mx::float32));
    auto expert_order = mx::argsort(masked, -1);
    auto top_indices = slice_axis(expert_order, -1,
                                  kExperts - kExpertsPerToken, kExperts);
    // torch.topk, used by the released model, returns the selected experts in
    // descending score order by default.  argsort returns ascending order, so
    // reverse the selected tail before gathering weights and reducing expert
    // outputs.  The expert set is the same either way, but FP32 top-k reduction
    // is order-sensitive and the discrepancy compounds across routed layers.
    top_indices = mx::flip(top_indices, -1);
    auto top_weights = mx::take_along_axis(scores, top_indices, -1);
    top_weights = top_weights /
                  (mx::sum(top_weights, -1, true) + Tensor(1e-20f, mx::float32));
    top_weights = top_weights * Tensor(2.5f, mx::float32);
    if (!dump_directory.empty()) {
        dump_text_stage(dump_directory,
                        "router_logits_layer_" + std::to_string(layer),
                        logits);
        dump_text_stage(dump_directory,
                        "router_indices_layer_" + std::to_string(layer),
                        mx::astype(top_indices, mx::float32));
        dump_text_stage(dump_directory,
                        "router_weights_layer_" + std::to_string(layer),
                        top_weights);
    }

    auto repeated = mx::repeat(flat, kExpertsPerToken, 0);
    auto expert_indices = mx::reshape(top_indices, {tokens * kExpertsPerToken});
    auto batched_input = mx::expand_dims(repeated, 1);

    auto expert_mm = [&](const Tensor &input, const std::string &name,
                         bool transpose_weight) {
        auto weight = weights.at(prefix + ".experts." + name);
        if (transpose_weight)
            weight = mx::transpose(weight, {0, 2, 1});
        weight = mx::contiguous(weight, false);
        auto output = mx::gather_mm(input, weight, std::nullopt,
                                    expert_indices, false);
        return mx::squeeze(output, 1);
    };

    auto gate = expert_mm(batched_input, "gate_proj", true);
    auto up = expert_mm(batched_input, "up_proj", true);
    auto activated = silu(gate) * up;
    // Match the released fused_moe_ops contract exactly: routing weights are
    // rounded to the expert compute dtype and applied to the intermediate
    // activation before the down projection.  Applying FP32 weights after the
    // down projection is algebraically equivalent in real arithmetic but not
    // across the model's BF16 boundaries, and the discrepancy accumulates over
    // nineteen routed layers.
    auto routed_weights = mx::reshape(mx::astype(top_weights, activated.dtype()),
                                      {tokens * kExpertsPerToken, 1});
    activated = activated * routed_weights;
    auto expert_output = expert_mm(mx::expand_dims(activated, 1),
                                   "down_proj", true);
    expert_output = mx::reshape(expert_output,
                                {tokens, kExpertsPerToken, kHidden});
    auto routed = mx::sum(mx::astype(expert_output, mx::float32), 1);
    auto shared = dense_mlp(flat, weights, prefix + ".shared_experts");
    return mx::reshape(mx::astype(routed, shared.dtype()) + shared, x.shape());
}

Tensor text_backbone(const Tensor &input, const Weights &weights,
                     int prompt_tokens, const Event &event,
                     std::atomic<bool> &cancelled,
                     const std::string &dump_directory) {
    const int sequence = input.shape(1);
    auto positions = mx::arange(sequence, mx::float32);
    auto inverse = 1.f /
                   mx::power(Tensor(600000.f, mx::float32),
                             mx::arange(0, kRotaryDim, 2, mx::float32) /
                                 float(kRotaryDim));
    auto angles = mx::reshape(positions, {1, sequence, 1}) *
                  mx::reshape(inverse, {1, 1, kRotaryDim / 2});
    auto doubled = mx::concatenate({angles, angles}, -1);
    auto cosine = mx::astype(mx::cos(doubled), input.dtype());
    auto sine = mx::astype(mx::sin(doubled), input.dtype());

    auto qi = mx::reshape(mx::arange(sequence, mx::int32), {sequence, 1});
    auto ki = mx::reshape(mx::arange(sequence, mx::int32), {1, sequence});
    auto forbidden = (qi < Tensor(prompt_tokens, mx::int32)) &
                     (ki >= Tensor(prompt_tokens, mx::int32));
    // The released pipeline builds this mask in the activation dtype with
    // torch.finfo(BF16).min rather than FP32 -inf.  Keep both the finite mask
    // value and the public BF16 SDPA boundary, otherwise layer-0 rounding moves
    // near-tied router scores across expert-selection thresholds.
    auto mask = mx::reshape(
        mx::where(forbidden,
                  Tensor(std::numeric_limits<float>::lowest(), input.dtype()),
                  Tensor(0.f, input.dtype())),
        {1, 1, sequence, sequence});

    auto hidden = input;
    for (int layer = 0; layer < 20; ++layer) {
        checkpoint(cancelled);
        event("llada_text_backbone", layer, 20);
        const auto prefix = "layers." + std::to_string(layer);
        auto residual = hidden;
        auto normalized = rms(hidden, weights.at(prefix + ".input_layernorm.weight"),
                              1e-6f);
        if (layer == 0)
            dump_text_stage(dump_directory, "layer_0_input_norm", normalized);
        auto qkv = weights.project(normalized, prefix + ".attention.query_key_value");
        if (layer == 0)
            dump_text_stage(dump_directory, "layer_0_qkv", qkv);
        auto query = heads(slice_axis(qkv, -1, 0, 2048), kAttentionHeads, kHeadDim);
        auto key = heads(slice_axis(qkv, -1, 2048, 2560), kKVHeads, kHeadDim);
        auto value = heads(slice_axis(qkv, -1, 2560, 3072), kKVHeads, kHeadDim);
        query = rms(query, weights.at(prefix + ".attention.query_layernorm.weight"),
                    1e-6f);
        key = rms(key, weights.at(prefix + ".attention.key_layernorm.weight"),
                  1e-6f);
        if (layer == 0) {
            dump_text_stage(dump_directory, "layer_0_q_norm", query);
            dump_text_stage(dump_directory, "layer_0_k_norm", key);
        }
        auto rotated = text_rope(query, key, cosine, sine);
        key = mx::repeat(rotated[1], kAttentionHeads / kKVHeads, 1);
        value = mx::repeat(value, kAttentionHeads / kKVHeads, 1);
        auto attention_output = weights.project(
            attend(rotated[0], key, value, false, mask),
            prefix + ".attention.dense");
        if (layer == 0)
            dump_text_stage(dump_directory, "layer_0_attention_dense", attention_output);
        hidden = residual + attention_output;

        residual = hidden;
        normalized = rms(hidden,
                         weights.at(prefix + ".post_attention_layernorm.weight"),
                         1e-6f);
        if (layer == 0)
            dump_text_stage(dump_directory, "layer_0_post_attention_norm", normalized);
        Tensor mlp_output = normalized;
        if (layer == 0) {
            auto gate = weights.project(normalized, prefix + ".mlp.gate_proj");
            auto up = weights.project(normalized, prefix + ".mlp.up_proj");
            if (!dump_directory.empty()) {
                dump_text_stage(dump_directory, "layer_0_mlp_gate", gate);
                dump_text_stage(dump_directory, "layer_0_mlp_up", up);
            }
            mlp_output = weights.project(silu(gate) * up,
                                         prefix + ".mlp.down_proj");
            dump_text_stage(dump_directory, "layer_0_mlp_down", mlp_output);
        } else {
            mlp_output = sparse_moe(normalized, weights, prefix + ".mlp",
                                    dump_directory, layer);
        }
        hidden = residual + mlp_output;
        mx::eval(hidden);
        dump_text_stage(dump_directory, "backbone_layer_" + std::to_string(layer), hidden);
    }
    event("llada_text_backbone", 20, 20);
    return rms(hidden, weights.at("norm.weight"), 1e-6f);
}

Tensor text_projection(const Tensor &input, const Weights &weights,
                       const Event &event, std::atomic<bool> &cancelled,
                       const std::string &dump_directory) {
    auto hidden = input;
    for (int layer = 0; layer < 6; ++layer) {
        checkpoint(cancelled);
        event("llada_text_projection", layer, 6);
        const auto prefix = "layers." + std::to_string(layer);
        auto normalized = rms_plain(hidden, 1e-6f);
        auto query = rms_plain(heads(weights.project(normalized,
                                                     prefix + ".self_attn.q_proj"),
                                     32, 64),
                               1e-6f);
        auto key = rms_plain(heads(weights.project(normalized,
                                                   prefix + ".self_attn.k_proj"),
                                   32, 64),
                             1e-6f);
        auto value = heads(weights.project(normalized,
                                           prefix + ".self_attn.v_proj"),
                           32, 64);
        hidden = hidden + weights.project(attend(query, key, value, false),
                                          prefix + ".self_attn.out_proj");
        normalized = rms_plain(hidden, 1e-6f);
        hidden = hidden + weights.project(
                              gelu_tanh(weights.project(normalized,
                                                       prefix + ".mlp.fc1")),
                              prefix + ".mlp.fc2");
        mx::eval(hidden);
        dump_text_stage(dump_directory, "projection_layer_" + std::to_string(layer), hidden);
    }
    event("llada_text_projection", 6, 6);
    return weights.project(hidden, "projector");
}

} // namespace

LLaDAConditioning llada_encode_text(const std::filesystem::path &root,
                                    const Tokens &tokens, Weights &text,
                                    Weights &query, Weights &projection,
                                    const Event &event,
                                    std::atomic<bool> &cancelled,
                                    const std::string &dump_directory) {
    require(tokens.valid > 0 && tokens.valid == int(tokens.ids.size()),
            "LLaDA native text encoder requires an unpadded prompt");

    try {
        if (text.bytes() == 0) {
            event("load_llada_text", 0, 3);
            text.load(root / "text_encoder", event, cancelled);
            text.remap_keys([](std::string key) {
                constexpr const char *prefix = "model.language_model.";
                if (key.starts_with(prefix))
                    key.erase(0, std::strlen(prefix));
                return key;
            });
            text.erase_prefix("model.lm_head");
            text.erase_prefix("lm_head");
            event("load_llada_text", 1, 3);
        }
        if (query.bytes() == 0) {
            query.load(root / "queryformer", event, cancelled);
            event("load_llada_text", 2, 3);
        }
        if (projection.bytes() == 0) {
            projection.load(root / "text_projection", event, cancelled);
            event("load_llada_text", 3, 3);
        }

        auto ids = Tensor(tokens.ids.data(), {1, tokens.valid}, mx::int32);
        auto embeddings = mx::take(text.at("word_embeddings.weight"), ids, 0);
        auto query_embeddings = queryformer(embeddings, query);
        dump_text_stage(dump_directory, "token_embeddings", embeddings);
        dump_text_stage(dump_directory, "queryformer", query_embeddings);
        auto combined = mx::concatenate({embeddings, query_embeddings}, 1);
        dump_text_stage(dump_directory, "combined", combined);
        auto hidden = text_backbone(combined, text, tokens.valid, event, cancelled,
                                    dump_directory);
        dump_text_stage(dump_directory, "backbone_final", hidden);
        auto projected = text_projection(hidden, projection, event, cancelled,
                                         dump_directory);
        dump_text_stage(dump_directory, "projection_final", projected);
        auto features = mx::squeeze(projected, 0);
        mx::eval(features);
        const int total = features.shape(0);
        mx::clear_cache();
        return {std::move(features), tokens.valid, total};
    } catch (...) {
        text.clear();
        query.clear();
        projection.clear();
        mx::clear_cache();
        throw;
    }
}

} // namespace tc

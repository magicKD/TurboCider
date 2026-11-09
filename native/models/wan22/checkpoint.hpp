#pragma once

#include "../../backends/mlx.hpp"

#include <array>
#include <string_view>

namespace tc::wan22 {

enum class CheckpointLayout { diffusers, comfy };

inline std::string comfy_key(const std::string &name, bool block) {
    using Pair = std::pair<std::string_view, std::string_view>;
    static constexpr std::array top{
        Pair{"head.head.", "proj_out."},
        Pair{"time_embedding.0.", "condition_embedder.time_embedder.linear_1."},
        Pair{"time_embedding.2.", "condition_embedder.time_embedder.linear_2."},
        Pair{"time_projection.1.", "condition_embedder.time_proj."},
        Pair{"text_embedding.0.", "condition_embedder.text_embedder.linear_1."},
        Pair{"text_embedding.2.", "condition_embedder.text_embedder.linear_2."},
    };
    static constexpr std::array blocks{
        Pair{"norm3.", "norm2."}, Pair{"ffn.0.", "ffn.net.0.proj."}, Pair{"ffn.2.", "ffn.net.2."},
        Pair{"self_attn.q.", "attn1.to_q."}, Pair{"self_attn.k.", "attn1.to_k."},
        Pair{"self_attn.v.", "attn1.to_v."}, Pair{"self_attn.o.", "attn1.to_out.0."},
        Pair{"self_attn.norm_q.", "attn1.norm_q."}, Pair{"self_attn.norm_k.", "attn1.norm_k."},
        Pair{"cross_attn.q.", "attn2.to_q."}, Pair{"cross_attn.k.", "attn2.to_k."},
        Pair{"cross_attn.v.", "attn2.to_v."}, Pair{"cross_attn.o.", "attn2.to_out.0."},
        Pair{"cross_attn.norm_q.", "attn2.norm_q."}, Pair{"cross_attn.norm_k.", "attn2.norm_k."},
    };
    if (block && name == "modulation") return "scale_shift_table";
    if (!block && name == "head.modulation") return "scale_shift_table";
    if (!block && (name == "patch_embedding.weight" || name == "patch_embedding.bias")) return name;
    auto replace = [&](const auto &mapping) -> std::string {
        for (const auto &[source, target] : mapping) {
            if (name.starts_with(source)) {
                const auto suffix = name.substr(source.size());
                require(suffix == "weight" || suffix == "bias", "invalid Wan2.2 Comfy weight suffix: " + name);
                return std::string(target) + suffix;
            }
        }
        return {};
    };
    auto mapped = block ? replace(blocks) : replace(top);
    require(!mapped.empty(), "unsupported Wan2.2 Comfy weight: " + name);
    return mapped;
}

// The converted checkpoint has mixed FP32 and FP16 tensors. The native graph
// is BF16, so normalize these weights explicitly before running either layout.
inline void prepare_comfy_weights(Weights &weights, bool block) {
    weights.remap_keys([&](const std::string &key) { return comfy_key(key, block); });
    const auto keys = weights.sorted_keys();
    std::vector<Tensor> values;
    values.reserve(keys.size());
    for (const auto &key : keys) {
        const auto &value = weights.at(key);
        require(value.dtype() == mx::float32 || value.dtype() == mx::float16 || value.dtype() == mx::bfloat16,
                "unsupported Wan2.2 Comfy weight dtype: " + key);
        values.push_back(value.dtype() == mx::bfloat16 ? value : mx::astype(value, mx::bfloat16));
    }
    weights.bind_arrays(keys, values);
    weights.materialize();
}

} // namespace tc::wan22

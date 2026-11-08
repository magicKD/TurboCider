#pragma once
#include "../../backends/mlx.hpp"
#include "../../backends/convrot_rotation.hpp"
#include <algorithm>
#include <cctype>

namespace tc::qwen21 {

inline bool convrot_encoder_checkpoint(const std::filesystem::path &path) {
    return path.filename() == "qwen3vl_8b_int8_convrot.safetensors";
}

inline std::filesystem::path encoder_checkpoint_source(const std::filesystem::path &root) {
    // Keep the existing floating-point source priority when both are present.
    for (const auto *name : {"text_encoders/qwen3vl_8b_bf16.safetensors",
                             "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
                             "text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"})
        if (std::filesystem::is_regular_file(root / name)) return root / name;
    throw std::runtime_error("missing Qwen Image 2.1 encoder: expected qwen3vl_8b_bf16.safetensors, "
                             "qwen3vl_8b_int8_convrot.safetensors or Qwen3-VL-8B-Instruct-Q4_K_M.gguf");
}

// The published Qwen3-VL checkpoint already uses model.* keys. Validate the
// actual ConvRot contract before either raw W8A8 execution or lossless Q8
// packing; merely finding an INT8 tensor is insufficient to establish H256.
inline void prepare_convrot_encoder(Weights &weights, bool w8a8) {
    const auto keys = weights.sorted_keys();
    std::vector<Tensor> tags;
    for (const auto &key : keys) if (key.ends_with(".comfy_quant")) {
        const auto prefix = key.substr(0, key.size() - 12);
        const auto &weight = weights.at(prefix + ".weight");
        const auto &scale = weights.at(prefix + ".weight_scale");
        const auto &tag = weights.at(key);
        require(weight.dtype() == mx::int8 && weight.ndim() == 2 && weight.shape(0) > 0 &&
                    weight.shape(1) > 0 && weight.shape(1) % 256 == 0 &&
                    scale.dtype() == mx::float32 && scale.shape() == mx::Shape{weight.shape(0), 1} &&
                    tag.dtype() == mx::uint8 && tag.ndim() == 1 && tag.size() > 0 && tag.size() <= 256,
                "invalid Qwen3-VL ConvRot source geometry: " + prefix);
        tags.push_back(tag);
    }
    require(!tags.empty(), "Qwen3-VL ConvRot checkpoint has no quantized matrices");
    mx::eval(tags);
    for (const auto &tag : tags) {
        std::string marker(reinterpret_cast<const char *>(tag.data<uint8_t>()), tag.size());
        marker.erase(std::remove_if(marker.begin(), marker.end(), [](unsigned char c) { return std::isspace(c); }), marker.end());
        require(marker == "{\"format\":\"int8_tensorwise\",\"convrot\":true,\"convrot_groupsize\":256}",
                "Qwen3-VL encoder requires the published int8_tensorwise / ConvRot H256 marker");
    }
    auto matrix = [&](const std::string &prefix, int rows, int columns) {
        require(weights.convrot(prefix) && weights.at(prefix + ".weight").shape() == mx::Shape{rows, columns},
                "Qwen3-VL ConvRot encoder matrix mismatch: " + prefix);
    };
    matrix("model.embed_tokens", 151936, 4096);
    for (int layer = 0; layer < 36; ++layer) {
        const auto prefix = "model.layers." + std::to_string(layer);
        for (const auto *name : {"q_proj", "o_proj"}) matrix(prefix + ".self_attn." + name, 4096, 4096);
        for (const auto *name : {"k_proj", "v_proj"}) matrix(prefix + ".self_attn." + name, 1024, 4096);
        for (const auto *name : {"gate_proj", "up_proj"}) matrix(prefix + ".mlp." + name, 12288, 4096);
        matrix(prefix + ".mlp.down_proj", 4096, 12288);
    }
    // Vision is deliberately still BF16 in this artifact. Reject integer
    // tensors lacking a recognized marker instead of treating codes as dense.
    for (const auto &key : keys) {
        const auto &value = weights.at(key);
        if (value.dtype() == mx::int8)
            require(key.ends_with(".weight") && weights.convrot(key.substr(0, key.size() - 7)),
                    "unmarked INT8 tensor in Qwen3-VL encoder: " + key);
        if (key.starts_with("model.visual."))
            require(value.dtype() == mx::bfloat16, "unsupported quantized Qwen3-VL visual tower: " + key);
    }
    // Image conditioning consumes hidden states, never vocabulary logits.
    // Dropping this unused 622 MB matrix also avoids packing it in the baseline.
    weights.erase_prefix("lm_head.");
    weights.set_convrot_w8a8(w8a8);
    if (!w8a8) weights.pack_convrot_q8(128, mx::float32);
}

inline Tensor encoder_embedding_rows(const Weights &weights, const Tensor &ids,
                                     const std::string &prefix) {
    if (!weights.convrot(prefix)) return weights.embedding_rows(ids, prefix);
    require(ids.ndim() == 1 && ids.dtype() == mx::int32 && ids.size() > 0,
            "Qwen3-VL embedding gather requires nonempty rank1 int32 IDs");
    const auto &weight = weights.at(prefix + ".weight");
    require(weight.ndim() == 2 && weight.shape(0) > 0 && weight.shape(1) > 0,
            "invalid Qwen3-VL ConvRot embedding table");
    // Callers validate host token IDs before constructing this gather. Gather
    // every plane first: decoding the full vocabulary would allocate >1 GiB.
    auto codes = mx::take(weight, ids, 0);
    Tensor rotated = codes;
    if (weight.dtype() == mx::int8) {
        const auto &scale = weights.at(prefix + ".weight_scale");
        require(weight.shape(1) % 256 == 0 && scale.dtype() == mx::float32 &&
                    scale.shape() == mx::Shape{weight.shape(0), 1},
                "invalid raw Qwen3-VL ConvRot embedding coefficients");
        rotated = mx::astype(codes, mx::float32) * mx::take(scale, ids, 0);
    } else {
        const auto &scale = weights.at(prefix + ".scales"), &bias = weights.at(prefix + ".biases");
        const int width = weight.shape(1) * 4;
        require(weight.dtype() == mx::uint32 && width % 256 == 0 && scale.ndim() == 2 &&
                    scale.shape(0) == weight.shape(0) && scale.shape(1) > 0 && width % scale.shape(1) == 0 &&
                    bias.shape() == scale.shape() && scale.dtype() == mx::float32 && bias.dtype() == mx::float32,
                "invalid packed Qwen3-VL ConvRot embedding coefficients");
        const int group = width / scale.shape(1);
        require(group == 32 || group == 64 || group == 128, "unsupported ConvRot embedding Q8 group");
        rotated = mx::dequantize(codes, mx::take(scale, ids, 0), mx::take(bias, ids, 0),
                                group, 8, "affine", std::nullopt, mx::float32);
    }
    // Comfy's regular H4 Kronecker power is symmetric and orthonormal, so
    // H256^-1 == H256. This is not MLX's Sylvester Hadamard ordering. Match
    // comfy-kitchen dequantize_embedding: inverse in F32, one final BF16 cast.
    return mx::astype(convrot_kernel::rotate(rotated, convrot_kernel::Rotation::SimdRegister), mx::bfloat16);
}
} // namespace tc::qwen21

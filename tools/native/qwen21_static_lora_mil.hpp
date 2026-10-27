#pragma once

// Isolated research emitter. The shipping runtime and its dynamic-weight MIL
// remain unchanged. Base BF16 values are converted to FP16, never LoRA-merged.
#include "../../native/backends/private/ane_mil.hpp"
#include <array>
#include <cstring>
#include <span>

namespace tc::qwen21::static_lora_lab {
struct FrozenProgram { std::string mil; std::vector<uint8_t> constants; };
inline void replace_once(std::string &text, const std::string &from, const std::string &to) {
    const auto at = text.find(from);
    if (at == std::string::npos || text.find(from, at + from.size()) != std::string::npos)
        throw std::invalid_argument("static LoRA emitter source contract changed");
    text.replace(at, from.size(), to);
}
inline FrozenProgram freeze(const ane::GraphShape &shape,
                            const std::array<std::span<const uint16_t>, 3> &weights) {
    if (shape.kind != ane::Kind::SwiGLU || shape.rows != 1024 || shape.hidden != 4096 ||
        (shape.width != 1024 && shape.width != 2048) || !shape.lora_inputs ||
        shape.tile_k != 4096 || shape.tile_n != 4096)
        throw std::invalid_argument("static LoRA pilot requires bounded original Qwen decode geometry");
    FrozenProgram result{ane::private_api::fp16_program(shape), std::vector<uint8_t>(64)};
    auto put = [&](size_t offset, auto value) {
        std::memcpy(result.constants.data() + offset, &value, sizeof(value));
    };
    put(0, uint32_t(3)); put(4, uint32_t(2));
    const std::array<std::string, 3> names{"wg", "wu", "wd"};
    for (size_t i = 0; i < weights.size(); ++i) {
        const int rows = i == 2 ? shape.hidden : shape.width;
        const int cols = i == 2 ? shape.width : shape.hidden;
        if (weights[i].size() != size_t(rows) * cols)
            throw std::invalid_argument("static LoRA constant extent mismatch");
        const uint64_t bytes = weights[i].size_bytes(), record = result.constants.size(), data = record + 64;
        result.constants.resize(data + bytes);
        put(record, uint32_t(0xdeadbeef)); put(record + 4, uint32_t(1));
        put(record + 8, bytes); put(record + 16, data);
        std::memcpy(result.constants.data() + data, weights[i].data(), bytes);
        const auto geometry = "[1, 1, " + std::to_string(rows) + ", " + std::to_string(cols) + "]";
        const auto plane = std::to_string(uint64_t(rows) * cols);
        const auto buffer = "tensor_buffer<fp16, shape=" + geometry + ", strides=[" + plane + ", " +
            plane + ", " + std::to_string(cols) + ", 1], interleave_factors=[1, 1, 1, 1]>";
        replace_once(result.mil, ", " + buffer + " " + names[i], "");
        const auto tensor = "tensor<fp16, " + geometry + "> " + names[i] + "_t";
        replace_once(result.mil, tensor + " = tensor_buffer_to_tensor<ios17>(input = " + names[i] + ")",
            tensor + " = const()[name = string(\"" + names[i] + "_t\"), val = tensor<fp16, " + geometry +
            ">(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(" + std::to_string(record) + ")))]");
    }
    // exp(-abs(g)) keeps the exponent finite. Preserve corrected gate/up and
    // return the complete corrected prefix hidden for one full down-LoRA.
    const auto tensor = "tensor<fp16, [1, 1, " + std::to_string(shape.width) + ", 1024]> ";
    replace_once(result.mil, tensor + "neg = mul(x = gc, y = fp16(-1))",
        tensor + "abs_gate = abs(x = gc);\n        " + tensor + "neg = mul(x = abs_gate, y = fp16(-1))");
    replace_once(result.mil, tensor + "silu = real_div(x = gc, y = denom)",
        tensor + "negative_gate = mul(x = gc, y = eg);\n        tensor<bool, [1, 1, " +
        std::to_string(shape.width) + ", 1024]> positive_gate = greater_equal(x = gc, y = fp16(0));\n        " +
        tensor + "numerator = select(cond = positive_gate, a = gc, b = negative_gate);\n        " +
        tensor + "silu = real_div(x = numerator, y = denom)");
    if (result.constants.size() > (64u << 20)) throw std::invalid_argument("static LoRA constants exceed Program bound");
    return result;
}
} // namespace tc::qwen21::static_lora_lab

#pragma once

// Native binding of the existing dense_gpu_projection.hpp MPP algorithm:
// 32x128 tiles, four SIMD groups, physical row-major BF16 source matrices and
// FP32 accumulation. Optional BF16 checkpoint bias preserves the output cast
// before addition. No compact source/transpose bank is synthesized.
#include "../backend.hpp"

#include <limits>
#include <optional>

namespace tc::metal_graph::dense_linear {

inline constexpr std::string_view precision_recipe = "dense-bf16-mpp32x128-fp32-accum-cast-before-bias-v2";

struct Geometry {
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
};

struct Parameters {
    uint32_t rows = 0;
    uint32_t input_stride = 0;
    uint32_t weight_stride = 0;
    uint32_t output_stride = 0;
};
static_assert(sizeof(Parameters) == 16);

// Shared production validation for the native wrapper and CPU-only contract
// checks. This descriptor does not pretend to own or construct a GPU buffer.
struct Layout {
    uint32_t rows = 0;
    uint32_t columns = 0;
    size_t row_stride_bytes = 0;
    size_t view_size_bytes = 0;
    size_t offset_bytes = 0;
    ScalarType type = ScalarType::Bytes;
};
struct CheckedBindings {
    Parameters parameters;
    size_t input_bytes = 0;
    size_t weight_bytes = 0;
    size_t output_bytes = 0;
    size_t bias_bytes = 0;
};

inline void validate_geometry(Geometry geometry) {
    // Includes Qwen's latent projections K=64 / N=64. Wider arbitrary tails
    // are not advertised until independently device-validated.
    if (geometry.input_columns == 0 || geometry.input_columns > 12288 || geometry.input_columns % 64 ||
        geometry.output_columns == 0 || geometry.output_columns > 32768 || geometry.output_columns % 64)
        throw std::invalid_argument("native linear requires bounded 64-aligned K and N");
}

inline size_t validate_layout(Layout layout) {
    if (layout.type != ScalarType::BF16 && layout.type != ScalarType::FP32)
        throw std::invalid_argument("native linear only accepts BF16 or explicit FP32 matrix storage");
    const auto element = scalar_bytes(layout.type);
    const auto maximum = std::numeric_limits<size_t>::max();
    if (layout.rows == 0 || layout.rows > uint32_t(std::numeric_limits<int>::max()) ||
        layout.columns == 0 || layout.columns > uint32_t(std::numeric_limits<int>::max()) ||
        layout.columns > maximum/element || layout.offset_bytes % element ||
        layout.row_stride_bytes % element ||
        layout.row_stride_bytes/element > size_t(std::numeric_limits<int>::max()) ||
        layout.view_size_bytes > maximum-layout.offset_bytes)
        throw std::invalid_argument("native linear matrix alignment, extent or stride overflow");
    const auto last = size_t(layout.columns)*element;
    if (layout.row_stride_bytes < last ||
        size_t(layout.rows-1) > (maximum-last)/layout.row_stride_bytes)
        throw std::invalid_argument("native linear matrix pitch/extent overflow");
    const auto extent = size_t(layout.rows-1)*layout.row_stride_bytes+last;
    if (extent > layout.view_size_bytes)
        throw std::invalid_argument("native linear matrix exceeds its retained view");
    return extent;
}

inline CheckedBindings validate_bindings(Geometry geometry, Layout input,
    Layout weight, Layout output, std::optional<Layout> bias = {}) {
    validate_geometry(geometry);
    CheckedBindings checked;
    checked.input_bytes = validate_layout(input);
    checked.weight_bytes = validate_layout(weight);
    checked.output_bytes = validate_layout(output);
    if (input.type != ScalarType::BF16 || weight.type != ScalarType::BF16 ||
        input.columns != geometry.input_columns ||
        weight.rows != geometry.output_columns || weight.columns != geometry.input_columns ||
        output.rows != input.rows || output.columns != geometry.output_columns)
        throw std::invalid_argument("native linear logical shape/input-weight dtype mismatch");
    if (bias) {
        checked.bias_bytes = validate_layout(*bias);
        if (bias->rows != 1 || bias->columns != geometry.output_columns ||
            bias->type != ScalarType::BF16)
            throw std::invalid_argument("native linear bias must be BF16 [1,N]");
    }
    checked.parameters = {input.rows,
        uint32_t(input.row_stride_bytes/scalar_bytes(input.type)),
        uint32_t(weight.row_stride_bytes/scalar_bytes(weight.type)),
        uint32_t(output.row_stride_bytes/scalar_bytes(output.type))};
    return checked;
}

inline bool ranges_overlap(size_t offset_a, size_t bytes_a, size_t offset_b, size_t bytes_b) {
    if (bytes_a == 0 || bytes_b == 0) return false;
    return offset_a <= offset_b ? offset_b-offset_a < bytes_a : offset_a-offset_b < bytes_b;
}

inline constexpr std::string_view source_body = R"metal(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct TCNativeLinearParams {
    uint rows, input_stride, weight_stride, output_stride;
};

template<typename Output, bool HasBias = false>
inline void tc_native_linear_body(device const bfloat* input,
    device const bfloat* weight, device Output* output,
    constant TCNativeLinearParams& p, uint2 group,
    device const bfloat* bias = nullptr) {
    uint row = group.y*32, col = group.x*128;
    auto a = tensor(const_cast<device bfloat*>(input),
                    dextents<int,2>{TC_LINEAR_K,int(p.rows)},
                    array<int,2>{1,int(p.input_stride)});
    auto b = tensor(const_cast<device bfloat*>(weight),
                    dextents<int,2>{TC_LINEAR_K,TC_LINEAR_N},
                    array<int,2>{1,int(p.weight_stride)});
    auto aa = a.slice(0,row), bb = b.slice(0,col);
    matmul2d<matmul2d_descriptor(32,128,TC_LINEAR_K,false,true,false),
             execution_simdgroups<4>> op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
    op.run(aa,bb,acc);
    for (uint i=0; i<acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto coord = acc.get_multidimensional_index(i);
        if (row+coord[1] < p.rows && col+coord[0] < TC_LINEAR_N) {
            // BF16 matmul output is rounded before the independent model bias
            // operation. Adding bias to FP32 acc then casting changes Qwen.
            Output value = Output(acc[i]);
            if constexpr (HasBias) value = Output(float(value)+float(bias[col+coord[0]]));
            output[ulong(row+coord[1])*p.output_stride+col+coord[0]] = value;
        }
    }
}

kernel void tc_native_linear_bf16(device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]], device bfloat* output [[buffer(2)]],
    constant TCNativeLinearParams& p [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]]) {
    tc_native_linear_body(input,weight,output,p,group);
}

// Deliberate FP32 output; the consumer is responsible for the model boundary.
kernel void tc_native_linear_f32(device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]], device float* output [[buffer(2)]],
    constant TCNativeLinearParams& p [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]]) {
    tc_native_linear_body(input,weight,output,p,group);
}

kernel void tc_native_linear_bias_bf16(device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]], device bfloat* output [[buffer(2)]],
    device const bfloat* bias [[buffer(3)]], constant TCNativeLinearParams& p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]]) {
    tc_native_linear_body<bfloat,true>(input,weight,output,p,group,bias);
}
kernel void tc_native_linear_bias_f32(device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]], device float* output [[buffer(2)]],
    device const bfloat* bias [[buffer(3)]], constant TCNativeLinearParams& p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]]) {
    tc_native_linear_body<float,true>(input,weight,output,p,group,bias);
}
)metal";

inline std::string source(Geometry geometry) {
    validate_geometry(geometry);
    return "#define TC_LINEAR_K "+std::to_string(geometry.input_columns)+
           "\n#define TC_LINEAR_N "+std::to_string(geometry.output_columns)+"\n"+
           std::string(source_body);
}

} // namespace tc::metal_graph::dense_linear

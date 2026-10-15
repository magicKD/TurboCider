#pragma once

#include "../backend.hpp"

// Native bindings of TurboCider's existing dense MPP projection and paired
// SwiGLU algorithms. The input/weight tensors keep their physical row strides;
// no weight transpose, widened bank or compact checkpoint copy is required.

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tc::metal_graph::dense_ffn {

inline constexpr std::string_view precision_recipe =
    "dense-bf16-mpp32x128-gate-sigmoid-fast-exp-denom-recip-bf16-silu-hidden-v3";
inline constexpr std::string_view parallel_recipe =
    "dense-channel-down-fp32-add-single-bf16-join-v1";

// Matches the MSL POD exactly. Strides are elements, and are validated to fit
// MPP's signed-int tensor extents before recording any graph steps.
struct Parameters {
    uint32_t rows = 0;
    uint32_t first_channel = 0;
    uint32_t input_stride = 0;
    uint32_t gate_stride = 0;
    uint32_t up_stride = 0;
    uint32_t hidden_stride = 0;
    uint32_t down_stride = 0;
    uint32_t output_stride = 0;
};
static_assert(sizeof(Parameters) == 32);
struct JoinParameters {
    uint32_t rows = 0, gpu_stride = 0, ane_stride = 0, output_stride = 0;
};
static_assert(sizeof(JoinParameters) == 16);

// CPU-visible layout contract used by the actual native wrapper. It carries
// metadata only and cannot manufacture storage ownership or device readiness.
struct Layout {
    uint32_t rows = 0, columns = 0;
    size_t row_stride_bytes = 0, view_size_bytes = 0, offset_bytes = 0;
    ScalarType type = ScalarType::Bytes;
};
inline size_t validate_layout(Layout m) {
    if (m.type != ScalarType::BF16 && m.type != ScalarType::FP32)
        throw std::invalid_argument("native FFN requires BF16 or FP32 matrix storage");
    const auto element = scalar_bytes(m.type), maximum = std::numeric_limits<size_t>::max();
    if (!m.rows || m.rows > uint32_t(std::numeric_limits<int>::max()) ||
        !m.columns || m.columns > uint32_t(std::numeric_limits<int>::max()) ||
        size_t(m.columns) > maximum / element || m.offset_bytes % element ||
        m.view_size_bytes > maximum - m.offset_bytes || m.row_stride_bytes % element ||
        m.row_stride_bytes / element > size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native FFN matrix alignment, extent or stride overflow");
    const auto tail = size_t(m.columns) * element;
    if (m.row_stride_bytes < tail || size_t(m.rows - 1) > (maximum - tail) / m.row_stride_bytes)
        throw std::invalid_argument("native FFN matrix pitch/extent overflow");
    const auto used = size_t(m.rows - 1) * m.row_stride_bytes + tail;
    if (used > m.view_size_bytes)
        throw std::invalid_argument("native FFN matrix exceeds retained buffer view");
    return used;
}

struct CheckedJoin {
    JoinParameters parameters;
    size_t gpu_bytes = 0, ane_bytes = 0, output_bytes = 0;
};
inline CheckedJoin validate_join(uint32_t hidden, Layout gpu, Layout ane, Layout output) {
    CheckedJoin result{{}, validate_layout(gpu), validate_layout(ane), validate_layout(output)};
    if (!hidden || hidden > 4096 || hidden % 128 || gpu.columns != hidden ||
        ane.columns != hidden || output.columns != hidden || gpu.rows != ane.rows ||
        gpu.rows != output.rows || gpu.type != ScalarType::FP32 ||
        ane.type != ScalarType::FP32 || output.type != ScalarType::BF16)
        throw std::invalid_argument("native FFN join requires matching FP32,FP32 to BF16 matrices");
    result.parameters = {gpu.rows, uint32_t(gpu.row_stride_bytes / 4),
        uint32_t(ane.row_stride_bytes / 4), uint32_t(output.row_stride_bytes / 2)};
    return result;
}
inline bool ranges_overlap(size_t a, size_t a_bytes, size_t b, size_t b_bytes) {
    return a <= b ? b - a < a_bytes : a - b < b_bytes;
}

struct Geometry {
    uint32_t hidden = 0;
    uint32_t width = 0;
    uint32_t channels = 0;
};

inline void validate_geometry(Geometry shape) {
    // Both local model shapes and small synthetic component shapes are valid.
    // This is an operator capability check, not a model/profile qualification.
    if (shape.hidden == 0 || shape.hidden > 4096 || shape.hidden % 128 ||
        shape.width == 0 || shape.width > 12288 || shape.width % 128 ||
        shape.channels == 0 || shape.channels > shape.width || shape.channels % 128)
        throw std::invalid_argument("native dense FFN requires bounded 128-aligned geometry");
}

inline constexpr std::string_view source_body = R"metal(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct TCNativeFfnParams {
    uint rows, first_channel, input_stride, gate_stride, up_stride;
    uint hidden_stride, down_stride, output_stride;
};

kernel void tc_native_ffn_gate_up(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device const bfloat* up [[buffer(2)]],
    device bfloat* hidden [[buffer(3)]],
    constant TCNativeFfnParams& p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]]) {
    uint row = group.y * 32, col = group.x * 128;
    auto a = tensor(const_cast<device bfloat*>(input),
                    dextents<int,2>{TC_FFN_H,int(p.rows)},
                    array<int,2>{1,int(p.input_stride)});
    auto bg = tensor(const_cast<device bfloat*>(gate) + ulong(p.first_channel)*p.gate_stride,
                     dextents<int,2>{TC_FFN_H,TC_FFN_C},
                     array<int,2>{1,int(p.gate_stride)});
    auto bu = tensor(const_cast<device bfloat*>(up) + ulong(p.first_channel)*p.up_stride,
                     dextents<int,2>{TC_FFN_H,TC_FFN_C},
                     array<int,2>{1,int(p.up_stride)});
    auto aa = a.slice(0,row), gg = bg.slice(0,col), uu = bu.slice(0,col);
    matmul2d<matmul2d_descriptor(32,128,TC_FFN_H,false,true,false),
             execution_simdgroups<4>> op;
    auto ag = op.template get_destination_cooperative_tensor<decltype(aa),decltype(gg),float>();
    auto au = op.template get_destination_cooperative_tensor<decltype(aa),decltype(uu),float>();
    op.run(aa,gg,ag);
    op.run(aa,uu,au);
    for (uint i = 0; i < ag.get_capacity(); ++i) {
        if (!ag.is_valid_element(i)) continue;
        auto coord = ag.get_multidimensional_index(i);
        if (row + coord[1] >= p.rows || col + coord[0] >= TC_FFN_C) continue;
        bfloat g = bfloat(ag[i]), u = bfloat(au[i]);
        // Match MLX's BF16 exp overload and each BF16 operation inside
        // Sigmoid<bfloat>, in addition to the SiLU and hidden output casts.
        // Explicit fast exp matches compiled MLX at BF16 rounding midpoints;
        // it does not relax the compile options of any other operation.
        bfloat exponential = bfloat(metal::fast::exp(metal::abs(float(g))));
        bfloat denominator = bfloat(1.0f + float(exponential));
        bfloat y = bfloat(1.0f / float(denominator));
        bfloat sigmoid = g < 0 ? y : bfloat(1.0f-float(y));
        bfloat activated = bfloat(float(g) * float(sigmoid));
        hidden[ulong(row+coord[1])*p.hidden_stride+col+coord[0]] =
            bfloat(float(activated) * float(u));
    }
}

template<typename Output>
inline void tc_native_ffn_down_body(device const bfloat* hidden,
    device const bfloat* weight, device Output* output,
    constant TCNativeFfnParams& p, uint2 group) {
    uint row = group.y * 32, col = group.x * 128;
    auto a = tensor(const_cast<device bfloat*>(hidden),
                    dextents<int,2>{TC_FFN_C,int(p.rows)},
                    array<int,2>{1,int(p.hidden_stride)});
    auto b = tensor(const_cast<device bfloat*>(weight) + p.first_channel,
                    dextents<int,2>{TC_FFN_C,TC_FFN_H},
                    array<int,2>{1,int(p.down_stride)});
    auto aa = a.slice(0,row), bb = b.slice(0,col);
    matmul2d<matmul2d_descriptor(32,128,TC_FFN_C,false,true,false),
             execution_simdgroups<4>> op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
    op.run(aa,bb,acc);
    for (uint i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto coord = acc.get_multidimensional_index(i);
        if (row+coord[1] < p.rows && col+coord[0] < TC_FFN_H)
            output[ulong(row+coord[1])*p.output_stride+col+coord[0]] = Output(acc[i]);
    }
}

kernel void tc_native_ffn_down_bf16(device const bfloat* hidden [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]], device bfloat* output [[buffer(2)]],
    constant TCNativeFfnParams& p [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]]) {
    tc_native_ffn_down_body(hidden,weight,output,p,group);
}

// A channel partial deliberately omits the final down-output narrowing. Its
// consumer must join the complement in FP32 and restore the model boundary.
kernel void tc_native_ffn_down_f32(device const bfloat* hidden [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]], device float* output [[buffer(2)]],
    constant TCNativeFfnParams& p [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]]) {
    tc_native_ffn_down_body(hidden,weight,output,p,group);
}
)metal";

inline constexpr std::string_view join_source_body = R"metal(
struct TCNativeFfnJoinParams { uint rows, gpu_stride, ane_stride, output_stride; };
kernel void tc_native_ffn_join_partials(
    device const float* gpu [[buffer(0)]], device const float* ane [[buffer(1)]],
    device bfloat* output [[buffer(2)]], constant TCNativeFfnJoinParams& p [[buffer(3)]],
    uint2 index [[thread_position_in_grid]]) {
    if (index.x >= TC_FFN_H || index.y >= p.rows) return;
    // No branch is narrowed before addition. This is a new hybrid precision
    // recipe; it does not inherit bit equivalence to a full-width BF16 down.
    const float sum = gpu[ulong(index.y)*p.gpu_stride+index.x] +
                      ane[ulong(index.y)*p.ane_stride+index.x];
    output[ulong(index.y)*p.output_stride+index.x] = bfloat(sum);
}
)metal";

inline std::string source(Geometry shape, bool parallel = false) {
    validate_geometry(shape);
    // MPP's K dimension is a C++ template constant, not a runtime parameter.
    // Bound the specialization at prepare, never compile inside encode/run.
    return "#define TC_FFN_H " + std::to_string(shape.hidden) +
           "\n#define TC_FFN_C " + std::to_string(shape.channels) + "\n" +
           std::string(source_body) + (parallel ? std::string(join_source_body) : std::string{});
}

} // namespace tc::metal_graph::dense_ffn

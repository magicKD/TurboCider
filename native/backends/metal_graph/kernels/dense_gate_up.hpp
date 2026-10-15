#pragma once

#include "dense_ffn.hpp"

// Paired base projections, before any ordered LoRA correction or activation.
// Derived from TurboCider's dense_ffn paired MPP algorithm; its existing source
// and preparation bundle remain unchanged. This independent source has one PSO.
namespace tc::metal_graph::dense_gate_up {

using Geometry = dense_ffn::Geometry;
inline constexpr std::string_view precision_recipe =
    "dense-bf16-mpp32x128-paired-base-separate-bf16-gate-up-v1";

// Element pitches, matching the independent MSL POD. BufferView byte offsets
// are bound by Backend and must not be added again by the kernel.
struct Parameters {
    uint32_t rows = 0, first_channel = 0, input_stride = 0;
    uint32_t gate_stride = 0, up_stride = 0;
    uint32_t gate_output_stride = 0, up_output_stride = 0;
};
static_assert(sizeof(Parameters) == 28);

struct Layout {
    uint32_t rows = 0, columns = 0;
    size_t row_stride_bytes = 0, view_size_bytes = 0, offset_bytes = 0;
    ScalarType type = ScalarType::Bytes;
    uintptr_t native_identity = 0;
};

struct CheckedBindings {
    Parameters parameters;
    size_t input_bytes = 0, gate_bytes = 0, up_bytes = 0;
    size_t gate_output_bytes = 0, up_output_bytes = 0;
};

inline size_t validate_layout(const Layout& matrix) {
    if (!matrix.native_identity || matrix.type != ScalarType::BF16)
        throw std::invalid_argument("native paired gate/up requires retained BF16 matrices");
    return dense_ffn::validate_layout({matrix.rows,matrix.columns,matrix.row_stride_bytes,
        matrix.view_size_bytes,matrix.offset_bytes,matrix.type});
}

inline bool overlaps(const Layout& a, size_t a_bytes, const Layout& b, size_t b_bytes) {
    return a.native_identity == b.native_identity &&
        dense_ffn::ranges_overlap(a.offset_bytes,a_bytes,b.offset_bytes,b_bytes);
}

inline CheckedBindings validate_bindings(Geometry geometry, const Layout& input,
    const Layout& gate, const Layout& up, const Layout& gate_output,
    const Layout& up_output, uint32_t first_channel) {
    dense_ffn::validate_geometry(geometry);
    CheckedBindings checked{{},validate_layout(input),validate_layout(gate),validate_layout(up),
        validate_layout(gate_output),validate_layout(up_output)};
    if (first_channel > geometry.width-geometry.channels ||
        input.columns != geometry.hidden ||
        gate.rows != geometry.width || gate.columns != geometry.hidden ||
        up.rows != geometry.width || up.columns != geometry.hidden ||
        gate_output.rows != input.rows || gate_output.columns != geometry.channels ||
        up_output.rows != input.rows || up_output.columns != geometry.channels)
        throw std::invalid_argument("native paired gate/up logical shape or full source slice mismatch");
    // Read operands may share a fused checkpoint allocation. Every writable
    // bounding extent, including row padding, must be disjoint from all reads
    // and the other output. External overlapping bytesNoCopy wrappers require
    // caller provenance; different native handles cannot prove physical alias.
    for (const auto& output : {std::pair{&gate_output,checked.gate_output_bytes},
                              std::pair{&up_output,checked.up_output_bytes}}) {
        for (const auto& source : {std::pair{&input,checked.input_bytes},
                                  std::pair{&gate,checked.gate_bytes},
                                  std::pair{&up,checked.up_bytes}})
            if (overlaps(*output.first,output.second,*source.first,source.second))
                throw std::invalid_argument("native paired gate/up output aliases an immutable source");
    }
    if (overlaps(gate_output,checked.gate_output_bytes,up_output,checked.up_output_bytes))
        throw std::invalid_argument("native paired gate/up outputs overlap");
    checked.parameters = {input.rows,first_channel,uint32_t(input.row_stride_bytes/2),
        uint32_t(gate.row_stride_bytes/2),uint32_t(up.row_stride_bytes/2),
        uint32_t(gate_output.row_stride_bytes/2),uint32_t(up_output.row_stride_bytes/2)};
    return checked;
}

inline constexpr std::string_view source_body = R"metal(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct TCNativeGateUpParams {
    uint rows, first_channel, input_stride, gate_stride, up_stride;
    uint gate_output_stride, up_output_stride;
};

kernel void tc_native_gate_up_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device const bfloat* up [[buffer(2)]],
    device bfloat* gate_output [[buffer(3)]],
    device bfloat* up_output [[buffer(4)]],
    constant TCNativeGateUpParams& p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]]) {
    uint row = group.y * 32, col = group.x * 128;
    auto a = tensor(const_cast<device bfloat*>(input),
                    dextents<int,2>{TC_GATE_UP_H,int(p.rows)},
                    array<int,2>{1,int(p.input_stride)});
    auto bg = tensor(const_cast<device bfloat*>(gate) + ulong(p.first_channel)*p.gate_stride,
                     dextents<int,2>{TC_GATE_UP_H,TC_GATE_UP_C},
                     array<int,2>{1,int(p.gate_stride)});
    auto bu = tensor(const_cast<device bfloat*>(up) + ulong(p.first_channel)*p.up_stride,
                     dextents<int,2>{TC_GATE_UP_H,TC_GATE_UP_C},
                     array<int,2>{1,int(p.up_stride)});
    auto aa = a.slice(0,row), gg = bg.slice(0,col), uu = bu.slice(0,col);
    matmul2d<matmul2d_descriptor(32,128,TC_GATE_UP_H,false,true,false),
             execution_simdgroups<4>> op;
    auto ag = op.template get_destination_cooperative_tensor<decltype(aa),decltype(gg),float>();
    auto au = op.template get_destination_cooperative_tensor<decltype(aa),decltype(uu),float>();
    op.run(aa,gg,ag);
    op.run(aa,uu,au);
    for (uint i = 0; i < ag.get_capacity(); ++i) {
        if (!ag.is_valid_element(i)) continue;
        auto coord = ag.get_multidimensional_index(i);
        if (row + coord[1] >= p.rows || col + coord[0] >= TC_GATE_UP_C) continue;
        // Both base projections retain their original independent BF16
        // boundaries. Ordered LoRA and SiLU are separate graph operations.
        gate_output[ulong(row+coord[1])*p.gate_output_stride+col+coord[0]] = bfloat(ag[i]);
        up_output[ulong(row+coord[1])*p.up_output_stride+col+coord[0]] = bfloat(au[i]);
    }
}
)metal";

inline std::string source(Geometry geometry) {
    dense_ffn::validate_geometry(geometry);
    return "#define TC_GATE_UP_H "+std::to_string(geometry.hidden)+
           "\n#define TC_GATE_UP_C "+std::to_string(geometry.channels)+"\n"+
           std::string(source_body);
}

} // namespace tc::metal_graph::dense_gate_up

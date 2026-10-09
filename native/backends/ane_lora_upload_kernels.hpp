#pragma once

// Research GPU producer: original typed B + F32 ranks directly to a transposed
// FP16 correction surface. No full correction array, B copy, or ANE MIL change.
// Prepend transfer_source and TC_RANK/TC_BF16/TC_BM/TC_BN/TC_SWAPPED defines.
namespace tc::ane::gpu {
inline constexpr const char *lora_upload_source = R"metal(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#if TC_BF16
using LoraOperand = bfloat;
#else
using LoraOperand = half;
#endif
struct RankParams { uint rows, pitch, begin, available; };
struct LoraUploadParams {
    uint rows, columns, b_pitch, target_pitch, boundary;
    float scale;
};
kernel void tc_ane_lora_rank_narrow(device const float *src [[buffer(0)]],
                                  device ushort *dst [[buffer(1)]],
                                  device atomic_uint *status [[buffer(2)]],
                                  constant RankParams &p [[buffer(3)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * TC_RANK) return;
    uint row = i / TC_RANK, col = i % TC_RANK;
    float value = row < p.available ? src[(p.begin + row) * p.pitch + col] : 0;
    ushort narrowed = TC_BF16 ? to_bfloat(value) : to_half(value);
    uint mask = TC_BF16 ? 0x7f80 : 0x7c00;
    if (!isfinite(value) || (narrowed & mask) == mask) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        narrowed = 0;
    }
    dst[i] = narrowed;
}
kernel void tc_ane_lora_b_upload(device LoraOperand *rank [[buffer(0)]],
                               device LoraOperand *up [[buffer(1)]],
                               device ushort *dst [[buffer(2)]],
                               device atomic_uint *status [[buffer(3)]],
                               constant LoraUploadParams &p [[buffer(4)]],
                               uint2 group [[threadgroup_position_in_grid]]) {
    using namespace mpp::tensor_ops;
    uint row = group.y * TC_BM, col = group.x * TC_BN;
    auto r = tensor(rank, dextents<int,2>{TC_RANK,int(p.rows)}, array<int,2>{1,TC_RANK});
    auto b = tensor(up, dextents<int,2>{TC_RANK,int(p.columns)}, array<int,2>{1,int(p.b_pitch)});
#if TC_SWAPPED
    auto a = b.slice(0,row), bb = r.slice(0,col);
#else
    auto a = r.slice(0,row), bb = b.slice(0,col);
#endif
    matmul2d<matmul2d_descriptor(TC_BM,TC_BN,TC_RANK,false,true,false),execution_simdgroups<4>> op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(a),decltype(bb),float>();
    op.run(a,bb,acc);
    for (uint i=0; i<acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto c = acc.get_multidimensional_index(i);
#if TC_SWAPPED
        uint channel = row+c[1], token = col+c[0];
#else
        uint token = row+c[1], channel = col+c[0];
#endif
        if (token>=p.rows || channel>=p.columns) continue;
        float value=acc[i]*p.scale;
        // Match the existing delta dtype boundary BEFORE the FP16 IOSurface
        // carrier. Integer conversions also retain finite half subnormals.
        if (p.boundary==1) value=as_type<float>(uint(to_bfloat(value))<<16);
        else if (p.boundary==0) value=from_half(to_half(value));
        ushort result=to_half(value);
        if (!isfinite(value) || (result&0x7c00)==0x7c00) {
            atomic_fetch_or_explicit(status,1u,memory_order_relaxed); result=0;
        }
        dst[channel*p.target_pitch+token]=result;
    }
}
)metal";
} // namespace tc::ane::gpu

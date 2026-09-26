#pragma once
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::z_metal {
// Opt-in fused two-reduction candidate. Preserve all thirty virtual SIMD
// partial sums and both BF16 boundaries while reducing physical thread count.
inline std::vector<mlx::core::array> gate_norm_virtual(
    const mlx::core::array &x, const mlx::core::array &res,
    const mlx::core::array &gw, const mlx::core::array &gm,
    const mlx::core::array &sw, const mlx::core::array &sm, int threads) {
    namespace mx = mlx::core;
    if ((threads != 128 && threads != 256 && threads != 512) ||
        x.ndim() != 3 || x.shape(0) != 1 || x.shape(2) != 3840 ||
        res.shape() != x.shape() || res.dtype() != x.dtype() ||
        gw.shape() != mx::Shape{3840} || sw.shape() != mx::Shape{3840} ||
        gm.size() != 3840 || sm.size() != 3840 ||
        gm.dtype() != x.dtype() || sm.dtype() != x.dtype())
        throw std::invalid_argument("invalid virtual gate norm geometry");
    bool precomputed = x.dtype() != mx::float32;
    auto gate = precomputed ? mx::tanh(gm) : gm;
    static auto kernel = mx::fast::metal_kernel(
        "tc_z_gate_norm_virtual", {"x","res","gw","gate","sw","scale"},
        {"value","feed"}, R"metal(
        uint tid = thread_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint row = threadgroup_position_in_grid.x;
        constexpr uint Chunks = (960 + Threads - 1) / Threads;
        threadgroup float sums[32];
        float4 values[Chunks];
        for (uint chunk=0; chunk<Chunks; ++chunk) {
            uint vt=chunk*Threads+tid;
            if (vt<960) {
                values[chunk]=float4(reinterpret_cast<device const vec<T,4>*>(x)[row*960+vt]);
                float sum=0.0f;
                for (uint i=0;i<4;++i) sum+=values[chunk][i]*values[chunk][i];
                sum=simd_sum(sum);
                if (lane==0) sums[vt/32]=sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum=lane<30?sums[lane]:0.0f;
        float inv=metal::precise::rsqrt(simd_sum(sum)/3840.0f+1e-5f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint chunk=0;chunk<Chunks;++chunk) {
            uint vt=chunk*Threads+tid;
            if (vt<960) {
                uint index=row*960+vt;
                float4 weights=float4(reinterpret_cast<device const vec<G,4>*>(gw)[vt]);
                vec<T,4> gates=reinterpret_cast<device const vec<T,4>*>(gate)[vt];
                vec<T,4> residual=reinterpret_cast<device const vec<T,4>*>(res)[index];
                vec<T,4> result;
                float partial=0.0f;
                for (uint i=0;i<4;++i) {
                    T n=T((values[chunk][i]*inv)*weights[i]);
                    T g=PRECOMPUTED?gates[i]:T(metal::precise::tanh(float(gates[i])));
                    T product=T(float(n)*float(g));
                    result[i]=T(float(residual[i])+float(product));
                    values[chunk][i]=float(result[i]);
                    partial+=values[chunk][i]*values[chunk][i];
                }
                reinterpret_cast<device vec<T,4>*>(value)[index]=result;
                partial=simd_sum(partial);
                if(lane==0) sums[vt/32]=partial;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum=lane<30?sums[lane]:0.0f;
        inv=metal::precise::rsqrt(simd_sum(sum)/3840.0f+1e-5f);
        for(uint chunk=0;chunk<Chunks;++chunk) {
            uint vt=chunk*Threads+tid;
            if(vt<960) {
                float4 weights=float4(reinterpret_cast<device const vec<S,4>*>(sw)[vt]);
                vec<T,4> scales=reinterpret_cast<device const vec<T,4>*>(scale)[vt];
                vec<T,4> result;
                for(uint i=0;i<4;++i) {
                    T n=T((values[chunk][i]*inv)*weights[i]);
                    T factor=T(1.0f+float(scales[i]));
                    result[i]=T(float(n)*float(factor));
                }
                reinterpret_cast<device vec<T,4>*>(feed)[row*960+vt]=result;
            }
        }
        )metal");
    return kernel({x,res,gw,gate,sw,sm},{x.shape(),x.shape()},{x.dtype(),x.dtype()},
        {x.shape(1)*threads,1,1},{threads,1,1},
        {{"T",x.dtype()},{"G",gw.dtype()},{"S",sw.dtype()},
         {"Threads",threads},{"PRECOMPUTED",precomputed}}, {}, false, {});
}
}

#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::z_metal {
namespace mx = mlx::core;

// Shape-specialized QKV projection whose epilogue directly produces the
// head-major SDPA layout. BN is one 128-wide head, so each threadgroup owns a
// complete head for a row tile and can apply the exact Q/K RMSNorm + RoPE
// boundary without materializing the 11520-wide token-major projection.
inline std::vector<mx::array> project_prepare_qkv(
    const mx::array &x, const mx::array &w, const mx::array &qw,
    const mx::array &kw, const mx::array &freqs) {
    if (x.ndim() != 3 || x.shape(0) != 1 || x.shape(2) != 3840 ||
        w.shape() != mx::Shape{11520,3840} ||
        qw.shape() != mx::Shape{128} || kw.shape() != mx::Shape{128} ||
        freqs.shape() != mx::Shape{x.shape(1),64,2} ||
        x.dtype() != mx::bfloat16 || w.dtype() != x.dtype())
        throw std::invalid_argument("Z-Image fused QKV projection geometry/dtype mismatch");
    constexpr int bm = 48;
    static auto kernel = mx::fast::metal_kernel(
        "tc_z_mpp_qkv_prepare_48x128", {"x","w","qw","kw","freqs"},
        {"q","k","v"}, R"metal(
        using namespace mpp::tensor_ops;
        uint row = threadgroup_position_in_grid.y * BM;
        uint col = threadgroup_position_in_grid.x * 128;
        auto a = tensor(const_cast<device T*>(x), dextents<int,2>{3840,M},
                        array<int,2>{1,3840});
        auto b = tensor(const_cast<device T*>(w), dextents<int,2>{3840,11520},
                        array<int,2>{1,3840});
        auto aa = a.slice(0,row), bb = b.slice(0,col);
        matmul2d<matmul2d_descriptor(BM,128,3840,false,true,false),
                 execution_simdgroups<4>> op;
        auto acc = op.template get_destination_cooperative_tensor<
            decltype(aa),decltype(bb),float>();
        op.run(aa,bb,acc);

        threadgroup T tile[BM * 128];
        for (uint i=0; i<acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            auto coord = acc.get_multidimensional_index(i);
            tile[coord[1]*128+coord[0]] = T(acc[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint kind = col / 3840;
        uint head = (col % 3840) / 128;
        for (uint local_row=sg; local_row<BM; local_row+=4) {
            uint token = row + local_row;
            if (token >= M) continue;
            uint d0 = lane * 4;
            uint src = local_row * 128 + d0;
            uint dst = (head * M + token) * 128 + d0;
            T values[4];
            float sum = 0.0f;
            for (uint i=0; i<4; ++i) {
                values[i] = tile[src+i];
                float value = float(values[i]);
                sum += value*value;
            }
            if (kind == 2) {
                for (uint i=0; i<4; ++i) v[dst+i] = values[i];
                continue;
            }
            float inv = metal::precise::rsqrt(simd_sum(sum)/128.0f+1e-5f);
            for (uint i=0; i<4; ++i) {
                uint d = d0+i;
                float weight = kind == 0 ? float(qw[d]) : float(kw[d]);
                values[i] = T((float(values[i])*inv)*weight);
            }
            device T *out = kind == 0 ? q : k;
            for (uint i=0; i<4; i+=2) {
                uint d = d0+i;
                float c = float(freqs[token*128+d]);
                float s = float(freqs[token*128+d+1]);
                float first = float(values[i]), second = float(values[i+1]);
                out[dst+i] = T(first*c-second*s);
                out[dst+i+1] = T(first*s+second*c);
            }
        }
        )metal", "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    const int m = x.shape(1);
    const mx::Shape shape{1,30,m,128};
    return kernel({x,w,qw,kw,freqs}, {shape,shape,shape},
                  {x.dtype(),x.dtype(),x.dtype()},
                  {90*128,(m+bm-1)/bm,1}, {128,1,1},
                  {{"T",x.dtype()},{"M",m},{"BM",bm}}, {}, false, {});
}

} // namespace tc::z_metal

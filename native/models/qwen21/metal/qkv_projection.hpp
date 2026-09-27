#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::qwen21::metal {
namespace mx = mlx::core;

// Experimental QKV+Q/K norm/RoPE screen. Each MPP 48x128 GEMM tile covers
// one complete head and writes directly into the SDPA head-major layout.
// Unlike post-projection fusion, no token-major QKV output is materialized.
inline std::vector<mx::array> project_prepare_qkv(
    const mx::array &x, const mx::array &w, const mx::array &qw,
    const mx::array &kw, const mx::array &cos, const mx::array &sin,
    int bm = 48) {
    if (x.ndim() != 3 || x.shape(0) != 1 || x.shape(2) != 4096 ||
        (bm != 16 && bm != 32 && bm != 48 && bm != 64) ||
        w.shape() != mx::Shape{12288, 4096} ||
        qw.shape() != mx::Shape{128} || kw.shape() != mx::Shape{128} ||
        cos.shape() != mx::Shape{x.shape(1), 64} || sin.shape() != cos.shape() ||
        x.dtype() != mx::bfloat16 || w.dtype() != x.dtype() ||
        qw.dtype() != x.dtype() || kw.dtype() != x.dtype() ||
        cos.dtype() != mx::float32 || sin.dtype() != cos.dtype())
        throw std::invalid_argument("Qwen21 fused QKV projection geometry/dtype mismatch");
    static auto kernel = mx::fast::metal_kernel(
        "tc_qwen21_mpp_qkv_prepare_screen", {"x","w","qw","kw","cos","sin"},
        {"q","k","v"}, R"metal(
        using namespace mpp::tensor_ops;
        uint row = threadgroup_position_in_grid.y * BM;
        uint col = threadgroup_position_in_grid.x * 128;
        auto a = tensor(const_cast<device T*>(x), dextents<int,2>{4096,M},
                        array<int,2>{1,4096});
        auto b = tensor(const_cast<device T*>(w), dextents<int,2>{4096,12288},
                        array<int,2>{1,4096});
        auto aa = a.slice(0,row), bb = b.slice(0,col);
        matmul2d<matmul2d_descriptor(BM,128,4096,false,true,false),
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
        uint kind = col / 4096;
        uint head = (col % 4096) / 128;
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
            float inv = metal::precise::rsqrt(simd_sum(sum)/128.0f+1e-6f);
            for (uint i=0; i<4; ++i) {
                uint d = d0+i;
                float weight = kind == 0 ? float(qw[d]) : float(kw[d]);
                values[i] = T((float(values[i])*inv)*weight);
            }
            device T *out = kind == 0 ? q : k;
            for (uint i=0; i<4; i+=2) {
                uint d = d0+i;
                float c = float(cos[token*64+d/2]);
                float s = float(sin[token*64+d/2]);
                float first = float(values[i]), second = float(values[i+1]);
                out[dst+i] = T(first*c-second*s);
                out[dst+i+1] = T(first*s+second*c);
            }
        }
        )metal", "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    const int m = x.shape(1);
    const mx::Shape shape{1,32,m,128};
    return kernel({x,w,qw,kw,cos,sin}, {shape,shape,shape},
                  {x.dtype(),x.dtype(),x.dtype()},
                  {96*128,(m+bm-1)/bm,1}, {128,1,1},
                  {{"T",x.dtype()},{"M",m},{"BM",bm}}, {}, false, {});
}
} // namespace tc::qwen21::metal

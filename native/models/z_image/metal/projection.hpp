#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::z_metal {
namespace mx = mlx::core;

// Experimental BF16 projection. Exact dimensions are template constants;
// FP32 accumulation retains the existing GEMM output rounding boundary.
inline mx::array projection(const mx::array &x, const mx::array &w) {
    if (x.ndim() != 3 || x.shape(0) != 1 || w.ndim() != 2 ||
        x.shape(2) != w.shape(1) || x.dtype() != mx::bfloat16 ||
        w.dtype() != x.dtype())
        throw std::invalid_argument("Z-Image Metal projection geometry/dtype mismatch");
    const int m = x.shape(1), n = w.shape(0), k = w.shape(1);
    // Healthy-state M4 Max sweeps at the 512 workload (1056 rows) favor
    // narrower output tiles than the pre-reboot screens. Keep the choices
    // shape-specific: the four dense projections have different occupancy
    // and K-loop costs even though they share the same epilogue.
    const int bm = n == 11520 && k == 3840 ? 48 :
                   n == 3840 && k == 3840 ? 16 : 32;
    const int bn = 128;
    static auto kernel = mx::fast::metal_kernel(
        "tc_z_mpp_projection", {"x", "w"}, {"out"}, R"metal(
        using namespace mpp::tensor_ops;
        uint row = threadgroup_position_in_grid.y * BM;
        uint col = threadgroup_position_in_grid.x * BN;
        auto a = tensor(const_cast<device T*>(x), dextents<int,2>{K,M}, array<int,2>{1,K});
        auto b = tensor(const_cast<device T*>(w), dextents<int,2>{K,N}, array<int,2>{1,K});
        auto aa = a.slice(0,row), bb = b.slice(0,col);
        matmul2d<matmul2d_descriptor(BM,BN,K,false,true,false),execution_simdgroups<4>> op;
        auto acc = op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
        op.run(aa,bb,acc);
        for (uint i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            auto coord = acc.get_multidimensional_index(i);
            if (row + coord[1] < M && col + coord[0] < N)
                out[(row + coord[1])*N + col + coord[0]] = T(acc[i]);
        }
        )metal", "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    return kernel({x,w}, {{1,m,n}}, {x.dtype()},
                  {((n+bn-1)/bn)*128,(m+bm-1)/bm,1}, {128,1,1},
                  {{"T",x.dtype()},{"M",m},{"N",n},{"K",k},{"BM",bm},{"BN",bn}},
                  {}, false, {})[0];
}
} // namespace tc::z_metal

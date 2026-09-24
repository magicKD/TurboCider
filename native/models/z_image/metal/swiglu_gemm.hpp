#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::z_metal {
namespace mx = mlx::core;

// Metal 4 TensorOps epilogue. Healthy-state sweeps at both 1056 and 4128 rows
// favor a 32x128 tile. Retain the pre-reboot 32x256 choice behind an ablation
// switch so same-binary qualification can distinguish the tile change.
inline mx::array swiglu_gemm(const mx::array &x, const mx::array &w,
                             const mx::array &up) {
    if (x.ndim() != 3 || x.shape(0) != 1 || x.shape(2) != 3840 ||
        w.shape() != mx::Shape{10240,3840} ||
        up.shape() != mx::Shape{1,x.shape(1),10240} ||
        x.dtype() != mx::bfloat16 || w.dtype() != x.dtype() || up.dtype() != x.dtype())
        throw std::invalid_argument("Z-Image Metal SwiGLU GEMM geometry mismatch");
    static auto kernel = mx::fast::metal_kernel(
        "tc_z_swiglu_gemm", {"x", "w", "up"}, {"out"}, R"metal(
        using namespace mpp::tensor_ops;
        uint row = threadgroup_position_in_grid.y * 32;
        uint col = threadgroup_position_in_grid.x * BN;
        // MPP traits require non-const element types. The operands are read-only.
        auto a = tensor(const_cast<device T*>(x), dextents<int,2>{3840,M}, array<int,2>{1,3840});
        auto b = tensor(const_cast<device T*>(w), dextents<int,2>{3840,10240}, array<int,2>{1,3840});
        auto aa = a.slice(0,row), bb = b.slice(0,col);
        matmul2d<matmul2d_descriptor(32,BN,3840,false,true,false),execution_simdgroups<4>> op;
        auto acc = op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
        op.run(aa,bb,acc);
        for (uint i=0; i<acc.get_capacity(); ++i) {
            auto coord = acc.get_multidimensional_index(i);
            if (row+coord[1] < M) {
                uint offset = (row+coord[1])*10240+col+coord[0];
                T gate = T(acc[i]);
                auto y = 1 / (1 + metal::exp(metal::abs(gate)));
                T sigmoid = gate < 0 ? y : 1-y;
                out[offset] = T(T(gate*sigmoid)*up[offset]);
            }
        }
        )metal", "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    const int bn = std::getenv("TURBOCIDER_Z_SWIGLU_LEGACY_32X256") ? 256 : 128;
    return kernel({x,w,up}, {up.shape()}, {x.dtype()},
                  {(10240/bn)*128,(x.shape(1)+31)/32,1}, {128,1,1},
                  {{"T",x.dtype()},{"M",x.shape(1)},{"BN",bn}}, {}, false, {})[0];
}

// Dual-projection variant: gate and up share the input tile and threadgroup
// residency. It removes the intermediate `up` dispatch and keeps two FP32
// accumulators. Full-image parity is required; epilogue arithmetic can still
// differ from the reference despite explicit BF16 output conversions.
inline mx::array swiglu_dual_gemm(const mx::array &x, const mx::array &gate_w,
                                  const mx::array &up_w) {
    if (x.ndim() != 3 || x.shape(0) != 1 || x.shape(2) != 3840 ||
        gate_w.shape() != mx::Shape{10240,3840} || up_w.shape() != gate_w.shape() ||
        x.dtype() != mx::bfloat16 || gate_w.dtype() != x.dtype() || up_w.dtype() != x.dtype())
        throw std::invalid_argument("Z-Image dual SwiGLU GEMM geometry mismatch");
    static auto kernel = mx::fast::metal_kernel(
        "tc_z_swiglu_dual_gemm_32x256", {"x", "gate", "up"}, {"out"}, R"metal(
        using namespace mpp::tensor_ops;
        uint row = threadgroup_position_in_grid.y * 32;
        uint col = threadgroup_position_in_grid.x * 256;
        auto a = tensor(const_cast<device T*>(x), dextents<int,2>{3840,M}, array<int,2>{1,3840});
        auto bg = tensor(const_cast<device T*>(gate), dextents<int,2>{3840,10240}, array<int,2>{1,3840});
        auto bu = tensor(const_cast<device T*>(up), dextents<int,2>{3840,10240}, array<int,2>{1,3840});
        auto aa = a.slice(0,row), gg = bg.slice(0,col), uu = bu.slice(0,col);
        matmul2d<matmul2d_descriptor(32,256,3840,false,true,false),execution_simdgroups<4>> op;
        auto ag = op.template get_destination_cooperative_tensor<decltype(aa),decltype(gg),float>();
        auto au = op.template get_destination_cooperative_tensor<decltype(aa),decltype(uu),float>();
        op.run(aa,gg,ag); op.run(aa,uu,au);
        for (uint i = 0; i < ag.get_capacity(); ++i) {
            if (!ag.is_valid_element(i)) continue;
            auto coord = ag.get_multidimensional_index(i);
            if (row + coord[1] < M) {
                uint offset = (row+coord[1])*10240+col+coord[0];
                T g = T(ag[i]), u = T(au[i]);
                auto y = 1 / (1 + metal::exp(metal::abs(float(g))));
                T sigmoid = g < 0 ? T(y) : T(1-y);
                out[offset] = T(T(g*sigmoid)*u);
            }
        }
        )metal", "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    return kernel({x,gate_w,up_w}, {{1,x.shape(1),10240}}, {x.dtype()},
                  {40*128,(x.shape(1)+31)/32,1}, {128,1,1},
                  {{"T",x.dtype()},{"M",x.shape(1)}}, {}, false, {})[0];
}

} // namespace tc::z_metal

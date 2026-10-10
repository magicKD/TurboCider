#pragma once
// Validation-only group32 row-major adaptation of the affine group-sum idea.
// Does not consume Splash's group64/tiled package or change production routing.
#include "backends/mlx.hpp"

namespace tc::qwen21_probe {
inline Tensor u4_projection(const Tensor &input, const Tensor &words,
        const Tensor &scales, const Tensor &biases) {
    require(input.ndim() == 3 && input.shape(0) == 1 && input.dtype() == mx::float16 &&
        words.ndim() == 2 && words.dtype() == mx::uint32 && scales.dtype() == mx::float16 &&
        biases.dtype() == mx::float16 && scales.shape() == biases.shape() && scales.shape(0) == words.shape(0) &&
        input.shape(2) == words.shape(1) * 8 && input.shape(2) == scales.shape(1) * 32 &&
        words.shape(0) % 128 == 0 && input.shape(2) % 32 == 0,
        "group32 direct U4 MPP requires aligned original row-major affine FP16 planes");
    const int m = input.shape(1), k = input.shape(2), n = words.shape(0), padded = ((m + 31) / 32) * 32;
    Tensor x = m == padded ? input : mx::pad(input, {{0, 0}, {0, padded - m}, {0, 0}});
    static auto sums_kernel = mx::fast::metal_kernel("tc_probe_u4_g32_sums_m32", {"x"}, {"sums"}, R"metal(
        uint origin = threadgroup_position_in_grid.x * 32;
        for (uint g=0;g<K/32;++g) for (uint r=simdgroup_index_in_threadgroup;r<32;r+=4) {
            float value=simd_sum(float(x[(origin+r)*K+g*32+thread_index_in_simdgroup]));
            if(thread_index_in_simdgroup==0) sums[(origin/32)*(K/32)*32+g*32+r]=value;
        }
    )metal");
    auto sums = sums_kernel({x}, {{padded / 32, k / 32, 32}}, {mx::float32},
        {padded / 32 * 128, 1, 1}, {128, 1, 1}, {{"K", k}}, {}, false, {})[0];
    static auto kernel = mx::fast::metal_kernel("tc_probe_u4_g32_direct_m32n128_sg4",
        {"x", "words", "scales", "biases", "sums"}, {"out"}, R"metal(
        using namespace mpp::tensor_ops;
        uint row=threadgroup_position_in_grid.y*32, col=threadgroup_position_in_grid.x*128;
        auto a=tensor(const_cast<device half*>(x),dextents<int,2>{K,M},array<int,2>{1,K});
        auto raw=reinterpret_cast<device uchar*>(const_cast<device uint*>(words));
        tensor<device uint4b_format,dextents<int,2>,tensor_inline> b(raw,dextents<int,2>{K,N},array<int,2>{1,K});
        constexpr auto descriptor=matmul2d_descriptor(32,128,32,false,true,false);
        matmul2d<descriptor,execution_simdgroups<4>> operation;
        auto aa=a.slice<32,32>(0,row); auto bb=b.slice<32,128>(0,col);
        auto accumulated=operation.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
        #pragma unroll
        for(ushort i=0;i<accumulated.get_capacity();++i) accumulated[i]=0;
        for(uint g=0;g<K/32;++g) {
            auto as=a.slice<32,32>(g*32,row); auto bs=b.slice<32,128>(g*32,col);
            auto partial=operation.template get_destination_cooperative_tensor<decltype(as),decltype(bs),float>();
            #pragma unroll
            for(ushort i=0;i<partial.get_capacity();++i) partial[i]=0;
            operation.run(as,bs,partial);
            #pragma unroll
            for(ushort i=0;i<accumulated.get_capacity();++i) {
                auto index=accumulated.get_multidimensional_index(i);
                uint meta=(col+index[0])*(K/32)+g;
                float sum=sums[(row/32)*(K/32)*32+g*32+index[1]];
                accumulated[i]+=partial[i]*float(scales[meta])+sum*float(biases[meta]);
            }
        }
        #pragma unroll
        for(ushort i=0;i<accumulated.get_capacity();++i) {
            auto index=accumulated.get_multidimensional_index(i);
            out[(row+index[1])*N+col+index[0]]=half(accumulated[i]);
        }
    )metal", "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    auto output = kernel({x, words, scales, biases, sums}, {{1, padded, n}}, {mx::float16},
        {n / 128 * 128, padded / 32, 1}, {128, 1, 1}, {{"M", padded}, {"N", n}, {"K", k}}, {}, false, {})[0];
    return m == padded ? output : mx::slice(output, {0, 0, 0}, {1, m, n});
}
} // namespace tc::qwen21_probe

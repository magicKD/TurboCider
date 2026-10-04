#pragma once

#include "affine_fp16.hpp"

namespace tc::z_metal {
// Experimental half operands + FP32 destination MPP. Preserve the native
// FP32 glue; unlike the ordinary half GEMM, do not narrow its dot output.
inline mx::array fp16_mpp_projection(const mx::array &input,const mx::array &dense) {
    if (input.ndim()!=3 || input.shape(0)!=1 || input.dtype()!=mx::float16 || dense.ndim()!=2 ||
        dense.dtype()!=mx::float16 || input.shape(2)!=dense.shape(1))
        throw std::invalid_argument("FP16 MPP projection geometry/dtype unsupported");
    const int m=input.shape(1),n=dense.shape(0),k=input.shape(2);
    const int bm=n==11520 && k==3840 ? 48 : n==3840 && k==3840 ? 16 : 32;
    constexpr int bn=128;
    static auto kernel=mx::fast::metal_kernel("tc_z_affine_f16_mpp_f32",{"x","w"},{"out"},R"metal(
        using namespace mpp::tensor_ops;
        uint row=threadgroup_position_in_grid.y*BM;
        uint col=threadgroup_position_in_grid.x*BN;
        auto a=tensor(const_cast<device half*>(x),dextents<int,2>{K,M},array<int,2>{1,K});
        auto b=tensor(const_cast<device half*>(w),dextents<int,2>{K,N},array<int,2>{1,K});
        auto aa=a.slice(0,row),bb=b.slice(0,col);
        matmul2d<matmul2d_descriptor(BM,BN,K,false,true,false),execution_simdgroups<4>> op;
        auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
        op.run(aa,bb,acc);
        for (uint i=0;i<acc.get_capacity();++i) {
            if (!acc.is_valid_element(i)) continue;
            auto coord=acc.get_multidimensional_index(i);
            if (row+coord[1]<M && col+coord[0]<N) out[(row+coord[1])*N+col+coord[0]]=acc[i];
        }
    )metal", "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    auto product=kernel({input,dense},{{1,m,n}},{mx::float32},
        {((n+bn-1)/bn)*128,(m+bm-1)/bm,1},{128,1,1},
        {{"M",m},{"N",n},{"K",k},{"BM",bm},{"BN",bn}},{},false,{})[0];
    return product;
}
inline mx::array affine_fp16_mpp_matmul(const mx::array &x,const mx::array &codes,
        const mx::array &scales,const mx::array &offsets,int bits,float divisor) {
    if (x.ndim()!=3 || x.shape(0)!=1 || x.dtype()!=mx::float32 || (divisor!=1.f && divisor!=64.f))
        throw std::invalid_argument("GPU affine FP16 MPP input/headroom unsupported");
    auto dense=affine_decode_fp16(codes,scales,offsets,x.shape(2),bits);
    auto input=mx::astype(divisor==1.f ? x : x/divisor,mx::float16);
    auto product=fp16_mpp_projection(input,dense);
    return divisor==1.f ? product : product*divisor;
}
// Independent recipe: per-row FP32 maxabs normalization avoids both a half
// input overflow and a half dot-output boundary. No clipping or A8 codes.
inline mx::array dense_fp16_mpp_dynamic(const mx::array &x,const mx::array &weight,float minimum_divisor) {
    if (x.ndim()!=3 || x.shape(0)!=1 || (x.dtype()!=mx::float32 && x.dtype()!=mx::bfloat16) ||
        weight.ndim()!=2 || weight.dtype()!=mx::float16 || weight.shape(1)!=x.shape(2) ||
        (minimum_divisor!=1.f && minimum_divisor!=64.f))
        throw std::invalid_argument("dynamic FP16 MPP source projection unsupported");
    auto wide=mx::astype(x,mx::float32);
    auto scale=mx::maximum(mx::max(mx::abs(wide),-1,true)/32752.f,mx::array(minimum_divisor,mx::float32));
    auto input=mx::astype(wide/scale,mx::float16);
    return fp16_mpp_projection(input,weight)*scale;
}
} // namespace tc::z_metal

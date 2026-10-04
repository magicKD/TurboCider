#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <limits>
#include <stdexcept>

namespace tc::z_metal {
namespace mx = mlx::core;

// Experimental compute recipe, not W8A8. The owner must prevalidate finite
// metadata and an FP16 reconstruction upper bound before submitting this.
// Direct FP32 q*scale+bias -> one FP16 RNE; no full FP32 matrix temporary.
inline void validate_affine_fp16(const mx::array &codes,const mx::array &scales,
                                const mx::array &offsets,int columns,int bits) {
    if (codes.ndim()!=2 || scales.ndim()!=2 || offsets.shape()!=scales.shape() || columns<=0 ||
        columns%32 || (bits!=4 && bits!=8) || codes.dtype()!=mx::uint32 ||
        scales.dtype()!=mx::float16 || offsets.dtype()!=mx::float16 ||
        scales.shape()!=mx::Shape{codes.shape(0),columns/32} || int64_t(codes.shape(1))!=int64_t(columns)*bits/32 ||
        !codes.flags().row_contiguous || !scales.flags().row_contiguous || !offsets.flags().row_contiguous)
        throw std::invalid_argument("GPU affine FP16 decode geometry/dtype/contiguity mismatch");
    const uint64_t elements=uint64_t(codes.shape(0))*uint64_t(columns);
    if (!elements || elements>uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("GPU affine FP16 decode extent unsupported");
}
inline mx::array affine_decode_fp16(const mx::array &codes, const mx::array &scales,
                                   const mx::array &offsets, int columns, int bits) {
    validate_affine_fp16(codes,scales,offsets,columns,bits);
    const uint64_t elements=uint64_t(codes.shape(0))*uint64_t(columns);
    static auto kernel=mx::fast::metal_kernel("tc_z_affine_decode_f16_rne", {"q","scales","offsets"}, {"out"}, R"metal(
        uint i=thread_position_in_grid.x;
        if (i>=ELEMENTS) return;
        uint word=q[i/(32/BITS)];
        uint code=(word >> ((i%(32/BITS))*BITS)) & ((1u<<BITS)-1u);
        float product=float(code)*float(scales[i/32]);
        float value=product+float(offsets[i/32]);
        out[i]=half(value);
    )metal", "", false, false, {});
    return kernel({codes,scales,offsets}, {{codes.shape(0),columns}}, {mx::float16},
        {int(elements),1,1}, {256,1,1}, {{"BITS",bits},{"ELEMENTS",int(elements)}}, {},false,{})[0];
}
// Native packed QMM with explicitly narrowed input/output; NO dense W decode.
inline mx::array affine_qmm_fp16_matmul(const mx::array &x,const mx::array &codes,
        const mx::array &scales,const mx::array &offsets,int bits,float divisor) {
    if (x.ndim()!=3 || x.shape(0)!=1 || x.dtype()!=mx::float32 || (divisor!=1.f && divisor!=64.f))
        throw std::invalid_argument("packed FP16 QMM input/headroom unsupported");
    validate_affine_fp16(codes,scales,offsets,x.shape(2),bits);
    auto input=mx::astype(divisor==1.f ? x : x/divisor,mx::float16);
    auto result=mx::astype(mx::quantized_matmul(input,codes,scales,offsets,true,32,bits,"affine"),mx::float32);
    return divisor==1.f ? result : result*divisor;
}
inline mx::array dense_fp16_matmul(const mx::array &x,const mx::array &weight,float divisor) {
    if (x.ndim()!=3 || x.shape(0)!=1 || (x.dtype()!=mx::float32 && x.dtype()!=mx::bfloat16) ||
        weight.ndim()!=2 || weight.dtype()!=mx::float16 || weight.shape(1)!=x.shape(2) ||
        (divisor!=1.f && divisor!=64.f))
        throw std::invalid_argument("source-float FP16 projection geometry/dtype/headroom unsupported");
    auto input=mx::astype(divisor==1.f ? x : mx::astype(x,mx::float32)/divisor,mx::float16);
    auto result=mx::astype(mx::matmul(input,mx::transpose(weight)),mx::float32);
    return divisor==1.f ? result : result*divisor;
}

// Restore the old native QMM result dtype before bias/nonlinearity. Down's
// divisor=64 is explicit, before narrowing; restoration is FP32, never clamp.
inline mx::array affine_fp16_matmul(const mx::array &x, const mx::array &codes,
        const mx::array &scales, const mx::array &offsets, int bits, float divisor) {
    if (x.ndim()<2 || (x.dtype()!=mx::float32 && x.dtype()!=mx::bfloat16 && x.dtype()!=mx::float16) ||
        (divisor!=1.f && divisor!=64.f))
        throw std::invalid_argument("GPU affine FP16 GEMM input/headroom unsupported");
    auto dense=affine_decode_fp16(codes,scales,offsets,x.shape(-1),bits);
    auto input=mx::astype(divisor==1.f ? x : mx::astype(x,mx::float32)/divisor,mx::float16);
    auto product=mx::astype(mx::matmul(input,mx::transpose(dense)),mx::float32);
    if (divisor!=1.f) product=product*divisor;
    return mx::astype(product,mx::promote_types(x.dtype(),scales.dtype()));
}
} // namespace tc::z_metal

#include "models/z_image/metal/affine_fp16.hpp"
#include "models/z_image/metal/affine_fp16_mpp.hpp"
#include "backends/mlx.hpp"
#include "core/gguf_decode.hpp"
#include <cmath>
#include <cstring>
#include <iostream>

namespace {
using namespace tc;
void check(bool value,const char *reason) { if (!value) throw std::runtime_error(reason); }
void test(int bits,int rows,int columns) {
    const int lanes=32/bits,groups=rows*columns/32;
    std::vector<uint32_t> words(size_t(rows)*columns/lanes,0);
    std::vector<uint16_t> scales(groups),offsets(groups),expected(size_t(rows)*columns);
    const float options[]={0.f,-0.f,0.0039215087890625f,-0.0625f,0.125f,0.000000059604644775390625f,1.0078125f};
    for (int i=0;i<groups;++i) {
        const float s=options[i%7];scales[i]=gguf::float_to_fp16_rne(s);
        offsets[i]=gguf::float_to_fp16_rne(-(bits==8 ? 128.f : 8.f)*s);
    }
    for (int i=0;i<rows*columns;++i) {
        const uint32_t code=uint32_t(i*37+19)&((1u<<bits)-1u);
        words[size_t(i)/lanes]|=code<<((i%lanes)*bits);
        const float s=gguf::fp16_to_float(scales[i/32]),b=gguf::fp16_to_float(offsets[i/32]);
        expected[i]=gguf::float_to_fp16_rne(float(code)*s+b);
    }
    std::vector<mx::float16_t> scale_values(groups),offset_values(groups),dense_values(expected.size());
    std::memcpy(scale_values.data(),scales.data(),scales.size()*2);
    std::memcpy(offset_values.data(),offsets.data(),offsets.size()*2);
    std::memcpy(dense_values.data(),expected.data(),expected.size()*2);
    Tensor q(words.data(),{rows,columns/lanes},mx::uint32),s(scale_values.data(),{rows,columns/32},mx::float16),
           b(offset_values.data(),{rows,columns/32},mx::float16);
    auto decoded=z_metal::affine_decode_fp16(q,s,b,columns,bits);mx::eval(decoded);
    const auto *actual=decoded.data<uint16_t>();
    for (size_t i=0;i<expected.size();++i)
        check(actual[i]==expected[i] || (!(actual[i]&0x7fff) && !(expected[i]&0x7fff)),"GPU affine RNE differs from scalar oracle");
    std::vector<float> values(size_t(5)*columns);
    for (size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.013f)*31.f;
    Tensor input(values.data(),{1,5,columns},mx::float32),dense(dense_values.data(),{rows,columns},mx::float16);
    for (float divisor:{1.f,64.f}) {
        auto result=z_metal::affine_fp16_matmul(input,q,s,b,bits,divisor);
        auto reference=mx::astype(mx::matmul(mx::astype(input/divisor,mx::float16),mx::transpose(dense)),mx::float32)*divisor;
        mx::eval({result,reference});
        check(result.dtype()==mx::float32 && mx::all(result==reference).item<bool>(),"GPU FP16 GEMM/restoration differs from same-order oracle");
        auto mpp=z_metal::affine_fp16_mpp_matmul(input,q,s,b,bits,divisor);
        auto wide_input=mx::astype(mx::astype(input/divisor,mx::float16),mx::float32);
        auto wide_reference=mx::matmul(wide_input,mx::transpose(mx::astype(dense,mx::float32)))*divisor;
        mx::eval({mpp,wide_reference});
        const float error=mx::max(mx::abs(mpp-wide_reference)).item<float>();
        const float magnitude=mx::max(mx::abs(wide_reference)).item<float>();
        check(mx::all(mx::isfinite(mpp)).item<bool>() && error<=std::max(1e-5f,magnitude*1e-5f),
              "FP16 MPP FP32 accumulator differs from FP32 same-operand reference");
        auto packed=z_metal::affine_qmm_fp16_matmul(input,q,s,b,bits,divisor);
        auto packed_reference=mx::astype(mx::quantized_matmul(mx::astype(input/divisor,mx::float16),
            q,s,b,true,32,bits,"affine"),mx::float32)*divisor;
        mx::eval({packed,packed_reference});
        check(packed.dtype()==mx::float32 && mx::all(packed==packed_reference).item<bool>(),
              "packed FP16 QMM differs from declared narrowing/restoration order");
        auto dynamic=z_metal::dense_fp16_mpp_dynamic(input,dense,divisor);
        mx::eval(dynamic);
        check(mx::all(mx::isfinite(dynamic)).item<bool>() &&
            mx::max(mx::abs(dynamic-wide_reference)).item<float>()<=std::max(1e-5f,magnitude*1e-5f),
            "dynamic FP16 projection changed declared same-operand normal range");
    }
    for (const char *fault: {"bits","headroom","dtype","metadata","columns"}) {
        bool rejected=false;
        try {
            if (!std::strcmp(fault,"bits")) (void)z_metal::affine_decode_fp16(q,s,b,columns,6);
            if (!std::strcmp(fault,"headroom")) (void)z_metal::affine_fp16_matmul(input,q,s,b,bits,32.f);
            if (!std::strcmp(fault,"dtype")) (void)z_metal::affine_decode_fp16(q,mx::astype(s,mx::float32),b,columns,bits);
            if (!std::strcmp(fault,"metadata")) (void)z_metal::affine_decode_fp16(q,s,mx::zeros({rows,columns/32+1},mx::float16),columns,bits);
            if (!std::strcmp(fault,"columns")) (void)z_metal::affine_decode_fp16(q,s,b,columns-1,bits);
        } catch (const std::invalid_argument &) {rejected=true;}
        check(rejected,"invalid GPU affine FP16 configuration accepted");
    }
    auto huge=input*1e8f;
    auto dynamic=z_metal::dense_fp16_mpp_dynamic(huge,dense,64.f);
    auto scale=mx::maximum(mx::max(mx::abs(huge),-1,true)/32752.f,Tensor(64.f));
    auto normalized=mx::astype(mx::astype(huge/scale,mx::float16),mx::float32);
    auto reference=mx::matmul(normalized,mx::transpose(mx::astype(dense,mx::float32)))*scale;
    mx::eval({dynamic,reference});
    const float magnitude=mx::max(mx::abs(reference)).item<float>();
    check(mx::all(mx::isfinite(dynamic)).item<bool>() &&
          mx::max(mx::abs(dynamic-reference)).item<float>()<=std::max(1e-5f,magnitude*1e-5f),
          "dynamic FP16 normalization overflowed/clipped or changed FP32 restore order");
}
}
int main() {
    try {
        tc::configure_streams();mlx::core::set_cache_limit(0);
        for (int bits:{4,8}) for (int rows:{1,7,33}) for (int columns:{32,64,128}) test(bits,rows,columns);
        std::cout<<"PASS GPU affine FP16: 18 shapes, scalar RNE, matmul/MPP/packed QMM FP32 restore, invalid geometry/headroom\n";
    } catch (const std::exception &error) { std::cerr<<error.what()<<'\n';return 1; }
}

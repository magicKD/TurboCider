#pragma once
// Projection-only prototype. Original signed I8 row codes are exactly
// representable as BF16/FP16 integers. Decode only a bounded TG tile, perform
// the integer-valued-weight dot with F32 accumulation, THEN apply the original
// F32 row scale. This reorders F32 arithmetic; it is not globally bit-exact.
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <limits>
#include <stdexcept>

namespace tc::convrot_factorized_lab {
namespace mx=mlx::core;
inline mx::array projection(const mx::array&x,const mx::array&words,const mx::array&row_scale,
    int rb,int re,int cb,int ce,int bm=32,int bn=64,int bk=64,mx::Dtype output=mx::float32) {
    if(x.ndim()!=3 || x.shape(0)!=1 || x.shape(1)<1 || !x.flags().row_contiguous ||
        (x.dtype()!=mx::bfloat16 && x.dtype()!=mx::float16) || words.ndim()!=2 || words.dtype()!=mx::uint32 ||
        !words.flags().row_contiguous || words.shape(1)<1 || words.shape(1)>std::numeric_limits<int>::max()/4 ||
        row_scale.ndim()!=2 || row_scale.shape(0)!=words.shape(0) || row_scale.shape(1)!=1 ||
        row_scale.dtype()!=mx::float32 || !row_scale.flags().row_contiguous || rb<0 || re<=rb || re>words.shape(0) ||
        cb<0 || ce<=cb || ce>words.shape(1)*4 || cb%32 || ce%32 || x.shape(2)!=ce-cb ||
        (bm!=32 && bm!=64) || (bn!=64 && bn!=128) || (bk!=32 && bk!=64 && bk!=128) ||
        (output!=mx::float32 && output!=x.dtype()) || int64_t(x.shape(1))*(re-rb)>std::numeric_limits<int>::max())
        throw std::invalid_argument("factored Q8 needs original row F32 scales, signed-q+128 codes and exact bounded physical ranges");
    const int m=x.shape(1),n=re-rb,k=ce-cb,wp=words.shape(1)*4;
    const int tile_k=k%bk?32:bk;
    static auto kernel=mx::fast::metal_kernel("tc_qwen_convrot_row_scale_factored_candidate",
        {"x","words","row_scale"},{"out"},R"metal(
        using namespace mpp::tensor_ops;
        const uint row=threadgroup_position_in_grid.y*BM,col=threadgroup_position_in_grid.x*BN;
        threadgroup T tile[BN*BK];
        auto a=tensor(const_cast<device T*>(x),dextents<int,2>{K,M},array<int,2>{1,K});
        auto b=tensor(tile,dextents<int,2>{BK,BN},array<int,2>{1,BK});
        matmul2d<matmul2d_descriptor(BM,BN,BK,false,true,false,
            matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<4>> op;
        auto aa=a.slice(0,row);
        auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(b),float>();
        for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0;
        for(uint kb=0;kb<K;kb+=BK) {
            for(uint q=thread_position_in_threadgroup.x;q<BN*(BK/4);q+=128) {
                const uint r=col+q/(BK/4),local_c=(q%(BK/4))*4,c=kb+local_c;
                uint word=0;
                if(r<N && c<K)word=words[(RB+r)*(WP/4)+(CB+c)/4];
                #pragma unroll
                for(uint j=0;j<4;++j) {
                    const int signed_code=int((word>>(j*8))&255u)-128;
                    tile[(q/(BK/4))*BK+local_c+j]=(r<N && c+j<K)?T(signed_code):T(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto input=a.slice(kb,row);op.run(input,b,acc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for(uint i=0;i<acc.get_capacity();++i) {
            if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);
            if(row+c[1]<M && col+c[0]<N)
                out[(row+c[1])*N+col+c[0]]=O(acc[i]*row_scale[RB+col+c[0]]);
        }
    )metal","#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    return kernel({x,words,row_scale},{{1,m,n}},{output},
        {((n+bn-1)/bn)*128,(m+bm-1)/bm,1},{128,1,1},
        {{"T",x.dtype()},{"O",output},{"M",m},{"N",n},{"K",k},{"WP",wp},{"RB",rb},{"CB",cb},
         {"BM",bm},{"BN",bn},{"BK",tile_k}}, {},false,{})[0];
}
} // namespace tc::convrot_factorized_lab

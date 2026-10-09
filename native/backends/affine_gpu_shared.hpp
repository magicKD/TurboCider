#pragma once
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <limits>
#include <stdexcept>

namespace tc::affine_gpu {
namespace mx=mlx::core;

inline int shared_effective_k_tile(int k,int requested) {
    if(k<=0 || k%32 || (requested!=32 && requested!=64 && requested!=128))
        throw std::invalid_argument("shared affine K tile requires aligned32 positive K and requested32/64/128");
    // A regular-device left tensor with a partial BK64/128 reduction can
    // read nonfinite padding on this SDK. Every K32 tile is fully in-range.
    // This is a declared tail policy, not a measured large-tile substitution.
    return k%requested?32:requested;
}

// Research-only: one typed decode per packed word into a bounded TG tile.
// All four SIMD groups consume that tile; no global dense weight bank,
// widened metadata, inverse basis conversion or layer-ahead prefetch.
inline mx::array projection_shared(const mx::array &x,const mx::array &words,
    const mx::array &scales,const mx::array &biases,int bits,int rb,int re,int cb,int ce,
    int bm=64,int bn=64,int bk=64,mx::Dtype output_dtype=mx::float32) {
    if(x.ndim()!=3 || x.shape(0)!=1 || x.shape(1)<32 || words.ndim()!=2 || words.shape(0)<=0 || words.shape(1)<=0 ||
        scales.ndim()!=2 || scales.shape(1)<=0 || scales.shape(0)!=words.shape(0) || biases.shape()!=scales.shape() ||
        (bits!=4 && bits!=8) || words.dtype()!=mx::uint32 || !words.flags().row_contiguous ||
        (x.dtype()!=mx::float16 && x.dtype()!=mx::bfloat16) || scales.dtype()!=x.dtype() || biases.dtype()!=x.dtype() ||
        (bm!=32 && bm!=64 && bm!=128) || (bn!=64 && bn!=128) || (bk!=32 && bk!=64 && bk!=128) ||
        (output_dtype!=mx::float32 && output_dtype!=x.dtype()))
        throw std::invalid_argument("shared affine projection requires original typed Q4/Q8, M>=32 and bounded BM/BN/BK");
    const int pack=32/bits;
    if(words.shape(1)>std::numeric_limits<int>::max()/pack)
        throw std::invalid_argument("shared affine physical width overflow");
    const int wp=words.shape(1)*pack,groups=scales.shape(1),group=wp/groups;
    if(wp%groups || (group!=32 && group!=64 && group!=128) || rb<0 || re<=rb || re>words.shape(0) ||
        cb<0 || ce<=cb || ce>wp || cb%group || ce%group || x.shape(2)!=ce-cb)
        throw std::invalid_argument("shared affine physical row/column/group mismatch");
    const int m=x.shape(1),n=re-rb,k=ce-cb;
    const int effective_bk=shared_effective_k_tile(k,bk);
    static auto kernel=mx::fast::metal_kernel("tc_affine_word_shared_decode_candidate",
        {"x","words","scales","biases"},{"out"},R"metal(
        using namespace mpp::tensor_ops;
        constexpr uint PACK=32/BITS;
        uint row=threadgroup_position_in_grid.y*BM,col=threadgroup_position_in_grid.x*BN;
        threadgroup T tile[BN*BK];
        auto a=tensor(const_cast<device T*>(x),dextents<int,2>{K,M},array<int,2>{1,K});
        auto b=tensor(tile,dextents<int,2>{BK,BN},array<int,2>{1,BK});
        matmul2d<matmul2d_descriptor(BM,BN,BK,false,true,false,
            matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<4>> op;
        auto aa=a.slice(0,row);
        auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(b),float>();
        for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0;
        for(uint kb=0;kb<K;kb+=BK) {
            for(uint q=thread_position_in_threadgroup.x;q<BN*(BK/PACK);q+=128) {
                uint r=col+q/(BK/PACK),local_c=(q%(BK/PACK))*PACK,c=kb+local_c;
                uint word=0;T scale=T(0),bias=T(0);
                if(r<N && c<K) {
                    uint physical_row=RB+r,physical_col=CB+c;
                    word=words[physical_row*(WP/PACK)+physical_col/PACK];
                    uint meta=physical_row*(WP/GROUP)+physical_col/GROUP;
                    scale=scales[meta];bias=biases[meta];
                }
                #pragma unroll
                for(uint j=0;j<PACK;++j) {
                    uchar code=uchar((word>>(j*BITS))&((1u<<BITS)-1));
                    tile[(q/(BK/PACK))*BK+local_c+j]=(r<N && c+j<K)?T(scale*code+bias):T(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto input=a.slice(kb,row);
            op.run(input,b,acc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for(uint i=0;i<acc.get_capacity();++i) {
            if(!acc.is_valid_element(i))continue;
            auto coord=acc.get_multidimensional_index(i);
            if(row+coord[1]<M && col+coord[0]<N)out[(row+coord[1])*N+col+coord[0]]=O(acc[i]);
        }
    )metal","#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    return kernel({x,words,scales,biases},{{1,m,n}},{output_dtype},
        {((n+bn-1)/bn)*128,(m+bm-1)/bm,1},{128,1,1},
        {{"T",x.dtype()},{"O",output_dtype},{"M",m},{"N",n},{"K",k},{"WP",wp},{"GROUP",group},
         {"BITS",bits},{"RB",rb},{"CB",cb},{"BM",bm},{"BN",bn},{"BK",effective_bk}}, {},false,{})[0];
}
}

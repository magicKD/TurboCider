#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::affine_gpu {
namespace mx=mlx::core;

// Research-only fused affine decode/GEMM. Decode each coefficient in the
// ORIGINAL FP16/BF16 metadata dtype into an MPP right-input register tile;
// accumulate in F32, without a global dense bank or shared-memory W tile.
// Raw GGUF/Comfy must first use their original explicit affine importer.
// No default route, activation quantization or integer-MAC/overlap claim.
inline mx::array projection_mpp_fp32(const mx::array &x,const mx::array &words,
    const mx::array &scales,const mx::array &biases,int bits,
    int row_begin,int row_end,int col_begin,int col_end,int bk=32,int sn=32,int sm=16) {
    if(x.ndim()!=3 || x.shape(0)!=1 || x.shape(1)<32 || words.ndim()!=2 ||
       scales.ndim()!=2 || biases.shape()!=scales.shape() || (bits!=4&&bits!=8) ||
       words.dtype()!=mx::uint32 || (x.dtype()!=mx::bfloat16&&x.dtype()!=mx::float16) ||
       scales.dtype()!=x.dtype() || biases.dtype()!=x.dtype() || !words.flags().row_contiguous ||
       scales.shape(0)!=words.shape(0) || scales.shape(1)<=0 ||
       (bk!=32&&bk!=64) || (sn!=32&&sn!=64) || (bk==64&&sn==64) ||
       (sm!=16&&sm!=32&&sm!=64) || (sm!=16 && (bk!=32 || sn!=32)))
        throw std::invalid_argument("MPP affine partial requires same-dtype Q4/Q8, M>=32 and validated SM/BK/SN tiles");
    // BK64/SN64 failed the actual numerical screen on the current SDK/device.
    // Do not silently substitute a tile or admit it on compile success alone.
    const int physical=words.shape(1)*(32/bits),groups=scales.shape(1);
    const int group=physical/groups;
    if(physical%groups || (group!=32&&group!=64&&group!=128) ||
       row_begin<0 || row_end<=row_begin || row_end>words.shape(0) || col_begin<0 || col_end<=col_begin ||
       col_end>physical || col_begin%group || col_end%group || x.shape(2)!=col_end-col_begin)
        throw std::invalid_argument("MPP affine partial physical range/group mismatch");
    const int m=x.shape(1),n=row_end-row_begin,k=col_end-col_begin,bn=2*sn;
    static auto kernel=mx::fast::metal_kernel("tc_affine_mpp_register_decode_fp32_candidate",
        {"x","words","scales","biases"},{"out"},R"metal(
        using namespace mpp::tensor_ops;
        constexpr uint BM=2*SM;
        uint warp=simdgroup_index_in_threadgroup;
        uint row=threadgroup_position_in_grid.y*BM+(warp/2)*SM;
        uint col=threadgroup_position_in_grid.x*(2*SN)+(warp%2)*SN;
        auto a=tensor(const_cast<device T*>(x),dextents<int,2>{K,M},array<int,2>{1,K});
        matmul2d<matmul2d_descriptor(SM,SN,BK,false,true,false,
            matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroup> op;
        auto b=op.template get_right_input_cooperative_tensor<T,T,float>();
        auto aa=a.slice(0,row);
        auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(b),float>();
        #pragma unroll
        for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0;
        for(uint kb=0;kb<K;kb+=BK) {
            #pragma unroll
            for(uint i=0;i<b.get_capacity();++i) {
                if(!b.is_valid_element(i))continue;
                auto coord=b.get_multidimensional_index(i);
                uint r=col+coord[1],c=kb+coord[0];
                T decoded=T(0);
                if(r<N && c<K) {
                    uint physical_row=ROW_BEGIN+r,physical_col=COL_BEGIN+c;
                    uint word=words[physical_row*(WP/(32/BITS))+physical_col/(32/BITS)];
                    uchar code=uchar((word>>((physical_col%(32/BITS))*BITS))&((1u<<BITS)-1));
                    uint meta=physical_row*(WP/GROUP)+physical_col/GROUP;
                    // Same typed affine expression as the original decoder.
                    // Widening metadata first would define different weights.
                    decoded=scales[meta]*code+biases[meta];
                }
                b[i]=decoded;
            }
            auto input=a.slice(kb,row);
            op.run(input,b,acc);
        }
        #pragma unroll
        for(uint i=0;i<acc.get_capacity();++i) {
            if(!acc.is_valid_element(i))continue;
            auto coord=acc.get_multidimensional_index(i);
            if(row+coord[1]<M && col+coord[0]<N)out[(row+coord[1])*N+col+coord[0]]=acc[i];
        }
    )metal","#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    return kernel({x,words,scales,biases},{{1,m,n}},{mx::float32},
        {((n+bn-1)/bn)*128,(m+2*sm-1)/(2*sm),1},{128,1,1},
        {{"T",x.dtype()},{"M",m},{"N",n},{"K",k},{"WP",physical},{"GROUP",group},{"BITS",bits},
         {"ROW_BEGIN",row_begin},{"COL_BEGIN",col_begin},{"BK",bk},{"SN",sn},{"SM",sm}}, {},false,{})[0];
}
} // namespace tc::affine_gpu

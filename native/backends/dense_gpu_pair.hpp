#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::dense_gpu {
namespace mx=mlx::core;

// One dispatch for two same-width output-row intervals of the SAME immutable
// physical dense source. No gather/concatenate/compact W bank. Each interval
// has its own row bound; complete BN alignment prevents a tile crossing the
// gate/up seam. Original operand dtype, F32 accumulation, one output cast.
// Research consumer only; not an implicit model/default replacement.
inline mx::array projection_row_pair(const mx::array &x,const mx::array &w,
    int first0,int first1,int count,int col_begin,int col_end,int bm=32,int bn=128,
    bool retain_fp32=false) {
    if(x.ndim()!=3 || x.shape(0)!=1 || x.shape(1)<=0 || w.ndim()!=2 || first0<0 || first1<0 ||
        count<=0 || first0>w.shape(0)-count || first1>w.shape(0)-count ||
        col_begin<0 || col_end<=col_begin || col_end>w.shape(1) || x.shape(2)!=col_end-col_begin ||
        !w.flags().row_contiguous || (x.dtype()!=mx::bfloat16&&x.dtype()!=mx::float16) || w.dtype()!=x.dtype() ||
        (bm!=16&&bm!=32&&bm!=64) || (bn!=64&&bn!=128) || count%bn)
        throw std::invalid_argument("paired physical projection requires same-dtype FP16/BF16, valid ranges and aligned pair seam");
    const int m=x.shape(1),k=col_end-col_begin,pitch=w.shape(1);
    const auto output_dtype=retain_fp32?mx::float32:x.dtype();
    static auto kernel=mx::fast::metal_kernel("tc_dense_projection_row_pair_candidate",{"x","w"},{"out"},R"metal(
        using namespace mpp::tensor_ops;
        uint row=threadgroup_position_in_grid.y*BM,col=threadgroup_position_in_grid.x*BN;
        uint segment=col/COUNT,local_col=col%COUNT;
        uint first=segment?FIRST1:FIRST0;
        auto a=tensor(const_cast<device T*>(x),dextents<int,2>{K,M},array<int,2>{1,K});
        auto b=tensor(const_cast<device T*>(w)+first*WP+COL_BEGIN,dextents<int,2>{K,COUNT},array<int,2>{1,WP});
        auto aa=a.slice(0,row),bb=b.slice(0,local_col);
        matmul2d<matmul2d_descriptor(BM,BN,K,false,true,false),execution_simdgroups<4>> op;
        auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
        for(uint i=0;i<acc.get_capacity();++i) {
            if(!acc.is_valid_element(i))continue;
            auto coord=acc.get_multidimensional_index(i);
            if(row+coord[1]<M && local_col+coord[0]<COUNT)out[(row+coord[1])*(2*COUNT)+col+coord[0]]=O(acc[i]);
        }
    )metal","#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    return kernel({x,w},{{1,m,2*count}},{output_dtype},
        {(2*count/bn)*128,(m+bm-1)/bm,1},{128,1,1},
        {{"T",x.dtype()},{"O",output_dtype},{"M",m},{"K",k},{"WP",pitch},{"COUNT",count},
         {"FIRST0",first0},{"FIRST1",first1},{"COL_BEGIN",col_begin},{"BM",bm},{"BN",bn}}, {},false,{})[0];
}
} // namespace tc::dense_gpu

#pragma once
#include "backends/mlx.hpp"
#include <mlx/fast.h>

namespace tc::research::lora_pair {
// Two ORIGINAL independent A matrices; contiguous F32 outputs keep both B
// consumers unchanged. One dispatch is not proof of one physical X read.
// Standalone candidate, no model/default route or persistent packed bank.
inline std::vector<Tensor> ranks(const Tensor &x,const Tensor &a0,const Tensor &a1,
    int begin,int end,int bm=16,int bn=64,bool same_group=false) {
    require(x.ndim()==3 && x.shape(0)==1 && x.shape(1)>0 && a0.ndim()==2 && a1.ndim()==2 &&
        a0.shape(0)>0 && a0.shape(0)==a1.shape(0) && (x.dtype()==mx::bfloat16 || x.dtype()==mx::float16) &&
        a0.dtype()==x.dtype() && a1.dtype()==x.dtype() && a0.flags().row_contiguous && a1.flags().row_contiguous &&
        begin>=0 && end>begin && end<=a0.shape(1) && end<=a1.shape(1) && x.shape(2)==end-begin &&
        (bm==16 || bm==32 || bm==64) && (bn==64 || bn==128) && a0.shape(0)%bn==0,
        "paired LoRA A requires original typed equal ranks, valid physical columns and bounded aligned tiles");
    static auto kernel=mx::fast::metal_kernel("tc_research_two_source_lora_ranks",{"x","a0","a1"},{"out0","out1"},R"metal(
        using namespace mpp::tensor_ops;
        uint row=threadgroup_position_in_grid.y*BM,col=threadgroup_position_in_grid.x*BN;
        auto left=tensor(const_cast<device T*>(x),dextents<int,2>{K,M},array<int,2>{1,K});auto aa=left.slice(0,row);
        matmul2d<matmul2d_descriptor(BM,BN,K,false,true,false),execution_simdgroups<4>> op;
        if constexpr(SAME_GROUP) {
            auto b0=tensor(const_cast<device T*>(a0)+BEGIN,dextents<int,2>{K,RANK},array<int,2>{1,PITCH0});
            auto b1=tensor(const_cast<device T*>(a1)+BEGIN,dextents<int,2>{K,RANK},array<int,2>{1,PITCH1});
            auto bb0=b0.slice(0,col),bb1=b1.slice(0,col);
            auto acc0=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb0),float>();
            auto acc1=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb1),float>();
            op.run(aa,bb0,acc0);op.run(aa,bb1,acc1);
            for(uint i=0;i<acc0.get_capacity();++i) {
                if(!acc0.is_valid_element(i))continue;auto coordinate=acc0.get_multidimensional_index(i);
                if(row+coordinate[1]<M && col+coordinate[0]<RANK) {
                    uint index=(row+coordinate[1])*RANK+col+coordinate[0];out0[index]=acc0[i];out1[index]=acc1[i];
                }
            }
        } else {
            uint segment=col/RANK,local=col%RANK;
            auto right=tensor(const_cast<device T*>(segment?a1:a0)+BEGIN,dextents<int,2>{K,RANK},array<int,2>{1,segment?PITCH1:PITCH0});
            auto bb=right.slice(0,local);auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
            for(uint i=0;i<acc.get_capacity();++i) {
                if(!acc.is_valid_element(i))continue;auto coordinate=acc.get_multidimensional_index(i);
                if(row+coordinate[1]<M && local+coordinate[0]<RANK) {
                    uint index=(row+coordinate[1])*RANK+local+coordinate[0];
                    if(segment)out1[index]=acc[i];else out0[index]=acc[i];
                }
            }
        }
    )metal","#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    const int m=x.shape(1),rank=a0.shape(0),k=end-begin;
    return kernel({x,a0,a1},{{1,m,rank},{1,m,rank}},{mx::float32,mx::float32},
        {(same_group?rank:2*rank)/bn*128,(m+bm-1)/bm,1},{128,1,1},
        {{"T",x.dtype()},{"M",m},{"RANK",rank},{"K",k},{"BEGIN",begin},{"PITCH0",a0.shape(1)},{"PITCH1",a1.shape(1)},
         {"BM",bm},{"BN",bn},{"SAME_GROUP",same_group}}, {},false,{});
}
} // namespace tc::research::lora_pair

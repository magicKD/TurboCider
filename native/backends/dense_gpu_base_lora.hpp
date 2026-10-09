#pragma once
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <cmath>
#include <stdexcept>

namespace tc::dense_gpu {
namespace mx=mlx::core;

// One dispatch, two cooperative matmuls: original dense base is narrowed to
// the model dtype in a small TG tile BEFORE the F32 scaled B contribution.
// F32 ranks are narrowed as in the admitted B epilogue recipe. No global
// base/delta output, W compaction, adapter merge or changed alpha order.
inline mx::array base_lora_epilogue(const mx::array &x,const mx::array &w,
    const mx::array &ranks,const mx::array &up,int rb,int re,int cb,int ce,
    int first,float scale,int bm=32,int bn=128) {
    if(x.ndim()!=3 || x.shape(0)!=1 || x.shape(1)<=0 || w.ndim()!=2 ||
        rb<0 || re<=rb || re>w.shape(0) || cb<0 || ce<=cb || ce>w.shape(1) || x.shape(2)!=ce-cb ||
        !w.flags().row_contiguous || (x.dtype()!=mx::bfloat16 && x.dtype()!=mx::float16) || w.dtype()!=x.dtype() ||
        ranks.ndim()!=3 || ranks.shape(0)!=1 || ranks.shape(1)!=x.shape(1) || ranks.shape(2)<=0 || ranks.dtype()!=mx::float32 ||
        up.ndim()!=2 || up.dtype()!=x.dtype() || !up.flags().row_contiguous || ranks.shape(2)!=up.shape(1) ||
        first<0 || first>up.shape(0)-(re-rb) || !std::isfinite(scale) ||
        (bm!=16 && bm!=32 && bm!=64) || (bn!=64 && bn!=128))
        throw std::invalid_argument("base/LoRA fused physical geometry/dtype/range mismatch");
    const int m=x.shape(1),n=re-rb,k=ce-cb,r=ranks.shape(2),pitch=w.shape(1);
    static auto kernel=mx::fast::metal_kernel("tc_base_lora_two_matmul_epilogue",{"x","w","ranks","up","scale"},{"out"},R"metal(
        using namespace mpp::tensor_ops;
        uint row=threadgroup_position_in_grid.y*BM,col=threadgroup_position_in_grid.x*BN;
        threadgroup T base_tile[BM*BN];
        {
            auto a=tensor(const_cast<device T*>(x),dextents<int,2>{K,M},array<int,2>{1,K});
            auto b=tensor(const_cast<device T*>(w)+RB*WP+CB,dextents<int,2>{K,N},array<int,2>{1,WP});
            auto aa=a.slice(0,row),bb=b.slice(0,col);
            matmul2d<matmul2d_descriptor(BM,BN,K,false,true,false),execution_simdgroups<4>> op;
            auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
            for(uint i=0;i<acc.get_capacity();++i) {
                if(!acc.is_valid_element(i))continue;
                auto c=acc.get_multidimensional_index(i);
                if(row+c[1]<M && col+c[0]<N)base_tile[c[1]*BN+c[0]]=T(acc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a=tensor(const_cast<device T*>(ranks),dextents<int,2>{R,M},array<int,2>{1,R});
        auto b=tensor(const_cast<device T*>(up)+FIRST*R,dextents<int,2>{R,N},array<int,2>{1,R});
        auto aa=a.slice(0,row),bb=b.slice(0,col);
        matmul2d<matmul2d_descriptor(BM,BN,R,false,true,false),execution_simdgroups<4>> op;
        auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
        for(uint i=0;i<acc.get_capacity();++i) {
            if(!acc.is_valid_element(i))continue;
            auto c=acc.get_multidimensional_index(i);
            if(row+c[1]<M && col+c[0]<N)
                out[(row+c[1])*N+col+c[0]]=T(float(base_tile[c[1]*BN+c[0]])+acc[i]*scale);
        }
    )metal","#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    return kernel({x,w,mx::astype(ranks,up.dtype()),up,mx::array(scale,mx::float32)},{{1,m,n}},{x.dtype()},
        {((n+bn-1)/bn)*128,(m+bm-1)/bm,1},{128,1,1},
        {{"T",x.dtype()},{"M",m},{"N",n},{"K",k},{"R",r},{"WP",pitch},{"RB",rb},{"CB",cb},{"FIRST",first},{"BM",bm},{"BN",bn}}, {},false,{})[0];
}
} // namespace tc::dense_gpu

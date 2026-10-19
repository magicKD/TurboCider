#pragma once
// Research only: FP16 X, ORIGINAL BF16 W, F32 partial. No W cast/repack.
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::mixed_hidden_lab {
namespace mx=mlx::core;
inline mx::array down(const mx::array&x,const mx::array&w,int rb,int re,int cb,int ce){
    if(x.ndim()!=3||x.shape(0)!=1||x.shape(1)<1||w.ndim()!=2||!x.flags().row_contiguous||!w.flags().row_contiguous||
       (x.dtype()!=mx::float16&&x.dtype()!=mx::bfloat16)||w.dtype()!=mx::bfloat16||rb<0||re<=rb||re>w.shape(0)||
       cb<0||ce<=cb||ce>w.shape(1)||x.shape(2)!=ce-cb)
        throw std::invalid_argument("mixed down needs contiguous FP16/BF16 X, original BF16 W and valid physical range");
    const int m=x.shape(1),n=re-rb,k=ce-cb,pitch=w.shape(1);
    static auto kernel=mx::fast::metal_kernel("tc_qwen_mixed_hidden_down_v1",{"x","w"},{"out"},R"metal(
        using namespace mpp::tensor_ops;
        uint row=threadgroup_position_in_grid.y*32,col=threadgroup_position_in_grid.x*128;
        auto a=tensor(const_cast<device A*>(x),dextents<int,2>{K,M},array<int,2>{1,K});
        auto b=tensor(const_cast<device B*>(w)+RB*WP+CB,dextents<int,2>{K,N},array<int,2>{1,WP});
        auto aa=a.slice(0,row);
        auto bb=b.slice(0,col);
        matmul2d<matmul2d_descriptor(32,128,K,false,true,false),execution_simdgroups<4>> op;
        auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
        for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);
            if(row+c[1]<M&&col+c[0]<N)out[(row+c[1])*N+col+c[0]]=acc[i];}
    )metal","#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    return kernel({x,w},{{1,m,n}},{mx::float32},{uint32_t((n+127)/128*128),uint32_t((m+31)/32),1},{128,1,1},
        {{"A",x.dtype()},{"B",mx::bfloat16},{"M",m},{"N",n},{"K",k},{"WP",pitch},{"RB",rb},{"CB",cb}},{},false,{})[0];
}
}

#pragma once
// Default-off native-base -> two independent B corrections -> typed SwiGLU.
// No global corrected gate/up, rank reuse, weight merge or base-GEMM change.
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <cmath>
#include <stdexcept>

namespace tc::qwen21::metal {
namespace mx=mlx::core;
inline mx::array prepare_lora_swiglu(const mx::array &base,const mx::array &rg,const mx::array &bg,
                        const mx::array &ru,const mx::array &bu,float gs,float us,
                        int bm=16,int bn=128,int stage=4) {
    if(base.ndim()!=3 || base.shape(0)!=1 || base.shape(1)<=0 || base.shape(1)>8192 ||
       base.shape(2)<=0 || base.shape(2)%2 || base.shape(2)>32768 || base.dtype()!=mx::bfloat16 ||
       rg.ndim()!=3 || ru.ndim()!=3 || rg.shape(0)!=1 || ru.shape(0)!=1 ||
       rg.shape(1)!=base.shape(1) || ru.shape(1)!=base.shape(1) || rg.dtype()!=mx::float32 || ru.dtype()!=mx::float32 ||
       rg.shape(2)<32 || rg.shape(2)>256 || ru.shape(2)<32 || ru.shape(2)>256 ||
       bg.shape()!=mx::Shape{base.shape(2)/2,rg.shape(2)} || bu.shape()!=mx::Shape{base.shape(2)/2,ru.shape(2)} ||
       bg.dtype()!=base.dtype() || bu.dtype()!=base.dtype() || !bg.flags().row_contiguous || !bu.flags().row_contiguous ||
       !std::isfinite(gs) || !std::isfinite(us) || (bm!=16&&bm!=32) || (bn!=64&&bn!=128) || stage<0 || stage>4)
        throw std::invalid_argument("paired LoRA B/SwiGLU physical source/dtype/tile/stage mismatch");
    const int m=base.shape(1),n=base.shape(2)/2;
    static auto kernel=mx::fast::metal_kernel("tc_qwen21_native_base_paired_b_swiglu_v1",
        {"base","rg","bg","ru","bu","gs","us"},{"out"},R"metal(
        using namespace mpp::tensor_ops;
        const uint row=threadgroup_position_in_grid.y*BM,col=threadgroup_position_in_grid.x*BN;
        threadgroup T gates[BM*BN];
        {
            auto a=tensor(const_cast<device T*>(rg),dextents<int,2>{RG,M},array<int,2>{1,RG});
            auto b=tensor(const_cast<device T*>(bg),dextents<int,2>{RG,N},array<int,2>{1,RG});
            auto aa=a.slice(0,row),bb=b.slice(0,col);
            matmul2d<matmul2d_descriptor(BM,BN,RG,false,true,false),execution_simdgroups<4>> op;
            auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
            for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);
                if(row+c[1]<M&&col+c[0]<N){float value=acc[i]*gs;value+=float(base[(row+c[1])*2*N+col+c[0]]);
                    gates[c[1]*BN+c[0]]=T(value);}}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto a=tensor(const_cast<device T*>(ru),dextents<int,2>{RU,M},array<int,2>{1,RU});
            auto b=tensor(const_cast<device T*>(bu),dextents<int,2>{RU,N},array<int,2>{1,RU});
            auto aa=a.slice(0,row),bb=b.slice(0,col);
            matmul2d<matmul2d_descriptor(BM,BN,RU,false,true,false),execution_simdgroups<4>> op;
            auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
            for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);
                if(row+c[1]<M&&col+c[0]<N){float value=acc[i]*us;value+=float(base[(row+c[1])*2*N+N+col+c[0]]);
                    const T g=gates[c[1]*BN+c[0]],u=T(value);
                    T result;
                    if constexpr(STAGE==0)result=g;
                    else if constexpr(STAGE==1)result=u;
                    else {
                        // Match the typed operator expression. The actual
                        // installed-runtime witness must independently pass.
                        const T sigmoid=TcQwenSigmoid{}(g);
                        const T activated=T(g*sigmoid);
                        if constexpr(STAGE==2)result=sigmoid;
                        else if constexpr(STAGE==3)result=activated;
                        else result=T(activated*u);
                    }
                    out[(row+c[1])*N+col+c[0]]=result;
                }}
        }
        )metal",R"metal(
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        // MLX unary_ops.h Sigmoid expression, retaining T-valued exp.
        struct TcQwenSigmoid {
            template<typename T> T operator()(T x) {
                auto y=1/(1+metal::exp(metal::abs(x)));
                return x<0?y:1-y;
            }
        };
        )metal");
    return kernel({base,mx::astype(rg,mx::bfloat16),bg,mx::astype(ru,mx::bfloat16),bu,mx::array(gs,mx::float32),mx::array(us,mx::float32)},
        {{1,m,n}},{mx::bfloat16},{uint32_t((n+bn-1)/bn*128),uint32_t((m+bm-1)/bm),1},{128,1,1},
        {{"T",mx::bfloat16},{"M",m},{"N",n},{"RG",rg.shape(2)},{"RU",ru.shape(2)},{"BM",bm},{"BN",bn},{"STAGE",stage}}, {},false,{})[0];
}
}

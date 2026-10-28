#pragma once
// Research only. Original base gate/up + independent online B corrections +
// typed SwiGLU, without publishing the global base or corrected projections.
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <cmath>
#include <stdexcept>

namespace tc::qwen21::base_swiglu_lab {
namespace mx=mlx::core;
inline mx::array prepare(const mx::array &x,const mx::array &w,
    const mx::array &rg,const mx::array &bg,const mx::array &ru,const mx::array &bu,
    float gs,float us,int bm=32,int bn=128,int stage=6) {
    if(x.ndim()!=3||x.shape(0)!=1||x.shape(1)<=0||x.shape(1)>8192||x.shape(2)<=0||x.shape(2)>4096||x.shape(2)%32||
       x.dtype()!=mx::bfloat16||w.ndim()!=2||w.shape(0)<=0||w.shape(0)%2||w.shape(0)>24576||w.shape(1)!=x.shape(2)||
       w.dtype()!=x.dtype()||!w.flags().row_contiguous||rg.ndim()!=3||ru.ndim()!=3||
       rg.shape(0)!=1||ru.shape(0)!=1||rg.shape(1)!=x.shape(1)||ru.shape(1)!=x.shape(1)||
       rg.shape(2)<32||rg.shape(2)>256||ru.shape(2)<32||ru.shape(2)>256||rg.dtype()!=mx::float32||ru.dtype()!=mx::float32||
       bg.shape()!=mx::Shape{w.shape(0)/2,rg.shape(2)}||bu.shape()!=mx::Shape{w.shape(0)/2,ru.shape(2)}||
       bg.dtype()!=x.dtype()||bu.dtype()!=x.dtype()||!bg.flags().row_contiguous||!bu.flags().row_contiguous||
       !std::isfinite(gs)||!std::isfinite(us)||(bm!=16&&bm!=32&&bm!=64)||(bn!=64&&bn!=128)||stage<0||stage>6)
        throw std::invalid_argument("base/paired-B/SwiGLU physical source/dtype/tile/stage mismatch");
    const int m=x.shape(1),k=x.shape(2),n=w.shape(0)/2;
    static auto kernel=mx::fast::metal_kernel("tc_qwen21_base_paired_b_swiglu_lab_v1",
        {"x","w","rg","bg","ru","bu","gs","us"},{"out"},R"metal(
        using namespace mpp::tensor_ops;
        const uint row=threadgroup_position_in_grid.y*BM,col=threadgroup_position_in_grid.x*BN;
        threadgroup T gates[BM*BN],base_tile[BM*BN];
        {
            auto a=tensor(const_cast<device T*>(x),dextents<int,2>{K,M},array<int,2>{1,K});
            auto b=tensor(const_cast<device T*>(w),dextents<int,2>{K,N},array<int,2>{1,K});
            auto aa=a.slice(0,row),bb=b.slice(0,col);
            matmul2d<matmul2d_descriptor(BM,BN,K,false,true,false),execution_simdgroups<4>> op;
            auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
            for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);
                if(row+c[1]<M&&col+c[0]<N){base_tile[c[1]*BN+c[0]]=T(acc[i]);
                    if constexpr(STAGE==0)out[(row+c[1])*N+col+c[0]]=T(acc[i]);}}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto a=tensor(const_cast<device T*>(rg),dextents<int,2>{RG,M},array<int,2>{1,RG});
            auto b=tensor(const_cast<device T*>(bg),dextents<int,2>{RG,N},array<int,2>{1,RG});
            auto aa=a.slice(0,row),bb=b.slice(0,col);
            matmul2d<matmul2d_descriptor(BM,BN,RG,false,true,false),execution_simdgroups<4>> op;
            auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
            for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);
                if(row+c[1]<M&&col+c[0]<N){float value=acc[i]*gs;value+=float(base_tile[c[1]*BN+c[0]]);
                    gates[c[1]*BN+c[0]]=T(value);
                    if constexpr(STAGE==2)out[(row+c[1])*N+col+c[0]]=T(value);}}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto a=tensor(const_cast<device T*>(x),dextents<int,2>{K,M},array<int,2>{1,K});
            auto b=tensor(const_cast<device T*>(w)+N*K,dextents<int,2>{K,N},array<int,2>{1,K});
            auto aa=a.slice(0,row),bb=b.slice(0,col);
            matmul2d<matmul2d_descriptor(BM,BN,K,false,true,false),execution_simdgroups<4>> op;
            auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
            for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);
                if(row+c[1]<M&&col+c[0]<N){base_tile[c[1]*BN+c[0]]=T(acc[i]);
                    if constexpr(STAGE==1)out[(row+c[1])*N+col+c[0]]=T(acc[i]);}}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto a=tensor(const_cast<device T*>(ru),dextents<int,2>{RU,M},array<int,2>{1,RU});
            auto b=tensor(const_cast<device T*>(bu),dextents<int,2>{RU,N},array<int,2>{1,RU});
            auto aa=a.slice(0,row),bb=b.slice(0,col);
            matmul2d<matmul2d_descriptor(BM,BN,RU,false,true,false),execution_simdgroups<4>> op;
            auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
            for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);
                if(row+c[1]<M&&col+c[0]<N){float value=acc[i]*us;value+=float(base_tile[c[1]*BN+c[0]]);
                    const T g=gates[c[1]*BN+c[0]],u=T(value);
                    if constexpr(STAGE>=3){T result;
                        if constexpr(STAGE==3)result=u;
                        else {const T sigmoid=TcBaseSigmoid{}(g),activated=T(g*sigmoid);
                            if constexpr(STAGE==4)result=sigmoid;
                            else if constexpr(STAGE==5)result=activated;
                            else result=T(activated*u);}
                        out[(row+c[1])*N+col+c[0]]=result;}
                }}
        }
        )metal",R"metal(
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        struct TcBaseSigmoid {template<typename T>T operator()(T x){
            auto y=1/(1+metal::exp(metal::abs(x)));return x<0?y:1-y;}};
        )metal");
    return kernel({x,w,mx::astype(rg,mx::bfloat16),bg,mx::astype(ru,mx::bfloat16),bu,mx::array(gs,mx::float32),mx::array(us,mx::float32)},
        {{1,m,n}},{mx::bfloat16},{uint32_t((n+bn-1)/bn*128),uint32_t((m+bm-1)/bm),1},{128,1,1},
        {{"T",mx::bfloat16},{"M",m},{"K",k},{"N",n},{"RG",rg.shape(2)},{"RU",ru.shape(2)},{"BM",bm},{"BN",bn},{"STAGE",stage}}, {},false,{})[0];
}
}

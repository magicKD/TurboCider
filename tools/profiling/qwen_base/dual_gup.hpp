#pragma once
// Research-only paired dense base GEMM + typed SwiGLU. Original W rows/pitch,
// no packed checkpoint copy. Sharing an X view does not prove physical reuse.
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::dual_gup_lab {
namespace mx=mlx::core;
inline mx::array prepare(const mx::array&x,const mx::array&w,int first,int count,
                        bool shared=false,int bm=32,int bn=64,int stage=4) {
    if(x.ndim()!=3||x.shape(0)!=1||x.shape(1)<1||x.shape(1)>8192||x.shape(2)<32||x.shape(2)>4096||x.shape(2)%32||
       x.dtype()!=mx::bfloat16||!x.flags().row_contiguous||w.ndim()!=2||w.shape(0)<2||w.shape(0)%2||w.shape(0)>32768||
       w.shape(1)!=x.shape(2)||w.dtype()!=x.dtype()||!w.flags().row_contiguous||first<0||count<1||first>w.shape(0)/2-count||
       (bm!=16&&bm!=32)||(bn!=64&&bn!=128)||stage<0||stage>4)
        throw std::invalid_argument("paired dense SwiGLU original BF16 source/range/tile/stage mismatch");
    const int m=x.shape(1),k=x.shape(2),f=w.shape(0)/2;
    auto make=[](bool tg) {
        const std::string start=R"metal(
        using namespace mpp::tensor_ops;
        const uint row=threadgroup_position_in_grid.y*BM,col=threadgroup_position_in_grid.x*BN;
        auto a=tensor(const_cast<device T*>(x),dextents<int,2>{K,M},array<int,2>{1,K});
        auto bg=tensor(const_cast<device T*>(w)+FIRST*K,dextents<int,2>{K,N},array<int,2>{1,K});
        auto bu=tensor(const_cast<device T*>(w)+(F+FIRST)*K,dextents<int,2>{K,N},array<int,2>{1,K});
        auto aa=a.slice(0,row),gg=bg.slice(0,col),uu=bu.slice(0,col);
        matmul2d<matmul2d_descriptor(BM,BN,K,false,true,false),execution_simdgroups<4>> op;
        )metal";
        std::string body;
        if(!tg)body=R"metal(
        auto gates=op.template get_destination_cooperative_tensor<decltype(aa),decltype(gg),float>();
        auto ups=op.template get_destination_cooperative_tensor<decltype(aa),decltype(uu),float>();
        static_assert(__is_same(decltype(gates),decltype(ups)),"paired fragment layouts must be identical");
        op.run(aa,gg,gates);op.run(aa,uu,ups);
        for(uint i=0;i<gates.get_capacity();++i) {
            if(!gates.is_valid_element(i))continue;auto c=gates.get_multidimensional_index(i);
            if(row+c[1]<M&&col+c[0]<N)out[(row+c[1])*N+col+c[0]]=TcDualValue<STAGE>{}(T(gates[i]),T(ups[i]));
        }
        )metal";
        else body=R"metal(
        threadgroup T gates[BM*BN];
        {
            auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(gg),float>();op.run(aa,gg,acc);
            for(uint i=0;i<acc.get_capacity();++i) {
                if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);
                if(row+c[1]<M&&col+c[0]<N)gates[c[1]*BN+c[0]]=T(acc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(uu),float>();op.run(aa,uu,acc);
            for(uint i=0;i<acc.get_capacity();++i) {
                if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);
                if(row+c[1]<M&&col+c[0]<N)out[(row+c[1])*N+col+c[0]]=TcDualValue<STAGE>{}(gates[c[1]*BN+c[0]],T(acc[i]));
            }
        }
        )metal";
        return mx::fast::metal_kernel(tg?"tc_qwen_base_dual_gup_shared_v1":"tc_qwen_base_dual_gup_register_v1",
            {"x","w"},{"out"},start+body,R"metal(
            #include <metal_tensor>
            #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
            struct TcDualSigmoid {template<typename T>T operator()(T x){
                auto y=1/(1+metal::exp(metal::abs(x)));return x<0?y:1-y;}};
            template<int Stage>struct TcDualValue {
                template<typename T>T operator()(T gate,T up){
                    if constexpr(Stage==0)return gate;
                    else if constexpr(Stage==1)return up;
                    else {const T sigmoid=TcDualSigmoid{}(gate),activated=T(gate*sigmoid);
                        if constexpr(Stage==2)return sigmoid;
                        else if constexpr(Stage==3)return activated;
                        else return T(activated*up);}
                }
            };
            )metal");
    };
    static auto reg=make(false),tg=make(true);const auto&kernel=shared?tg:reg;
    return kernel({x,w},{{1,m,count}},{mx::bfloat16},
        {uint32_t((count+bn-1)/bn*128),uint32_t((m+bm-1)/bm),1},{128,1,1},
        {{"T",mx::bfloat16},{"M",m},{"N",count},{"K",k},{"F",f},{"FIRST",first},{"BM",bm},{"BN",bn},{"STAGE",stage}}, {},false,{})[0];
}
}

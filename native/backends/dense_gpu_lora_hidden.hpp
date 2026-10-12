#pragma once
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tc::dense_gpu {
namespace mx=mlx::core;

// Corrected gate/up + SwiGLU hidden, original physical base and B operands.
// Only two small typed TG tiles, never global gate/up or merged weights.
// Preserve base narrowing and corrected gate/up narrowing. Activation
// lowering is independently screened against the compiled model consumer.
inline mx::array lora_hidden(const mx::array &x,const mx::array &w,
    const mx::array &rg,const mx::array &bg,const mx::array &ru,const mx::array &bu,
    int gate_begin,int up_begin,int count,int gate_b_begin,int up_b_begin,
    float gate_scale,float up_scale,int bm=32,int bn=128,bool round_sigmoid=true) {
    if(x.ndim()!=3 || x.shape(0)!=1 || x.shape(1)<=0 || w.ndim()!=2 || w.shape(1)!=x.shape(2) ||
        !w.flags().row_contiguous || (x.dtype()!=mx::float16 && x.dtype()!=mx::bfloat16) || w.dtype()!=x.dtype() ||
        count<=0 || gate_begin<0 || up_begin<0 || gate_begin>w.shape(0)-count || up_begin>w.shape(0)-count ||
        rg.ndim()!=3 || ru.ndim()!=3 || rg.shape(0)!=1 || ru.shape(0)!=1 || rg.shape(1)!=x.shape(1) || ru.shape(1)!=x.shape(1) ||
        rg.dtype()!=mx::float32 || ru.dtype()!=mx::float32 || rg.shape(2)<=0 || ru.shape(2)<=0 ||
        bg.ndim()!=2 || bu.ndim()!=2 || !bg.flags().row_contiguous || !bu.flags().row_contiguous ||
        bg.dtype()!=x.dtype() || bu.dtype()!=x.dtype() || bg.shape(1)!=rg.shape(2) || bu.shape(1)!=ru.shape(2) ||
        gate_b_begin<0 || up_b_begin<0 || gate_b_begin>bg.shape(0)-count || up_b_begin>bu.shape(0)-count ||
        !std::isfinite(gate_scale) || !std::isfinite(up_scale) || (bm!=16 && bm!=32) || (bn!=64 && bn!=128))
        throw std::invalid_argument("fused LoRA hidden source/range/dtype mismatch");
    auto gemm=[](const std::string &input,const std::string &weight,const std::string &k,
                const std::string &offset,const std::string &pitch,const std::string &write) {
        return "{\n auto a=tensor(const_cast<device T*>("+input+"),dextents<int,2>{"+k+",M},array<int,2>{1,"+k+"});\n"
            "auto b=tensor(const_cast<device T*>("+weight+")+"+offset+",dextents<int,2>{"+k+",N},array<int,2>{1,"+pitch+"});\n"
            "auto aa=a.slice(0,row),bb=b.slice(0,col);\nmatmul2d<matmul2d_descriptor(BM,BN,"+k+",false,true,false),execution_simdgroups<4>> op;\n"
            "auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);\n"
            "for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);\n"
            "if(row+c[1]<M && col+c[0]<N){"+write+"}}}\n";
    };
    static auto kernel=mx::fast::metal_kernel("tc_lora_corrected_hidden_four_matmul",
        {"x","w","rg","bg","ru","bu","scaleg","scaleu"},{"out"},
        std::string("using namespace mpp::tensor_ops;\nuint row=threadgroup_position_in_grid.y*BM,col=threadgroup_position_in_grid.x*BN;\nthreadgroup T base_tile[BM*BN],gate_tile[BM*BN];\n")+
        gemm("x","w","K","GATE*K","K","base_tile[c[1]*BN+c[0]]=T(acc[i]);")+
        "threadgroup_barrier(mem_flags::mem_threadgroup);\n"+
        gemm("rg","bg","RG","GB*RG","RG","gate_tile[c[1]*BN+c[0]]=T(float(base_tile[c[1]*BN+c[0]])+acc[i]*scaleg);")+
        "threadgroup_barrier(mem_flags::mem_threadgroup);\n"+
        gemm("x","w","K","UP*K","K","base_tile[c[1]*BN+c[0]]=T(acc[i]);")+
        "threadgroup_barrier(mem_flags::mem_threadgroup);\n"+
        gemm("ru","bu","RU","UB*RU","RU",R"metal(
            T g=gate_tile[c[1]*BN+c[0]],u=T(float(base_tile[c[1]*BN+c[0]])+acc[i]*scaleu);
            float s=1.0f/(1.0f+metal::precise::exp(-float(g)));
            if constexpr(ROUND_SIGMOID)s=float(T(s));
            T activated=T(float(g)*s);
            out[(row+c[1])*N+col+c[0]]=T(float(activated)*float(u));
        )metal"),"#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    const int m=x.shape(1),n=count;
    return kernel({x,w,mx::astype(rg,x.dtype()),bg,mx::astype(ru,x.dtype()),bu,
        mx::array(gate_scale,mx::float32),mx::array(up_scale,mx::float32)},{{1,m,n}},{x.dtype()},
        {((n+bn-1)/bn)*128,(m+bm-1)/bm,1},{128,1,1},
        {{"T",x.dtype()},{"M",m},{"N",n},{"K",x.shape(2)},{"RG",rg.shape(2)},{"RU",ru.shape(2)},
         {"GATE",gate_begin},{"UP",up_begin},{"GB",gate_b_begin},{"UB",up_b_begin},{"BM",bm},{"BN",bn},{"ROUND_SIGMOID",round_sigmoid}}, {},false,{})[0];
}
} // namespace tc::dense_gpu

#pragma once
// Default-off corrected-QKV decode pilot. Original online A ranks; base + B + Q/K norm/RoPE.
// F32 A-ranks are supplied by the original online producer, never precomputed
// across calls. Preserve BF16 base and corrected-projection rounding points.
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tc::qwen21::metal {
namespace mx=mlx::core;
inline std::string corrected_qkv_source(bool direct) {
    std::string source=R"metal(
        using namespace mpp::tensor_ops;
        const uint row=threadgroup_position_in_grid.y*BM;
        const uint col=threadgroup_position_in_grid.x*128;
        const uint kind=col/H, head=(col%H)/128;
        device T *output=out+kind*H*M;
        threadgroup T tile[BM*128];
        {
            auto a=tensor(const_cast<device T*>(x),dextents<int,2>{H,M},array<int,2>{1,H});
            /*TC_QKV_BASE_BINDING*/
            matmul2d<matmul2d_descriptor(BM,128,H,false,true,false),execution_simdgroups<4>> op;
            auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
            op.run(aa,bb,acc);
            for(uint i=0;i<acc.get_capacity();++i) {
                if(!acc.is_valid_element(i))continue;
                auto at=acc.get_multidimensional_index(i);
                tile[at[1]*128+at[0]]=T(acc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            device const T *lr=kind==0?rq:(kind==1?rk:rv);
            device const T *bu=kind==0?bq:(kind==1?bk:bv);
            auto a=tensor(const_cast<device T*>(lr),dextents<int,2>{R,M},array<int,2>{1,R});
            auto b=tensor(const_cast<device T*>(bu),dextents<int,2>{R,H},array<int,2>{1,R});
            auto aa=a.slice(0,row),bb=b.slice(0,col%H);
            matmul2d<matmul2d_descriptor(BM,128,R,false,true,false),execution_simdgroups<4>> op;
            auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
            op.run(aa,bb,acc);
            for(uint i=0;i<acc.get_capacity();++i) {
                if(!acc.is_valid_element(i))continue;
                auto at=acc.get_multidimensional_index(i);
                float value=acc[i]*scales[kind];
                value+=float(tile[at[1]*128+at[0]]);
                tile[at[1]*128+at[0]]=T(value);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
        for(uint local=sg;local<BM;local+=4) {
            const uint token=row+local;if(token>=M)continue;
            const uint d0=lane*4,dst=(head*M+token)*128+d0;
            T values[4];float sum=0.f;
            for(uint i=0;i<4;++i){values[i]=tile[local*128+d0+i];float z=float(values[i]);sum+=z*z;}
            if(kind==2){for(uint i=0;i<4;++i)output[dst+i]=values[i];continue;}
            const float inv=metal::precise::rsqrt(simd_sum(sum)/128.f+1e-6f);
            for(uint i=0;i<4;++i) {
                const uint d=d0+i;const float weight=kind==0?float(qw[d]):float(kw[d]);
                // Native BF16 RMSNorm narrows the normalized value BEFORE
                // its typed gamma multiplication, then narrows that product.
                values[i]=T(float(T(float(values[i])*inv))*weight);
            }
            {
                // Match the native F32 Multiply nodes before Add/Subtract.
                // Keep this scope local: base/B and RMSNorm are unchanged.
#pragma clang fp contract(off)
                for(uint i=0;i<4;i+=2) {
                    const uint d=d0+i;const float c=cosine[token*64+d/2],s=sine[token*64+d/2];
                    const float a=float(values[i]),b=float(values[i+1]);
                    output[dst+i]=T(a*c-b*s);output[dst+i+1]=T(b*c+a*s);
                }
            }
        }
        )metal";
    const std::string marker="/*TC_QKV_BASE_BINDING*/";
    const auto at=source.find(marker);
    if(at==std::string::npos)throw std::logic_error("QKV source binding marker missing");
    source.replace(at,marker.size(),direct?R"metal(device const T *source=kind==0?wq:(kind==1?wk:wv);
            auto b=tensor(const_cast<device T*>(source),dextents<int,2>{H,H},array<int,2>{1,H});
            auto aa=a.slice(0,row),bb=b.slice(0,col%H);)metal":R"metal(auto b=tensor(const_cast<device T*>(w),dextents<int,2>{H,3*H},array<int,2>{1,H});
            auto aa=a.slice(0,row),bb=b.slice(0,col);)metal");
    return source;
}
inline std::vector<mx::array> prepare_corrected_qkv_impl(
        const mx::array &x,const std::vector<mx::array> &bases,
        const std::array<mx::array,3> &ranks,const std::array<mx::array,3> &up,
        const std::array<float,3> &scale,const mx::array &qw,const mx::array &kw,
        const mx::array &cosine,const mx::array &sine,int bm,bool direct) {
    if(x.ndim()!=3 || x.shape(0)!=1 || x.shape(1)<=0 || x.shape(1)>8192 ||
       x.shape(2)<=0 || x.shape(2)>4096 || x.shape(2)%128 ||
       x.dtype()!=mx::bfloat16 || (bm!=16 && bm!=32 && bm!=48 && bm!=64))
        throw std::invalid_argument("QKV LoRA prepare input/tile mismatch");
    const int m=x.shape(1),h=x.shape(2);
    if(bases.size()!=size_t(direct?3:1))throw std::invalid_argument("QKV base binding count mismatch");
    for(const auto &base:bases)
        if(base.shape()!=mx::Shape{direct?h:3*h,h} || base.dtype()!=x.dtype() || !base.flags().row_contiguous)
            throw std::invalid_argument("QKV original base geometry/dtype/stride mismatch");
    if(qw.shape()!=mx::Shape{128} || kw.shape()!=qw.shape() ||
       qw.dtype()!=x.dtype() || kw.dtype()!=x.dtype() || cosine.shape()!=mx::Shape{m,64} ||
       sine.shape()!=cosine.shape() || cosine.dtype()!=mx::float32 || sine.dtype()!=mx::float32)
        throw std::invalid_argument("QKV LoRA prepare base/norm/position mismatch");
    if(ranks[0].ndim()!=3 || ranks[0].shape(0)!=1 || ranks[0].shape(1)!=m)
        throw std::invalid_argument("QKV LoRA prepare rank shape mismatch");
    const int r=ranks[0].shape(2);
    if(r!=64 && r!=128 && r!=256)throw std::invalid_argument("QKV LoRA prepare bounded rank required");
    for(size_t i=0;i<3;++i)
        if(ranks[i].shape()!=mx::Shape{1,m,r} || ranks[i].dtype()!=mx::float32 ||
           up[i].shape()!=mx::Shape{h,r} || up[i].dtype()!=x.dtype() || !up[i].flags().row_contiguous ||
           !std::isfinite(scale[i]))throw std::invalid_argument("QKV LoRA prepare factor/scale mismatch");
    static const std::string header="#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
    static auto kernel=mx::fast::metal_kernel("tc_qwen21_corrected_qkv_prepare_v2_rope_rounding",
        {"x","w","rq","rk","rv","bq","bk","bv","scales","qw","kw","cosine","sine"},
        {"out"},corrected_qkv_source(false),header);
    static auto direct_kernel=mx::fast::metal_kernel("tc_qwen21_corrected_qkv_direct_v1",
        {"x","wq","wk","wv","rq","rk","rv","bq","bk","bv","scales","qw","kw","cosine","sine"},
        {"out"},corrected_qkv_source(true),header);
    // Match the established joint B consumer's explicit F32 -> BF16 narrowing.
    const auto rq=mx::astype(ranks[0],mx::bfloat16),rk=mx::astype(ranks[1],mx::bfloat16),rv=mx::astype(ranks[2],mx::bfloat16);
    const mx::array scales(scale.data(),{3},mx::float32);
    // One primitive output and three independent single-output slices avoid
    // compiled multi-output sibling cycles pinning captured model constants.
    std::vector<mx::array> inputs{x};inputs.insert(inputs.end(),bases.begin(),bases.end());
    for(const auto &value:std::vector<mx::array>{rq,rk,rv,up[0],up[1],up[2],scales,qw,kw,cosine,sine})inputs.push_back(value);
    auto packed=(direct?direct_kernel:kernel)(inputs,
        {{3,1,h/128,m,128}},{x.dtype()},
        {uint32_t((3*h/128)*128),uint32_t((m+bm-1)/bm),1},{128,1,1},
        {{"T",x.dtype()},{"M",m},{"H",h},{"R",r},{"BM",bm}}, {},false,{})[0];
    std::vector<mx::array> outputs;
    for(int i=0;i<3;++i)
        outputs.push_back(mx::squeeze(mx::slice(packed,{i,0,0,0,0},{i+1,1,h/128,m,128}),0));
    return outputs;
}

inline std::vector<mx::array> prepare_corrected_qkv(
        const mx::array &x,const mx::array &packed_base,
        const std::array<mx::array,3> &ranks,const std::array<mx::array,3> &up,
        const std::array<float,3> &scale,const mx::array &qw,const mx::array &kw,
        const mx::array &cosine,const mx::array &sine,int bm=32) {
    return prepare_corrected_qkv_impl(x,{packed_base},ranks,up,scale,qw,kw,cosine,sine,bm,false);
}
inline std::vector<mx::array> prepare_corrected_qkv_direct(
        const mx::array &x,const std::array<mx::array,3> &bases,
        const std::array<mx::array,3> &ranks,const std::array<mx::array,3> &up,
        const std::array<float,3> &scale,const mx::array &qw,const mx::array &kw,
        const mx::array &cosine,const mx::array &sine,int bm=32) {
    return prepare_corrected_qkv_impl(x,std::vector<mx::array>(bases.begin(),bases.end()),ranks,up,scale,qw,kw,cosine,sine,bm,true);
}

} // namespace tc::qwen21::metal

#pragma once
// Research-only counterfactual stages. Never installed as a runtime route.
// F32 A-ranks are supplied by the original online producer, never precomputed
// across calls. Preserve BF16 base and corrected-projection rounding points.
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <array>
#include <cmath>
#include <stdexcept>

namespace tc::qwen21::qkv_numeric_lab {
namespace mx=mlx::core;
inline std::vector<mx::array> prepare_stage(
        const mx::array &x,const mx::array &packed_base,
        const std::array<mx::array,3> &ranks,const std::array<mx::array,3> &up,
        const std::array<float,3> &scale,const mx::array &qw,const mx::array &kw,
        const mx::array &cosine,const mx::array &sine,int stage,int bm=32,
        const std::array<mx::array,3> *native_base=nullptr,int rope_mode=0) {
    if(x.ndim()!=3 || x.shape(0)!=1 || x.shape(1)<=0 || x.shape(1)>8192 ||
       x.shape(2)<=0 || x.shape(2)>4096 || x.shape(2)%128 ||
       x.dtype()!=mx::bfloat16 || (bm!=16 && bm!=32 && bm!=48 && bm!=64))
        throw std::invalid_argument("QKV LoRA prepare input/tile mismatch");
    if(stage<0||stage>3)throw std::invalid_argument("bounded numeric stage required");
    if(rope_mode<0||rope_mode>1)throw std::invalid_argument("bounded RoPE contraction control required");
    const int m=x.shape(1),h=x.shape(2);
    if(native_base)for(const auto &base:*native_base)
        if(base.shape()!=x.shape() || base.dtype()!=x.dtype() || !base.flags().row_contiguous)
            throw std::invalid_argument("actual original contiguous base required");
    if(packed_base.shape()!=mx::Shape{3*h,h} || packed_base.dtype()!=x.dtype() ||
       !packed_base.flags().row_contiguous || qw.shape()!=mx::Shape{128} || kw.shape()!=qw.shape() ||
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
    static auto kernel=mx::fast::metal_kernel("tc_qwen21_qkv_numeric_stage_v1",
        {"x","w","baseq","basek","basev","rq","rk","rv","bq","bk","bv","scales","qw","kw","cosine","sine"},
        {"out"},R"metal(
        using namespace mpp::tensor_ops;
        const uint row=threadgroup_position_in_grid.y*BM;
        const uint col=threadgroup_position_in_grid.x*128;
        const uint kind=col/H, head=(col%H)/128;
        device T *output=out+kind*H*M;
        threadgroup T tile[BM*128];
        if constexpr(HAS_BASE) {
            device const T *source=kind==0?baseq:kind==1?basek:basev;
            for(uint index=thread_index_in_threadgroup;index<BM*128;index+=128) {
                const uint local=index/128,d=index%128;
                tile[index]=row+local<M?source[(row+local)*H+col%H+d]:T(0);
            }
        } else {
            auto a=tensor(const_cast<device T*>(x),dextents<int,2>{H,M},array<int,2>{1,H});
            auto b=tensor(const_cast<device T*>(w),dextents<int,2>{H,3*H},array<int,2>{1,H});
            auto aa=a.slice(0,row),bb=b.slice(0,col);
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
        if constexpr(STAGE==0) {
            for(uint index=thread_index_in_threadgroup;index<BM*128;index+=128) {
                const uint local=index/128,d=index%128;
                if(row+local<M)output[(head*M+row+local)*128+d]=tile[index];
            }
            return;
        }
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
        if constexpr(STAGE==1) {
            for(uint index=thread_index_in_threadgroup;index<BM*128;index+=128) {
                const uint local=index/128,d=index%128;
                if(row+local<M)output[(head*M+row+local)*128+d]=tile[index];
            }
            return;
        }
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
            if constexpr(STAGE==2) {for(uint i=0;i<4;++i)output[dst+i]=values[i];continue;}
            if constexpr(ROPE_MODE==1) {
#pragma clang fp contract(off)
                for(uint i=0;i<4;i+=2) {
                    const uint d=d0+i;const float c=cosine[token*64+d/2],s=sine[token*64+d/2];
                    const float a=float(values[i]),b=float(values[i+1]);
                    output[dst+i]=T(a*c-b*s);output[dst+i+1]=T(b*c+a*s);
                }
            } else for(uint i=0;i<4;i+=2) {
                const uint d=d0+i;const float c=cosine[token*64+d/2],s=sine[token*64+d/2];
                const float a=float(values[i]),b=float(values[i+1]);
                output[dst+i]=T(a*c-b*s);output[dst+i+1]=T(b*c+a*s);
            }
        }
        )metal","#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    // Match the established joint B consumer's explicit F32 -> BF16 narrowing.
    const auto rq=mx::astype(ranks[0],mx::bfloat16),rk=mx::astype(ranks[1],mx::bfloat16),rv=mx::astype(ranks[2],mx::bfloat16);
    const mx::array scales(scale.data(),{3},mx::float32);
    // One primitive output and three independent single-output slices avoid
    // compiled multi-output sibling cycles pinning captured model constants.
    auto packed=kernel({x,packed_base,native_base?(*native_base)[0]:x,native_base?(*native_base)[1]:x,native_base?(*native_base)[2]:x,rq,rk,rv,up[0],up[1],up[2],scales,qw,kw,cosine,sine},
        {{3,1,h/128,m,128}},{x.dtype()},
        {uint32_t((3*h/128)*128),uint32_t((m+bm-1)/bm),1},{128,1,1},
        {{"T",x.dtype()},{"M",m},{"H",h},{"R",r},{"BM",bm},{"STAGE",stage},{"HAS_BASE",bool(native_base)},{"ROPE_MODE",rope_mode}}, {},false,{})[0];
    std::vector<mx::array> outputs;
    for(int i=0;i<3;++i)
        outputs.push_back(mx::squeeze(mx::slice(packed,{i,0,0,0,0},{i+1,1,h/128,m,128}),0));
    return outputs;
}

} // namespace tc::qwen21::qkv_numeric_lab

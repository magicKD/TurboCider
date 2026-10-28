#pragma once
// Research only: complete base + online B correction + Q/K norm/RoPE.
// F32 A-ranks are supplied by the original online producer, never precomputed
// across calls. Preserve BF16 base and corrected-projection rounding points.
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <array>
#include <cmath>
#include <stdexcept>

namespace tc::qwen21::qkv_lora_lab {
namespace mx=mlx::core;
inline std::vector<mx::array> prepare(
        const mx::array &x,const mx::array &packed_base,
        const std::array<mx::array,3> &ranks,const std::array<mx::array,3> &up,
        const std::array<float,3> &scale,const mx::array &qw,const mx::array &kw,
        const mx::array &cosine,const mx::array &sine,int bm=32) {
    if(x.ndim()!=3 || x.shape(0)!=1 || x.shape(1)<=0 || x.shape(1)>8192 ||
       x.shape(2)<=0 || x.shape(2)>4096 || x.shape(2)%128 ||
       x.dtype()!=mx::bfloat16 || (bm!=16 && bm!=32 && bm!=48 && bm!=64))
        throw std::invalid_argument("QKV LoRA prepare input/tile mismatch");
    const int m=x.shape(1),h=x.shape(2);
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
    static auto kernel=mx::fast::metal_kernel("tc_qwen21_corrected_qkv_prepare_lab",
        {"x","w","rq","rk","rv","bq","bk","bv","scales","qw","kw","cosine","sine"},
        {"out"},R"metal(
        using namespace mpp::tensor_ops;
        const uint row=threadgroup_position_in_grid.y*BM;
        const uint col=threadgroup_position_in_grid.x*128;
        const uint kind=col/H, head=(col%H)/128;
        device T *output=out+kind*H*M;
        threadgroup T tile[BM*128];
        {
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
            for(uint i=0;i<4;i+=2) {
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
    auto packed=kernel({x,packed_base,rq,rk,rv,up[0],up[1],up[2],scales,qw,kw,cosine,sine},
        {{3,1,h/128,m,128}},{x.dtype()},
        {uint32_t((3*h/128)*128),uint32_t((m+bm-1)/bm),1},{128,1,1},
        {{"T",x.dtype()},{"M",m},{"H",h},{"R",r},{"BM",bm}}, {},false,{})[0];
    std::vector<mx::array> outputs;
    for(int i=0;i<3;++i)
        outputs.push_back(mx::squeeze(mx::slice(packed,{i,0,0,0,0},{i+1,1,h/128,m,128}),0));
    return outputs;
}

// Second-stage consumer of [base Q|K|V|rank Q|K|V]. The first GEMM accumulates
// in F32 and narrows each rank at its sole B-input boundary, as joint B does.
// No F32 rank is consumed elsewhere on this bounded ordinary QKV route.
inline std::vector<mx::array> finish_projected(
        const mx::array &projected,const std::array<mx::array,3> &up,
        const std::array<float,3> &scale,const mx::array &qw,const mx::array &kw,
        const mx::array &cosine,const mx::array &sine,int bm=32) {
    if(projected.ndim()!=3 || projected.shape(0)!=1 || projected.shape(1)<=0 || projected.shape(1)>8192 ||
       projected.dtype()!=mx::bfloat16 || up[0].ndim()!=2 || (bm!=16 && bm!=32 && bm!=48 && bm!=64))
        throw std::invalid_argument("projected QKV finish input/tile mismatch");
    const int m=projected.shape(1),h=up[0].shape(0),r=up[0].shape(1);
    if(h<=0 || h>4096 || h%128 || (r!=64 && r!=128 && r!=256) || projected.shape(2)!=3*h+3*r ||
       qw.shape()!=mx::Shape{128} || kw.shape()!=qw.shape() || qw.dtype()!=mx::bfloat16 || kw.dtype()!=qw.dtype() ||
       cosine.shape()!=mx::Shape{m,64} || sine.shape()!=cosine.shape() || cosine.dtype()!=mx::float32 || sine.dtype()!=cosine.dtype())
        throw std::invalid_argument("projected QKV finish geometry/dtype mismatch");
    for(int i=0;i<3;++i)if(up[i].shape()!=mx::Shape{h,r} || up[i].dtype()!=mx::bfloat16 ||
        !up[i].flags().row_contiguous || !std::isfinite(scale[i]))throw std::invalid_argument("projected QKV finish factor mismatch");
    static auto kernel=mx::fast::metal_kernel("tc_qwen21_joint_projection_finish_lab",
        {"projected","bq","bk","bv","scales","qw","kw","cosine","sine"},{"out"},R"metal(
        using namespace mpp::tensor_ops;
        const uint row=threadgroup_position_in_grid.y*BM,col=threadgroup_position_in_grid.x*128;
        const uint kind=col/H,head=(col%H)/128;
        device const T *bu=kind==0?bq:(kind==1?bk:bv);
        auto a=tensor(const_cast<device T*>(projected)+3*H+kind*R,dextents<int,2>{R,M},array<int,2>{1,3*H+3*R});
        auto b=tensor(const_cast<device T*>(bu),dextents<int,2>{R,H},array<int,2>{1,R});
        auto aa=a.slice(0,row),bb=b.slice(0,col%H);
        matmul2d<matmul2d_descriptor(BM,128,R,false,true,false),execution_simdgroups<4>> op;
        auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
        op.run(aa,bb,acc);threadgroup T tile[BM*128];
        for(uint i=0;i<acc.get_capacity();++i) {
            if(!acc.is_valid_element(i))continue;auto at=acc.get_multidimensional_index(i);
            if(row+at[1]<M) {
                float value=acc[i]*scales[kind];value+=float(projected[(row+at[1])*(3*H+3*R)+col+at[0]]);
                tile[at[1]*128+at[0]]=T(value);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
        device T *output=out+kind*H*M;
        for(uint local=sg;local<BM;local+=4) {
            const uint token=row+local;if(token>=M)continue;
            const uint d0=lane*4,dst=(head*M+token)*128+d0;
            T values[4];float sum=0.f;
            for(uint i=0;i<4;++i){values[i]=tile[local*128+d0+i];float z=float(values[i]);sum+=z*z;}
            if(kind==2){for(uint i=0;i<4;++i)output[dst+i]=values[i];continue;}
            const float inv=metal::precise::rsqrt(simd_sum(sum)/128.f+1e-6f);
            for(uint i=0;i<4;++i){const float weight=kind==0?float(qw[d0+i]):float(kw[d0+i]);
                values[i]=T(float(T(float(values[i])*inv))*weight);}
            for(uint i=0;i<4;i+=2){const uint d=d0+i;const float c=cosine[token*64+d/2],s=sine[token*64+d/2];
                const float a=float(values[i]),b=float(values[i+1]);
                output[dst+i]=T(a*c-b*s);output[dst+i+1]=T(b*c+a*s);}
        }
        )metal","#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    const mx::array scales(scale.data(),{3},mx::float32);
    auto packed=kernel({projected,up[0],up[1],up[2],scales,qw,kw,cosine,sine},{{3,1,h/128,m,128}},{mx::bfloat16},
        {uint32_t((3*h/128)*128),uint32_t((m+bm-1)/bm),1},{128,1,1},
        {{"T",mx::bfloat16},{"M",m},{"H",h},{"R",r},{"BM",bm}}, {},false,{})[0];
    std::vector<mx::array> outputs;
    for(int i=0;i<3;++i)outputs.push_back(mx::squeeze(mx::slice(packed,{i,0,0,0,0},{i+1,1,h/128,m,128}),0));
    return outputs;
}
} // namespace tc::qwen21::qkv_lora_lab

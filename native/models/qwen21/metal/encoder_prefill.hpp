#pragma once
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::qwen21::metal {
namespace mx=mlx::core;
inline mx::array encoder_rms(const mx::array &x,const mx::array &weight,float epsilon) {
    if(x.ndim()!=3 || x.shape(0)!=1 || x.shape(1)<=0 || x.shape(2)<=0 || x.shape(2)>8192 ||
        weight.shape()!=mx::Shape{x.shape(2)} || x.dtype()!=weight.dtype() ||
        (x.dtype()!=mx::float16 && x.dtype()!=mx::bfloat16) || epsilon!=1e-6f)
        throw std::invalid_argument("encoder fused RMS requires original typed rank3 rows and epsilon1e-6");
    static auto kernel=mx::fast::metal_kernel("tc_qwen_vl_prefill_rms",{"x","w"},{"out"},R"metal(
        uint row=threadgroup_position_in_grid.x,tid=thread_position_in_threadgroup.x;
        uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
        threadgroup float partials[4];float sum=0.0f;
        for(uint d=tid;d<H;d+=128){float value=float(x[row*H+d]);sum+=value*value;}
        sum=simd_sum(sum);if(lane==0)partials[sg]=sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(sg==0){float total=simd_sum(lane<4?partials[lane]:0.0f);if(lane==0)partials[0]=metal::precise::rsqrt(total/float(H)+1e-6f);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint d=tid;d<H;d+=128)out[row*H+d]=T((float(x[row*H+d])*partials[0])*float(w[d]));
    )metal");
    return kernel({x,weight},{x.shape()},{x.dtype()},{x.shape(1)*128,1,1},{128,1,1},
        {{"T",x.dtype()},{"H",x.shape(2)}},{},false,{})[0];
}

// Qwen3-VL uses split-half (NeoX) RoPE, NOT DiT's interleaved pairs. Keep
// activation rounding after F32 RMS/scale and at both RoPE products.
inline std::vector<mx::array> encoder_qk(const mx::array &q,const mx::array &k,
        const mx::array &qw,const mx::array &kw,const mx::array &cos,const mx::array &sin,float epsilon) {
    if(q.ndim()!=3 || k.ndim()!=3 || q.shape(0)!=1 || k.shape(0)!=1 || q.shape(1)<=0 || q.shape(1)!=k.shape(1) ||
        q.shape(2)!=4096 || k.shape(2)!=1024 || qw.shape()!=mx::Shape{128} || kw.shape()!=qw.shape() ||
        cos.shape()!=mx::Shape{1,1,q.shape(1),128} || sin.shape()!=cos.shape() ||
        q.dtype()!=k.dtype() || q.dtype()!=qw.dtype() || q.dtype()!=kw.dtype() || q.dtype()!=cos.dtype() || q.dtype()!=sin.dtype() ||
        (q.dtype()!=mx::float16 && q.dtype()!=mx::bfloat16) || epsilon!=1e-6f)
        throw std::invalid_argument("encoder fused Q/K RMS/NeoX RoPE requires original32Q/8KV heads and typed128 channels");
    static auto kernel=mx::fast::metal_kernel("tc_qwen_vl_prefill_qk",{"q","k","qw","kw","cos","sin"},{"qo","ko"},R"metal(
        uint task=thread_position_in_grid.x/32,lane=thread_index_in_simdgroup;
        if(task>=M*40)return;uint row=task/40,h=task%40;bool query=h<32;
        uint head=query?h:h-32,heads=query?32:8,base=row*heads*128+head*128;
        float a[4],z[4],sum=0.0f;
        for(uint i=0;i<4;++i){uint d=lane+i*32;a[i]=query?float(q[base+d]):float(k[base+d]);sum+=a[i]*a[i];}
        float inverse=metal::precise::rsqrt(simd_sum(sum)/128.0f+1e-6f);
        for(uint i=0;i<4;++i){uint d=lane+i*32;z[i]=float(T((a[i]*inverse)*(query?float(qw[d]):float(kw[d]))));}
        for(uint i=0;i<4;++i){uint d=lane+i*32;float rotated=i<2?-z[i+2]:z[i-2];
            T first=T(z[i]*float(cos[row*128+d])),second=T(rotated*float(sin[row*128+d]));T value=T(first+second);
            uint dst=(head*M+row)*128+d;if(query)qo[dst]=value;else ko[dst]=value;}
    )metal");
    const int m=q.shape(1);
    return kernel({q,k,qw,kw,cos,sin},{{1,32,m,128},{1,8,m,128}},{q.dtype(),k.dtype()},
        {m*40*32,1,1},{128,1,1},{{"T",q.dtype()},{"M",m}},{},false,{});
}
}

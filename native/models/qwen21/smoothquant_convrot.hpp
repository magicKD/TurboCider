#pragma once
#include "../../backends/convrot_rotation.hpp"
namespace tc::qwen21 {
// Symmetric, orthogonal regular H4 tensor powers. Scaling and rotation are
// FP32, with one final FP16 cast before Cider's per-token A8 quantization.
// This is the same order used by the frozen 2026-10-17 experiment.
inline Tensor sq_convrot_rotate(const Tensor &input){
 auto x=mx::astype(input,mx::float32);int k=x.shape(-1);
 if(k%256==0)return convrot_kernel::rotate(x,convrot_kernel::Rotation::SimdRegister);
 int g=k%64==0?64:k%16==0?16:k%4==0?4:0;
 if(!g)throw std::invalid_argument("ConvRot K must be divisible by 4");
 static auto kernel=mx::fast::metal_kernel("qwen21_regular_hadamard_small",{"x","columns","group_size"},{"out"},R"metal(
  threadgroup float current[256],next[256];
  uint lane=thread_index_in_threadgroup,K=uint(columns),G=uint(group_size);
  uint base=threadgroup_position_in_grid.y*K+threadgroup_position_in_grid.x*G;
  if(lane<G)current[lane]=x[base+lane];threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint stride=1;stride<G;stride*=4){
   if(lane<G){uint digit=(lane/stride)&3,first=lane-digit*stride;
    float a=current[first],b=current[first+stride],c=current[first+2*stride],d=current[first+3*stride];
    switch(digit){case 0:next[lane]=a+b+c-d;break;case 1:next[lane]=a+b-c+d;break;case 2:next[lane]=a-b+c+d;break;default:next[lane]=-a+b+c+d;}
   }threadgroup_barrier(mem_flags::mem_threadgroup);if(lane<G)current[lane]=next[lane];threadgroup_barrier(mem_flags::mem_threadgroup);
  }if(lane<G)out[base+lane]=current[lane]*rsqrt(float(G));
 )metal");
 return kernel({x,Tensor(k),Tensor(g)},{x.shape()},{mx::float32},{k/g*256,int(x.size()/k),1},{256,1,1},{},{},false,{})[0];
}
}

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_attention.h"
#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>

namespace rv {
namespace {
NSString* const source = @R"metal(
#include <metal_stdlib>
using namespace metal;

struct Geometry { uint h, w, planes, query_heads, kv_heads, condition; };
#include <metal_simdgroup_matrix>

// A group computes eight rows by eight adjacent queries. Flattening the rectangle of eligible keys
// lets 32-key tiles span scanlines. Each SIMD group keeps its query fragments,
// eight output matrices, maxima and normalizers in registers across tiles.
kernel void local_attention_block32(
    device const ushort* q [[buffer(0)]], device const ushort* k [[buffer(1)]],
    device const ushort* v [[buffer(2)]], device ushort* y [[buffer(3)]],
    constant Geometry& g [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint group_rows=8, queries=group_rows*8, threads=group_rows*32;
  constexpr ushort score_stride=32;
  threadgroup half shared_qk[queries][64], vt[32][64], prob[queries][32];
  threadgroup float scratch[queries][score_stride];
#define qt shared_qk
#define kt shared_qk
  const uint tiles_x=(g.w+7u)/8u;
  const uint top=(group.x/tiles_x)*group_rows, left=(group.x%tiles_x)*8u;
  const uint n=g.h*g.w, query_base=(group.z*g.query_heads+group.y)*n*64u;
  const uint kv_head=group.y/(g.query_heads/g.kv_heads);
  const uint kv_base=(group.z*g.kv_heads+kv_head)*g.planes*n*64u;
  for(uint i=tid;i<queries*64u;i+=threads){
    uint r=i/64, d=i%64;
    uint yq=min(top+r/8,g.h-1u), xq=min(left+r%8,g.w-1u);
    qt[r][d]=half(as_type<float>(uint(q[query_base+(yq*g.w+xq)*64u+d])<<16));
  }
  // Read the actual register-to-matrix element mapping once, via a known
  // matrix. Row scaling and epilogue stores then need no assumed lane layout.
  if(tid<64) scratch[tid/8][tid%8]=float(tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 index_matrix;
  simdgroup_load(index_matrix,&scratch[0][0],score_stride);
  const ushort2 indices=ushort2(reinterpret_cast<thread float2&>(index_matrix.thread_elements()));
  simdgroup_half8x8 query_frag[8];
  simdgroup_float8x8 output[8];
  const simdgroup_float8x8 zero=make_filled_simdgroup_matrix<float,8,8>(0.0f);
  for(ushort d=0;d<8;++d){
    simdgroup_load(query_frag[d],&qt[sg*8][d*8],64);
    output[d]=zero;
  }
  // The first K tile reuses the query staging memory.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_alpha[8];
  float maxima[8], sums[8];
  for(ushort r=0;r<8;++r){ maxima[r]=-1.0e30f; sums[r]=0; }
  for(uint plane=0;plane<g.planes;++plane){
    const int radius=(g.condition || plane<2)?24:12;
    const int ymin=max(0,int(top)-radius), ymax=min(int(g.h),int(min(top+group_rows-1u,g.h-1))+radius);
    const int xmin=max(0,int(left)-radius), xmax=min(int(g.w),int(min(left+7u,g.w-1))+radius);
    const uint width=uint(xmax-xmin), count=width*uint(ymax-ymin);
    for(uint base=0;base<count;base+=32u){
      for(uint i=tid;i<32u*64u;i+=threads){
        uint idx=min(base+i/64,count-1u), d=i%64;
        uint pos=(uint(ymin)+idx/width)*g.w+uint(xmin)+idx%width;
        uint address=kv_base+(plane*n+pos)*64u+d;
        kt[i/64][d]=half(as_type<float>(uint(k[address])<<16));
        vt[i/64][d]=half(as_type<float>(uint(v[address])<<16));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(ushort section=0;section<4;++section){
        simdgroup_float8x8 score=zero;
        for(ushort d=0;d<8;++d){
          simdgroup_half8x8 key_frag;
          simdgroup_load(key_frag,&kt[section*8][d*8],64,ulong2(0),true);
          simdgroup_multiply_accumulate(score,query_frag[d],key_frag,score);
        }
        simdgroup_store(score,&scratch[sg*8][section*8],score_stride);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const uint idx=base+lane;
      const int ky=ymin+int(idx/width), kx=xmin+int(idx%width);
      const int qy=int(min(top+sg,g.h-1u));
      for(ushort r=0;r<8;++r){
        const int qx=int(min(left+r,g.w-1u));
        const bool active=idx<count && ky>=qy-radius && ky<qy+radius && kx>=qx-radius && kx<qx+radius;
        const float score=active?scratch[sg*8+r][lane]*0.125f:-1.0e30f;
        const float next_max=max(maxima[r],simd_max(score)), alpha=fast::exp(maxima[r]-next_max);
        const float e=active?fast::exp(score-next_max):0.0f;
        prob[sg*8+r][lane]=half(e);
        row_alpha[r]=alpha;
        sums[r]=fma(sums[r],alpha,simd_sum(e));
        maxima[r]=next_max;
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const float2 scales=float2(row_alpha[indices.x/8],row_alpha[indices.y/8]);
      for(ushort d=0;d<8;++d){
        reinterpret_cast<thread float2&>(output[d].thread_elements())*=scales;
      }
      for(ushort section=0;section<4;++section){
        simdgroup_half8x8 probabilities;
        simdgroup_load(probabilities,&prob[sg*8][section*8],32);
        for(ushort d=0;d<8;++d){
          simdgroup_half8x8 values;
          simdgroup_load(values,&vt[section*8][d*8],64);
          simdgroup_multiply_accumulate(output[d],probabilities,values,output[d]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for(ushort d=0;d<8;++d){
    const float2 values=reinterpret_cast<thread float2&>(output[d].thread_elements());
    for(ushort j=0;j<2;++j){
      const ushort elem=indices[j], r=elem/8, c=elem%8;
      const uint yq=top+sg, xq=left+r;
      if(yq<g.h && xq<g.w){
        const uint bits=as_type<uint>(values[j]/sums[r]);
        y[query_base+(yq*g.w+xq)*64u+d*8u+c]=ushort((bits+0x7fffu+((bits>>16)&1u))>>16);
      }
    }
  }
#undef qt
#undef kt
}
)metal";

struct Geometry { uint32_t h, w, planes, query_heads, kv_heads, condition; };

constexpr NSUInteger query_rows = 8;
constexpr NSUInteger threads = query_rows * 32;

struct Pipeline {
  id<MTLComputePipelineState> function;
  Pipeline() {
    id<MTLDevice> device = at::mps::getCurrentMPSStream()->device();
    NSError* error = nil;
    auto library = [device newLibraryWithSource:source options:nil error:&error];
    TORCH_CHECK(library, "Metal Flash attention compilation failed: ", error.localizedDescription.UTF8String);
    function = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"local_attention_block32"]
                                                     error:&error];
    TORCH_CHECK(function && function.threadExecutionWidth == 32 && function.maxTotalThreadsPerThreadgroup >= threads,
                "Metal Flash attention pipeline failed: ",
                error ? error.localizedDescription.UTF8String : "unexpected SIMD width");
  }
};

id<MTLBuffer> storage(const at::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}
}

at::Tensor metal_local_attention(const at::Tensor& query, const at::Tensor& key,
                                 const at::Tensor& value, int height, int width,
                                 bool condition_only) {
  @autoreleasepool {
    TORCH_CHECK(height > 0 && width > 0 && query.dim() == 4 && key.dim() == 4 &&
        value.sizes() == key.sizes() && query.size(0) == key.size(0) &&
        query.size(3) == 64 && key.size(3) == 64 && query.size(1) > 0 && key.size(1) > 0 &&
        query.size(1) % key.size(1) == 0 && query.device().is_mps() &&
        key.device() == query.device() && value.device() == query.device() &&
        query.scalar_type() == at::kBFloat16 && query.scalar_type() == key.scalar_type() &&
        value.scalar_type() == key.scalar_type(), "unsupported Metal Flash attention tensors");
    const int64_t n = int64_t(height) * width;
    TORCH_CHECK(query.size(2) == n && key.size(2) % n == 0 &&
                key.size(2) / n >= 1 && key.size(2) / n <= 4, "invalid Flash attention geometry");
    static Pipeline kernel;
    auto q = query.contiguous(), k = key.contiguous(), v = value.contiguous();
    auto out = at::empty(q.sizes(), q.options());
    Geometry geometry{uint32_t(height), uint32_t(width), uint32_t(key.size(2)/n),
                      uint32_t(query.size(1)), uint32_t(key.size(1)), uint32_t(condition_only)};
    auto* stream = at::mps::getCurrentMPSStream();
    const auto tile_rows = (height + query_rows - 1) / query_rows;
    const auto query_groups = ((width + 7) / 8) * tile_rows;
    dispatch_sync(stream->queue(), ^{
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:kernel.function];
      [enc setBuffer:storage(q) offset:q.storage_offset()*2 atIndex:0];
      [enc setBuffer:storage(k) offset:k.storage_offset()*2 atIndex:1];
      [enc setBuffer:storage(v) offset:v.storage_offset()*2 atIndex:2];
      [enc setBuffer:storage(out) offset:out.storage_offset()*2 atIndex:3];
      [enc setBytes:&geometry length:sizeof(geometry) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(query_groups, query.size(1), query.size(0))
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    });
    return out;
  }
}
}

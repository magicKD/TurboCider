#pragma once

// Standalone research harness, not a model-facing runtime or default backend.
// Every call is joined. A timeout disables reuse and its completion handler
// retains source allocation owners, surfaces and scratch until GPU completion.
#include "../../native/backends/ane_gpu.hpp"
#include "../../native/backends/ane_transfer_kernels.hpp"
#include "../../native/backends/ane_lora_upload_kernels.hpp"
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <mutex>

namespace tc::research {
using namespace ane;
struct LoraUploadProjection {
    DeviceMatrixView ranks, up;
    gpu::Surface destination;
    int begin_row=0, first_channel=0;
    float scale=1.f;
    DType boundary=DType::BF16;
};
class LoraUpload {
    id<MTLDevice> device_=MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> narrow_,projection_;
    std::vector<id<MTLBuffer>> scratch_;
    int rank_,capacity_,bm_,bn_;
    DType operand_;
    bool swapped_,disabled_=false;
    struct Pending {
        std::vector<LoraUploadProjection> sources;
        std::vector<id<MTLBuffer>> scratch;
        id<MTLBuffer> status;
        std::mutex mutex;
        std::condition_variable cv;
        bool done=false,ok=false;
    };
    static void check(bool ok,const char *message) {
        if (!ok) throw std::invalid_argument(message);
    }
    size_t validate(const DeviceMatrixView &v,DType dtype) const {
        check(v.owner && v.buffer && v.rows>0 && v.rows<=1048576 && v.cols==rank_ && v.dtype==dtype,
              "rank upload source geometry/dtype/owner");
        id<MTLBuffer> buffer=(__bridge id<MTLBuffer>)v.buffer;
        const size_t item=dtype==DType::FP32?4:2;
        const size_t pitch=v.row_stride_bytes?v.row_stride_bytes:size_t(rank_)*item;
        check(buffer.device==device_ && v.buffer_bytes<=buffer.length && v.offset_bytes%item==0 &&
              pitch>=size_t(rank_)*item && pitch%item==0 && pitch<=UINT32_MAX &&
              pitch<=SIZE_MAX/size_t(v.rows) && v.offset_bytes<=v.buffer_bytes &&
              size_t(v.rows-1)*pitch+size_t(rank_)*item<=v.buffer_bytes-v.offset_bytes,
              "rank upload source extent/device");
        return pitch;
    }
    static bool aliases_surface(const DeviceMatrixView &v,const gpu::Surface &surface) {
        id<MTLBuffer> buffer=(__bridge id<MTLBuffer>)v.buffer;
        if (!buffer.contents) return false;
        const auto source=reinterpret_cast<uintptr_t>(buffer.contents);
        const auto target=reinterpret_cast<uintptr_t>(surface.data());
        const size_t bytes=IOSurfaceGetAllocSize(static_cast<IOSurfaceRef>(surface.native_iosurface()));
        // Compare entire backing intervals, not ObjC buffer/IOSurface handles.
        // A second no-copy buffer may refer to the same physical allocation.
        return source<=target ? target-source<buffer.length : source-target<bytes;
    }
  public:
    LoraUpload(int rank,DType operand,int capacity=4224,bool swapped=true,int bm=16,int bn=128)
        :rank_(rank),capacity_(capacity),bm_(bm),bn_(bn),operand_(operand),swapped_(swapped) {
        check((rank==64 || rank==128 || rank==256 || rank==512) && capacity>0 && capacity<=4224 &&
              (operand==DType::BF16 || operand==DType::FP16) && (bm==16 || bm==32) && (bn==64 || bn==128),
              "rank upload unsupported specialization/capacity");
        queue_=[device_ newCommandQueue];check(queue_!=nil,"rank upload command queue unavailable");
        const std::string source="#define TC_RANK "+std::to_string(rank)+"\n#define TC_BF16 "+
            std::to_string(operand==DType::BF16)+"\n#define TC_BM "+std::to_string(bm)+
            "\n#define TC_BN "+std::to_string(bn)+"\n#define TC_SWAPPED "+std::to_string(swapped)+"\n"+
            gpu::transfer_source+gpu::lora_upload_source;
        NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
        options.mathMode=MTLMathModeSafe;options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
        options.languageVersion=MTLLanguageVersion4_0;
        id<MTLLibrary> library=[device_ newLibraryWithSource:@(source.c_str()) options:options error:&error];
        if (!library) throw std::runtime_error(std::string("rank upload shader: ")+error.localizedDescription.UTF8String);
        auto pipeline=[&](NSString *name) {
            id<MTLComputePipelineState> p=[device_ newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
            if (!p) throw std::runtime_error(std::string("rank upload pipeline: ")+error.localizedDescription.UTF8String);
            return p;
        };
        narrow_=pipeline(@"tc_ane_lora_rank_narrow");projection_=pipeline(@"tc_ane_lora_b_upload");
        check(projection_.threadExecutionWidth==32 && projection_.maxTotalThreadsPerThreadgroup>=128,
              "rank upload MPP SIMD capacity unavailable");
        for (int i=0;i<2;++i) {
            scratch_.push_back([device_ newBufferWithLength:size_t(capacity)*rank*2 options:MTLResourceStorageModePrivate]);
            check(scratch_.back()!=nil,"rank upload bounded scratch allocation failed");
        }
    }
    uint64_t scratch_bytes() const { return uint64_t(capacity_)*rank_*4; }
    gpu::Completion run(std::vector<LoraUploadProjection> sources,uint32_t &flags,
                        std::chrono::milliseconds timeout=std::chrono::seconds(30)) {
        check(!disabled_ && !sources.empty() && sources.size()<=2,"rank upload disabled or too many projections");
        for (const auto &p:sources) {
            validate(p.ranks,DType::FP32);validate(p.up,operand_);
            check(p.destination.element()==gpu::Element::FP16 && !p.destination.is_view() &&
                  p.destination.columns()<=uint32_t(capacity_) && p.destination.rows()>0 &&
                  p.destination.pitch()%2==0 && p.destination.pitch()/2<=UINT32_MAX &&
                  p.begin_row>=0 && p.begin_row<=p.ranks.rows && p.first_channel>=0 &&
                  p.first_channel<=p.up.rows && int(p.destination.rows())<=p.up.rows-p.first_channel &&
                  std::isfinite(p.scale) && (p.boundary==DType::BF16 || p.boundary==DType::FP16 || p.boundary==DType::FP32),
                  "rank upload destination/range/scale/boundary");
            check(p.ranks.buffer!=p.up.buffer,"rank upload source aliases");
            for (const auto &target:sources)
                check(!aliases_surface(p.ranks,target.destination) && !aliases_surface(p.up,target.destination),
                      "rank upload source/destination allocations overlap");
        }
        for (size_t i=0;i<sources.size();++i) for (size_t j=0;j<i;++j)
            check(sources[i].destination.native_iosurface()!=sources[j].destination.native_iosurface(),
                  "rank upload destinations alias");
        auto state=std::make_shared<Pending>();state->sources=std::move(sources);state->scratch=scratch_;
        state->status=[device_ newBufferWithLength:4 options:MTLResourceStorageModeShared];
        check(state->status!=nil,"rank upload status allocation failed");
        id<MTLCommandBuffer> command=[queue_ commandBuffer];check(command!=nil,"rank upload command buffer unavailable");
        id<MTLBlitCommandEncoder> clear=[command blitCommandEncoder];check(clear!=nil,"rank upload clear encoder unavailable");
        [clear fillBuffer:state->status range:NSMakeRange(0,4) value:0];[clear endEncoding];
        for (size_t i=0;i<state->sources.size();++i) {
            const auto &p=state->sources[i];
            const uint32_t rows=p.destination.columns();
            struct RankParams {uint32_t rows,pitch,begin,available;} rp{rows,uint32_t(validate(p.ranks,DType::FP32)/4),
                uint32_t(p.begin_row),std::min(rows,uint32_t(p.ranks.rows-p.begin_row))};
            id<MTLComputeCommandEncoder> narrow=[command computeCommandEncoder];check(narrow!=nil,"rank narrow encoder unavailable");
            [narrow setComputePipelineState:narrow_];
            [narrow setBuffer:(__bridge id<MTLBuffer>)p.ranks.buffer offset:p.ranks.offset_bytes atIndex:0];
            [narrow setBuffer:state->scratch[i] offset:0 atIndex:1];[narrow setBuffer:state->status offset:0 atIndex:2];
            [narrow setBytes:&rp length:sizeof(rp) atIndex:3];
            [narrow dispatchThreads:MTLSizeMake(size_t(rows)*rank_,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [narrow endEncoding];
            const size_t destination_bytes=IOSurfaceGetAllocSize(static_cast<IOSurfaceRef>(p.destination.native_iosurface()));
            id<MTLBuffer> destination=[device_ newBufferWithBytesNoCopy:p.destination.data() length:destination_bytes
                options:MTLResourceStorageModeShared deallocator:nil];
            check(destination!=nil,"rank upload surface buffer binding failed");
            struct Params {uint32_t rows,columns,b_pitch,target_pitch,boundary;float scale;} q{rows,p.destination.rows(),
                uint32_t(validate(p.up,operand_)/2),uint32_t(p.destination.pitch()/2),uint32_t(p.boundary),p.scale};
            id<MTLComputeCommandEncoder> project=[command computeCommandEncoder];check(project!=nil,"rank B encoder unavailable");
            [project setComputePipelineState:projection_];[project setBuffer:state->scratch[i] offset:0 atIndex:0];
            [project setBuffer:(__bridge id<MTLBuffer>)p.up.buffer offset:p.up.offset_bytes+size_t(p.first_channel)*q.b_pitch*2 atIndex:1];
            [project setBuffer:destination offset:0 atIndex:2];[project setBuffer:state->status offset:0 atIndex:3];
            [project setBytes:&q length:sizeof(q) atIndex:4];
            const uint32_t m=swapped_?q.columns:q.rows,n=swapped_?q.rows:q.columns;
            [project dispatchThreadgroups:MTLSizeMake((n+bn_-1)/bn_,(m+bm_-1)/bm_,1)
                threadsPerThreadgroup:MTLSizeMake(128,1,1)];[project endEncoding];
        }
        [command addCompletedHandler:^(id<MTLCommandBuffer> done) {
            {std::lock_guard lock(state->mutex);state->done=true;state->ok=done.status==MTLCommandBufferStatusCompleted;}
            state->cv.notify_all();
        }];[command commit];
        std::unique_lock lock(state->mutex);
        if (!state->cv.wait_for(lock,timeout,[&]{return state->done;})) {
            disabled_=true;flags=0;return {false,true,"rank upload completion deadline exceeded"};
        }
        flags=*static_cast<uint32_t *>(state->status.contents);
        if (!state->ok) disabled_=true;
        return {state->ok && flags==0,false,state->ok ? flags?"rank upload nonfinite/overflow":"" : "rank upload GPU command failed"};
    }
};
} // namespace tc::research

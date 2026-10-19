// Diagnostic subprocess only. Reuse the unchanged selector/CB observer and
// sample compute-encoder boundaries ONLY after an encoded Metal event wait.
// Descriptor-based encoder creation changes instrumentation, not stock code.
#include "observer.mm"
#include <algorithm>
#include <memory>
#include <thread>

#ifndef TC_HANDOFF_SOURCE_SHA
#define TC_HANDOFF_SOURCE_SHA "unbound"
#endif
namespace {
struct HandoffEvent { uint64_t event, value; bool wait; };
struct HandoffState {
    std::vector<HandoffEvent> events;
    id<MTLCounterSampleBuffer> counter = nil;
    uint64_t begin = 0, end = 0, gpu_begin = 0, gpu_end = 0;
    int request = -1;
    bool ok = false;
};
std::mutex handoff_mutex;
std::condition_variable handoff_changed;
std::map<void*,std::shared_ptr<HandoffState>> handoff_encoded;
std::vector<std::shared_ptr<HandoffState>> handoff_completed;
std::map<void*,uint64_t> handoff_events;
NSMutableArray *handoff_retained_events;
id<MTLDevice> handoff_device;
id<MTLCounterSet> handoff_counter_set;
uint64_t handoff_pending=0,handoff_dropped=0;
constexpr size_t handoff_capacity=50000;
using EventFn=void(*)(id,SEL,id,uint64_t);
using ComputeFn=id<MTLComputeCommandEncoder>(*)(id,SEL);
EventFn handoff_original_signal=nullptr,handoff_original_wait=nullptr;
ComputeFn handoff_original_compute=nullptr;
CommitFn handoff_original_commit=nullptr;
uint64_t handoff_anchor_begin=0,handoff_anchor_end=0,handoff_gpu_anchor=0,handoff_cpu_anchor=0;

std::shared_ptr<HandoffState> handoff_state(id buffer) {
    auto &state=handoff_encoded[(__bridge void*)buffer];
    if(!state)state=std::make_shared<HandoffState>();
    return state;
}
void handoff_event(id buffer,id event,uint64_t value,bool wait) {
    std::lock_guard lock(handoff_mutex);
    auto [it,inserted]=handoff_events.emplace((__bridge void*)event,0);
    if(inserted){it->second=handoff_events.size();[handoff_retained_events addObject:event];}
    handoff_state(buffer)->events.push_back({it->second,value,wait});
}
void handoff_signal_call(id self,SEL sel,id event,uint64_t value) {
    handoff_event(self,event,value,false);handoff_original_signal(self,sel,event,value);
}
void handoff_wait_call(id self,SEL sel,id event,uint64_t value) {
    handoff_event(self,event,value,true);handoff_original_wait(self,sel,event,value);
}
id<MTLComputeCommandEncoder> handoff_compute_call(id self,SEL sel) {
    std::shared_ptr<HandoffState> state;
    {std::lock_guard lock(handoff_mutex);auto it=handoff_encoded.find((__bridge void*)self);
        if(it!=handoff_encoded.end() && !it->second->counter &&
           std::any_of(it->second->events.begin(),it->second->events.end(),[](auto e){return e.wait;}))state=it->second;}
    if(!state)return handoff_original_compute(self,sel);
    MTLCounterSampleBufferDescriptor *descriptor=[MTLCounterSampleBufferDescriptor new];
    descriptor.counterSet=handoff_counter_set;descriptor.storageMode=MTLStorageModeShared;descriptor.sampleCount=2;
    NSError *error=nil;
    state->counter=[handoff_device newCounterSampleBufferWithDescriptor:descriptor error:&error];
    if(!state->counter)throw std::runtime_error("handoff encoder counter allocation failed");
    MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];pass.dispatchType=MTLDispatchTypeSerial;
    pass.sampleBufferAttachments[0].sampleBuffer=state->counter;
    pass.sampleBufferAttachments[0].startOfEncoderSampleIndex=0;
    pass.sampleBufferAttachments[0].endOfEncoderSampleIndex=1;
    id<MTLComputeCommandEncoder> encoder=[(id<MTLCommandBuffer>)self computeCommandEncoderWithDescriptor:pass];
    if(!encoder)throw std::runtime_error("handoff timestamped encoder creation failed");
    return encoder;
}
void handoff_commit_call(id self,SEL sel) {
    std::shared_ptr<HandoffState> state;
    {std::lock_guard lock(handoff_mutex);auto it=handoff_encoded.find((__bridge void*)self);
        if(it!=handoff_encoded.end()){state=std::move(it->second);handoff_encoded.erase(it);++handoff_pending;}}
    if(state){state->begin=now();state->request=scope.load();
        [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> done){
            state->end=now();state->ok=done.status==MTLCommandBufferStatusCompleted;
            if(done.GPUStartTime>0 && done.GPUEndTime>=done.GPUStartTime){state->gpu_begin=uint64_t(done.GPUStartTime*1e9);state->gpu_end=uint64_t(done.GPUEndTime*1e9);}
            {std::lock_guard lock(handoff_mutex);if(handoff_completed.size()<handoff_capacity)handoff_completed.push_back(state);else ++handoff_dropped;--handoff_pending;}
            handoff_changed.notify_all();
        }];}
    // Includes the unchanged base observer's own forwarding and completion.
    handoff_original_commit(self,sel);
}
bool handoff_drain() {
    std::unique_lock lock(handoff_mutex);
    return handoff_changed.wait_for(lock,std::chrono::seconds(10),[]{return handoff_pending==0;});
}
}

extern "C" int tc_qwen_handoff_install(){@autoreleasepool{
    if(tc_qwen_profile_install(1))return -1;
    try{
        handoff_device=MTLCreateSystemDefaultDevice();
        if(![handoff_device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])throw std::runtime_error("stage-boundary sampling unsupported");
        for(id<MTLCounterSet> candidate in handoff_device.counterSets)
            if([candidate.name isEqual:MTLCommonCounterSetTimestamp])handoff_counter_set=candidate;
        if(!handoff_counter_set)throw std::runtime_error("GPU timestamp counter unavailable");
        handoff_retained_events=[NSMutableArray array];
        id<MTLCommandQueue> queue=[handoff_device newCommandQueue];id<MTLCommandBuffer> buffer=[queue commandBuffer];
        Class cls=object_getClass(buffer);
        handoff_original_signal=reinterpret_cast<EventFn>(hook(cls,"encodeSignalEvent:value:",reinterpret_cast<IMP>(handoff_signal_call),4));
        handoff_original_wait=reinterpret_cast<EventFn>(hook(cls,"encodeWaitForEvent:value:",reinterpret_cast<IMP>(handoff_wait_call),4));
        handoff_original_compute=reinterpret_cast<ComputeFn>(hook(cls,"computeCommandEncoder",reinterpret_cast<IMP>(handoff_compute_call),2));
        handoff_original_commit=reinterpret_cast<CommitFn>(hook(cls,"commit",reinterpret_cast<IMP>(handoff_commit_call),2));
        handoff_anchor_begin=now();[handoff_device sampleTimestamps:&handoff_cpu_anchor gpuTimestamp:&handoff_gpu_anchor];handoff_anchor_end=now();
        return 0;
    }catch(...){restore();return -1;}
}}

extern "C" int tc_qwen_handoff_self_test(){@autoreleasepool{
    if(tc_qwen_profile_self_test())return -1;
    try{
        scope.store(-4);
        NSError *error=nil;
        id<MTLLibrary> library=[handoff_device newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;kernel void increment(device uint*x[[buffer(0)]],uint i[[thread_position_in_grid]]){if(i<64)x[i]+=1;}" options:nil error:&error];
        if(!library)return -1;
        id<MTLComputePipelineState> pipeline=[handoff_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"increment"] error:&error];
        id<MTLBuffer> values=[handoff_device newBufferWithLength:256 options:MTLResourceStorageModeShared];
        id<MTLSharedEvent> event=[handoff_device newSharedEvent];id<MTLCommandQueue> queue=[handoff_device newCommandQueue];
        if(!pipeline||!values||!event||!queue)return -1;
        for(int i=0;i<64;++i)static_cast<uint32_t*>(values.contents)[i]=uint32_t(i);
        id<MTLCommandBuffer> input=[queue commandBuffer],output=[queue commandBuffer];
        [input encodeSignalEvent:event value:1];[output encodeWaitForEvent:event value:1];
        id<MTLComputeCommandEncoder> encoder=[output computeCommandEncoder];[encoder setComputePipelineState:pipeline];[encoder setBuffer:values offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(64,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];
        [input commit];[output commit];[output waitUntilCompleted];
        if(!handoff_drain()||tc_qwen_profile_wait(1000))return -1;
        for(int i=0;i<64;++i)if(static_cast<uint32_t*>(values.contents)[i]!=uint32_t(i+1))return -1;
        // Falsify the assumption that encoder samples exclude an earlier
        // command-buffer event wait. A host-released gate delays actual work
        // for20ms; the encoder interval must remain much shorter.
        scope.store(-5);id<MTLSharedEvent> gate=[handoff_device newSharedEvent];
        id<MTLSharedEvent> delayed=[handoff_device newSharedEvent];
        input=[queue commandBuffer];output=[queue commandBuffer];
        [input encodeWaitForEvent:gate value:1];[input encodeSignalEvent:delayed value:1];
        [output encodeWaitForEvent:delayed value:1];encoder=[output computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];[encoder setBuffer:values offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(64,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];
        [input commit];[output commit];
        std::thread release([gate]{@autoreleasepool{std::this_thread::sleep_for(std::chrono::milliseconds(20));gate.signaledValue=1;}});
        [output waitUntilCompleted];release.join();
        if(!handoff_drain()||tc_qwen_profile_wait(1000))return -1;
        for(int i=0;i<64;++i)if(static_cast<uint32_t*>(values.contents)[i]!=uint32_t(i+2))return -1;
        std::lock_guard lock(handoff_mutex);int sampled=0;
        for(const auto &state:handoff_completed)if(state->request==-4 && state->counter){
            NSData *data=[state->counter resolveCounterRange:NSMakeRange(0,2)];
            if(data.length!=2*sizeof(MTLCounterResultTimestamp))return -1;
            const auto *ticks=static_cast<const MTLCounterResultTimestamp*>(data.bytes);
            if(!state->ok||!ticks[0].timestamp||ticks[0].timestamp==MTLCounterErrorValue||ticks[1].timestamp==MTLCounterErrorValue||ticks[1].timestamp<=ticks[0].timestamp)return -1;
            ++sampled;
        }
        return sampled==1?0:-1;
    }catch(...){return -1;}
}}

extern "C" char *tc_qwen_handoff_finish(){@autoreleasepool{
    const bool drained=handoff_drain();
    uint64_t end_before=now(),cpu_end=0,gpu_end=0;[handoff_device sampleTimestamps:&cpu_end gpuTimestamp:&gpu_end];const uint64_t end_after=now();
    char *base=tc_qwen_profile_finish();if(!base)return nullptr;
    NSData *bytes=[NSData dataWithBytes:base length:std::strlen(base)];tc_qwen_profile_string_free(base);
    NSMutableDictionary *result=[NSJSONSerialization JSONObjectWithData:bytes options:NSJSONReadingMutableContainers error:nil];
    if(!result)return nullptr;
    NSMutableArray *records=[NSMutableArray array];
    {std::lock_guard lock(handoff_mutex);
        for(const auto &state:handoff_completed){NSMutableArray *events=[NSMutableArray array];
            for(const auto &e:state->events)[events addObject:@{@"event":@(e.event),@"value":@(e.value),@"wait":@(e.wait)}];
            NSMutableDictionary *row=[@{@"request":@(state->request),@"begin_ns":@(state->begin),@"end_ns":@(state->end),@"ok":@(state->ok),
                @"gpu_begin_ns":@(state->gpu_begin),@"gpu_end_ns":@(state->gpu_end),@"events":events,@"encoder_sampled":@(bool(state->counter))} mutableCopy];
            if(state->counter){NSData *data=[state->counter resolveCounterRange:NSMakeRange(0,2)];
                row[@"resolved_bytes"]=@(data.length);
                if(data.length==2*sizeof(MTLCounterResultTimestamp)){const auto *ticks=static_cast<const MTLCounterResultTimestamp*>(data.bytes);
                    row[@"encoder_begin_ticks"]=@(ticks[0].timestamp);row[@"encoder_end_ticks"]=@(ticks[1].timestamp);}}
            [records addObject:row];
        }
        result[@"handoff"]=@{@"schema":@"tc-metal-event-encoder-timestamps-v1",@"source_sha256":@TC_HANDOFF_SOURCE_SHA,
            @"drained":@(drained),@"pending":@(handoff_pending),@"dropped":@(handoff_dropped),@"uncommitted_event_buffers":@(handoff_encoded.size()),
            @"records":records,@"calibration":@{@"begin_host_before_ns":@(handoff_anchor_begin),@"begin_host_after_ns":@(handoff_anchor_end),
                @"begin_gpu_ticks":@(handoff_gpu_anchor),@"begin_cpu_ticks":@(handoff_cpu_anchor),@"end_host_before_ns":@(end_before),
                @"end_host_after_ns":@(end_after),@"end_gpu_ticks":@(gpu_end),@"end_cpu_ticks":@(cpu_end)},
            @"scope":@"Diagnostic encoder-boundary GPU timestamp samples after encoded event waits; not ANE hardware time, per-shader busy time, or stock performance."};
    }
    handoff_retained_events=nil;
    return encode(result);
}}

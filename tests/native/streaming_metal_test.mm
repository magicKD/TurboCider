#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "streaming/context.hpp"
#include <cassert>
#include <iostream>

using namespace tc::streaming;
constexpr uint32_t elements=4096, groups=13, passes=3;
constexpr uint64_t bytes=elements*sizeof(float);

class MetalSlots final : public ModelSlotAdapter {
    id<MTLDevice> device;
    id<MTLComputePipelineState> kernel;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> output;
    std::vector<id<MTLBuffer>> slots;
    id<MTLCommandBuffer> last;
    uint64_t seq=0;
public:
    explicit MetalSlots(id<MTLDevice> gpu) : device(gpu) {
        NSError *error=nil;
        NSString *source=@"#include <metal_stdlib>\nusing namespace metal;\n"
                         "kernel void copy_weight(device const float *src [[buffer(0)]],"
                         " device float *dst [[buffer(1)]],uint i [[thread_position_in_grid]])"
                         "{dst[i]=src[i]*2.0f;}";
        id<MTLLibrary> lib=[device newLibraryWithSource:source options:nil error:&error];
        if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
        kernel=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"copy_weight"] error:&error];
        if(!kernel)throw std::runtime_error(error.localizedDescription.UTF8String);
        queue=[device newCommandQueue];
        output=[device newBufferWithLength:bytes*groups*passes options:MTLResourceStorageModeShared];
        assert(queue && output);
    }
    void create_pool(const PoolLayout &p) override {
        for(const auto &s:p.slots){
            id<MTLBuffer> buffer=[device newBufferWithLength:s.capacity_bytes options:MTLResourceStorageModeShared];
            if(!buffer)throw std::runtime_error("Metal test allocation failed");
            slots.push_back(buffer);
        }
    }
    FillJob make_fill_job(const Group &,const tc_stream_slot_ticket_v1 &t) override {
        return {t,this,[](void *user,const tc_stream_slot_ticket_v1 *ticket,
                         const std::atomic<bool> *cancel,uint64_t *loaded){
            @autoreleasepool {
                if(cancel->load())return -1;
                auto &self=*static_cast<MetalSlots *>(user);
                float *destination=(float *)self.slots[ticket->slot].contents;
                const float value=float(1+ticket->item.pass*1000+ticket->item.group);
                for(uint32_t i=0;i<elements;++i)destination[i]=value;
                *loaded=bytes;return 0;
            }
        }};
    }
    void encode_prefix(uint32_t) override {}
    void prepare_group(const Group &,const tc_stream_slot_ticket_v1 &) override {}
    ReaderSet encode_group(const Group &,const tc_stream_slot_ticket_v1 &ticket,CompletionMailbox &mailbox) override {
        @autoreleasepool {
            id<MTLCommandBuffer> cb=[queue commandBuffer];
            id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:kernel];
            [enc setBuffer:slots[ticket.slot] offset:0 atIndex:0];
            [enc setBuffer:output offset:bytes*(ticket.item.pass*groups+ticket.item.group) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(elements,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
            const auto t=ticket;
            tc_stream_reader_fence_v1 fence{1,++seq};
            CompletionMailbox *sink=&mailbox;
            [cb addCompletedHandler:^(id<MTLCommandBuffer> completed){
                tc_stream_completion_v1 event{};
                event.struct_size=sizeof(event);event.version=TC_STREAM_SLOT_ABI_V1;
                event.kind=TC_STREAM_READER_COMPLETE;event.ticket=t;event.fence=fence;
                event.status=completed.status==MTLCommandBufferStatusCompleted?0:-1;
                sink->post(event);
            }];
            [cb commit];last=cb;
            ReaderSet readers;readers.count=1;readers.fences[0]=fence;return readers;
        }
    }
    bool drain() noexcept override {
        if(last){[last waitUntilCompleted];return last.status==MTLCommandBufferStatusCompleted;}
        return true;
    }
    void destroy_pool() noexcept override {slots.clear();last=nil;}
    void verify() {
        const float *data=(const float *)output.contents;
        for(uint32_t p=0;p<passes;++p)for(uint32_t g=0;g<groups;++g)
            for(uint32_t i=0;i<elements;++i)
                assert(data[(p*groups+g)*elements+i]==float(1+p*1000+g)*2.0f);
    }
};

int main(){
    @autoreleasepool {
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        if(!device){std::cout<<"SKIP: no Metal device is available\n";return 0;}
        for(uint32_t k=1;k<=3;++k){
            StageLayout layout;layout.id="metal-synthetic";layout.group_size=1;
            layout.slot_count=k;layout.distance=k-1;layout.workers=1;layout.pass_count=passes;
            PoolLayout pool;pool.id=0;
            for(uint32_t i=0;i<k;++i)pool.slots.push_back({{bytes},bytes});
            layout.pools.push_back(pool);
            for(uint32_t i=0;i<groups;++i)layout.groups.push_back({i,0,i%k,{i},{bytes},bytes});
            auto adapter=std::make_shared<MetalSlots>(device);
            StageExecutor executor(1,1,adapter);std::atomic<bool> cancelled{false};
            auto counters=executor.run(layout,cancelled);
            assert(counters.pool_creates==1 && counters.slot_bundles==k && counters.fills==groups*passes);
            adapter->verify();
        }
        std::cout<<"PASS: real Metal slot reuse K=1/2/3, 3 passes, output equality; synthetic, not model certification\n";
    }
}

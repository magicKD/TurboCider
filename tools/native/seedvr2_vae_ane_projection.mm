// Complete-VAE integration of the public shared-surface projection experiment.
// Not linked into ordinary native/App builds. No private ANE APIs or fallback.
#include "seedvr2_vae_ane_projection.hpp"
#include "seedvr2_vae_upscale_io.hpp"
namespace tc::seedvr2 {
namespace {
using namespace tc::seedvr2_upscale_probe;
struct Geometry {uint32_t h,w,th,tw,y,x,cs,rs,ws;};
// Same index/FP16-rounding recipe as the separately witnessed layer probe.
const char *projection_shader=R"METAL(
#include <metal_stdlib>
using namespace metal;
struct P {uint h,w,th,tw,y,x,cs,rs,ws;};
kernel void tc_seedvr2_ane_projection_pack(device const half *src [[buffer(0)]],device half *dst [[buffer(1)]],
    constant P &p [[buffer(2)]],uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half tile[16][17];
    uint lc=tid%16,lp=tid/16,c=group.x*16+lc,pos=group.y*16+lp;
    half value=half(0);
    if(c<256 && pos<p.th*p.tw)value=src[((p.y+pos/p.tw)*p.w+p.x+pos%p.tw)*256+c];
    tile[lp][lc]=value;threadgroup_barrier(mem_flags::mem_threadgroup);
    c=group.x*16+lp;pos=group.y*16+lc;
    if(c<256 && pos<p.th*p.tw)dst[c*p.cs+(pos/p.tw)*p.rs+(pos%p.tw)*p.ws]=tile[lc][lp];
}
kernel void tc_seedvr2_ane_projection_consume(device const half *src [[buffer(0)]],device half *dst [[buffer(1)]],
    constant P &p [[buffer(2)]],device const half *bias [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half tile[16][17];
    uint lp=tid%16,lc=tid/16,c=group.x*16+lc,pos=group.y*16+lp;
    half value=half(0);
    if(c<256 && pos<4*p.th*p.tw) {
        uint row=pos/(2*p.tw),col=pos%(2*p.tw),phase=(row%2)*2+col%2;
        value=half(float(src[(phase*256+c)*p.cs+(row/2)*p.rs+(col/2)*p.ws])+float(bias[phase*256+c]));
    }
    tile[lc][lp]=value;threadgroup_barrier(mem_flags::mem_threadgroup);
    c=group.x*16+lp;pos=group.y*16+lc;
    if(c<256 && pos<4*p.th*p.tw)
        dst[((2*p.y+pos/(2*p.tw))*(2*p.w)+2*p.x+pos%(2*p.tw))*256+c]=tile[lp][lc];
}
)METAL";
struct ProjectionIO {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue=nil;
    id<MTLComputePipelineState> pack=nil,consume=nil;
    ProjectionIO() {
        check(device!=nil,"projection Metal device unavailable");queue=[device newCommandQueue];check(queue!=nil,"projection queue unavailable");
        auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
        options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
        NSError *error=nil;auto library=[device newLibraryWithSource:@(projection_shader) options:options error:&error];
        if(!library)throw std::runtime_error(std::string("projection shader: ")+error.localizedDescription.UTF8String);
        pack=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"tc_seedvr2_ane_projection_pack"] error:&error];
        consume=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"tc_seedvr2_ane_projection_consume"] error:&error];
        check(pack && consume && pack.maxTotalThreadsPerThreadgroup>=256 && consume.maxTotalThreadsPerThreadgroup>=256,
              "projection transpose kernels unavailable");
    }
    std::unique_ptr<Submission> submit(const Tensor &x,const Tensor &z,const Tensor &bias,
        Geometry geometry,int columns,Slot *producer,int next,Slot *consumer,int previous) {
        auto pending=std::make_unique<Submission>();pending->owners={x,z,bias};
        pending->command=[queue commandBuffer];check(pending->command!=nil,"projection command allocation failed");
        auto encode=[&](id<MTLComputePipelineState> kernel,id<MTLBuffer> source,id<MTLBuffer> target,Geometry p,bool packing){
            auto encoder=[pending->command computeCommandEncoder];check(encoder!=nil,"projection encoder allocation failed");
            [encoder setComputePipelineState:kernel];[encoder setBuffer:source offset:0 atIndex:0];[encoder setBuffer:target offset:0 atIndex:1];
            [encoder setBytes:&p length:sizeof(p) atIndex:2];if(!packing)[encoder setBuffer:buffer(bias,device) offset:0 atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(16,((packing?1:4)*p.th*p.tw+15)/16,1)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];
        };
        if(producer){
            auto p=geometry;p.y=(next/columns)*p.th;p.x=(next%columns)*p.tw;
            p.cs=producer->input.array.strides[1].unsignedIntValue;p.rs=producer->input.array.strides[2].unsignedIntValue;
            [pending->surfaces addObject:producer->input.array];
            encode(pack,buffer(x,device),producer->input.metal,p,true);
        }
        if(consumer){
            check(consumer->last_tile==previous,"projection output reused before its consumer");
            auto p=geometry;p.y=(previous/columns)*p.th;p.x=(previous%columns)*p.tw;
            p.cs=consumer->output.array.strides[1].unsignedIntValue;p.rs=consumer->output.array.strides[2].unsignedIntValue;
            [pending->surfaces addObject:consumer->output.array];
            encode(consume,consumer->output.metal,buffer(z,device),p,false);
        }
        [pending->command commit];return pending;
    }
};
struct ProjectionRuntime {
    ProjectionIO io;
    MLModel *model=nil;
    std::vector<std::unique_ptr<Slot>> slots;
    std::shared_ptr<ExperimentalProjectionStats> stats;
    std::atomic<bool> &cancelled;
    std::atomic_flag active=ATOMIC_FLAG_INIT;
    ProjectionRuntime(const std::filesystem::path &manifest,bool cpu,std::atomic<bool> &stop,
                      std::shared_ptr<ExperimentalProjectionStats> observation):stats(std::move(observation)),cancelled(stop){
        @autoreleasepool {
            checkpoint(cancelled);NSError *error=nil;auto content=[NSData dataWithContentsOfFile:@(manifest.c_str())];
            check(content && content.length<65536,"projection manifest unavailable/too big");
            NSDictionary *spec=[NSJSONSerialization JSONObjectWithData:content options:0 error:&error];
            check([spec isKindOfClass:NSDictionary.class] &&
                [spec[@"schema"] isEqual:@"turbocider-seedvr2-vae-upscale-probe-v1"] &&
                [spec[@"checkpoint_sha256"] isEqual:@"20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"] &&
                [spec[@"layer"] isEqual:@"decoder.up_blocks.2.upsamplers.0.upscale_conv"] &&
                [spec[@"qualified"] isEqual:@NO] && [spec[@"variant"] isEqual:@"projection"] &&
                [spec[@"precision"] isEqual:@"fp16"] && [spec[@"bias_applied"] isEqual:@NO] &&
                [spec[@"channel_order"] isEqual:@"spatial_row_spatial_col_channel"] &&
                [spec[@"compute_units"] isEqual:@"cpuAndNeuralEngine"],"original unqualified projection-only manifest required");
            const int th=[spec[@"rows"] intValue],tw=[spec[@"cols"] intValue];
            check((th==32 || th==64 || th==128) && tw==th &&
                [spec[@"input_shape"] isEqual:shape(256,th,tw)] && [spec[@"output_shape"] isEqual:shape(1024,th,tw)] &&
                [spec[@"original_shape"] isEqual:@[@1024,@256,@1,@1,@1]],"projection manifest geometry differs");
            check([spec[@"compiled"] isKindOfClass:NSDictionary.class] &&
                [spec[@"compiled"][@"path"] isEqual:@"upscale.mlmodelc"],"projection compiled path differs");
            const auto path=manifest.parent_path()/"upscale.mlmodelc";
            auto config=[MLModelConfiguration new];config.computeUnits=cpu?MLComputeUnitsCPUOnly:MLComputeUnitsCPUAndNeuralEngine;
            auto started=Clock::now();model=[MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(path.c_str())] configuration:config error:&error];
            if(!model)throw std::runtime_error(std::string("projection model load: ")+error.localizedDescription.UTF8String);
            stats->load_seconds=elapsed(started);stats->cpu_control=cpu;stats->tile_rows=th;stats->tile_cols=tw;
            for(int i=0;i<2;++i)slots.push_back(std::make_unique<Slot>(io.device,256,1024,th,tw,th,tw));
            auto input=model.modelDescription.inputDescriptionsByName[@"x"].multiArrayConstraint;
            auto output=model.modelDescription.outputDescriptionsByName[@"y"].multiArrayConstraint;
            check(model.modelDescription.inputDescriptionsByName.count==1 && model.modelDescription.outputDescriptionsByName.count==1 &&
                input.dataType==MLMultiArrayDataTypeFloat16 && output.dataType==MLMultiArrayDataTypeFloat16 &&
                [input.shape isEqual:slots[0]->input.array.shape] && [output.shape isEqual:slots[0]->output.array.shape],
                "projection compiled descriptors differ from backing");
            for(const auto &slot:slots)stats->slot_bytes+=slot->input.bytes()+slot->output.bytes();
            checkpoint(cancelled);
        }
    }
    Tensor run(const Tensor &input,const Tensor &weight,const Tensor &bias){
        check(!active.test_and_set(),"projection runtime cannot be used concurrently");
        struct Unlock{std::atomic_flag &value;~Unlock(){value.clear();}}unlock{active};
        @autoreleasepool {
            checkpoint(cancelled);const auto started=Clock::now();
            require(input.ndim()==5 && input.shape(0)==1 && input.shape(1)==1 && input.shape(4)==256 &&
                input.dtype()==mx::float16 && weight.shape()==mx::Shape{1024,1,1,1,256} && weight.dtype()==mx::float16 &&
                bias.shape()==mx::Shape{1024} && bias.dtype()==mx::float16,"projection runtime operand recipe differs");
            const int h=input.shape(2),w=input.shape(3),th=stats->tile_rows,tw=stats->tile_cols;
            require(h>=th && w>=tw && h<=512 && w<=512 && h%th==0 && w%tw==0,"projection requires complete fixed tiles within a 1024 canvas");
            auto x=mx::contiguous(input);if(x.offset()!=0)x=mx::copy(x);
            auto shift=mx::contiguous(bias);if(shift.offset()!=0)shift=mx::copy(shift);
            // Poison catches incomplete tile writes through the unchanged finite
            // pixel gate. It is included in this full-VAE experiment's timing.
            auto z=mx::full({1,1,h*2,w*2,256},std::numeric_limits<float>::quiet_NaN(),mx::float16);
            mx::eval({x,shift,z});mx::synchronize();checkpoint(cancelled);
            const int columns=w/tw,tiles=(h/th)*columns;
            Geometry geometry{uint32_t(h),uint32_t(w),uint32_t(th),uint32_t(tw),0,0,0,0,1};
            auto predict=[&](Slot &slot,int tile){
                checkpoint(cancelled);NSError *error=nil;
                auto result=[model predictionFromFeatures:slot.provider options:slot.options error:&error];
                if(!result)throw std::runtime_error(std::string("projection predict: ")+error.localizedDescription.UTF8String);
                ++stats->coreml_calls;auto output=[result featureValueForName:@"y"].multiArrayValue;
                // Accessing MLMultiArray.dataPointer locks its pixel buffer,
                // which then cannot be reused as a Core ML output backing.
                // Inspect identity through pixelBuffer, never CPU data access.
                check(output && output.dataType==MLMultiArrayDataTypeFloat16 && output.pixelBuffer==slot.output.pixel &&
                    [output.shape isEqual:slot.output.array.shape] &&
                    [output.strides isEqual:slot.output.array.strides],"projection backing/shape/stride not honored");
                ++stats->backing_hits;slot.last_tile=tile;checkpoint(cancelled);
            };
            auto first=io.submit(x,z,shift,geometry,columns,slots[0].get(),0,nullptr,-1);first->join();first.reset();
            for(int tile=0;tile<tiles;++tile){
                checkpoint(cancelled);
                auto future=io.submit(x,z,shift,geometry,columns,tile+1<tiles?slots[(tile+1)%2].get():nullptr,tile+1,
                    tile>0?slots[(tile-1)%2].get():nullptr,tile-1);
                predict(*slots[tile%2],tile);future->join();
            }
            auto final=io.submit(x,z,shift,geometry,columns,nullptr,-1,slots[(tiles-1)%2].get(),tiles-1);final->join();final.reset();
            checkpoint(cancelled);++stats->frames;stats->boundary_seconds+=elapsed(started);return z;
        }
    }
};
}
ExperimentalUpscaleProjection make_experimental_ane_projection(const std::filesystem::path &manifest,bool cpu,
    std::atomic<bool> &cancelled,std::shared_ptr<ExperimentalProjectionStats> &stats){
    stats=std::make_shared<ExperimentalProjectionStats>();
    auto runtime=std::make_shared<ProjectionRuntime>(manifest,cpu,cancelled,stats);
    return [runtime](const Tensor &input,const Tensor &weight,const Tensor &bias){return runtime->run(input,weight,bias);};
}
}

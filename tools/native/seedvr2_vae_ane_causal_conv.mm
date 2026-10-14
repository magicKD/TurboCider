// Public fixed-tile causal convolution. Direct separate-frame/halo packing;
// no full temporal concatenation. Offline tooling, not an App/default backend.
#include "seedvr2_vae_ane_causal_conv.hpp"
#include "seedvr2_vae_upscale_io.hpp"
namespace tc::seedvr2 {
namespace {
using namespace tc::seedvr2_upscale_probe;
struct Geometry{uint32_t h,w,th,tw,hi,wi,y,x,cs,rs;};
const char *shader=R"METAL(
#include <metal_stdlib>
using namespace metal;
struct P{uint h,w,th,tw,hi,wi,y,x,cs,rs;};
kernel void tc_seedvr2_ane_causal_pack(device const ushort *current [[buffer(0)]],device const ushort *past0 [[buffer(1)]],
    device const ushort *past1 [[buffer(2)]],device ushort *out [[buffer(3)]],constant P &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]){
    threadgroup ushort tile[16][17];
    uint lc=tid%16,lp=tid/16,c=group.x*16+lc,pos=group.y*16+lp;
    ushort value=0;
    if(c<768 && pos<p.hi*p.wi){
        int iy=int(p.y+pos/p.wi)-1,ix=int(p.x+pos%p.wi)-1;
        if(iy>=0 && iy<int(p.h) && ix>=0 && ix<int(p.w)){
            uint channel=c%256,index=(uint(iy)*p.w+uint(ix))*256+channel;
            value=c<256?past0[index]:(c<512?past1[index]:current[index]);
        }
    }
    tile[lp][lc]=value;threadgroup_barrier(mem_flags::mem_threadgroup);
    c=group.x*16+lp;pos=group.y*16+lc;
    if(c<768 && pos<p.hi*p.wi)out[c*p.cs+(pos/p.wi)*p.rs+pos%p.wi]=tile[lc][lp];
}
kernel void tc_seedvr2_ane_causal_consume(device const half *src [[buffer(0)]],device half *out [[buffer(1)]],
    device const half *bias [[buffer(2)]],constant P &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]){
    threadgroup half tile[16][17];
    uint lp=tid%16,lc=tid/16,c=group.x*16+lc,pos=group.y*16+lp;
    half value=0;
    if(c<128 && pos<p.th*p.tw)value=half(float(src[c*p.cs+(pos/p.tw)*p.rs+pos%p.tw])+float(bias[c]));
    tile[lc][lp]=value;threadgroup_barrier(mem_flags::mem_threadgroup);
    c=group.x*16+lp;pos=group.y*16+lc;
    if(c<128 && pos<p.th*p.tw && p.y+pos/p.tw<p.h && p.x+pos%p.tw<p.w)
        out[((p.y+pos/p.tw)*p.w+p.x+pos%p.tw)*128+c]=tile[lp][lc];
}
)METAL";
struct CausalIO{
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue=nil;
    id<MTLComputePipelineState> pack=nil,consume=nil;
    CausalIO(){
        check(device!=nil,"causal Metal device unavailable");queue=[device newCommandQueue];check(queue!=nil,"causal Metal queue unavailable");
        auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
        options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;NSError *error=nil;
        auto library=[device newLibraryWithSource:@(shader) options:options error:&error];
        if(!library)throw std::runtime_error(std::string("causal shader: ")+error.localizedDescription.UTF8String);
        pack=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"tc_seedvr2_ane_causal_pack"] error:&error];
        consume=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"tc_seedvr2_ane_causal_consume"] error:&error];
        check(pack && consume && pack.maxTotalThreadsPerThreadgroup>=256 && consume.maxTotalThreadsPerThreadgroup>=256,
            "causal transpose kernels unavailable");
    }
    std::unique_ptr<Submission> submit(const Tensor &current,const Tensor &past0,const Tensor &past1,const Tensor &out,
        const Tensor &bias,Geometry geometry,int columns,Slot *producer,int next,Slot *consumer,int previous){
        auto pending=std::make_unique<Submission>();pending->owners={current,past0,past1,out,bias};
        pending->command=[queue commandBuffer];check(pending->command!=nil,"causal command allocation failed");
        if(producer){
            auto p=geometry;p.y=(next/columns)*p.th;p.x=(next%columns)*p.tw;
            p.cs=producer->input.array.strides[1].unsignedIntValue;p.rs=producer->input.array.strides[2].unsignedIntValue;
            [pending->surfaces addObject:producer->input.array];auto encoder=[pending->command computeCommandEncoder];check(encoder!=nil,"causal pack encoder failed");
            [encoder setComputePipelineState:pack];[encoder setBuffer:buffer(current,device) offset:0 atIndex:0];
            [encoder setBuffer:buffer(past0,device) offset:0 atIndex:1];[encoder setBuffer:buffer(past1,device) offset:0 atIndex:2];
            [encoder setBuffer:producer->input.metal offset:0 atIndex:3];[encoder setBytes:&p length:sizeof(p) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(48,(p.hi*p.wi+15)/16,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];
        }
        if(consumer){
            check(consumer->last_tile==previous,"causal output slot reused before consumption");
            auto p=geometry;p.y=(previous/columns)*p.th;p.x=(previous%columns)*p.tw;
            p.cs=consumer->output.array.strides[1].unsignedIntValue;p.rs=consumer->output.array.strides[2].unsignedIntValue;
            [pending->surfaces addObject:consumer->output.array];auto encoder=[pending->command computeCommandEncoder];check(encoder!=nil,"causal consume encoder failed");
            [encoder setComputePipelineState:consume];[encoder setBuffer:consumer->output.metal offset:0 atIndex:0];
            [encoder setBuffer:buffer(out,device) offset:0 atIndex:1];[encoder setBuffer:buffer(bias,device) offset:0 atIndex:2];
            [encoder setBytes:&p length:sizeof(p) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(8,(p.th*p.tw+15)/16,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];
        }
        [pending->command commit];return pending;
    }
};
struct CausalRuntime{
    CausalIO io;MLModel *model=nil;std::vector<std::unique_ptr<Slot>> slots;
    std::shared_ptr<ExperimentalCausalConvStats> stats;std::atomic<bool> &cancelled;bool verify_io;
    std::atomic_flag active=ATOMIC_FLAG_INIT;
    CausalRuntime(const std::filesystem::path &manifest,bool cpu,std::atomic<bool> &stop,
        std::shared_ptr<ExperimentalCausalConvStats> observed,bool verify):stats(std::move(observed)),cancelled(stop),verify_io(verify){
        @autoreleasepool{
            checkpoint(cancelled);auto content=[NSData dataWithContentsOfFile:@(manifest.c_str())];check(content && content.length<65536,"causal manifest unavailable/too big");
            NSError *error=nil;NSDictionary *spec=[NSJSONSerialization JSONObjectWithData:content options:0 error:&error];
            check([spec isKindOfClass:NSDictionary.class] && [spec[@"schema"] isEqual:@"turbocider-seedvr2-vae-conv-probe-v1"] &&
                [spec[@"checkpoint_sha256"] isEqual:@"20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"] &&
                [spec[@"layer"] isEqual:@"decoder.up_blocks.3.resnets.0.conv1"] && [spec[@"original_shape"] isEqual:@[@128,@256,@3,@3,@3]] &&
                [spec[@"qualified"] isEqual:@NO] && [spec[@"precision"] isEqual:@"fp16"] && [spec[@"bias_applied"] isEqual:@NO] &&
                [spec[@"source_operand_recipe"] isEqual:@"original-fp16-temporal-channel-stack"] &&
                [spec[@"input_layout"] isEqual:@"N(T*C)HW"] && [spec[@"output_layout"] isEqual:@"NOHW"] &&
                [spec[@"compute_units"] isEqual:@"cpuAndNeuralEngine"] && [spec[@"output_channels"] isEqual:@128] &&
                [spec[@"total_output_channels"] isEqual:@128],"original full-output causal bank required");
            const int th=[spec[@"rows"] intValue],tw=[spec[@"cols"] intValue];
            check((th==32 || th==64 || th==128) && tw==th && [spec[@"input_rows"] intValue]==th+2 &&
                [spec[@"input_cols"] intValue]==tw+2 && [spec[@"temporal_taps"] intValue]==3 &&
                [spec[@"input_channels"] intValue]==256 && [spec[@"compiled"] isKindOfClass:NSDictionary.class] &&
                [spec[@"compiled"][@"path"] isEqual:@"conv.mlmodelc"],"causal bank geometry/path differs");
            auto config=[MLModelConfiguration new];config.computeUnits=cpu?MLComputeUnitsCPUOnly:MLComputeUnitsCPUAndNeuralEngine;
            const auto path=manifest.parent_path()/"conv.mlmodelc";auto start=Clock::now();
            model=[MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(path.c_str())] configuration:config error:&error];
            if(!model)throw std::runtime_error(std::string("causal Core ML load: ")+error.localizedDescription.UTF8String);
            stats->load_seconds=elapsed(start);stats->cpu_control=cpu;stats->tile_rows=th;stats->tile_cols=tw;
            for(int i=0;i<2;++i)slots.push_back(std::make_unique<Slot>(io.device,768,128,th+2,tw+2,th,tw));
            auto in=model.modelDescription.inputDescriptionsByName[@"x"].multiArrayConstraint;
            auto out=model.modelDescription.outputDescriptionsByName[@"y"].multiArrayConstraint;
            check(model.modelDescription.inputDescriptionsByName.count==1 && model.modelDescription.outputDescriptionsByName.count==1 &&
                in.dataType==MLMultiArrayDataTypeFloat16 && out.dataType==MLMultiArrayDataTypeFloat16 &&
                [in.shape isEqual:slots[0]->input.array.shape] && [out.shape isEqual:slots[0]->output.array.shape],"causal model descriptors differ from backing");
            for(const auto &slot:slots)stats->slot_bytes+=slot->input.bytes()+slot->output.bytes();checkpoint(cancelled);
        }
    }
    void witness_pack(Slot &slot,int tile,Geometry p,int columns,const std::array<Tensor,3> &sources){
        if(!verify_io)return;p.y=(tile/columns)*p.th;p.x=(tile%columns)*p.tw;
        const auto *packed=static_cast<const uint16_t *>(slot.input.metal.contents);
        const auto cs=slot.input.array.strides[1].unsignedLongLongValue,rs=slot.input.array.strides[2].unsignedLongLongValue;
        for(int tap=0;tap<3;++tap)for(uint32_t c=0;c<256;++c)for(uint32_t row=0;row<p.hi;++row)for(uint32_t col=0;col<p.wi;++col){
            const int iy=int(p.y+row)-1,ix=int(p.x+col)-1;
            const uint16_t expected=iy>=0 && iy<int(p.h) && ix>=0 && ix<int(p.w)?sources[tap].data<uint16_t>()[(size_t(iy)*p.w+ix)*256+c]:0;
            check(expected==packed[(tap*256+c)*cs+row*rs+col],"causal temporal/halo pack differs bitwise");++stats->pack_checked_words;
        }
    }
    void witness_consume(Slot &slot,int tile,Geometry p,int columns,const Tensor &result,const Tensor &bias){
        if(!verify_io)return;p.y=(tile/columns)*p.th;p.x=(tile%columns)*p.tw;
        const auto *raw=static_cast<const _Float16 *>(slot.output.metal.contents);const auto *actual=result.data<uint16_t>();
        const auto *shift=bias.data<mx::float16_t>();const auto cs=slot.output.array.strides[1].unsignedLongLongValue,rs=slot.output.array.strides[2].unsignedLongLongValue;
        for(uint32_t row=0;row<p.th && p.y+row<p.h;++row)for(uint32_t col=0;col<p.tw && p.x+col<p.w;++col)for(uint32_t c=0;c<128;++c){
            const _Float16 expected=_Float16(float(raw[c*cs+row*rs+col])+float(shift[c]));
            check(std::memcmp(&expected,actual+((size_t(p.y+row)*p.w+p.x+col)*128+c),2)==0,"causal bias/assembly differs bitwise");++stats->consume_checked_words;
        }
    }
    Tensor run(const Tensor &input,const FrameHistory &previous,const Tensor &weight,const Tensor &bias){
        check(!active.test_and_set(),"causal runtime cannot be used concurrently");
        struct Unlock{std::atomic_flag &value;~Unlock(){value.clear();}}unlock{active};
        @autoreleasepool{
            checkpoint(cancelled);auto started=Clock::now();
            require(input.ndim()==5 && input.shape(0)==1 && input.shape(1)==1 && input.shape(4)==256 && input.dtype()==mx::float16 &&
                previous.size()==2 && weight.shape()==mx::Shape{128,3,3,3,256} && weight.dtype()==mx::float16 &&
                bias.shape()==mx::Shape{128} && bias.dtype()==mx::float16,"causal runtime operands differ");
            for(const auto &past:previous)require(past.shape()==input.shape() && past.dtype()==mx::float16,"causal history geometry differs");
            const int h=input.shape(2),w=input.shape(3),th=stats->tile_rows,tw=stats->tile_cols;
            require(h>0 && w>0 && h<=1024 && w<=1024,"causal runtime canvas exceeds experiment scope");
            auto prepare=[](const Tensor &value){auto out=mx::contiguous(value);mx::eval(out);
                if(out.offset()!=0){out=mx::full(out.shape(),out,out.dtype());mx::eval(out);}return out;};
            std::array<Tensor,3> sources{prepare(previous[0]),prepare(previous[1]),prepare(input)};auto shift=prepare(bias);
            auto z=mx::full({1,1,h,w,128},std::numeric_limits<float>::quiet_NaN(),mx::float16);mx::eval(z);mx::synchronize();checkpoint(cancelled);
            const int columns=(w+tw-1)/tw,tiles=((h+th-1)/th)*columns;
            Geometry geometry{uint32_t(h),uint32_t(w),uint32_t(th),uint32_t(tw),uint32_t(th+2),uint32_t(tw+2),0,0,0,0};
            auto submit=[&](Slot *producer,int next,Slot *consumer,int last){return io.submit(sources[2],sources[0],sources[1],z,shift,geometry,columns,producer,next,consumer,last);};
            auto predict=[&](Slot &slot,int tile){checkpoint(cancelled);NSError *error=nil;
                auto result=[model predictionFromFeatures:slot.provider options:slot.options error:&error];
                if(!result)throw std::runtime_error(std::string("causal predict: ")+error.localizedDescription.UTF8String);
                ++stats->coreml_calls;auto out=[result featureValueForName:@"y"].multiArrayValue;
                check(out && out.dataType==MLMultiArrayDataTypeFloat16 && out.pixelBuffer==slot.output.pixel &&
                    [out.shape isEqual:slot.output.array.shape] && [out.strides isEqual:slot.output.array.strides],"causal output backing/shape/stride not honored");
                ++stats->backing_hits;slot.last_tile=tile;checkpoint(cancelled);
            };
            auto first=submit(slots[0].get(),0,nullptr,-1);first->join();first.reset();witness_pack(*slots[0],0,geometry,columns,sources);
            for(int tile=0;tile<tiles;++tile){
                checkpoint(cancelled);auto future=submit(tile+1<tiles?slots[(tile+1)%2].get():nullptr,tile+1,tile>0?slots[(tile-1)%2].get():nullptr,tile-1);
                predict(*slots[tile%2],tile);future->join();future.reset();
                if(tile+1<tiles)witness_pack(*slots[(tile+1)%2],tile+1,geometry,columns,sources);
                if(tile>0)witness_consume(*slots[(tile-1)%2],tile-1,geometry,columns,z,shift);
            }
            auto final=submit(nullptr,-1,slots[(tiles-1)%2].get(),tiles-1);final->join();final.reset();
            witness_consume(*slots[(tiles-1)%2],tiles-1,geometry,columns,z,shift);
            checkpoint(cancelled);++stats->frames;stats->boundary_seconds+=elapsed(started);return z;
        }
    }
};
}
ExperimentalCausalConv make_experimental_ane_causal_conv(const std::filesystem::path &manifest,bool cpu,
    std::atomic<bool> &cancelled,std::shared_ptr<ExperimentalCausalConvStats> &stats,bool verify_io){
    stats=std::make_shared<ExperimentalCausalConvStats>();auto runtime=std::make_shared<CausalRuntime>(manifest,cpu,cancelled,stats,verify_io);
    return [runtime](const Tensor &input,const FrameHistory &past,const Tensor &weight,const Tensor &bias){return runtime->run(input,past,weight,bias);};
}
}

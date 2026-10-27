// Complete original-parameter spatial projection + separate bias + shuffle.
// Offline public Core ML/Metal scheduling only, never a production VAE backend.
#include "seedvr2_vae_upscale_io.hpp"
#include <iostream>
#include <thread>
#include <atomic>
using namespace tc::seedvr2_upscale_probe;
namespace {
struct Params { uint32_t h,w,th,tw,y,x,cs,rs,ws,shuffled; float sign; };
const char *shader=R"METAL(
#include <metal_stdlib>
using namespace metal;
struct P { uint h,w,th,tw,y,x,cs,rs,ws,shuffled; float sign; };
kernel void pack(device const half *src [[buffer(0)]],device half *dst [[buffer(1)]],
                 constant P &p [[buffer(2)]],uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half tile[16][17];
    uint lc=tid%16,lp=tid/16,c=group.x*16+lc,pos=group.y*16+lp;
    half value=half(0);
    if(c<256 && pos<p.th*p.tw)value=half(float(src[((p.y+pos/p.tw)*p.w+p.x+pos%p.tw)*256+c])*p.sign);
    tile[lp][lc]=value;threadgroup_barrier(mem_flags::mem_threadgroup);
    c=group.x*16+lp;pos=group.y*16+lc;
    if(c<256 && pos<p.th*p.tw)dst[c*p.cs+(pos/p.tw)*p.rs+(pos%p.tw)*p.ws]=tile[lc][lp];
}
kernel void consume(device const half *src [[buffer(0)]],device half *dst [[buffer(1)]],
                    constant P &p [[buffer(2)]],device const half *bias [[buffer(3)]],
                    uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half tile[16][17];
    uint lp=tid%16,lc=tid/16,c=group.x*16+lc,pos=group.y*16+lp;
    half value=half(0);
    if(c<256 && pos<4*p.th*p.tw) {
        uint row=pos/(2*p.tw),col=pos%(2*p.tw),phase=(row%2)*2+col%2;
        uint sc=p.shuffled?c:phase*256+c,sr=p.shuffled?row:row/2,sx=p.shuffled?col:col/2;
        // The completed projection is already FP16; add bias with the same
        // separate FP16 output rounding as native matmul + bias.
        value=half(float(src[sc*p.cs+sr*p.rs+sx*p.ws])+float(bias[phase*256+c]));
    }
    tile[lc][lp]=value;threadgroup_barrier(mem_flags::mem_threadgroup);
    c=group.x*16+lp;pos=group.y*16+lc;
    if(c<256 && pos<4*p.th*p.tw)dst[((2*p.y+pos/(2*p.tw))*(2*p.w)+2*p.x+pos%(2*p.tw))*256+c]=tile[lp][lc];
}
)METAL";
struct IO {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue=nil;
    id<MTLComputePipelineState> pack=nil,consume=nil;
    IO() {
        check(device!=nil,"Metal device unavailable");queue=[device newCommandQueue];check(queue!=nil,"Metal queue missing");
        auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
        options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
        NSError *error=nil;auto library=[device newLibraryWithSource:@(shader) options:options error:&error];
        if(!library)throw std::runtime_error(std::string("shader: ")+error.localizedDescription.UTF8String);
        pack=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"pack"] error:&error];
        consume=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"consume"] error:&error];
        check(pack && consume && pack.maxTotalThreadsPerThreadgroup>=256 && consume.maxTotalThreadsPerThreadgroup>=256,
              "transpose pipelines unavailable");
    }
    std::unique_ptr<Submission> submit(const mx::array &x,const mx::array &z,const mx::array &bias,
        Params geometry,int columns,Slot *producer,int next,Slot *consumer,int previous,
        const mx::array *gpu_projection=nullptr,int gpu_tile=-1) {
        auto submission=std::make_unique<Submission>();submission->owners={x,z,bias};
        submission->command=[queue commandBuffer];check(submission->command!=nil,"command allocation failed");
        auto encode=[&](id<MTLComputePipelineState> kernel,id<MTLBuffer> source,id<MTLBuffer> target,Params p,bool packing) {
            auto encoder=[submission->command computeCommandEncoder];check(encoder!=nil,"encoder allocation failed");
            [encoder setComputePipelineState:kernel];[encoder setBuffer:source offset:0 atIndex:0];
            [encoder setBuffer:target offset:0 atIndex:1];[encoder setBytes:&p length:sizeof(p) atIndex:2];
            if(!packing)[encoder setBuffer:buffer(bias,device) offset:0 atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(16,((packing?1:4)*p.th*p.tw+15)/16,1)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];
        };
        if(producer) {
            auto p=geometry;p.y=(next/columns)*p.th;p.x=(next%columns)*p.tw;
            p.cs=producer->input.array.strides[1].unsignedIntValue;
            p.rs=producer->input.array.strides[2].unsignedIntValue;p.ws=1;
            [submission->surfaces addObject:producer->input.array];
            encode(pack,buffer(x,device),producer->input.metal,p,true);
        }
        if(consumer) {
            check(consumer->last_tile==previous,"surface slot overwritten before its consumer");
            auto p=geometry;p.y=(previous/columns)*p.th;p.x=(previous%columns)*p.tw;
            p.cs=consumer->output.array.strides[1].unsignedIntValue;
            p.rs=consumer->output.array.strides[2].unsignedIntValue;p.ws=1;
            [submission->surfaces addObject:consumer->output.array];
            encode(consume,consumer->output.metal,buffer(z,device),p,false);
        }
        if(gpu_projection) {
            submission->owners.push_back(*gpu_projection);
            auto p=geometry;p.y=(gpu_tile/columns)*p.th;p.x=(gpu_tile%columns)*p.tw;
            p.shuffled=0;p.cs=1;p.rs=p.tw*1024;p.ws=1024;
            encode(consume,buffer(*gpu_projection,device),buffer(z,device),p,false);
        }
        [submission->command commit];return submission;
    }
};
NSDictionary *plan(NSURL *url,MLModelConfiguration *config) {
    __block MLComputePlan *value=nil;auto ready=dispatch_semaphore_create(0);
    [MLComputePlan loadContentsOfURL:url configuration:config completionHandler:^(MLComputePlan *result,NSError *) {
        value=result;dispatch_semaphore_signal(ready);
    }];
    if(dispatch_semaphore_wait(ready,dispatch_time(DISPATCH_TIME_NOW,15*NSEC_PER_SEC)) || !value.modelStructure.program)
        return @{@"available":@NO,@"scope":@"anticipated only; not physical occupancy"};
    MLComputePlan *completed=value;NSMutableArray *operations=[NSMutableArray array];
    std::function<void(MLModelStructureProgramBlock *,int)> walk=[&](MLModelStructureProgramBlock *block,int depth) {
        check(depth<16 && operations.count<10000,"compute plan exceeds bounds");
        for(MLModelStructureProgramOperation *op in block.operations) {
            auto device=[completed computeDeviceUsageForMLProgramOperation:op].preferredComputeDevice;
            NSString *kind=[device isKindOfClass:MLNeuralEngineComputeDevice.class]?@"neural_engine":
                [device isKindOfClass:MLCPUComputeDevice.class]?@"cpu":@"other_or_unknown";
            [operations addObject:@{@"operator":op.operatorName,@"preferred":kind}];
            for(MLModelStructureProgramBlock *nested in op.blocks)walk(nested,depth+1);
        }
    };
    for(MLModelStructureProgramFunction *fn in completed.modelStructure.program.functions.allValues)walk(fn.block,0);
    return @{@"available":@YES,@"operations":operations,@"scope":@"anticipated; not actual placement/occupancy"};
}
NSDictionary *quality(const mx::array &value,NSData *reference,const mx::array &bias,int h,int w,float sign) {
    check(value.flags().row_contiguous && value.size()==size_t(h)*w*1024,"output extent/layout changed");
    const auto *actual=value.data<mx::float16_t>();const auto *expected=static_cast<const float *>(reference.bytes);
    const auto *shift=bias.data<mx::float16_t>();double nx=0,ny=0,dot=0,error=0,maximum=0;
    for(int row=0;row<h;++row)for(int col=0;col<w;++col)for(int phase=0;phase<4;++phase)for(int c=0;c<256;++c) {
        const int oc=phase*256+c;
        const double a=float(actual[((size_t(row)*2+phase/2)*(w*2)+col*2+phase%2)*256+c]);
        const double b=float(sign*expected[(size_t(row)*w+col)*1024+oc]+float(shift[oc]));
        check(std::isfinite(a) && std::isfinite(b),"nonfinite projection/bias/shuffle output");
        nx+=b*b;ny+=a*a;dot+=a*b;error+=(a-b)*(a-b);maximum=std::max(maximum,std::abs(a-b));
    }
    const double l2=std::sqrt(error/std::max(nx,1e-20)),cosine=dot/std::max(std::sqrt(nx*ny),1e-20);
    check(l2<=.05 && cosine>=.999,"unchanged complete projection FP32 numerical gate failed");
    return @{@"passed":@YES,@"finite":@YES,@"relative_l2":@(l2),@"cosine":@(cosine),@"max_abs":@(maximum)};
}
}
int main(int argc,char **argv) {
    @autoreleasepool {
        try {
            check(argc==3,"usage: upscale-probe FIXTURE.json gpu_matmul|gpu_fused|gpu_conv2d|gpu_tiled|ne_serial|ne_pipeline|cpu_serial");
            const std::string route=argv[2];const bool gpu=route=="gpu_matmul" || route=="gpu_conv2d";
            const bool fused=route=="gpu_fused",tiled=route=="gpu_tiled",pipeline=route=="ne_pipeline";
            check(gpu || fused || tiled || pipeline || route=="ne_serial" || route=="cpu_serial","unknown route");
            NSError *error=nil;auto data=[NSData dataWithContentsOfFile:@(argv[1])];check(data && data.length<65536,"fixture unavailable/big");
            NSDictionary *spec=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            check([spec isKindOfClass:NSDictionary.class] && [spec[@"schema"] isEqual:@"seedvr2-upscale-boundary-fixture-v1"],"fixture schema differs");
            const int h=[spec[@"rows"] intValue],w=[spec[@"cols"] intValue],th=[spec[@"tile_rows"] intValue],
                tw=[spec[@"tile_cols"] intValue],iterations=[spec[@"iterations"] intValue];
            const bool shuffled=[spec[@"variant"] isEqual:@"shuffle"];
            check(h>=32 && h<=512 && w==h && (th==32 || th==64 || th==128) && tw==th &&
                h%th==0 && w%tw==0 && iterations>=3 && iterations<=30,"fixture geometry/repeats out of bounds");
            auto read=[&](NSString *key,size_t bytes){auto content=[NSData dataWithContentsOfFile:spec[key] options:NSDataReadingMappedIfSafe error:&error];
                check(content && content.length==bytes,"operand extent differs");return content;};
            NSData *input=read(@"input",size_t(h)*w*256*2),*weight=read(@"weight",1024*256*2),
                *bias_data=read(@"bias",1024*2),*reference=read(@"reference",size_t(h)*w*1024*4);
            mx::set_default_device(mx::Device::gpu);mx::set_cache_limit(1ull<<30);IO io;
            mx::array x(static_cast<const mx::float16_t *>(input.bytes),{1,h,w,256},mx::float16),
                wf(static_cast<const mx::float16_t *>(weight.bytes),{1024,256},mx::float16),
                bias(static_cast<const mx::float16_t *>(bias_data.bytes),{1024},mx::float16);
            auto wn=mx::negative(wf);mx::eval({x,wf,wn,bias});
            mx::array z(0);if(!gpu){z=mx::zeros({1,h*2,w*2,256},mx::float16);mx::eval(z);}
            std::vector<std::unique_ptr<Slot>> slots;MLModel *model=nil;NSDictionary *compute_plan=@{@"available":@NO};double load=0;
            if(!gpu && !tiled && !fused) {
                auto config=[MLModelConfiguration new];config.computeUnits=route=="cpu_serial"?MLComputeUnitsCPUOnly:MLComputeUnitsCPUAndNeuralEngine;
                auto url=[NSURL fileURLWithPath:spec[@"compiled"]];auto begin=Clock::now();
                model=[MLModel modelWithContentsOfURL:url configuration:config error:&error];
                if(!model)throw std::runtime_error(std::string("Core ML load: ")+error.localizedDescription.UTF8String);
                load=elapsed(begin);compute_plan=plan(url,config);
                for(int i=0;i<(pipeline?2:1);++i)slots.push_back(std::make_unique<Slot>(io.device,256,shuffled?256:1024,th,tw,shuffled?th*2:th,shuffled?tw*2:tw));
                auto in=model.modelDescription.inputDescriptionsByName[@"x"].multiArrayConstraint;
                auto out=model.modelDescription.outputDescriptionsByName[@"y"].multiArrayConstraint;
                check(in.dataType==MLMultiArrayDataTypeFloat16 && out.dataType==MLMultiArrayDataTypeFloat16 &&
                    [in.shape isEqual:slots[0]->input.array.shape] && [out.shape isEqual:slots[0]->output.array.shape],"model descriptor differs from exact backing");
            }
            const int columns=w/tw,tiles=(h/th)*columns;
            Params geometry{uint32_t(h),uint32_t(w),uint32_t(th),uint32_t(tw),0,0,0,0,1,uint32_t(shuffled),1.f};
            uint64_t checked_pack=0,checked_consume=0,checked_default=0,calls=0,hits=0;
            if(!slots.empty()) {
                auto &slot=*slots[0];const auto *original=static_cast<const _Float16 *>(input.bytes);
                for(int tile=0;tile<tiles;++tile) {
                    auto pending=io.submit(x,z,bias,geometry,columns,&slot,tile,nullptr,-1);pending->join();
                    const auto *packed=static_cast<const _Float16 *>(slot.input.metal.contents);
                    const auto cs=slot.input.array.strides[1].unsignedLongLongValue,rs=slot.input.array.strides[2].unsignedLongLongValue;
                    const int oy=(tile/columns)*th,ox=(tile%columns)*tw;
                    for(int c=0;c<256;++c)for(int row=0;row<th;++row)for(int col=0;col<tw;++col) {
                        const auto expected=original[((size_t(oy+row)*w+ox+col)*256+c)];
                        check(std::memcmp(&expected,packed+size_t(c)*cs+size_t(row)*rs+col,2)==0,"actual GPU input pack differs bitwise");++checked_pack;
                    }
                    auto *out=static_cast<_Float16 *>(slot.output.metal.contents);
                    const auto ocs=slot.output.array.strides[1].unsignedLongLongValue,ors=slot.output.array.strides[2].unsignedLongLongValue;
                    const int channels=shuffled?256:1024,rows=shuffled?th*2:th,cols=shuffled?tw*2:tw;
                    for(int c=0;c<channels;++c)for(int row=0;row<rows;++row)for(int col=0;col<cols;++col)
                        out[size_t(c)*ocs+size_t(row)*ors+col]=_Float16(float((tile*17+c*13+row*7+col*11)%1024-512)/1024.f);
                    slot.last_tile=tile;pending=io.submit(x,z,bias,geometry,columns,nullptr,-1,&slot,tile);pending->join();
                    const auto *assembled=z.data<mx::float16_t>(),*shift=bias.data<mx::float16_t>();
                    for(int row=0;row<th*2;++row)for(int col=0;col<tw*2;++col)for(int c=0;c<256;++c) {
                        const int oc=((row%2)*2+col%2)*256+c,sc=shuffled?c:oc,sr=shuffled?row:row/2,sx=shuffled?col:col/2;
                        const _Float16 value=_Float16(float((tile*17+sc*13+sr*7+sx*11)%1024-512)/1024.f);
                        const _Float16 expected=_Float16(float(value)+float(shift[oc]));
                        const auto *actual=assembled+((size_t(oy*2+row)*(w*2)+ox*2+col)*256+c);
                        check(std::memcmp(&expected,actual,2)==0,"actual GPU bias/shuffle assembly differs bitwise");++checked_consume;
                    }
                }
            }
            if(fused) {
                for(float sign:{1.f,-1.f,1.f}) {
                    @autoreleasepool {
                    auto projected=mx::matmul(mx::reshape(x,{h*w,256}),mx::transpose(sign>0?wf:wn));mx::eval(projected);
                    auto full=geometry;full.th=h;full.tw=w;
                    auto pending=io.submit(x,z,bias,full,1,nullptr,-1,nullptr,-1,&projected,0);pending->join();
                    auto native_projection=mx::matmul(mx::reshape(x,{h*w,256}),mx::transpose(sign>0?wf:wn));
                    auto expected=mx::contiguous(mx::reshape(mx::transpose(mx::reshape(native_projection+bias,{1,h,w,2,2,256}),
                        {0,1,3,2,4,5}),{1,h*2,w*2,256}));mx::eval(expected);
                    check(std::memcmp(z.data<mx::float16_t>(),expected.data<mx::float16_t>(),z.nbytes())==0,
                        "fused GPU bias/shuffle changed native FP16 output words");checked_default+=z.size();
                    }
                }
            }
            mx::clear_cache();mx::reset_peak_memory();NSMutableArray *observations=[NSMutableArray array];std::vector<double> times;
            uint64_t peak=footprint();double maximum_l2=0,minimum_cosine=1;
            for(int iteration=0;iteration<iterations+3;++iteration) {
                @autoreleasepool {
                    const float sign=iteration%3==1?-1.f:1.f;geometry.sign=sign;
                    if(!gpu)std::fill(z.data<mx::float16_t>(),z.data<mx::float16_t>()+z.size(),mx::float16_t(std::numeric_limits<float>::quiet_NaN()));
                    double prediction_seconds=0,gpu_seconds=0,wait_seconds=0;
                    auto join=[&](std::unique_ptr<Submission> &pending){auto at=Clock::now();gpu_seconds+=pending->join();wait_seconds+=elapsed(at);pending.reset();};
                    auto predict=[&](Slot &slot,int tile){auto at=Clock::now();
                        auto result=[model predictionFromFeatures:slot.provider options:slot.options error:&error];prediction_seconds+=elapsed(at);
                        if(!result)throw std::runtime_error(std::string("Core ML predict: ")+error.localizedDescription.UTF8String);
                        auto out=[result featureValueForName:@"y"].multiArrayValue;++calls;
                        check(out && out.dataType==MLMultiArrayDataTypeFloat16 && out.pixelBuffer==slot.output.pixel &&
                            [out.shape isEqual:slot.output.array.shape] && [out.strides isEqual:slot.output.array.strides],"output backing/shape/stride not honored");
                        ++hits;slot.last_tile=tile;};
                    auto begin=Clock::now();mx::array result(0);
                    if(gpu) {
                        // Match the production expression's ownership: extra
                        // named projection/biased views distort the active peak.
                        result=mx::reshape((route=="gpu_matmul"?
                            mx::matmul(mx::reshape(x,{h*w,256}),mx::transpose(sign>0?wf:wn)):
                            mx::reshape(mx::conv2d(x,mx::reshape(sign>0?wf:wn,{1024,1,1,256})),{h*w,1024}))+bias,{1,h,w,2,2,256});
                        result=mx::contiguous(mx::reshape(mx::transpose(result,{0,1,3,2,4,5}),{1,h*2,w*2,256}));mx::eval(result);
                    } else if(fused) {
                        auto projected=mx::matmul(mx::reshape(x,{h*w,256}),mx::transpose(sign>0?wf:wn));mx::eval(projected);
                        auto full=geometry;full.th=h;full.tw=w;
                        auto pending=io.submit(x,z,bias,full,1,nullptr,-1,nullptr,-1,&projected,0);join(pending);result=z;
                    } else if(tiled) {
                        for(int tile=0;tile<tiles;++tile) {
                            const int y=(tile/columns)*th,x0=(tile%columns)*tw;
                            auto patch=mx::contiguous(mx::slice(x,{0,y,x0,0},{1,y+th,x0+tw,256}));
                            auto projected=mx::matmul(mx::reshape(patch,{th*tw,256}),mx::transpose(sign>0?wf:wn));mx::eval(projected);
                            auto pending=io.submit(x,z,bias,geometry,columns,nullptr,-1,nullptr,-1,&projected,tile);join(pending);
                        }
                        result=z;
                    } else {
                        auto first=io.submit(x,z,bias,geometry,columns,slots[0].get(),0,nullptr,-1);join(first);
                        if(pipeline) {
                            for(int tile=0;tile<tiles;++tile) {
                                auto future=io.submit(x,z,bias,geometry,columns,tile+1<tiles?slots[(tile+1)%2].get():nullptr,tile+1,
                                    tile>0?slots[(tile-1)%2].get():nullptr,tile-1);
                                predict(*slots[tile%2],tile);join(future);
                            }
                        } else {
                            predict(*slots[0],0);
                            for(int tile=1;tile<tiles;++tile) {
                                auto step=io.submit(x,z,bias,geometry,columns,slots[0].get(),tile,slots[0].get(),tile-1);join(step);predict(*slots[0],tile);
                            }
                        }
                        auto final=io.submit(x,z,bias,geometry,columns,nullptr,-1,slots[pipeline?(tiles-1)%2:0].get(),tiles-1);join(final);result=z;
                    }
                    const double total=elapsed(begin);auto checked=quality(result,reference,bias,h,w,sign);
                    maximum_l2=std::max(maximum_l2,[checked[@"relative_l2"] doubleValue]);minimum_cosine=std::min(minimum_cosine,[checked[@"cosine"] doubleValue]);
                    peak=std::max(peak,footprint());[observations addObject:@{@"warm":@(iteration>=3),@"sign":@(sign),@"total_seconds":@(total),
                        @"coreml_sum_seconds":@(prediction_seconds),@"metal_gpu_command_seconds":@(gpu_seconds),@"wait_seconds":@(wait_seconds),@"quality":checked}];
                    if(iteration>=3)times.push_back(total);
                }
            }
            std::sort(times.begin(),times.end());uint64_t bytes=0;for(const auto &slot:slots)bytes+=slot->input.bytes()+slot->output.bytes();
            NSDictionary *report=@{@"schema":@"seedvr2-upscale-boundary-observation-v1",@"route":@(route.c_str()),@"variant":spec[@"variant"],
                @"rows":@(h),@"cols":@(w),@"tile_rows":@(th),@"tile_cols":@(tw),@"tiles":@(tiles),@"load_seconds":@(load),
                @"warm_median_seconds":@((times[(times.size()-1)/2]+times[times.size()/2])/2),@"observations":observations,
                @"coreml_calls":@(calls),@"output_backing_identity_hits":@(hits),@"slot_bytes":@(bytes),
                @"bitwise_pack_checked_values":@(checked_pack),@"bitwise_consumer_checked_values":@(checked_consume),@"compute_plan":compute_plan,
                @"bitwise_default_checked_values":@(checked_default),
                @"mlx_active_peak_bytes":@(mx::get_peak_memory()),@"sampled_process_footprint_max_bytes":@(peak),
                @"quality":@{@"passed":@YES,@"finite":@YES,@"relative_l2":@(maximum_l2),@"cosine":@(minimum_cosine)},
                @"signed_input_aba_checked":@YES,@"qualified":@NO,@"physical_ane_occupancy_measured":@NO,
                @"scope":@"complete prepared spatial projection+separate FP16 bias+shuffle; pack/sync/assembly included, host reference/witness/poison excluded; not VAE/App or hard process peak"};
            auto json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingSortedKeys error:&error];
            check(json!=nil,"report serialization failed");std::cout.write(static_cast<const char *>(json.bytes),json.length);std::cout<<'\n';
        } catch(const std::exception &failure) { std::cerr<<failure.what()<<'\n'; return 1; }
    }
}

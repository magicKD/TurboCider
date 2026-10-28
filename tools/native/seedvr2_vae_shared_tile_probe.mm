// Offline original-weight convolution boundary, not a production VAE backend.
// Public IOSurface IO + Metal pack/consume + optional disjoint GPU suffix.
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <mlx/mlx.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mx=mlx::core;
namespace {
using Clock=std::chrono::steady_clock;
void check(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
uint64_t footprint(){task_vm_info_data_t value{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    check(task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&value),&count)==KERN_SUCCESS,"footprint unavailable");
    return value.phys_footprint;}
id<MTLBuffer> buffer(const mx::array &value,id<MTLDevice> device) {
    check(value.is_available() && value.flags().row_contiguous && value.offset()==0,"unmaterialized/noncontiguous MLX operand");
    auto result=(__bridge id<MTLBuffer>)const_cast<void *>(value.buffer().ptr());
    check(result && result.device==device && result.length>=value.nbytes(),"MLX Metal extent/device mismatch");
    return result;
}
NSArray *shape(int channels,int rows,int cols){return @[@1,@(channels),@(rows),@(cols)];}
struct Surface {
    CVPixelBufferRef pixel=nullptr;
    MLMultiArray *array=nil;
    id<MTLBuffer> metal=nil;
    int channels,rows,cols;
    Surface(id<MTLDevice> device,int c,int h,int w):channels(c),rows(h),cols(w) {
        NSDictionary *attributes=@{(id)kCVPixelBufferIOSurfacePropertiesKey:@{}};
        check(CVPixelBufferCreate(kCFAllocatorDefault,w,c*h,kCVPixelFormatType_OneComponent16Half,
            (__bridge CFDictionaryRef)attributes,&pixel)==kCVReturnSuccess && pixel,"surface allocation failed");
        try {
            array=[[MLMultiArray alloc] initWithPixelBuffer:pixel shape:shape(c,h,w)];
            auto surface=CVPixelBufferGetIOSurface(pixel);
            check(array && surface && array.dataType==MLMultiArrayDataTypeFloat16,"surface binding failed");
            auto address=IOSurfaceGetBaseAddress(surface);const size_t bytes=IOSurfaceGetAllocSize(surface),page=getpagesize();
            check(address && bytes && uintptr_t(address)%page==0 && bytes<=SIZE_MAX-(page-1),"surface VM address/size invalid");
            const size_t mapped_bytes=(bytes+page-1)/page*page;
            // CoreVideo can report logical bytes + metadata rather than a page
            // multiple. Metal requires whole pages. Verify the owning mapped
            // VM region before wrapping the tail; kernels never touch beyond
            // validated logical strides. Never guess a readable tail page.
            mach_vm_address_t region=reinterpret_cast<mach_vm_address_t>(address);mach_vm_size_t extent=0;
            vm_region_basic_info_data_64_t information{};mach_msg_type_number_t count=VM_REGION_BASIC_INFO_COUNT_64;
            mach_port_t object=MACH_PORT_NULL;
            const auto status=mach_vm_region(mach_task_self(),&region,&extent,VM_REGION_BASIC_INFO_64,
                reinterpret_cast<vm_region_info_t>(&information),&count,&object);
            if(object!=MACH_PORT_NULL)mach_port_deallocate(mach_task_self(),object);
            const auto begin=reinterpret_cast<mach_vm_address_t>(address);
            check(status==KERN_SUCCESS && region<=begin && begin-region<=extent && mapped_bytes<=extent-(begin-region) &&
                (information.protection&(VM_PROT_READ|VM_PROT_WRITE))==(VM_PROT_READ|VM_PROT_WRITE),
                "surface rounded VM mapping is not fully readable/writable");
            metal=[device newBufferWithBytesNoCopy:address length:mapped_bytes options:MTLResourceStorageModeShared deallocator:nil];
            check(metal && metal.contents==address,"public no-copy Metal surface binding failed");
            check(array.strides.count==4 && array.strides[3].unsignedLongLongValue==1 &&
                array.strides[2].unsignedLongLongValue==CVPixelBufferGetBytesPerRow(pixel)/2 &&
                array.strides[1].unsignedLongLongValue==size_t(h)*CVPixelBufferGetBytesPerRow(pixel)/2,
                "surface strides differ from public pixel pitch");
        }catch(...){array=nil;metal=nil;CVPixelBufferRelease(pixel);pixel=nullptr;throw;}
    }
    ~Surface(){metal=nil;array=nil;if(pixel)CVPixelBufferRelease(pixel);}
    Surface(const Surface &)=delete;
    size_t bytes()const{return metal.length;}
};
struct Slot {
    Surface input,output;
    MLDictionaryFeatureProvider *provider=nil;
    MLPredictionOptions *options=nil;
    int last_tile=-1;
    Slot(id<MTLDevice> device,int cin,int cout,int hi,int wi,int ho,int wo)
        :input(device,cin,hi,wi),output(device,cout,ho,wo) {
        NSError *error=nil;
        provider=[[MLDictionaryFeatureProvider alloc] initWithDictionary:
            @{@"x":[MLFeatureValue featureValueWithMultiArray:input.array]} error:&error];
        check(provider!=nil,"provider allocation failed");
        options=[MLPredictionOptions new];options.outputBackings=@{@"y":output.array};
    }
};
// A submission retains both MLX allocations and the MLMultiArrays that own the
// external IOSurfaces. Exceptional unwinding drains before releasing any owner.
struct Submission {
    id<MTLCommandBuffer> command=nil;
    std::vector<mx::array> owners;
    NSMutableArray *surfaces=[NSMutableArray array];
    bool joined=false;
    ~Submission(){if(command && !joined)[command waitUntilCompleted];}
    double join(){[command waitUntilCompleted];joined=true;
        check(command.status==MTLCommandBufferStatusCompleted,"Metal pack/consume failed");
        return std::max(0.,command.GPUEndTime-command.GPUStartTime);}
};
struct Params {
    uint32_t cin,taps,full_hi,full_wi,full_ho,full_wo,tile_hi,tile_wi,tile_ho,tile_wo;
    uint32_t y,x,channels,total_channels,cs,rs,ws;
    float sign;
};
const char *shader=R"METAL(
#include <metal_stdlib>
using namespace metal;
struct P { uint cin,taps,full_hi,full_wi,full_ho,full_wo,tile_hi,tile_wi,tile_ho,tile_wo;
           uint y,x,channels,total_channels,cs,rs,ws; float sign; };
kernel void pack_tile(device const half *source [[buffer(0)]],device half *target [[buffer(1)]],
                      constant P &p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.cin*p.taps*p.tile_hi*p.tile_wi)return;
    uint col=i%p.tile_wi,row=(i/p.tile_wi)%p.tile_hi,channel=i/(p.tile_wi*p.tile_hi);
    uint tap=channel/p.cin,c=channel%p.cin;
    uint at=((tap*p.full_hi+p.y+row)*p.full_wi+p.x+col)*p.cin+c;
    target[channel*p.cs+row*p.rs+col*p.ws]=half(float(source[at])*p.sign);
}
kernel void consume_tile(device const half *source [[buffer(0)]],device half *target [[buffer(1)]],
                         constant P &p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.channels*p.tile_ho*p.tile_wo)return;
    uint c=i%p.channels,col=(i/p.channels)%p.tile_wo,row=i/(p.channels*p.tile_wo);
    target[((p.y+row)*p.full_wo+p.x+col)*p.total_channels+c]=source[c*p.cs+row*p.rs+col*p.ws];
}
kernel void pack_tiled(device const half *source [[buffer(0)]],device half *target [[buffer(1)]],
                       constant P &p [[buffer(2)]],uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half tile[16][17];
    uint local_c=tid%16,local_pos=tid/16,c=group.x*16+local_c,pos=group.y*16+local_pos;
    half value=half(0);
    if(c<p.cin && pos<p.tile_hi*p.tile_wi){
        uint row=pos/p.tile_wi,col=pos%p.tile_wi;
        value=half(float(source[((group.z*p.full_hi+p.y+row)*p.full_wi+p.x+col)*p.cin+c])*p.sign);
    }
    tile[local_pos][local_c]=value;threadgroup_barrier(mem_flags::mem_threadgroup);
    c=group.x*16+local_pos;pos=group.y*16+local_c;
    if(c<p.cin && pos<p.tile_hi*p.tile_wi)
        target[(group.z*p.cin+c)*p.cs+(pos/p.tile_wi)*p.rs+(pos%p.tile_wi)*p.ws]=tile[local_c][local_pos];
}
kernel void consume_tiled(device const half *source [[buffer(0)]],device half *target [[buffer(1)]],
                          constant P &p [[buffer(2)]],uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half tile[16][17];
    uint local_pos=tid%16,local_c=tid/16,c=group.x*16+local_c,pos=group.y*16+local_pos;
    half value=half(0);
    if(c<p.channels && pos<p.tile_ho*p.tile_wo)value=source[c*p.cs+(pos/p.tile_wo)*p.rs+(pos%p.tile_wo)*p.ws];
    tile[local_c][local_pos]=value;threadgroup_barrier(mem_flags::mem_threadgroup);
    c=group.x*16+local_pos;pos=group.y*16+local_c;
    if(c<p.channels && pos<p.tile_ho*p.tile_wo)
        target[((p.y+pos/p.tile_wo)*p.full_wo+p.x+pos%p.tile_wo)*p.total_channels+c]=tile[local_pos][local_c];
}
kernel void merge_suffix(device const half *source [[buffer(0)]],device half *target [[buffer(1)]],
                         constant P &p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint suffix=p.total_channels-p.channels;
    if(i>=p.full_ho*p.full_wo*suffix)return;
    target[(i/suffix)*p.total_channels+p.channels+i%suffix]=source[i];
}
)METAL";
struct MetalIO {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue=nil;
    id<MTLComputePipelineState> pack=nil,consume=nil,merge=nil;
    bool tiled;
    explicit MetalIO(bool transpose):tiled(transpose){
        check(device!=nil,"Metal device unavailable");queue=[device newCommandQueue];check(queue!=nil,"Metal queue unavailable");
        auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
        options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
        NSError *error=nil;auto library=[device newLibraryWithSource:@(shader) options:options error:&error];
        if(!library)throw std::runtime_error(std::string("tile shader compilation failed: ")+error.localizedDescription.UTF8String);
        auto make=[&](NSString *name){auto value=[device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
            check(value!=nil,"tile pipeline unavailable");return value;};
        pack=make(tiled?@"pack_tiled":@"pack_tile");consume=make(tiled?@"consume_tiled":@"consume_tile");merge=make(@"merge_suffix");
        check(pack.maxTotalThreadsPerThreadgroup>=256 && consume.maxTotalThreadsPerThreadgroup>=256 &&
              merge.maxTotalThreadsPerThreadgroup>=256,"required tile threadgroup capacity unavailable");
    }
    std::unique_ptr<Submission> submit(const mx::array &source,const mx::array &target,Params geometry,
                    Slot *producer,int producer_tile,Slot *consumer,int consumer_tile,int columns,
                    const mx::array *suffix=nullptr) {
        auto pending=std::make_unique<Submission>();pending->owners={source,target};
        pending->command=[queue commandBuffer];check(pending->command!=nil,"command allocation failed");
        auto encode=[&](id<MTLComputePipelineState> state,id<MTLBuffer> a,id<MTLBuffer> b,Params p,size_t threads){
            auto encoder=[pending->command computeCommandEncoder];check(encoder!=nil,"encoder allocation failed");
            [encoder setComputePipelineState:state];[encoder setBuffer:a offset:0 atIndex:0];[encoder setBuffer:b offset:0 atIndex:1];
            [encoder setBytes:&p length:sizeof(p) atIndex:2];
            if(tiled && state==pack)[encoder dispatchThreadgroups:MTLSizeMake((p.cin+15)/16,(p.tile_hi*p.tile_wi+15)/16,p.taps)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            else if(tiled && state==consume)[encoder dispatchThreadgroups:MTLSizeMake((p.channels+15)/16,(p.tile_ho*p.tile_wo+15)/16,1)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            else [encoder dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];};
        if(producer){
            auto p=geometry;p.y=(producer_tile/columns)*p.tile_ho;p.x=(producer_tile%columns)*p.tile_wo;
            p.cs=producer->input.array.strides[1].unsignedIntValue;p.rs=producer->input.array.strides[2].unsignedIntValue;
            p.ws=producer->input.array.strides[3].unsignedIntValue;
            [pending->surfaces addObject:producer->input.array];
            encode(pack,buffer(source,device),producer->input.metal,p,size_t(p.cin)*p.taps*p.tile_hi*p.tile_wi);
        }
        if(consumer){
            check(consumer->last_tile==consumer_tile,"consumer slot was overwritten or never predicted");
            auto p=geometry;p.y=(consumer_tile/columns)*p.tile_ho;p.x=(consumer_tile%columns)*p.tile_wo;
            p.cs=consumer->output.array.strides[1].unsignedIntValue;p.rs=consumer->output.array.strides[2].unsignedIntValue;
            p.ws=consumer->output.array.strides[3].unsignedIntValue;
            [pending->surfaces addObject:consumer->output.array];
            encode(consume,consumer->output.metal,buffer(target,device),p,size_t(p.channels)*p.tile_ho*p.tile_wo);
        }
        if(suffix){pending->owners.push_back(*suffix);
            encode(merge,buffer(*suffix,device),buffer(target,device),geometry,
                   size_t(geometry.full_ho)*geometry.full_wo*(geometry.total_channels-geometry.channels));}
        [pending->command commit];return pending;
    }
};
NSDictionary *quality(const mx::array &value,NSData *reference,int h,int w,int channels,float sign) {
    check(value.flags().row_contiguous && value.size()==size_t(h)*w*channels,"assembled output ABI differs");
    const auto *a=value.data<mx::float16_t>();const auto *b=static_cast<const float *>(reference.bytes);
    double nx=0,ny=0,dot=0,error=0,maximum=0;
    for(int r=0;r<h;++r)for(int col=0;col<w;++col)for(int c=0;c<channels;++c){
        const double x=float(a[(size_t(r)*w+col)*channels+c]),y=sign*b[(size_t(c)*h+r)*w+col];
        check(std::isfinite(x) && std::isfinite(y),"nonfinite assembled/reference value");
        nx+=y*y;ny+=x*x;dot+=x*y;error+=(x-y)*(x-y);maximum=std::max(maximum,std::abs(x-y));}
    const double l2=std::sqrt(error/std::max(nx,1e-20)),cosine=dot/std::max(std::sqrt(nx*ny),1e-20);
    check(l2<=.05 && cosine>=.999,"unchanged full-spatial-convolution FP32 quality gate failed");
    return @{@"passed":@YES,@"finite":@YES,@"relative_l2":@(l2),@"cosine":@(cosine),@"max_abs":@(maximum)};
}
}

int main(int argc,char **argv) {
    if(argc!=3 && argc!=4){std::cerr<<"usage: shared-tile-probe FIXTURE.json ROUTE [scalar|transpose]\n";return 2;}
    @autoreleasepool {
        try {
            const std::string route=argv[2];
            const std::string kernel=argc==4?argv[3]:"scalar";
            check(kernel=="scalar" || kernel=="transpose","unknown pack kernel");
            const bool gpu=route=="gpu3d" || route=="gpu2d",hybrid=route=="hybrid3d" || route=="hybrid2d",
                pipeline=route=="ne_pipeline" || hybrid;
            check(gpu || hybrid || route=="cpu_serial" || route=="ne_serial" || pipeline,"unknown route");
            NSError *error=nil;NSData *data=[NSData dataWithContentsOfFile:@(argv[1])];
            check(data && data.length<65536,"fixture missing/unbounded");
            NSDictionary *spec=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            check([spec isKindOfClass:NSDictionary.class] && [spec[@"schema"] isEqual:@"seedvr2-vae-shared-tile-fixture-v1"],"fixture schema mismatch");
            const int cin=[spec[@"cin"] intValue],taps=[spec[@"taps"] intValue],ks=[spec[@"ks"] intValue],
                cout=[spec[@"cout"] intValue],core_cout=[spec[@"core_cout"] intValue],
                h=[spec[@"rows"] intValue],w=[spec[@"cols"] intValue],th=[spec[@"tile_rows"] intValue],tw=[spec[@"tile_cols"] intValue],
                iterations=[spec[@"iterations"] intValue];
            check(cin>0 && cin<=512 && cout>0 && cout<=4096 && core_cout>0 && core_cout<=cout && (taps==1 || taps==3) && (ks==1 || ks==3) &&
                h>0 && h<=1024 && w>0 && w<=1024 && th>=32 && th<=256 && tw>=32 && tw<=256 &&
                h%th==0 && w%tw==0 && iterations>=3 && iterations<=30,"invalid bounded tile geometry");
            check(gpu || (hybrid?core_cout<cout:core_cout==cout),"Core ML coverage does not match route");
            const int hi=h+ks-1,wi=w+ks-1,thi=th+ks-1,twi=tw+ks-1,columns=w/tw,tiles=(h/th)*columns;
            check(uint64_t(cin)*taps*hi*wi*2+uint64_t(cout)*h*w*6<=(2ull<<30),"full fixture exceeds 2-GiB bound");
            NSData *input=[NSData dataWithContentsOfFile:spec[@"input"]],*weight=[NSData dataWithContentsOfFile:spec[@"weight"]],
                *reference=[NSData dataWithContentsOfFile:spec[@"reference"]];
            check(input.length==uint64_t(cin)*taps*hi*wi*2 && weight.length==uint64_t(cout)*taps*ks*ks*cin*2 &&
                reference.length==uint64_t(cout)*h*w*4,"fixture bytes differ");
            mx::set_default_device(mx::Device::gpu);mx::set_cache_limit(512ull<<20);
            mx::array x(static_cast<const mx::float16_t *>(input.bytes),{1,taps,hi,wi,cin},mx::float16),
                wf(static_cast<const mx::float16_t *>(weight.bytes),{cout,taps,ks,ks,cin},mx::float16);
            // GPU-only returns its convolution allocation directly; do not
            // charge it for an unused Core ML assembly workspace as well.
            auto wneg=-wf;auto z=gpu?mx::array(0):mx::zeros({1,h,w,cout},mx::float16);mx::eval({x,wf,wneg,z});
            auto full_pack=[&](){return mx::reshape(mx::contiguous(mx::transpose(x,{0,2,3,1,4})),{1,hi,wi,taps*cin});};
            auto filter2d=[&](const mx::array &value){return mx::reshape(mx::contiguous(mx::transpose(value,{0,2,3,1,4})),{value.shape(0),ks,ks,taps*cin});};
            auto wp=filter2d(wf),wn=filter2d(wneg);mx::eval({wp,wn});
            mx::array suffix_w(0),suffix_n(0),suffix_p2d(0),suffix_n2d(0);
            if(hybrid){suffix_w=mx::slice(wf,{core_cout,0,0,0,0},{cout,taps,ks,ks,cin});suffix_n=-suffix_w;
                suffix_p2d=filter2d(suffix_w);suffix_n2d=filter2d(suffix_n);mx::eval({suffix_w,suffix_n,suffix_p2d,suffix_n2d});}
            MetalIO io(kernel=="transpose");MLModel *model=nil;NSMutableArray *backing_receipts=[NSMutableArray array];
            std::vector<std::unique_ptr<Slot>> slots;double load=0;uint64_t calls=0,hits=0;
            if(!gpu){auto config=[MLModelConfiguration new];config.computeUnits=route=="cpu_serial"?MLComputeUnitsCPUOnly:MLComputeUnitsCPUAndNeuralEngine;
                auto started=Clock::now();model=[MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:spec[@"compiled"]] configuration:config error:&error];load=elapsed(started);
                if(!model)throw std::runtime_error(std::string("Core ML load: ")+(error?error.localizedDescription.UTF8String:"unknown"));
                check(model.modelDescription.inputDescriptionsByName.count==1 && model.modelDescription.outputDescriptionsByName.count==1 &&
                    [model.modelDescription.inputDescriptionsByName[@"x"].multiArrayConstraint.shape isEqual:shape(cin*taps,thi,twi)] &&
                    [model.modelDescription.outputDescriptionsByName[@"y"].multiArrayConstraint.shape isEqual:shape(core_cout,th,tw)] &&
                    model.modelDescription.inputDescriptionsByName[@"x"].multiArrayConstraint.dataType==MLMultiArrayDataTypeFloat16 &&
                    model.modelDescription.outputDescriptionsByName[@"y"].multiArrayConstraint.dataType==MLMultiArrayDataTypeFloat16,"compiled fixed IO ABI differs");
                for(int index=0;index<(pipeline?2:1);++index)slots.push_back(std::make_unique<Slot>(io.device,cin*taps,core_cout,thi,twi,th,tw));
                for(size_t a=0;a<slots.size();++a)for(size_t b=0;b<a;++b)
                    check(slots[a]->input.metal.contents!=slots[b]->input.metal.contents &&
                          slots[a]->output.metal.contents!=slots[b]->output.metal.contents,"ring allocations alias");
            }
            Params geometry{uint32_t(cin),uint32_t(taps),uint32_t(hi),uint32_t(wi),uint32_t(h),uint32_t(w),uint32_t(thi),uint32_t(twi),
                uint32_t(th),uint32_t(tw),0,0,uint32_t(core_cout),uint32_t(cout),0,0,0,1.f};
            uint64_t checked_pack=0,checked_consume=0;
            if(!gpu){
                // Independent exact indexing witness, outside the timed loop:
                // compare GPU pack to original host NDHWC, then write a known
                // source pattern and verify every assembled prefix element.
                auto &slot=*slots[0];const auto *host=static_cast<const _Float16 *>(input.bytes);
                for(float sign:{1.f,-1.f,1.f}){
                    geometry.sign=sign;
                    for(int tile=0;tile<tiles;++tile){
                        auto packed=io.submit(x,z,geometry,&slot,tile,nullptr,-1,columns);packed->join();
                        auto *actual=static_cast<const _Float16 *>(slot.input.metal.contents);
                        const size_t cs=slot.input.array.strides[1].unsignedLongLongValue,rs=slot.input.array.strides[2].unsignedLongLongValue;
                        const int oy=(tile/columns)*th,ox=(tile%columns)*tw;
                        for(int c=0;c<cin*taps;++c)for(int row=0;row<thi;++row)for(int col=0;col<twi;++col){
                            const _Float16 expected=_Float16(sign*float(host[((size_t(c/cin)*hi+oy+row)*wi+ox+col)*cin+c%cin]));
                            check(std::memcmp(&expected,&actual[size_t(c)*cs+size_t(row)*rs+col],2)==0,"GPU temporal/halo pack bitwise witness differs");++checked_pack;}
                        auto *out=static_cast<_Float16 *>(slot.output.metal.contents);
                        const size_t ocs=slot.output.array.strides[1].unsignedLongLongValue,ors=slot.output.array.strides[2].unsignedLongLongValue;
                        for(int c=0;c<core_cout;++c)for(int row=0;row<th;++row)for(int col=0;col<tw;++col)
                            out[size_t(c)*ocs+size_t(row)*ors+col]=_Float16(sign*((tile*17+c*13+row*7+col*11)%1024-512)/1024.f);
                        slot.last_tile=tile;auto consumed=io.submit(x,z,geometry,nullptr,-1,&slot,tile,columns);consumed->join();
                    }
                    const auto *assembled=z.data<mx::float16_t>();
                    for(int row=0;row<h;++row)for(int col=0;col<w;++col)for(int c=0;c<core_cout;++c){
                        const int tile=(row/th)*columns+col/tw;
                        const _Float16 expected=_Float16(sign*((tile*17+c*13+(row%th)*7+(col%tw)*11)%1024-512)/1024.f);
                        check(std::memcmp(&expected,&assembled[(size_t(row)*w+col)*cout+c],2)==0,"GPU tile assembly bitwise witness differs");++checked_consume;}
                }
            }
            NSMutableArray *observations=[NSMutableArray array];std::vector<double> warm;uint64_t peak=footprint();
            double max_l2=0,min_cosine=1;
            for(int iteration=0;iteration<iterations+3;++iteration){
                @autoreleasepool {
                    const float sign=iteration%3==1?-1.f:1.f;geometry.sign=sign;
                    if(!gpu)std::fill(z.data<mx::float16_t>(),z.data<mx::float16_t>()+z.size(),
                        mx::float16_t(std::numeric_limits<float>::quiet_NaN()));
                    double core_seconds=0,gpu_seconds=0,producer_consumer_wait=0;
                    auto predict=[&](Slot &slot,int tile){auto started=Clock::now();
                        auto result=[model predictionFromFeatures:slot.provider options:slot.options error:&error];core_seconds+=elapsed(started);
                        if(!result)throw std::runtime_error(std::string("Core ML prediction: ")+(error?error.localizedDescription.UTF8String:"unknown"));
                        auto output=[result featureValueForName:@"y"].multiArrayValue;++calls;
                        check(output && output.pixelBuffer==slot.output.pixel && [output.shape isEqual:slot.output.array.shape] &&
                              [output.strides isEqual:slot.output.array.strides],"output surface backing not honored exactly");
                        ++hits;slot.last_tile=tile;};
                    auto join=[&](std::unique_ptr<Submission> &pending){auto started=Clock::now();gpu_seconds+=pending->join();producer_consumer_wait+=elapsed(started);pending.reset();};
                    auto started=Clock::now();mx::array result(0),suffix(0);
                    if(gpu){result=route=="gpu3d"?mx::conv3d(x,sign>0?wf:wneg):mx::conv2d(full_pack(),sign>0?wp:wn);mx::eval(result);}
                    else {
                        auto initial=io.submit(x,z,geometry,slots[0].get(),0,nullptr,-1,columns);join(initial);
                        if(hybrid){suffix=route=="hybrid3d"?mx::conv3d(x,sign>0?suffix_w:suffix_n):
                            mx::conv2d(full_pack(),sign>0?suffix_p2d:suffix_n2d);mx::async_eval(suffix);}
                        if(pipeline){
                            for(int tile=0;tile<tiles;++tile){
                                auto future=io.submit(x,z,geometry,tile+1<tiles?slots[(tile+1)%2].get():nullptr,tile+1,
                                    tile>0?slots[(tile-1)%2].get():nullptr,tile-1,columns);
                                predict(*slots[tile%2],tile);join(future);
                            }
                        }else {
                            predict(*slots[0],0);
                            for(int tile=1;tile<tiles;++tile){auto consume_previous=io.submit(x,z,geometry,nullptr,-1,slots[0].get(),tile-1,columns);join(consume_previous);
                                auto pack_next=io.submit(x,z,geometry,slots[0].get(),tile,nullptr,-1,columns);join(pack_next);predict(*slots[0],tile);}
                        }
                        if(hybrid)mx::eval(suffix);
                        auto final=io.submit(x,z,geometry,nullptr,-1,slots[pipeline?(tiles-1)%2:0].get(),tiles-1,columns,hybrid?&suffix:nullptr);join(final);
                        result=z;
                    }
                    const double total=elapsed(started);
                    auto checked=quality(result,reference,h,w,cout,sign);max_l2=std::max(max_l2,[checked[@"relative_l2"] doubleValue]);
                    min_cosine=std::min(min_cosine,[checked[@"cosine"] doubleValue]);peak=std::max(peak,footprint());
                    [observations addObject:@{@"warm":@(iteration>=3),@"sign":@(sign),@"total_seconds":@(total),
                        @"coreml_sum_seconds":@(core_seconds),@"metal_gpu_command_seconds":@(gpu_seconds),
                        @"producer_consumer_wait_seconds":@(producer_consumer_wait),@"quality":checked}];
                    if(iteration>=3)warm.push_back(total);
                }
            }
            std::sort(warm.begin(),warm.end());uint64_t slot_bytes=0;
            for(const auto &slot:slots){slot_bytes+=slot->input.bytes()+slot->output.bytes();[backing_receipts addObject:@{
                @"input_bytes":@(slot->input.bytes()),@"output_bytes":@(slot->output.bytes()),@"input_strides":slot->input.array.strides,
                @"output_strides":slot->output.array.strides}];}
            NSDictionary *report=@{@"schema":@"seedvr2-vae-shared-tile-observation-v1",@"route":@(route.c_str()),@"rows":@(h),@"cols":@(w),
                @"pack_kernel":@(kernel.c_str()),
                @"tile_rows":@(th),@"tile_cols":@(tw),@"tiles":@(tiles),@"cout":@(cout),@"core_cout":@(core_cout),
                @"warm_median_seconds":@((warm[(warm.size()-1)/2]+warm[warm.size()/2])/2),@"observations":observations,
                @"load_seconds":@(load),@"coreml_calls":@(calls),@"output_backing_identity_hits":@(hits),@"slots":@(slots.size()),
                @"slot_bytes":@(slot_bytes),@"slot_receipts":backing_receipts,@"sampled_process_footprint_max_bytes":@(peak),
                @"mlx_active_peak_bytes":@(mx::get_peak_memory()),@"quality":@{@"passed":@YES,@"finite":@YES,@"relative_l2":@(max_l2),@"cosine":@(min_cosine)},
                @"bitwise_pack_checked_values":@(checked_pack),@"bitwise_consumer_checked_values":@(checked_consume),
                @"poison_output_before_each_frame":@(!gpu),
                @"signed_input_aba_checked":@YES,@"qualified":@NO,@"physical_ane_occupancy_measured":@NO,
                @"scope":@"complete synthetic spatial convolution; GPU pack/producer completion, public Core ML IO, GPU consumer layout and optional disjoint suffix/join included; no VAE norm/bias/history/request qualification"};
            NSData *encoded=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingSortedKeys error:&error];check(encoded!=nil,"serialization failed");
            std::cout<<[[[NSString alloc] initWithData:encoded encoding:NSUTF8StringEncoding] UTF8String]<<'\n';
        }catch(const std::exception &error){try{mx::synchronize();}catch(...){}
            std::cerr<<error.what()<<'\n';return 1;}
    }
}

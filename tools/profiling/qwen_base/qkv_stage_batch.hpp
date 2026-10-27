#pragma once
// Narrow research W producer: original BF16 Q/K/V, same H128/RNE math.
// Two bounded compact scale entries, ONE codes/scales bank. No W copies.
#include "../../../native/backends/ane_gpu.hpp"
#include "../../../native/backends/ane_transfer_kernels.hpp"
#include "../../../native/backends/ane_w8_kernels.hpp"
#include "../../../native/backends/ane_w8a8_math.hpp"
#include "../../../native/backends/mlx.hpp"
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstring>
#include <condition_variable>
#include <mutex>

namespace tc::qkv_stage_lab {
namespace ag=ane::gpu;
constexpr int H=4096,N=12288;
struct Params {
    uint32_t source_pitch,physical_cols,encoding,dtype,group_size;
    uint32_t meta_pitch,meta_dtype,offset_pitch,offset_dtype,has_offset;
    uint32_t row_begin,rows,column_begin,columns,block,code_pitch,scale_pitch,transpose;
    uint32_t source_aligned,seed_low,seed_high,basis,activation_group,has_column_scale,inverse_column_scale;
    float norm;
};
inline constexpr const char* fused_source=R"metal(
kernel void tc_qkv_cached_codes(device const uchar*q[[buffer(0)]],device const uchar*k[[buffer(1)]],device const uchar*v[[buffer(2)]],
    device const ushort*sq[[buffer(3)]],device const ushort*sk[[buffer(4)]],device const ushort*sv[[buffer(5)]],
    device char*dst[[buffer(6)]],device ushort*scales[[buffer(7)]],device atomic_uint*status[[buffer(8)]],
    constant float*signs[[buffer(9)]],constant W8Params&p[[buffer(10)]],constant uint&first_part[[buffer(11)]],
    uint2 tile[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]]) {
    const uint global=tile.x+first_part*4096,part=global/4096,row=global%4096;
    device const uchar*src=part==0?q:part==1?k:v;
    device const ushort*cached=part==0?sq:part==1?sk:sv;
    float values[16];w8_sylvester_register_load(src,src,src,status,p,signs,row,tile.y*tc_w8_block,lane,values);
    const ushort bits=cached[row];const float scale=from_half(bits);
    if(!(scale>0)||!isfinite(scale))atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
    // Codes read the independent cached scalar, never a racing destination.
    if(tile.y==0&&lane==0)scales[ulong(global)*p.scale_pitch/2]=bits;
    for(uint j=0;j<tc_w8_block/32;++j){const uint col=tile.y*tc_w8_block+lane+j*32;
        float code=clamp((values[j]/scale)*128.f,-127.f,127.f);
        if(!isfinite(code)){atomic_fetch_or_explicit(status,8u,memory_order_relaxed);code=0;}
        dst[ulong(global)*p.code_pitch+col]=char(rint(code));}
}
)metal";

class Stager {
    struct Cached {std::array<Tensor,3> weights;std::array<id<MTLBuffer>,3> scales;};
    struct Pending {
        std::mutex mutex;std::condition_variable cv;bool done=false,ok=false;
        id<MTLBuffer> status;std::array<Tensor,3> weights;std::array<id<MTLBuffer>,3> cache;
        std::array<std::shared_ptr<void>,3> source_allocations;
        ag::Surface codes,scales;
        Pending(const Cached&c,ag::Surface w,ag::Surface s):weights(c.weights),cache(c.scales),codes(w),scales(s){
            for(int j=0;j<3;++j)source_allocations[j]=weights[j].data_shared_ptr();}
    };
    id<MTLDevice> metal_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;
    id<MTLComputePipelineState> codes_,copy_,fused_;
    id<MTLBuffer> signs_;
    std::vector<Cached> cache_;
    std::shared_ptr<Pending> pending_;
    bool failed_=false;
    Params params()const{return{H*2,H,0,1,32,0,0,0,0,0,0,H,0,H,128,uint32_t(codes.pitch()),uint32_t(scales.pitch()),0,
        1,20260930,0,0,0,0,0,1.f/std::sqrt(128.f)};}
    void validate(const std::array<Tensor,3>&w)const{
        require(!failed_&&!pending_,"batch W producer disabled or still in flight");
        for(const auto&a:w){require(a.shape()==mx::Shape{H,H}&&a.dtype()==mx::bfloat16&&a.is_available()&&a.flags().row_contiguous&&
                a.buffer().ptr()&&a.offset()%2==0&&a.buffer_size()>=size_t(a.offset())+size_t(H)*H*2,"ready original contiguous BF16 QKV required");
            id<MTLBuffer>b=(__bridge id<MTLBuffer>)a.buffer().ptr();
            require(b.device==metal_&&b.length>=a.buffer_size(),"QKV source device/extent differs");}
    }
  public:
    ag::Device device{true,true,0};
    ag::Surface codes{device,N,H,ag::Element::I8},scales{device,N,1,ag::Element::FP16};
    double setup_seconds=0,prime_seconds=0;
    uint64_t calls=0,native_calls=0,custom_calls=0,command_buffers=0,compute_dispatches=0,native_scale_hits=0,native_scale_misses=0;
    static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
    Stager(){const auto start=Clock::now();metal_=(__bridge id<MTLDevice>)device.native_device();queue_=[metal_ newCommandQueue];
        require(queue_!=nil,"batch W queue unavailable");NSError*error=nil;MTLCompileOptions*options=[MTLCompileOptions new];options.fastMathEnabled=NO;
        const auto source=std::string(ag::transfer_source)+ag::w8_source+fused_source;
        library_=[metal_ newLibraryWithSource:@(source.c_str()) options:options error:&error];
        require(library_!=nil,"batch W shader compile failed: "+std::string(error.localizedDescription.UTF8String?:"unknown"));
        MTLFunctionConstantValues*fc=[MTLFunctionConstantValues new];const bool specialized=true;const uint32_t encoding=0,dtype=1,block=128;
        [fc setConstantValue:&specialized type:MTLDataTypeBool atIndex:0];[fc setConstantValue:&encoding type:MTLDataTypeUInt atIndex:1];
        [fc setConstantValue:&dtype type:MTLDataTypeUInt atIndex:2];[fc setConstantValue:&block type:MTLDataTypeUInt atIndex:3];
        auto pipeline=[&](NSString*name,bool constants){id<MTLFunction>f=constants?[library_ newFunctionWithName:name constantValues:fc error:&error]:[library_ newFunctionWithName:name];
            require(f!=nil,"batch W function missing");id<MTLComputePipelineState>p=[metal_ newComputePipelineStateWithFunction:f error:&error];
            require(p!=nil&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=64,"batch W pipeline capacity unsupported");return p;};
        codes_=pipeline(@"tc_ane_sylvester_register_codes",true);copy_=pipeline(@"tc_ane_w8_scale_copy",false);fused_=pipeline(@"tc_qkv_cached_codes",true);
        signs_=[metal_ newBufferWithLength:512*sizeof(float) options:MTLResourceStorageModeShared];require(signs_!=nil,"batch W signs allocation failed");
        for(int i=0;i<512;++i)static_cast<float*>(signs_.contents)[i]=float(ane::rotation_sign(20260930,i));
        setup_seconds=seconds(start);
    }
    ~Stager(){if(pending_)try{finish();}catch(...) {}}
    void native(const std::array<Tensor,3>&w){validate(w);std::vector<ag::QuantStage>tickets;const auto before=device.scale_cache_stats();
        try{for(int j=0;j<3;++j){const auto&a=w[j];ane::DeviceWeightView v{const_cast<void*>(a.buffer().ptr()),a.buffer_size(),size_t(a.offset()),H*2,H,H,
                ane::DeviceWeightEncoding::Dense,ane::DType::BF16,32,{},{},std::make_shared<Tensor>(a),a.data_shared_ptr()};v.immutable_generation=true;
            tickets.push_back(device.stage_w8(v,{0,H,0,H,128,20260930,false},codes.slice_rows(j*H,H),scales.slice_rows(j*H,H)));}}
        catch(...){for(auto&t:tickets)try{(void)t.finish();}catch(...){}throw;}
        bool ok=true;for(auto&t:tickets){const auto c=t.finish();ok=ok&&c.ok&&!c.timed_out&&t.validation_flags()==0;}
        require(ok,"native W source validation failed");const auto after=device.scale_cache_stats();
        const auto misses=after.misses-before.misses,hits=after.hits-before.hits;require(misses+hits==3,"native compact scale accounting differs");
        native_scale_hits+=hits;native_scale_misses+=misses;
        ++calls;++native_calls;command_buffers+=3;compute_dispatches+=6+misses;
    }
    void prime(const std::array<Tensor,3>&w){validate(w);require(cache_.size()<2,"bounded two-source metadata fixture only");const auto start=Clock::now();native(w);
        Cached c{w,{nil,nil,nil}};for(int j=0;j<3;++j){c.scales[j]=[metal_ newBufferWithLength:H*2 options:MTLResourceStorageModeShared];require(c.scales[j]!=nil,"compact scale allocation failed");
            for(int r=0;r<H;++r)std::memcpy(static_cast<char*>(c.scales[j].contents)+r*2,static_cast<const char*>(scales.data())+size_t(j*H+r)*scales.pitch(),2);}
        cache_.push_back(std::move(c));prime_seconds+=seconds(start);
    }
    void submit(const std::array<Tensor,3>&w,int arm){validate(w);require(arm>=1&&arm<=3,"custom W arm must be1..3");
        const Cached*c=nullptr;for(const auto&v:cache_)if(v.weights[0].id()==w[0].id()&&v.weights[1].id()==w[1].id()&&v.weights[2].id()==w[2].id())c=&v;
        require(c,"unprimed QKV generation rejected before GPU submission");auto state=std::make_shared<Pending>(*c,codes,scales);
        state->status=[metal_ newBufferWithLength:4 options:MTLResourceStorageModeShared];require(state->status!=nil,"batch W status allocation failed");*static_cast<uint32_t*>(state->status.contents)=0;
        const auto wl=codes.metal_buffer_lease(),sl=scales.metal_buffer_lease();wl.validate((__bridge void*)metal_);sl.validate((__bridge void*)metal_);
        id<MTLBuffer>wb=(__bridge id<MTLBuffer>)wl.native_buffer,sb=(__bridge id<MTLBuffer>)sl.native_buffer;
        id<MTLCommandBuffer>cb=[queue_ commandBuffer];require(cb!=nil,"batch W command allocation failed");const auto p=params();
        if(arm==1)for(int j=0;j<3;++j){id<MTLComputeCommandEncoder>copy=[cb computeCommandEncoder];require(copy!=nil,"scale copy encoder failed");
            const uint32_t cp[4]{H,1,uint32_t(scales.pitch()/2),0};[copy setComputePipelineState:copy_];[copy setBuffer:c->scales[j] offset:0 atIndex:0];
            [copy setBuffer:sb offset:sl.offset_bytes+size_t(j*H)*scales.pitch() atIndex:1];[copy setBytes:cp length:sizeof(cp) atIndex:2];
            [copy dispatchThreads:MTLSizeMake(H,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[copy endEncoding];
            id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];require(e!=nil,"codes encoder failed");[e setComputePipelineState:codes_];
            [e setBuffer:(__bridge id<MTLBuffer>)w[j].buffer().ptr() offset:w[j].offset() atIndex:0];[e setBuffer:state->status offset:0 atIndex:1];[e setBuffer:state->status offset:0 atIndex:2];
            [e setBuffer:sb offset:sl.offset_bytes+size_t(j*H)*scales.pitch() atIndex:3];[e setBuffer:wb offset:wl.offset_bytes+size_t(j*H)*codes.pitch() atIndex:4];
            [e setBuffer:state->status offset:0 atIndex:5];[e setBytes:&p length:sizeof(p) atIndex:6];[e setBuffer:signs_ offset:0 atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake(H,H/128,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];}
        else for(uint32_t j=0;j<uint32_t(arm==2?3:1);++j){id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];require(e!=nil,"fused codes encoder failed");[e setComputePipelineState:fused_];
            for(int part=0;part<3;++part){[e setBuffer:(__bridge id<MTLBuffer>)w[part].buffer().ptr() offset:w[part].offset() atIndex:part];[e setBuffer:c->scales[part] offset:0 atIndex:3+part];}
            [e setBuffer:wb offset:wl.offset_bytes atIndex:6];[e setBuffer:sb offset:sl.offset_bytes atIndex:7];[e setBuffer:state->status offset:0 atIndex:8];
            [e setBuffer:signs_ offset:0 atIndex:9];[e setBytes:&p length:sizeof(p) atIndex:10];[e setBytes:&j length:sizeof(j) atIndex:11];
            [e dispatchThreadgroups:MTLSizeMake(arm==2?H:N,H/128,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];}
        [cb addCompletedHandler:^(id<MTLCommandBuffer>b){{std::lock_guard lock(state->mutex);state->ok=b.status==MTLCommandBufferStatusCompleted;state->done=true;}state->cv.notify_all();}];
        pending_=state;[cb commit];++calls;++custom_calls;++command_buffers;compute_dispatches+=arm==1?6:arm==2?3:1;
    }
    uint32_t finish(){require(bool(pending_),"batch W no pending command");auto state=pending_;std::unique_lock lock(state->mutex);
        if(!state->cv.wait_for(lock,std::chrono::seconds(30),[&]{return state->done;})){failed_=true;throw ane::CapabilityError("batch W timeout; submitted owners retained by completion");}
        const auto flags=*static_cast<const uint32_t*>(state->status.contents);const bool ok=state->ok;pending_.reset();
        if(!ok||flags){failed_=true;throw ane::CapabilityError("batch W validation/completion failed");}return flags;
    }
    uint64_t metadata_bytes()const{return cache_.size()*3*H*2;}
};
}

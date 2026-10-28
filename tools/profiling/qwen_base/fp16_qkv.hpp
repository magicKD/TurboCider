#pragma once
// Private fixed512 FP16 QKV research alternative. Dynamic original BF16 W,
// one current-layer FP16 surface, exact integer cast with no W8 rotation.
#include "qkv_ane.hpp"
#include "../../../native/backends/ane_transfer_kernels.hpp"
#import <Metal/Metal.h>
#include <condition_variable>
#include <mutex>

namespace tc::fp16_qkv_lab {
namespace pa=ane::private_api;
namespace ag=ane::gpu;
namespace qlab=qwen21::base_qkv_lab;
inline constexpr const char*cast_source=R"metal(
kernel void tc_qkv_fp16_weights(device const ushort*q[[buffer(0)]],device const ushort*k[[buffer(1)]],device const ushort*v[[buffer(2)]],
    device ushort*dst[[buffer(3)]],device atomic_uint*status[[buffer(4)]],constant uint&pitch[[buffer(5)]],uint2 pos[[thread_position_in_grid]]) {
    if(pos.x>=4096||pos.y>=12288)return;uint part=pos.y/4096,row=pos.y%4096;
    device const ushort*src=part==0?q:part==1?k:v;const float value=as_type<float>(uint(src[ulong(row)*4096+pos.x])<<16);
    ushort bits=to_half(value);
    if(!isfinite(value)||(bits&0x7c00)==0x7c00){atomic_fetch_or_explicit(status,1u,memory_order_relaxed);bits=0;}
    dst[ulong(pos.y)*pitch/2+pos.x]=bits;
}
)metal";

inline std::string single_program(){
    // Match the current W8 emitter's ONE whole MatMul, but FP16 ports and no
    // Q/DQ. Native fp16_program() is retained as a tiled lowering control.
    auto shape=[](int r,int c){return "[1, 1, "+std::to_string(r)+", "+std::to_string(c)+"]";};
    auto type=[&](int r,int c){return "tensor<fp16, "+shape(r,c)+">";};
    auto buffer=[&](int r,int c){const auto plane=std::to_string(uint64_t(r)*c);return "tensor_buffer<fp16, shape="+shape(r,c)+
        ", strides=["+plane+", "+plane+", "+std::to_string(c)+", 1], interleave_factors=[1, 1, 1, 1]>";};
    return "program(1.3)\n{\n    func main_ane<ios18>("+buffer(4096,512)+" x, "+buffer(12288,4096)+" w) {\n        "+
        type(4096,512)+" xt = tensor_buffer_to_tensor<ios17>(input = x);\n        "+type(12288,4096)+
        " wt = tensor_buffer_to_tensor<ios17>(input = w);\n        "+type(12288,512)+
        " yn = matmul(transpose_x = bool(false), transpose_y = bool(false), x = wt, y = xt);\n        "+buffer(12288,512)+
        " y = tensor_to_tensor_buffer<ios17>(input = yn, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([6291456, 6291456, 512, 1]));\n    } -> (y);\n}\n";
}

class Graph {
    struct Job {
        std::array<Tensor,3>weights;std::array<std::shared_ptr<void>,3>allocations;
        pa::Surface destination;id<MTLBuffer>status;std::mutex mutex;std::condition_variable cv;bool done=false,ok=false;
        Job(const std::array<Tensor,3>&w,pa::Surface out):weights(w),destination(out){for(int i=0;i<3;++i)allocations[i]=weights[i].data_shared_ptr();}
    };
    id<MTLDevice>metal_;
    id<MTLCommandQueue>queue_;
    id<MTLComputePipelineState>cast_;
    std::shared_ptr<Job>pending_;
    bool disabled_=false;
    uint64_t timeline_=0;
    void join_weights(){require(!disabled_&&bool(pending_),"FP16 QKV missing or disabled W producer");auto j=pending_;std::unique_lock lock(j->mutex);
        if(!j->cv.wait_for(lock,std::chrono::seconds(30),[&]{return j->done;})){disabled_=true;throw ane::CapabilityError("FP16 W producer timeout; completion retains owners");}
        const bool valid=j->ok&&!*static_cast<const uint32_t*>(j->status.contents);pending_.reset();
        if(!valid){disabled_=true;throw ane::CapabilityError("FP16 W conversion nonfinite/overflow or GPU failure");}}
  public:
    pa::Device device{false,true,0};
    pa::Surface x{device,4096,512,pa::Element::FP16},w{device,12288,4096,pa::Element::FP16},y{device,12288,512,pa::Element::FP16};
    std::unique_ptr<pa::Program>program,single;
    uint64_t stages=0,predictions=0,tiled_predictions=0,single_predictions=0;
    double setup_seconds=0;
    static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
    explicit Graph(bool single_only=false){const auto t=Clock::now();metal_=(__bridge id<MTLDevice>)device.native_device();queue_=[metal_ newCommandQueue];require(queue_!=nil,"FP16 W queue unavailable");
        MTLCompileOptions*o=[MTLCompileOptions new];o.fastMathEnabled=NO;NSError*error=nil;const auto shader=std::string(ag::transfer_source)+cast_source;
        auto library=[metal_ newLibraryWithSource:@(shader.c_str()) options:o error:&error];require(library!=nil,"FP16 W cast shader failed: "+std::string(error.localizedDescription.UTF8String?:"unknown"));
        cast_=[metal_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"tc_qkv_fp16_weights"] error:&error];require(cast_!=nil&&cast_.maxTotalThreadsPerThreadgroup>=256,"FP16 cast pipeline capacity unavailable");
        const auto mil=pa::fp16_program({ane::Kind::Matmul,512,4096,12288,4096,512,false});
        std::vector<std::string>names;
        if(!single_only){program=std::make_unique<pa::Program>(device,mil,std::span<const uint8_t>{},pa::default_cache_directory());names=program->inputs();std::sort(names.begin(),names.end());
            require(names==std::vector<std::string>{"w","x"}&&program->outputs()==std::vector<std::string>{"y"},"FP16 fixed QKV graph ABI mismatch");}
        single=std::make_unique<pa::Program>(device,single_program(),std::span<const uint8_t>{},pa::default_cache_directory());names=single->inputs();std::sort(names.begin(),names.end());
        require(names==std::vector<std::string>{"w","x"}&&single->outputs()==std::vector<std::string>{"y"},"single FP16 QKV graph ABI mismatch");setup_seconds=seconds(t);
    }
    ~Graph(){if(pending_)try{join_weights();}catch(...) {}}
    size_t slot_bytes()const{return x.bytes()+w.bytes()+y.bytes();}
    void stage(const std::array<Tensor,3>&weights){require(!disabled_&&!pending_,"FP16 QKV bank has an active producer");
        for(const auto&a:weights){require(a.is_available()&&a.shape()==mx::Shape{4096,4096}&&a.dtype()==mx::bfloat16&&a.flags().row_contiguous&&
                a.buffer().ptr()&&a.offset()%2==0&&a.buffer_size()>=size_t(a.offset())+size_t(4096)*4096*2,"ready original BF16 QKV source required");
            id<MTLBuffer>b=(__bridge id<MTLBuffer>)a.buffer().ptr();require(b.device==metal_&&b.length>=a.buffer_size(),"FP16 W source device/extent mismatch");}
        auto state=std::make_shared<Job>(weights,w);state->status=[metal_ newBufferWithLength:4 options:MTLResourceStorageModeShared];require(state->status!=nil,"FP16 W status allocation failed");*static_cast<uint32_t*>(state->status.contents)=0;
        const auto lease=w.metal_buffer_lease();lease.validate((__bridge void*)metal_);auto cb=[queue_ commandBuffer];require(cb!=nil,"FP16 W command unavailable");
        auto e=[cb computeCommandEncoder];require(e!=nil,"FP16 W encoder unavailable");[e setComputePipelineState:cast_];
        for(int i=0;i<3;++i)[e setBuffer:(__bridge id<MTLBuffer>)weights[i].buffer().ptr() offset:weights[i].offset() atIndex:i];
        [e setBuffer:(__bridge id<MTLBuffer>)lease.native_buffer offset:lease.offset_bytes atIndex:3];[e setBuffer:state->status offset:0 atIndex:4];
        const uint32_t pitch=uint32_t(w.pitch());[e setBytes:&pitch length:sizeof(pitch) atIndex:5];
        [e dispatchThreads:MTLSizeMake(4096,12288,1) threadsPerThreadgroup:MTLSizeMake(32,8,1)];[e endEncoding];
        [cb addCompletedHandler:^(id<MTLCommandBuffer>b){{std::lock_guard lock(state->mutex);state->ok=b.status==MTLCommandBufferStatusCompleted;state->done=true;}state->cv.notify_all();}];
        pending_=state;[cb commit];++stages;
    }
    template<class GPU>qlab::Projection run(const Tensor&input,GPU&&gpu,bool minimal=false,bool prepack=false){
        require(!disabled_&&input.shape()==mx::Shape{1,1024,4096},"FP16 QKV fixed input/disabled mismatch");qlab::Projection p{Tensor(0.f)};
        auto start=Clock::now();join_weights();p.stage_seconds=seconds(start);
        auto allocation=mx::allocator::malloc(size_t(512)*12288*2);require(allocation.ptr(),"FP16 QKV owned tail allocation failed");Tensor tail(allocation,{1,512,12288},mx::bfloat16);
        require(timeline_<=UINT64_MAX-2,"FP16 QKV timeline exhausted");const auto ready=++timeline_,done=++timeline_;
        // Transposed A16 upload and independent BF16 restore retain the same
        // leading-ready/ANE-done contract; no rotation or W8 scale epilogue.
        std::vector<ag::Upload>uploads{{qlab::device_view(input,512,4096,512),x,0,1.f}};
        if(prepack){const auto prepared_start=Clock::now();auto upload=device.prepare_upload(std::move(uploads));upload.submit();const auto c=upload.finish();
            require(c.ok&&!c.timed_out&&upload.validation_flags()==0,"FP16 prepacked input upload invalid/incomplete");p.activation_seconds=seconds(prepared_start);uploads.clear();}
        auto io=device.prepare_transfer(std::move(uploads),{{y,qlab::device_view(tail,512,12288),0,ane::DType::BF16,1.f}},ready,done);
        std::pair<std::string,pa::Surface>ins[]{{"x",x},{"w",w}},outs[]{{"y",y}};auto ticket=(minimal?single:program)->enqueue(ins,outs,ready,done,io.failure_callback());std::optional<Tensor>head;
        try{start=Clock::now();io.submit();head=gpu(slice_axis(input,1,0,512));require(!head->is_available(),"FP16 GPU head reused materialized output");mx::async_eval(*head);p.submit_seconds=seconds(start);}
        catch(...){try{(void)ticket.finish();(void)io.finish();}catch(...){}throw;}
        start=Clock::now();const auto a=ticket.finish(),b=io.finish();
        if(!(a.ok&&a.physical_complete&&!a.retired&&!a.timed_out&&b.ok&&!b.timed_out&&io.validation_flags()==0)){disabled_=true;throw ane::CapabilityError("FP16 QKV prediction/restore incomplete or nonfinite");}
        p.join_seconds=seconds(start);p.value=mx::concatenate({*head,tail},1);mx::eval(p.value);++predictions;
        if(minimal)++single_predictions;else ++tiled_predictions;return p;
    }
};
}

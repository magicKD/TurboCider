#pragma once
// Research-only owner for one static FP16 ANE prefix and complete runtime LoRA.
#include "qwen21_static_lora_mil.hpp"
#include "../../native/models/qwen21/runtime_ffn_graphs.hpp"
#include "../../native/backends/private/ane_program.hpp"
#include "../../native/backends/dense_gpu_lora_b.hpp"
#include "../../native/platform/apple/platform.hpp"
#include "../../native/runtime/memory_manifest.hpp"
#include <mlx/allocator.h>
#include <algorithm>
#import <Metal/Metal.h>

namespace tc::qwen21::static_lora_lab {
namespace pa=ane::private_api;
namespace ff=runtime_ffn;
enum class ReadyFence { None, QueueTail, EventValue };
enum class SubmitOrder { TransferFirst, GpuFirst };
inline ane::DeviceMatrixView prefix_device_view(const Tensor &a) {
    require(a.ndim()==3&&a.shape(0)==1&&a.flags().row_contiguous&&a.is_available()&&a.buffer().ptr()&&
        (a.dtype()==mx::bfloat16||a.dtype()==mx::float16),"ready owned device matrix required");
    return {const_cast<void*>(a.buffer().ptr()),a.buffer_size(),size_t(a.offset()),a.shape(1),a.shape(2),
        size_t(a.shape(2))*2,a.dtype()==mx::bfloat16?ane::DType::BF16:ane::DType::FP16,
        std::make_shared<Tensor>(a),a.data_shared_ptr()};
}
class PreparedPrefix {
    static constexpr int M=1024,H=4096,F=12288;
    int width_;
    bool shared_;
    pa::Device device_;
    std::unique_ptr<pa::Program> program_;
    std::vector<std::pair<std::string,pa::Surface>> inputs_,outputs_;
    uint64_t timeline_=0;
    ff::Function corrections_,suffix_,down_,ranks_,packed_suffix_;
    static std::string stem(int block){require(block>=0&&block<32,"bounded FFN layer required");return "transformer_blocks."+std::to_string(block)+".img_mlp.";}
    static double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
  public:
    uint64_t successful_calls=0;
    size_t constant_bytes=0;
    bool compiled=false;
    double setup_seconds=0;
    std::string constants_sha256,mil_sha256;
    double corrections_ready_seconds=0,submission_seconds=0,exposed_wait_seconds=0,post_join_seconds=0;
    double input_ready_wait_seconds=0;
    uint64_t input_ready_fences=0;
    uint64_t event_value_fences=0,ready_before_private_completion=0;
    uint64_t gpu_first_submissions=0;
    double gpu_submit_seconds=0,transfer_submit_seconds=0;
    uint64_t packed_suffix_calls=0;
    size_t packed_suffix_bytes=0;
    double packed_suffix_setup_seconds=0;
    bool last_private_complete_at_ready=false;
    PreparedPrefix(const Weights &w,int width,bool shared,const std::filesystem::path &cache,int block=0)
        :width_(width),shared_(shared),
         corrections_(shared?ff::corrections_shared_ranks(w,stem(block),0,width):ff::corrections(w,stem(block),0,width)),
         suffix_(ff::channels(w,stem(block),width,F-width,H,F,false,shared)),down_(ff::down_add(w,stem(block))),ranks_(ff::input_ranks(w,stem(block))) {
        require(width==1024||width==2048,"bounded prefix share required");const auto start=Clock::now();
        const auto &gu=w.at(stem(block)+"gate_up.weight"),&d=w.at(stem(block)+"out.weight");
        require(gu.shape()==mx::Shape{2*F,H}&&d.shape()==mx::Shape{H,F}&&gu.dtype()==mx::bfloat16&&d.dtype()==mx::bfloat16,
                "original BF16 base FFN geometry required");
        std::array<Tensor,3> values{mx::contiguous(mx::astype(slice_axis(gu,0,0,width),mx::float16)),
            mx::contiguous(mx::astype(slice_axis(gu,0,F,F+width),mx::float16)),
            mx::contiguous(mx::astype(slice_axis(d,1,0,width),mx::float16))};
        std::array<std::span<const uint16_t>,3> spans;
        for(size_t i=0;i<3;++i){mx::eval(values[i]);require(mx::all(mx::isfinite(values[i])).item<bool>(),"base FP16 constant overflow");
            spans[i]={reinterpret_cast<const uint16_t*>(values[i].data<mx::float16_t>()),values[i].size()};}
        auto frozen=freeze({ane::Kind::SwiGLU,M,H,width,4096,4096,true},spans);constant_bytes=frozen.constants.size();
        constants_sha256=memory_sha256_hex({reinterpret_cast<const char*>(frozen.constants.data()),frozen.constants.size()});
        mil_sha256=memory_sha256_hex(frozen.mil);program_=std::make_unique<pa::Program>(device_,frozen.mil,frozen.constants,cache);
        compiled=program_->compiled_now();auto names=program_->inputs();std::sort(names.begin(),names.end());
        require(names==std::vector<std::string>{"dg","du","x"}&&program_->outputs()==std::vector<std::string>{"y"},"static LoRA interface differs");
        for(const auto &[name,rows]:std::array<std::pair<const char*,int>,3>{{{"x",H},{"dg",width},{"du",width}}})
            inputs_.emplace_back(name,pa::Surface(device_,rows,M,pa::Element::FP16));
        outputs_.emplace_back("y",pa::Surface(device_,H+width,M,pa::Element::FP16));setup_seconds=elapsed(start);
    }
    void prepare_packed_suffix(const Weights &w,int block=0){
        require(!packed_suffix_&&!shared_&&w.runtime_lora_bf16_fp32_ranks()&&w.runtime_lora_b_epilogue(),"bounded independent packed suffix recipe required");
        const auto start=Clock::now();const auto p=stem(block);const auto &gu=w.at(p+"gate_up.weight");
        require(!w.has(p+"gate_up.bias")&&!w.quantized(p+"gate_up")&&!w.convrot(p+"gate_up")&&!w.nvfp4(p+"gate_up"),"unbiased original dense suffix required");
        const int count=F-width_;auto packed=mx::contiguous(mx::concatenate({slice_axis(gu,0,width_,F),slice_axis(gu,0,F+width_,2*F)},0));mx::eval(packed);
        const auto digest=[](const Tensor &a){auto b=mx::view(mx::contiguous(a),mx::uint8);mx::eval(b);
            return memory_sha256_hex({reinterpret_cast<const char*>(b.data<uint8_t>()),b.nbytes()});};
        for(int i=0;i<2;++i)require(digest(slice_axis(packed,0,i*count,(i+1)*count))==digest(slice_axis(gu,0,i*F+width_,(i+1)*F)),"packed suffix changed original BF16 base");
        const auto snapshot=w.runtime_lora_snapshot();require(!snapshot.fp16_rank_math,"F32 independent A ranks required");
        const auto binding=std::find_if(snapshot.bindings.begin(),snapshot.bindings.end(),[&](const auto &b){return b.prefix==p+"gate_up";});
        require(binding!=snapshot.bindings.end()&&binding->ordered_terms.size()==2,"two separate gate/up terms required");
        std::array<Tensor,2> up{Tensor(0.f),Tensor(0.f)};std::array<float,2> scales{1.f,1.f};
        for(int i=0;i<2;++i){const auto &term=binding->ordered_terms[i];
            require(term.output_start==i*F&&term.output_end==(i+1)*F&&term.down.shape()==mx::Shape{128,H}&&
                term.up.shape()==mx::Shape{F,128}&&term.down.dtype()==mx::bfloat16&&term.up.dtype()==mx::bfloat16,
                "original separate r128 gate/up factors required");up[i]=term.up;scales[i]=term.scale;}
        packed_suffix_bytes=packed.nbytes();
        packed_suffix_=mx::compile(ff::Function([&w,p,packed,up,scales,first=width_,count](const auto &a){
            auto base=mx::matmul(a[0],mx::transpose(packed));auto low=w.lora_input_ranks(a[0],p+"gate_up",0,H);
            require(low.size()==2,"separate online gate/up ranks required");
            auto gate=dense_gpu::lora_b_epilogue(low[0],up[0],first,F,scales[0],mx::bfloat16,base,0);
            auto update=dense_gpu::lora_b_epilogue(low[1],up[1],first,F,scales[1],mx::bfloat16,base,count);
            auto hidden=silu(gate)*update;auto partial=w.project_base_slice(hidden,p+"out",0,H,first,F,false);
            return std::vector<Tensor>{partial,hidden};
        }));packed_suffix_setup_seconds=elapsed(start);
    }
    std::vector<Tensor> suffix_control(const Tensor &input,bool packed)const{
        require(!shared_&&(!packed||bool(packed_suffix_)),"prepared independent suffix required");
        return (packed?packed_suffix_:suffix_)({input});
    }
    void poison(){std::memset(outputs_[0].second.data(),0xff,outputs_[0].second.bytes());}
    Tensor run(const Tensor &input,bool hybrid=true,ReadyFence fence=ReadyFence::None,SubmitOrder order=SubmitOrder::TransferFirst,bool packed_suffix=false){
        require(input.shape()==mx::Shape{1,M,H}&&input.dtype()==mx::bfloat16,"static prefix input geometry mismatch");
        require(order!=SubmitOrder::GpuFirst||(hybrid&&fence==ReadyFence::None),"GPU-first requires an independent unfenced complement");
        require(!packed_suffix||(hybrid&&bool(packed_suffix_)),"prepared packed GPU complement required");
        const auto correction_start=Clock::now();std::vector<Tensor> inputs{input};
        if(shared_){auto values=ranks_({input});inputs.insert(inputs.end(),values.begin(),values.end());}
        auto deltas=corrections_(inputs);mx::eval(deltas);corrections_ready_seconds+=elapsed(correction_start);
        const auto submit_start=Clock::now();auto memory=mx::allocator::malloc(size_t(M)*(H+width_)*2);require(memory.ptr(),"output allocation failed");
        Tensor packed(memory,{1,M,H+width_},mx::bfloat16);auto pv=prefix_device_view(packed);
        const uint64_t ready=timeline_+1,done=timeline_+2;timeline_=done;
        auto transfer=device_.prepare_transfer({{prefix_device_view(input),inputs_[0].second,0,1.f},
            {prefix_device_view(deltas[0]),inputs_[1].second,0,1.f},{prefix_device_view(deltas[1]),inputs_[2].second,0,1.f}},
            {{outputs_[0].second,pv,0,ane::DType::BF16,1.f,std::nullopt,std::nullopt,std::nullopt,ane::gpu::RowScalePolicy::Positive}},ready,done);
        auto ticket=program_->enqueue(inputs_,outputs_,ready,done,transfer.failure_callback());std::vector<Tensor> gpu;
        try{
            const auto submit_gpu=[&]{const auto start=Clock::now();gpu=(packed_suffix?packed_suffix_:suffix_)(inputs);
                require(std::all_of(gpu.begin(),gpu.end(),[](const auto &a){return !a.is_available();}),"completed cached suffix returned");
                mx::async_eval(gpu);gpu_submit_seconds+=elapsed(start);};
            if(order==SubmitOrder::GpuFirst){submit_gpu();++gpu_first_submissions;}
            const auto transfer_start=Clock::now();transfer.submit();transfer_submit_seconds+=elapsed(transfer_start);
            last_private_complete_at_ready=false;
            if(fence!=ReadyFence::None){const auto ready_start=Clock::now();
                if(fence==ReadyFence::EventValue){
                    // Device::wait submits on the transfer queue AFTER its
                    // ANE-dependent restore. It is a queue-tail fence, not an
                    // isolated ready wait. Wait directly on the borrowed real
                    // event; never advance its value or enqueue another CB.
                    id<MTLSharedEvent> event=(__bridge id<MTLSharedEvent>)device_.shared_event();
                    require([event waitUntilSignaledValue:ready timeoutMS:30000],"GPU input ready event deadline exceeded");
                    ++event_value_fences;
                }else require(device_.wait(ready,std::chrono::seconds(30)),"transfer queue-tail fence failed");
                input_ready_wait_seconds+=elapsed(ready_start);++input_ready_fences;
                last_private_complete_at_ready=ticket.status().physical_complete;
                if(!last_private_complete_at_ready)++ready_before_private_completion;
            }
            if(hybrid&&order==SubmitOrder::TransferFirst)submit_gpu();submission_seconds+=elapsed(submit_start);
            const auto wait_start=Clock::now();const auto completion=ticket.finish(),copied=transfer.finish();
            exposed_wait_seconds+=elapsed(wait_start);
            require(completion.ok&&completion.physical_complete&&!completion.retired&&copied.ok&&transfer.validation_flags()==0,
                    "static LoRA handoff failed; partial output rejected");
            if(!hybrid){gpu=suffix_(inputs);mx::eval(gpu);}
        }catch(...){try{(void)ticket.finish();}catch(...){}try{(void)transfer.finish();}catch(...){}mx::synchronize();throw;}
        const auto post_start=Clock::now();auto prefix=slice_axis(packed,-1,0,H),hidden=slice_axis(packed,-1,H,H+width_);
        auto base=mx::astype(mx::astype(prefix,mx::float32)+mx::astype(gpu[0],mx::float32),mx::bfloat16);
        auto whole_hidden=mx::concatenate({hidden,gpu[1]},-1);auto output=down_({whole_hidden,base})[0];mx::eval(output);
        post_join_seconds+=elapsed(post_start);require(output.shape()==input.shape()&&output.dtype()==input.dtype(),"incomplete static FFN output");
        ++successful_calls;if(packed_suffix)++packed_suffix_calls;return output;
    }
};
} // namespace tc::qwen21::static_lora_lab

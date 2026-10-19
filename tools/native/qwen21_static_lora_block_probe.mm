// Ordinary Qwen21 r128/joint block observer; static FP16 prefix + complete
// runtime LoRA. Bounded research only, never selected by a product request.
#include "qwen21_static_lora_mil.hpp"
#include "../../native/models/qwen21/pipeline.hpp"
#include "../../native/models/qwen21/runtime_ffn_graphs.hpp"
#include "../../native/models/qwen21/viggle_adapter.hpp"
#include "../../native/backends/private/ane_program.hpp"
#include "../../native/platform/apple/bridge.hpp"
#include "../../native/platform/apple/platform.hpp"
#include "turbocider/turbocider.h"
#include <mlx/allocator.h>
#include <array>
#include <bit>
#include <fstream>
#include <iostream>

using namespace tc;
namespace pa = ane::private_api;
namespace ff = qwen21::runtime_ffn;
namespace lab {
constexpr int M = 1024, H = 4096, F = 12288;
void emitter_contract() {
    int tested=0,rejected=0;
    for(int width:{1024,2048}) {
        std::array<std::vector<uint16_t>,3> matrices;
        for(size_t i=0;i<matrices.size();++i)matrices[i].assign(size_t(width)*H,uint16_t(0x3c00+i));
        std::array<std::span<const uint16_t>,3> spans{matrices[0],matrices[1],matrices[2]};
        const ane::GraphShape shape{ane::Kind::SwiGLU,M,H,width,4096,4096,true};
        const auto program=qwen21::static_lora_lab::freeze(shape,spans);
        auto read=[&](size_t offset,auto &value){std::memcpy(&value,program.constants.data()+offset,sizeof(value));};
        uint32_t count=0,version=0;read(0,count);read(4,version);
        require(count==3 && version==2,"constant file header contract mismatch");
        size_t offset=64;
        for(size_t i=0;i<3;++i) {
            uint32_t magic=0,type=0;uint64_t bytes=0,data=0;
            read(offset,magic);read(offset+4,type);read(offset+8,bytes);read(offset+16,data);
            require(offset%64==0 && magic==0xdeadbeef && type==1 && bytes==matrices[i].size()*2 &&
                data==offset+64 && std::memcmp(program.constants.data()+data,matrices[i].data(),bytes)==0,
                "constant metadata/source payload contract mismatch");
            offset=data+bytes;++tested;
        }
        require(offset==program.constants.size() && program.mil.find("input = wg)")==std::string::npos &&
            program.mil.find("input = wu)")==std::string::npos && program.mil.find("input = wd)")==std::string::npos &&
            program.mil.find("input = dg)")!=std::string::npos && program.mil.find("input = du)")!=std::string::npos &&
            program.mil.find("values = (dp_0_0, hidden)")!=std::string::npos &&
            program.mil.find("abs(x = gc)")!=std::string::npos,"frozen graph adapter/hidden ABI mismatch");
        for(int bad=0;bad<6;++bad) {
            auto invalid=shape;auto values=spans;
            if(bad==0)invalid.rows=512;if(bad==1)invalid.width=4096;if(bad==2)invalid.hidden=2048;
            if(bad==3)invalid.lora_inputs=false;if(bad==4)invalid.tile_k=2048;
            if(bad==5)values[2]=values[2].first(values[2].size()-1);
            bool denied=false;try{(void)qwen21::static_lora_lab::freeze(invalid,values);}catch(const std::exception&){denied=true;}
            require(denied,"unbounded/malformed static emitter input accepted");++rejected;
        }
    }
    std::cout<<"emitter_contract_passed="<<tested<<" rejected="<<rejected<<" device_calls=0\n";
}
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
NSString *ns(const std::string &s) { return [NSString stringWithUTF8String:s.c_str()]; }
std::string hash(const Tensor &a) {
    auto b = mx::view(mx::contiguous(a), mx::uint8); mx::eval(b);
    return memory_sha256_hex({reinterpret_cast<const char *>(b.data<uint8_t>()), b.nbytes()});
}
NSDictionary *difference(const Tensor &a, const Tensor &b) {
    require(a.shape() == b.shape(), "comparison shape mismatch");
    auto x = mx::contiguous(mx::astype(a, mx::float32)), y = mx::contiguous(mx::astype(b, mx::float32));
    mx::eval(x, y);
    double error = 0, energy = 0, maximum = 0, dot = 0, other = 0;
    size_t changed = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double u = x.data<float>()[i], v = y.data<float>()[i], d = u-v;
        require(std::isfinite(u) && std::isfinite(v), "nonfinite full consumer output");
        error += d*d; energy += u*u; other += v*v; dot += u*v;
        maximum = std::max(maximum, std::abs(d)); changed += u != v;
    }
    return @{@"relative_l2": @(std::sqrt(error/std::max(energy,1e-30))),
        @"cosine": @(dot/std::sqrt(std::max(energy*other,1e-60))),
        @"max_abs": @(maximum), @"changed": @(changed), @"elements": @(x.size())};
}
Tensor rotate(const Tensor &x, const Tensor &cosine, const Tensor &sine) {
    auto shape=x.shape(), paired_shape=shape; paired_shape.back()/=2; paired_shape.push_back(2);
    auto paired=mx::reshape(mx::astype(x,mx::float32),paired_shape);
    auto a=mx::squeeze(slice_axis(paired,-1,0,1),-1), b=mx::squeeze(slice_axis(paired,-1,1,2),-1);
    auto c=mx::reshape(cosine,{1,1,shape[2],shape[3]/2}), s=mx::reshape(sine,{1,1,shape[2],shape[3]/2});
    return mx::astype(mx::reshape(mx::stack({a*c-b*s,a*s+b*c},-1),shape),x.dtype());
}
ff::Function front(const Weights &w) {
    return mx::compile(ff::Function([&w](const std::vector<Tensor> &a) {
        const std::string p="transformer_blocks.0.";
        auto x=mx::fast::layer_norm(a[0],{},{},1e-6f)*(Tensor(1.f,a[0].dtype())+a[1]);
        auto q=linear(x,w,p+"attn.to_q"), k=linear(x,w,p+"attn.to_k");
        auto v=heads(linear(x,w,p+"attn.to_v"),32,128);
        q=rotate(mx::fast::rms_norm(heads(q,32,128),w.at(p+"attn.norm_q.weight"),1e-6f),a[5],a[6]);
        k=rotate(mx::fast::rms_norm(heads(k,32,128),w.at(p+"attn.norm_k.weight"),1e-6f),a[5],a[6]);
        auto attention=attend(q,mx::concatenate({a[7],k},2),mx::concatenate({a[8],v},2));
        auto hidden=a[0]+mx::tanh(a[2])*linear(attention,w,p+"attn.to_out.0");
        x=mx::fast::layer_norm(hidden,{},{},1e-6f)*(Tensor(1.f,hidden.dtype())+a[3]);
        return std::vector<Tensor>{hidden,x};
    }));
}
ane::DeviceMatrixView device_view(const Tensor &a) {
    require(a.ndim()==3 && a.shape(0)==1 && a.flags().row_contiguous && a.is_available() &&
        a.buffer().ptr() && (a.dtype()==mx::bfloat16 || a.dtype()==mx::float16), "ready owned device matrix required");
    return {const_cast<void *>(a.buffer().ptr()),a.buffer_size(),size_t(a.offset()),a.shape(1),a.shape(2),
        size_t(a.shape(2))*2,a.dtype()==mx::bfloat16?ane::DType::BF16:ane::DType::FP16,
        std::make_shared<Tensor>(a),a.data_shared_ptr()};
}
Tensor allocate_output(int cols) {
    auto b=mx::allocator::malloc(size_t(M)*cols*2); require(b.ptr(), "output allocation failed");
    return Tensor(b,{1,M,cols},mx::bfloat16);
}
class Prefix {
    int width_;
    bool shared_ranks_;
    pa::Device device_;
    std::unique_ptr<pa::Program> program_;
    std::vector<std::pair<std::string,pa::Surface>> inputs_,outputs_;
    uint64_t timeline_=0;
    ff::Function corrections_,suffix_,down_,ranks_;
public:
    uint64_t successful_calls=0;
    size_t constant_bytes=0;
    bool compiled=false;
    double setup_seconds=0;
    std::string constants_sha256,mil_sha256;
    double corrections_ready_seconds=0,submission_seconds=0,exposed_wait_seconds=0,post_join_seconds=0;
    Prefix(const Weights &w,int width,bool shared_ranks,const std::filesystem::path &cache)
        : width_(width),shared_ranks_(shared_ranks),
          corrections_(shared_ranks?ff::corrections_shared_ranks(w,"transformer_blocks.0.img_mlp.",0,width):
                                    ff::corrections(w,"transformer_blocks.0.img_mlp.",0,width)),
          suffix_(ff::channels(w,"transformer_blocks.0.img_mlp.",width,F-width,H,F,false,shared_ranks)),
          down_(ff::down_add(w,"transformer_blocks.0.img_mlp.")),
          ranks_(ff::input_ranks(w,"transformer_blocks.0.img_mlp.")) {
        const auto start=Clock::now();
        const auto &gu=w.at("transformer_blocks.0.img_mlp.gate_up.weight");
        const auto &d=w.at("transformer_blocks.0.img_mlp.out.weight");
        std::array<Tensor,3> values{
            mx::contiguous(mx::astype(slice_axis(gu,0,0,width),mx::float16)),
            mx::contiguous(mx::astype(slice_axis(gu,0,F,F+width),mx::float16)),
            mx::contiguous(mx::astype(slice_axis(d,1,0,width),mx::float16))};
        std::array<std::span<const uint16_t>,3> spans;
        for(size_t i=0;i<values.size();++i) {
            mx::eval(values[i]); require(mx::all(mx::isfinite(values[i])).item<bool>(),"base FP16 constant overflow");
            spans[i]={reinterpret_cast<const uint16_t *>(values[i].data<mx::float16_t>()),values[i].size()};
        }
        auto frozen=qwen21::static_lora_lab::freeze({ane::Kind::SwiGLU,M,H,width,4096,4096,true},spans);
        constant_bytes=frozen.constants.size();
        constants_sha256=memory_sha256_hex({reinterpret_cast<const char *>(frozen.constants.data()),frozen.constants.size()});
        mil_sha256=memory_sha256_hex(frozen.mil);
        program_=std::make_unique<pa::Program>(device_,frozen.mil,frozen.constants,cache);
        compiled=program_->compiled_now();
        auto names=program_->inputs(); std::sort(names.begin(),names.end());
        require(names==std::vector<std::string>{"dg","du","x"} && program_->outputs()==std::vector<std::string>{"y"},
                "static LoRA compiled interface differs");
        for(const auto &[name,rows]:std::array<std::pair<const char *,int>,3>{{{"x",H},{"dg",width},{"du",width}}})
            inputs_.emplace_back(name,pa::Surface(device_,rows,M,pa::Element::FP16));
        outputs_.emplace_back("y",pa::Surface(device_,H+width,M,pa::Element::FP16));
        setup_seconds=elapsed(start);
    }
    void poison() { std::memset(outputs_[0].second.data(),0xff,outputs_[0].second.bytes()); }
    Tensor run(const Tensor &input,bool hybrid) {
        require(input.shape()==mx::Shape{1,M,H} && input.dtype()==mx::bfloat16,"static prefix input geometry mismatch");
        const auto correction_start=Clock::now();
        std::vector<Tensor> inputs{input};
        if(shared_ranks_) {auto values=ranks_({input});inputs.insert(inputs.end(),values.begin(),values.end());}
        auto deltas=corrections_(inputs); mx::eval(deltas);
        corrections_ready_seconds+=elapsed(correction_start);
        const auto submit_start=Clock::now();
        auto packed=allocate_output(H+width_); auto pv=device_view(packed);
        const uint64_t ready=timeline_+1, done=timeline_+2; timeline_=done;
        auto transfer=device_.prepare_transfer({{device_view(input),inputs_[0].second,0,1.f},
            {device_view(deltas[0]),inputs_[1].second,0,1.f},{device_view(deltas[1]),inputs_[2].second,0,1.f}},
            {{outputs_[0].second,pv,0,ane::DType::BF16,1.f,std::nullopt,std::nullopt,std::nullopt,ane::gpu::RowScalePolicy::Positive}},ready,done);
        auto ticket=program_->enqueue(inputs_,outputs_,ready,done,transfer.failure_callback());
        std::vector<Tensor> gpu;
        try {
            transfer.submit();
            if(hybrid) { gpu=suffix_(inputs); mx::async_eval(gpu); }
            submission_seconds+=elapsed(submit_start);
            const auto wait_start=Clock::now();
            const auto completion=ticket.finish(), copied=transfer.finish();
            exposed_wait_seconds+=elapsed(wait_start);
            require(completion.ok && completion.physical_complete && !completion.retired && copied.ok &&
                    transfer.validation_flags()==0,"static LoRA prefix/handoff failed; no partial output published");
            if(!hybrid) { gpu=suffix_(inputs); mx::eval(gpu); }
        } catch (...) {
            try {(void)ticket.finish();}catch(...){}
            try {(void)transfer.finish();}catch(...){}
            mx::synchronize(); throw;
        }
        const auto post_start=Clock::now();
        auto prefix=slice_axis(packed,-1,0,H), hidden=slice_axis(packed,-1,H,H+width_);
        auto base=mx::astype(mx::astype(prefix,mx::float32)+mx::astype(gpu[0],mx::float32),mx::bfloat16);
        auto whole_hidden=mx::concatenate({hidden,gpu[1]},-1);
        auto output=down_({whole_hidden,base})[0]; mx::eval(output);
        post_join_seconds+=elapsed(post_start);
        require(output.shape()==input.shape() && output.dtype()==input.dtype(),"incomplete static FFN output");
        ++successful_calls;
        return output;
    }
};
void measure(const qwen21::DiagnosticBlockObservation &o,int width,bool shared_ranks,
             const std::filesystem::path &out,NSMutableDictionary *report) {
    require(o.layer==0 && o.steady && o.arguments.size()==9 && o.outputs.size()==1 &&
        o.weights.has_runtime_loras() && o.arguments[0].shape()==mx::Shape{1,M,H},"real ordinary student decode block required");
    const auto &w=o.weights;
    const auto loras=w.runtime_lora_snapshot();size_t applied_terms=0;
    for(const auto &binding:loras.bindings)applied_terms+=binding.ordered_terms.size();
    report[@"runtime_lora_applied_terms"]=@(applied_terms);
    report[@"resolved_projection_bindings"]=@(loras.bindings.size());
    require(applied_terms==227,"student runtime adapter binding incomplete");
    const auto source_gate=hash(w.at("transformer_blocks.0.img_mlp.gate_up.weight"));
    const auto source_down=hash(w.at("transformer_blocks.0.img_mlp.out.weight"));
    report[@"source_gate_up_sha256"]=ns(source_gate);report[@"source_down_sha256"]=ns(source_down);
    std::vector<std::string> source_hashes;
    for(const auto &a:o.arguments)source_hashes.push_back(hash(a));
    const auto output_hash=hash(o.outputs[0]);
    auto producer=front(w), full=ff::full(w,"transformer_blocks.0.img_mlp.");
    auto finish=mx::compile(ff::Function([](const std::vector<Tensor> &a) {
        return std::vector<Tensor>{a[0]+mx::tanh(a[1])*a[2]};
    }));
    auto initial=producer(o.arguments); mx::eval(initial);
    auto ffn_reference=full({initial[1]})[0]; mx::eval(ffn_reference);
    auto staged=finish({initial[0],o.arguments[4],ffn_reference})[0]; mx::eval(staged);
    auto staged_quality=difference(o.outputs[0],staged);
    report[@"staged_gpu_quality"]=staged_quality;
    require([staged_quality[@"changed"] unsignedLongLongValue]==0,"staged GPU did not exactly reproduce ordinary compiled block");
    mx::save_safetensors((out/"real-input-and-gpu-reference.safetensors").string(),
        {{"input",initial[1]},{"ffn_reference",ffn_reference},{"block_reference",o.outputs[0]}});
    report[@"input_sha256"]=ns(hash(initial[1])); report[@"ordinary_block_sha256"]=ns(output_hash);
    report[@"prefix_tokens"]=@(o.prefix_tokens);
    Prefix prefix(w,width,shared_ranks,out/"private-cache");
    report[@"private_setup_seconds"]=@(prefix.setup_seconds); report[@"constant_bytes"]=@(prefix.constant_bytes);
    report[@"compile_selector_called"]=@(prefix.compiled);
    report[@"static_constants_sha256"]=ns(prefix.constants_sha256);report[@"static_mil_sha256"]=ns(prefix.mil_sha256);
    NSMutableArray *samples=[NSMutableArray array]; report[@"samples"]=samples;
    const std::array<const char *,4> names{"gpu-original","gpu-staged","private-serial","private-hybrid"};
    Tensor last_ffn=ffn_reference,last_block=staged;
    for(int iteration=-3;iteration<12;++iteration)for(int visit=0;visit<4;++visit) {
        const int arm=(visit+iteration+4)%4;
        prefix.poison(); // poison is outside every arm's timed boundary
        const auto start=Clock::now(); Tensor result=o.outputs[0];std::optional<Tensor> evaluated_feed;
        if(arm==0) { auto values=o.gpu_block(o.arguments); mx::eval(values); result=values[0]; }
        else {
            auto values=producer(o.arguments); mx::eval(values);
            auto feed=arm==1?full({values[1]})[0]:prefix.run(values[1],arm==3);
            result=finish({values[0],o.arguments[4],feed})[0]; mx::eval(result);
            evaluated_feed=feed;
            if(arm==3) {last_ffn=feed; last_block=result;}
        }
        mx::synchronize(); const double seconds=elapsed(start);
        auto quality=difference(o.outputs[0],result);
        require([quality[@"relative_l2"] doubleValue]<=.001 && [quality[@"cosine"] doubleValue]>=.999,
                "whole-block fixed numerical gate failed");
        if(arm<2)require([quality[@"changed"] unsignedLongLongValue]==0,"GPU replay changed ordinary output");
        id feed_quality=NSNull.null;
        if(evaluated_feed) {
            feed_quality=difference(ffn_reference,*evaluated_feed);
            require([feed_quality[@"relative_l2"] doubleValue]<=.03 && [feed_quality[@"cosine"] doubleValue]>=.999,
                    "full FFN first/warm/measured numerical gate failed");
            if(arm==1)require([feed_quality[@"changed"] unsignedLongLongValue]==0,"GPU FFN replay changed");
        }
        [samples addObject:@{@"iteration":@(iteration),@"arm":@(names[arm]),@"seconds":@(seconds),
                            @"quality":quality,@"ffn_quality":feed_quality}];
    }
    auto ffn_quality=difference(ffn_reference,last_ffn);
    require([ffn_quality[@"relative_l2"] doubleValue]<=.03 && [ffn_quality[@"cosine"] doubleValue]>=.999,
            "complete FFN numerical gate failed");
    report[@"ffn_quality"]=ffn_quality; report[@"successful_private_calls"]=@(prefix.successful_calls);
    report[@"private_phase_totals"]=@{@"scope":@"all serial/hybrid first/warm/measured host spans; waits overlap GPU; not kernel attribution",
        @"corrections_ready_seconds":@(prefix.corrections_ready_seconds),@"submission_seconds":@(prefix.submission_seconds),
        @"exposed_private_wait_seconds":@(prefix.exposed_wait_seconds),@"post_join_seconds":@(prefix.post_join_seconds)};
    mx::save_safetensors((out/"real-private-output.safetensors").string(),{{"ffn",last_ffn},{"block",last_block}});
    for(size_t i=0;i<o.arguments.size();++i)require(hash(o.arguments[i])==source_hashes[i],"borrowed block input changed");
    require(hash(o.outputs[0])==output_hash,"borrowed original output changed");
    require(hash(w.at("transformer_blocks.0.img_mlp.gate_up.weight"))==source_gate &&
            hash(w.at("transformer_blocks.0.img_mlp.out.weight"))==source_down,"original base FFN source changed");
    report[@"borrowed_values_unchanged"]=@YES;
    report[@"status"]=@"measured";
}
} // namespace lab

int main(int argc,const char **argv) {
    @autoreleasepool {
        if(argc==2 && std::string_view(argv[1])=="--emitter-contract") {
            try {lab::emitter_contract();return 0;}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
        }
        if(argc!=7) {std::cerr<<"usage: probe MODEL ADAPTER FRESH_OUTPUT SHARE separate|shared REFERENCE_OR_DASH\n";return 2;}
        const auto out=std::filesystem::absolute(argv[3]);
        bool created=false;
        NSMutableDictionary *report=[@{@"schema":@"tc-qwen-static-lora-block-v1",@"status":@"starting",
            @"scope":@"ordinary layer0 decode replay; complete r128/joint LoRA; full attention/FFN/residual host boundary; not request or image quality",
            @"qualification_passed":@NO,@"physical_overlap_qualified":@NO,@"image_export":@NO} mutableCopy];
        try {
            require(!std::filesystem::exists(out) && std::filesystem::create_directories(out),"fresh owned output required");
            created=true;
            require(std::filesystem::space(out).available>(2ull<<30),"static pilot needs 2GiB free reserve");
            const auto adapter=std::filesystem::canonical(argv[2]);
            const auto *pin=qwen21::viggle_v021_adapter(adapter.filename().string());
            require(pin && pin->rank=="r128" && sha256_file(adapter)==pin->sha256,"original pinned r128 adapter required");
            const int width=std::stoi(argv[4]); require(width==1024 || width==2048,"bounded prefix share required");
            const std::string recipe=argv[5];require(recipe=="separate" || recipe=="shared","rank sharing recipe required");
            const bool shared_ranks=recipe=="shared", editing=std::string_view(argv[6])!="-";
            setenv("TURBOCIDER_QWEN21_LORA_BF16_AB","1",1);
            configure_streams(); std::atomic<bool> cancelled{false};
            qwen21::Session session(argv[1]); bool finished=false; int observations=0;
            session.set_diagnostic_block_observer([&](const auto &o) {
                require(o.layer==0 && observations<2 && o.steady==(observations==1),"bounded ordinary block observation order differs");
                ++observations; if(!o.steady)return;
                lab::measure(o,width,shared_ranks,out,report); throw qwen21::DiagnosticBlockReplayComplete();
            });
            Request r; r.model="qwen-image-2.1"; r.operation=editing?"image.edit":"image.generate";
            r.prompt=editing?"Change the teapot to bright red. Preserve its shape, handle, spout, table and background.":
                "A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.";
            r.width=r.height=512; r.steps=6; r.seed=73; r.audio=false; r.execution="gpu";
            r.residency="resident"; r.allow_approximation=true; r.lora_strategy="inference_time";
            r.loras.push_back({adapter.string(),1.f,"transformer"});
            r.qwen21_dit_cache="off"; r.qwen21_dit_cache_explicit=true;
            r.output=(out/"must-not-export.png").string();
            if(editing) {InputAsset ref;ref.kind="image";ref.role="reference";ref.path=std::filesystem::canonical(argv[6]).string();
                r.inputs.push_back(ref);r.qwen21_reference_size=512;report[@"reference_sha256"]=lab::ns(sha256_file(ref.path));}
            report[@"adapter_sha256"]=lab::ns(std::string(pin->sha256));report[@"operation"]=lab::ns(r.operation);
            report[@"prefix_channels"]=@(width);report[@"seed"]=@(r.seed);
            report[@"input_rank_recipe"]=lab::ns(recipe);
            const Event event=[](const std::string &phase,int n,int total){std::cerr<<phase<<' '<<n<<'/'<<total<<'\n';};
            try {(void)session.generate(r,event,cancelled);}catch(const qwen21::DiagnosticBlockReplayComplete &){finished=true;}
            session.set_diagnostic_block_observer({});session.unload();mx::synchronize();
            require(finished && observations==2 && !std::filesystem::exists(r.output),"probe did not stop at step1/layer0");
            char *identity=tc_runtime_build_identity();require(identity,"loaded runtime identity missing");
            report[@"loaded_runtime_id"]=lab::ns(identity);tc_string_free(identity);
            report[@"completed_full_dit_steps"]=@1;report[@"stopped_step"]=@1;report[@"stopped_layer"]=@0;
        } catch(const std::exception &error) {report[@"status"]=@"failed";report[@"error"]=lab::ns(error.what());}
        if(created) {
            std::ofstream file(out/"native.json",std::ios::out|std::ios::trunc);file<<json(report)<<'\n';file.close();
        }
        std::cout<<json(report)<<'\n';return [report[@"status"] isEqual:@"measured"]?0:1;
    }
}

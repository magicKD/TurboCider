#include "qwen21_static_lora_prefix.hpp"
#include "qwen21_qkv_lora_prepare.hpp"
#include "../../native/models/qwen21/pipeline.hpp"
#include "../../native/models/qwen21/viggle_adapter.hpp"
#include "../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>
#include <map>

#ifndef TC_COMBO_CPP_SHA
#define TC_COMBO_CPP_SHA "unbound"
#define TC_COMBO_QKV_SHA "unbound"
#define TC_COMBO_PREFIX_SHA "unbound"
#define TC_COMBO_MIL_SHA "unbound"
#endif
using namespace tc;
namespace lab {
using Fn=qwen21::DiagnosticGpuBlock;
constexpr int M=1024,H=4096,F=12288;
constexpr std::array<const char*,9> names{"gpu-original","gpu-staged","gpu-qkv","private","private-qkv","private-event","private-qkv-event",
    "private-packed","private-qkv-packed"};
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
NSString *ns(const std::string &s){return [NSString stringWithUTF8String:s.c_str()];}
std::string hash(const Tensor &a){auto b=mx::view(mx::contiguous(a),mx::uint8);mx::eval(b);
    return memory_sha256_hex({reinterpret_cast<const char*>(b.data<uint8_t>()),b.nbytes()});}
NSDictionary *difference(const Tensor &reference,const Tensor &value){
    require(reference.shape()==value.shape(),"full comparison geometry differs");
    auto a=mx::contiguous(mx::astype(reference,mx::float32)),b=mx::contiguous(mx::astype(value,mx::float32));mx::eval(a,b);
    double error=0,energy=0,other=0,dot=0,maximum=0;size_t changed=0;
    for(size_t i=0;i<a.size();++i){const double x=a.data<float>()[i],y=b.data<float>()[i],d=x-y;
        require(std::isfinite(x)&&std::isfinite(y),"nonfinite complete consumer");
        error+=d*d;energy+=x*x;other+=y*y;dot+=x*y;maximum=std::max(maximum,std::abs(d));changed+=x!=y;}
    return @{@"relative_l2":@(std::sqrt(error/std::max(energy,1e-30))),
        @"cosine":@(energy==0&&other==0?1.:dot/std::sqrt(std::max(energy*other,1e-60))),
        @"max_abs":@(maximum),@"changed":@(changed),@"elements":@(a.size())};
}
void gate(NSDictionary *q,double limit,bool exact=false){
    require([q[@"relative_l2"] doubleValue]<=limit&&[q[@"cosine"] doubleValue]>=.999,"complete consumer numeric gate failed");
    if(exact)require([q[@"changed"] unsignedLongLongValue]==0,"strong ordinary GPU replay changed");
}
Tensor rotate(const Tensor &x,const Tensor &cosine,const Tensor &sine){
    auto shape=x.shape(),pairs=shape;pairs.back()/=2;pairs.push_back(2);auto p=mx::reshape(mx::astype(x,mx::float32),pairs);
    auto a=mx::squeeze(slice_axis(p,-1,0,1),-1),b=mx::squeeze(slice_axis(p,-1,1,2),-1);
    auto c=mx::reshape(cosine,{1,1,shape[2],shape[3]/2}),s=mx::reshape(sine,{1,1,shape[2],shape[3]/2});
    return mx::astype(mx::reshape(mx::stack({a*c-b*s,a*s+b*c},-1),shape),x.dtype());
}
struct QkvBank {
    Tensor packed{0.f};std::array<Tensor,3> up{Tensor(0.f),Tensor(0.f),Tensor(0.f)};
    std::array<float,3> scales{1,1,1};
    const std::array<std::string,3> prefixes{"transformer_blocks.0.attn.to_q","transformer_blocks.0.attn.to_k","transformer_blocks.0.attn.to_v"};
    double setup_seconds=0;
    explicit QkvBank(const Weights &w){const auto start=Clock::now();
        require(w.runtime_lora_bf16_fp32_ranks()&&w.runtime_lora_b_epilogue(),"fixed joint A/B recipe required");
        const auto snapshot=w.runtime_lora_snapshot();require(!snapshot.fp16_rank_math,"F32 A ranks required");std::vector<Tensor> bases;
        for(int i=0;i<3;++i){const auto &p=prefixes[i];const auto &base=w.at(p+".weight");
            require(base.shape()==mx::Shape{H,H}&&base.dtype()==mx::bfloat16&&!w.has(p+".bias")&&!w.quantized(p)&&!w.convrot(p)&&!w.nvfp4(p),
                    "original unbiased BF16 QKV required");
            auto found=std::find_if(snapshot.bindings.begin(),snapshot.bindings.end(),[&](const auto &x){return x.prefix==p;});
            require(found!=snapshot.bindings.end()&&found->ordered_terms.size()==1,"one full QKV adapter term required");
            const auto &t=found->ordered_terms[0];require(t.down.shape()==mx::Shape{128,H}&&t.up.shape()==mx::Shape{H,128}&&
                t.down.dtype()==mx::bfloat16&&t.up.dtype()==mx::bfloat16&&t.output_start==0&&t.output_end==H,"complete r128 QKV factors required");
            up[i]=t.up;scales[i]=t.scale;bases.push_back(base);
        }
        packed=mx::contiguous(mx::concatenate(bases,0));mx::eval(packed);
        for(int i=0;i<3;++i)require(hash(slice_axis(packed,0,i*H,(i+1)*H))==hash(bases[i]),"QKV pack changed base values");
        setup_seconds=elapsed(start);
    }
    std::vector<Tensor> project(const Weights &w,const Tensor &x,const Tensor &c,const Tensor &s,bool fused) const {
        if(fused){std::array<Tensor,3> ranks{Tensor(0.f),Tensor(0.f),Tensor(0.f)};
            for(int i=0;i<3;++i){auto low=w.lora_input_ranks(x,prefixes[i],0,H);require(low.size()==1,"complete online A rank required");ranks[i]=low[0];}
            return qwen21::qkv_lora_lab::prepare(x,packed,ranks,up,scales,w.at("transformer_blocks.0.attn.norm_q.weight"),
                w.at("transformer_blocks.0.attn.norm_k.weight"),c,s,32);
        }
        std::vector<Tensor> result;for(int i=0;i<3;++i){auto v=heads(linear(x,w,prefixes[i]),32,128);
            if(i<2)v=rotate(mx::fast::rms_norm(v,w.at(i==0?"transformer_blocks.0.attn.norm_q.weight":"transformer_blocks.0.attn.norm_k.weight"),1e-6f),c,s);
            result.push_back(v);}return result;
    }
};
std::vector<Tensor> front(const Weights &w,const QkvBank &bank,const std::vector<Tensor> &a,bool fused){
    const std::string p="transformer_blocks.0.";auto x=mx::fast::layer_norm(a[0],{},{},1e-6f)*(Tensor(1.f,a[0].dtype())+a[1]);
    auto qkv=bank.project(w,x,a[5],a[6],fused);
    auto attention=attend(qkv[0],mx::concatenate({a[7],qkv[1]},2),mx::concatenate({a[8],qkv[2]},2));
    auto hidden=a[0]+mx::tanh(a[2])*linear(attention,w,p+"attn.to_out.0");
    x=mx::fast::layer_norm(hidden,{},{},1e-6f)*(Tensor(1.f,hidden.dtype())+a[3]);return {hidden,x};
}
Fn full_qkv(const Weights &w,const QkvBank &bank){return mx::compile(Fn([&w,&bank](const std::vector<Tensor> &a){
    auto value=front(w,bank,a,true);auto gu=linear(value[1],w,"transformer_blocks.0.img_mlp.gate_up");
    auto ff=silu(slice_axis(gu,-1,0,F))*slice_axis(gu,-1,F,2*F);
    return std::vector<Tensor>{value[0]+mx::tanh(a[4])*linear(ff,w,"transformer_blocks.0.img_mlp.out")};
}));}
std::map<std::string,std::string> layer_hashes(const Weights &w){std::map<std::string,std::string> result;
    for(const auto &key:w.sorted_keys())if(key.starts_with("transformer_blocks.0."))result.emplace(key,hash(w.at(key)));
    for(const auto &b:w.runtime_lora_snapshot().bindings)if(b.prefix.starts_with("transformer_blocks.0."))
        for(size_t i=0;i<b.ordered_terms.size();++i){const auto key=b.prefix+".term"+std::to_string(i);
            result.emplace(key+".A",hash(b.ordered_terms[i].down));result.emplace(key+".B",hash(b.ordered_terms[i].up));}
    return result;
}
void measure(const qwen21::DiagnosticBlockObservation &o,const std::filesystem::path &out,NSMutableDictionary *report){
    require(o.layer==0&&o.steady&&o.arguments.size()==9&&o.outputs.size()==1&&o.arguments[0].shape()==mx::Shape{1,M,H},"real complete ordinary decode block required");
    const auto &w=o.weights;size_t terms=0;for(const auto &b:w.runtime_lora_snapshot().bindings)terms+=b.ordered_terms.size();
    require(terms==227,"complete original student adapter required");report[@"runtime_lora_applied_terms"]=@(terms);report[@"prefix_tokens"]=@(o.prefix_tokens);
    const auto weights_before=layer_hashes(w);const auto original_hash=hash(o.outputs[0]);std::vector<std::string> input_hashes;
    for(const auto &x:o.arguments)input_hashes.push_back(hash(x));QkvBank bank(w);
    auto ordinary=mx::compile(Fn([&w,&bank](const auto &a){return front(w,bank,a,false);}));
    auto fused=mx::compile(Fn([&w,&bank](const auto &a){return front(w,bank,a,true);}));
    auto gpu_fused=full_qkv(w,bank),full=qwen21::runtime_ffn::full(w,"transformer_blocks.0.img_mlp.");
    auto finish=mx::compile(Fn([](const auto &a){return std::vector<Tensor>{a[0]+mx::tanh(a[1])*a[2]};}));
    std::array<std::vector<Tensor>,2> contexts{ordinary(o.arguments),fused(o.arguments)};
    std::array<Tensor,2> ffn_reference{full({contexts[0][1]})[0],full({contexts[1][1]})[0]};mx::eval(ffn_reference[0],ffn_reference[1]);
    auto staged=finish({contexts[0][0],o.arguments[4],ffn_reference[0]})[0];mx::eval(staged);gate(difference(o.outputs[0],staged),.001,true);
    auto qkv_only=gpu_fused(o.arguments)[0];mx::eval(qkv_only);gate(difference(o.outputs[0],qkv_only),.001);
    auto changed=o.arguments;changed[0]=mx::astype(changed[0]*Tensor(.984375f,mx::bfloat16)+Tensor(.00390625f,mx::bfloat16),mx::bfloat16);mx::eval(changed[0]);
    auto alt_reference=o.gpu_block(changed)[0],alt_qkv=gpu_fused(changed)[0];mx::eval(alt_reference,alt_qkv);gate(difference(alt_reference,alt_qkv),.001);
    report[@"fresh_gpu_arguments_verified"]=@YES;
    qwen21::static_lora_lab::PreparedPrefix prefix(w,2048,false,out/"private-cache");
    prefix.prepare_packed_suffix(w);report[@"packed_suffix_setup_seconds"]=@(prefix.packed_suffix_setup_seconds);
    report[@"packed_suffix_bytes"]=@(prefix.packed_suffix_bytes);
    report[@"qkv_pack_and_verify_seconds"]=@(bank.setup_seconds);report[@"qkv_packed_bytes"]=@(bank.packed.nbytes());
    report[@"private_setup_seconds"]=@(prefix.setup_seconds);report[@"private_constant_bytes"]=@(prefix.constant_bytes);
    report[@"private_compiled_now"]=@(prefix.compiled);report[@"private_constants_sha256"]=ns(prefix.constants_sha256);report[@"private_mil_sha256"]=ns(prefix.mil_sha256);
    auto alt_input=mx::astype(contexts[0][1]*Tensor(.984375f,mx::bfloat16)+Tensor(.00390625f,mx::bfloat16),mx::bfloat16);mx::eval(alt_input);
    auto plain_suffix=prefix.suffix_control(contexts[0][1],false),packed_suffix=prefix.suffix_control(contexts[0][1],true);
    mx::eval(plain_suffix);mx::eval(packed_suffix);
    gate(difference(plain_suffix[0],packed_suffix[0]),.001,true);gate(difference(plain_suffix[1],packed_suffix[1]),.001,true);
    auto alt_plain_suffix=prefix.suffix_control(alt_input,false),alt_packed_suffix=prefix.suffix_control(alt_input,true);
    mx::eval(alt_plain_suffix);mx::eval(alt_packed_suffix);
    gate(difference(alt_plain_suffix[0],alt_packed_suffix[0]),.001,true);gate(difference(alt_plain_suffix[1],alt_packed_suffix[1]),.001,true);
    report[@"packed_suffix_control_exact"]=@YES;
    mx::save_safetensors((out/"packed-suffix-controls.safetensors").string(),{{"partial_reference",plain_suffix[0]},
        {"hidden_reference",plain_suffix[1]},{"partial_packed",packed_suffix[0]},{"hidden_packed",packed_suffix[1]}});
    auto alt_ffn=full({alt_input})[0];mx::eval(alt_ffn);
    const auto event_fence=qwen21::static_lora_lab::ReadyFence::EventValue;
    prefix.poison();auto first=prefix.run(contexts[0][1],true,event_fence);gate(difference(ffn_reference[0],first),.03);const auto first_hash=hash(first);
    prefix.poison();auto second=prefix.run(alt_input,true,event_fence);gate(difference(alt_ffn,second),.03);
    require(hash(first)==first_hash&&hash(second)!=first_hash,"private output owner or fresh input failure");
    prefix.poison();auto restored=prefix.run(contexts[0][1],true,event_fence);require(hash(restored)==first_hash,"private A/B/A restoration failed");
    report[@"private_fresh_input_restoration"]=@YES;
    const auto no_fence=qwen21::static_lora_lab::ReadyFence::None;
    const auto transfer_first=qwen21::static_lora_lab::SubmitOrder::TransferFirst;
    prefix.poison();auto gpu_a=prefix.run(contexts[0][1],true,no_fence,transfer_first,true);gate(difference(ffn_reference[0],gpu_a),.03);const auto gpu_a_hash=hash(gpu_a);
    prefix.poison();auto gpu_b=prefix.run(alt_input,true,no_fence,transfer_first,true);gate(difference(alt_ffn,gpu_b),.03);
    require(hash(gpu_a)==gpu_a_hash&&hash(gpu_b)!=gpu_a_hash,"packed suffix output ownership/fresh input failure");
    prefix.poison();auto gpu_restored=prefix.run(contexts[0][1],true,no_fence,transfer_first,true);
    require(hash(gpu_restored)==gpu_a_hash&&gpu_a_hash==first_hash,"packed suffix A/B/A restoration or schedule identity failure");
    report[@"packed_suffix_fresh_input_restoration"]=@YES;report[@"preflight_private_calls"]=@6;
    report[@"preflight_ready_wait"]=@"event-value-and-packed-suffix";
    report[@"schedule_recipe"]=@"packed-suffix-v4";report[@"measured_samples_per_arm"]=@18;
    mx::save_safetensors((out/"inputs-reference.safetensors").string(),{{"input",o.arguments[0]},{"block_reference",o.outputs[0]},
        {"ffn_input_original",contexts[0][1]},{"ffn_input_qkv",contexts[1][1]},{"ffn_reference_original",ffn_reference[0]},{"ffn_reference_qkv",ffn_reference[1]}});
    NSMutableArray *samples=[NSMutableArray array];report[@"samples"]=samples;std::vector<Tensor> last(names.size(),o.outputs[0]);
    std::array<Tensor,6> last_private{first,first,first,first,first,first};
    for(int iteration=-3;iteration<18;++iteration)for(int visit=0;visit<int(names.size());++visit){
        const int arm=(visit+iteration+int(names.size()))%int(names.size());const bool is_private=arm>=3,use_qkv=arm==2||arm==4||arm==6||arm==8;
        const bool producer_fence=arm==5||arm==6,packed_arm=arm>=7;
        const auto fence=producer_fence?event_fence:qwen21::static_lora_lab::ReadyFence::None;
        const auto order=transfer_first;
        const std::array<double,7> before{prefix.corrections_ready_seconds,prefix.submission_seconds,prefix.input_ready_wait_seconds,
            prefix.exposed_wait_seconds,prefix.post_join_seconds,prefix.gpu_submit_seconds,prefix.transfer_submit_seconds};
        if(is_private)prefix.poison();const auto start=Clock::now();Tensor value(0.f);std::optional<Tensor> feed;
        id producer_time=NSNull.null;
        if(arm==0||arm==2){value=(arm==0?o.gpu_block(o.arguments):gpu_fused(o.arguments))[0];require(!value.is_available(),"cached completed GPU output returned");mx::eval(value);}
        else{auto context=(use_qkv?fused:ordinary)(o.arguments);mx::eval(context);producer_time=@(elapsed(start));
            feed=is_private?prefix.run(context[1],true,fence,order,packed_arm):full({context[1]})[0];
            value=finish({context[0],o.arguments[4],*feed})[0];mx::eval(value);}
        mx::synchronize();const double wall=elapsed(start);auto q=difference(o.outputs[0],value);gate(q,.001,arm<2);
        id matched=NSNull.null,chain=NSNull.null;
        if(feed){matched=difference(ffn_reference[use_qkv?1:0],*feed);chain=difference(ffn_reference[0],*feed);gate(matched,.03,arm==1);gate(chain,.03,arm==1);}
        if(is_private)last_private[arm-3]=*feed;last[arm]=value;
        id phases=NSNull.null;
        if(is_private)phases=@{@"corrections_ready_seconds":@(prefix.corrections_ready_seconds-before[0]),
            @"submission_seconds":@(prefix.submission_seconds-before[1]),@"input_ready_wait_seconds":@(prefix.input_ready_wait_seconds-before[2]),
            @"exposed_wait_seconds":@(prefix.exposed_wait_seconds-before[3]),@"post_join_seconds":@(prefix.post_join_seconds-before[4]),
            @"producer_fence":@(producer_fence),@"ready_wait_kind":producer_fence?@"event-value":@"none",
            @"private_complete_at_ready":producer_fence?@(prefix.last_private_complete_at_ready):NSNull.null,
            @"submission_order":@"transfer-first",@"gpu_suffix_recipe":packed_arm?@"packed-base":@"original",
            @"gpu_submit_seconds":@(prefix.gpu_submit_seconds-before[5]),@"transfer_submit_seconds":@(prefix.transfer_submit_seconds-before[6])};
        [samples addObject:@{@"iteration":@(iteration),@"arm":@(names[arm]),@"seconds":@(wall),@"quality":q,
            @"matched_input_ffn_quality":matched,@"original_chain_ffn_quality":chain,@"private_phases":phases,@"producer_seconds":producer_time}];
    }
    std::unordered_map<std::string,Tensor> outputs;for(size_t arm=0;arm<names.size();++arm)outputs.emplace(names[arm],last[arm]);
    outputs.emplace("private_ffn",last_private[0]);outputs.emplace("private_qkv_ffn",last_private[1]);
    outputs.emplace("private_event_ffn",last_private[2]);outputs.emplace("private_qkv_event_ffn",last_private[3]);
    outputs.emplace("private_packed_ffn",last_private[4]);outputs.emplace("private_qkv_packed_ffn",last_private[5]);
    mx::save_safetensors((out/"complete-outputs.safetensors").string(),outputs);
    require(prefix.successful_calls==132&&prefix.input_ready_fences==45&&prefix.event_value_fences==45&&prefix.gpu_first_submissions==0&&prefix.packed_suffix_calls==45,
            "all actual private jobs/event-value fences/packed consumers required");
    require(prefix.ready_before_private_completion>0,"no observed ready fence preceded physical Private completion");
    report[@"successful_private_calls"]=@(prefix.successful_calls);report[@"input_ready_fences"]=@(prefix.input_ready_fences);
    report[@"event_value_fences"]=@(prefix.event_value_fences);report[@"ready_before_private_completion"]=@(prefix.ready_before_private_completion);
    report[@"gpu_first_submissions"]=@(prefix.gpu_first_submissions);
    report[@"packed_suffix_calls"]=@(prefix.packed_suffix_calls);
    report[@"private_phase_totals"]=@{@"scope":@"preflight/warm/measured host spans; waits overlap GPU, not physical-engine trace",
        @"corrections_ready_seconds":@(prefix.corrections_ready_seconds),@"submission_seconds":@(prefix.submission_seconds),
        @"exposed_wait_seconds":@(prefix.exposed_wait_seconds),@"post_join_seconds":@(prefix.post_join_seconds),
        @"input_ready_wait_seconds":@(prefix.input_ready_wait_seconds)};
    for(size_t i=0;i<o.arguments.size();++i)require(hash(o.arguments[i])==input_hashes[i],"borrowed input changed");
    require(hash(o.outputs[0])==original_hash&&layer_hashes(w)==weights_before,"borrowed weights/factors/output changed");
    NSMutableDictionary *source=[NSMutableDictionary dictionary];for(const auto &[key,value]:weights_before)source[ns(key)]=ns(value);
    report[@"actual_layer_source_sha256"]=source;report[@"input_sha256"]=ns(input_hashes[0]);report[@"ordinary_block_sha256"]=ns(original_hash);
    report[@"borrowed_values_unchanged"]=@YES;report[@"status"]=@"measured";
}
} // namespace lab

int main(int argc,const char **argv){@autoreleasepool{
    if(argc==2&&std::string_view(argv[1])=="--describe"){
        NSMutableArray *arms=[NSMutableArray array];for(const auto *name:lab::names)[arms addObject:@(name)];char *id=tc_runtime_build_identity();if(!id)return 1;
        auto info=@{@"arms":arms,@"primary_recipe":@"private-qkv-packed",@"schedule_recipe":@"packed-suffix-v4",@"ready_wait_kind":@"mixed",
            @"preflight_ready_wait":@"event-value-and-packed-suffix",@"static_prefix_channels":@2048,@"shared_ranks":@NO,
            @"measured_samples_per_arm":@18,@"warmups_per_arm":@3,@"cpp_sha256":@TC_COMBO_CPP_SHA,@"qkv_sha256":@TC_COMBO_QKV_SHA,
            @"prefix_sha256":@TC_COMBO_PREFIX_SHA,@"mil_sha256":@TC_COMBO_MIL_SHA,@"loaded_runtime_id":lab::ns(id)};
        tc_string_free(id);std::cout<<json(info)<<'\n';return 0;
    }
    if(argc!=5){std::cerr<<"usage: probe MODEL PINNED_R128 FRESH_OUTPUT REFERENCE_OR_DASH\n";return 2;}
    const auto out=std::filesystem::absolute(argv[3]);bool created=false;
    NSMutableDictionary *report=[@{@"schema":@"tc-qwen-qkv-private-combo-block-v1",@"status":@"starting",@"qualification_passed":@NO,
        @"image_export":@NO,@"static_prefix_channels":@2048,@"shared_ranks":@NO,
        @"scope":@"ordinary r128/joint step1 layer0 full-block; original GPU/QKV/static Private/combination; not full request/image quality/physical overlap"} mutableCopy];
    try{
        require(!std::filesystem::exists(out)&&std::filesystem::create_directories(out),"fresh owned output required");created=true;
        require(std::filesystem::space(out).available>(2ull<<30),"2GiB free reserve required");
        const auto adapter=std::filesystem::canonical(argv[2]);const auto *pin=qwen21::viggle_v021_adapter(adapter.filename().string());
        require(pin&&pin->rank=="r128"&&sha256_file(adapter)==pin->sha256,"original pinned r128 required");
        setenv("TURBOCIDER_QWEN21_LORA_BF16_AB","1",1);configure_streams();std::atomic<bool> cancelled{false};qwen21::Session session(argv[1]);int count=0;bool finished=false;
        session.set_diagnostic_block_observer([&](const auto &o){require(o.layer==0&&count<2&&o.steady==(count==1),"bounded observation order differs");
            ++count;if(!o.steady)return;lab::measure(o,out,report);throw qwen21::DiagnosticBlockReplayComplete();});
        const bool edit=std::string_view(argv[4])!="-";Request r;r.model="qwen-image-2.1";r.operation=edit?"image.edit":"image.generate";
        r.prompt=edit?"Change the teapot to bright red. Preserve its shape, handle, spout, table and background.":
            "A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.";
        r.width=r.height=512;r.steps=6;r.seed=73;r.audio=false;r.execution="gpu";r.residency="resident";r.allow_approximation=true;
        r.lora_strategy="inference_time";r.loras.push_back({adapter.string(),1.f,"transformer"});r.qwen21_dit_cache="off";r.qwen21_dit_cache_explicit=true;
        r.output=(out/"must-not-export.png").string();if(edit){InputAsset ref;ref.kind="image";ref.role="reference";ref.path=std::filesystem::canonical(argv[4]).string();
            r.inputs.push_back(ref);r.qwen21_reference_size=512;report[@"reference_sha256"]=lab::ns(sha256_file(ref.path));}
        report[@"adapter_sha256"]=lab::ns(std::string(pin->sha256));report[@"operation"]=lab::ns(r.operation);report[@"seed"]=@(r.seed);
        const Event event=[](const std::string &p,int n,int total){std::cerr<<p<<' '<<n<<'/'<<total<<'\n';};
        try{(void)session.generate(r,event,cancelled);}catch(const qwen21::DiagnosticBlockReplayComplete&){finished=true;}
        session.set_diagnostic_block_observer({});session.unload();mx::synchronize();require(finished&&count==2&&!std::filesystem::exists(r.output),"probe did not stop before image decoding");
        char *id=tc_runtime_build_identity();require(id,"loaded runtime missing");report[@"loaded_runtime_id"]=lab::ns(id);tc_string_free(id);
        report[@"completed_full_dit_steps"]=@1;report[@"stopped_step"]=@1;report[@"stopped_layer"]=@0;
    }catch(const std::exception &e){report[@"status"]=@"failed";report[@"error"]=lab::ns(e.what());}
    if(created){std::ofstream file(out/"native.json");file<<json(report)<<'\n';}std::cout<<json(report)<<'\n';return [report[@"status"] isEqual:@"measured"]?0:1;
}}

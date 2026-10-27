// Real layer0 complete block followed by a DEPENDENT projection fixture.
// The next normalization reuses layer0's modulation, not natural layer1 state.
// This isolates the FFN->QKV staging edge; it is not a two-block model forward.
#include "../../../native/models/qwen21/pipeline.hpp"
#include "../../../native/models/qwen21/runtime_ffn_graphs.hpp"
#include "../../../native/models/qwen21/runtime_private_qkv.hpp"
#include "../../../native/models/qwen21/metal/base_dual_swiglu.hpp"
#include "../../../native/backends/ane_ffn.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <map>

#ifndef TC_FRONTIER_NATIVE_ID
#error helper must bind the exact native C++ ABI
#endif
#ifndef TC_FRONTIER_CPP_SHA
#error helper source identity required
#endif
namespace {
using namespace tc;
using Fn=qwen21::DiagnosticGpuBlock;
using Scope=void(*)(int);
constexpr int M=1024,H=4096,F=12288,C=6144,A=6;
const std::array<const char*,A> names{"original-post","dual-post","original-pre","dual-pre","original-hidden","dual-hidden"};
NSString *ns(const std::string&s){return [NSString stringWithUTF8String:s.c_str()];}
void write(const std::filesystem::path&p,NSDictionary*v){std::ofstream f(p);f<<json(v)<<'\n';require(bool(f),"frontier report write failed");}
std::string hash(const Tensor&x){auto b=mx::view(mx::contiguous(x),mx::uint8);mx::eval(b);
    return memory_sha256_hex({reinterpret_cast<const char*>(b.data<uint8_t>()),b.nbytes()});}
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
Tensor rope(const Tensor&x,const Tensor&c,const Tensor&s){auto shape=x.shape(),pairs=shape;pairs.back()/=2;pairs.push_back(2);
    auto p=mx::reshape(mx::astype(x,mx::float32),pairs),a=mx::squeeze(slice_axis(p,-1,0,1),-1),b=mx::squeeze(slice_axis(p,-1,1,2),-1);
    auto co=mx::reshape(c,{1,1,shape[2],64}),si=mx::reshape(s,{1,1,shape[2],64});
    return mx::astype(mx::reshape(mx::stack({a*co-b*si,a*si+b*co},-1),shape),x.dtype());}
NSDictionary *prefetch(const QwenPrivateQkvPrefetchMetrics&s){return @{@"attempts":@(s.attempts),@"submissions":@(s.submissions),
    @"claims":@(s.claims),@"producer_submissions":@(s.producer_submissions),@"weight_completions":@(s.weight_completions),
    @"completed_hits":@(s.completed_hits),@"discards":@(s.discards),@"retired_claims":@(s.retired_claims),@"failures":@(s.failures),
    @"requested":@(s.requested),@"enabled":@(s.enabled),@"future_pending":@(s.future_pending)};}

void measure(const qwen21::DiagnosticBlockObservation&o,const std::filesystem::path&out,const char*descriptor,
             int repeats,NSMutableDictionary*report,Scope scope){
    require(o.layer==0&&o.steady&&o.arguments.size()==9&&o.arguments[0].shape()==mx::Shape{1,M,H}&&o.outputs.size()==1,
        "original base first decode block0 required");
    const auto&w=o.weights;require(!w.has_runtime_loras(),"base only");
    auto qsource=[&](int l){const auto p="transformer_blocks."+std::to_string(l)+".attn.to_";
        return std::array<Tensor,3>{w.at(p+"q.weight"),w.at(p+"k.weight"),w.at(p+"v.weight")};};
    std::array<std::array<Tensor,3>,2> qweights{qsource(0),qsource(1)};
    std::map<std::string,std::string> source_hash;
    for(int l=0;l<2;++l)for(int j=0;j<3;++j){const std::string path="transformer_blocks."+std::to_string(l)+".attn.to_"+std::string(j==0?"q":j==1?"k":"v")+".weight";
        qweights[l][j]=w.at(path);require(qweights[l][j].shape()==mx::Shape{H,H}&&qweights[l][j].dtype()==mx::bfloat16,"original QKV source required");
        source_hash[path]=hash(qweights[l][j]);}
    const std::string prefix="transformer_blocks.0.img_mlp.";
    const auto&gu=w.at(prefix+"gate_up.weight");const auto&down=w.at(prefix+"out.weight");
    require(gu.shape()==mx::Shape{2*F,H}&&down.shape()==mx::Shape{H,F}&&gu.dtype()==mx::bfloat16&&down.dtype()==mx::bfloat16,"original FFN source required");
    source_hash[prefix+"gate_up.weight"]=hash(gu);source_hash[prefix+"out.weight"]=hash(down);
    auto halves=mx::split(gu,2,0);mx::eval(halves);std::vector<ane::FfnWeight> sources{{halves[0]},{halves[1]},{down}};
    auto full=qwen21::runtime_ffn::full(w,prefix);
    auto prepare=mx::compile(Fn([&w](const auto&a){const std::string p="transformer_blocks.0.";
        auto x=mx::fast::layer_norm(a[0],{},{},1e-6f)*(Tensor(1.f,a[0].dtype())+a[1]);
        auto q=heads(linear(x,w,p+"attn.to_q"),32,128),k=heads(linear(x,w,p+"attn.to_k"),32,128),v=heads(linear(x,w,p+"attn.to_v"),32,128);
        q=rope(mx::fast::rms_norm(q,w.at(p+"attn.norm_q.weight"),1e-6f),a[5],a[6]);
        k=rope(mx::fast::rms_norm(k,w.at(p+"attn.norm_k.weight"),1e-6f),a[5],a[6]);
        auto att=attend(q,mx::concatenate({a[7],k},2),mx::concatenate({a[8],v},2));
        auto h=a[0]+mx::tanh(a[2])*linear(att,w,p+"attn.to_out.0");
        return std::vector<Tensor>{h,mx::fast::layer_norm(h,{},{},1e-6f)*(Tensor(1.f,h.dtype())+a[3])};}));
    auto finish=mx::compile(Fn([](const auto&a){return std::vector<Tensor>{a[0]+mx::tanh(a[1])*a[2]};}));
    auto norm=mx::compile(Fn([](const auto&a){return std::vector<Tensor>{mx::fast::layer_norm(a[0],{},{},1e-6f)*(Tensor(1.f,a[0].dtype())+a[1])};}));
    auto project=mx::compile(Fn([](const auto&a){return std::vector<Tensor>{mx::concatenate({mx::matmul(a[0],mx::transpose(a[1])),
        mx::matmul(a[0],mx::transpose(a[2])),mx::matmul(a[0],mx::transpose(a[3]))},-1)};}));
    auto original=o.arguments,alternate=o.arguments;alternate[0]=mx::astype(alternate[0]*Tensor(.984375f,mx::bfloat16)+Tensor(.00390625f,mx::bfloat16),mx::bfloat16);mx::eval(alternate[0]);
    std::vector<std::string> input_hashes;for(const auto&a:original)input_hashes.push_back(hash(a));const auto borrowed_hash=hash(o.outputs[0]);
    std::unordered_map<std::string,Tensor> payload;
    for(int v=0;v<2;++v){const auto&args=v?alternate:original;auto parts=prepare(args);auto reference=o.gpu_block(args)[0];
        auto staged=finish({parts[0],args[4],full({parts[1]})[0]})[0];mx::eval(reference,staged);require(hash(reference)==hash(staged),"staged full GPU block arithmetic differs");
        if(!v)require(hash(reference)==borrowed_hash,"ordinary captured GPU control changed");
        payload.emplace(v?"gpu_b":"gpu_a",reference);payload.emplace(v?"ffn_input_b":"ffn_input_a",parts[1]);}
    setenv("TURBOCIDER_ANE_BACKEND","private",1);setenv("TURBOCIDER_ALLOW_PRIVATE_ANE","1",1);
    setenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH","w8a8",1);setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","6144",1);
    setenv("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM","1",1);setenv("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION","sigmoid-complement",1);
    setenv("TURBOCIDER_PRIVATE_ANE_SCALE_CACHE","1",1);setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE","1",1);
    setenv("TURBOCIDER_PRIVATE_ANE_PREFETCH","0",1);setenv("TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE","1",1);
    setenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1",1);setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS","1",1);
    setenv("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN","1",1);setenv("TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES","0",1);
    if(scope)scope(0);std::atomic<bool> cancelled{false};ane::HybridFfn runtime(descriptor,H,F,1536ull<<20,cancelled);
    ane::HybridFfn::SourceScope source_scope(&runtime);qwen21::RuntimePrivateQkv qkv(512ull<<20,cancelled);
    require(runtime.available()&&runtime.supports_after_submit()&&qkv.available(),"fixed joint fixture unavailable");
    qkv.begin_request("frontier-original-bf16-source",true);const auto program_key=qkv.metrics().program_key;
    uint64_t edge_calls=0;NSMutableArray*samples=[NSMutableArray array];report[@"samples"]=samples;
    auto run=[&](int arm,int variant){
        const auto&args=variant==1?alternate:original;const int target=variant==2?0:1;
        require(runtime.plan_block(0,M).split()&&qkv.plan(M),"fixed joint plan declined");runtime.stage_weights(0,M,sources);
        auto parts=prepare(args);bool staged=false,fallback=false;
        auto ahead=[&]{require(!staged&&qkv.prefetch(target,M,qweights[target]),"single QKV future did not submit once");staged=true;++edge_calls;};
        auto feed=runtime.run(0,parts[1],[&](const Tensor&x){fallback=true;return full({x})[0];},cancelled,nullptr,
            [&](const Tensor&x,int first,int count){require(first==0&&count==C,"fixed GPU complementary range changed");
                if(arm/2==1)ahead();
                auto h=arm%2?qwen21::metal::prepare_base_dual_swiglu(x,gu,0,C):
                    silu(w.project_slice(x,prefix+"gate_up",0,C,0,H,false))*w.project_slice(x,prefix+"gate_up",F,F+C,0,H,false);
                if(arm/2==2){mx::async_eval(h);ahead();}
                return std::make_pair(w.project_base_slice_fp32(h,prefix+"out",0,H,0,C),h);
            },{},[&]{if(arm/2==0)ahead();});
        require(staged&&!fallback,"fixture lost one-hop or fell back");
        auto block=finish({parts[0],args[4],feed})[0];
        require(!block.is_available(),"complete block reused an already materialized output");
        qkv.stage(target,M,qweights[target]);
        // Explicit bound fixture: reuse current layer's scale for next norm.
        auto x=norm({block,args[1]})[0];
        auto raw=qkv.run(target,x,[&](const Tensor&v){return project({v,qweights[target][0],qweights[target][1],qweights[target][2]})[0];},cancelled);
        mx::eval(block,raw);return std::make_pair(block,raw);
    };
    std::array<std::array<std::string,2>,3> anchors;
    for(int v=0;v<4;++v)for(int arm=0;arm<A;++arm){const int variant=v==3?0:v;if(scope)scope(1+v*A+arm);
        auto values=run(arm,variant);const std::array<std::string,2> hashes{hash(values.first),hash(values.second)};
        if(!arm&&v<3)anchors[variant]=hashes;else require(hashes==anchors[variant],"frontier/fusion/weight/input restoration changed full output bits");
        if(v<3){const auto name=std::string(names[arm])+"_"+std::string(v==0?"a":v==1?"b":"w");
            payload.emplace(name+"_block",values.first);payload.emplace(name+"_qkv",values.second);}
    }
    require(anchors[0][1]!=anchors[1][1]&&anchors[0][1]!=anchors[2][1],"changed inputs/weights returned stale QKV");
    std::array<std::optional<std::pair<Tensor,Tensor>>,A> last;
    for(int rep=-2;rep<repeats;++rep)for(int j=0;j<A;++j){const int arm=(j+rep+A)%A;const int variant=bool(rep&1);
        if(scope)scope(100+(rep+2)*A+j);const auto before=runtime.metrics();const auto qb=qkv.metrics();const auto pb=qkv.prefetch_metrics();
        const auto start=Clock::now();auto value=run(arm,variant);const auto wall=seconds(start);
        const auto after=runtime.metrics();const auto qa=qkv.metrics();const auto pa=qkv.prefetch_metrics();
        [samples addObject:@{@"arm":@(names[arm]),@"rep":@(rep),@"warmup":@(rep<0),@"input_variant":@(bool(variant)),@"seconds":@(wall),
            @"ffn_caller_seconds":@(after.runtime_weight_wall_seconds-before.runtime_weight_wall_seconds),
            @"ffn_async_wait_seconds":@(after.runtime_weight_async_ane_wait_seconds-before.runtime_weight_async_ane_wait_seconds),
            @"qkv_weight_wait_seconds":@(pa.weight_join_wait_seconds-pb.weight_join_wait_seconds),
            @"qkv_predict_restore_join_seconds":@(qa.join_seconds-qb.join_seconds)}];
        // Completion is inside run(). Do not stream CPU checksums between
        // performance samples; independently retain each arm's final outputs.
        last[arm]=std::move(value);
    }
    for(int arm=0;arm<A;++arm)require(last[arm]&&hash(last[arm]->first)==anchors[1][0]&&hash(last[arm]->second)==anchors[1][1],
        "last measured alternate input did not restore complete outputs");
    if(scope)scope(-3);source_scope.finish();qkv.drain();const auto m=runtime.metrics();const auto q=qkv.metrics();const auto p=qkv.prefetch_metrics();
    const uint64_t calls=A*(repeats+6);
    require(edge_calls==calls&&m.runtime_calls==calls&&m.runtime_weight_hybrid_blocks==calls&&m.runtime_weight_program_instances==1&&
        !m.runtime_failed&&m.runtime_weight_fallback_blocks==0&&m.runtime_weight_overflow_retries==0&&m.runtime_weight_headroom==1&&
        q.predictions==calls&&q.weight_stages==calls&&q.completed_blocks==calls&&q.validation_calls==1&&q.validation_passed&&!q.failed&&
        q.fallback_blocks==0&&q.program_key==program_key,"actual fixed programs or successful workload counts differ");
    require(p.attempts==calls&&p.submissions==calls&&p.producer_submissions==3*calls&&p.claims==calls&&p.weight_completions==calls&&
        p.completed_hits==calls&&!p.future_pending&&!p.discards&&!p.retired_claims&&!p.failures,"one-bank future/completion counts differ");
    for(const auto&[name,digest]:source_hash)require(hash(w.at(name))==digest,"original checkpoint weights changed");
    for(size_t i=0;i<original.size();++i)require(hash(original[i])==input_hashes[i],"borrowed source input changed");
    require(hash(o.outputs[0])==borrowed_hash,"borrowed complete block output changed");
    NSMutableDictionary*hashes=[NSMutableDictionary dictionary];for(const auto&[name,digest]:source_hash)hashes[ns(name)]=ns(digest);
    mx::save_safetensors((out/"arrays.safetensors").string(),payload);
    report[@"source_arrays_sha256"]=hashes;report[@"metrics"]=to_dictionary(m);report[@"prefetch"]=prefetch(p);
    report[@"qkv"]=@{@"predictions":@(q.predictions),@"weight_stages":@(q.weight_stages),@"completed_blocks":@(q.completed_blocks),
        @"validation_calls":@(q.validation_calls),@"validation_passed":@(q.validation_passed),@"failed":@(q.failed),@"fallback_blocks":@(q.fallback_blocks),
        @"program_key":ns(q.program_key),@"slot_bytes":@(q.slot_bytes),@"estimated_bytes":@(q.estimated_bytes)};
    report[@"source_unchanged"]=@YES;report[@"all_outputs_bits_exact"]=@YES;report[@"single_bank"]=@YES;
    report[@"prefix_tokens"]=@(o.prefix_tokens);report[@"status"]=@"complete";
}
}
extern "C" __attribute__((visibility("default")))
int tc_qwen_frontier_probe(const char*model,const char*reference,const char*descriptor,const char*output,int repeats,void(*scope)(int)){@autoreleasepool{
    const std::filesystem::path out(output?output:"");NSMutableDictionary*report=[@{@"schema":@"tc-qwen-frontier-fixture-v2",@"status":@"starting",
        @"timing_checksum_policy":@"post-loop-last-per-arm",@"last_owned_output_bytes":@(uint64_t(A)*M*4*H*2),
        @"complete_request":@NO,@"image_export":@NO,@"natural_two_block_forward":@NO,@"next_norm":@"reused-layer0-modulation-fixture",
        @"current_block_qkv":@"gpu",@"completed_full_dit_steps":@1,@"stopped_step":@1,@"stopped_layer":@0,
        @"repeats":@(repeats),@"warmups":@2,@"cpp_sha256":@TC_FRONTIER_CPP_SHA,@"compiled_native_id":@TC_FRONTIER_NATIVE_ID,
        @"extra_owned_weight_bytes":@0,@"scope":@"real complete layer0 block plus dependent next-weight QKV scheduling fixture; not full40 speed, natural two-block math or physical overlap/INT8 proof"}mutableCopy];
    try{require(model&&reference&&descriptor&&output&&(repeats==16||repeats==32)&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"bounded fresh frontier fixture required");
        char*id=tc_runtime_build_identity();require(id,"runtime identity missing");std::string identity(id);tc_string_free(id);report[@"loaded_runtime_id"]=ns(identity);
        require(identity==TC_FRONTIER_NATIVE_ID,"frontier helper/native C++ ABI mismatch");configure_streams();qwen21::Session session(model);
        Request r;r.model="qwen-image-2.1";r.operation=std::string_view(reference)=="-"?"image.generate":"image.edit";
        r.width=r.height=512;r.steps=40;r.frames=1;r.audio=false;r.seed=73;r.execution="gpu";r.residency="resident";r.allow_approximation=false;
        r.qwen21_dit_cache="off";r.qwen21_dit_cache_explicit=true;r.qwen21_reference_size=1024;r.output=(out/"not-exported.png").string();
        r.prompt=r.operation=="image.generate"?"A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.":"Change the teapot to bright red. Preserve its shape, handle, spout, table and background.";
        if(r.operation=="image.edit")r.inputs.push_back({.kind="image",.role="reference",.path=reference});
        report[@"actual_source_request"]=@{@"model":ns(r.model),@"operation":ns(r.operation),@"width":@(r.width),@"height":@(r.height),@"steps":@(r.steps),
            @"seed":@(r.seed),@"audio":@(r.audio),@"lora_count":@(r.loras.size()),@"reference_count":@(r.inputs.size()),@"reference_size":@(r.qwen21_reference_size)};
        session.set_diagnostic_block_observer([&](const qwen21::DiagnosticBlockObservation&o){if(o.layer||!o.steady)return;
            measure(o,out,descriptor,repeats,report,scope);throw qwen21::DiagnosticBlockReplayComplete();});
        std::atomic<bool>cancelled{false};try{(void)session.generate(r,[](const auto&,int,int){},cancelled);throw std::runtime_error("frontier capture did not stop");}
        catch(const qwen21::DiagnosticBlockReplayComplete&){require([report[@"status"]isEqual:@"complete"],"frontier capture incomplete");}
    }catch(const std::exception&e){report[@"status"]=@"failed";report[@"error"]=ns(e.what());std::cerr<<e.what()<<'\n';}
    if(scope)scope(-3);if(std::filesystem::is_directory(out))write(out/"report.json",report);return[report[@"status"]isEqual:@"complete"]?0:1;
}}

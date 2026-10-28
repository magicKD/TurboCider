#include "qwen21_base_swiglu_prepare.hpp"
#include "../../native/models/qwen21/metal/lora_swiglu.hpp"
#include "../../native/models/qwen21/metal/lora_qkv_prepare.hpp"
#include "../../native/models/qwen21/pipeline.hpp"
#include "../../native/models/qwen21/viggle_adapter.hpp"
#include "../../native/backends/dense_gpu_lora_b.hpp"
#include "../../native/platform/apple/bridge.hpp"
#include "../../native/platform/apple/platform.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>
#include <map>

#ifndef TC_BASE_SWIGLU_CPP_SHA
#define TC_BASE_SWIGLU_CPP_SHA "unbound"
#define TC_BASE_SWIGLU_HEADER_SHA "unbound"
#endif
using namespace tc;
namespace bs {
using Fn=qwen21::DiagnosticGpuBlock;
constexpr std::array<const char*,4> names{"gpu-original","strong-control","base-paired-b32","base-paired-b64"};
constexpr int warmups=3,samples=16;
NSString *ns(const std::string &s){return [NSString stringWithUTF8String:s.c_str()];}
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
std::string hash(const Tensor &a){auto bytes=mx::view(mx::contiguous(a),mx::uint8);mx::eval(bytes);
    return memory_sha256_hex({reinterpret_cast<const char*>(bytes.data<uint8_t>()),bytes.nbytes()});}
NSDictionary *quality(const Tensor &reference,const Tensor &value){
    require(reference.shape()==value.shape()&&reference.dtype()==value.dtype(),"complete typed output geometry differs");
    auto a=mx::contiguous(mx::astype(reference,mx::float32)),b=mx::contiguous(mx::astype(value,mx::float32));mx::eval(a,b);
    double error=0,energy=0,other=0,dot=0,maximum=0;size_t changed=0;
    for(size_t i=0;i<a.size();++i){const double x=a.data<float>()[i],y=b.data<float>()[i],d=x-y;
        require(std::isfinite(x)&&std::isfinite(y),"nonfinite complete output");
        error+=d*d;energy+=x*x;other+=y*y;dot+=x*y;maximum=std::max(maximum,std::abs(d));changed+=x!=y;}
    return @{@"relative_l2":@(std::sqrt(error/std::max(energy,1e-30))),@"max_abs":@(maximum),@"changed":@(changed),
        @"cosine":@(energy==0&&other==0?1.:dot/std::sqrt(std::max(energy*other,1e-60))),@"elements":@(a.size()),@"bits_exact":@(hash(reference)==hash(value))};
}
std::vector<Tensor> reference(const Tensor &x,const Tensor &w,const Tensor &rg,const Tensor &bg,const Tensor &ru,const Tensor &bu,float gs,float us){
    const int n=w.shape(0)/2;auto base=mx::matmul(x,mx::transpose(w));
    auto g=dense_gpu::lora_b_epilogue(rg,bg,0,n,gs,mx::bfloat16,base,0);
    auto u=dense_gpu::lora_b_epilogue(ru,bu,0,n,us,mx::bfloat16,base,n);
    return {slice_axis(base,-1,0,n),slice_axis(base,-1,n,2*n),g,u,mx::sigmoid(g),silu(g),silu(g)*u};
}
bool kernel_contract(){
    NSMutableArray *rows=[NSMutableArray array];bool production=true,all=true;size_t rejected=0;
    for(int fixture=0;fixture<3;++fixture){
        const int m=fixture==0?33:fixture==1?65:1024,k=fixture==0?128:fixture==1?256:4096,n=fixture==0?257:fixture==1?513:12288;
        const int rg=fixture==1?64:128,ru=fixture==0?64:fixture==1?256:128;
        auto x=mx::astype(mx::reshape(mx::sin(mx::arange((m+2)*k,mx::float32)*.013f)*.125f,{1,m+2,k}),mx::bfloat16);x=slice_axis(x,1,1,m+1);
        if(fixture==1)x=mx::expand_dims(mx::transpose(mx::astype(mx::reshape(mx::cos(mx::arange(m*k,mx::float32)*.007f)*.1f,{k,m}),mx::bfloat16)),0);
        auto w=mx::astype(mx::reshape(mx::cos(mx::arange((2*n+4)*k,mx::float32)*.0013f)*.02f,{2*n+4,k}),mx::bfloat16);w=slice_axis(w,0,2,2*n+2);
        auto g=mx::reshape(mx::sin(mx::arange(m*rg,mx::float32)*.013f)*.1f,{1,m,rg});
        auto u=mx::reshape(mx::cos(mx::arange(m*ru,mx::float32)*.011f)*.1f,{1,m,ru});
        auto bg=mx::astype(mx::reshape(mx::sin(mx::arange((n+2)*rg,mx::float32)*.007f)*.03f,{n+2,rg}),mx::bfloat16);bg=slice_axis(bg,0,1,n+1);
        auto bu=mx::astype(mx::reshape(mx::cos(mx::arange((n+2)*ru,mx::float32)*.009f)*.03f,{n+2,ru}),mx::bfloat16);bu=slice_axis(bu,0,1,n+1);
        mx::eval(x,w,g,u,bg,bu);const float gs=.75f,us=-.5f;auto expected=reference(x,w,g,bg,u,bu,gs,us);mx::eval(expected);
        for(int bm:{16,32,64})for(int bn:{64,128})for(int stage=0;stage<7;++stage){
            auto actual=qwen21::base_swiglu_lab::prepare(x,w,g,bg,u,bu,gs,us,bm,bn,stage);mx::eval(actual);
            auto q=quality(expected[stage],actual);const bool exact=[q[@"bits_exact"] boolValue];all=all&&exact;if(fixture==2)production=production&&exact;
            [rows addObject:@{@"fixture":@(fixture),@"m":@(m),@"n":@(n),@"k":@(k),@"bm":@(bm),@"bn":@(bn),@"stage":@(stage),@"quality":q}];
        }
        for(int bad=0;bad<10;++bad){bool denied=false;try{
            qwen21::base_swiglu_lab::prepare(bad==0?mx::astype(x,mx::float32):x,bad==1?mx::astype(w,mx::float16):w,
                bad==2?mx::astype(g,mx::bfloat16):g,bad==3?slice_axis(bg,0,0,n-1):bg,u,bu,
                bad==4?std::numeric_limits<float>::infinity():gs,us,bad==5?128:32,bad==6?32:128,bad==7?-1:bad==8?7:6);
            if(bad==9)qwen21::base_swiglu_lab::prepare(x,mx::transpose(w),g,bg,u,bu,gs,us);
        }catch(const std::invalid_argument&){denied=true;}require(denied,"invalid source/tile/stage accepted");++rejected;}
    }
    std::cout<<json(@{@"schema":@"tc-qwen-base-swiglu-kernel-v1",@"status":@"complete",@"all_fixtures_bitwise_exact":@(all),
        @"production_geometry_bitwise_exact":@(production),@"rejected":@(rejected),@"cases":rows,@"qualification_passed":@NO})<<'\n';return production;
}
struct Binding {
    std::array<Tensor,3> qkv_up{Tensor(0.f),Tensor(0.f),Tensor(0.f)};std::array<float,3> qkv_scale{};
    std::array<Tensor,2> ff_up{Tensor(0.f),Tensor(0.f)};std::array<float,2> ff_scale{};std::array<size_t,2> ff_index{};
    explicit Binding(const Weights &w){
        size_t qkv=0,ff=0,total=0;
        for(const auto &b:w.runtime_lora_snapshot().bindings){total+=b.ordered_terms.size();
            for(size_t i=0;i<b.ordered_terms.size();++i){const auto &t=b.ordered_terms[i];
                if(b.prefix=="transformer_blocks.0.img_mlp.gate_up"){
                    const int kind=t.output_start==0&&t.output_end==12288?0:t.output_start==12288&&t.output_end==24576?1:-1;
                    require(kind>=0&&t.down.shape()==mx::Shape{128,4096}&&t.up.shape()==mx::Shape{12288,128}&&t.source_adapter_ordinal==0,"complete independent r128 FFN terms required");
                    ff_up[kind]=t.up;ff_scale[kind]=t.scale;ff_index[kind]=i;++ff;}}
            for(int kind=0;kind<3;++kind)if(b.prefix==std::string("transformer_blocks.0.attn.")+(kind==0?"to_q":kind==1?"to_k":"to_v")){
                require(b.ordered_terms.size()==1,"one full QKV term required");const auto &t=b.ordered_terms[0];
                require(t.down.shape()==mx::Shape{128,4096}&&t.up.shape()==mx::Shape{4096,128}&&t.output_start==0&&t.output_end==4096&&t.source_adapter_ordinal==0,"original QKV r128 required");
                qkv_up[kind]=t.up;qkv_scale[kind]=t.scale;++qkv;}}
        require(total==227&&qkv==3&&ff==2&&ff_index[0]!=ff_index[1],"full adapter coverage differs");
        require(w.at("transformer_blocks.0.img_mlp.gate_up.weight").shape()==mx::Shape{24576,4096},"original gate/up geometry required");
    }
    Tensor hidden(const Weights &w,const Tensor &x,int mode,int stage=6)const{
        const std::string name="transformer_blocks.0.img_mlp.gate_up";auto ranks=w.lora_input_ranks(x,name,0,4096);
        require(ranks.size()==2,"two online independent A ranks required");
        if(mode==1){require(stage==6,"control stage unsupported");auto base=mx::matmul(x,mx::transpose(w.at(name+".weight")));
            return qwen21::metal::prepare_lora_swiglu(base,ranks[ff_index[0]],ff_up[0],ranks[ff_index[1]],ff_up[1],ff_scale[0],ff_scale[1],16,128);}
        return qwen21::base_swiglu_lab::prepare(x,w.at(name+".weight"),ranks[ff_index[0]],ff_up[0],ranks[ff_index[1]],ff_up[1],ff_scale[0],ff_scale[1],mode==2?32:64,128,stage);
    }
    std::vector<Tensor> qkv(const Weights &w,const Tensor &x,const Tensor &c,const Tensor &s)const{
        std::array<Tensor,3> ranks{Tensor(0.f),Tensor(0.f),Tensor(0.f)},base=ranks;
        for(int kind=0;kind<3;++kind){const auto name=std::string("transformer_blocks.0.attn.")+(kind==0?"to_q":kind==1?"to_k":"to_v");
            auto r=w.lora_input_ranks(x,name,0,4096);require(r.size()==1,"original QKV online rank required");ranks[kind]=r[0];base[kind]=w.at(name+".weight");}
        return qwen21::metal::prepare_corrected_qkv_direct(x,base,ranks,qkv_up,qkv_scale,w.at("transformer_blocks.0.attn.norm_q.weight"),
            w.at("transformer_blocks.0.attn.norm_k.weight"),c,s,32);
    }
};
Fn block(const Weights &w,const Binding &binding,int mode,int tap=0){return mx::compile(Fn([&w,&binding,mode,tap](const std::vector<Tensor>&a){
    const std::string p="transformer_blocks.0.";
    auto x=mx::fast::layer_norm(a[0],{},{},1e-6f)*(Tensor(1.f,a[0].dtype())+a[1]);auto qkv=binding.qkv(w,x,a[5],a[6]);
    auto attended=attend(qkv[0],mx::concatenate({a[7],qkv[1]},2),mx::concatenate({a[8],qkv[2]},2));
    auto h=a[0]+mx::tanh(a[2])*linear(attended,w,p+"attn.to_out.0");
    x=mx::fast::layer_norm(h,{},{},1e-6f)*(Tensor(1.f,h.dtype())+a[3]);if(tap==1)return std::vector<Tensor>{x};
    auto ff=binding.hidden(w,x,mode);if(tap==2)return std::vector<Tensor>{ff};
    return std::vector<Tensor>{h+mx::tanh(a[4])*linear(ff,w,p+"img_mlp.out")};
}));}
void measure(const qwen21::DiagnosticBlockObservation &o,const std::filesystem::path &out,NSMutableDictionary *report,bool reverse){
    require(o.layer==0&&o.steady&&o.arguments.size()==9&&o.outputs.size()==1&&o.arguments[0].shape()==mx::Shape{1,1024,4096},"actual ordinary r128 decode layer0 required");
    const auto &w=o.weights;std::map<std::string,std::string> sources;std::vector<std::string> arguments;
    for(const auto &name:w.sorted_keys())if(name.starts_with("transformer_blocks.0."))sources[name]=hash(w.at(name));
    for(const auto &b:w.runtime_lora_snapshot().bindings)if(b.prefix.starts_with("transformer_blocks.0."))
        for(size_t i=0;i<b.ordered_terms.size();++i){const auto name=b.prefix+".term"+std::to_string(i);sources[name+".A"]=hash(b.ordered_terms[i].down);sources[name+".B"]=hash(b.ordered_terms[i].up);}
    for(const auto &a:o.arguments)arguments.push_back(hash(a));const auto expected=hash(o.outputs[0]);
    const auto setup=Clock::now();Binding binding(w);report[@"binding_seconds"]=@(elapsed(setup));report[@"extra_owned_weight_bytes"]=@0;
    report[@"runtime_lora_applied_terms"]=@227;report[@"prefix_tokens"]=@(o.prefix_tokens);report[@"order"]=reverse?@"reverse":@"forward";
    report[@"ordinary_block_sha256"]=ns(expected);std::vector<Fn> functions{o.gpu_block};for(int mode=1;mode<4;++mode)functions.push_back(block(w,binding,mode));
    NSMutableArray *first=[NSMutableArray array];report[@"first_replays"]=first;
    for(int mode=1;mode<4;++mode){const auto start=Clock::now();auto value=functions[mode](o.arguments)[0];mx::eval(value);const double wall=elapsed(start);
        auto q=quality(o.outputs[0],value);[first addObject:@{@"arm":@(names[mode]),@"seconds":@(wall),@"quality":q}];
        mx::save_safetensors((out/(std::string(names[mode])+"-initial.safetensors")).string(),{{"block",value}});
        if(![q[@"bits_exact"] boolValue]){
            auto x=block(w,binding,1,1)(o.arguments)[0];mx::eval(x);auto ranks=w.lora_input_ranks(x,"transformer_blocks.0.img_mlp.gate_up",0,4096);
            auto refs=reference(x,w.at("transformer_blocks.0.img_mlp.gate_up.weight"),ranks[binding.ff_index[0]],binding.ff_up[0],ranks[binding.ff_index[1]],binding.ff_up[1],binding.ff_scale[0],binding.ff_scale[1]);mx::eval(refs);
            NSMutableArray *stages=[NSMutableArray array];report[@"failure_stages"]=stages;
            for(int stage=0;stage<7;++stage){auto actual=binding.hidden(w,x,mode,stage);mx::eval(actual);auto sq=quality(refs[stage],actual);
                [stages addObject:@{@"stage":@(stage),@"quality":sq}];mx::save_safetensors((out/("stage-"+std::to_string(stage)+".safetensors")).string(),{{"reference",refs[stage]},{"candidate",actual}});}
            throw std::runtime_error("complete block bits differ; no timing permitted");}}
    auto changed=o.arguments;changed[0]=mx::astype(changed[0]*Tensor(.984375f,mx::bfloat16)+Tensor(.00390625f,mx::bfloat16),mx::bfloat16);mx::eval(changed[0]);
    auto fresh=functions[0](changed)[0];mx::eval(fresh);require(hash(fresh)!=expected,"perturbed input reused old output");
    for(int mode=1;mode<4;++mode){auto value=functions[mode](changed)[0];mx::eval(value);require(hash(value)==hash(fresh),"perturbed output bits differ");
        auto restored=functions[mode](o.arguments)[0];mx::eval(restored);require(hash(restored)==expected,"A-B-A restored output bits differ");}
    report[@"fresh_arguments_verified"]=@YES;
    std::unordered_map<std::string,Tensor> inputs{{"block-reference",o.outputs[0]},{"changed-reference",fresh}};
    for(size_t i=0;i<o.arguments.size();++i)inputs.emplace("argument-"+std::to_string(i),o.arguments[i]);
    auto hidden=block(w,binding,1,2)(o.arguments)[0];mx::eval(hidden);inputs.emplace("hidden-reference",hidden);
    for(int mode=2;mode<4;++mode){auto ff=block(w,binding,mode,2)(o.arguments)[0];mx::eval(ff);require(hash(ff)==hash(hidden),"complete FFN hidden bits differ");inputs.emplace(names[mode],ff);}
    mx::save_safetensors((out/"inputs-and-hidden.safetensors").string(),inputs);inputs.clear();
    // No per-call readback/hash between timing samples. First/fresh/restored
    // and final full outputs are checked outside this balanced timing window.
    NSMutableArray *timings=[NSMutableArray array];report[@"samples"]=timings;std::vector<Tensor> last(4,o.outputs[0]);
    for(int iteration=-warmups;iteration<samples;++iteration)for(int visit=0;visit<4;++visit){const int mode=((reverse?3-visit:visit)+iteration+4)%4;
        const auto start=Clock::now();auto value=functions[mode](o.arguments)[0];require(!value.is_available(),"new invocation returned completed cached tensor");
        mx::eval(value);mx::synchronize();const double wall=elapsed(start);last[mode]=value;
        [timings addObject:@{@"iteration":@(iteration),@"visit":@(visit),@"arm":@(names[mode]),@"seconds":@(wall)}];}
    NSMutableArray *final=[NSMutableArray array];report[@"final_quality"]=final;std::unordered_map<std::string,Tensor> outputs;
    for(int mode=0;mode<4;++mode){auto q=quality(o.outputs[0],last[mode]);[final addObject:@{@"arm":@(names[mode]),@"quality":q}];
        require([q[@"bits_exact"] boolValue],"final full block bits differ");outputs.emplace(names[mode],last[mode]);}
    mx::save_safetensors((out/"complete-block-outputs.safetensors").string(),outputs);
    for(size_t i=0;i<o.arguments.size();++i)require(hash(o.arguments[i])==arguments[i],"borrowed argument changed");require(hash(o.outputs[0])==expected,"borrowed ordinary output changed");
    for(const auto &name:w.sorted_keys())if(name.starts_with("transformer_blocks.0."))require(hash(w.at(name))==sources.at(name),"base source changed");
    for(const auto &b:w.runtime_lora_snapshot().bindings)if(b.prefix.starts_with("transformer_blocks.0."))
        for(size_t i=0;i<b.ordered_terms.size();++i){const auto name=b.prefix+".term"+std::to_string(i);require(hash(b.ordered_terms[i].down)==sources.at(name+".A")&&hash(b.ordered_terms[i].up)==sources.at(name+".B"),"adapter source changed");}
    NSMutableDictionary *source=[NSMutableDictionary dictionary];for(const auto &[name,digest]:sources)source[ns(name)]=ns(digest);report[@"actual_layer_source_sha256"]=source;
    report[@"input_sha256"]=ns(arguments[0]);report[@"borrowed_values_unchanged"]=@YES;report[@"all_outputs_lazy_before_eval"]=@YES;
    report[@"timing_readback_policy"]=@"first-fresh-restored-and-final-only-v1";report[@"status"]=@"measured";
}
}
int main(int argc,const char **argv){@autoreleasepool{
    if(argc==2&&std::string_view(argv[1])=="--kernel-contract"){try{configure_streams();return bs::kernel_contract()?0:1;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
    if(argc==2&&std::string_view(argv[1])=="--describe"){
        char *id=tc_runtime_build_identity();if(!id)return 1;NSMutableArray *arms=[NSMutableArray array];for(const auto *name:bs::names)[arms addObject:@(name)];
        auto d=@{@"recipe":@"original-base-paired-b-swiglu-v2-b64",@"cpp_sha256":@TC_BASE_SWIGLU_CPP_SHA,@"header_sha256":@TC_BASE_SWIGLU_HEADER_SHA,
            @"arms":arms,@"primary":@"base-paired-b64",@"strong_control":@"strong-control",@"warmups":@(bs::warmups),@"samples_per_arm":@(bs::samples),@"loaded_runtime_id":bs::ns(id)};
        tc_string_free(id);std::cout<<json(d)<<'\n';return 0;}
    if(argc!=6){std::cerr<<"usage: probe MODEL PINNED_R128 FRESH_OUTPUT REF_OR_DASH forward|reverse\n";return 2;}
    const auto out=std::filesystem::absolute(argv[3]);bool created=false;
    NSMutableDictionary *report=[@{@"schema":@"tc-qwen-base-swiglu-block-v1",@"status":@"starting",@"qualification_passed":@NO,@"image_export":@NO,
        @"scope":@"step1/layer0 full block; online A/B, full attention/down/residual; not complete request or per-sample numeric proof"} mutableCopy];
    try{
        require(!std::filesystem::exists(out)&&std::filesystem::create_directories(out),"fresh owned output required");created=true;
        require(std::filesystem::space(out).available>(3ull<<30),"3GiB disk reserve required");const std::string order=argv[5];require(order=="forward"||order=="reverse","known order required");
        const auto adapter=std::filesystem::canonical(argv[2]);const auto *pin=qwen21::viggle_v021_adapter(adapter.filename().string());require(pin&&pin->rank=="r128"&&sha256_file(adapter)==pin->sha256,"original pinned r128 required");
        for(const char *flag:{"TURBOCIDER_QWEN21_CORRECTED_QKV","TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT","TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX",
            "TURBOCIDER_QWEN21_PREFIX_SNAPSHOT","TURBOCIDER_QWEN21_LORA_SWIGLU","TURBOCIDER_QWEN21_ASYNC_STEP_PAIR"})setenv(flag,"0",1);
        setenv("TURBOCIDER_QWEN21_LORA_BF16_AB","1",1);configure_streams();qwen21::Session session(argv[1]);int observations=0;bool finished=false;std::atomic<bool> cancelled{false};
        session.set_diagnostic_block_observer([&](const auto &o){require(o.layer==0&&observations<2&&o.steady==(observations==1),"bounded observation sequence differs");++observations;
            if(!o.steady)return;bs::measure(o,out,report,order=="reverse");throw qwen21::DiagnosticBlockReplayComplete();});
        const bool edit=std::string_view(argv[4])!="-";Request r;r.model="qwen-image-2.1";r.operation=edit?"image.edit":"image.generate";
        r.prompt=edit?"Change the teapot to bright red. Preserve its shape, handle, spout, table and background.":"A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.";
        r.width=r.height=512;r.steps=6;r.seed=73;r.audio=false;r.execution="gpu";r.residency="resident";r.allow_approximation=true;r.lora_strategy="inference_time";
        r.qwen21_dit_cache="off";r.qwen21_dit_cache_explicit=true;r.loras.push_back({adapter.string(),1.f,"transformer"});r.output=(out/"must-not-export.png").string();
        if(edit){InputAsset a;a.kind="image";a.role="reference";a.path=std::filesystem::canonical(argv[4]).string();r.inputs.push_back(a);r.qwen21_reference_size=512;report[@"reference_sha256"]=bs::ns(sha256_file(a.path));}
        report[@"adapter_sha256"]=bs::ns(sha256_file(adapter));report[@"operation"]=bs::ns(r.operation);report[@"seed"]=@(r.seed);
        try{(void)session.generate(r,[](const auto &phase,int n,int total){std::cerr<<phase<<' '<<n<<'/'<<total<<'\n';},cancelled);}catch(const qwen21::DiagnosticBlockReplayComplete&){finished=true;}
        session.set_diagnostic_block_observer({});session.unload();mx::synchronize();require(finished&&observations==2&&!std::filesystem::exists(r.output),"bounded replay must stop before VAE/PNG");
        report[@"completed_full_dit_steps"]=@1;report[@"stopped_step"]=@1;report[@"stopped_layer"]=@0;
        char *id=tc_runtime_build_identity();require(id,"loaded runtime missing");report[@"loaded_runtime_id"]=bs::ns(id);tc_string_free(id);
    }catch(const std::exception&e){report[@"status"]=@"failed";report[@"error"]=bs::ns(e.what());}
    if(created){std::ofstream file(out/"native.json");file<<json(report)<<'\n';}std::cout<<json(report)<<'\n';return [report[@"status"] isEqual:@"measured"]?0:1;
}}

#define main tc_previous_qkv_probe_entry
#include "qwen21_qkv_lora_block_probe.mm"
#undef main
#include "qwen21_lora_swiglu_prepare.hpp"

#ifndef TC_SWIGLU_CPP_SHA
#define TC_SWIGLU_CPP_SHA "unbound"
#define TC_SWIGLU_HEADER_SHA "unbound"
#define TC_SWIGLU_HELPER_SHA "unbound"
#define TC_SWIGLU_UNARY_SHA "unbound"
#define TC_SWIGLU_BF16_SHA "unbound"
#endif

namespace swiglu {
constexpr std::array<const char*,7> names{"gpu-original","gpu-replay","direct-qkv","paired-b-swiglu-b16","paired-b-swiglu-b32",
    "direct-qkv-paired-b-swiglu-b16","direct-qkv-paired-b-swiglu-b32"};
constexpr int warmups=3,samples=21;
struct Binding {
    std::array<Tensor,3> qkv_up{Tensor(0.f),Tensor(0.f),Tensor(0.f)};
    std::array<float,3> qkv_scale{};
    std::array<Tensor,2> ff_up{Tensor(0.f),Tensor(0.f)};
    std::array<float,2> ff_scale{};std::array<size_t,2> ff_index{};
    explicit Binding(const Weights &w) {
        auto snapshot=w.runtime_lora_snapshot();size_t qkv_terms=0,ff_terms=0,total=0;
        for(const auto &b:snapshot.bindings){total+=b.ordered_terms.size();
            for(size_t i=0;i<b.ordered_terms.size();++i){const auto &t=b.ordered_terms[i];
                if(b.prefix=="transformer_blocks.0.img_mlp.gate_up"){
                    require(t.down.shape()==mx::Shape{128,4096}&&t.up.shape()==mx::Shape{12288,128}&&t.source_adapter_ordinal==0,
                        "full independent original r128 gate/up sources required");
                    const int kind=t.output_start==0&&t.output_end==12288?0:t.output_start==12288&&t.output_end==24576?1:-1;
                    require(kind>=0,"full gate/up adapter range required");ff_up[kind]=t.up;ff_scale[kind]=t.scale;ff_index[kind]=i;++ff_terms;
                }
            }
            for(int i=0;i<3;++i){const auto key=std::string("transformer_blocks.0.attn.")+(i==0?"to_q":i==1?"to_k":"to_v");
                if(b.prefix!=key)continue;require(b.ordered_terms.size()==1,"single original QKV correction required");const auto &t=b.ordered_terms[0];
                require(t.down.shape()==mx::Shape{128,4096}&&t.up.shape()==mx::Shape{4096,128}&&t.output_start==0&&t.output_end==4096&&t.source_adapter_ordinal==0,
                    "complete original r128 QKV factors required");qkv_up[i]=t.up;qkv_scale[i]=t.scale;++qkv_terms;
            }
        }
        require(total==227&&qkv_terms==3&&ff_terms==2&&ff_index[0]!=ff_index[1],"pinned full r128 source coverage differs");
        require(w.at("transformer_blocks.0.img_mlp.gate_up.weight").shape()==mx::Shape{24576,4096},"original gate/up physical source differs");
    }
    Tensor hidden(const Weights &w,const Tensor &x,int bm) const {
        const std::string p="transformer_blocks.0.img_mlp.gate_up";
        auto base=mx::matmul(x,mx::transpose(w.at(p+".weight")));
        auto ranks=w.lora_input_ranks(x,p,0,4096);require(ranks.size()==2,"two independently recomputed A ranks required");
        return qwen21::swiglu_lab::finish(base,ranks[ff_index[0]],ff_up[0],ranks[ff_index[1]],ff_up[1],ff_scale[0],ff_scale[1],bm,128);
    }
    std::vector<Tensor> project(const Weights &w,const Tensor &x,const Tensor &c,const Tensor &s,bool direct) const {
        if(!direct){std::vector<Tensor> result;
            for(int i=0;i<3;++i){auto value=heads(linear(x,w,std::string("transformer_blocks.0.attn.")+(i==0?"to_q":i==1?"to_k":"to_v")),32,128);
                if(i<2)value=lab::rotate(mx::fast::rms_norm(value,w.at(i==0?"transformer_blocks.0.attn.norm_q.weight":"transformer_blocks.0.attn.norm_k.weight"),1e-6f),c,s);
                result.push_back(value);}return result;
        }
        std::array<Tensor,3> ranks{Tensor(0.f),Tensor(0.f),Tensor(0.f)},base=ranks;
        for(int i=0;i<3;++i){const auto key=std::string("transformer_blocks.0.attn.")+(i==0?"to_q":i==1?"to_k":"to_v");
            auto r=w.lora_input_ranks(x,key,0,4096);require(r.size()==1,"one online QKV rank per original projection required");ranks[i]=r[0];base[i]=w.at(key+".weight");}
        return qwen21::metal::prepare_corrected_qkv_direct(x,base,ranks,qkv_up,qkv_scale,w.at("transformer_blocks.0.attn.norm_q.weight"),
            w.at("transformer_blocks.0.attn.norm_k.weight"),c,s,32);
    }
};
lab::Fn block(const Weights &w,const Binding &binding,int mode,int tap=0){return mx::compile(lab::Fn([&w,&binding,mode,tap](const std::vector<Tensor> &a){
    const std::string p="transformer_blocks.0.";const bool direct=mode==2||mode>=5;
    auto x=mx::fast::layer_norm(a[0],{},{},1e-6f)*(Tensor(1.f,a[0].dtype())+a[1]);
    auto qkv=binding.project(w,x,a[5],a[6],direct);
    auto attended=attend(qkv[0],mx::concatenate({a[7],qkv[1]},2),mx::concatenate({a[8],qkv[2]},2));
    auto hidden=a[0]+mx::tanh(a[2])*linear(attended,w,p+"attn.to_out.0");
    x=mx::fast::layer_norm(hidden,{},{},1e-6f)*(Tensor(1.f,hidden.dtype())+a[3]);
    if(tap==1)return std::vector<Tensor>{x};
    Tensor ff=x;
    if(mode>=3)ff=binding.hidden(w,x,(mode==4||mode==6)?32:16);
    else {auto gu=linear(x,w,p+"img_mlp.gate_up");ff=silu(slice_axis(gu,-1,0,12288))*slice_axis(gu,-1,12288,24576);}
    if(tap==2)return std::vector<Tensor>{ff};
    return std::vector<Tensor>{hidden+mx::tanh(a[4])*linear(ff,w,p+"img_mlp.out")};
}));}
void exact(NSDictionary *q){require([q[@"changed"] unsignedLongLongValue]==0,"complete paired B/SwiGLU consumer changed output bits");}
void measure(const qwen21::DiagnosticBlockObservation &o,const std::filesystem::path &out,NSMutableDictionary *report,bool reverse){
    require(o.layer==0&&o.steady&&o.arguments.size()==9&&o.outputs.size()==1&&o.arguments[0].shape()==mx::Shape{1,1024,4096},"actual ordinary r128 decode layer0 required");
    const auto &w=o.weights;std::map<std::string,std::string> source;
    for(const auto &name:w.sorted_keys())if(name.starts_with("transformer_blocks.0."))source.emplace(name,lab::hash(w.at(name)));
    for(const auto &b:w.runtime_lora_snapshot().bindings)if(b.prefix.starts_with("transformer_blocks.0."))
        for(size_t i=0;i<b.ordered_terms.size();++i){const auto key=b.prefix+".term"+std::to_string(i);source.emplace(key+".A",lab::hash(b.ordered_terms[i].down));source.emplace(key+".B",lab::hash(b.ordered_terms[i].up));}
    std::vector<std::string> input_hashes;for(const auto &a:o.arguments)input_hashes.push_back(lab::hash(a));const auto expected_hash=lab::hash(o.outputs[0]);
    const auto setup=Clock::now();Binding binding(w);report[@"binding_seconds"]=@(lab::elapsed(setup));report[@"extra_owned_weight_bytes"]=@0;
    report[@"runtime_lora_applied_terms"]=@227;report[@"prefix_tokens"]=@(o.prefix_tokens);report[@"order"]=reverse?@"reverse":@"forward";
    std::vector<lab::Fn> functions{o.gpu_block};for(int mode=1;mode<int(names.size());++mode)functions.push_back(block(w,binding,mode));
    NSMutableArray *first=[NSMutableArray array];report[@"first_replays"]=first;
    for(int mode=1;mode<int(names.size());++mode){const auto start=Clock::now();auto value=functions[mode](o.arguments)[0];mx::eval(value);
        const double wall=lab::elapsed(start);auto q=lab::difference(o.outputs[0],value);const auto bits=lab::hash(value);
        [first addObject:@{@"arm":@(names[mode]),@"seconds":@(wall),@"quality":q,@"output_sha256":lab::ns(bits)}];require(bits==expected_hash,"first replay output bits differ");
        mx::save_safetensors((out/(std::string(names[mode])+"-initial.safetensors")).string(),{{"block",value}});exact(q);}
    auto changed=o.arguments;changed[0]=mx::astype(changed[0]*Tensor(.984375f,mx::bfloat16)+Tensor(.00390625f,mx::bfloat16),mx::bfloat16);mx::eval(changed[0]);
    auto fresh=functions[0](changed)[0];mx::eval(fresh);const auto fresh_hash=lab::hash(fresh);require(fresh_hash!=expected_hash,"perturbed input reused old output");
    for(size_t mode=1;mode<functions.size();++mode){auto value=functions[mode](changed)[0];mx::eval(value);exact(lab::difference(fresh,value));require(lab::hash(value)==fresh_hash,"perturbed output bits differ");
        auto restored=functions[mode](o.arguments)[0];mx::eval(restored);exact(lab::difference(o.outputs[0],restored));require(lab::hash(restored)==expected_hash,"restored output bits differ");}
    report[@"fresh_arguments_verified"]=@YES;
    std::unordered_map<std::string,Tensor> diagnostics{{"block_reference",o.outputs[0]},{"changed_reference",fresh}};
    for(size_t i=0;i<o.arguments.size();++i)diagnostics.emplace("argument-"+std::to_string(i),o.arguments[i]);
    // Single-output tap functions, only outside the timing loop. They do not
    // replace the ordinary single-output complete block denominator.
    for(int mode:{1,3,4}){auto value=block(w,binding,mode,2)(o.arguments)[0];mx::eval(value);diagnostics.emplace(std::string(names[mode])+"-hidden",value);}
    exact(lab::difference(diagnostics.at("gpu-replay-hidden"),diagnostics.at("paired-b-swiglu-b16-hidden")));
    exact(lab::difference(diagnostics.at("gpu-replay-hidden"),diagnostics.at("paired-b-swiglu-b32-hidden")));
    require(lab::hash(diagnostics.at("gpu-replay-hidden"))==lab::hash(diagnostics.at("paired-b-swiglu-b16-hidden"))&&
        lab::hash(diagnostics.at("gpu-replay-hidden"))==lab::hash(diagnostics.at("paired-b-swiglu-b32-hidden")),"complete hidden bits differ");
    mx::save_safetensors((out/"inputs-and-hidden.safetensors").string(),diagnostics);diagnostics.clear();
    NSMutableArray *timings=[NSMutableArray array];report[@"samples"]=timings;std::vector<Tensor> last(names.size(),o.outputs[0]);
    for(int iteration=-warmups;iteration<samples;++iteration)for(int visit=0;visit<int(names.size());++visit){
        const int position=reverse?int(names.size())-1-visit:visit;const int mode=(position+iteration+int(names.size()))%int(names.size());
        const auto start=Clock::now();auto value=functions[mode](o.arguments)[0];require(!value.is_available(),"new call returned completed cached tensor");
        mx::eval(value);mx::synchronize();const double wall=lab::elapsed(start);auto q=lab::difference(o.outputs[0],value);exact(q);const auto bits=lab::hash(value);
        require(bits==expected_hash,"live complete output bits differ");last[mode]=value;
        [timings addObject:@{@"iteration":@(iteration),@"visit":@(visit),@"arm":@(names[mode]),@"seconds":@(wall),@"quality":q,@"output_sha256":lab::ns(bits)}];
    }
    std::unordered_map<std::string,Tensor> outputs;for(size_t mode=0;mode<names.size();++mode)outputs.emplace(names[mode],last[mode]);
    mx::save_safetensors((out/"complete-block-outputs.safetensors").string(),outputs);
    for(size_t i=0;i<o.arguments.size();++i)require(lab::hash(o.arguments[i])==input_hashes[i],"borrowed argument changed");
    require(lab::hash(o.outputs[0])==expected_hash,"borrowed ordinary output changed");
    for(const auto &name:w.sorted_keys())if(name.starts_with("transformer_blocks.0."))require(lab::hash(w.at(name))==source.at(name),"base source changed");
    for(const auto &b:w.runtime_lora_snapshot().bindings)if(b.prefix.starts_with("transformer_blocks.0."))
        for(size_t i=0;i<b.ordered_terms.size();++i){const auto key=b.prefix+".term"+std::to_string(i);require(lab::hash(b.ordered_terms[i].down)==source.at(key+".A")&&lab::hash(b.ordered_terms[i].up)==source.at(key+".B"),"adapter source changed");}
    NSMutableDictionary *hashes=[NSMutableDictionary dictionary];for(const auto &[name,digest]:source)hashes[lab::ns(name)]=lab::ns(digest);
    report[@"actual_layer_source_sha256"]=hashes;report[@"input_sha256"]=lab::ns(input_hashes[0]);report[@"ordinary_block_sha256"]=lab::ns(expected_hash);
    report[@"borrowed_values_unchanged"]=@YES;report[@"all_outputs_lazy_before_eval"]=@YES;report[@"bitwise_outputs_checked"]=@YES;report[@"status"]=@"measured";
}
void kernel_test(){
    size_t cases=0,rejected=0;bool sigmoid_witness=false;
    for(int fixture=0;fixture<6;++fixture){
        const int m=fixture==0?1:fixture==1?33:fixture==2?129:fixture==3?65:1024;
        const int n=fixture==0?83:fixture==4?12288:256,rg=fixture==2?256:128,ru=fixture==3?64:128;
        auto base=mx::astype(mx::reshape(mx::sin(mx::arange((m+2)*2*n,mx::float32)*.017f)*8.f,{1,m+2,2*n}),mx::bfloat16);
        base=slice_axis(base,1,1,m+1);
        if(fixture==5){auto negative_zero=mx::full({1,m,n},-0.f,mx::bfloat16);
            base=mx::concatenate({negative_zero,mx::ones({1,m,n},mx::bfloat16)},2);}
        auto g=mx::reshape(mx::sin(mx::arange(m*rg,mx::float32)*.013f)*.1f,{1,m,rg});
        auto u=mx::reshape(mx::cos(mx::arange(m*ru,mx::float32)*.011f)*.1f,{1,m,ru});
        auto bg=mx::astype(mx::reshape(mx::sin(mx::arange((n+2)*rg,mx::float32)*.007f)*.03f,{n+2,rg}),mx::bfloat16);
        auto bu=mx::astype(mx::reshape(mx::cos(mx::arange((n+2)*ru,mx::float32)*.009f)*.03f,{n+2,ru}),mx::bfloat16);
        bg=slice_axis(bg,0,1,n+1);bu=slice_axis(bu,0,1,n+1);mx::eval(base,g,u,bg,bu);
        if(fixture==5){g=mx::zeros_like(g);u=mx::zeros_like(u);}
        const float gs=fixture==3?0.f:fixture==5?-0.f:.75f,us=-.5f;
        auto reference=mx::compile([n,gs,us,&bg,&bu](const std::vector<Tensor>&a){
            auto gate=dense_gpu::lora_b_epilogue(a[1],bg,0,n,gs,mx::bfloat16,a[0],0);
            auto up=dense_gpu::lora_b_epilogue(a[2],bu,0,n,us,mx::bfloat16,a[0],n);
            return std::vector<Tensor>{gate,up,mx::sigmoid(gate),silu(gate),silu(gate)*up};
        });
        auto expected=reference({base,g,u});mx::eval(expected);
        auto widened=mx::astype(mx::sigmoid(mx::astype(expected[0],mx::float32)),mx::bfloat16);mx::eval(widened);
        sigmoid_witness=sigmoid_witness||[lab::difference(expected[2],widened)[@"changed"] unsignedLongLongValue]>0;
        for(int bm:{16,32})for(int bn:{64,128}){
            for(int stage=0;stage<5;++stage){auto actual=qwen21::swiglu_lab::finish(base,g,bg,u,bu,gs,us,bm,bn,stage);mx::eval(actual);
                auto q=lab::difference(expected[stage],actual);
                require([q[@"changed"] unsignedLongLongValue]==0&&lab::hash(expected[stage])==lab::hash(actual),"paired B/SwiGLU typed stage/bit mismatch fixture="+std::to_string(fixture)+" stage="+std::to_string(stage));}
            ++cases;
        }
        for(int bad=0;bad<9;++bad){bool denied=false;try{
            qwen21::swiglu_lab::finish(bad==0?mx::astype(base,mx::float32):base,bad==1?mx::astype(g,mx::bfloat16):g,
                bad==2?slice_axis(bg,0,0,n-1):bg,u,bad==3?mx::astype(bu,mx::float16):bu,
                bad==4?std::numeric_limits<float>::infinity():gs,us,bad==5?64:16,bad==6?32:128,bad==7?-1:bad==8?5:4);
            }catch(const std::invalid_argument&){denied=true;}require(denied,"invalid paired B/SwiGLU contract accepted");++rejected;}
    }
    require(sigmoid_witness,"installed BF16 sigmoid boundary witness missing");
    std::cout<<"kernel_cases="<<cases<<" exact_stages="<<cases*5<<" rejected="<<rejected<<" actual_sigmoid_typed_rounding_witness=true\n";
}
}

int main(int argc,const char **argv){@autoreleasepool{
    if(argc==2&&std::string_view(argv[1])=="--kernel-contract"){
        try{configure_streams();swiglu::kernel_test();return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
    if(argc==2&&std::string_view(argv[1])=="--describe"){
        char *runtime=tc_runtime_build_identity();if(!runtime)return 1;NSMutableArray *arms=[NSMutableArray array];for(const auto *name:swiglu::names)[arms addObject:@(name)];
        auto info=@{@"recipe":@"native-base-paired-b-swiglu-v2-bitwise",@"arms":arms,@"primary":@"direct-qkv-paired-b-swiglu-b16",@"strong_control":@"direct-qkv",
            @"warmups":@(swiglu::warmups),@"samples_per_arm":@(swiglu::samples),@"cpp_sha256":@TC_SWIGLU_CPP_SHA,@"header_sha256":@TC_SWIGLU_HEADER_SHA,
            @"helper_sha256":@TC_SWIGLU_HELPER_SHA,@"unary_header_sha256":@TC_SWIGLU_UNARY_SHA,@"bf16_math_sha256":@TC_SWIGLU_BF16_SHA,@"loaded_runtime_id":lab::ns(runtime)};
        tc_string_free(runtime);std::cout<<json(info)<<'\n';return 0;
    }
    if(argc!=6){std::cerr<<"usage: probe MODEL PINNED_R128 FRESH_OUTPUT REFERENCE_OR_DASH forward|reverse\n";return 2;}
    const auto out=std::filesystem::absolute(argv[3]);bool created=false;
    NSMutableDictionary *report=[@{@"schema":@"tc-qwen-lora-swiglu-block-v1",@"status":@"starting",@"qualification_passed":@NO,@"image_export":@NO,
        @"scope":@"real step1/layer0 complete block, all online A/B/attention/FFN/down/residual; not full request, cached first step or picture quality"} mutableCopy];
    try{
        require(!std::filesystem::exists(out)&&std::filesystem::create_directories(out),"fresh owned output required");created=true;
        require(std::filesystem::space(out).available>(3ull<<30),"3GiB disk reserve required");
        const std::string order=argv[5];require(order=="forward"||order=="reverse","known fixed order required");
        const auto adapter=std::filesystem::canonical(argv[2]);const auto *pin=qwen21::viggle_v021_adapter(adapter.filename().string());
        require(pin&&pin->rank=="r128"&&sha256_file(adapter)==pin->sha256,"original pinned r128 required");
        setenv("TURBOCIDER_QWEN21_CORRECTED_QKV","0",1);setenv("TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT","0",1);setenv("TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX","0",1);
        setenv("TURBOCIDER_QWEN21_PREFIX_SNAPSHOT","0",1);setenv("TURBOCIDER_QWEN21_LORA_BF16_AB","1",1);configure_streams();
        qwen21::Session session(argv[1]);int observations=0;bool finished=false;std::atomic<bool> cancelled{false};
        session.set_diagnostic_block_observer([&](const auto &o){require(o.layer==0&&observations<2&&o.steady==(observations==1),"bounded real observation sequence differs");
            ++observations;if(!o.steady)return;swiglu::measure(o,out,report,order=="reverse");throw qwen21::DiagnosticBlockReplayComplete();});
        const bool edit=std::string_view(argv[4])!="-";Request r;r.model="qwen-image-2.1";r.operation=edit?"image.edit":"image.generate";
        r.prompt=edit?"Change the teapot to bright red. Preserve its shape, handle, spout, table and background.":"A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.";
        r.width=r.height=512;r.steps=6;r.seed=73;r.audio=false;r.execution="gpu";r.residency="resident";r.allow_approximation=true;r.lora_strategy="inference_time";
        r.qwen21_dit_cache="off";r.qwen21_dit_cache_explicit=true;r.loras.push_back({adapter.string(),1.f,"transformer"});r.output=(out/"must-not-export.png").string();
        if(edit){InputAsset input;input.kind="image";input.role="reference";input.path=std::filesystem::canonical(argv[4]).string();r.inputs.push_back(input);r.qwen21_reference_size=512;report[@"reference_sha256"]=lab::ns(sha256_file(input.path));}
        report[@"adapter_sha256"]=lab::ns(sha256_file(adapter));report[@"operation"]=lab::ns(r.operation);report[@"seed"]=@(r.seed);
        try{(void)session.generate(r,[](const auto &phase,int n,int total){std::cerr<<phase<<' '<<n<<'/'<<total<<'\n';},cancelled);}catch(const qwen21::DiagnosticBlockReplayComplete&){finished=true;}
        session.set_diagnostic_block_observer({});session.unload();mx::synchronize();require(finished&&observations==2&&!std::filesystem::exists(r.output),"bounded replay must stop before VAE/PNG");
        char *runtime=tc_runtime_build_identity();require(runtime,"loaded runtime identity missing");report[@"loaded_runtime_id"]=lab::ns(runtime);tc_string_free(runtime);
        report[@"completed_full_dit_steps"]=@1;report[@"stopped_step"]=@1;report[@"stopped_layer"]=@0;
    }catch(const std::exception &e){report[@"status"]=@"failed";report[@"error"]=lab::ns(e.what());}
    if(created){std::ofstream file(out/"native.json");file<<json(report)<<'\n';}std::cout<<json(report)<<'\n';return [report[@"status"] isEqual:@"measured"]?0:1;
}}

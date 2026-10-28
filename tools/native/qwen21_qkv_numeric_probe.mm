// Research-only same-input stage counterfactuals, linked to the sealed v2
// library. Reuse the earlier real-block oracle/helpers, not its entry point.
#define main tc_qkv_previous_probe_entry
#include "qwen21_qkv_lora_block_probe.mm"
#undef main
#include "qwen21_qkv_numeric_prepare.hpp"

#ifndef TC_NUMERIC_CPP_SHA
#define TC_NUMERIC_CPP_SHA "unbound"
#define TC_NUMERIC_HEADER_SHA "unbound"
#define TC_NUMERIC_HELPER_SHA "unbound"
#define TC_NUMERIC_LAB_SHA "unbound"
#endif

namespace numeric {
constexpr std::array<const char*,6> names{"gpu-original","gpu-replay","mpp-prepare","native-base-b16","native-base-b32","no-contract-prepare"};
using lab::Fn;using lab::ns;using lab::hash;using lab::difference;
std::vector<Tensor> project(const Weights &w,const lab::Binding &bank,const Tensor &x,const Tensor &c,const Tensor &s,int mode){
    if(mode==1)return bank.project(w,x,c,s,1);
    std::array<Tensor,3> ranks{Tensor(0.f),Tensor(0.f),Tensor(0.f)},base=ranks;
    for(int i=0;i<3;++i){ranks[i]=w.lora_input_ranks(x,bank.prefixes[i],0,4096)[0];
        if(mode==3||mode==4)base[i]=mx::matmul(x,mx::transpose(w.at(bank.prefixes[i]+".weight")));}
    return qwen21::qkv_numeric_lab::prepare_stage(x,bank.packed,ranks,bank.up,bank.scales,
        w.at("transformer_blocks.0.attn.norm_q.weight"),w.at("transformer_blocks.0.attn.norm_k.weight"),c,s,3,
        mode==3?16:32,(mode==3||mode==4)?&base:nullptr,mode==5?1:0);
}
Fn block(const Weights &w,const lab::Binding &bank,int mode){return mx::compile(Fn([&w,&bank,mode](const auto &a){
    const std::string p="transformer_blocks.0.";
    auto x=mx::fast::layer_norm(a[0],{},{},1e-6f)*(Tensor(1.f,a[0].dtype())+a[1]);auto qkv=project(w,bank,x,a[5],a[6],mode);
    auto attended=attend(qkv[0],mx::concatenate({a[7],qkv[1]},2),mx::concatenate({a[8],qkv[2]},2));
    auto hidden=a[0]+mx::tanh(a[2])*linear(attended,w,p+"attn.to_out.0");
    x=mx::fast::layer_norm(hidden,{},{},1e-6f)*(Tensor(1.f,hidden.dtype())+a[3]);auto gu=linear(x,w,p+"img_mlp.gate_up");
    auto ff=silu(slice_axis(gu,-1,0,12288))*slice_axis(gu,-1,12288,24576);
    return std::vector<Tensor>{hidden+mx::tanh(a[4])*linear(ff,w,p+"img_mlp.out")};
}));}
void measure(const qwen21::DiagnosticBlockObservation &o,const std::filesystem::path &out,NSMutableDictionary *report){
    require(o.layer==0&&o.steady&&o.arguments.size()==9&&o.outputs.size()==1&&o.arguments[0].shape()==mx::Shape{1,1024,4096},"real ordinary layer0 decode required");
    const auto &w=o.weights;size_t terms=0;for(const auto &b:w.runtime_lora_snapshot().bindings)terms+=b.ordered_terms.size();
    require(terms==227,"complete original adapter required");report[@"runtime_lora_applied_terms"]=@(terms);
    std::vector<std::string> before;for(const auto &a:o.arguments)before.push_back(hash(a));const auto original=hash(o.outputs[0]);
    std::map<std::string,std::string> sources;
    for(const auto &key:w.sorted_keys())if(key.starts_with("transformer_blocks.0."))sources.emplace(key,hash(w.at(key)));
    for(const auto &b:w.runtime_lora_snapshot().bindings)if(b.prefix.starts_with("transformer_blocks.0."))
        for(size_t i=0;i<b.ordered_terms.size();++i){const auto key=b.prefix+".term"+std::to_string(i);
            sources.emplace(key+".A",hash(b.ordered_terms[i].down));sources.emplace(key+".B",hash(b.ordered_terms[i].up));}
    lab::Binding bank(w);auto x=mx::fast::layer_norm(o.arguments[0],{},{},1e-6f)*(Tensor(1.f,o.arguments[0].dtype())+o.arguments[1]);mx::eval(x);
    auto reference=mx::compile(Fn([&w,&bank](const auto &a){std::vector<Tensor> base,corrected,normed,roped;
        for(int i=0;i<3;++i){auto b=mx::matmul(a[0],mx::transpose(w.at(bank.prefixes[i]+".weight")));
            auto y=heads(linear(a[0],w,bank.prefixes[i]),32,128);auto n=i<2?mx::fast::rms_norm(y,
                w.at(i==0?"transformer_blocks.0.attn.norm_q.weight":"transformer_blocks.0.attn.norm_k.weight"),1e-6f):y;
            base.push_back(heads(b,32,128));corrected.push_back(y);normed.push_back(n);roped.push_back(i<2?lab::rotate(n,a[1],a[2]):n);}
        base.insert(base.end(),corrected.begin(),corrected.end());base.insert(base.end(),normed.begin(),normed.end());base.insert(base.end(),roped.begin(),roped.end());return base;
    }));
    auto refs=reference({x,o.arguments[5],o.arguments[6]});mx::eval(refs);
    std::array<Tensor,3> base{Tensor(0.f),Tensor(0.f),Tensor(0.f)},ranks=base;
    for(int i=0;i<3;++i){base[i]=mx::matmul(x,mx::transpose(w.at(bank.prefixes[i]+".weight")));ranks[i]=w.lora_input_ranks(x,bank.prefixes[i],0,4096)[0];mx::eval(base[i],ranks[i]);}
    auto norm=mx::compile(Fn([&w](const auto &a){return std::vector<Tensor>{
        mx::fast::rms_norm(a[0],w.at("transformer_blocks.0.attn.norm_q.weight"),1e-6f),
        mx::fast::rms_norm(a[1],w.at("transformer_blocks.0.attn.norm_k.weight"),1e-6f),a[2]};}));
    auto rope=mx::compile(Fn([](const auto &a){return std::vector<Tensor>{lab::rotate(a[0],a[3],a[4]),lab::rotate(a[1],a[3],a[4]),a[2]};}));
    std::unordered_map<std::string,Tensor> arrays{{"normalized_input",x}};
    for(int stage=0;stage<4;++stage)for(int i=0;i<3;++i)arrays.emplace("reference-"+std::to_string(stage)+"-"+std::to_string(i),refs[stage*3+i]);
    NSMutableArray *stages=[NSMutableArray array];report[@"stage_quality"]=stages;
    for(int bm:{16,32})for(bool supplied:{false,true}){
        std::array<std::vector<Tensor>,4> values;
        const std::string label=std::string(supplied?"native-base":"mpp-base")+"-b"+std::to_string(bm);
        for(int stage=0;stage<4;++stage){values[stage]=qwen21::qkv_numeric_lab::prepare_stage(x,bank.packed,ranks,bank.up,bank.scales,
            w.at("transformer_blocks.0.attn.norm_q.weight"),w.at("transformer_blocks.0.attn.norm_k.weight"),o.arguments[5],o.arguments[6],stage,bm,supplied?&base:nullptr);mx::eval(values[stage]);
            for(int i=0;i<3;++i){auto q=difference(refs[stage*3+i],values[stage][i]);
                [stages addObject:@{@"recipe":ns(label),@"stage":@(stage),@"projection":@(i),@"quality":q}];
                arrays.emplace(label+"-"+std::to_string(stage)+"-"+std::to_string(i),values[stage][i]);}
        }
        auto matched_norm=norm(values[1]);auto matched_rope=rope({values[2][0],values[2][1],values[2][2],o.arguments[5],o.arguments[6]});mx::eval(matched_norm);mx::eval(matched_rope);
        for(int i=0;i<3;++i){[stages addObject:@{@"recipe":ns(label+"-matched-norm"),@"stage":@2,@"projection":@(i),@"quality":difference(matched_norm[i],values[2][i])}];
            [stages addObject:@{@"recipe":ns(label+"-matched-rope"),@"stage":@3,@"projection":@(i),@"quality":difference(matched_rope[i],values[3][i])}];
            arrays.emplace(label+"-matched-norm-"+std::to_string(i),matched_norm[i]);arrays.emplace(label+"-matched-rope-"+std::to_string(i),matched_rope[i]);}
        if(!supplied&&bm==32){auto actual=qwen21::metal::prepare_corrected_qkv(x,bank.packed,ranks,bank.up,bank.scales,
            w.at("transformer_blocks.0.attn.norm_q.weight"),w.at("transformer_blocks.0.attn.norm_k.weight"),o.arguments[5],o.arguments[6],32);mx::eval(actual);
            for(int i=0;i<3;++i)require(hash(actual[i])==hash(values[3][i]),"stage decomposition changed the actual shipping-pilot expression bits");}
        auto no_contract=qwen21::qkv_numeric_lab::prepare_stage(x,bank.packed,ranks,bank.up,bank.scales,
            w.at("transformer_blocks.0.attn.norm_q.weight"),w.at("transformer_blocks.0.attn.norm_k.weight"),o.arguments[5],o.arguments[6],3,bm,supplied?&base:nullptr,1);mx::eval(no_contract);
        for(int i=0;i<3;++i){[stages addObject:@{@"recipe":ns(label+"-no-contract"),@"stage":@3,@"projection":@(i),@"quality":difference(refs[9+i],no_contract[i])}];
            arrays.emplace(label+"-no-contract-"+std::to_string(i),no_contract[i]);}
    }
    report[@"actual_full_expression_exact"]=@YES;report[@"numeric_recipe"]=@"rope-contract-v2";
    mx::save_safetensors((out/"stage-outputs.safetensors").string(),arrays);
    std::vector<Fn> functions{o.gpu_block};for(int i=1;i<int(names.size());++i)functions.push_back(numeric::block(w,bank,i));
    auto changed=o.arguments;changed[0]=mx::astype(changed[0]*Tensor(.984375f,mx::bfloat16)+Tensor(.00390625f,mx::bfloat16),mx::bfloat16);mx::eval(changed[0]);
    auto fresh=functions[0](changed)[0];mx::eval(fresh);
    for(size_t i=1;i<functions.size();++i){auto value=functions[i](changed)[0];mx::eval(value);lab::gate(difference(fresh,value),i==1);}
    report[@"fresh_arguments_verified"]=@YES;NSMutableArray *samples=[NSMutableArray array];report[@"samples"]=samples;
    std::unordered_map<std::string,Tensor> blocks{{"reference",o.outputs[0]}};
    for(int iteration=-3;iteration<12;++iteration)for(int visit=0;visit<6;++visit){const int mode=(visit+iteration+6)%6;const auto start=Clock::now();auto value=functions[mode](o.arguments)[0];
        require(!value.is_available(),"cached completed block returned");mx::eval(value);mx::synchronize();const auto wall=lab::elapsed(start);
        auto q=difference(o.outputs[0],value);lab::gate(q,mode<2);if(mode<2)require(hash(value)==original,"ordinary GPU control changed output bits");blocks.insert_or_assign(names[mode],value);
        [samples addObject:@{@"iteration":@(iteration),@"arm":@(names[mode]),@"seconds":@(wall),@"quality":q}];}
    mx::save_safetensors((out/"block-outputs.safetensors").string(),blocks);
    for(size_t i=0;i<o.arguments.size();++i)require(hash(o.arguments[i])==before[i],"borrowed block argument changed");
    require(hash(o.outputs[0])==original,"borrowed ordinary output changed");
    for(const auto &key:w.sorted_keys())if(key.starts_with("transformer_blocks.0."))require(hash(w.at(key))==sources.at(key),"borrowed original source changed");
    for(const auto &b:w.runtime_lora_snapshot().bindings)if(b.prefix.starts_with("transformer_blocks.0."))for(size_t i=0;i<b.ordered_terms.size();++i){const auto key=b.prefix+".term"+std::to_string(i);
        require(hash(b.ordered_terms[i].down)==sources.at(key+".A")&&hash(b.ordered_terms[i].up)==sources.at(key+".B"),"borrowed original factors changed");}
    report[@"borrowed_values_unchanged"]=@YES;report[@"status"]=@"measured";report[@"prefix_tokens"]=@(o.prefix_tokens);
    NSMutableDictionary *hashes=[NSMutableDictionary dictionary];for(const auto &[key,value]:sources)hashes[ns(key)]=ns(value);report[@"actual_layer_source_sha256"]=hashes;
}
}

int main(int argc,const char **argv){@autoreleasepool{
    if(argc==2&&std::string_view(argv[1])=="--describe"){
        char *identity=tc_runtime_build_identity();if(!identity)return 1;
        auto descriptor=@{@"recipe":@"rope-contract-v2",@"cpp_sha256":@TC_NUMERIC_CPP_SHA,@"header_sha256":@TC_NUMERIC_HEADER_SHA,
            @"helper_sha256":@TC_NUMERIC_HELPER_SHA,@"lab_sha256":@TC_NUMERIC_LAB_SHA,@"runtime_build_id":lab::ns(identity),
            @"primary":@"no-contract-prepare",@"warmups":@3,@"samples_per_arm":@12};
        tc_string_free(identity);std::cout<<json(descriptor)<<'\n';return 0;
    }
    if(argc==2&&std::string_view(argv[1])=="--rope-compile-contract"){
        configure_streams();auto x=mx::ones({1,33,128},mx::bfloat16),w=mx::ones({384,128},mx::bfloat16);
        std::array<Tensor,3> ranks{mx::ones({1,33,64},mx::float32),mx::ones({1,33,64},mx::float32),mx::ones({1,33,64},mx::float32)};
        std::array<Tensor,3> up{mx::ones({128,64},mx::bfloat16),mx::ones({128,64},mx::bfloat16),mx::ones({128,64},mx::bfloat16)};
        auto gamma=mx::ones({128},mx::bfloat16),c=mx::ones({33,64},mx::float32),s=mx::zeros({33,64},mx::float32);
        auto y=qwen21::qkv_numeric_lab::prepare_stage(x,w,ranks,up,{1.f,1.f,1.f},gamma,gamma,c,s,3,32,nullptr,1);mx::eval(y);
        std::cout<<"rope_scoped_no_contract_compiled=1 elements="<<y[0].size()*3<<'\n';return 0;
    }
    if(argc!=5){std::cerr<<"usage: numeric-probe MODEL PINNED_R128 FRESH_OUTPUT REFERENCE_OR_DASH\n";return 2;}
    const auto out=std::filesystem::absolute(argv[3]);bool created=false;NSMutableDictionary *report=[@{@"schema":@"tc-qwen-qkv-numeric-v1",@"status":@"starting",
        @"qualification_passed":@NO,@"image_export":@NO,@"scope":@"real step1/layer0 stage counterfactuals and complete block; not a full-image or unique global-cause claim"} mutableCopy];
    try{require(!std::filesystem::exists(out)&&std::filesystem::create_directories(out),"fresh owned output required");created=true;
        require(std::filesystem::space(out).available>(3ull<<30),"3GiB free reserve required");
        const auto adapter=std::filesystem::canonical(argv[2]);const auto *pin=qwen21::viggle_v021_adapter(adapter.filename().string());
        require(pin&&pin->rank=="r128"&&sha256_file(adapter)==pin->sha256,"original pinned r128 required");
        setenv("TURBOCIDER_QWEN21_CORRECTED_QKV","0",1);setenv("TURBOCIDER_QWEN21_LORA_BF16_AB","1",1);configure_streams();
        std::atomic<bool> cancelled{false};qwen21::Session session(argv[1]);int observations=0;bool finished=false;
        session.set_diagnostic_block_observer([&](const auto &o){require(o.layer==0&&observations<2&&o.steady==(observations==1),"bounded ordinary observation order required");
            ++observations;if(!o.steady)return;numeric::measure(o,out,report);throw qwen21::DiagnosticBlockReplayComplete();});
        const bool edit=std::string_view(argv[4])!="-";Request r;r.model="qwen-image-2.1";r.operation=edit?"image.edit":"image.generate";
        r.prompt=edit?"Change the teapot to bright red. Preserve its shape, handle, spout, table and background.":"A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.";
        r.width=r.height=512;r.steps=6;r.seed=73;r.audio=false;r.execution="gpu";r.residency="resident";r.allow_approximation=true;r.lora_strategy="inference_time";
        r.loras.push_back({adapter.string(),1.f,"transformer"});r.qwen21_dit_cache="off";r.qwen21_dit_cache_explicit=true;r.output=(out/"must-not-export.png").string();
        if(edit){InputAsset ref;ref.kind="image";ref.role="reference";ref.path=std::filesystem::canonical(argv[4]).string();r.inputs.push_back(ref);r.qwen21_reference_size=512;}
        const Event event=[](const auto &phase,int n,int total){std::cerr<<phase<<' '<<n<<'/'<<total<<'\n';};
        try{(void)session.generate(r,event,cancelled);}catch(const qwen21::DiagnosticBlockReplayComplete&){finished=true;}
        session.set_diagnostic_block_observer({});session.unload();mx::synchronize();require(finished&&observations==2&&!std::filesystem::exists(r.output),"numeric probe must stop before VAE/PNG");
        char *identity=tc_runtime_build_identity();require(identity,"loaded runtime identity missing");report[@"runtime_build_id"]=lab::ns(identity);tc_string_free(identity);
        report[@"completed_full_dit_steps"]=@1;report[@"stopped_step"]=@1;report[@"stopped_layer"]=@0;
    }catch(const std::exception &e){report[@"status"]=@"failed";report[@"error"]=lab::ns(e.what());}
    if(created){std::ofstream file(out/"native.json");file<<json(report)<<'\n';}std::cout<<json(report)<<'\n';return [report[@"status"] isEqual:@"measured"]?0:1;
}}

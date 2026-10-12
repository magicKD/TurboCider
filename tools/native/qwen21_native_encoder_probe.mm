#define main qwen21_gpu_encoder_reference_main
#include "qwen21_edit_encoder_probe.mm"
#undef main
#include "../../native/models/qwen21/native_encoder.hpp"
#include "../../native/models/qwen21/pipeline.hpp"
#include "../../native/backends/metal_graph/kernels/dense_attention.hpp"

namespace {
NSDictionary* native_stats(const tc::qwen21::NativeEncoderStats& s) {
    return @{@"vision_layers":@(s.vision_layers),@"language_layers":@(s.language_layers),@"graph_submissions":@(s.graph_submissions),@"dispatches":@(s.dispatches),@"future_weight_windows":@(s.future_weight_windows),@"skipped_future_windows":@(s.skipped_future_windows),@"preparation_seconds":@(s.preparation_seconds),@"source_ready_seconds":@(s.source_ready_seconds),@"gpu_wait_seconds":@(s.gpu_wait_seconds),@"ane_ffns":@(s.ane_ffns),@"ane_chunks":@(s.ane_chunks),@"ane_padded_rows":@(s.ane_padded_rows),@"ane_producer_seconds":@(s.ane_producer_seconds),@"ane_branch_seconds":@(s.ane_branch_seconds),@"ane_consumer_seconds":@(s.ane_consumer_seconds),@"physically_complete":@(s.physically_complete)};
}
void contracts() {
    using namespace tc::metal_graph;using namespace tc::metal_graph::dense_attention;
    validate_geometry({32,128,1,8,ScalarType::BF16});validate_geometry({16,80,1,0,ScalarType::FP32});
    Layout q{32,33,128,256,33*256,32*33*256,0,ScalarType::BF16};auto kv=q;kv.heads=8;kv.view_size_bytes=8*33*256;
    auto b=validate_bindings({32,128,1,8,ScalarType::BF16},q,kv,kv,q,{true,0});need(b.parameters.reserved0==4,"GQA head mapping missing");
    for(auto geom:{AttentionGeometry{32,128,1,7,ScalarType::BF16},AttentionGeometry{16,80,1,0,ScalarType::BF16},AttentionGeometry{16,128,1,0,ScalarType::FP32}}){bool rejected=false;try{validate_geometry(geom);}catch(const std::exception&){rejected=true;}need(rejected,"bad encoder attention ABI accepted");}
    cpu();std::cout<<"native_encoder_cpu=PASS device=0 models=0\n";
}
void generated_native(const std::filesystem::path& out) {
    tc::configure_streams();mx::set_cache_limit(256ULL<<20);std::atomic<bool> cancelled=false;
    NSMutableArray* records=[NSMutableArray array];tc::qwen21::NativeEncoder native;
    for(int rows:{33,129})for(bool final_norm:{false,true}) {
        tc::Weights weights;std::vector<std::string> names;std::vector<Tensor> values;
        auto bind=[&](std::string n,Tensor x){names.push_back(std::move(n));values.push_back(std::move(x));};
        bind("model.embed_tokens.weight",signal({1,512},1));bind("model.norm.weight",mx::ones({512},mx::bfloat16));
        for(int i=0;i<2;++i){auto p="model.layers."+std::to_string(i);
            for(const char* n:{"input_layernorm","post_attention_layernorm"})bind(p+"."+n+".weight",mx::ones({512},mx::bfloat16));
            for(const char* n:{"q_norm","k_norm"})bind(p+".self_attn."+n+".weight",mx::ones({128},mx::bfloat16));
            for(const char* n:{"q_proj","o_proj"})bind(p+".self_attn."+n+".weight",signal({512,512},i+2));
            for(const char* n:{"k_proj","v_proj"})bind(p+".self_attn."+n+".weight",signal({256,512},i+4));
            for(const char* n:{"gate_proj","up_proj"})bind(p+".mlp."+n+".weight",signal({1024,512},i+6));
            bind(p+".mlp.down_proj.weight",signal({512,1024},i+8));
        }
        weights.bind_arrays(names,values);tc::Weights original;original.bind_arrays(names,values);mx::eval(values);
        auto x=signal({1,rows,512},12),idx=mx::arange(rows,mx::int32);auto positions=mx::astype(mx::stack({idx,idx/4,idx%4}),mx::int32);std::vector<Tensor> delta{signal({1,rows,512},13),signal({1,rows,512},14)};
        tc::qwen21::TextConfig c;c.layers=2;c.heads=4;c.kv_heads=2;c.final_norm=final_norm;
        tc::qwen21::TextEncoder oracle(original,c);auto reference=oracle.encode_embeddings(x,positions,rows,{},cancelled,delta);mx::eval(reference);mx::synchronize();
        auto actual=native.language(weights,x,positions,rows,delta,{},cancelled,c);
        [records addObject:@{@"kind":@"full-two-language-layers-GQA-mRoPE-DeepStack",@"rows":@(rows),@"final_norm":@(final_norm),@"error":error(actual,reference,.025)}];
    }
    for(int grid:{4,8}) {
        tc::Weights weights,original;std::vector<std::string> names;std::vector<Tensor> values;
        auto bind=[&](std::string n,Tensor x){names.push_back(std::move(n));values.push_back(std::move(x));};
        const int h=288,f=80,merged=512,width=h*4;
        auto normalization=[&](const std::string& p,int n){bind(p+".weight",mx::ones({n},mx::bfloat16));bind(p+".bias",signal({n},18));};
        auto projection=[&](const std::string& p,int n,int k,int salt){bind(p+".weight",signal({n,k},salt));bind(p+".bias",signal({n},salt+1));};
        bind("model.visual.patch_embed.proj.weight",signal({h,3,2,4,4},3));bind("model.visual.patch_embed.proj.bias",signal({h},4));bind("model.visual.pos_embed.weight",signal({16,h},5));
        for(int i=0;i<2;++i){auto p="model.visual.blocks."+std::to_string(i);normalization(p+".norm1",h);normalization(p+".norm2",h);projection(p+".attn.qkv",h*3,h,i+7);projection(p+".attn.proj",h,h,i+9);projection(p+".mlp.linear_fc1",f,h,i+10);projection(p+".mlp.linear_fc2",h,f,i+11);}
        auto merger=[&](const std::string& p,bool post){normalization(p+".norm",post?width:h);projection(p+".linear_fc1",width,width,16);projection(p+".linear_fc2",merged,width,17);};
        merger("model.visual.merger",false);merger("model.visual.deepstack_merger_list.0",true);merger("model.visual.deepstack_merger_list.1",true);
        weights.bind_arrays(names,values);original.bind_arrays(names,values);mx::eval(values);mx::synchronize();
        tc::qwen21::VisionConfig c;c.patch=4;c.hidden=h;c.heads=4;c.layers=2;c.position_side=4;c.deepstack_layers={0,1};
        auto patches=signal({grid*grid,96},22,mx::float32);tc::qwen21::VisionEncoder oracle(original,c);auto reference=oracle.encode(patches,grid,grid,{},cancelled);mx::eval(reference.merged);mx::eval(reference.deepstack);mx::synchronize();
        auto actual=native.vision(weights,patches,grid,grid,{},cancelled,c);need(actual.deepstack.size()==2,"missing native deepstack mergers");
        [records addObject:@{@"kind":@"complete-two-vision-layers-FP32-D72-rope-mergers",@"grid":@(grid),@"error":error(actual.merged,reference.merged,.025),@"deep0":error(actual.deepstack[0],reference.deepstack[0],.025),@"deep1":error(actual.deepstack[1],reference.deepstack[1],.025)}];
    }
    save(out,@{@"status":@"pass",@"scope":@"generated complete language and vision layers, source tails, GQA, RoPE, FP32 norms, residual/DeepStack/mergers",@"vision_bf16_gemm":@(tc::qwen21::native_encoder_vision_bf16()),@"records":records,@"native":native_stats(native.statistics()),@"models_loaded":@0});
}
void full(const std::filesystem::path& root,const std::filesystem::path& out,const std::filesystem::path& image,tc::qwen21::NativeEncoder* prepared=nullptr,bool components=false) {
    tc::configure_streams();mx::set_cache_limit(256ULL<<20);std::atomic<bool> stop=false;
    const bool metal=tc::qwen21::native_encoder_requested();auto start=tc::Clock::now();
    auto lease=tc::streaming::SourceLease::capture({{.logical_id="text",.path=root/"text_encoders/qwen3vl_8b_bf16.safetensors"},{.logical_id="tokenizer",.path=root/"processor/tokenizer.json"}});
    tc::Weights weights;weights.load_lease(lease,{"text"},[](const std::string&,int,int){},stop);auto fd=lease->duplicate_fd("tokenizer");tc::Tokenizer tokenizer(fd.get(),lease->file("tokenizer").bytes);
    std::unique_ptr<tc::qwen21::NativeEncoder> created;tc::qwen21::NativeEncoder* native=nullptr;
    if(metal){if(prepared){native=prepared;native->reset_statistics();}else{created=std::make_unique<tc::qwen21::NativeEncoder>();native=created.get();}}
    const auto image_digest=tc::sha256_file(image);auto vision_start=tc::Clock::now();auto pixels=tc::qwen21::resize_reference(tc::load_rgba_image_tensor(image),1024);auto alpha=tc::slice_axis(pixels,3,3,4);auto input=tc::qwen21::vision_input(tc::slice_axis(pixels,3,0,3)*alpha+(1.f-alpha));
    const int expected_vision=components?2:27,expected_language=components?2:36;tc::qwen21::VisionConfig vc;if(components){vc.layers=2;vc.deepstack_layers={0,1};}
    int visual_layers=0;tc::qwen21::VisionFeatures feature{Tensor(0.f),{}};
    auto event=[&](const std::string& stage,int n,int total){if(stage=="qwen21_vision_encode"){need(total==expected_vision,"wrong visual layers");visual_layers=n;}};
    if(native)feature=native->vision(weights,input.patches,input.grid_height,input.grid_width,event,stop,vc);
    else{tc::qwen21::VisionEncoder vision(weights,vc);feature=vision.encode(input.patches,input.grid_height,input.grid_width,event,stop);}
    mx::eval(feature.merged);mx::eval(feature.deepstack);mx::synchronize();const auto vision_seconds=elapsed(vision_start);
    weights.erase_prefix("model.visual.");mx::clear_cache();auto tokens=tokenizer.raw_bounded(tc::qwen21::reference_prompt_template("Edit the reference image. Keep the subject distinct. Warm studio light.",1),tc::Tokenizer::qwen21_limit);
    auto assembled=tc::qwen21::assemble_prompt(tokens,weights.at("model.embed_tokens.weight"),{{std::move(feature),input.grid_height,input.grid_width}});mx::eval(assembled.embeddings);mx::eval(assembled.deepstack_deltas);mx::synchronize();weights.erase("model.embed_tokens.weight");
    auto language_start=tc::Clock::now();tc::qwen21::TextConfig config;config.final_norm=false;if(components)config.layers=2;int language_layers=0;
    auto progress=[&](const std::string& stage,int n,int total){if(stage=="qwen21_text_encode"){need(total==expected_language,"wrong language layers");language_layers=n;}};
    Tensor hidden(0.f);
    if(native)hidden=native->language(weights,assembled.embeddings,assembled.positions,assembled.embeddings.shape(1),assembled.deepstack_deltas,progress,stop,config);
    else{tc::qwen21::TextEncoder encoder(weights,config);hidden=encoder.encode_embeddings(assembled.embeddings,assembled.positions,assembled.embeddings.shape(1),progress,stop,assembled.deepstack_deltas,[&](const std::string& p){weights.erase_prefix(p);});}
    auto retained=assembled.retain(hidden);mx::eval(retained);mx::synchronize();const auto language_seconds=elapsed(language_start),total_seconds=elapsed(start);
    need(visual_layers==expected_vision && language_layers==expected_language,"incomplete encoder");auto f=mx::astype(retained,mx::float32),finite=mx::all(mx::isfinite(f));mx::eval(f,finite);mx::synchronize();need(finite.item<bool>(),"nonfinite conditioning");
    lease->revalidate_after_drain();need(tc::sha256_file(image)==image_digest,"reference changed");
    auto path=out.string()+".conditioning.f32";need(!std::filesystem::exists(path),"existing conditioning");std::ofstream data(path,std::ios::binary);data.write(reinterpret_cast<const char*>(f.data<float>()),f.nbytes());need(bool(data),"conditioning write failed");
    save(out,@{@"status":@"pass",@"scope":components?@"real first-two vision+language layers, reference1024":@"complete single-reference vision27+language36, reference1024, no DiT/VAE",@"completed_vision_layers":@(visual_layers),@"completed_language_layers":@(language_layers),@"metal":@(metal),@"vision_bf16_gemm":@(tc::qwen21::native_encoder_vision_bf16()),@"prepared_reused":@(prepared!=nullptr),@"rows":@(assembled.embeddings.shape(1)),@"vision_seconds":@(vision_seconds),@"language_seconds":@(language_seconds),@"total_encoder_seconds":@(total_seconds),@"retained_elements":@(f.size()),@"conditioning_path":@(path.c_str()),@"native":native?native_stats(native->statistics()):@{},@"source_stat_revalidated":@YES,@"models_downloaded":@0,@"model_exports":@0});
}
}
int main(int argc,char** argv){@autoreleasepool{try{
    char* raw=tc_runtime_build_identity();need(raw,"missing build identity");std::string id(raw);tc_string_free(raw);need(id==TC_EDIT_ENCODER_EXPECTED_BUILD_ID,"wrong linked runtime");
    if(argc==2 && std::string_view(argv[1])=="--cpu-contract"){contracts();return 0;}
    if(argc==3 && std::string_view(argv[1])=="--generated"){generated_native(argv[2]);return 0;}
    if(argc==5 && std::string_view(argv[1])=="--full"){full(argv[2],argv[3],argv[4]);return 0;}
    if(argc==5 && std::string_view(argv[1])=="--components"){full(argv[2],argv[3],argv[4],nullptr,true);return 0;}
    if(argc==5 && std::string_view(argv[1])=="--reuse"){tc::qwen21::NativeEncoder native;for(int n=0;n<2;++n)full(argv[2],std::string(argv[3])+"."+std::to_string(n)+".json",argv[4],&native);save(argv[3],@{@"status":@"pass",@"fresh_source_requests":@2,@"conditioning_cache":@NO});return 0;}
    throw std::invalid_argument("usage --cpu-contract / --generated OUT / --full MODEL OUT IMAGE / --reuse MODEL OUT IMAGE");
}catch(const std::exception& e){std::cerr<<"native_encoder_probe=FAIL "<<e.what()<<'\n';return 1;}}}

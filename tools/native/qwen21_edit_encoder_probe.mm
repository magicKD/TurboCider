// Bounded correctness and local 1/2/3-reference conditioning probe. No ANE,
// checkpoint export, model download, denoising or parameter sweep.
#include "../../native/models/qwen21/encoder_attention.hpp"
#include "../../native/models/qwen21/conditioning.hpp"
#include "../../native/media/image.hpp"
#include "../../native/runtime/streaming/source_lease.hpp"
#include "../../native/platform/apple/bridge.hpp"
#include "../../native/platform/apple/platform.hpp"
#include <turbocider/turbocider.h>
#include <fstream>
#include <iostream>

namespace mx=tc::mx;
using tc::Tensor;
namespace {
void need(bool b,const std::string& s) {tc::require(b,s);}
double elapsed(tc::Clock::time_point t) {return std::chrono::duration<double>(tc::Clock::now()-t).count();}
void save(const std::filesystem::path& p,id value) {
    need(!std::filesystem::exists(p),"receipt must be new");std::ofstream f(p);f<<tc::json(value)<<'\n';need(bool(f),"receipt write failed");
}
NSDictionary* error(const Tensor& a,const Tensor& b,double limit=.02) {
    need(a.shape()==b.shape() && a.dtype()==b.dtype(),"output shape/dtype changed");
    auto af=mx::astype(a,mx::float32),bf=mx::astype(b,mx::float32);
    auto finite=mx::logical_and(mx::all(mx::isfinite(af)),mx::all(mx::isfinite(bf)));
    auto energy=mx::sum(bf*bf),difference=mx::sum((af-bf)*(af-bf));
    mx::eval(finite,energy,difference);mx::synchronize();
    need(finite.item<bool>() && energy.item<float>()>0,"nonfinite or degenerate reference");
    double rrms=std::sqrt(double(difference.item<float>())/energy.item<float>());
    std::cerr<<"rrms="<<rrms<<" limit="<<limit<<" elements="<<a.size()<<'\n';
    need(std::isfinite(rrms) && rrms<=limit,"encoder RRMS exceeds predeclared limit");
    return @{@"rrms":@(rrms),@"limit":@(limit),@"finite":@YES,@"elements":@(a.size())};
}
Tensor signal(mx::Shape shape,int seed,mx::Dtype dtype=mx::bfloat16) {
    size_t n=1;for(int x:shape)n*=x;
    return mx::astype(mx::reshape(mx::sin(mx::arange(int(n),mx::float32)*(.137f+seed*.001f))* .03f,shape),dtype);
}
void cpu() {
    using tc::qwen21::encoder_attention::language_plan;
    for(int n:{1,127,128,1040,2064,3088}) {
        auto p=language_plan(n,n,true);need(p.causal_without_mask && p.grouped_kv==(n>=128),"wrong causal/GQA plan");
        auto old=language_plan(n,n,false);need(!old.causal_without_mask && !old.grouped_kv,"reference changed");
        if(n>1)need(!language_plan(n,n-1,true).causal_without_mask,"padding lost mask");
    }
    for(auto pair:{std::pair{0,0},std::pair{4,0},std::pair{4,5}}) {
        bool rejected=false;try {(void)language_plan(pair.first,pair.second,true);}catch(const std::exception&){rejected=true;}
        need(rejected,"invalid plan accepted");
    }
    std::cout<<"encoder_attention_cpu=PASS device=0 models=0\n";
}
void generated(const std::filesystem::path& out) {
    tc::configure_streams();mx::set_cache_limit(256ULL<<20);
    std::atomic<bool> cancelled=false;NSMutableArray* records=[NSMutableArray array];
    // Full language blocks cover mRoPE, early DeepStack, raw/final norm and
    // padded/unpadded causal keys, not only the attention kernel in isolation.
    tc::Weights weights;std::vector<std::string> names;std::vector<Tensor> values;
    auto bind=[&](std::string n,Tensor v) {names.push_back(std::move(n));values.push_back(std::move(v));};
    bind("model.embed_tokens.weight",signal({1,512},1));bind("model.norm.weight",mx::ones({512},mx::bfloat16));
    for(int i=0;i<2;++i) {
        auto p="model.layers."+std::to_string(i);
        for(auto n:{"input_layernorm","post_attention_layernorm"})bind(p+"."+n+".weight",mx::ones({512},mx::bfloat16));
        for(auto n:{"q_norm","k_norm"})bind(p+".self_attn."+n+".weight",mx::ones({128},mx::bfloat16));
        for(auto n:{"q_proj","o_proj"})bind(p+".self_attn."+n+".weight",signal({512,512},i+2));
        for(auto n:{"k_proj","v_proj"})bind(p+".self_attn."+n+".weight",signal({256,512},i+4));
        for(auto n:{"gate_proj","up_proj"})bind(p+".mlp."+n+".weight",signal({1024,512},i+6));
        bind(p+".mlp.down_proj.weight",signal({512,1024},i+8));
    }
    weights.bind_arrays(names,values);mx::eval(values);mx::synchronize();
    for(int count:{129,257,385})for(bool padded:{false,true}) {
        std::cerr<<"generated language rows="<<count<<" padded="<<padded<<'\n';
        auto x=signal({1,count,512},12);auto index=mx::arange(count,mx::int32);
        auto positions=mx::stack({index,index/16,index%16});
        std::vector<Tensor> deltas{signal({1,count,512},13),signal({1,count,512},14)};
        tc::qwen21::TextConfig c;c.layers=2;c.heads=4;c.kv_heads=2;c.final_norm=padded;
        c.efficient_attention=false;tc::qwen21::TextEncoder old(weights,c);
        c.efficient_attention=true;tc::qwen21::TextEncoder fast(weights,c);
        const int valid=padded?count-7:count;
        auto a=old.encode_embeddings(x,positions,valid,{},cancelled,deltas);
        auto b=fast.encode_embeddings(x,positions,valid,{},cancelled,deltas);
        [records addObject:@{@"kind":@"two-language-blocks",@"rows":@(count),@"padded":@(padded),@"error":error(b,a)}];
    }
    for(auto dtype:{mx::float32,mx::bfloat16})for(int count:{64,192,256}) {
        std::cerr<<"generated visual rows="<<count<<" fp32="<<(dtype==mx::float32)<<'\n';
        auto q=signal({1,16,count,72},21,dtype),k=signal({1,16,count,72},22,dtype),v=signal({1,16,count,72},23,dtype);
        auto a=tc::qwen21::encoder_attention::vision(q,k,v,false);
        auto b=tc::qwen21::encoder_attention::vision(q,k,v,true);
        [records addObject:@{@"kind":@"visual-72-to-80",@"rows":@(count),@"fp32":@(dtype==mx::float32),@"error":error(b,a,.005)}];
    }
    mx::synchronize();save(out,@{@"status":@"pass",@"scope":@"generated; six full language pairs and six visual attention pairs",@"records":records,@"ane_calls":@0});
}
void real(const std::filesystem::path& root,const std::filesystem::path& out,const std::vector<std::filesystem::path>& images) {
    tc::configure_streams();mx::set_cache_limit(256ULL<<20);std::atomic<bool> cancelled=false;
    const auto lease=tc::streaming::SourceLease::capture({{.logical_id="text",.path=root/"text_encoders/qwen3vl_8b_bf16.safetensors"},
        {.logical_id="tokenizer",.path=root/"processor/tokenizer.json"}});
    tc::Weights weights;weights.load_lease(lease,{"text"},[](const std::string&,int,int){},cancelled);
    std::vector<Tensor> visual_sources;uint64_t visual_bytes=0;
    for(const auto& key:weights.sorted_keys())if(key.starts_with("model.visual.")) {
        visual_sources.push_back(weights.at(key));visual_bytes+=visual_sources.back().nbytes();
    }
    need(!visual_sources.empty() && visual_bytes<2ULL<<30,"unexpected vision source budget");
    mx::eval(visual_sources);mx::synchronize();
    std::array<std::vector<tc::qwen21::VisualReference>,2> refs;
    std::array<std::vector<double>,2> vision_seconds;
    std::vector<std::string> hashes;NSMutableArray* visual_records=[NSMutableArray array];
    for(size_t i=0;i<images.size();++i) {
        hashes.push_back(tc::sha256_file(images[i]));
        auto pixels=tc::qwen21::resize_reference(tc::load_rgba_image_tensor(images[i]),1024);
        auto alpha=tc::slice_axis(pixels,3,3,4);
        auto input=tc::qwen21::vision_input(tc::slice_axis(pixels,3,0,3)*alpha+(1.f-alpha));
        mx::eval(input.patches);mx::synchronize();
        std::array<std::optional<tc::qwen21::VisionFeatures>,2> result;
        for(size_t ordinal=0;ordinal<2;++ordinal) {
            const size_t arm=i%2?1-ordinal:ordinal;
            std::cerr<<"real vision image="<<(i+1)<<" arm="<<arm<<" rows="<<input.patches.shape(0)<<std::endl;
            tc::qwen21::VisionConfig config;config.efficient_attention=arm==1;
            tc::qwen21::VisionEncoder encoder(weights,config);
            auto begin=tc::Clock::now();auto feature=encoder.encode(input.patches,input.grid_height,input.grid_width,{},cancelled);
            mx::eval(feature.merged);mx::eval(feature.deepstack);mx::synchronize();
            vision_seconds[arm].push_back(elapsed(begin));result[arm]=std::move(feature);
        }
        NSMutableArray* deep=[NSMutableArray array];
        need(result[0]->deepstack.size()==3 && result[1]->deepstack.size()==3,"DeepStack levels changed");
        for(size_t j=0;j<3;++j)[deep addObject:error(result[1]->deepstack[j],result[0]->deepstack[j],.01)];
        [visual_records addObject:@{@"image":@(i+1),@"patch_rows":@(input.patches.shape(0)),@"order":i%2?@"BA":@"AB",
            @"reference_seconds":@(vision_seconds[0].back()),@"optimized_seconds":@(vision_seconds[1].back()),
            @"merged_error":error(result[1]->merged,result[0]->merged,.01),@"deepstack_error":deep}];
        for(size_t arm=0;arm<2;++arm)refs[arm].push_back({std::move(*result[arm]),input.grid_height,input.grid_width});
    }
    save(out.string()+".visual.json",@{@"status":@"pass",@"visual_pairs":visual_records,
        @"physical_completion":@"actual mx::eval + mx::synchronize",@"ane_calls":@0});
    // Free the vision source tower before gathering all language weights.
    visual_sources.clear();weights.erase_prefix("model.visual.");mx::clear_cache();mx::synchronize();
    std::vector<Tensor> language_sources;uint64_t language_bytes=0;
    for(const auto& key:weights.sorted_keys())if(key.starts_with("model.layers.") || key=="model.embed_tokens.weight") {
        language_sources.push_back(weights.at(key));language_bytes+=language_sources.back().nbytes();
    }
    const auto baseline=tc::observe_process_memory();
    need(baseline.available && baseline.system_available_bytes>language_bytes+(4ULL<<30),"insufficient available memory for bounded resident language comparison");
    need(language_bytes<17ULL<<30,"unexpected language source budget");
    mx::eval(language_sources);mx::synchronize();
    auto fd=lease->duplicate_fd("tokenizer");tc::Tokenizer tokenizer(fd.get(),lease->file("tokenizer").bytes);
    NSMutableArray* pairs=[NSMutableArray array];
    for(size_t n=1;n<=3;++n) {
        const std::string prompt="Arrange the subjects from the reference images on a pale wooden table. Keep each subject distinct. Warm studio light.";
        auto tokens=tokenizer.raw_bounded(tc::qwen21::reference_prompt_template(prompt,n),tc::Tokenizer::qwen21_limit);
        std::array<std::optional<Tensor>,2> outputs;std::array<double,2> text_seconds{};
        int expanded=0;
        for(size_t ordinal=0;ordinal<2;++ordinal) {
            const size_t arm=n==2?1-ordinal:ordinal;
            std::cerr<<"real language references="<<n<<" arm="<<arm<<std::endl;
            std::vector<tc::qwen21::VisualReference> selected(refs[arm].begin(),refs[arm].begin()+n);
            auto assembled=tc::qwen21::assemble_prompt(tokens,weights.at("model.embed_tokens.weight"),selected);
            mx::eval(assembled.embeddings);mx::eval(assembled.deepstack_deltas);mx::synchronize();
            expanded=assembled.embeddings.shape(1);
            tc::qwen21::TextConfig config;config.final_norm=false;config.efficient_attention=arm==1;
            tc::qwen21::TextEncoder encoder(weights,config);
            auto begin=tc::Clock::now();auto result=assembled.retain(encoder.encode_embeddings(assembled.embeddings,
                assembled.positions,expanded,{},cancelled,assembled.deepstack_deltas));
            mx::eval(result);mx::synchronize();text_seconds[arm]=elapsed(begin);outputs[arm]=std::move(result);
        }
        std::array<double,2> visual_total{};
        for(size_t arm=0;arm<2;++arm)for(size_t i=0;i<n;++i)visual_total[arm]+=vision_seconds[arm][i];
        // Preserve the initial failed 2% screen. The user accepts 5-8% tensor
        // screening plus final-image comparison; use its stricter 5% here.
        NSDictionary* comparison=error(*outputs[1],*outputs[0],.05);
        NSDictionary* pair=@{@"reference_images":@(n),@"expanded_tokens":@(expanded),@"order":n==2?@"BA":@"AB",
            @"reference_text_seconds":@(text_seconds[0]),@"optimized_text_seconds":@(text_seconds[1]),
            @"reference_vision_sum_seconds":@(visual_total[0]),@"optimized_vision_sum_seconds":@(visual_total[1]),
            @"conditioning_error":comparison,@"strict_2_percent_pass":@([comparison[@"rrms"] doubleValue]<=.02)};
        [pairs addObject:pair];
        mx::synchronize();lease->revalidate_after_drain();
        save(out.string()+".language-"+std::to_string(n)+".json",@{@"status":@"pass-user-5-percent-screen",@"pair":pair,
            @"source_stat_revalidated":@YES,@"physical_completion":@"actual mx::eval + mx::synchronize"});
    }
    for(size_t i=0;i<images.size();++i)need(tc::sha256_file(images[i])==hashes[i],"reference content changed");
    NSMutableArray* source_hashes=[NSMutableArray array];for(const auto& h:hashes)[source_hashes addObject:@(h.c_str())];
    save(out,@{@"status":@"pass",@"scope":@"local 1024px visual towers and 1/2/3-image language conditioning; no VAE/DiT/ANE",
        @"visual_pairs":visual_records,@"conditioning_pairs":pairs,@"visual_source_bytes":@(visual_bytes),
        @"language_source_bytes":@(language_bytes),@"language_sources_preloaded":@YES,@"cache_limit_bytes":@(256ULL<<20),
        @"source_stat_revalidated":@YES,@"reference_sha256":source_hashes,@"physical_completion":@"actual mx::eval + mx::synchronize",
        @"timing_scope":@"one sample/order per image/count; source loading, resizing and assembly excluded; first real-shape compile included; no stable performance qualification",
        @"quality_scope":@"5% user tensor screen; strict initial 2% failure retained; final-image acceptance is separate",
        @"system_available_before_language_bytes":@(baseline.system_available_bytes),@"mlx_peak_bytes":@(mx::get_peak_memory())});
}
}
int main(int argc,char** argv) {
    @autoreleasepool {try {
        char* identity=tc_runtime_build_identity();need(identity,"missing runtime identity");
        const std::string build(identity);tc_string_free(identity);
        need(build==TC_EDIT_ENCODER_EXPECTED_BUILD_ID,"loaded library differs from explicitly linked runtime");
        if(argc==1 || (argc==2 && std::string_view(argv[1])=="--cpu-contract")) {cpu();return 0;}
        need(argc>=3,"explicit device mode required");
        if(std::string_view(argv[1])=="--generated" && argc==3)generated(argv[2]);
        else if(std::string_view(argv[1])=="--real" && argc==7)real(argv[2],argv[3],{argv[4],argv[5],argv[6]});
        else throw std::invalid_argument("usage: --generated OUTPUT or --real MODEL OUTPUT REF1 REF2 REF3");
        std::cout<<"encoder_probe=PASS\n";return 0;
    }catch(const std::exception& e) {std::cerr<<"encoder_probe=FAIL "<<e.what()<<'\n';return 1;}}
}

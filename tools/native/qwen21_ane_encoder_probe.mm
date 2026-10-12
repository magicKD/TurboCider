// Native experiment only: existing local 1..3 images, layer-0 real language
// FFN inputs, two finite channel candidates. No model export or denoising.
#define main qwen21_gpu_encoder_reference_main
#include "qwen21_edit_encoder_probe.mm"
#undef main
#include "../../native/models/qwen21/encoder_runtime.hpp"
#include "../../native/backends/private/ane_mil.hpp"

namespace {
NSDictionary* timing_error(const Tensor& value,const Tensor& reference) {return error(value,reference,.05);}
NSDictionary* graph_stats(const tc::qwen21::EncoderRuntimeStats& s) {
    return @{@"gpu_graph_builds":@(s.gpu_graph_builds),@"gpu_graph_calls":@(s.gpu_graph_calls),
        @"input_ready_seconds":@(s.input_ready_seconds),@"gpu_graph_encode_host_seconds":@(s.gpu_graph_encode_host_seconds),
        @"preparation_reused":@(s.preparation_reused),@"bucket_rows":@(s.bucket_rows),
        @"next_source_submissions":@(s.next_source_submissions),@"next_source_skips":@(s.next_source_skips)};
}
void generated_bridge(const std::filesystem::path& out) {
    tc::configure_streams();mx::set_cache_limit(256ULL<<20);NSMutableArray* records=[NSMutableArray array];
    // Physical pitch/range, FP32 partial, and changing dynamic weights all
    // matter here. A compiled graph must not reuse the first layer's weights.
    for(auto dtype:{mx::bfloat16,mx::float16}) {
        auto fn=mx::compile([](const std::vector<Tensor>& a) {
            return std::vector<Tensor>{tc::dense_gpu::projection_range(a[0],a[1],0,128,256,512,32,true)};
        });
        auto x=signal({1,129,256},31,dtype);std::optional<Tensor> previous;
        for(int seed:{32,47}) {
            auto w=signal({128,512},seed,dtype);
            auto reference=mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(tc::slice_axis(w,1,256,512),mx::float32)));
            auto actual=fn({x,w})[0];auto diff=error(actual,reference,.0001);
            need(actual.dtype()==mx::float32,"MPP partial narrowed before join");
            if(previous) {auto changed=mx::any(actual!=*previous);mx::eval(changed);need(changed.item<bool>(),"dynamic weights reused stale compiled constants");}
            previous=actual;
            [records addObject:@{@"seed":@(seed),@"bf16":@(dtype==mx::bfloat16),@"error":diff}];
        }
    }
    mx::synchronize();save(out,@{@"status":@"pass",@"scope":@"BF16/FP16 physical range and dynamic weight rebind, FP32 partial",@"records":records,@"ane_calls":@0});
}
void policy_cpu() {
    using namespace tc::qwen21::encoder_runtime;
    tc::Request r;r.model="qwen-image-2.1";r.operation="image.edit";r.execution="gpu";r.width=r.height=512;
    r.allow_approximation=true;size_t checks=0;
    for(int count=0;count<=4;++count)for(int size:{512,1024,768}) {
        r.inputs.resize(count);r.qwen21_reference_size=size;
        need(supported_request(r)==(count>=1 && count<=3 && size!=768),"wrong encoder request domain");++checks;
    }
    r.inputs.resize(1);r.qwen21_reference_size=1024;
    r.allow_approximation=false;need(!supported_request(r),"approximate encoder allowed without opt-in");r.allow_approximation=true;++checks;
    r.prompt_enhance=true;need(!supported_request(r),"prompt enhancement entered experimental encoder");r.prompt_enhance=false;++checks;
    for(auto field:{&r.execution,&r.operation,&r.model,&r.hybrid_mlp_mode,&r.qwen21_denoiser_backend}) {
        auto old=*field;*field="unsupported";need(!supported_request(r),"invalid route accepted");*field=old;++checks;
    }
    for(int rows:{1,1023,1024,1074,2104,3134,32768}){need(eligible_rows(rows)==(rows>=1024),"wrong row threshold");++checks;}
    for(int rows:{0,-1,32769}){bool rejected=false;try{(void)eligible_rows(rows);}catch(const std::exception&){rejected=true;}need(rejected,"invalid rows accepted");++checks;}
    for(int rows=512;rows<=2048;rows+=128){validate_bucket(rows);++checks;}
    for(int rows:{0,128,511,513,2050,2176}){bool rejected=false;try{validate_bucket(rows);}catch(const std::exception&){rejected=true;}need(rejected,"invalid program bucket accepted");++checks;}
    const char* bucket_key="TURBOCIDER_QWEN21_ENCODER_RUNTIME_BUCKET";
    const char* original=std::getenv(bucket_key);const std::optional<std::string> saved=original?std::optional<std::string>(original):std::nullopt;
    setenv(bucket_key,"balanced",1);
    for(auto [rows,want]:{std::pair{1071,1152},std::pair{2101,1152},std::pair{3131,1664},std::pair{3328,1664}}) {
        need(select_bucket(rows,1152)==want,"wrong chunk/padding selection");++checks;
    }
    bool missing_rows=false;try{(void)select_bucket(0,1152);}catch(const std::exception&){missing_rows=true;}
    need(missing_rows,"balanced bucket admitted missing rows");++checks;
    setenv(bucket_key,"invalid",1);bool invalid_bucket=false;try{(void)select_bucket(3131,1152);}catch(const std::exception&){invalid_bucket=true;}
    need(invalid_bucket,"invalid bucket silently selected");++checks;
    if(saved)setenv(bucket_key,saved->c_str(),1);else unsetenv(bucket_key);
    cpu();std::cout<<"encoder_runtime_policy=PASS checks="<<checks<<" device=0 models=0\n";
}
void screen(const std::filesystem::path& root,const std::filesystem::path& out,
            const std::filesystem::path& manifest,const std::vector<std::filesystem::path>& images) {
    tc::configure_streams();mx::set_cache_limit(256ULL<<20);std::atomic<bool> cancelled=false;
    auto observation=tc::observe_process_memory();need(observation.available && observation.system_available_bytes>8ULL<<30,"insufficient memory for bounded encoder component");
    auto lease=tc::streaming::SourceLease::capture({{.logical_id="text",.path=root/"text_encoders/qwen3vl_8b_bf16.safetensors"},
        {.logical_id="tokenizer",.path=root/"processor/tokenizer.json"}});
    tc::Weights weights;weights.load_lease(lease,{"text"},[](const std::string&,int,int){},cancelled);
    std::vector<tc::qwen21::VisualReference> refs;std::vector<std::string> hashes;
    tc::qwen21::VisionEncoder vision(weights);
    for(const auto& image:images) {
        hashes.push_back(tc::sha256_file(image));
        auto p=tc::qwen21::resize_reference(tc::load_rgba_image_tensor(image),1024);auto alpha=tc::slice_axis(p,3,3,4);
        auto input=tc::qwen21::vision_input(tc::slice_axis(p,3,0,3)*alpha+(1.f-alpha));
        auto feature=vision.encode(input.patches,input.grid_height,input.grid_width,{},cancelled);
        mx::eval(feature.merged);mx::eval(feature.deepstack);mx::synchronize();
        refs.push_back({std::move(feature),input.grid_height,input.grid_width});
    }
    weights.erase_prefix("model.visual.");mx::clear_cache();
    auto fd=lease->duplicate_fd("tokenizer");tc::Tokenizer tokenizer(fd.get(),lease->file("tokenizer").bytes);
    std::vector<Tensor> inputs;
    for(size_t n=1;n<=3;++n) {
        auto tokens=tokenizer.raw_bounded(tc::qwen21::reference_prompt_template("Arrange the reference subjects on a pale wooden table. Keep each subject distinct. Warm studio light.",n),tc::Tokenizer::qwen21_limit);
        std::vector<tc::qwen21::VisualReference> selected(refs.begin(),refs.begin()+n);
        auto assembled=tc::qwen21::assemble_prompt(tokens,weights.at("model.embed_tokens.weight"),selected);
        std::optional<Tensor> captured;
        tc::qwen21::TextFfnHooks capture{[](int,const std::string&,int){},[&](int layer,const Tensor& value,const tc::qwen21::TextFfnHooks::Gpu& gpu){
            need(layer==0,"capture layer changed");mx::eval(value);mx::synchronize();captured=value;return gpu(value);}};
        tc::qwen21::TextConfig config;config.layers=1;config.final_norm=false;
        tc::qwen21::TextEncoder encoder(weights,config);
        // The first FFN's input precedes the first DeepStack residual. One
        // level suffices for this one-layer capture; no shortened full encoding.
        auto result=encoder.encode_embeddings(assembled.embeddings,assembled.positions,assembled.embeddings.shape(1),{},cancelled,
            {assembled.deepstack_deltas.front()},{},&capture);
        mx::eval(result);mx::synchronize();need(bool(captured),"missing real FFN input");inputs.push_back(*captured);
    }
    refs.clear();weights.erase("model.embed_tokens.weight");
    const std::string prefix="model.layers.0";
    auto eager=[&](const Tensor& value) {
        return tc::linear(tc::silu(tc::linear(value,weights,prefix+".mlp.gate_proj"))*tc::linear(value,weights,prefix+".mlp.up_proj"),weights,prefix+".mlp.down_proj");
    };
    auto full=mx::compile([&](const std::vector<Tensor>& a){return std::vector<Tensor>{eager(a[0])};});
    auto gpu=[&](const Tensor& value){return full({value})[0];};
    NSMutableArray* candidates=[NSMutableArray array];
    // Program/source preparations are reported separately, not hidden in
    // warm timings; each measured hybrid includes stage+input+GPU+ANE+join.
    for(int channels:{3072,6144}) {
        setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS",std::to_string(channels).c_str(),1);
        std::cerr<<"candidate channels="<<channels<<std::endl;
        auto start=tc::Clock::now();tc::qwen21::EncoderRuntime bridge(weights,manifest,1ULL<<30,cancelled);
        const auto setup=elapsed(start);NSMutableArray* pairs=[NSMutableArray array];
        for(size_t j=0;j<inputs.size();++j) {
            const auto& input=inputs[j];const int rows=input.shape(1);
            auto hybrid=[&]{bridge.prepare(0,prefix,rows);return bridge.execute(0,input,gpu);};
            // One warm pair for this shape; no sweep or repeated request.
            auto oracle=eager(input);auto warm_gpu=gpu(input);mx::eval(oracle,warm_gpu);mx::synchronize();
            auto gpu_error=timing_error(warm_gpu,oracle);
            auto warm=hybrid();mx::eval(warm);mx::synchronize();auto warm_error=timing_error(warm,oracle);
            for(int order=0;order<2;++order) {
                std::array<double,2> seconds{};std::array<std::optional<Tensor>,2> outputs;
                for(int ordinal=0;ordinal<2;++ordinal) {
                    const int arm=order?1-ordinal:ordinal;mx::synchronize();auto t=tc::Clock::now();
                    auto result=arm?hybrid():gpu(input);mx::eval(result);mx::synchronize();seconds[arm]=elapsed(t);outputs[arm]=std::move(result);
                }
                auto diff=timing_error(*outputs[1],*outputs[0]);
                NSDictionary* record=@{@"references":@(j+1),@"rows":@(rows),@"order":order?@"BA":@"AB",
                    @"gpu_seconds":@(seconds[0]),@"hybrid_seconds":@(seconds[1]),@"saving_percent":@(100*(seconds[0]-seconds[1])/seconds[0]),
                    @"error":diff,@"gpu_compile_vs_original":gpu_error,@"warm_error":warm_error};
                [pairs addObject:record];
                save(out.string()+".c"+std::to_string(channels)+".n"+std::to_string(j+1)+"."+std::to_string(order)+".json",record);
                std::cerr<<"complete refs="<<(j+1)<<" order="<<order<<" GPU="<<seconds[0]<<" hybrid="<<seconds[1]<<std::endl;
            }
        }
        auto metrics=bridge.finish();need(metrics.runtime_weight_hybrid_blocks==9 && metrics.runtime_failures==0 && metrics.runtime_weight_fallback_blocks==0,
            "not all nine component FFNs ran the hybrid path");
        [candidates addObject:@{@"ane_channels":@(channels),@"setup_seconds":@(setup),@"pairs":pairs,
            @"gpu_recipe":@(tc::qwen21::encoder_runtime::gpu_recipe_identity().c_str()),@"graph_stats":graph_stats(bridge.statistics()),
            @"stage_specialized":@(metrics.runtime_weight_stage_specialized),@"source_recipe":@(metrics.runtime_weight_source_recipe.c_str()),
            @"successful_ffns":@(metrics.runtime_weight_hybrid_blocks),@"ane_evaluations":@(metrics.runtime_calls),
            @"stage_seconds":@(metrics.runtime_weight_stage_seconds),@"stage_wait_seconds":@(metrics.runtime_weight_stage_wait_seconds),
            @"gpu_submit_seconds":@(metrics.runtime_weight_gpu_seconds),
            @"async_ane_host_wait_seconds":@(metrics.runtime_weight_async_ane_wait_seconds),@"post_join_seconds":@(metrics.runtime_weight_post_join_seconds),
            @"estimated_slot_bytes":@(metrics.runtime_weight_estimated_bytes),@"owner_destroyed_after_drain":@(metrics.runtime_weight_session_released)}];
    }
    lease->revalidate_after_drain();for(size_t i=0;i<images.size();++i)need(tc::sha256_file(images[i])==hashes[i],"reference content drift");
    save(out,@{@"status":@"pass",@"scope":@"real layer-0 prefill FFN for 1..3 references; two candidates, warm+AB+BA per shape; no full encoder/denoising qualification",
        @"candidates":candidates,@"source_stat_revalidated":@YES,@"physical_completion":@"mx::eval+synchronize; existing private executor finish/drain before owner release",
        @"cache_limit_bytes":@(256ULL<<20),@"mlx_peak_bytes":@(mx::get_peak_memory()),@"downloaded_models":@0,@"model_exports":@0});
}
void encode_full(const std::filesystem::path& root,const std::filesystem::path& out,
                 const std::filesystem::path& manifest,int channels,const std::vector<std::filesystem::path>& images,
                 tc::qwen21::EncoderRuntimeCache* cache=nullptr,bool calibrate=false) {
    need(images.size()==3 && (channels==0 || channels==6144),"full screen only covers selected three-reference candidate");
    tc::configure_streams();mx::set_cache_limit(256ULL<<20);std::atomic<bool> cancelled=false;
    auto start=tc::Clock::now();
    std::vector<tc::streaming::SourceFileIdentity> sources{{.logical_id="text",.path=root/"text_encoders/qwen3vl_8b_bf16.safetensors"},
        {.logical_id="tokenizer",.path=root/"processor/tokenizer.json"}};
    if(channels)sources.push_back({.logical_id="descriptor",.path=manifest});
    auto lease=tc::streaming::SourceLease::capture(sources);tc::Weights weights;
    weights.load_lease(lease,{"text"},[](const std::string&,int,int){},cancelled);
    auto fd=lease->duplicate_fd("tokenizer");tc::Tokenizer tokenizer(fd.get(),lease->file("tokenizer").bytes);
    const auto load_seconds=elapsed(start);auto visual_start=tc::Clock::now();
    std::vector<tc::qwen21::VisualReference> refs;tc::qwen21::VisionEncoder vision(weights);
    std::vector<std::string> hashes;
    for(const auto& image:images) {
        hashes.push_back(tc::sha256_file(image));auto p=tc::qwen21::resize_reference(tc::load_rgba_image_tensor(image),1024);
        auto alpha=tc::slice_axis(p,3,3,4);auto input=tc::qwen21::vision_input(tc::slice_axis(p,3,0,3)*alpha+(1.f-alpha));
        auto feature=vision.encode(input.patches,input.grid_height,input.grid_width,{},cancelled);
        mx::eval(feature.merged);mx::eval(feature.deepstack);mx::synchronize();
        refs.push_back({std::move(feature),input.grid_height,input.grid_width});
    }
    const double visual_seconds=elapsed(visual_start);
    // Identical bounded phase policy in both arms; no resident full checkpoint.
    weights.erase_prefix("model.visual.");mx::clear_cache();auto assembly_start=tc::Clock::now();
    auto tokens=tokenizer.raw_bounded(tc::qwen21::reference_prompt_template("Arrange the reference subjects on a pale wooden table. Keep each subject distinct. Warm studio light.",images.size()),tc::Tokenizer::qwen21_limit);
    auto assembled=tc::qwen21::assemble_prompt(tokens,weights.at("model.embed_tokens.weight"),refs);
    mx::eval(assembled.embeddings);mx::eval(assembled.deepstack_deltas);mx::synchronize();weights.erase("model.embed_tokens.weight");
    const double assembly_seconds=elapsed(assembly_start);const int rows=assembled.embeddings.shape(1);
    std::unique_ptr<tc::qwen21::EncoderRuntime> bridge;std::optional<tc::qwen21::TextFfnHooks> hooks;
    std::unique_ptr<tc::qwen21::EncoderCalibration> calibration;
    std::vector<std::pair<int,Tensor>> samples;std::vector<std::string> sample_names;std::vector<Tensor> sample_values;
    std::optional<tc::qwen21::EncoderScaleProfile> profile;
    auto prep_start=tc::Clock::now();
    if(channels){
        setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","6144",1);
        auto profile_path=tc::qwen21::encoder_profile_path("TURBOCIDER_QWEN21_ENCODER_S1_PROFILE");
        if(!profile_path.empty())profile=tc::qwen21::EncoderScaleProfile::load(profile_path,root/"text_encoders/qwen3vl_8b_bf16.safetensors",3,1024,6144);
        bridge=std::make_unique<tc::qwen21::EncoderRuntime>(weights,manifest,1ULL<<30,cancelled,rows,cache,profile?&*profile:nullptr);hooks=bridge->hooks();
    }
    if(calibrate) {
        const auto existing=tc::qwen21::encoder_profile_path("TURBOCIDER_ENCODER_PROBE_EXISTING_PROFILE");
        if(!existing.empty())profile=tc::qwen21::EncoderScaleProfile::load(existing,root/"text_encoders/qwen3vl_8b_bf16.safetensors",3,1024,6144);
        std::string joined;for(const auto& h:hashes)joined+=h+";";
        if(!profile)calibration=std::make_unique<tc::qwen21::EncoderCalibration>(weights,root/"text_encoders/qwen3vl_8b_bf16.safetensors",3,1024,6144,joined);
        hooks=tc::qwen21::TextFfnHooks{[](int,const std::string&,int){},[&](int layer,const Tensor& x,const tc::qwen21::TextFfnHooks::Gpu& gpu) {
            const auto p="model.layers."+std::to_string(layer);if(calibration)calibration->observe(layer,p,x);
            if(layer==0 || layer==17 || layer==35) {
                samples.emplace_back(layer,x);
                for(const char* n:{"gate_proj","up_proj","down_proj"}) {
                    auto key=p+".mlp."+n+".weight";sample_names.push_back(key);sample_values.push_back(weights.at(key));
                }
            }
            return gpu(x);
        }};
    }
    const double setup_seconds=elapsed(prep_start);auto language_start=tc::Clock::now();int completed=0;
    tc::qwen21::TextConfig config;config.final_norm=false;tc::qwen21::TextEncoder encoder(weights,config);
    auto text=assembled.retain(encoder.encode_embeddings(assembled.embeddings,assembled.positions,rows,
        [&](const std::string& stage,int done,int total){if(stage=="qwen21_text_encode"){completed=done;need(total==36,"wrong full layer count");}},
        cancelled,assembled.deepstack_deltas,[&](const std::string& p){weights.erase_prefix(p);},hooks?&*hooks:nullptr));
    mx::eval(text);mx::synchronize();std::optional<tc::HybridMetrics> metrics;tc::qwen21::EncoderRuntimeStats statistics;
    if(bridge){metrics=bridge->finish();statistics=bridge->statistics();bridge.reset();}
    const double language_seconds=elapsed(language_start),total_seconds=elapsed(start);
    need(completed==36 && text.dtype()==mx::bfloat16 && text.shape(0)==1 && text.shape(2)==4096,"incomplete full encoder output");
    if(metrics)need(metrics->runtime_weight_hybrid_blocks==36 && metrics->runtime_failures==0 && metrics->runtime_weight_fallback_blocks==0,
        "full encoder did not execute all 36 hybrid FFNs");
    auto value=mx::astype(text,mx::float32),finite=mx::all(mx::isfinite(value));mx::eval(value,finite);mx::synchronize();need(finite.item<bool>(),"nonfinite retained conditioning");
    lease->revalidate_after_drain();for(size_t i=0;i<images.size();++i)need(tc::sha256_file(images[i])==hashes[i],"reference changed");
    if(calibrate) {
        const auto existing=tc::qwen21::encoder_profile_path("TURBOCIDER_ENCODER_PROBE_EXISTING_PROFILE");
        const auto profile_path=existing.empty()?std::filesystem::path(out.string()+".s1.safetensors"):existing;
        if(calibration)calibration->write(profile_path);
        weights.clear();refs.clear();assembled.deepstack_deltas.clear();mx::clear_cache();
        tc::Weights sampled;sampled.bind_arrays(sample_names,sample_values);
        profile=tc::qwen21::EncoderScaleProfile::load(profile_path,root/"text_encoders/qwen3vl_8b_bf16.safetensors",3,1024,6144);
        NSMutableArray* records=[NSMutableArray array];
        setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","6144",1);
        for(int recipe:{0,1,2}) {
            const bool scaled=recipe==2;
            setenv("TURBOCIDER_QWEN21_ENCODER_RUNTIME_BUCKET",recipe==0?"1152":"1664",1);
            const auto setup_start=tc::Clock::now();
            tc::qwen21::EncoderRuntimeCache prepared_cache;
            setenv("TURBOCIDER_QWEN21_ENCODER_RUNTIME_REUSE","1",1);
            tc::qwen21::EncoderRuntime component(sampled,manifest,1ULL<<30,cancelled,rows,&prepared_cache,scaled?&*profile:nullptr);
            const auto component_setup=elapsed(setup_start);
            for(const auto& [layer,x]:samples) {
                const auto p="model.layers."+std::to_string(layer);
                auto gpu=[&](const Tensor& v){return tc::linear(tc::silu(tc::linear(v,sampled,p+".mlp.gate_proj"))*tc::linear(v,sampled,p+".mlp.up_proj"),sampled,p+".mlp.down_proj");};
                auto hybrid=[&]{component.prepare(layer,p,rows);return component.execute(layer,x,gpu);};
                auto a=gpu(x),b=hybrid();mx::eval(a,b);mx::synchronize();
                for(int order=0;order<2;++order) {
                    std::array<double,2> timings{};std::array<std::optional<Tensor>,2> values;
                    for(int ordinal=0;ordinal<2;++ordinal) {
                        const int arm=order?1-ordinal:ordinal;mx::synchronize();auto t=tc::Clock::now();
                        auto v=arm?hybrid():gpu(x);mx::eval(v);mx::synchronize();timings[arm]=elapsed(t);values[arm]=v;
                    }
                    auto diff=error(*values[1],*values[0],1.0);const bool numeric_pass=[diff[@"rrms"] doubleValue]<=.05;
                    NSDictionary* record=@{@"s1":@(scaled),@"bucket_rows":@(component.statistics().bucket_rows),@"layer":@(layer),@"order":order?@"BA":@"AB",@"rows":@(rows),
                        @"gpu_seconds":@(timings[0]),@"hybrid_seconds":@(timings[1]),@"saving_percent":@(100*(timings[0]-timings[1])/timings[0]),
                        @"error":diff,@"screen_rrms_limit":@.05,@"numeric_pass":@(numeric_pass),@"setup_seconds":@(component_setup)};
                    [records addObject:record];save(out.string()+".r"+std::to_string(recipe)+".l"+std::to_string(layer)+".o"+std::to_string(order)+".json",record);
                }
            }
            auto m=component.finish();need(m.runtime_weight_hybrid_blocks==9 && m.runtime_failures==0,"component execution incomplete");
            if(recipe==0) {
                std::vector<Tensor> rebound=sample_values;
                for(size_t n=0;n<sample_names.size();++n)if(sample_names[n]=="model.layers.0.mlp.up_proj.weight")rebound[n]=mx::astype(rebound[n]*.5f,mx::bfloat16);
                tc::Weights changed;changed.bind_arrays(sample_names,rebound);mx::eval(rebound);mx::synchronize();
                tc::qwen21::EncoderRuntime reused(changed,manifest,1ULL<<30,cancelled,rows,&prepared_cache);
                const auto& x=samples.front().second;const std::string p="model.layers.0";
                auto gpu=[&](const Tensor& v){return tc::linear(tc::silu(tc::linear(v,changed,p+".mlp.gate_proj"))*tc::linear(v,changed,p+".mlp.up_proj"),changed,p+".mlp.down_proj");};
                reused.prepare(0,p,rows);auto actual=reused.execute(0,x,gpu),expected=gpu(x);mx::eval(actual,expected);mx::synchronize();
                auto diff=error(actual,expected,.05);auto rm=reused.finish();auto rs=reused.statistics();
                need(rs.preparation_reused && rs.gpu_graph_builds==0 && rm.runtime_weight_hybrid_blocks==1,"prepared cache was rebuilt or counters not reset");
                save(out.string()+".cache-rebind.json",@{@"status":@"pass",@"up_weight_multiplier":@.5,@"error":diff,@"graph_stats":graph_stats(rs),@"hybrid_ffns":@(rm.runtime_weight_hybrid_blocks)});
            }
            prepared_cache.clear();
        }
        save(out.string()+".calibration-screen.json",@{@"status":@"pass",@"scope":@"36-layer GPU-only calibration plus real layers 0/17/35 complete FFN AB/BA; capture timing excluded from baseline",
            @"profile_path":@(profile_path.c_str()),@"profile_sha256":@(profile->digest.c_str()),@"calibration_layers":@36,@"records":records,
            @"quality_limit":@.05,@"minimum_saving_percent":@5,@"profile_qualified":@NO,@"models_downloaded":@0,@"model_exports":@0});
        mx::synchronize();lease->revalidate_after_drain();
    }
    const auto data_path=out.string()+".conditioning.f32";need(!std::filesystem::exists(data_path),"existing conditioning receipt");
    std::ofstream data(data_path,std::ios::binary);data.write(reinterpret_cast<const char*>(value.data<float>()),value.nbytes());need(bool(data),"cannot write small retained conditioning");data.close();
    save(out,@{@"status":@"pass",@"scope":@"complete 27-layer vision x3 and 36-layer language encoder; no DiT/VAE/image qualification",
        @"ane_channels":@(channels),@"rows":@(rows),@"retained_rows":@(text.shape(1)),@"completed_layers":@(completed),
        @"total_encoder_seconds":@(total_seconds),@"load_seconds":@(load_seconds),@"vision_seconds":@(visual_seconds),
        @"assembly_seconds":@(assembly_seconds),@"program_setup_seconds":@(setup_seconds),@"language_seconds":@(language_seconds),
        @"hybrid_ffns":@(metrics?metrics->runtime_weight_hybrid_blocks:0),@"ane_evaluations":@(metrics?metrics->runtime_calls:0),
        @"gpu_recipe":channels?@(tc::qwen21::encoder_runtime::gpu_recipe_identity().c_str()):@"full-gpu-original",
        @"graph_stats":graph_stats(statistics),@"stage_specialized":@(metrics?metrics->runtime_weight_stage_specialized:false),
        @"stage_seconds":@(metrics?metrics->runtime_weight_stage_seconds:0),@"stage_wait_seconds":@(metrics?metrics->runtime_weight_stage_wait_seconds:0),
        @"async_ane_host_wait_seconds":@(metrics?metrics->runtime_weight_async_ane_wait_seconds:0),@"post_join_seconds":@(metrics?metrics->runtime_weight_post_join_seconds:0),
        @"pre_seconds":@(metrics?metrics->runtime_weight_pre_seconds:0),
        @"prefetch_submissions":@(metrics?metrics->runtime_weight_prefetch_submissions:0),@"prefetch_hits":@(metrics?metrics->runtime_weight_prefetch_hits:0),
        @"prefetch_failures":@(metrics?metrics->runtime_weight_prefetch_failures:0),@"prefetch_wait_seconds":@(metrics?metrics->runtime_weight_prefetch_wait_seconds:0),
        @"s1_hybrid_ffns":@(metrics?metrics->runtime_weight_s1_hybrid_blocks:0),
        @"physical_completion":@"eval+synchronize; private finish/drain; phase owner destroyed before source generation check",
        @"source_stat_revalidated":@YES,@"retained_conditioning_path":@(data_path.c_str()),@"conditioning_elements":@(value.size()),
        @"mlx_peak_bytes":@(mx::get_peak_memory()),@"models_downloaded":@0,@"model_exports":@0});
}
}
int main(int argc,char** argv) {
 @autoreleasepool {try {
    char* raw=tc_runtime_build_identity();need(raw,"missing runtime identity");std::string id(raw);tc_string_free(raw);
    need(id==TC_EDIT_ENCODER_EXPECTED_BUILD_ID,"wrong linked runtime");
    if(argc==2 && std::string_view(argv[1])=="--cpu-contract"){policy_cpu();return 0;}
    if(argc==3 && std::string_view(argv[1])=="--shape-contract") {
        auto descriptor=tc::ane::runtime_template_descriptor(argv[2]);
        need(!descriptor.has_public_artifact(),"expected shape-only descriptor");
        for(int channels:{3072,6144})for(int bucket:{1152,1664}) {
            auto shape=descriptor.shape;shape.rows=bucket;tc::qwen21::encoder_runtime::validate_bucket(shape.rows);shape.width=channels;
            auto spec=tc::ane::private_api::w8_swiglu_program(shape,20260930,1.f);
            need(!spec.mil.empty() && !spec.constants.empty(),"incomplete W8A8 MIL geometry");
        }
        std::cout<<"encoder_shape=PASS device=0 models=0\n";return 0;
    }
    if(argc==3 && std::string_view(argv[1])=="--generated"){generated(argv[2]);return 0;}
    if(argc==3 && std::string_view(argv[1])=="--generated-bridge"){generated_bridge(argv[2]);return 0;}
    if(argc==8 && std::string_view(argv[1])=="--calibrate-screen") {
        encode_full(argv[2],argv[3],argv[4],0,{argv[5],argv[6],argv[7]},nullptr,true);return 0;
    }
    if(argc==9 && std::string_view(argv[1])=="--encode-reuse") {
        tc::qwen21::EncoderRuntimeCache cache;
        for(int n=0;n<2;++n)encode_full(argv[2],std::string(argv[3])+"."+std::to_string(n)+".json",argv[4],std::stoi(argv[5]),{argv[6],argv[7],argv[8]},&cache);
        cache.clear();mx::synchronize();
        save(argv[3],@{@"status":@"pass",@"scope":@"two complete encoders with fresh weights/source leases; one prepared cache; no conditioning cache",@"requests":@2,@"cache_evicted":@YES});return 0;
    }
    if(argc==9 && std::string_view(argv[1])=="--encode-full") {
        encode_full(argv[2],argv[3],argv[4],std::stoi(argv[5]),{argv[6],argv[7],argv[8]});return 0;
    }
    need(argc==8 && std::string_view(argv[1])=="--ffn-screen","usage: --ffn-screen MODEL OUT MANIFEST REF1 REF2 REF3");
    screen(argv[2],argv[3],argv[4],{argv[5],argv[6],argv[7]});return 0;
 }catch(const std::exception& e){std::cerr<<"ane_encoder_probe=FAIL "<<e.what()<<std::endl;return 1;}}
}

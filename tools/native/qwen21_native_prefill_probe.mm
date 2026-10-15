#define main qwen21_gpu_encoder_reference_main
#include "qwen21_edit_encoder_probe.mm"
#undef main
#include "../../native/backends/metal_graph/native_prefill_ffn.hpp"
#include "../../native/backends/metal_graph/mlx_bridge.hpp"
#include "../../native/models/qwen21/pipeline.hpp"
#include "../../native/models/qwen21/text_encoder.hpp"

namespace {
namespace mg=tc::metal_graph;
NSDictionary* metrics(const mg::NativePrefillFfnStats& s){return @{@"calls":@(s.calls),@"chunks":@(s.chunks),@"valid_rows":@(s.valid_rows),@"padded_rows":@(s.padded_rows),@"producer_seconds":@(s.producer_seconds),@"branch_seconds":@(s.branch_seconds),@"consumer_seconds":@(s.consumer_seconds),@"total_seconds":@(s.total_seconds)};}
mg::MatrixView view(mg::Backend& b,const Tensor& t){auto x=mg::import_mlx_matrix(b,t);return {x.storage,x.rows,x.columns,x.row_stride_bytes};}
Tensor allocation(int rows,int cols){return mg::allocate_mlx_matrix(rows,cols,mg::ScalarType::BF16,true);}
NSDictionary* difference(const Tensor& a,const Tensor& b){
    auto af=mx::astype(a,mx::float32),bf=mx::astype(b,mx::float32);
    auto finite=mx::logical_and(mx::all(mx::isfinite(af)),mx::all(mx::isfinite(bf))),e=mx::sum(bf*bf),d=mx::sum((af-bf)*(af-bf));mx::eval(finite,e,d);mx::synchronize();
    need(finite.item<bool>() && e.item<float>()>0,"nonfinite/degenerate prefill output");const double rrms=std::sqrt(double(d.item<float>())/e.item<float>());
    return @{@"rrms":@(rrms),@"finite":@YES,@"elements":@(a.size())};
}
void screen(const Tensor& x,const Tensor& gate,const Tensor& up,const Tensor& down,uint32_t bucket,const std::filesystem::path& cache,const std::filesystem::path& out,const char* source){
    need(x.shape(0)==1 && x.shape(2)==4096 && x.dtype()==mx::bfloat16 && x.shape(1)<=16384,"invalid real prefill input");
    auto mlx=mx::compile([](const std::vector<Tensor>& a){auto hidden=tc::silu(mx::matmul(a[0],mx::transpose(a[1])))*mx::matmul(a[0],mx::transpose(a[2]));return std::vector<Tensor>{mx::matmul(hidden,mx::transpose(a[3]))};});
    mx::eval(x,gate,up,down);mx::synchronize();auto oracle=mlx({x,gate,up,down})[0];mx::eval(oracle);mx::synchronize();
    mg::Backend backend;auto gpu=mg::NativeFfn::prepare(backend,{4096,12288,12288});auto tmp=allocation(x.shape(1),12288),gpu_out=allocation(x.shape(1),4096);
    auto run_gpu=[&]{mg::CommandGraph g;gpu.encode(g,{view(backend,x),view(backend,gate),view(backend,up),view(backend,down),view(backend,tmp),view(backend,gpu_out),0});auto ticket=backend.submit(std::move(g));auto c=ticket.wait();need(c.ok && c.physically_complete,"native GPU oracle failed");};
    run_gpu();auto gpu_error=difference(gpu_out,oracle);need([gpu_error[@"rrms"] doubleValue]<.025,"native GPU FFN differs from model oracle");
    NSMutableArray* records=[NSMutableArray array];
    for(uint32_t channels:{3072U,6144U}){
        std::cerr<<"real-prefill source="<<source<<" rows="<<x.shape(1)<<" bucket="<<bucket<<" channels="<<channels<<std::endl;
        auto started=tc::Clock::now();mg::NativeHybridFfnOptions options;options.rows=bucket;options.ane_channels=channels;options.program_cache=cache;options.register_weight_staging=true;
        auto hybrid=mg::NativePrefillFfn::prepare(backend,std::move(options));const auto setup=elapsed(started);auto hybrid_out=allocation(x.shape(1),4096);
        auto run_hybrid=[&]{hybrid->execute(view(backend,x),view(backend,gate),view(backend,up),view(backend,down),view(backend,hybrid_out));};
        run_hybrid();auto warm_error=difference(hybrid_out,oracle);NSMutableArray* pairs=[NSMutableArray array];
        for(int order=0;order<2;++order){std::array<double,2> seconds{};
            for(int ordinal=0;ordinal<2;++ordinal){const int arm=order?1-ordinal:ordinal;auto t=tc::Clock::now();if(arm)run_hybrid();else run_gpu();seconds[arm]=elapsed(t);}
            auto err=difference(hybrid_out,oracle);
            [pairs addObject:@{@"order":order?@"BA":@"AB",@"native_gpu_seconds":@(seconds[0]),@"hybrid_seconds":@(seconds[1]),@"saving_percent":@(100*(seconds[0]-seconds[1])/seconds[0]),@"error":err}];
        }
        auto stats=hybrid->statistics();need(hybrid->healthy() && stats.calls==3 && stats.chunks==3ULL*((x.shape(1)+int(bucket)-1)/int(bucket)),"incorrect physical completion/chunk count");
        [records addObject:@{@"channels":@(channels),@"setup_seconds":@(setup),@"warm_error":warm_error,@"pairs":pairs,@"hybrid":metrics(stats)}];
        save(out.string()+".c"+std::to_string(channels)+".json",records.lastObject);
    }
    save(out,@{@"status":@"pass",@"scope":@(source),@"rows":@(x.shape(1)),@"bucket_rows":@(bucket),@"reference":@"original MLX BF16 checkpoint FFN, same real input",@"native_gpu_error":gpu_error,@"max_rrms":@.05,@"minimum_component_gain_percent":@5,@"candidates":records,@"models_downloaded":@0,@"model_exports":@0,@"full_request_qualified":@NO});
}
Tensor random_tensor(mx::Shape shape,uint64_t seed,float sigma){
    size_t n=1;for(int d:shape)n*=d;std::vector<float> data(n);
    for(auto& v:data){seed^=seed>>12;seed^=seed<<25;seed^=seed>>27;const auto bits=seed*2685821657736338717ULL;v=(float(uint32_t(bits>>40))*(2.f/16777216.f)-1.f)*sigma*std::sqrt(3.f);}
    return mx::astype(Tensor(data.data(),shape,mx::float32),mx::bfloat16);
}
void generated_prefill(const std::filesystem::path& out,const std::filesystem::path& cache){
    tc::configure_streams();mg::Backend b;mg::NativeHybridFfnOptions options;options.rows=32;options.hidden=128;options.width=1024;options.ane_channels=512;options.program_cache=cache;
    auto hybrid=mg::NativePrefillFfn::prepare(b,options);NSMutableArray* records=[NSMutableArray array];
    for(int rows:{1,33,65}){
        auto x=random_tensor({1,rows,128},20261008+rows,.1f),g=random_tensor({1024,128},7+rows,1.f/std::sqrt(128.f)),u=random_tensor({1024,128},9,1.f/std::sqrt(128.f)),d=random_tensor({128,1024},11,1.f/std::sqrt(1024.f));
        auto expected=mx::matmul(tc::silu(mx::matmul(x,mx::transpose(g)))*mx::matmul(x,mx::transpose(u)),mx::transpose(d));
        mx::eval(x,g,u,d,expected);mx::synchronize();auto output=mg::allocate_mlx_matrix(rows,128,mg::ScalarType::BF16,true);
        auto xv=view(b,x),gv=view(b,g),uv=view(b,u),dv=view(b,d),ov=view(b,output);bool rejected=false;
        try{hybrid->execute(xv,gv,uv,dv,xv);}catch(const std::invalid_argument&){rejected=true;}need(rejected && hybrid->healthy(),"alias failure retired/reused a source");
        hybrid->execute(xv,gv,uv,dv,ov);auto err=difference(output,expected);std::cerr<<"generated rows="<<rows<<" rrms="<<[err[@"rrms"] doubleValue]<<std::endl;need([err[@"rrms"] doubleValue]<=.08,"generated prefill exceeded 8% screen");
        [records addObject:@{@"rows":@(rows),@"chunks":@((rows+31)/32),@"error":err}];
    }
    const auto s=hybrid->statistics();need(hybrid->healthy() && s.calls==3 && s.chunks==6 && s.valid_rows==99 && s.padded_rows==93,"padded row accounting wrong");
    // A host source-preparation exception after both branch submissions may
    // retire the result, but it must not release any physically live source.
    auto controller=mg::NativeHybridFfn::prepare(b,options);
    auto x=random_tensor({1,32,128},1,.1f),g=random_tensor({1024,128},2,.08f),u=random_tensor({1024,128},3,.08f),d=random_tensor({128,1024},4,.03f);
    mx::eval(x,g,u,d);mx::synchronize();auto h=mg::allocate_mlx_matrix(32,512,mg::ScalarType::BF16,true);
    auto gp=mg::allocate_mlx_matrix(32,128,mg::ScalarType::FP32,true),ap=mg::allocate_mlx_matrix(32,128,mg::ScalarType::FP32,true),o=mg::allocate_mlx_matrix(32,128,mg::ScalarType::BF16,true);
    mg::NativeHybridFfnPlan plan;plan.input=view(b,x);plan.gate=view(b,g);plan.up=view(b,u);plan.down=view(b,d);plan.gpu_partial=view(b,gp);plan.ane_partial=view(b,ap);plan.published_output=view(b,o);
    controller.gpu_complement().encode(plan.gpu_ffn,{plan.input,plan.gate,plan.up,plan.down,view(b,h),plan.gpu_partial,0});
    controller.gpu_complement().encode_join_partials(plan.after_join,plan.gpu_partial,plan.ane_partial,plan.published_output);
    bool invoked=false;auto completion=controller.execute_until(std::move(plan),std::chrono::steady_clock::now()+std::chrono::seconds(30),[&]{invoked=true;throw std::runtime_error("injected-next-source-failure");});
    need(invoked && !completion.completion.ok && completion.completion.retired && completion.completion.all_work_physically_complete,"source preparation exception released live work");
    save(out,@{@"status":@"pass",@"scope":@"generated short/tail/multi-chunk and changing weights, source alias rejection and delayed next-source failure",@"records":records,@"hybrid":metrics(s),@"source_failure_drained":@YES,@"models_loaded":@0});
}
void encoder(const std::filesystem::path& root,const std::filesystem::path& out,const std::filesystem::path& cache,const std::filesystem::path& image){
    tc::configure_streams();mx::set_cache_limit(256ULL<<20);std::atomic<bool> stop=false;
    auto lease=tc::streaming::SourceLease::capture({{.logical_id="text",.path=root/"text_encoders/qwen3vl_8b_bf16.safetensors"},{.logical_id="tokenizer",.path=root/"processor/tokenizer.json"}});
    tc::Weights weights;weights.load_lease(lease,{"text"},[](const std::string&,int,int){},stop);auto fd=lease->duplicate_fd("tokenizer");tc::Tokenizer tokenizer(fd.get(),lease->file("tokenizer").bytes);
    const auto hash=tc::sha256_file(image);auto pixels=tc::qwen21::resize_reference(tc::load_rgba_image_tensor(image),1024),alpha=tc::slice_axis(pixels,3,3,4);auto input=tc::qwen21::vision_input(tc::slice_axis(pixels,3,0,3)*alpha+(1.f-alpha));
    tc::qwen21::VisionEncoder visual(weights);auto feature=visual.encode(input.patches,input.grid_height,input.grid_width,{},stop);mx::eval(feature.merged);mx::eval(feature.deepstack);mx::synchronize();weights.erase_prefix("model.visual.");mx::clear_cache();
    auto tokens=tokenizer.raw_bounded(tc::qwen21::reference_prompt_template("Edit the reference image. Keep the subject distinct. Warm studio light.",1),tc::Tokenizer::qwen21_limit);
    auto assembled=tc::qwen21::assemble_prompt(tokens,weights.at("model.embed_tokens.weight"),{{std::move(feature),input.grid_height,input.grid_width}});
    tc::qwen21::TextConfig config;config.layers=1;config.final_norm=false;tc::qwen21::TextEncoder text(weights,config);
    bool called=false;tc::qwen21::TextFfnHooks hooks{[](int,const std::string&,int){},[&](int layer,const Tensor& x,const tc::qwen21::TextFfnHooks::Gpu& gpu){
        need(layer==0 && !called,"unexpected encoder capture");called=true;std::string p="model.layers.0.mlp.";
        screen(x,weights.at(p+"gate_proj.weight"),weights.at(p+"up_proj.weight"),weights.at(p+"down_proj.weight"),1152,cache,out,"real single-image encoder layer0 FFN after complete vision27");return gpu(x);}};
    auto result=text.encode_embeddings(assembled.embeddings,assembled.positions,assembled.embeddings.shape(1),{},stop,{assembled.deepstack_deltas.front()},{},&hooks);mx::eval(result);mx::synchronize();
    need(called && tc::sha256_file(image)==hash,"missing capture/reference drift");lease->revalidate_after_drain();
}
void prefill(const std::filesystem::path& root,const std::filesystem::path& out,const std::filesystem::path& cache,const std::filesystem::path& image){
    tc::configure_streams();mx::set_cache_limit(256ULL<<20);std::atomic<bool> stop=false;tc::qwen21::Session session(root);session.load({},stop);
    tc::Request r;r.model="qwen-image-2.1";r.operation="image.edit";r.execution="gpu";r.width=r.height=512;r.frames=1;r.audio=false;r.steps=1;r.seed=42;r.allow_approximation=true;r.residency="component_staged";r.qwen21_dit_cache="off";r.qwen21_reference_size=1024;r.prompt="Edit the reference image. Keep the subject distinct. Warm studio light.";r.output=out.string()+".must-not-be-generated.png";
    r.inputs.push_back({.kind="image",.role="reference",.path=image.string()});bool called=false;
    session.set_diagnostic_prefill_observer([&](int layer,const Tensor& x,const tc::Weights& weights){need(layer==0 && !called,"unexpected DiT prefill capture");called=true;auto gu=weights.at("transformer_blocks.0.img_mlp.gate_up.weight"),down=weights.at("transformer_blocks.0.img_mlp.out.weight");
        screen(x,tc::slice_axis(gu,0,0,12288),tc::slice_axis(gu,0,12288,24576),down,2688,cache,out,"real single-image edit step0 layer0 FFN, no KV cache, base model");throw tc::qwen21::DiagnosticBlockReplayComplete();});
    bool bounded=false;try{(void)session.generate(r,{},stop);}catch(const tc::qwen21::DiagnosticBlockReplayComplete&){bounded=true;}
    session.set_diagnostic_prefill_observer({});session.unload();mx::synchronize();need(called && bounded && !std::filesystem::exists(r.output),"diagnostic did not stop before denoiser/decode");
}
void prewarmed_edit(const std::filesystem::path& root,const std::filesystem::path& out,const std::filesystem::path& cache,const std::filesystem::path& image,bool hybrid){
    tc::configure_streams();mx::set_cache_limit(256ULL<<20);std::atomic<bool> stop=false;auto process_start=tc::Clock::now();tc::qwen21::Session session(root);
    auto load_start=tc::Clock::now();session.load({},stop);const auto load_seconds=elapsed(load_start);double prewarm_seconds=0;
    if(hybrid){auto t=tc::Clock::now();session.prewarm_native_prefill(cache);prewarm_seconds=elapsed(t);need(session.prewarm_native_prefill(cache)==0,"idle prewarm rebuilt its resources");}
    tc::Request r;r.model="qwen-image-2.1";r.operation="image.edit";r.execution="gpu";r.width=r.height=512;r.frames=1;r.audio=false;r.steps=1;r.seed=42;r.allow_approximation=true;r.residency="component_staged";r.qwen21_dit_cache="off";r.qwen21_reference_size=1024;r.prompt="Edit the reference image. Keep the subject distinct. Warm studio light.";r.output=out.string()+".png";r.inputs.push_back({.kind="image",.role="reference",.path=image.string()});
    auto started=tc::Clock::now();auto result=session.generate(r,{},stop);const auto generation_seconds=elapsed(started);
    need(result.actual_steps==1 && result.lora_applied_projections==0 && result.reference_tokens==4096 && !result.prompt_cache_hit,"wrong prewarmed edit scope");
    if(hybrid)need(result.native_prefill && result.native_prefill->preparation_reused && result.native_prefill->calls==32 && result.native_prefill->chunks==64 && result.native_prefill->physically_complete,"prewarm not reused or first-step execution incomplete");
    NSDictionary* value=tc::to_dictionary(result);save(out,@{@"status":@"pass",@"hybrid":@(hybrid),@"model_load_seconds":@(load_seconds),@"prewarm_wall_seconds":@(prewarm_seconds),@"generation_wall_seconds":@(generation_seconds),@"cold_model_plus_prepare_plus_generation_seconds":@(elapsed(process_start)),@"result":value,@"conditioning_cache_hit":@(result.prompt_cache_hit),@"kv_cache_hit":@NO,@"quality_qualified":@NO});
    session.unload();mx::synchronize();
}

}
int main(int argc,char** argv){@autoreleasepool{try{
    char* raw=tc_runtime_build_identity();need(raw,"missing identity");std::string id(raw);tc_string_free(raw);need(id==TC_EDIT_ENCODER_EXPECTED_BUILD_ID,"wrong linked runtime");
    if(argc==4 && std::string_view(argv[1])=="--generated"){generated_prefill(argv[2],argv[3]);return 0;}
    if(argc==7 && std::string_view(argv[1])=="--prewarmed-edit"){need(std::string_view(argv[6])=="gpu" || std::string_view(argv[6])=="hybrid","bad prewarm arm");prewarmed_edit(argv[2],argv[3],argv[4],argv[5],std::string_view(argv[6])=="hybrid");return 0;}
    need(argc==6,"usage --encoder/--prefill MODEL OUT CACHE IMAGE");const std::string mode=argv[1];
    need(std::filesystem::path(argv[4]).is_absolute() && !std::filesystem::exists(argv[3]),"cache/output ownership invalid");
    if(mode=="--encoder")encoder(argv[2],argv[3],argv[4],argv[5]);else if(mode=="--prefill")prefill(argv[2],argv[3],argv[4],argv[5]);else throw std::invalid_argument("bad probe mode");
    return 0;
}catch(const std::exception& e){std::cerr<<"native_prefill_probe=FAIL "<<e.what()<<'\n';return 1;}}}

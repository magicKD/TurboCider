#include "pipeline.hpp"
#include "bf16_streaming.hpp"
#include "conditioning.hpp"
#include "diagnostic_options.hpp"
#include "scheduler.hpp"
#include "vae.hpp"
#include "../../media/image.hpp"
#include <mlx/random.h>
#include <cstdlib>

namespace tc::qwen21 {
namespace {
uint32_t option(const char *name,uint32_t fallback,uint32_t maximum) {
    const char *raw=std::getenv(name);if(!raw)return fallback;
    const auto text=std::string_view(raw);
    require(!text.empty() && text.size()<=2 && text.find_first_not_of("0123456789")==std::string_view::npos,
        "Qwen BF16 stream option requires a bounded integer");
    const auto n=std::stoul(std::string(text));require(n<=maximum,"Qwen BF16 stream option exceeds bound");return uint32_t(n);
}
}
RunResult Session::run_bf16_streamed(const Request &r,const Event &event,std::atomic<bool> &cancelled,
        bool warmup,bool prepare_only) {
    require(!prepare_only && r.model=="qwen-image-2.1" && r.operation=="image.generate" && r.inputs.empty() &&
        r.width==512 && r.height==512 && r.steps>=2 && r.steps<=40 && r.execution=="gpu" &&
        r.residency=="component_staged" && r.loras.empty() && !r.prompt_enhance && r.hybrid_mlp_mode=="auto" &&
        r.ane_manifest.empty() && r.encoder_ane_manifest.empty() && !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
        !r.streaming.active() && !r.streaming_offload && !r.memory_constrained.enabled && !r.memory_budget_bytes &&
        transformer_source_.extension()==".safetensors" && encoder_source_.extension()==".safetensors",
        "Qwen private original-BF16 streaming needs component-staged512 base GPU generation/warmup; public RAM authority not implemented");
    for(const char *flag:{"TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS","TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME",
        "TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU","TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC","TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV",
        "TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC","TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN",
        "TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN","TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC",
        "TURBOCIDER_QWEN21_METAL_QK_ROPE","TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE","TURBOCIDER_QWEN21_PROFILE_GPU_BLOCKS",
        "TURBOCIDER_QWEN21_PROFILE_GPU_OPS","TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS","TURBOCIDER_QWEN21_GGUF_SHARED_DOWN"})
        require(!option_enabled(std::getenv(flag)),"Qwen original-BF16 streaming excludes another math/cache/profiling recipe");
    const auto start=Clock::now();auto elapsed=[](auto begin){return std::chrono::duration<double>(Clock::now()-begin).count();};
    const auto encoder_prefix=option("TURBOCIDER_QWEN21_BF16_STREAM_ENCODER_PREFIX",24,34);
    const auto dit_prefix=option("TURBOCIDER_QWEN21_BF16_STREAM_DIT_PREFIX",24,30);
    const auto budget_gib=option("TURBOCIDER_QWEN21_BF16_STREAM_WEIGHT_GIB",11,20);
    require(budget_gib>=1,"Qwen BF16 managed weight budget must be positive");
    const uint64_t budget=uint64_t(budget_gib)<<30;
    checkpoint(cancelled);mx::synchronize();unload();mx::reset_peak_memory();mx::set_cache_limit(r.allocator_cache_bytes);
    RunResult result;result.request=r;result.plan=make_plan(r);result.prepared=prepare_only;result.warmup=warmup;
    result.backend="mlx_cpp_metal_qwen21_bf16_streaming_experimental";result.precision="bf16_dit+bf16_text+bf16_vae";
    result.checkpoint=transformer_source_.filename().string();
    result.selection="private original-BF16 component-staged encoder+DiT, bounded two-slot CPU prefetch, dynamic GPU weights and evaluated last-reader retirement; not a public RAM cap/physical overlap certificate";
    result.qwen_bf16_streaming.emplace();result.qwen_ffn_phases.emplace();result.qwen_ffn_phases->policy="gpu";
    auto emit=[&](const char *phase,int completed,int total){if(event)event(phase,completed,total);};
    auto dump=[&](const char *name,const Tensor &tensor) {
        if(r.dump.empty())return;
        std::filesystem::create_directories(r.dump);
        mx::save_safetensors((std::filesystem::path(r.dump)/(std::string(name)+".safetensors")).string(),{{"tensor",tensor}});
    };
    Tensor text(0.f);const auto text_start=Clock::now();
    bf16_stream_scope([&] {
        emit("load_qwen21_text",0,1);
        Bf16StreamBank bank(bf16_stream_plan(encoder_source_,true,36,encoder_prefix,1,cancelled),budget,cancelled);
        emit("load_qwen21_text",1,1);
        Tokenizer tokenizer(root_/"processor");auto tokens=tokenizer.raw_bounded(reference_prompt_template(r.prompt,0),Tokenizer::qwen21_limit);
        auto assembled=assemble_prompt(tokens,bank.resident(),{});
        TextConfig config;config.final_norm=false;TextEncoder encoder(bank.resident(),config);
        encoder.set_layer_weights([&](int layer)->const Weights &{return bank.acquire(layer);},[&](int layer){bank.retire(layer);});
        bank.begin_pass(0);
        text=assembled.retain(encoder.encode_embeddings(assembled.embeddings,assembled.positions,assembled.embeddings.shape(1),event,cancelled));
        mx::eval(text);require(mx::all(mx::isfinite(text)).item<bool>(),"nonfinite streamed Qwen encoder output");
        bank.finish_pass();bank.finish();result.qwen_bf16_streaming->encoder=bank.metrics();
    });
    mx::clear_cache();result.timings.text=elapsed(text_start);dump("qwen21_text",text);
    result.text_tokens=result.valid_text_tokens=text.shape(1);result.total_tokens=text.shape(1)+1024;
    auto schedule=sigmas(r.width,r.height,r.steps);mx::eval(schedule);
    auto latents=mx::astype(mx::random::normal({1,1024,64},mx::float32,mx::random::key(r.seed)),mx::bfloat16);
    text=mx::astype(text,mx::bfloat16);mx::eval(text);
    if(!r.noise_path.empty()) {
        auto [values,metadata]=mx::load_safetensors(r.noise_path);require(values.count("tensor") || values.count("initial"),"Qwen noise file missing tensor");
        const auto &noise=values.at(values.count("tensor") ? "tensor" : "initial");require(noise.shape()==latents.shape(),"Qwen noise shape mismatch");
        latents=mx::astype(noise,mx::bfloat16);
    }
    dump("qwen21_initial",latents);
    emit("load_qwen21_transformer",0,1);const auto denoise_start=Clock::now();
    bf16_stream_scope([&] {
        Bf16StreamBank bank(bf16_stream_plan(transformer_source_,false,32,dit_prefix,uint32_t(r.steps),cancelled),budget,cancelled);
        emit("load_qwen21_transformer",1,1);Transformer dit(bank.resident());
        dit.set_layer_weights([&](int layer)->const Weights &{return bank.acquire(layer);},[&](int layer){bank.retire(layer);});
        emit("denoise",0,r.steps);
        for(int step=0;step<r.steps;++step) {
            checkpoint(cancelled);bank.begin_pass(uint32_t(step));const auto step_start=Clock::now();
            const bool reuse=dit.prefix_matches(text,32,32,{});
            auto noise=dit.forward(latents,text,schedule.data<float>()[step],32,32,true);
            latents=latents+noise*Tensor(schedule.data<float>()[step+1]-schedule.data<float>()[step],latents.dtype());
            mx::eval(latents);require(mx::all(mx::isfinite(latents)).item<bool>(),"nonfinite streamed Qwen latent");
            bank.finish_pass();auto &phase=reuse ? result.qwen_ffn_phases->decode : result.qwen_ffn_phases->prefill;
            ++phase.steps;phase.rows=reuse ? 1024 : uint64_t(result.total_tokens);phase.step_seconds+=elapsed(step_start);
            ++result.actual_steps;emit("denoise",step+1,r.steps);
        }
        bank.finish();result.qwen_bf16_streaming->denoiser=bank.metrics();
    });
    mx::clear_cache();result.timings.denoise=elapsed(denoise_start);dump("qwen21_latents",latents);
    // Both component weight banks are gone before VAE loading, not merely
    // labels changed on still-resident arrays. Source leases/graphs drained.
    checkpoint(cancelled);const auto decode_start=Clock::now();Weights vae;
    emit("load_qwen21_vae",0,1);vae.load_file(root_/"vae/qwen_image_2.1_vae_bf16.safetensors");vae.materialize();emit("load_qwen21_vae",1,1);
    VAE decoder(vae);auto spatial=mx::transpose(mx::reshape(latents,{1,32,32,64}),{0,3,1,2});
    auto pixels=mx::transpose(decoder.decode(spatial,event,cancelled),{0,2,3,1});mx::eval(pixels);
    require(mx::all(mx::isfinite(pixels)).item<bool>(),"nonfinite streamed Qwen pixels");dump("qwen21_pixels",pixels);
    result.timings.decode=elapsed(decode_start);
    if(!warmup){checkpoint(cancelled);emit("export",0,1);save_rgba_png(pixels,r.output);emit("export",1,1);}
    result.timings.wall=elapsed(start);result.active_bytes=mx::get_active_memory();result.peak_bytes=mx::get_peak_memory();return result;
}
}

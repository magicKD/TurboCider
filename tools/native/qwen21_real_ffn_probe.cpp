// Bounded replay of original BF16 decode inputs through the production bridge.
// Includes weight staging, A8 conversion, LoRA, final join and synchronization.
// Offline component evidence only: never registers a model/profile in the App.
#include "../../native/backends/ane_ffn.hpp"
#include "../../native/backends/ane_smoothquant.hpp"
#include "../../native/models/qwen21/runtime_ffn_graphs.hpp"
#include "../../native/platform/apple/platform.hpp"
#include "turbocider/turbocider.h"
#include <array>
#include <bit>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

using namespace tc;
namespace cal = ane::calibration;
namespace ff = qwen21::runtime_ffn;
struct Sample { int layer,step; Tensor input; std::string sha; };
struct Layer {
    std::vector<Tensor> weights;
    ff::Function full,channels,corrections,down;
};
static double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
static void env(const char *key,const std::string &value) { ::setenv(key,value.c_str(),1); }
static void metrics(std::ostream &out,const HybridMetrics &a,const HybridMetrics &b) {
#define SPAN(name) out << ",\"" #name "\":" << b.runtime_weight_##name-a.runtime_weight_##name
    SPAN(stage_seconds); SPAN(stage_wait_seconds); SPAN(pre_seconds); SPAN(lora_input_ready_seconds);
    SPAN(a8_wait_seconds); SPAN(async_ane_wait_seconds); SPAN(post_join_seconds); SPAN(wall_seconds);
    SPAN(fallback_blocks); SPAN(overflow_retries); SPAN(scale_cache_hits); SPAN(scale_cache_misses);
#undef SPAN
    out << ",\"calls\":" << b.runtime_calls-a.runtime_calls << ",\"headroom\":" << b.runtime_weight_headroom;
}
int main(int argc,char **argv) {
    try {
        require(argc==8 || argc==9,"usage: checkpoint capture-dir shape-json output.jsonl owned-cache width hadamard|sq-hadamard|comfy [lora]");
        configure_streams();
        const auto checkpoint=std::filesystem::canonical(argv[1]), capture=std::filesystem::canonical(argv[2]);
        const int width=std::stoi(argv[6]); const std::string recipe=argv[7];
        require((width==4096 || width==6144) && (recipe=="hadamard" || recipe=="sq-hadamard" || recipe=="comfy"),"unbounded replay configuration");
        require(std::filesystem::is_regular_file(capture/"complete.json"),"capture lacks completion receipt");
        const bool lora=argc==9;
        const char *fence_env=std::getenv("TURBOCIDER_FFN_REPLAY_FENCE");
        require(!fence_env || std::string(fence_env)=="0" || std::string(fence_env)=="1","replay fence requires 0 or 1");
        const bool fence=fence_env && std::string(fence_env)=="1";
        std::string adapter;
        if(lora) {
            const auto p=std::filesystem::canonical(argv[8]);
            adapter=p.string()+":"+std::to_string(std::filesystem::file_size(p))+":"+
                std::to_string(static_cast<long long>(std::filesystem::last_write_time(p).time_since_epoch().count()))+
                ":"+std::to_string(std::bit_cast<uint32_t>(1.f))+":"+sha256_file(p);
        }
        const auto identity=ane::smoothquant::checkpoint_fingerprint(checkpoint);
        std::vector<Sample> samples;
        std::string prompt,trajectory_id;
        for(const auto &entry:std::filesystem::directory_iterator(capture)) {
            if(entry.path().extension()!=".safetensors")continue;
            require(!entry.is_symlink() && entry.file_size()<9*1024*1024,"invalid bounded replay input");
            auto [arrays,m]=mx::load_safetensors(entry.path().string());
            require(m.at("capture_recipe")=="qwen21-real-ffn-bf16-v1" && m.at("trajectory")=="forced-gpu" &&
                m.at("model_identity")==identity && m.at("adapter_identity")==adapter,"replay source/model/adapter provenance mismatch");
            if(prompt.empty())prompt=m.at("prompt");
            require(m.at("prompt")==prompt,"mixed prompt capture");
            const auto trajectory=m.at("seed")+":"+m.at("total_steps");
            if(trajectory_id.empty())trajectory_id=trajectory;
            require(trajectory==trajectory_id,"mixed seed/step-count trajectory");
            const int layer=std::stoi(m.at("layer")), step=std::stoi(m.at("step"));
            require((layer==0 || layer==15 || layer==31) && step>0,"invalid replay point");
            require(arrays.size()==1 && arrays.at("input").shape()==mx::Shape({1,1024,4096}) &&
                arrays.at("input").dtype()==mx::bfloat16,"replay must contain complete original BF16 input");
            samples.push_back({layer,step,arrays.at("input"),sha256_file(entry.path())});
        }
        require(samples.size()==9,"expected three depths and three steps");
        std::sort(samples.begin(),samples.end(),[](const auto &a,const auto &b){return std::tie(a.step,a.layer)<std::tie(b.step,b.layer);});
        for(size_t i=0;i<samples.size();++i) {
            require(samples[i].layer==std::array<int,3>{0,15,31}[i%3],"missing/duplicate depth");
            if(i%3) require(samples[i].step==samples[i-1].step,"mixed step group");
            else if(i) require(samples[i].step>samples[i-1].step,"duplicate step group");
            mx::eval(samples[i].input);
        }
        require(samples[0].step==1,"missing early calibration input");
        Weights weights; weights.load_file(checkpoint); weights.set_runtime_lora_fp16(false);
        std::atomic<bool> cancelled{false};
        Event event=[](const std::string &,int,int){};
        if(lora)require(weights.apply_loras({{argv[8],1.f,"transformer"}},"transformer",event,cancelled,true)==227,"LoRA binding mismatch");
        std::map<int,Layer> layers;
        for(int ordinal:{0,15,31}) {
            auto prefix="transformer_blocks."+std::to_string(ordinal)+".img_mlp.";
            auto gu=mx::split(weights.at(prefix+"gate_up.weight"),2,0);
            auto &v=layers.emplace(ordinal,Layer{{gu[0],gu[1],weights.at(prefix+"out.weight")},
                ff::full(weights,prefix),ff::channels(weights,prefix,0,12288-width),
                ff::corrections(weights,prefix,12288-width,width),ff::down_add(weights,prefix)}).first->second;
            mx::eval(v.weights);
        }
        if (const char *raw=std::getenv("TURBOCIDER_FFN_REPLAY_EXPORT_CALIBRATION")) {
            const auto directory=std::filesystem::path(raw);
            require(directory.is_absolute() && std::filesystem::create_directory(directory),"calibration export needs a fresh absolute directory");
            std::filesystem::create_directories(directory/"inputs/block0");
            const auto input_path=directory/"inputs/block0/step1.npy";
            mx::save(input_path.string(),mx::astype(mx::squeeze(samples[0].input,0),mx::float32));
            if(lora) {
                std::filesystem::create_directories(directory/"corrections/block0");
                auto fn=ff::corrections(weights,"transformer_blocks.0.img_mlp.",0,width);
                auto deltas=fn({samples[0].input});mx::eval(deltas);
                const auto gate=directory/"corrections/block0/step1.gate.npy",up=directory/"corrections/block0/step1.up.npy";
                mx::save(gate.string(),mx::astype(mx::squeeze(deltas[0],0),mx::float32));
                mx::save(up.string(),mx::astype(mx::squeeze(deltas[1],0),mx::float32));
                std::ofstream binding(directory/"corrections/binding.json");
                binding<<"{\"schema_version\":1,\"checkpoint_sha256\":"<<cal::quote(sha256_file(checkpoint))
                    <<",\"adapter_identity\":"<<cal::quote(adapter)<<",\"width\":"<<width<<",\"samples\":{\"block0/step1.npy\":{\"input_sha256\":"
                    <<cal::quote(sha256_file(input_path))<<",\"gate_sha256\":"<<cal::quote(sha256_file(gate))
                    <<",\"up_sha256\":"<<cal::quote(sha256_file(up))<<"}}}\n";
                binding.close();require(bool(binding),"failed to write correction binding");
            }
            std::cout<<directory<<'\n';return 0;
        }
        // Explicit bounded experimental configuration, no native-auto fitting.
        env("TURBOCIDER_ANE_BACKEND","private");env("TURBOCIDER_ALLOW_PRIVATE_ANE","1");
        env("TURBOCIDER_PRIVATE_ANE_DATA_PATH",recipe=="comfy"?"convrot_w8a8":"w8a8");
        env("TURBOCIDER_PRIVATE_ANE_CHANNELS",std::to_string(width));env("TURBOCIDER_PRIVATE_ANE_ROWS","request");
        env("TURBOCIDER_PRIVATE_ANE_CACHE_DIR",std::filesystem::absolute(argv[5]).string());
        env("TURBOCIDER_PRIVATE_ANE_GPU_IO","1");env("TURBOCIDER_RUNTIME_ANE_CHUNKS","1");
        env("TURBOCIDER_RUNTIME_ANE_PROFILE","0");env("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1");
        env("TURBOCIDER_PRIVATE_ANE_PREFETCH","0");env("TURBOCIDER_PRIVATE_ANE_SCALE_CACHE","1");
        env("TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE",fence?"1":"0");env("TURBOCIDER_PRIVATE_ANE_A8_SINGLE_PASS","0");
        env("TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN","0");env("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN","0");
        const auto setup=Clock::now();
        ane::HybridFfn bridge(argv[3],4096,12288,2ull<<30,cancelled,lora,std::nullopt,{},nullptr,std::nullopt,std::nullopt,std::nullopt,1024);
        require(bridge.available(),bridge.reason());bridge.begin_request(adapter);
        if(recipe=="sq-hadamard") {
            std::vector<Tensor> scales(32,mx::ones({1,4096},mx::float32));
            for(size_t i=0;i<3;++i) {
                const auto &s=samples[i];const auto &v=layers.at(s.layer);
                auto activation=mx::max(mx::abs(mx::astype(s.input,mx::float32)),{0,1});
                auto maximum=mx::maximum(mx::max(mx::abs(mx::astype(v.weights[0],mx::float32)),0),
                                         mx::max(mx::abs(mx::astype(v.weights[1],mx::float32)),0));
                auto scale=mx::clip(mx::sqrt(mx::maximum(activation,Tensor(1e-8f))/mx::maximum(maximum,Tensor(1e-8f))),Tensor(1.f/16),Tensor(16.f));
                scale=mx::where((activation==Tensor(0.f))|(maximum==Tensor(0.f)),mx::ones_like(scale),scale);
                scales[s.layer]=mx::contiguous(mx::reshape(scale,{1,4096}));mx::eval(scales[s.layer]);
            }
            // Diagnostic-only partial-depth scales. Unobserved ones are never
            // run and this vector is never written as a production sidecar.
            bridge.set_smoothquant(sha256_file(capture/"complete.json"),[&]{return scales;});
        }
        const auto setup_seconds=seconds(setup);
        require(!std::filesystem::exists(argv[4]),"refusing to overwrite replay results");
        std::ofstream out(argv[4]);out<<std::setprecision(12);
        char *raw_build=tc_runtime_build_identity();require(raw_build,"missing runtime build identity");
        const std::string build_identity(raw_build);tc_string_free(raw_build);
        out<<"{\"scope\":\"complete FFN replay, 3 depths; not attention/residual/E2E or physical overlap proof\",\"recipe\":"<<cal::quote(recipe)
           <<",\"runtime_build_identity\":"<<cal::quote(build_identity)<<",\"model_stat_identity\":"<<cal::quote(identity)
           <<",\"capture_receipt_sha256\":"<<cal::quote(sha256_file(capture/"complete.json"))
           <<",\"width\":"<<width<<",\"lora\":"<<(lora?"true":"false")<<",\"launch_fence\":"<<(fence?"true":"false")<<",\"setup_seconds\":"<<setup_seconds
           <<",\"calibration_step\":1,\"samples\":9,\"warmup_passes\":1,\"measured_passes\":2}\n";
        // Cycle all depths BEFORE repeating, so a hot single weight bank
        // cannot stand in for the layer-changing runtime staging path.
        for(int pass=0;pass<3;++pass)for(const auto &s:samples) {
            auto &v=layers.at(s.layer);auto gpu=[&](const Tensor &x){return v.full({x})[0];};
            auto channel=[&](const Tensor &x,int first,int count) {
                require(first==0 && count==12288-width,"channel changed");auto a=v.channels({x});return std::make_pair(a[0],a[1]);
            };
            auto corrections=[&](const Tensor &x,int first,int count) {
                require(first==12288-width && count==width,"LoRA channel changed");auto a=v.corrections({x});return std::make_pair(a[0],a[1]);
            };
            ane::HybridFfn::Adapter a{
                [](const Tensor &)->std::pair<Tensor,Tensor>{throw std::runtime_error("full correction unexpectedly requested");},
                [&](const Tensor &h,const Tensor &b){return v.down({h,b})[0];},corrections};
            const auto before=bridge.metrics();
            std::optional<Tensor> ref,actual;
            auto run_gpu=[&]{auto start=Clock::now();ref=gpu(s.input);mx::eval(*ref);return seconds(start);};
            auto run_hybrid=[&]{auto start=Clock::now();
                require(bridge.plan_block(s.layer,1024).split(),"replay silently selected GPU");
                bridge.stage(s.layer,1024,v.weights);
                actual=bridge.run(s.layer,s.input,gpu,cancelled,lora?&a:nullptr,channel);mx::eval(*actual);return seconds(start);};
            double gs,hs;
            if(pass%2){hs=run_hybrid();gs=run_gpu();}else{gs=run_gpu();hs=run_hybrid();}
            auto r=mx::astype(*ref,mx::float32),y=mx::astype(*actual,mx::float32),d=y-r;
            const double rrms=mx::sqrt(mx::sum(d*d)/mx::sum(r*r)).item<float>();
            const double cosine=(mx::sum(y*r)/mx::sqrt(mx::sum(y*y)*mx::sum(r*r))).item<float>();
            require(std::isfinite(rrms)&&std::isfinite(cosine),"nonfinite replay output");
            auto after=bridge.metrics();
            out<<"{\"pass\":"<<pass<<",\"layer\":"<<s.layer<<",\"step\":"<<s.step<<",\"held_out\":"<<(s.step!=1?"true":"false")
               <<",\"input_sha256\":"<<cal::quote(s.sha)<<",\"gpu_seconds\":"<<gs<<",\"hybrid_seconds\":"<<hs
               <<",\"relative_rmse\":"<<rrms<<",\"cosine\":"<<cosine;
            metrics(out,before,after);out<<"}\n";out.flush();
            require(after.runtime_calls==before.runtime_calls+1 && after.runtime_weight_fallback_blocks==0 && after.runtime_failures==0,
                "replay did not complete its declared ANE path: "+bridge.reason());
        }
        bridge.drain();
        require(ane::smoothquant::checkpoint_fingerprint(checkpoint)==identity,"checkpoint changed during replay");
        std::cout<<argv[4]<<'\n';
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}

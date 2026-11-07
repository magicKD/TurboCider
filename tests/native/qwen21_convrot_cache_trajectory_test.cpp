#include "models/qwen21/bf16_streaming.hpp"
#include "models/qwen21/transformer.hpp"
#include "models/qwen21/scheduler.hpp"
#include <iostream>

using namespace tc;
using namespace tc::qwen21;

int main(int argc,char **argv) {
    try {
        require(argc==3,"expected original DiT and a matching 512px Turbo8 correctness dump directory");
        configure_streams();mx::set_cache_limit(0);
        require(convrot_w8a8_available(),"the full trajectory check requires actual M5 W8A8");
        auto load=[&](const char *name) {
            auto [values,metadata]=mx::load_safetensors(std::filesystem::path(argv[2])/(std::string(name)+".safetensors"));
            require(values.count("tensor"),"missing owned diagnostic tensor");
            auto tensor=values.at("tensor");mx::eval(tensor);return tensor;
        };
        auto text=load("qwen21_text"),latents=load("qwen21_initial"),expected=load("qwen21_latents");
        require(latents.shape()==mx::Shape{1,1024,64} && expected.shape()==latents.shape(),
            "full eviction trajectory requires 512px latent geometry");
        std::atomic<bool> cancel{false};
        auto plan=bf16_stream_cache_plan(bf16_stream_plan(argv[1],false,32,0,8,cancel,
            StreamWeightMode::ConvRotW8A8),uint64_t(1)<<30);
        require(plan.prefix==1,"one-GiB fixture did not select one real DiT layer");
        Bf16StreamBank bank(std::move(plan),uint64_t(1)<<30,cancel);
        Transformer dit(bank.resident());
        dit.set_layer_weights([&](int layer)->const Weights&{return bank.acquire(layer);},
            [&](int layer){bank.retire(layer);});
        auto schedule=turbo_sigmas(512,512,8);mx::eval(schedule);
        const auto *sigma=schedule.data<float>();
        const auto before=convrot_w8a8_stats();
        for(uint32_t step=0;step<8;++step) {
            bf16_stream_scope([&] {
                bank.begin_pass(step);
                auto velocity=dit.forward(latents,text,sigma[step],32,32,true);
                latents=latents+velocity*Tensor(sigma[step+1]-sigma[step],latents.dtype());
                mx::eval(latents);bank.finish_pass();
            });
            if(step==2) {
                const auto active=mx::get_active_memory();
                require(bank.evict_cached_layers(),"full-model cache did not evict");
                require(mx::get_active_memory()<active,"full-model optional backing stayed live");
            }
        }
        bank.finish();
        const auto metrics=bank.metrics();
        require(mx::all(latents==expected).item<bool>(),"cache eviction changed the full eight-step 512px latent trajectory");
        require(metrics.cache_evictions==1 && metrics.final_prefix==0 && metrics.completed_prefix_layers==3 &&
            metrics.completed_streamed_layers==253 && metrics.fills==253 && metrics.completed_reader_fences==253 &&
            metrics.prefix_per_pass==std::vector<uint32_t>{1,1,1,0,0,0,0,0},"incomplete full-model eviction coverage");
        require(convrot_w8a8_stats().projections-before.projections==1536,"cache eviction changed actual W8A8 projection coverage");
        std::cout<<"PASS real 512px Turbo8: forced cache eviction after step 3; exact final latents; 1536 W8A8 projections; 253 drained refills; optional memory released\n";
        return 0;
    }catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}

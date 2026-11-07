#include "models/qwen21/bf16_streaming.hpp"
#include "models/qwen21/transformer.hpp"
#include "models/qwen21/scheduler.hpp"
#include <cstdlib>
#include <iostream>
#include <string_view>

using namespace tc;
using namespace tc::qwen21;

int main(int argc,char **argv) {
    try {
        require(argc==3,"expected original DiT and a matching 512px Turbo8 correctness dump directory");
        const auto* compact_control=std::getenv("TURBOCIDER_QWEN21_STREAM_COMPACT_KV");
        require(!compact_control || std::string_view(compact_control)=="0" || std::string_view(compact_control)=="1",
            "stream prefix compaction control requires 0 or 1");
        const bool compact_requested=!compact_control || std::string_view(compact_control)=="1";
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
        require(text.ndim()==3 && text.shape(0)==1 && text.shape(1)>0 && text.shape(2)==4096,
            "full eviction trajectory requires complete text rows at the original hidden width");
        constexpr uint32_t prefix_arrays=2*32;
        const uint64_t prefix_logical_bytes=uint64_t(prefix_arrays)*uint64_t(text.shape(1))*4096*latents.itemsize();
        constexpr uint64_t prefix_alignment_allowance=uint64_t(prefix_arrays)*16384;
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
        const auto quant_before=convrot_w8a8_quant_stats();
        const bool shared=convrot_shared_a8_enabled();
        uint64_t previous_prefix_backing=0;
        for(uint32_t step=0;step<8;++step) {
            const auto step_quant=convrot_w8a8_quant_stats();
            bf16_stream_scope([&] {
                bank.begin_pass(step);
                auto velocity=dit.forward(latents,text,sigma[step],32,32,true);
                latents=latents+velocity*Tensor(sigma[step+1]-sigma[step],latents.dtype());
                mx::eval(latents);bank.finish_pass();
            });
            require(convrot_w8a8_quant_stats().dispatches-step_quant.dispatches==
                uint64_t(shared && step>0 ? 128 : 192),
                "shared A8 changed prefill storage or missed decode projection groups");
            // prefix_storage is a read-only ownership inspection: it must
            // not turn an unfinished cache into a successful evaluated one.
            // The ordinary streamed forward/eval above must finish all K/V.
            const auto prefix=dit.prefix_storage();
            require(prefix.arrays==prefix_arrays && prefix.logical_bytes==prefix_logical_bytes && prefix.evaluated,
                "streamed trajectory lost complete evaluated prefix K/V storage");
            if(compact_requested) {
                require(prefix.compact,"streamed prefix K/V did not own compact independent storage");
                require(prefix.backing_bytes<=prefix_logical_bytes+prefix_alignment_allowance,
                    "compact prefix K/V retained target rows or excessive backing");
            }
            // Both policies reuse the same immutable prefix through decode.
            // The legacy arm has no assumed waste ratio or fixed byte count.
            if(step>0)require(prefix.backing_bytes<=previous_prefix_backing,
                "prefix K/V backing grew during the cached decode trajectory");
            previous_prefix_backing=prefix.backing_bytes;
            const auto active_before_eviction=mx::get_active_memory();
            if(step==2) {
                const auto active=mx::get_active_memory();
                require(bank.evict_cached_layers(),"full-model cache did not evict");
                require(mx::get_active_memory()<active,"full-model optional backing stayed live");
            }
            std::cout<<"{\"prefix_storage_step\":"<<step
                <<",\"compact_control\":\""<<(compact_control?compact_control:"default")<<"\""
                <<",\"compact_requested\":"<<(compact_requested?"true":"false")
                <<",\"prefix_arrays\":"<<prefix.arrays<<",\"prefix_logical_bytes\":"<<prefix.logical_bytes
                <<",\"prefix_backing_bytes\":"<<prefix.backing_bytes
                <<",\"prefix_evaluated\":"<<(prefix.evaluated?"true":"false")
                <<",\"prefix_compact\":"<<(prefix.compact?"true":"false")
                <<",\"active_before_optional_eviction_bytes\":"<<active_before_eviction
                <<",\"active_bytes\":"<<mx::get_active_memory()
                <<",\"optional_cache_evicted_this_step\":"<<(step==2?"true":"false")<<"}"<<std::endl;
        }
        bank.finish();
        const auto metrics=bank.metrics();
        require(mx::all(latents==expected).item<bool>(),"cache eviction changed the full eight-step 512px latent trajectory");
        require(metrics.cache_evictions==1 && metrics.final_prefix==0 && metrics.completed_prefix_layers==3 &&
            metrics.completed_streamed_layers==253 && metrics.fills==253 && metrics.completed_reader_fences==253 &&
            metrics.prefix_per_pass==std::vector<uint32_t>{1,1,1,0,0,0,0,0},"incomplete full-model eviction coverage");
        require(convrot_w8a8_stats().projections-before.projections==1536,"cache eviction changed actual W8A8 projection coverage");
        const auto quantizations=convrot_w8a8_quant_stats().dispatches-quant_before.dispatches;
        require(quantizations==uint64_t(shared ? 1088 : 1536),"incomplete physical A8 dispatch coverage");
        std::cout<<"PASS real 512px Turbo8: forced cache eviction after step 3; exact final latents; 1536 W8A8 projections; 253 drained refills; optional memory released; physical A8 dispatches="<<quantizations
            <<"; prefix logical/backing bytes="<<prefix_logical_bytes<<'/'<<previous_prefix_backing
            <<"; compact requested="<<(compact_requested?"true":"false")<<'\n';
        return 0;
    }catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}

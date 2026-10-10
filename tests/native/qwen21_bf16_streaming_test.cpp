#include "models/qwen21/bf16_streaming.hpp"
#include "models/qwen21/transformer.hpp"
#include "models/qwen21/text_encoder.hpp"
#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

using namespace tc;
using namespace tc::qwen21;
namespace {
template<class F>void rejects(F fn){bool failed=false;try{fn();}catch(const std::exception &){failed=true;}require(failed,"invalid BF16 stream accepted");}
void equal(const Tensor &a,const Tensor &b,const char *reason) {
    mx::eval({a,b});require(a.shape()==b.shape() && a.dtype()==b.dtype() && mx::all(a==b).item<bool>(),reason);
}
}
int main(int argc,char **argv) {try {
    require(argc==5,"encoder,denoiser,malformed,owned-change fixtures required");configure_streams();mx::set_cache_limit(0);
    std::atomic<bool> cancel{false};Weights text_weights,dit_weights;
    text_weights.load_file(argv[1]);dit_weights.load_file(argv[2]);text_weights.materialize();dit_weights.materialize();
    Tokens tokens;tokens.ids={4,7,11,4};tokens.valid=4;
    TextConfig text_config;text_config.layers=4;text_config.heads=text_config.kv_heads=1;text_config.final_norm=false;
    TextEncoder text_reference(text_weights,text_config);auto expected_text=text_reference.encode(tokens,{},cancel);mx::eval(expected_text);
    for(uint32_t prefix:{0u,1u,2u}) {
        Bf16StreamBank bank(bf16_stream_plan(argv[1],true,4,prefix,1,cancel),16ull<<20,cancel);
        TextEncoder encoder(bank.resident(),text_config);
        encoder.set_layer_weights([&](int layer)->const Weights &{return bank.acquire(layer);},[&](int layer){bank.retire(layer);});
        bank.begin_pass(0);auto actual=encoder.encode(tokens,{},cancel);equal(actual,expected_text,"streamed encoder changed original BF16 hidden");
        bank.finish_pass();bank.finish();const auto metrics=bank.metrics();
        require(metrics.drained && metrics.completed_layers==4 && metrics.fills==4-prefix && metrics.completed_reader_fences==4-prefix,
            "encoder pass/ticket coverage incomplete");
    }
    TransformerConfig config;config.layers=4;config.heads=1;config.context_dim=128;
    auto input=mx::astype(mx::reshape(mx::sin(mx::arange(4*64,mx::float32)*.01f),{1,4,64}),mx::bfloat16);
    auto context=mx::astype(mx::reshape(mx::cos(mx::arange(3*128,mx::float32)*.013f),{1,3,128}),mx::bfloat16);mx::eval({input,context});
    for(uint32_t prefix:{0u,1u,2u}) {
        Bf16StreamBank bank(bf16_stream_plan(argv[2],false,4,prefix,3,cancel),16ull<<20,cancel);
        Transformer reference(dit_weights,config),streamed(bank.resident(),config);
        streamed.set_layer_weights([&](int layer)->const Weights &{return bank.acquire(layer);},[&](int layer){bank.retire(layer);});
        auto a=input,b=input;
        for(uint32_t pass=0;pass<3;++pass) {
            const float sigma=.9f-.2f*pass;auto expected=reference.forward(a,context,sigma,2,2,true);
            bank.begin_pass(pass);auto actual=streamed.forward(b,context,sigma,2,2,true);
            equal(actual,expected,"streamed compiled DiT captured stale slot weights or changed original BF16 arithmetic");
            a=a+expected*Tensor(-.2f,mx::bfloat16);b=b+actual*Tensor(-.2f,mx::bfloat16);mx::eval({a,b});bank.finish_pass();
        }
        bank.finish();const auto metrics=bank.metrics();
        require(metrics.completed_passes==3 && metrics.completed_layers==12 && metrics.completed_prefix_layers==prefix*3 &&
            metrics.completed_streamed_layers==(4-prefix)*3 && metrics.fills==(4-prefix)*3 &&
            metrics.reader_fences==metrics.fills && metrics.completed_reader_fences==metrics.fills && metrics.drained,
            "DiT three-pass source/last-reader coverage incomplete");
        equal(a,b,"streamed complete Euler sequence changed BF16 latents");
    }
    {
        auto plan=bf16_stream_plan(argv[2],false,4,1,1,cancel);
        rejects([&]{Bf16StreamBank floor(plan,1,cancel);});
        Bf16StreamBank bank(std::move(plan),16ull<<20,cancel);bank.begin_pass(0);
        rejects([&]{bank.acquire(1);}); // no source may skip a prefix/layer
    }
    {
        Bf16StreamBank bank(bf16_stream_plan(argv[2],false,4,0,1,cancel),16ull<<20,cancel);
        bank.begin_pass(0);cancel.store(true);rejects([&]{bank.acquire(0);});
    }
    cancel.store(false);rejects([&]{bf16_stream_plan(argv[3],true,4,0,1,cancel);});
    {
        const auto path=std::filesystem::canonical(argv[4]);
        require(path.filename()=="owned-change.safetensors" && path.parent_path().filename().string().starts_with("tc-qwen-bf16-stream-"),
            "refusing to mutate a non-owned BF16 fixture");
        Bf16StreamBank bank(bf16_stream_plan(path,false,4,0,1,cancel),16ull<<20,cancel);
        bank.begin_pass(0);const int fd=::open(path.c_str(),O_WRONLY|O_CLOEXEC);require(fd>=0,"cannot open owned fault fixture");
        const int status=::ftruncate(fd,0);::close(fd);require(status==0,"cannot truncate owned fault fixture");
        rejects([&]{bank.acquire(0);});
    }
    mx::synchronize();std::cout<<"PASS Qwen original-BF16 streaming: encoder/compiled DiT exact across prefixes0/1/2, three KV/Euler passes, slot epochs, budget/cancel/source-change/closure negatives\n";
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}

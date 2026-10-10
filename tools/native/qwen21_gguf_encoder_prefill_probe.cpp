#include "models/qwen21/gguf_weights.hpp"
#include "models/qwen21/conditioning.hpp"
#include "models/qwen21/encoder_prefill.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace tc;
int main(int argc,char **argv) {try {
    require(argc==4,"usage: qwen21-gguf-encoder-prefill-probe model-root prompt hot-repeats");
    const auto root=std::filesystem::path(argv[1]);const int repeats=std::stoi(argv[3]);
    require(repeats>=3 && repeats<=9,"bounded cyclic3..9 component samples required");
    configure_streams();mx::set_cache_limit(0);std::atomic<bool> cancelled{false};
    auto elapsed=[](auto start){return std::chrono::duration<double>(Clock::now()-start).count();};
    Weights weights;const auto start=Clock::now();
    auto source=qwen21::load_gguf_component(root/"text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",weights,true,{},cancelled);
    const auto load_seconds=elapsed(start);
    Tokenizer tokenizer(root/"processor");auto tokens=tokenizer.raw_bounded(qwen21::reference_prompt_template(argv[2],0),Tokenizer::qwen21_limit);
    auto input=qwen21::assemble_prompt(tokens,weights,{});mx::eval({input.embeddings,input.positions});
    qwen21::TextConfig legacy;legacy.final_norm=false;auto compiled=legacy;compiled.compiled_gpu_blocks=true;
    qwen21::TextEncoder plain(weights,legacy),optimized(weights,compiled);
    auto fusion_start=Clock::now();Weights fused_weights=weights;auto fused_pack=qwen21::pack_encoder_prefill(fused_weights,cancelled);
    const auto fusion_seconds=elapsed(fusion_start);auto prefill=compiled;prefill.fused_gpu_prefill=true;
    qwen21::TextEncoder fused(fused_weights,prefill);
    const std::vector<const qwen21::TextEncoder *> encoders{&plain,&optimized,&fused};
    auto encode=[&](const qwen21::TextEncoder &encoder) {
        auto hidden=input.retain(encoder.encode_embeddings(input.embeddings,input.positions,input.embeddings.shape(1),{},cancelled));
        mx::eval(hidden);require(mx::all(mx::isfinite(hidden)).item<bool>(),"nonfinite real GGUF encoder output");return hidden;
    };
    auto reference=encode(plain);
    std::vector<double> first,errors;std::vector<std::vector<double>> times(3);
    auto error=[&](const Tensor &value) {
        auto a=mx::astype(value,mx::float32),b=mx::astype(reference,mx::float32);
        return mx::sqrt(mx::sum(mx::square(a-b))/mx::maximum(mx::sum(mx::square(b)),Tensor(1e-20f))).item<float>();
    };
    for(const auto *encoder:encoders) {
        auto t=Clock::now();auto y=encode(*encoder);first.push_back(elapsed(t));errors.push_back(error(y));
        require(std::isfinite(errors.back()) && errors.back()<.08,"compiled prefill exceeds8% conditioning screening budget");
    }
    for(int iteration=0;iteration<repeats;++iteration)for(int offset=0;offset<3;++offset) {
        const auto arm=(iteration+offset)%3;const auto t=Clock::now();auto y=encode(*encoders[arm]);times[arm].push_back(elapsed(t));
        require(error(y)<.08,"hot real encoder exceeds conditioning screening budget");
    }
    source->bank->check_unchanged();const auto metrics=source->bank->metrics();
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen21-gguf-encoder-prefill-v1\",\"qualification_passed\":false,"
        <<"\"scope\":\"actual original mixed-GGUF GPU language encoder; shared prepared inputs/source; no ANE/DiT/request claim\","
        <<"\"source_sha256\":\""<<metrics.source_sha256<<"\",\"source_load_seconds\":"<<load_seconds
        <<",\"rows\":"<<input.embeddings.shape(1)<<",\"retained_rows\":"<<reference.shape(1)<<",\"layers\":36,\"recipes\":[";
    for(size_t i=0;i<3;++i) {
        if(i)std::cout<<',';std::cout<<"{\"name\":\""<<(i==2 ? "fused_gpu_prefill" : i ? "compiled_native_gqa" : "legacy_fp32_repeated_kv")
            <<"\",\"first_seconds\":"<<first[i]<<",\"relative_l2\":"<<errors[i]<<",\"seconds\":[";
        for(size_t j=0;j<times[i].size();++j){if(j)std::cout<<',';std::cout<<times[i][j];}std::cout<<"]}";
    }
    std::cout<<"],\"fused_source_pack_seconds\":"<<fusion_seconds<<",\"qkv_fused_layers\":"<<fused_pack.qkv
        <<",\"qk_fused_layers\":"<<fused_pack.qk
        <<",\"gate_up_fused_layers\":"<<fused_pack.gate_up<<",\"mlx_logical_peak_bytes\":"<<mx::get_peak_memory()<<"}\n";
    return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}

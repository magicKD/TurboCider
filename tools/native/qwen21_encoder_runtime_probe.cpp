#include "../../native/models/qwen21/conditioning.hpp"
#include "../../native/backends/ane_ffn.hpp"
#include "../../native/media/image.hpp"
#include <iomanip>
#include <iostream>

// Actual local language/vision weights, but no DiT, downloads, activation
// dumps, or media output. Hot samples retain source owners and executor;
// production's request-local encoder reloads them, so this is not E2E proof.
int main(int argc, char **argv) {
    try {
        using namespace tc;
        require(argc >= 6 && argc <= 8,
            "usage: qwen21-encoder-runtime-probe model-root manifest prompt reference-size warm-repeats [ref1 [ref2]]");
        const auto root=std::filesystem::path(argv[1]);
        const int reference_size=std::stoi(argv[4]),repeats=std::stoi(argv[5]);
        require(reference_size==512 && repeats>=1 && repeats<=9,"probe requires 512 reference size and 1..9 repeats");
        configure_streams();std::atomic<bool> cancelled{false};
        Weights weights;weights.load_file(root/"text_encoders/qwen3vl_8b_bf16.safetensors");
        weights.materialize();
        Tokenizer tokenizer(root/"processor");
        auto tokens=tokenizer.raw_bounded(qwen21::reference_prompt_template(argv[3],argc-6),Tokenizer::qwen21_limit);
        std::vector<qwen21::VisualReference> references;
        qwen21::VisionEncoder vision(weights);
        const auto vision_start=Clock::now();
        for (int i=6;i<argc;++i) {
            auto pixels=qwen21::resize_reference(load_rgba_image_tensor(argv[i]),reference_size);
            auto alpha=slice_axis(pixels,3,3,4);
            auto input=qwen21::vision_input(slice_axis(pixels,3,0,3)*alpha+(1.f-alpha));
            references.push_back({vision.encode(input.patches,input.grid_height,input.grid_width,{},cancelled),
                input.grid_height,input.grid_width});
        }
        auto assembled=qwen21::assemble_prompt(tokens,weights.at("model.embed_tokens.weight"),references);
        mx::eval(assembled.embeddings);
        for(const auto &delta:assembled.deepstack_deltas)mx::eval(delta);
        const double vision_seconds=std::chrono::duration<double>(Clock::now()-vision_start).count();
        qwen21::TextConfig config;config.final_norm=false;
        qwen21::TextEncoder gpu(weights,config);
        auto encode=[&](const qwen21::TextEncoder &encoder) {
            auto output=assembled.retain(encoder.encode_embeddings(assembled.embeddings,assembled.positions,
                assembled.embeddings.shape(1),{},cancelled,assembled.deepstack_deltas));
            mx::eval(output);return output;
        };
        auto start=Clock::now();auto expected=encode(gpu);
        const double gpu_first=std::chrono::duration<double>(Clock::now()-start).count();
        start=Clock::now();
        ane::HybridFfn runtime(argv[2],4096,12288,1ull<<30,cancelled);runtime.begin_request();
        qwen21::TextEncoder hybrid(weights,config,&runtime);
        auto actual=encode(hybrid);runtime.drain();
        const double hybrid_first=std::chrono::duration<double>(Clock::now()-start).count();
        auto error=[&](const Tensor &value) {
            return mx::sqrt(mx::sum(mx::square(mx::astype(value,mx::float32)-mx::astype(expected,mx::float32)))/
                mx::sum(mx::square(mx::astype(expected,mx::float32)))).item<float>();
        };
        const double cold_error=error(actual);
        std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen21-encoder-component-v1\","
            "\"scope\":\"host language component; retained source owners/executor; not production request or physical overlap proof\","
            "\"rows\":"<<assembled.embeddings.shape(1)<<",\"references\":"<<references.size()
            <<",\"vision_assembly_seconds\":"<<vision_seconds<<",\"gpu_first_seconds\":"<<gpu_first
            <<",\"hybrid_first_with_setup_seconds\":"<<hybrid_first<<",\"cold_rel_l2\":"<<cold_error
            <<",\"samples\":[";
        for(int i=0;i<repeats;++i) {
            double gpu_seconds=0,hybrid_seconds=0;
            Tensor candidate(0.f);const auto calls_before=runtime.metrics().runtime_calls;
            auto measure_gpu=[&] {start=Clock::now();expected=encode(gpu);
                gpu_seconds=std::chrono::duration<double>(Clock::now()-start).count();};
            auto measure_hybrid=[&] {runtime.begin_request();start=Clock::now();candidate=encode(hybrid);runtime.drain();
                hybrid_seconds=std::chrono::duration<double>(Clock::now()-start).count();};
            if(i%2) {measure_hybrid();measure_gpu();} else {measure_gpu();measure_hybrid();}
            if(i)std::cout<<',';
            std::cout<<"{\"iteration\":"<<i<<",\"order\":\""<<(i%2?"hybrid,gpu":"gpu,hybrid")
                <<"\",\"gpu_seconds\":"<<gpu_seconds<<",\"hybrid_seconds\":"<<hybrid_seconds
                <<",\"rel_l2\":"<<error(candidate)<<",\"actual_calls\":"<<runtime.metrics().runtime_calls-calls_before<<'}';
        }
        auto metrics=runtime.metrics();
        require(!metrics.runtime_failed,"actual encoder component failed");
        std::cout<<"],\"backend\":\""<<metrics.runtime_weight_backend<<"\",\"data_path\":\""<<metrics.runtime_weight_data_path
            <<"\",\"partition_axis\":\""<<metrics.runtime_weight_partition_axis<<"\",\"ane_channels\":"<<metrics.runtime_weight_ane_channels
            <<",\"bucket\":"<<metrics.bucket<<",\"calls_session_total\":"<<metrics.runtime_calls
            <<",\"load_seconds\":"<<metrics.load_seconds<<",\"warmup_seconds\":"<<metrics.zero_input_warmup_seconds
            <<",\"stage_seconds_session_total\":"<<metrics.runtime_weight_stage_seconds
            <<",\"pre_ffn_seconds_session_total\":"<<metrics.runtime_weight_pre_seconds
            <<",\"hybrid_ffn_seconds_session_total\":"<<metrics.runtime_weight_wall_seconds
            <<",\"prediction_seconds_session_total\":"<<metrics.prediction_seconds
            <<",\"scale_cache_hits\":"<<metrics.runtime_weight_scale_cache_hits
            <<",\"scale_cache_misses\":"<<metrics.runtime_weight_scale_cache_misses
            <<",\"fallback_blocks\":"<<metrics.runtime_weight_fallback_blocks
            <<",\"qualification_passed\":false}\n";
    } catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}

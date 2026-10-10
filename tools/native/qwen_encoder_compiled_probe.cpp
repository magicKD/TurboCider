#include "../../native/models/qwen21/conditioning.hpp"
#include "../../native/backends/ane_ffn.hpp"
#include "../../native/media/image.hpp"
#include <iomanip>
#include <iostream>
#include <cmath>

int main(int argc,char **argv) {try {
    using namespace tc;
    require(argc>=5 && argc<=7,"usage: qwen-encoder-compiled-probe model-root manifest prompt repeats [ref1 [ref2]]");
    const auto root=std::filesystem::path(argv[1]);const int repeats=std::stoi(argv[4]);
    require(repeats>=3 && repeats<=9,"need3..9 cyclic hot component samples");
    configure_streams();std::atomic<bool> cancelled{false};
    Weights weights;weights.load_file(root/"text_encoders/qwen3vl_8b_bf16.safetensors");weights.materialize();
    const int refs=argc-5;
    Tokenizer tokenizer(root/"processor");
    auto tokens=tokenizer.raw_bounded(qwen21::reference_prompt_template(argv[3],refs),Tokenizer::qwen21_limit);
    std::vector<qwen21::VisualReference> references;qwen21::VisionEncoder vision(weights);
    for(int i=5;i<argc;++i) {
        auto pixels=qwen21::resize_reference(load_rgba_image_tensor(argv[i]),512);
        auto alpha=slice_axis(pixels,3,3,4);
        auto input=qwen21::vision_input(slice_axis(pixels,3,0,3)*alpha+(1.f-alpha));
        references.push_back({vision.encode(input.patches,input.grid_height,input.grid_width,{},cancelled),input.grid_height,input.grid_width});
    }
    auto assembled=qwen21::assemble_prompt(tokens,weights.at("model.embed_tokens.weight"),references);
    mx::eval({assembled.embeddings,assembled.positions});for(const auto &delta:assembled.deepstack_deltas)mx::eval(delta);
    qwen21::TextConfig original;original.final_norm=false;
    auto compiled=original;compiled.compiled_gpu_blocks=true;
    qwen21::TextEncoder gpu(weights,original),gpu_compiled(weights,compiled);
    auto encode=[&](const qwen21::TextEncoder &encoder) {
        auto y=assembled.retain(encoder.encode_embeddings(assembled.embeddings,assembled.positions,assembled.embeddings.shape(1),{},cancelled,assembled.deepstack_deltas));
        mx::eval(y);return y;
    };
    auto expected=encode(gpu);auto start=Clock::now();
    ane::HybridFfn runtime(argv[2],4096,12288,1ull<<30,cancelled);runtime.begin_request();
    const auto setup_seconds=std::chrono::duration<double>(Clock::now()-start).count();
    qwen21::TextEncoder hybrid(weights,original,&runtime),hybrid_compiled(weights,compiled,&runtime);
    const std::vector<std::string> names{"gpu_eager","gpu_compiled","hybrid_eager","hybrid_compiled"};
    const std::vector<const qwen21::TextEncoder*> encoders{&gpu,&gpu_compiled,&hybrid,&hybrid_compiled};
    std::vector<double> cold,errors;std::vector<std::vector<double>> times(4);
    auto error=[&](const Tensor &value) {
        return mx::sqrt(mx::sum(mx::square(mx::astype(value,mx::float32)-mx::astype(expected,mx::float32)))/
            mx::sum(mx::square(mx::astype(expected,mx::float32)))).item<float>();
    };
    for(size_t i=0;i<encoders.size();++i) {
        if(i>=2)runtime.begin_request();start=Clock::now();auto y=encode(*encoders[i]);if(i>=2)runtime.drain();
        cold.push_back(std::chrono::duration<double>(Clock::now()-start).count());
        errors.push_back(error(y));require(std::isfinite(errors.back()) && errors.back()<.08,"encoder conditioning exceeds8% component budget");
    }
    for(int iteration=0;iteration<repeats;++iteration)for(size_t j=0;j<encoders.size();++j) {
        const auto i=(iteration+j)%encoders.size();if(i>=2)runtime.begin_request();start=Clock::now();
        auto y=encode(*encoders[i]);if(i>=2)runtime.drain();times[i].push_back(std::chrono::duration<double>(Clock::now()-start).count());
        const auto e=error(y);require(std::isfinite(e) && e<.08,"hot encoder conditioning exceeds8% budget");
    }
    const auto metrics=runtime.metrics();require(!metrics.runtime_failed && metrics.runtime_weight_fallback_blocks==0,"encoder component fell back or failed");
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-encoder-compiled-component-v1\","
        "\"scope\":\"real language conditioning; shared prepared vision/embedding/mRoPE inputs and original retained source/executor; no DiT/media or whole-request/physical qualification\","
        "\"rows\":"<<assembled.embeddings.shape(1)<<",\"references\":"<<refs<<",\"setup_seconds\":"<<setup_seconds<<",\"recipes\":[";
    for(size_t i=0;i<4;++i) {
        if(i)std::cout<<',';std::cout<<"{\"name\":\""<<names[i]<<"\",\"first_seconds\":"<<cold[i]<<",\"relative_l2\":"<<errors[i]<<",\"seconds\":[";
        for(size_t j=0;j<times[i].size();++j){if(j)std::cout<<',';std::cout<<times[i][j];}std::cout<<"]}";
    }
    std::cout<<"],\"actual_ane_calls\":"<<metrics.runtime_calls<<",\"ane_channels\":"<<metrics.runtime_weight_ane_channels
        <<",\"bucket\":"<<metrics.bucket<<",\"qualification_passed\":false}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;} }

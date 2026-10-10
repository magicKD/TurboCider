// Bounded BaseOnly component replay. No model downloads, ANE program exports,
// adapter mutations or complete-model materialization.
#include "../../native/backends/ane_smoothquant.hpp"
#include "../../native/backends/metal_graph/mlx_ffn_island.hpp"
#include "../../native/models/qwen21/runtime_ffn_graphs.hpp"
#include "../../native/platform/apple/platform.hpp"
#include "turbocider/turbocider.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace tc;
namespace mg = tc::metal_graph;
namespace ff = tc::qwen21::runtime_ffn;
struct Sample { int step; Tensor input; std::string digest; };
struct Result {
    int step;
    std::string digest;
    double rrms;
    std::array<double,2> gpu_ms{}, native_wall_ms{}, native_device_ms{};
};
static double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
static std::string quote(const std::string &s) {
    std::string out = "\"";
    for (unsigned char c:s) {
        require(c>=32,"unexpected control character in replay identity");
        if (c=='"' || c=='\\') out+='\\';
        out+=char(c);
    }
    return out+'"';
}
static double relative_rms(const Tensor &candidate,const Tensor &reference) {
    auto a=mx::astype(candidate,mx::float32),b=mx::astype(reference,mx::float32);
    auto delta=a-b;
    auto metric=mx::sqrt(mx::sum(delta*delta)/
        mx::maximum(mx::sum(b*b),Tensor(1e-20f)));
    mx::eval(metric);
    const double value=metric.item<float>();
    require(std::isfinite(value),"native FFN produced nonfinite metric");
    return value;
}
static void pair(std::ostream &out,const std::array<double,2> &a) {
    out<<'['<<a[0]<<','<<a[1]<<']';
}
int main(int argc,char **argv) {
    try {
        require(argc==5,"usage: checkpoint capture-dir output.json 12288|6144");
        configure_streams();
        const auto checkpoint=std::filesystem::canonical(argv[1]);
        const auto capture=std::filesystem::canonical(argv[2]);
        const std::filesystem::path output=argv[3];
        const int channels=std::stoi(argv[4]);
        require(channels==12288 || channels==6144,"unbounded native replay width");
        require(!std::filesystem::exists(output),"refusing to overwrite replay result");
        require(std::filesystem::is_regular_file(capture/"complete.json"),
                "capture lacks completion receipt");
        const auto identity=ane::smoothquant::checkpoint_fingerprint(checkpoint);
        std::vector<Sample> samples;
        std::string prompt,seed;
        for(int step:{1,19}) {
            const auto path=capture/("layer-0-step-"+std::to_string(step)+".safetensors");
            require(!std::filesystem::is_symlink(path) &&
                    std::filesystem::file_size(path)<9*1024*1024,
                    "bounded original input file required");
            auto [arrays,m]=mx::load_safetensors(path.string());
            require(m.at("capture_recipe")=="qwen21-real-ffn-bf16-v1" &&
                    m.at("trajectory")=="forced-gpu" && m.at("adapter_identity").empty() &&
                    m.at("model_identity")==identity && m.at("total_steps")=="20" &&
                    m.at("layer")=="0" && m.at("step")==std::to_string(step),
                    "original BaseOnly capture/model provenance mismatch");
            if(samples.empty()) { prompt=m.at("prompt");seed=m.at("seed"); }
            require(m.at("prompt")==prompt && m.at("seed")==seed,
                    "mixed prompt/seed trajectory");
            require(arrays.size()==1 && arrays.at("input").shape()==mx::Shape{1,1024,4096} &&
                    arrays.at("input").dtype()==mx::bfloat16,
                    "replay requires full original BF16 matrix");
            samples.push_back({step,arrays.at("input"),sha256_file(path)});
            mx::eval(samples.back().input);
        }
        Weights weights;
        weights.load_file(checkpoint);
        const std::string prefix="transformer_blocks.0.img_mlp.";
        auto fused=weights.at(prefix+"gate_up.weight");
        auto gate=slice_axis(fused,0,0,12288);
        auto up=slice_axis(fused,0,12288,24576);
        auto down=weights.at(prefix+"out.weight");
        require(fused.shape()==mx::Shape{24576,4096} &&
                down.shape()==mx::Shape{4096,12288} &&
                fused.dtype()==mx::bfloat16 && down.dtype()==mx::bfloat16,
                "native FFN source must be local dense BF16 Qwen");
        // Only this layer's selected source tensors are evaluated. Import
        // retains allocator owners and preserves fused source row offsets.
        mx::eval({gate,up,down});
        ff::Function gpu=channels==12288 ? ff::full(weights,prefix) :
            ff::channels(weights,prefix,0,channels);
        mg::Backend backend;
        const auto prepare_start=Clock::now();
        mg::MlxFfnIsland native(backend,{4096,12288,uint32_t(channels)});
        const double prepare_ms=milliseconds(prepare_start);
        std::vector<Result> results;
        for(const auto &s:samples) {
            auto reference=gpu({s.input});mx::eval(reference);
            auto candidate=native.run(s.input,gate,up,down);
            const double rrms=relative_rms(candidate.output,reference[0]);
            // User screening tolerance; not a full-model quality certificate.
            require(rrms<=0.08,"native FFN RRMS exceeds bounded screening tolerance");
            Result result{s.step,s.digest,rrms,{},{},{}};
            for(size_t round=0;round<2;++round) {
                const auto run_gpu=[&] {
                    const auto start=Clock::now();
                    auto value=gpu({s.input});mx::eval(value);
                    result.gpu_ms[round]=milliseconds(start);
                };
                const auto run_native=[&] {
                    const auto start=Clock::now();
                    auto value=native.run(s.input,gate,up,down);
                    result.native_wall_ms[round]=milliseconds(start);
                    result.native_device_ms[round]=value.completion.gpu_seconds*1000.;
                };
                if(round==0) { run_gpu();run_native(); }
                else { run_native();run_gpu(); }
            }
            results.push_back(std::move(result));
        }
        require(ane::smoothquant::checkpoint_fingerprint(checkpoint)==identity,
                "checkpoint stat identity changed during component replay");
        char *raw=tc_runtime_build_identity();
        require(raw,"missing existing GPU library identity");
        const std::string gpu_build(raw);tc_string_free(raw);
        std::ofstream out(output);
        out<<std::setprecision(12)
           <<"{\"schema\":\"native-qwen-ffn-replay-v1\","
           <<"\"scope\":\"BaseOnly layer0 FFN W0 island; not native DiT, ANE, E2E or qualification\","
           <<"\"source_identity_kind\":\"canonical-stat-identity\","
           <<"\"source_identity\":"<<quote(identity)
           <<",\"existing_gpu_library_build\":"<<quote(gpu_build)
           <<",\"native_recipe\":"<<quote(std::string(mg::NativeFfn::precision_recipe()))
           <<",\"device\":"<<quote(backend.device_name())
           <<",\"channels\":"<<channels<<",\"rows\":1024,\"prepare_ms\":"<<prepare_ms
           <<",\"native_wall_scope\":\"allocation/import/encode/submit/physical completion; PSO prep separate\","
           <<"\"samples\":[";
        for(size_t i=0;i<results.size();++i) {
            if(i)out<<',';
            const auto &r=results[i];
            out<<"{\"layer\":0,\"step\":"<<r.step<<",\"input_sha256\":"<<quote(r.digest)
               <<",\"rrms\":"<<r.rrms<<",\"gpu_ms\":";pair(out,r.gpu_ms);
            out<<",\"native_wall_ms\":";pair(out,r.native_wall_ms);
            out<<",\"native_device_ms\":";pair(out,r.native_device_ms);out<<'}';
        }
        out<<"]}\n";out.close();
        require(bool(out),"failed writing component receipt");
        std::cout<<output<<'\n';
        return 0;
    } catch(const std::exception &error) {
        std::cerr<<"native FFN replay failed: "<<error.what()<<'\n';return 1;
    }
}

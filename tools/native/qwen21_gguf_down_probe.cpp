#include "models/qwen21/gguf_weights.hpp"
#include <iomanip>
#include <iostream>

using namespace tc;
int main(int argc,char **argv) {try {
    require(argc==2,"usage: qwen21-gguf-down-probe original-denoiser");
    configure_streams();mx::set_cache_limit(0);
    streaming::SourceFileIdentity source;source.logical_id="denoiser";source.path=argv[1];
    auto lease=streaming::SourceLease::capture_verified({source});
    const std::string name="model.diffusion_model.transformer_blocks.0.img_mlp.out.weight";
    MemoryLedger ledger(uint64_t(256)<<20);Weights weights;
    streaming::GgufKImportOptions options;options.enabled=true;options.floating_dtype=mx::float16;
    options.include_tensor=[name](std::string_view key){return key==name;};
    streaming::GgufPackedBank bank(lease,"denoiser",ledger,uint64_t(1)<<20,true,0,6,options);bank.load(weights);
    const auto prefix=name.substr(0,name.size()-7);
    using Function=std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen21-packed-shared-down-v1\",\"source_sha256\":\""<<lease->file("denoiser").content_digest
        <<"\",\"scope\":\"real original layer0 affine Q4 down, synthetic input, complete evaluated host projection spans; not model/performance qualification\",\"qualification_passed\":false,\"cases\":[";
    bool comma=false;
    for(int m:{1024,1048})for(bool partial:{false,true}) {
        const int k=partial?8192:12288;
        auto input=mx::astype(mx::reshape(mx::sin(mx::arange(m*k,mx::float32)*.00017f)*.125f,{1,m,k}),mx::float16);mx::eval(input);
        weights.set_qwen_affine_shared_down(false);
        Function control=mx::compile([&weights,prefix,partial,k](const std::vector<Tensor>&a) {
            return std::vector<Tensor>{partial ? weights.project_base_slice(a[0],prefix,0,4096,0,k,false) : weights.project(a[0],prefix)};
        });
        auto oracle=control({input})[0];mx::eval(oracle);
        weights.set_qwen_affine_shared_down(true);
        Function candidate=mx::compile([&weights,prefix,partial,k](const std::vector<Tensor>&a) {
            return std::vector<Tensor>{partial ? weights.project_base_slice(a[0],prefix,0,4096,0,k,false) : weights.project(a[0],prefix)};
        });
        auto actual=candidate({input})[0];mx::eval(actual);
        const auto difference=mx::astype(actual,mx::float32)-mx::astype(oracle,mx::float32);
        const double error=mx::sqrt(mx::sum(mx::square(difference))/mx::maximum(mx::sum(mx::square(mx::astype(oracle,mx::float32))),Tensor(1e-20f))).item<float>();
        require(actual.dtype()==mx::float16 && actual.shape()==oracle.shape() && mx::all(mx::isfinite(actual)).item<bool>() &&
                std::isfinite(error) && error<=.05,"shared packed down exceeds original typed QMM budget");
        for(int warm=0;warm<3;++warm){mx::eval(control({input}));mx::eval(candidate({input}));}
        std::vector<double> times[2];
        for(int sample=0;sample<9;++sample)for(int visit=0;visit<2;++visit) {
            const int index=(visit+sample)%2;const auto started=Clock::now();
            mx::eval((index?candidate:control)({input}));times[index].push_back(std::chrono::duration<double,std::milli>(Clock::now()-started).count());
        }
        if(comma)std::cout<<',';comma=true;
        std::cout<<"{\"M\":"<<m<<",\"K\":"<<k<<",\"relative_l2\":"<<error<<",\"qmm_ms\":[";
        for(size_t i=0;i<times[0].size();++i)std::cout<<(i?",":"")<<times[0][i];
        std::cout<<"],\"shared_mpp_ms\":[";
        for(size_t i=0;i<times[1].size();++i)std::cout<<(i?",":"")<<times[1][i];std::cout<<"]}";
    }
    bank.check_unchanged();std::cout<<"],\"warmups\":3,\"hot_samples\":9}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

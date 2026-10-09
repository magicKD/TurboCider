#include "../../native/backends/mlx.hpp"
#include <iomanip>
#include <iostream>

int main(int argc,char **argv){try {
    using namespace tc;require(argc==2,"usage: convrot-narrow-partial-probe original-local-checkpoint");configure_streams();
    Weights weights;weights.load_file(argv[1],"layers.0.feed_forward.");weights.pack_convrot_q8();weights.materialize();
    weights.set_metal_convrot(true);weights.set_affine_fp32_mpp(true);
    require(weights.at("w2.weight").shape()==mx::Shape{3840,2560} && weights.at("w2.scales").dtype()==mx::bfloat16,
            "original packed ConvRot legacy BF16-scale down source mismatch");
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-convrot-narrow-partial-component-v1\",\"scope\":\"original layer0 ConvRot packed g32/BF16-scale down, synthetic BF16 hidden; compiled rotation plus MPP F32 partial vs original packed-QMM BF16 then F32 widen; every invocation executes the consumer; serial host spans, not FFN/model/physical overlap qualification\",\"cases\":[";
    bool comma=false;
    for(int m:{33,1024,1056})for(int k:{5120,6144,8192}) {
        auto hidden=mx::reshape(mx::astype(mx::sin(mx::arange(m*k,mx::float32)*.001f)*.125f,mx::bfloat16),{1,m,k});mx::eval(hidden);
        using Fn=std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;
        const std::vector<Fn> recipes{
            mx::compile([&,k](const std::vector<Tensor>&a){return std::vector<Tensor>{weights.project_base_slice_fp32(a[0],"w2",0,3840,0,k)};}),
            mx::compile([&,k](const std::vector<Tensor>&a){return std::vector<Tensor>{mx::astype(weights.project_base_slice(a[0],"w2",0,3840,0,k,false),mx::float32)};})};
        auto expected=recipes[0]({hidden})[0];mx::eval(expected);std::vector<float> errors;
        std::vector<std::vector<double>> samples(2);
        for(const auto &fn:recipes) {
            auto actual=fn({hidden})[0];mx::eval(actual);
            const float error=mx::sqrt(mx::sum(mx::square(actual-expected))/mx::maximum(mx::sum(mx::square(expected)),Tensor(1e-20f))).item<float>();
            require(mx::all(mx::isfinite(actual)).item<bool>() && std::isfinite(error) && error<.05f,"narrow partial exceeded5% component budget");
            errors.push_back(error);for(int warm=0;warm<3;++warm)mx::eval(fn({hidden}));
        }
        for(int hot=0;hot<15;++hot)for(int visit=0;visit<2;++visit) {
            const int recipe=(hot+visit)%2;const auto start=Clock::now();mx::eval(recipes[recipe]({hidden}));
            samples[recipe].push_back(std::chrono::duration<double>(Clock::now()-start).count());
        }
        if(comma)std::cout<<',';comma=true;std::cout<<"{\"M\":"<<m<<",\"K\":"<<k<<",\"N\":3840,\"physical_K\":10240,\"recipes\":[";
        for(int recipe=0;recipe<2;++recipe){if(recipe)std::cout<<',';
            std::cout<<"{\"name\":\""<<(recipe?"BF16_QMM_then_F32":"MPP_F32")<<"\",\"relative_l2\":"<<errors[recipe]<<",\"seconds\":[";
            for(size_t i=0;i<samples[recipe].size();++i){if(i)std::cout<<',';std::cout<<samples[recipe][i];}std::cout<<"]}";}
        std::cout<<"]}";
    }
    std::cout<<"],\"qualification_passed\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}

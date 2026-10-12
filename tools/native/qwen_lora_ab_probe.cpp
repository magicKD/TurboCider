#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include "../../native/backends/dense_gpu_lora_b.hpp"
#include <iomanip>
#include <iostream>

int main(int argc,char **argv){try {
    using namespace tc;require(argc==2,"usage: qwen-lora-ab-probe original-local-adapter");configure_streams();
    Weights w;w.load_file(argv[1],"transformer.transformer_blocks.0.img_mlp.");w.materialize();
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-lora-joint-ab-v1\",\"scope\":\"real layer0 BF16 A/B, synthetic activation; complete online A-rank plus B/scale/delta cast, four compiled policies; serial host spans, not model/visual/physical GPU qualification\",\"cases\":[";
    bool comma=false;
    for(int m:{1024,2096,3144})for(int part:{0,1,2}) {
        const std::string stem=part==2?"out":"gate_layer";const auto &a=w.at(stem+".lora_A.weight"),&b=w.at(stem+".lora_B.weight");
        const int k=a.shape(1),first=part==1?7168:0,n=part==2?4096:part==1?5120:7168;
        require(a.dtype()==mx::bfloat16 && b.dtype()==mx::bfloat16 && a.shape(0)==256,"real BF16 r256 adapter mismatch");
        auto x=mx::reshape(mx::astype(mx::sin(mx::arange(m*k,mx::float32)*.001f)*.125f,mx::bfloat16),{1,m,k});mx::eval(x);
        using Fn=std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;std::vector<Fn> recipes;
        for(int recipe=0;recipe<4;++recipe)recipes.push_back(mx::compile([&,recipe,k,first,n](const std::vector<Tensor> &v) {
            auto low=(recipe&1)?dense_gpu::projection_range(v[0],a,0,256,0,k,16,true,64):
                mx::matmul(mx::astype(v[0],mx::float32),mx::transpose(mx::astype(a,mx::float32)));
            auto y=(recipe&2)?dense_gpu::lora_b_epilogue(low,b,first,first+n,.75f,mx::bfloat16):
                mx::astype(mx::matmul(low,mx::transpose(mx::astype(slice_axis(b,0,first,first+n),mx::float32)))*Tensor(.75f,mx::float32),mx::bfloat16);
            return std::vector<Tensor>{y};
        }));
        auto expected=recipes[0]({x})[0];mx::eval(expected);std::vector<float> errors;std::vector<std::vector<double>> samples(4);
        for(auto &fn:recipes) {
            auto y=fn({x})[0];mx::eval(y);
            auto af=mx::astype(y,mx::float32),ef=mx::astype(expected,mx::float32);
            const auto error=mx::sqrt(mx::sum(mx::square(af-ef))/mx::maximum(mx::sum(mx::square(ef)),Tensor(1e-20f))).item<float>();
            require(std::isfinite(error) && error<=.05f,"joint A/B exceeded5% component delta budget");errors.push_back(error);
            for(int warm=0;warm<3;++warm)mx::eval(fn({x}));
        }
        for(int sample=0;sample<15;++sample)for(int visit=0;visit<4;++visit) {
            const int i=(sample+visit)%4;const auto start=Clock::now();mx::eval(recipes[i]({x}));
            samples[i].push_back(std::chrono::duration<double>(Clock::now()-start).count());
        }
        if(comma)std::cout<<',';comma=true;std::cout<<"{\"M\":"<<m<<",\"K\":"<<k<<",\"N\":"<<n<<",\"first\":"<<first<<",\"projection\":\""<<stem<<"\",\"recipes\":[";
        const char *names[]={"original","bf16_a_only","fused_b_only","bf16_a_fused_b"};
        for(int i=0;i<4;++i){if(i)std::cout<<',';std::cout<<"{\"name\":\""<<names[i]<<"\",\"delta_rel_l2\":"<<errors[i]<<",\"seconds\":[";
            for(size_t j=0;j<samples[i].size();++j){if(j)std::cout<<',';std::cout<<samples[i][j];}std::cout<<"]}";}
        std::cout<<"]}";
    }
    std::cout<<"],\"qualification_passed\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}

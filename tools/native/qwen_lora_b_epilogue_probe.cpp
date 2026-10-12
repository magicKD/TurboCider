#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_lora_b.hpp"
#include <iomanip>
#include <iostream>

using namespace tc;
int main(int argc,char **argv){try {
    require(argc==2,"usage: qwen-lora-b-epilogue-probe local-original-adapter");configure_streams();
    Weights weights;weights.load_file(argv[1],"transformer.transformer_blocks.0.img_mlp.");weights.materialize();
    struct Recipe{int bm,bn;};const std::vector<Recipe> recipes{{0,0},{0,1},{16,64},{16,128},{32,64},{32,128},{64,64},{64,128}};
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-lora-b-fused-epilogue-v1\",\"scope\":\"real layer0 BF16 A/B, synthetic activation/prepared original F32 ranks, eager and compiled original F32 controls; rank narrowing, B/scale/optional base-add/output cast included; serial host spans, not full-model/visual/physical GPU qualification\",\"cases\":[";
    bool comma=false;
    for(int m:{1056,3137})for(int part:{0,1,2})for(bool add:{false,true}) {
        const bool down=part==2;const std::string stem=down?"out":"gate_layer";
        const auto &a=weights.at(stem+".lora_A.weight"),&b=weights.at(stem+".lora_B.weight");
        require(a.dtype()==mx::bfloat16 && b.dtype()==mx::bfloat16 && b.shape(1)==256,"real BF16 adapter fixture mismatch");
        const int k=a.shape(1),first=part==1?7168:0,n=down?4096:part==1?5120:7168;
        auto x=mx::astype(mx::reshape(mx::sin(mx::arange(m*k,mx::float32)*.001f)*.125f,{1,m,k}),mx::bfloat16);
        auto low=mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(a,mx::float32)));mx::eval(low);
        auto base=mx::astype(mx::reshape(mx::cos(mx::arange(m*(n+64),mx::float32)*.001f)*.03f,{1,m,n+64}),mx::bfloat16);mx::eval(base);
        auto original=[&,first,n,add](const std::vector<Tensor> &args) {
            auto y=mx::matmul(args[0],mx::transpose(mx::astype(slice_axis(b,0,first,first+n),mx::float32)))*Tensor(.75f,mx::float32);
            if(add)y=y+mx::astype(slice_axis(args[1],-1,17,17+n),mx::float32);
            return std::vector<Tensor>{mx::astype(y,mx::bfloat16)};
        };
        auto compiled=mx::compile(original);
        auto run=[&](size_t i) {
            if(i==0)return original({low,base})[0];if(i==1)return compiled({low,base})[0];
            return dense_gpu::lora_b_epilogue(low,b,first,first+n,.75f,mx::bfloat16,
                add?std::optional<Tensor>(base):std::nullopt,add?17:0,recipes[i].bm,recipes[i].bn);
        };
        auto expected=run(0);mx::eval(expected);std::vector<float> errors;std::vector<std::vector<double>> samples(recipes.size());
        for(size_t i=0;i<recipes.size();++i) {
            auto y=run(i);mx::eval(y);const float error=mx::sqrt(mx::sum(mx::square(mx::astype(y,mx::float32)-mx::astype(expected,mx::float32)))/mx::maximum(mx::sum(mx::square(mx::astype(expected,mx::float32))),Tensor(1e-20f))).item<float>();
            require(y.dtype()==mx::bfloat16 && std::isfinite(error) && error<=.05f,"fused B approximate component budget exceeded");errors.push_back(error);
            for(int warm=0;warm<3;++warm)mx::eval(run(i));
        }
        for(int sample=0;sample<15;++sample)for(size_t visit=0;visit<recipes.size();++visit) {
            const auto i=(sample+visit)%recipes.size();const auto start=Clock::now();mx::eval(run(i));samples[i].push_back(std::chrono::duration<double>(Clock::now()-start).count());
        }
        if(comma)std::cout<<',';comma=true;
        std::cout<<"{\"M\":"<<m<<",\"projection\":\""<<stem<<"\",\"first\":"<<first<<",\"N\":"<<n<<",\"add_base\":"<<(add?"true":"false")<<",\"recipes\":[";
        for(size_t i=0;i<recipes.size();++i) {
            std::cout<<(i?",":"")<<"{\"bm\":"<<recipes[i].bm<<",\"bn\":"<<recipes[i].bn<<",\"relative_l2\":"<<errors[i]<<",\"seconds\":[";
            for(size_t j=0;j<samples[i].size();++j)std::cout<<(j?",":"")<<samples[i][j];std::cout<<"]}";
        }
        std::cout<<"]}";
    }
    std::cout<<"],\"qualification_passed\":false}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}

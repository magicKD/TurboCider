#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include <iomanip>
#include <iostream>

using namespace tc;
int main(int argc,char **argv){try {
    require(argc==2,"usage: qwen-lora-b-operand-probe local-original-adapter");configure_streams();
    Weights weights;weights.load_file(argv[1],"transformer.transformer_blocks.0.img_mlp.");weights.materialize();
    struct Recipe{int bm,bn;};const std::vector<Recipe> recipes{{0,0},{16,64},{16,128},{32,64},{32,128},{64,64},{64,128}};
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-lora-b-bf16-operands-f32-delta-v1\","
        "\"scope\":\"original local BF16 adapter A/B, prepared original F32 ranks from synthetic BF16 activation; includes rank narrowing/B projection/F32 scaling; serial operator host spans, not whole-model/visual/GPU timestamp qualification\",\"cases\":[";
    bool comma=false;
    for(int rows:{1056,2113,3137})for(int part:{0,1,2}) {
        const bool down=part==2;const auto stem=down?"out":"gate_layer";
        const auto &a=weights.at(std::string(stem)+".lora_A.weight"),&b=weights.at(std::string(stem)+".lora_B.weight");
        require(a.dtype()==mx::bfloat16 && b.dtype()==mx::bfloat16 && a.shape(0)==256 && b.shape(1)==256,
            "original local BF16 LoRA geometry mismatch");
        const int k=a.shape(1),first=part==1?7168:0,count=down?4096:part==1?5120:7168;
        auto x=mx::astype(mx::reshape(mx::sin(mx::arange(rows*k,mx::float32)*.001f)*.125f,{1,rows,k}),mx::bfloat16);
        auto ranks=mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(a,mx::float32)));mx::eval(ranks);
        auto run=[&](size_t index) {
            if(index==0)return mx::matmul(ranks,mx::transpose(mx::astype(slice_axis(b,0,first,first+count),mx::float32)))*Tensor(.75f,mx::float32);
            return dense_gpu::projection_range(mx::astype(ranks,mx::bfloat16),b,first,first+count,0,256,
                recipes[index].bm,true,recipes[index].bn)*Tensor(.75f,mx::float32);
        };
        auto expected=run(0);mx::eval(expected);
        std::vector<float> errors,maxima;std::vector<std::vector<double>> samples(recipes.size());
        for(size_t index=0;index<recipes.size();++index) {
            auto y=run(index);mx::eval(y);const float error=mx::sqrt(mx::sum(mx::square(y-expected))/mx::maximum(mx::sum(mx::square(expected)),Tensor(1e-20f))).item<float>();
            require(y.dtype()==mx::float32 && std::isfinite(error) && error<=.05f,"approximate B/delta component budget exceeded");
            errors.push_back(error);maxima.push_back(mx::max(mx::abs(y-expected)).item<float>());
            for(int warm=0;warm<3;++warm)mx::eval(run(index));
        }
        for(int iteration=0;iteration<15;++iteration)for(size_t visit=0;visit<recipes.size();++visit) {
            const auto index=(visit+iteration)%recipes.size();const auto start=Clock::now();auto y=run(index);mx::eval(y);
            samples[index].push_back(std::chrono::duration<double>(Clock::now()-start).count());
        }
        if(comma)std::cout<<',';comma=true;
        std::cout<<"{\"M\":"<<rows<<",\"projection\":\""<<stem<<"\",\"first\":"<<first<<",\"N\":"<<count<<",\"rank\":256,\"recipes\":[";
        for(size_t index=0;index<recipes.size();++index) {
            std::cout<<(index?",":"")<<"{\"bm\":"<<recipes[index].bm<<",\"bn\":"<<recipes[index].bn<<",\"relative_l2\":"<<errors[index]<<",\"max_abs\":"<<maxima[index]<<",\"seconds\":[";
            for(size_t sample=0;sample<samples[index].size();++sample)std::cout<<(sample?",":"")<<samples[index][sample];
            std::cout<<"]}";
        }
        std::cout<<"]}";
    }
    std::cout<<"],\"qualification_passed\":false}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}

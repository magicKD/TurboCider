#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_base_lora.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include "../../native/backends/dense_gpu_lora_b.hpp"
#include <iomanip>
#include <iostream>

int main(int argc,char **argv){try {
    using namespace tc;require(argc==3,"usage: qwen-base-lora-probe original-checkpoint original-adapter");configure_streams();
    Weights base,adapter;base.load_file(argv[1],"transformer_blocks.0.img_mlp.gate_up.");base.materialize();
    adapter.load_file(argv[2],"transformer.transformer_blocks.0.img_mlp.");adapter.materialize();
    const auto &w=base.at("weight");require(w.shape()==mx::Shape{24576,4096} && w.dtype()==mx::bfloat16,"original gate/up source mismatch");
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-base-lora-fused-v1\",\"scope\":\"original layer0 dense BF16 base/A/B, synthetic input/prepared joint F32 ranks; compiled MLX base+B and MPP base+B vs fused two-matmul dispatch; same physical sources, base rounding boundary retained; host spans, not full FFN/model/physical GPU trace\",\"cases\":[";
    bool comma=false;
    for(int m:{1024,2096,3144})for(int n:{5120,7168})for(int part:{0,1}) {
        const auto stem=part?"proj":"gate_layer";const auto &a=adapter.at(std::string(stem)+".lora_A.weight"),&b=adapter.at(std::string(stem)+".lora_B.weight");
        const int rb=part*12288;
        auto x=mx::reshape(mx::astype(mx::sin(mx::arange(m*4096,mx::float32)*.001f)*.125f,mx::bfloat16),{1,m,4096});
        auto ranks=dense_gpu::projection_range(x,a,0,256,0,4096,16,true,64);mx::eval({x,ranks});
        using Fn=std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;
        std::vector<Fn> recipes;const std::vector<std::pair<int,int>> tiles{{16,64},{16,128},{32,64},{32,128},{64,64},{64,128}};
        recipes.push_back(mx::compile([&,rb,n](const std::vector<Tensor> &v){
            auto y=mx::matmul(v[0],mx::transpose(slice_axis(w,0,rb,rb+n)));
            return std::vector<Tensor>{dense_gpu::lora_b_epilogue(v[1],b,0,n,.75f,mx::bfloat16,y)};
        }));
        recipes.push_back(mx::compile([&,rb,n](const std::vector<Tensor> &v){
            auto y=dense_gpu::projection_range(v[0],w,rb,rb+n,0,4096,32,false,128);
            return std::vector<Tensor>{dense_gpu::lora_b_epilogue(v[1],b,0,n,.75f,mx::bfloat16,y)};
        }));
        for(auto [bm,bn]:tiles)recipes.push_back(mx::compile([&,rb,n,bm,bn](const std::vector<Tensor> &v){
            return std::vector<Tensor>{dense_gpu::base_lora_epilogue(v[0],w,v[1],b,rb,rb+n,0,4096,0,.75f,bm,bn)};
        }));
        auto expected=recipes[0]({x,ranks})[0];mx::eval(expected);std::vector<float> errors;std::vector<std::vector<double>> samples(recipes.size());
        for(auto &fn:recipes) {
            auto y=fn({x,ranks})[0];mx::eval(y);auto yf=mx::astype(y,mx::float32),ef=mx::astype(expected,mx::float32);
            const float e=mx::sqrt(mx::sum(mx::square(yf-ef))/mx::maximum(mx::sum(mx::square(ef)),Tensor(1e-20f))).item<float>();
            require(std::isfinite(e) && e<=.05f,"fused base/LoRA exceeded5% component budget");errors.push_back(e);
            for(int warm=0;warm<3;++warm)mx::eval(fn({x,ranks}));
        }
        for(int sample=0;sample<15;++sample)for(size_t visit=0;visit<recipes.size();++visit) {
            const auto i=(sample+visit)%recipes.size();const auto start=Clock::now();mx::eval(recipes[i]({x,ranks}));
            samples[i].push_back(std::chrono::duration<double>(Clock::now()-start).count());
        }
        if(comma)std::cout<<',';comma=true;std::cout<<"{\"M\":"<<m<<",\"N\":"<<n<<",\"base_row_begin\":"<<rb<<",\"physical_pitch\":4096,\"recipes\":[";
        for(size_t i=0;i<recipes.size();++i){if(i)std::cout<<',';std::cout<<"{\"bm\":"<<(i<2?0:tiles[i-2].first)<<",\"bn\":"<<(i<2?int(i):tiles[i-2].second)<<",\"relative_l2\":"<<errors[i]<<",\"seconds\":[";
            for(size_t j=0;j<samples[i].size();++j){if(j)std::cout<<',';std::cout<<samples[i][j];}std::cout<<"]}";}
        std::cout<<"]}";
    }
    std::cout<<"],\"qualification_passed\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}

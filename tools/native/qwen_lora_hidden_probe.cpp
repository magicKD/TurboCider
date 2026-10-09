#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include "../../native/backends/dense_gpu_lora_b.hpp"
#include "../../native/backends/dense_gpu_lora_hidden.hpp"
#include <iomanip>
#include <iostream>

int main(int argc,char **argv){try {
    using namespace tc;require(argc==3,"usage: qwen-lora-hidden-probe original-checkpoint original-adapter");configure_streams();
    Weights base,adapter;base.load_file(argv[1],"transformer_blocks.0.img_mlp.gate_up.");base.materialize();
    adapter.load_file(argv[2],"transformer.transformer_blocks.0.img_mlp.");adapter.materialize();
    const auto &w=base.at("weight"),&ag=adapter.at("gate_layer.lora_A.weight"),&bg=adapter.at("gate_layer.lora_B.weight"),
        &au=adapter.at("proj.lora_A.weight"),&bu=adapter.at("proj.lora_B.weight");
    require(w.shape()==mx::Shape{24576,4096} && w.dtype()==mx::bfloat16,"original fused gate/up source mismatch");
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-lora-hidden-fused-v1\",\"scope\":\"real layer0 BF16 base/A/B, synthetic input/prepared joint F32 ranks; complete corrected gate/up plus compiled SiLU/multiply vs single four-matmul hidden dispatch; host span, not down/FFN/model/physical GPU trace\",\"cases\":[";
    bool comma=false;
    for(int m:{1024,2096,3144})for(int n:{5120,7168}) {
        auto x=mx::reshape(mx::astype(mx::sin(mx::arange(m*4096,mx::float32)*.001f)*.125f,mx::bfloat16),{1,m,4096});
        auto rg=dense_gpu::projection_range(x,ag,0,256,0,4096,16,true,64),ru=dense_gpu::projection_range(x,au,0,256,0,4096,16,true,64);mx::eval({x,rg,ru});
        using Fn=std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;std::vector<Fn> recipes;
        for(bool mpp:{false,true})recipes.push_back(mx::compile([&,mpp,n](const std::vector<Tensor> &v) {
            auto g=mpp?dense_gpu::projection_range(v[0],w,0,n,0,4096,32,false,128):mx::matmul(v[0],mx::transpose(slice_axis(w,0,0,n)));
            auto u=mpp?dense_gpu::projection_range(v[0],w,12288,12288+n,0,4096,32,false,128):mx::matmul(v[0],mx::transpose(slice_axis(w,0,12288,12288+n)));
            g=dense_gpu::lora_b_epilogue(v[1],bg,0,n,.75f,mx::bfloat16,g);
            u=dense_gpu::lora_b_epilogue(v[2],bu,0,n,.75f,mx::bfloat16,u);
            return std::vector<Tensor>{silu(g)*u};
        }));
        struct Recipe{int bm,bn;bool rounded;};std::vector<Recipe> selected;
        for(int bm:{16,32})for(int bn:{64,128})for(bool rounded:{false,true}){
            selected.push_back({bm,bn,rounded});recipes.push_back(mx::compile([&,bm,bn,rounded,n](const std::vector<Tensor> &v){
                return std::vector<Tensor>{dense_gpu::lora_hidden(v[0],w,v[1],bg,v[2],bu,0,12288,n,0,0,.75f,.75f,bm,bn,rounded)};
            }));
        }
        auto expected=recipes[0]({x,rg,ru})[0];mx::eval(expected);std::vector<float> errors;std::vector<std::vector<double>> samples(recipes.size());
        for(auto &fn:recipes){auto y=fn({x,rg,ru})[0];mx::eval(y);auto yf=mx::astype(y,mx::float32),ef=mx::astype(expected,mx::float32);
            const float e=mx::sqrt(mx::sum(mx::square(yf-ef))/mx::maximum(mx::sum(mx::square(ef)),Tensor(1e-20f))).item<float>();
            require(std::isfinite(e) && e<=.05f,"hidden approximate component budget exceeded");errors.push_back(e);
            for(int warm=0;warm<3;++warm)mx::eval(fn({x,rg,ru}));
        }
        for(int sample=0;sample<15;++sample)for(size_t visit=0;visit<recipes.size();++visit){const auto i=(sample+visit)%recipes.size();
            const auto start=Clock::now();mx::eval(recipes[i]({x,rg,ru}));samples[i].push_back(std::chrono::duration<double>(Clock::now()-start).count());}
        if(comma)std::cout<<',';comma=true;std::cout<<"{\"M\":"<<m<<",\"N\":"<<n<<",\"recipes\":[";
        for(size_t i=0;i<recipes.size();++i){if(i)std::cout<<',';std::cout<<"{\"bm\":"<<(i<2?0:selected[i-2].bm)<<",\"bn\":"<<(i<2?int(i):selected[i-2].bn)
            <<",\"round_sigmoid\":"<<(i>=2 && selected[i-2].rounded?"true":"false")<<",\"relative_l2\":"<<errors[i]<<",\"seconds\":[";
            for(size_t j=0;j<samples[i].size();++j){if(j)std::cout<<',';std::cout<<samples[i][j];}std::cout<<"]}";}
        std::cout<<"]}";
    }
    std::cout<<"],\"qualification_passed\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}

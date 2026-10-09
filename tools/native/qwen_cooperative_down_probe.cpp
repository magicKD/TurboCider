#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_cooperative.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include <iomanip>
#include <iostream>

int main(int argc,char **argv){try {
    using namespace tc;require(argc==2 || argc==3,"usage: qwen-cooperative-down-probe original-local-checkpoint [register|tensor]");
    const std::string right_input=argc==3?argv[2]:"tensor";
    require(right_input=="register" || right_input=="tensor","right input must be register or tensor");
    const bool register_right=right_input=="register";
    configure_streams();
    Weights w;w.load_file(argv[1],"transformer_blocks.0.img_mlp.out.");w.materialize();const auto &down=w.at("weight");
    require(down.shape()==mx::Shape{4096,12288} && down.dtype()==mx::bfloat16,"original down physical source mismatch");
    struct Tile{int sm,bk,sn;};const std::vector<Tile> tiles{{16,32,32},{16,32,64},{16,64,32},{32,32,32},{64,32,32}};
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-cooperative-down-v1\",\"right_input\":\""<<right_input<<"\",\"scope\":\"real layer0 original BF16 down physical pitch12288, synthetic hidden; compiled original multi-SIMD vs K-blocked per-SIMD F32 partial, no widened/compacted W; unaligned K uses masked cooperative inputs for both variants; serial host spans, not complete FFN/model/device trace\",\"cases\":[";
    bool comma=false;
    for(int m:{1024,2096,3144})for(int k:{3072,5120,7168,10752}) {
        auto x=mx::reshape(mx::astype(mx::sin(mx::arange(m*k,mx::float32)*.001f)*.125f,mx::bfloat16),{1,m,k});mx::eval(x);
        using Fn=std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;std::vector<Fn> recipes;
        recipes.push_back(mx::compile([&,k](const std::vector<Tensor> &v){return std::vector<Tensor>{dense_gpu::projection_range(v[0],down,0,4096,0,k,32,true)};}));
        for(auto t:tiles)recipes.push_back(mx::compile([&,k,t](const std::vector<Tensor> &v){return std::vector<Tensor>{dense_gpu::projection_cooperative_range(v[0],down,0,4096,0,k,true,t.bk,t.sn,t.sm,register_right)};}));
        auto expected=recipes[0]({x})[0];mx::eval(expected);std::vector<float> errors;std::vector<std::vector<double>> times(recipes.size());
        for(auto &fn:recipes){auto y=fn({x})[0];mx::eval(y);
            const float e=mx::sqrt(mx::sum(mx::square(y-expected))/mx::maximum(mx::sum(mx::square(expected)),Tensor(1e-20f))).item<float>();
            require(std::isfinite(e) && e<.05f,"cooperative F32 partial exceeded5% component budget");errors.push_back(e);
            for(int warm=0;warm<3;++warm)mx::eval(fn({x}));
        }
        for(int it=0;it<15;++it)for(size_t j=0;j<recipes.size();++j){const auto i=(it+j)%recipes.size();const auto start=Clock::now();mx::eval(recipes[i]({x}));times[i].push_back(std::chrono::duration<double>(Clock::now()-start).count());}
        if(comma)std::cout<<',';comma=true;std::cout<<"{\"M\":"<<m<<",\"K\":"<<k<<",\"N\":4096,\"recipes\":[";
        for(size_t i=0;i<recipes.size();++i){if(i)std::cout<<',';std::cout<<"{\"SM\":"<<(i?tiles[i-1].sm:0)<<",\"BK\":"<<(i?tiles[i-1].bk:0)<<",\"SN\":"<<(i?tiles[i-1].sn:0)<<",\"relative_l2\":"<<errors[i]<<",\"seconds\":[";
            for(size_t j=0;j<times[i].size();++j){if(j)std::cout<<',';std::cout<<times[i][j];}std::cout<<"]}";}
        std::cout<<"]}";
    }
    std::cout<<"],\"qualification_passed\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}

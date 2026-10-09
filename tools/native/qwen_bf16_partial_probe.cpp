#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include <iomanip>
#include <iostream>

// Component only. Original physical BF16 source, synthetic BF16 hidden; no
// model/adapter downloads, source mutation, dense sidecar or ANE execution.
int main(int argc,char **argv) { try {
    using namespace tc;
    require(argc==2,"usage: qwen-bf16-partial-probe original-local-DiT-checkpoint");
    configure_streams();
    Weights weights;weights.load_file(argv[1],"transformer_blocks.0.img_mlp.out.");weights.materialize();
    const auto &down=weights.at("weight");
    require(down.shape()==mx::Shape{4096,12288} && down.dtype()==mx::bfloat16,
            "original Qwen down source geometry/dtype mismatch");
    using Fn=std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;
    const std::vector<std::string> names{"mpp_fp32","mpp_bf16_widen","mlx_bf16_widen"};
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-bf16-partial-component-v1\","
        "\"scope\":\"original layer0 BF16 physical down pitch12288, synthetic hidden; compiled partial/eval/widen; MLX may internally compact a strided operand, no explicit retained W copy; not full FFN/model/ANE/physical trace\","
        "\"source_bytes\":"<<down.nbytes()<<",\"cases\":[";
    bool comma=false;
    for(int m:{33,302,512,1024,2096,3144}) for(int k:{5120,7168,9216}) {
        auto x=mx::reshape(mx::astype(mx::sin(mx::arange(m*k,mx::float32)*.001f)*.125f,mx::bfloat16),{1,m,k});mx::eval(x);
        std::vector<Fn> recipes;
        recipes.push_back(mx::compile([&,k](const std::vector<Tensor> &v) {
            return std::vector<Tensor>{dense_gpu::projection_range(v[0],down,0,4096,0,k,32,true)};
        }));
        recipes.push_back(mx::compile([&,k](const std::vector<Tensor> &v) {
            return std::vector<Tensor>{mx::astype(dense_gpu::projection_range(v[0],down,0,4096,0,k,32,false),mx::float32)};
        }));
        recipes.push_back(mx::compile([&,k](const std::vector<Tensor> &v) {
            auto selected=slice_axis(down,1,0,k);
            return std::vector<Tensor>{mx::astype(mx::matmul(v[0],mx::transpose(selected)),mx::float32)};
        }));
        auto expected=recipes[0]({x})[0];mx::eval(expected);
        std::vector<double> errors;std::vector<std::vector<double>> times(recipes.size());
        for(auto &fn:recipes) {
            auto y=fn({x})[0];mx::eval(y);
            require(y.dtype()==mx::float32 && y.shape()==expected.shape() && mx::all(mx::isfinite(y)).item<bool>(),
                    "partial output dtype/shape/finite contract failed");
            const auto error=mx::sqrt(mx::sum(mx::square(y-expected))/mx::maximum(mx::sum(mx::square(expected)),Tensor(1e-20f))).item<float>();
            require(std::isfinite(error) && error<.05f,"BF16 partial exceeded5% component budget");errors.push_back(error);
            for(int warm=0;warm<3;++warm)mx::eval(fn({x}));
        }
        for(int it=0;it<15;++it) for(size_t j=0;j<recipes.size();++j) {
            const auto index=(it+j)%recipes.size();const auto start=Clock::now();mx::eval(recipes[index]({x}));
            times[index].push_back(std::chrono::duration<double>(Clock::now()-start).count());
        }
        if(comma)std::cout<<',';comma=true;
        std::cout<<"{\"M\":"<<m<<",\"K\":"<<k<<",\"N\":4096,\"recipes\":[";
        for(size_t i=0;i<recipes.size();++i) {
            if(i)std::cout<<',';
            std::cout<<"{\"name\":\""<<names[i]<<"\",\"relative_l2\":"<<errors[i]<<",\"seconds\":[";
            for(size_t j=0;j<times[i].size();++j) {if(j)std::cout<<',';std::cout<<times[i][j];}
            std::cout<<"]}";
        }
        std::cout<<"]}";
    }
    std::cout<<"],\"qualification_passed\":false}\n";
} catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;} }

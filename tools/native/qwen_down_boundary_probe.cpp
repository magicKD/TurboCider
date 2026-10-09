#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include <iomanip>
#include <iostream>

int main(int argc,char **argv) { try {
    using namespace tc;
    require(argc==2,"usage: qwen-down-boundary-probe local-BF16-checkpoint");
    configure_streams();Weights weights;weights.load_file(argv[1],"transformer_blocks.0.img_mlp.out.");weights.materialize();
    const auto &w=weights.at("weight");require(w.shape()==mx::Shape{4096,12288} && w.dtype()==mx::bfloat16,"original Qwen down fixture mismatch");
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-down-boundary-v1\",\"scope\":\"original layer0 BF16 physical source, synthetic hidden; MPP F32 vs MLX BF16 partial widened to F32; eager/compiled serial host spans; not complete FFN/model/physical GPU qualification\",\"cases\":[";
    bool comma=false;
    for(int m:{1024,2096,3144})for(int k:{5120,7168,10752}) {
        auto x=mx::reshape(mx::astype(mx::sin(mx::arange(m*k,mx::float32)*.001f)*.125f,mx::bfloat16),{1,m,k});mx::eval(x);
        auto mpp=[k](const std::vector<Tensor> &a){return std::vector<Tensor>{dense_gpu::projection_range(a[0],a[1],0,4096,0,k,32,true)};};
        auto mlx=[k](const std::vector<Tensor> &a){return std::vector<Tensor>{mx::astype(mx::matmul(a[0],mx::transpose(slice_axis(a[1],1,0,k))),mx::float32)};};
        using Fn=std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;
        std::vector<Fn> recipes{mpp,mlx,mx::compile(mpp),mx::compile(mlx)};
        auto baseline=mpp({x,w})[0];mx::eval(baseline);
        std::vector<double> errors;std::vector<std::vector<double>> times(4);
        for(auto &fn:recipes) {
            auto y=fn({x,w})[0];mx::eval(y);
            const auto rel=mx::sqrt(mx::sum(mx::square(y-baseline))/mx::maximum(mx::sum(mx::square(baseline)),Tensor(1e-20f))).item<float>();
            require(std::isfinite(rel) && rel<.05f,"down partial exceeded5% approximate component budget");errors.push_back(rel);
            for(int warm=0;warm<3;++warm)mx::eval(fn({x,w}));
        }
        for(int it=0;it<15;++it)for(int j=0;j<4;++j) {
            const int i=(it+j)%4;const auto start=Clock::now();mx::eval(recipes[i]({x,w}));
            times[i].push_back(std::chrono::duration<double>(Clock::now()-start).count());
        }
        if(comma)std::cout<<',';comma=true;
        std::cout<<"{\"M\":"<<m<<",\"K\":"<<k<<",\"N\":4096,\"physical_pitch\":12288,\"recipes\":[";
        const char *names[]={"mpp_f32","mlx_bf16_partial","compiled_mpp_f32","compiled_mlx_bf16_partial"};
        for(int i=0;i<4;++i){if(i)std::cout<<',';std::cout<<"{\"name\":\""<<names[i]<<"\",\"rel_l2_vs_f32_partial\":"<<errors[i]<<",\"seconds\":[";
            for(size_t j=0;j<times[i].size();++j){if(j)std::cout<<',';std::cout<<times[i][j];}std::cout<<"]}";}
        std::cout<<"]}";
    }
    std::cout<<"],\"qualification_passed\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;} }

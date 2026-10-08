#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_pair.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include <iomanip>
#include <iostream>

using namespace tc;
int main(int argc,char **argv){try {
    require(argc==2,"usage: qwen-gate-up-pair-probe local-diffusion-checkpoint");configure_streams();
    Weights weights;weights.load_file(argv[1],"transformer_blocks.0.img_mlp.gate_up.");weights.materialize();
    const auto &w=weights.at("weight");require(w.dtype()==mx::bfloat16 && w.shape()==mx::Shape{24576,4096},"real Qwen gate/up fixture mismatch");
    struct Recipe{const char *name;int bm,bn;};
    const std::vector<Recipe> recipes{{"original_mlx_two_matmuls",0,0},{"mpp_two_ranges",32,128},
        {"pair_m16n64",16,64},{"pair_m32n64",32,64},{"pair_m32n128",32,128},{"pair_m64n64",64,64},{"pair_m64n128",64,128}};
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-paired-gate-up-component-v1\","
        "\"scope\":\"real layer0 BF16 weights, synthetic input, same physical source, serial two-output operator host span, no artificial control concat, not FFN/request/GPU timestamp/LoRA qualification\",\"cases\":[";
    bool comma=false;
    for(int rows:{1056,3137})for(int count:{3072,7168,10752}) {
        auto x=mx::astype(mx::reshape(mx::sin(mx::arange(rows*4096,mx::float32)*.001f)*.125f,{1,rows,4096}),mx::bfloat16);mx::eval(x);
        auto run=[&](size_t index)->std::vector<Tensor> {
            if(index==0)return {mx::matmul(x,mx::transpose(slice_axis(w,0,0,count))),
                mx::matmul(x,mx::transpose(slice_axis(w,0,12288,12288+count)))};
            if(index==1)return {dense_gpu::projection_range(x,w,0,count,0,4096),
                dense_gpu::projection_range(x,w,12288,12288+count,0,4096)};
            return mx::split(dense_gpu::projection_row_pair(x,w,0,12288,count,0,4096,recipes[index].bm,recipes[index].bn),2,-1);
        };
        auto reference=run(0);mx::eval(reference);
        std::vector<float> errors;std::vector<std::vector<double>> samples(recipes.size());
        for(size_t index=0;index<recipes.size();++index) {
            auto y=run(index);mx::eval(y);
            auto diff0=mx::astype(y[0],mx::float32)-mx::astype(reference[0],mx::float32);
            auto diff1=mx::astype(y[1],mx::float32)-mx::astype(reference[1],mx::float32);
            const float error=mx::sqrt((mx::sum(mx::square(diff0))+mx::sum(mx::square(diff1)))/
                (mx::sum(mx::square(mx::astype(reference[0],mx::float32)))+mx::sum(mx::square(mx::astype(reference[1],mx::float32))))).item<float>();
            require(std::isfinite(error) && error<.002f,"paired real-weight component numerical gate failed");errors.push_back(error);
            for(int warm=0;warm<3;++warm)mx::eval(run(index));
        }
        for(int iteration=0;iteration<15;++iteration)for(size_t visit=0;visit<recipes.size();++visit) {
            const auto index=(visit+iteration)%recipes.size();const auto start=Clock::now();
            auto y=run(index);mx::eval(y);samples[index].push_back(std::chrono::duration<double>(Clock::now()-start).count());
        }
        if(comma)std::cout<<',';comma=true;
        std::cout<<"{\"M\":"<<rows<<",\"count_per_half\":"<<count<<",\"K\":4096,\"source_rows\":24576,\"recipes\":[";
        for(size_t index=0;index<recipes.size();++index) {
            std::cout<<(index?",":"")<<"{\"name\":\""<<recipes[index].name<<"\",\"rel_l2_vs_original_mlx\":"<<errors[index]<<",\"seconds\":[";
            for(size_t sample=0;sample<samples[index].size();++sample)std::cout<<(sample?",":"")<<samples[index][sample];
            std::cout<<"]}";
        }
        std::cout<<"]}";
    }
    std::cout<<"],\"qualification_passed\":false}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}

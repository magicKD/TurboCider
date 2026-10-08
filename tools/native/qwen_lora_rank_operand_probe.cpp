#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include <iomanip>
#include <iostream>

using namespace tc;
int main(int argc,char **argv){try {
    require(argc==2,"usage: qwen-lora-rank-operand-probe local-original-adapter");configure_streams();
    Weights weights;weights.load_file(argv[1],"transformer.transformer_blocks.0.img_mlp.");weights.materialize();
    const auto &gate=weights.at("gate_layer.lora_A.weight");
    const auto &down=weights.at("out.lora_A.weight");
    require(gate.dtype()==mx::bfloat16 && down.dtype()==mx::bfloat16 && gate.shape()==mx::Shape{256,4096} &&
        down.shape()==mx::Shape{256,12288},"original local BF16 LoRA A geometry mismatch");
    struct Recipe{int bm,bn;};const std::vector<Recipe> recipes{{0,0},{16,64},{32,64},{32,128},{64,64},{64,128}};
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-lora-bf16-operands-f32-ranks-v1\","
        "\"scope\":\"original local BF16 adapter A, synthetic BF16 activation, original FP32 ranks vs direct BF16 operands/F32 accumulator; serial operator host spans, not B/delta/model/visual/GPU timestamp qualification\",\"cases\":[";
    bool comma=false;
    for(int rows:{1056,3137})for(int columns:{4096,7168,12288}) {
        const auto &w=columns==4096?gate:down;
        auto x=mx::astype(mx::reshape(mx::sin(mx::arange(rows*columns,mx::float32)*.001f)*.125f,{1,rows,columns}),mx::bfloat16);mx::eval(x);
        auto run=[&](size_t index) {
            if(index==0)return mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(slice_axis(w,1,0,columns),mx::float32)));
            return dense_gpu::projection_range(x,w,0,256,0,columns,recipes[index].bm,true,recipes[index].bn);
        };
        auto expected=run(0);mx::eval(expected);
        std::vector<float> errors,maxima;std::vector<std::vector<double>> samples(recipes.size());
        for(size_t index=0;index<recipes.size();++index) {
            auto y=run(index);mx::eval(y);const float error=mx::sqrt(mx::sum(mx::square(y-expected))/mx::sum(mx::square(expected))).item<float>();
            require(y.dtype()==mx::float32 && std::isfinite(error) && error<1e-5f,"direct BF16 source rank arithmetic screen failed");
            errors.push_back(error);maxima.push_back(mx::max(mx::abs(y-expected)).item<float>());
            for(int warm=0;warm<3;++warm)mx::eval(run(index));
        }
        for(int iteration=0;iteration<15;++iteration)for(size_t visit=0;visit<recipes.size();++visit) {
            const auto index=(visit+iteration)%recipes.size();const auto start=Clock::now();
            auto y=run(index);mx::eval(y);samples[index].push_back(std::chrono::duration<double>(Clock::now()-start).count());
        }
        if(comma)std::cout<<',';comma=true;
        std::cout<<"{\"M\":"<<rows<<",\"K\":"<<columns<<",\"rank\":256,\"physical_pitch\":"<<w.shape(1)<<",\"recipes\":[";
        for(size_t index=0;index<recipes.size();++index) {
            std::cout<<(index?",":"")<<"{\"bm\":"<<recipes[index].bm<<",\"bn\":"<<recipes[index].bn<<",\"rel_l2\":"<<errors[index]
                <<",\"max_abs\":"<<maxima[index]<<",\"seconds\":[";
            for(size_t sample=0;sample<samples[index].size();++sample)std::cout<<(sample?",":"")<<samples[index][sample];
            std::cout<<"]}";
        }
        std::cout<<"]}";
    }
    std::cout<<"],\"qualification_passed\":false}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}

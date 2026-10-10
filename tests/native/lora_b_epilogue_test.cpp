#include "../../native/backends/dense_gpu_lora_b.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include <iostream>

namespace mx=mlx::core;
int main(){try {
    int cases=0;float maximum=0;
    for(auto storage:{mx::float16,mx::bfloat16})for(int m:{1,33,67})for(bool strided:{false,true}) {
        auto source=mx::reshape(mx::sin(mx::arange(m*64,mx::float32)*.017f)*.2f,strided?mx::Shape{64,m}:mx::Shape{m,64});
        auto low=mx::expand_dims(strided?mx::transpose(source):source,0);
        auto up=mx::astype(mx::reshape(mx::cos(mx::arange(384*64,mx::float32)*.003f)*.04f,{384,64}),storage);
        mx::eval({low,up});
        for(auto dtype:{mx::float32,mx::float16,mx::bfloat16})for(bool add:{false,true})for(int bm:{16,32,64})for(int bn:{64,128}) {
            auto base=mx::astype(mx::reshape(mx::sin(mx::arange(m*320,mx::float32)*.011f)*.02f,{1,m,320}),dtype);mx::eval(base);
            auto raw=tc::dense_gpu::projection_range(mx::astype(low,storage),up,64,257,0,64,bm,true,bn)*mx::array(-.75f,mx::float32);
            auto expected=mx::astype(add?raw+mx::astype(mx::slice(base,{0,0,17},{1,m,210}),mx::float32):raw,dtype);
            auto actual=tc::dense_gpu::lora_b_epilogue(low,up,64,257,-.75f,dtype,add?std::optional<mx::array>(base):std::nullopt,add?17:0,bm,bn);
            mx::eval({expected,actual});
            float error=mx::sqrt(mx::sum(mx::square(mx::astype(actual,mx::float32)-mx::astype(expected,mx::float32)))/
                mx::maximum(mx::sum(mx::square(mx::astype(expected,mx::float32))),mx::array(1e-20f))).item<float>();
            if(actual.shape()!=expected.shape() || actual.dtype()!=dtype || !std::isfinite(error) || error>.005f)
                throw std::runtime_error("fused B epilogue changed scale/base/cast boundary excessively");
            maximum=std::max(maximum,error);++cases;
        }
        for(int first:{-1,384}) {
            bool rejected=false;try{tc::dense_gpu::lora_b_epilogue(low,up,first,385,1.f,mx::float32);}catch(const std::invalid_argument&){rejected=true;}
            if(!rejected)throw std::runtime_error("invalid B range accepted");
        }
    }
    std::cout<<"PASS fused LoRA B epilogue cases="<<cases<<" maximum="<<maximum<<": two source dtypes, F32 ranks/strides, tails, physical offsets, three output dtypes, six tiles, negative scale and optional base\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}

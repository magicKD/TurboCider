#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_lora_hidden.hpp"
#include "../../native/backends/dense_gpu_base_lora.hpp"
#include <iostream>
#include <limits>

int main(){try {
    using namespace tc;configure_streams();int cases=0;float maximum=0;
    for(auto dtype:{mx::float16,mx::bfloat16})for(int m:{1,33,67})for(int bm:{16,32})for(int bn:{64,128})for(bool rounded:{false,true}) {
        constexpr int k=96,n=83,rg=64,ru=32,g=7,u=109,bg=11,bu=13;
        auto x=mx::reshape(mx::astype(mx::sin(mx::arange(m*k,mx::float32)*.011f)*.12f,dtype),{1,m,k});
        auto w=mx::astype(mx::reshape(mx::cos(mx::arange((u+n+17)*k,mx::float32)*.013f)*.25f,{u+n+17,k}),dtype);
        auto ag=mx::reshape(mx::sin(mx::arange(m*rg,mx::float32)*.009f)*.025f,{1,m,rg});
        auto au=mx::reshape(mx::cos(mx::arange(m*ru,mx::float32)*.009f)*.025f,{1,m,ru});
        auto b1=mx::astype(mx::reshape(mx::cos(mx::arange((n+32)*rg,mx::float32)*.017f)*.035f,{n+32,rg}),dtype);
        auto b2=mx::astype(mx::reshape(mx::sin(mx::arange((n+32)*ru,mx::float32)*.017f)*.035f,{n+32,ru}),dtype);
        auto gate=dense_gpu::base_lora_epilogue(x,w,ag,b1,g,g+n,0,k,bg,.75f,bm,bn);
        auto up=dense_gpu::base_lora_epilogue(x,w,au,b2,u,u+n,0,k,bu,-.25f,bm,bn);
        auto expected=silu(gate)*up;
        auto actual=dense_gpu::lora_hidden(x,w,ag,b1,au,b2,g,u,n,bg,bu,.75f,-.25f,bm,bn,rounded);mx::eval({expected,actual});
        auto af=mx::astype(actual,mx::float32),ef=mx::astype(expected,mx::float32);
        const float e=mx::sqrt(mx::sum(mx::square(af-ef))/mx::maximum(mx::sum(mx::square(ef)),Tensor(1e-20f))).item<float>();
        require(std::isfinite(e) && e<.05f,"hidden activation component exceeded5% screen");maximum=std::max(maximum,e);++cases;
    }
    auto x=mx::zeros({1,1,96},mx::bfloat16),w=mx::zeros({192,96},mx::bfloat16),r=mx::zeros({1,1,64},mx::float32),b=mx::zeros({96,64},mx::bfloat16);
    for(int bad=0;bad<5;++bad){bool rejected=false;try{
        dense_gpu::lora_hidden(x,w,bad==0?mx::astype(r,mx::float16):r,b,r,b,0,96,83,bad==1?20:0,0,
            bad==2?std::numeric_limits<float>::quiet_NaN():.75f,.75f,bad==3?64:32,bad==4?63:128,true);
    }catch(const std::invalid_argument &){rejected=true;}require(rejected,"malformed fused hidden contract accepted");}
    std::cout<<"PASS fused LoRA hidden cases="<<cases<<" maximum_relative_l2="<<maximum<<": typed base/corrected boundaries, distinct ranks/scales, source offsets, activation lowering and5 invalid contract rejections\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}

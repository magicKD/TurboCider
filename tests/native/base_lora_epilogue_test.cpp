#include "../../native/backends/dense_gpu_base_lora.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include "../../native/backends/dense_gpu_lora_b.hpp"
#include <iostream>
#include <limits>

namespace mx=mlx::core;
int main(){try {
    int cases=0;float maximum=0;
    for(auto dtype:{mx::float16,mx::bfloat16})for(int m:{1,33,67})for(int bm:{16,32,64})for(int bn:{64,128})for(float scale:{.75f,-.25f}) {
        constexpr int k=96,n=83,r=64,cb=32,rb=7,first=11;
        auto x=mx::reshape(mx::astype(mx::sin(mx::arange(m*k,mx::float32)*.011f)*.12f,dtype),{1,m,k});
        auto w=mx::astype(mx::reshape(mx::cos(mx::arange((n+19)*(k+64),mx::float32)*.013f)*.025f,{n+19,k+64}),dtype);
        auto low=mx::reshape(mx::sin(mx::arange(m*r,mx::float32)*.009f)*.025f,{1,m,r});
        auto b=mx::astype(mx::reshape(mx::cos(mx::arange((n+32)*r,mx::float32)*.017f)*.035f,{n+32,r}),dtype);
        auto base=tc::dense_gpu::projection_range(x,w,rb,rb+n,cb,cb+k,bm,false,bn);
        auto expected=tc::dense_gpu::lora_b_epilogue(low,b,first,first+n,scale,dtype,base,0,bm,bn);
        auto actual=tc::dense_gpu::base_lora_epilogue(x,w,low,b,rb,rb+n,cb,cb+k,first,scale,bm,bn);mx::eval({expected,actual});
        const auto error=mx::max(mx::abs(mx::astype(actual,mx::float32)-mx::astype(expected,mx::float32))).item<float>();maximum=std::max(maximum,error);
        if(error!=0)throw std::runtime_error("fused base/B changed the admitted sequential rounding oracle");++cases;
    }
    auto x=mx::zeros({1,1,96},mx::bfloat16),w=mx::zeros({96,96},mx::bfloat16),r=mx::zeros({1,1,64},mx::float32),b=mx::zeros({96,64},mx::bfloat16);
    for(int bad=0;bad<6;++bad){bool rejected=false;try {
        tc::dense_gpu::base_lora_epilogue(bad==0?mx::astype(x,mx::float32):x,w,bad==1?mx::astype(r,mx::float16):r,b,
            0,bad==2?97:83,0,96,bad==3?20:0,bad==4?std::numeric_limits<float>::infinity():.75f,bad==5?17:32,128);
    }catch(const std::invalid_argument &){rejected=true;}if(!rejected)throw std::runtime_error("malformed fused base/source contract accepted");}
    std::cout<<"PASS fused base/LoRA cases="<<cases<<" maximum="<<maximum<<": original physical base/B ranges, typed base boundary, F32 scaled B, tails, six tiles and6 invalid contract rejections\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}

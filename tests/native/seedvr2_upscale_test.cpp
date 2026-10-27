#include "../../native/models/seedvr2/metal/upscale.hpp"
#include <cstring>
#include <iostream>
namespace mx=mlx::core;
int main() {
    try {
        mx::set_default_device(mx::Device::gpu);int cases=0;
        for(int c:{3,16,256,512})for(int t:{1,2})for(bool temporal:{false,true})for(bool initial:{false,true}) {
            const int b=2,h=3,w=5,tr=temporal?2:1;
            auto x=mx::astype(mx::random::normal({b,t,h,w,4*tr*c},mx::float32,mx::random::key(19)),mx::float16);
            auto bias=mx::astype(mx::random::normal({4*tr*c},mx::float32,mx::random::key(23))*.1f,mx::float16);
            auto original=mx::reshape(x+bias,{b,t,h,w,2,2,tr,c});
            original=mx::reshape(mx::transpose(original,{0,1,6,2,4,3,5,7}),{b,t*tr,h*2,w*2,c});
            if(temporal && initial)original=mx::concatenate({mx::slice(original,{0,0,0,0,0},{b,1,h*2,w*2,c}),
                mx::slice(original,{0,2,0,0,0},{b,t*tr,h*2,w*2,c})},1);
            original=mx::contiguous(original);auto fused=tc::seedvr2::metal::upscale_bias_shuffle(x,bias,initial,temporal);
            mx::eval({original,fused});
            tc::require(fused.shape()==original.shape() && std::memcmp(fused.data<uint16_t>(),original.data<uint16_t>(),fused.nbytes())==0,
                "fused upsampling changed native phase/FP16 words");++cases;
        }
        auto x=mx::astype(mx::reshape(mx::arange(16*4*3,mx::float32),{1,1,4,4,12})*.0000001f,mx::float16);
        auto bias=mx::zeros({12},mx::float16);
        auto strided=mx::transpose(x,{0,1,3,2,4});
        auto original=mx::contiguous(mx::reshape(mx::transpose(mx::reshape(strided+bias,{1,1,4,4,2,2,1,3}),
            {0,1,6,2,4,3,5,7}),{1,1,8,8,3}));
        auto fused=tc::seedvr2::metal::upscale_bias_shuffle(strided,bias,true,false);mx::eval({original,fused});
        tc::require(std::memcmp(fused.data<uint16_t>(),original.data<uint16_t>(),fused.nbytes())==0,
            "fused upsampling changed strided/subnormal operands");++cases;
        bool rejected=false;try{tc::seedvr2::metal::upscale_bias_shuffle(mx::astype(x,mx::float32),bias,false,false);}catch(const std::exception &){rejected=true;}
        tc::require(rejected,"unsupported projection precision was accepted");
        std::cout<<"PASS fused separate FP16 bias/shuffle, phases/edges/strides/subnormals cases="<<cases<<'\n';
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}

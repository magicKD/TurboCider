#include "runtime.hpp"
#include "flux_vae_ops.hpp"
namespace tc {
Tensor Flux::encode_image(const Tensor& image,const Event& event,std::atomic<bool>& cancelled) {
    FluxVaeOps ops{vae_};
    auto x=ops.conv(image,"encoder.conv_in");
    for(int block=0;block<4;++block) {
        checkpoint(cancelled);event("image_encode",block,4);
        auto p="encoder.down_blocks."+std::to_string(block);
        for(int i=0;i<2;++i)x=ops.residual(x,p+".resnets."+std::to_string(i));
        if(block<3) {
            x=mx::pad(x,{{0,0},{0,1},{0,1},{0,0}},Tensor(0.f,x.dtype()));
            x=ops.conv(x,p+".downsamplers.0.conv",2,0);
        }
        mx::eval(x);
    }
    x=ops.residual(x,"encoder.mid_block.resnets.0");
    x=ops.attention(x,"encoder.mid_block.attentions.0");
    x=ops.residual(x,"encoder.mid_block.resnets.1");
    x=ops.conv(silu(ops.group_norm(x,"encoder.conv_norm_out")),"encoder.conv_out");
    x=slice_axis(ops.conv(x,"quant_conv"),3,0,32);
    int h=x.shape(1),w=x.shape(2);
    require(h%2==0&&w%2==0,"encoded image spatial dimensions must be even");
    x=mx::transpose(x,{0,3,1,2});
    x=mx::reshape(mx::transpose(mx::reshape(x,{1,32,h/2,2,w/2,2}),{0,1,3,5,2,4}),{1,128,h/2,w/2});
    auto mean=mx::reshape(vae_.at("bn.running_mean"),{1,128,1,1});
    auto std=mx::sqrt(mx::reshape(vae_.at("bn.running_var"),{1,128,1,1})+Tensor(1e-4f,mx::bfloat16));
    x=(x-mean)/std;
    x=mx::transpose(mx::reshape(x,{1,128,(h/2)*(w/2)}),{0,2,1});mx::eval(x);
    event("image_encode",4,4);return x;
}
}

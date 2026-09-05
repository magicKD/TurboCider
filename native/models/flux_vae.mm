#include "runtime.hpp"
namespace tc {
Tensor Flux::decode(const Tensor& latent,int height,int width,const Event&event,std::atomic<bool>&cancelled,const std::string&dump){
 auto trace=[&](const std::string&name,const Tensor&a){if(!dump.empty())mx::save_safetensors((std::filesystem::path(dump)/("vae_"+name+".safetensors")).string(),{{"tensor",a}});};
 auto&w=vae_;int h=height/16,ww=width/16;
 auto packed=mx::transpose(mx::reshape(latent,{1,h,ww,128}),{0,3,1,2});
 packed=packed*mx::sqrt(mx::reshape(w.at("bn.running_var"),{1,128,1,1})+Tensor(1e-4f,mx::bfloat16))+mx::reshape(w.at("bn.running_mean"),{1,128,1,1});
 auto x=mx::reshape(mx::transpose(mx::reshape(packed,{1,32,2,2,h,ww}),{0,1,4,2,5,3}),{1,32,h*2,ww*2});
 x=mx::transpose(x,{0,2,3,1});trace("unpack",x);
 auto conv=[&](const Tensor&a,const std::string&p){auto weight=mx::transpose(w.at(p+".weight"),{0,2,3,1});int pad=weight.shape(1)/2;auto z=mx::conv2d(a,weight,{1,1},{pad,pad});return z+w.at(p+".bias");};
 auto gn=[&](const Tensor&a,const std::string&p){int n=a.shape(1)*a.shape(2),c=a.shape(3);auto f=mx::reshape(mx::astype(a,mx::float32),{1,n,32,c/32});f=mx::reshape(mx::transpose(f,{0,2,1,3}),{1,32,n*c/32});f=mx::fast::layer_norm(f,{}, {},1e-6f);auto z=mx::reshape(mx::transpose(mx::reshape(f,{1,32,n,c/32}),{0,2,1,3}),a.shape());return mx::astype(z*mx::astype(w.at(p+".weight"),mx::float32)+mx::astype(w.at(p+".bias"),mx::float32),mx::bfloat16);};
 auto res=[&](const Tensor&a,const std::string&p){auto z=conv(silu(gn(a,p+".norm1")),p+".conv1");z=conv(silu(gn(z,p+".norm2")),p+".conv2");return z+(w.has(p+".conv_shortcut.weight")?conv(a,p+".conv_shortcut"):a);};
 x=conv(x,"post_quant_conv");trace("post",x);x=conv(x,"decoder.conv_in");trace("in",x);x=res(x,"decoder.mid_block.resnets.0");trace("mid0",x);
 {std::string p="decoder.mid_block.attentions.0";int n=x.shape(1)*x.shape(2),c=x.shape(3);auto z=gn(x,p+".group_norm");auto q=heads(mx::reshape(linear(z,w,p+".to_q"),{1,n,c}),1,c),k=heads(mx::reshape(linear(z,w,p+".to_k"),{1,n,c}),1,c),v=heads(mx::reshape(linear(z,w,p+".to_v"),{1,n,c}),1,c);x=x+linear(mx::reshape(attend(q,k,v),x.shape()),w,p+".to_out.0");}
 trace("attention",x);x=res(x,"decoder.mid_block.resnets.1");mx::eval(x);trace("mid1",x);
 for(int i=0;i<4;++i){checkpoint(cancelled);event("vae_decode",i,4);auto p="decoder.up_blocks."+std::to_string(i);
  for(int j=0;j<3;++j){x=res(x,p+".resnets."+std::to_string(j));mx::eval(x);}
  if(i<3)x=conv(mx::repeat(mx::repeat(x,2,1),2,2),p+".upsamplers.0.conv");mx::eval(x);trace("up"+std::to_string(i),x);
 }
 x=gn(x,"decoder.conv_norm_out");trace("norm",x);x=conv(silu(x),"decoder.conv_out");mx::eval(x);event("vae_decode",4,4);return x;
}
}

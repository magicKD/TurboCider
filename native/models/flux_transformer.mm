#include "runtime.hpp"
#include "../backends/coreml.hpp"
#include <cmath>
namespace tc {
Tensor Flux::denoise(const Tensor&latent,const Tensor&text,float sigma,int height,int width,const Event&event,std::atomic<bool>&cancelled,const std::vector<float>&reference_ids){
 auto&w=transformer_;int nt=text.shape(1),ni=latent.shape(1),n=nt+ni;
 auto t=Tensor(sigma*1000.f,latent.dtype());
 auto freq=mx::exp(mx::arange(128,mx::float32)*(-std::log(10000.f)/128.f));
 auto angle=mx::reshape(mx::astype(t,mx::float32)*freq,{1,128});
 auto emb=mx::concatenate({mx::cos(angle),mx::sin(angle)},-1);
 auto temb=mx::astype(linear(silu(linear(emb,w,"time_guidance_embed.timestep_embedder.linear_1")),w,"time_guidance_embed.timestep_embedder.linear_2"),mx::bfloat16);
 auto mod=[&](const std::string&p,int count){return mx::split(mx::expand_dims(linear(silu(temb),w,p+".linear"),1),count,-1);};
 auto mi=mod("double_stream_modulation_img",6),mt=mod("double_stream_modulation_txt",6),ms=mod("single_stream_modulation",3);
 std::vector<float> ids(n*4,0);
 for(int i=0;i<nt;++i)ids[i*4+3]=i;
 for(int i=0;i<ni;++i){ids[(nt+i)*4+1]=i/(width/16);ids[(nt+i)*4+2]=i%(width/16);}
 if(!reference_ids.empty()){require(reference_ids.size()<=size_t(ni)*4,"reference ID overflow");std::copy(reference_ids.begin(),reference_ids.end(),ids.end()-reference_ids.size());}
 auto coord=Tensor(ids.data(),{n,4},mx::float32);std::vector<Tensor> cosines,sines;
 auto rf=1.f/mx::power(Tensor(2000.f),mx::arange(0,32,2,mx::float32)/32.f);
 for(int axis=0;axis<4;++axis){auto angles=slice_axis(coord,1,axis,axis+1)*rf;cosines.push_back(mx::cos(angles));sines.push_back(mx::sin(angles));}
 auto cos=mx::concatenate(cosines,-1),sin=mx::concatenate(sines,-1);
 auto x=linear(latent,w,"x_embedder"),c=linear(text,w,"context_embedder");
 auto ff=[&](const Tensor&a,const std::string&p){auto parts=mx::split(linear(a,w,p+".linear_in"),2,-1);return linear(silu(parts[0])*parts[1],w,p+".linear_out");};
 auto qnorm=[&](const Tensor& a,const Tensor& weight,float eps){return mx::astype(mx::fast::rms_norm(mx::astype(a,mx::float32),weight,eps),a.dtype());};
 auto qkv=[&](const Tensor&a,const std::string&p,bool ctx){
  auto q=heads(linear(a,w,p+(ctx?".add_q_proj":".to_q")),heads_,128),k=heads(linear(a,w,p+(ctx?".add_k_proj":".to_k")),heads_,128),v=heads(linear(a,w,p+(ctx?".add_v_proj":".to_v")),heads_,128);
  return std::vector<Tensor>{qnorm(q,w.at(p+(ctx?".norm_added_q.weight":".norm_q.weight")),1e-5f),qnorm(k,w.at(p+(ctx?".norm_added_k.weight":".norm_k.weight")),1e-5f),v};
 };
 for(int i=0;i<dual_layers_;++i){checkpoint(cancelled);event("transformer_block",i,dual_layers_+single_layers_);auto p="transformer_blocks."+std::to_string(i);
  auto a=qkv(norm(x)*(1+mi[1])+mi[0],p+".attn",false),b=qkv(norm(c)*(1+mt[1])+mt[0],p+".attn",true);
  auto q=rope_pairs(mx::concatenate({b[0],a[0]},2),cos,sin),k=rope_pairs(mx::concatenate({b[1],a[1]},2),cos,sin),v=mx::concatenate({b[2],a[2]},2);
  auto att=attend(q,k,v);
  x=x+mi[2]*linear(slice_axis(att,1,nt,n),w,p+".attn.to_out.0");
  c=c+mt[2]*linear(slice_axis(att,1,0,nt),w,p+".attn.to_add_out");
  x=x+mi[5]*ff(norm(x)*(1+mi[4])+mi[3],p+".ff");
  c=c+mt[5]*ff(norm(c)*(1+mt[4])+mt[3],p+".ff_context");mx::eval({x,c});
 }
 x=mx::concatenate({c,x},1);
 auto sqnorm=[&](const Tensor&a,const Tensor&weight,float eps){return mx::astype(mx::fast::rms_norm(mx::astype(a,mx::float32),weight,eps),mx::bfloat16);};
 for(int i=0;i<single_layers_;++i){checkpoint(cancelled);event("transformer_block",dual_layers_+i,dual_layers_+single_layers_);auto p="single_transformer_blocks."+std::to_string(i)+".attn";
  auto a=norm(x)*(1+ms[1])+ms[0];
  if(hybrid_){
   auto packed=mx::astype(a,mx::float16);
   if(n<hybrid_->rows)packed=mx::concatenate({packed,mx::zeros({1,hybrid_->rows-n,3072},mx::float16)},1);
   mx::eval({a,packed});
   // Compile the independent GPU branch once per shape/dtype. Weights are
   // explicit inputs, so the cached graph cannot retain an old model session.
   static auto attention_graph=mx::compile([](const std::vector<Tensor>& args){
    auto proj=mx::matmul(args[0],mx::transpose(slice_axis(args[3],0,0,9216)));
    auto parts=mx::split(proj,3,-1);
    auto q=mx::astype(mx::fast::rms_norm(mx::astype(heads(parts[0],24,128),mx::float32),args[4],1e-5f),mx::bfloat16);
    auto k=mx::astype(mx::fast::rms_norm(mx::astype(heads(parts[1],24,128),mx::float32),args[5],1e-5f),mx::bfloat16);
    auto att=attend(rope_pairs(q,args[1],args[2]),rope_pairs(k,args[1],args[2]),heads(parts[2],24,128));
    return std::vector<Tensor>{mx::matmul(att,mx::transpose(slice_axis(args[6],1,0,3072)))};
   });
   auto gpu=attention_graph({a,cos,sin,w.at(p+".to_qkv_mlp_proj.weight"),w.at(p+".norm_q.weight"),w.at(p+".norm_k.weight"),w.at(p+".to_out.weight")})[0];
   mx::async_eval({gpu});
   auto ane=slice_axis(hybrid_->predict(i,packed),1,0,n);
   x=x+ms[2]*(gpu+mx::astype(ane,gpu.dtype()));
  }else{
   auto proj=linear(a,w,p+".to_qkv_mlp_proj");
   auto parts=mx::split(proj,mx::Shape{hidden_,hidden_*2,hidden_*3,hidden_*6},-1);
   require(parts.size()==5,"Flux single projection must contain Q/K/V and two MLP parts");
   require(parts[0].shape(-1)==hidden_&&parts[1].shape(-1)==hidden_&&parts[2].shape(-1)==hidden_&&parts[3].shape(-1)==hidden_*3&&parts[4].shape(-1)==hidden_*3,"Flux single projection has an unsupported packed layout");
   auto q=rope_pairs(sqnorm(heads(parts[0],heads_,128),w.at(p+".norm_q.weight"),1e-5f),cos,sin);
   auto k=rope_pairs(sqnorm(heads(parts[1],heads_,128),w.at(p+".norm_k.weight"),1e-5f),cos,sin);
   auto att=attend(q,k,heads(parts[2],heads_,128));auto mlp=silu(parts[3])*parts[4];
   x=x+ms[2]*linear(mx::concatenate({att,mlp},-1),w,p+".to_out");
  }
  mx::eval(x);
 }
 auto outmod=mx::split(mx::expand_dims(linear(silu(temb),w,"norm_out.linear"),1),2,-1);
 return linear(norm(slice_axis(x,1,nt,n))*(1+outmod[0])+outmod[1],w,"proj_out");
}
}

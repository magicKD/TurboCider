#include "runtime.hpp"
#include <cmath>
namespace tc {
Tensor Flux::encode(const Tokens&t,const Event&event,std::atomic<bool>&cancelled){
 Weights w;w.load(root_/"text_encoder",event,cancelled);
 w.apply_loras(active_loras_,"text_encoder",event,cancelled);
 int n=int(t.ids.size());
 auto ids=Tensor(t.ids.data(),{1,n},mx::int32);
 auto x=mx::take(w.at("model.embed_tokens.weight"),ids,0);
 auto freq=1.f/mx::power(Tensor(1000000.f),mx::arange(0,128,2,mx::float32)/128.f);
 auto angle=mx::reshape(mx::arange(n,mx::float32),{1,n,1})*mx::reshape(freq,{1,1,64});
 angle=mx::concatenate({angle,angle},-1);
 auto cos=mx::expand_dims(mx::astype(mx::cos(angle),mx::bfloat16),1);
 auto sin=mx::expand_dims(mx::astype(mx::sin(angle),mx::bfloat16),1);
 auto pos=mx::arange(n,mx::int32);auto qi=mx::reshape(pos,{n,1}),ki=mx::reshape(pos,{1,n});
 auto forbidden=mx::logical_or(ki>qi,ki>=Tensor(t.valid));
 auto mask=mx::reshape(mx::where(forbidden,Tensor(-INFINITY,mx::bfloat16),Tensor(0.f,mx::bfloat16)),{1,1,n,n});
 std::vector<Tensor> layers;
 auto rotate=[&](const Tensor& v){auto h=mx::split(v,2,-1);return v*cos+mx::concatenate({-h[1],h[0]},-1)*sin;};
 // FLUX.2 consumes hidden states 9, 18 and 27.  Layers after the last tap
 // cannot affect those tensors, so stop at layer 27 while still validating
 // the installed encoder's full configured depth in the constructor.
 constexpr int output_layer=27;
 require(text_layers_ >= output_layer, "FLUX text encoder must expose hidden state 27");
 for(int i=0;i<output_layer;++i){checkpoint(cancelled);event("text_encode",i,output_layer);std::string p="model.layers."+std::to_string(i);
  auto a=rms(x,w.at(p+".input_layernorm.weight"),1e-6f);
  auto q=heads(linear(a,w,p+".self_attn.q_proj"),text_heads_,128);
  auto k=heads(linear(a,w,p+".self_attn.k_proj"),text_kv_heads_,128);
  auto v=heads(linear(a,w,p+".self_attn.v_proj"),text_kv_heads_,128);
  q=rotate(rms(q,w.at(p+".self_attn.q_norm.weight"),1e-6f));
  k=rotate(rms(k,w.at(p+".self_attn.k_norm.weight"),1e-6f));
  k=mx::repeat(k,text_heads_/text_kv_heads_,1);v=mx::repeat(v,text_heads_/text_kv_heads_,1);
  x=x+linear(attend(q,k,v,true,mask),w,p+".self_attn.o_proj");
  a=rms(x,w.at(p+".post_attention_layernorm.weight"),1e-6f);
  x=x+linear(silu(linear(a,w,p+".mlp.gate_proj"))*linear(a,w,p+".mlp.up_proj"),w,p+".mlp.down_proj");
  mx::eval(x);if(i==8||i==17||i==26)layers.push_back(x);
 }
 auto result=mx::concatenate(layers,-1);mx::eval(result);
 event("text_encode",output_layer,output_layer);return result;
}
}

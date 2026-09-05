#include "ltx_components.hpp"
#include <cmath>
#include <limits>
namespace tc {
std::pair<Tensor,Tensor> ltx_project_hidden(const std::vector<Tensor>& states,const Tensor& mask,const Weights& projection) {
    require(!states.empty(),"missing Gemma hidden states");
    auto encoded=mx::stack(states,-1);auto variance=mx::mean(encoded*encoded,2,true);
    auto normalized=encoded*mx::rsqrt(variance+Tensor(1e-6f,encoded.dtype()));
    auto flat=mx::reshape(normalized,{encoded.shape(0),encoded.shape(1),encoded.shape(2)*encoded.shape(3)});
    flat=flat*mx::astype(mx::expand_dims(mask,-1),flat.dtype());mx::eval(flat);
    auto project=[&](const std::string& name){int target=projection.at(name+".weight").shape(0);auto scale=Tensor(float(std::sqrt(double(target)/encoded.shape(2))),flat.dtype());auto result=linear(flat*scale,projection,name);mx::eval(result);return result;};
    return {project("video_aggregate_embed"),project("audio_aggregate_embed")};
}
std::vector<Tensor> ltx_gemma_hidden(NSDictionary *config,const Weights& weights,
    const Tensor& ids,const Tensor& padding,const Event& event,std::atomic<bool>& cancel) {
    require([config[@"model_type"] isEqual:@"gemma4_unified"],"LTX requires Gemma4 unified config");
    NSDictionary *c=config[@"text_config"];
    require([c isKindOfClass:NSDictionary.class]&&[c[@"model_type"] isEqual:@"gemma4_unified_text"],"invalid Gemma4 text configuration");
    auto integer=[&](NSString *key,int maximum){id v=c[key];require([v isKindOfClass:NSNumber.class],"Gemma4 configuration field missing");int n=[v intValue];require(n>0&&n<=maximum&&double(n)==[v doubleValue],"invalid Gemma4 dimensions");return n;};
    int hidden=integer(@"hidden_size",16384),layers=integer(@"num_hidden_layers",128),heads=integer(@"num_attention_heads",256);
    int local_dim=integer(@"head_dim",1024),global_dim=integer(@"global_head_dim",1024);
    int local_kv=integer(@"num_key_value_heads",heads),global_kv=integer(@"num_global_key_value_heads",heads),window=integer(@"sliding_window",65536);
    float eps=[c[@"rms_norm_eps"] floatValue];require(std::isfinite(eps)&&eps>0,"invalid Gemma4 norm epsilon");
    require([c[@"num_kv_shared_layers"] intValue]==0&&! [c[@"use_bidirectional_attention"] isEqual:@"all"],"unsupported Gemma4 shared-KV or bidirectional tower");
    NSArray *types=c[@"layer_types"];require([types isKindOfClass:NSArray.class]&&types.count==NSUInteger(layers),"Gemma4 layer schedule mismatch");
    require(ids.ndim()==2&&ids.shape(0)==1&&ids.shape(1)>0&&ids.shape(1)<=4096&&ids.shape()==padding.shape(),"invalid Gemma4 token/mask shape");
    int length=ids.shape(1);auto& embedding=weights.at("embed_tokens.weight");require(embedding.ndim()==2&&embedding.shape(1)==hidden,"Gemma4 embedding shape mismatch");
    require(embedding.dtype()==mx::bfloat16||embedding.dtype()==mx::float16||embedding.dtype()==mx::float32,"Gemma4 pack must contain floating point weights");
    require(mx::all((ids>=0)&(ids<embedding.shape(0))).item<bool>(),"Gemma4 token out of vocabulary");
    Tensor x=mx::take(embedding,ids,0)*Tensor(float(std::sqrt(double(hidden))),embedding.dtype());
    auto normalize=[&](const Tensor& a,const std::string& name){auto f=mx::astype(a,mx::float32);auto y=f*mx::rsqrt(mx::mean(mx::square(f),-1,true)+eps);if(!name.empty())y=y*mx::astype(weights.at(name+".weight"),mx::float32);return mx::astype(y,a.dtype());};
    struct Attention {int dim,kv;Tensor cos,sin,mask;};
    std::map<std::string,Attention> geometry;
    for(bool sliding:{true,false}) {
        std::string name=sliding?"sliding_attention":"full_attention";
        NSDictionary *rope=c[@"rope_parameters"][@(name.c_str())];
        require([rope isKindOfClass:NSDictionary.class]&&[rope[@"rope_type"] isEqual:sliding?@"default":@"proportional"],"Gemma4 RoPE flavor mismatch");
        int dim=sliding?local_dim:global_dim,kv=sliding?local_kv:global_kv;require(dim%2==0&&heads%kv==0,"invalid Gemma4 grouped heads");
        float theta=[rope[@"rope_theta"] floatValue],factor=sliding?1.f:[rope[@"partial_rotary_factor"] floatValue];
        require(std::isfinite(theta)&&theta>0&&factor>0&&factor<=1,"invalid Gemma4 rotary parameters");int count=int(factor*dim/2);
        auto frequencies=1.f/mx::power(Tensor(theta),mx::arange(0,2*count,2,mx::float32)/dim);
        if(count<dim/2)frequencies=mx::concatenate({frequencies,mx::zeros({dim/2-count},mx::float32)});
        auto angles=mx::reshape(mx::arange(length,mx::float32),{1,length,1})*mx::reshape(frequencies,{1,1,dim/2});angles=mx::concatenate({angles,angles},-1);
        auto q=mx::reshape(mx::arange(length),{length,1}),k=mx::reshape(mx::arange(length),{1,length});
        auto visible=k<=q;if(sliding)visible=visible&(k>q-window);
        visible=mx::reshape(visible,{1,1,length,length})&mx::reshape(padding!=0,{1,1,1,length});
        auto empty=mx::logical_not(mx::any(visible,-1,true));visible=visible|(empty&mx::reshape(q==k,{1,1,length,length}));
        auto mask=mx::where(visible,Tensor(0.f,x.dtype()),Tensor(-std::numeric_limits<float>::infinity(),x.dtype()));
        geometry.emplace(name,Attention{dim,kv,mx::astype(mx::expand_dims(mx::cos(angles),2),x.dtype()),mx::astype(mx::expand_dims(mx::sin(angles),2),x.dtype()),mask});
    }
    std::vector<Tensor> states{x};
    for(int i=0;i<layers;++i) {
        checkpoint(cancel);event("gemma_layer",i,layers);
        auto name=std::string([types[i] UTF8String]);require(geometry.count(name),"unknown Gemma4 attention type");auto& g=geometry.at(name);
        auto p="layers."+std::to_string(i),a=p+".self_attn";
        auto input=normalize(x,p+".input_layernorm");
        auto split=[&](const Tensor& t,int n){return mx::reshape(t,{1,length,n,g.dim});};
        auto rotate=[&](const Tensor& t){return t*g.cos+mx::concatenate({-slice_axis(t,-1,g.dim/2,g.dim),slice_axis(t,-1,0,g.dim/2)},-1)*g.sin;};
        auto q=mx::transpose(rotate(normalize(split(linear(input,weights,a+".q_proj"),heads),a+".q_norm")),{0,2,1,3});
        auto raw=split(linear(input,weights,a+".k_proj"),g.kv);
        auto v=([c[@"attention_k_eq_v"] boolValue]&&name=="full_attention")?raw:split(linear(input,weights,a+".v_proj"),g.kv);
        auto k=mx::transpose(rotate(normalize(raw,a+".k_norm")),{0,2,1,3});v=mx::transpose(normalize(v,""),{0,2,1,3});
        auto repeat=[&](const Tensor&t){return mx::reshape(mx::broadcast_to(mx::expand_dims(t,2),{1,g.kv,heads/g.kv,length,g.dim}),{1,heads,length,g.dim});};
        auto attention=mx::fast::scaled_dot_product_attention(q,repeat(k),repeat(v),1.f,"",g.mask);
        attention=mx::reshape(mx::transpose(attention,{0,2,1,3}),{1,length,heads*g.dim});
        x=x+normalize(linear(attention,weights,a+".o_proj"),p+".post_attention_layernorm");
        input=normalize(x,p+".pre_feedforward_layernorm");auto gate=linear(input,weights,p+".mlp.gate_proj");
        auto scalar=[&](float value){return Tensor(value,gate.dtype());};
        auto activation=scalar(.5f)*gate*(scalar(1.f)+mx::tanh(scalar(float(std::sqrt(2.0/M_PI)))*(gate+scalar(.044715f)*mx::power(gate,scalar(3.f)))));
        x=x+normalize(linear(activation*linear(input,weights,p+".mlp.up_proj"),weights,p+".mlp.down_proj"),p+".post_feedforward_layernorm");
        x=x*weights.at(p+".layer_scalar");mx::eval(x);states.push_back(x);
    }
    states.back()=normalize(states.back(),"norm");mx::eval(states.back());event("gemma_layer",layers,layers);return states;
}
}

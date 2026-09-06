#include "runtime.hpp"
#include <cmath>
#include <cstring>
#include <regex>
namespace tc {
void Weights::load(const std::filesystem::path& p,const Event& event,std::atomic<bool>& cancelled){
 if(!values_.empty())return;
 require(std::filesystem::is_directory(p),"missing component: "+p.string());
 std::vector<std::filesystem::path> shards;
 for(auto& f:std::filesystem::directory_iterator(p))if(f.path().extension()==".safetensors")shards.push_back(f.path());
 std::sort(shards.begin(),shards.end());require(!shards.empty(),"no safetensors in "+p.string());
 try {int i=0;for(auto& f:shards){checkpoint(cancelled);event("load_"+p.filename().string(),i,int(shards.size()));auto data=mx::load_safetensors(f.string());for(auto& [k,v]:data.first){require(!values_.count(k),"duplicate tensor: "+k);values_.emplace(k,v);}++i;} }catch(...){clear();throw;}
}
void Weights::load_file(const std::filesystem::path& path,const std::string& prefix) {
 auto data=mx::load_safetensors(path.string());
 for(auto&[name,value]:data.first)if(name.starts_with(prefix)){auto key=name.substr(prefix.size());require(!values_.count(key),"duplicate weight: "+key);values_.emplace(key,value);}
 require(!values_.empty(),"no tensors match component prefix");
}
const Tensor& Weights::at(const std::string& k)const{auto i=values_.find(k);require(i!=values_.end(),"missing weight: "+k);return i->second;}
bool Weights::has(const std::string& k)const{return values_.count(k);}
namespace {
struct LoRAPair { std::optional<Tensor> down,up,alpha; };
static bool remove_suffix(std::string& value,const std::string& suffix) {
 if(!value.ends_with(suffix))return false;value.resize(value.size()-suffix.size());return true;
}
static std::string strip_lora_prefix(std::string value) {
 for(const auto& prefix:{"base_model.model.","transformer.","diffusion_model.","model."})
  if(value.starts_with(prefix)){value.erase(0,strlen(prefix));break;}
 return value;
}
static std::string kohya_to_bfl(const std::string& stem) {
 std::smatch m;
 if(std::regex_match(stem,m,std::regex("^(?:lora_unet_|lycoris_)(double_blocks|single_blocks)_([0-9]+)_(.+)$"))) {
  static const std::map<std::string,std::string> tails={
   {"img_attn_qkv","img_attn.qkv"},{"img_attn_proj","img_attn.proj"},{"txt_attn_qkv","txt_attn.qkv"},{"txt_attn_proj","txt_attn.proj"},
   {"img_mlp_0","img_mlp.0"},{"img_mlp_2","img_mlp.2"},{"txt_mlp_0","txt_mlp.0"},{"txt_mlp_2","txt_mlp.2"},
   {"img_mod_lin","img_mod.lin"},{"txt_mod_lin","txt_mod.lin"},{"linear1","linear1"},{"linear2","linear2"},{"modulation_lin","modulation.lin"}};
  auto found=tails.find(m[3]);if(found!=tails.end())return std::string(m[1])+"."+std::string(m[2])+"."+found->second;
 }
 return stem;
}
static std::vector<std::string> lora_targets(std::string stem) {
 stem=kohya_to_bfl(strip_lora_prefix(stem));
 std::vector<std::string> result{stem};
 std::smatch m;
 if(std::regex_match(stem,m,std::regex("^double_blocks\\.([0-9]+)\\.(.+)$"))) {
  std::string p="transformer_blocks."+std::string(m[1])+".", tail=m[2];
  if(tail=="img_attn.qkv")return {p+"attn.to_q",p+"attn.to_k",p+"attn.to_v"};
  if(tail=="txt_attn.qkv")return {p+"attn.add_q_proj",p+"attn.add_k_proj",p+"attn.add_v_proj"};
  static const std::map<std::string,std::string> map={{"img_attn.proj","attn.to_out.0"},{"txt_attn.proj","attn.to_add_out"},{"img_mlp.0","ff.linear_in"},{"img_mlp.2","ff.linear_out"},{"txt_mlp.0","ff_context.linear_in"},{"txt_mlp.2","ff_context.linear_out"},{"img_mod.lin","double_stream_modulation_img.linear"},{"txt_mod.lin","double_stream_modulation_txt.linear"}};
  if(auto found=map.find(tail);found!=map.end())return {p+found->second};
 }
 if(std::regex_match(stem,m,std::regex("^single_blocks\\.([0-9]+)\\.(.+)$"))) {
  std::string p="single_transformer_blocks."+std::string(m[1])+".attn.",tail=m[2];
  if(tail=="linear1")return {p+"to_qkv_mlp_proj"};if(tail=="linear2")return {p+"to_out"};
  if(tail=="modulation.lin")return {"single_stream_modulation.linear"};
 }
 static const std::map<std::string,std::string> aliases={{"img_in","x_embedder"},{"txt_in","context_embedder"},{"time_in.in_layer","time_guidance_embed.timestep_embedder.linear_1"},{"time_in.out_layer","time_guidance_embed.timestep_embedder.linear_2"},{"final_layer.adaLN_modulation.1","norm_out.linear"},{"final_layer.linear","proj_out"}};
 if(auto found=aliases.find(stem);found!=aliases.end())return {found->second};
 return result;
}
}
size_t Weights::apply_loras(const std::vector<LoRAAsset>& adapters,const std::string& role,const Event& event,std::atomic<bool>& cancel) {
 size_t applied=0;
 for(size_t adapter_index=0;adapter_index<adapters.size();++adapter_index) {
  const auto& adapter=adapters[adapter_index];if(adapter.role!=role)continue;
  require(std::filesystem::is_regular_file(adapter.path),"LoRA file missing: "+adapter.path);
  checkpoint(cancel);event("load_lora",int(adapter_index),int(adapters.size()));
  auto data=mx::load_safetensors(adapter.path);std::map<std::string,LoRAPair> pairs;
  for(auto&[raw,value]:data.first) {
   std::string stem=raw;
   if(remove_suffix(stem,".lora_A.default.weight")||remove_suffix(stem,".lora_A.weight")||remove_suffix(stem,".lora_down.weight")||remove_suffix(stem,".lora.down.weight")||remove_suffix(stem,".lora_A"))pairs[stem].down=value;
   else if(remove_suffix(stem,".lora_B.default.weight")||remove_suffix(stem,".lora_B.weight")||remove_suffix(stem,".lora_up.weight")||remove_suffix(stem,".lora.up.weight")||remove_suffix(stem,".lora_B"))pairs[stem].up=value;
   else if(remove_suffix(stem,".alpha")||remove_suffix(stem,".lora_alpha"))pairs[stem].alpha=value;
  }
  size_t adapter_applied=0;
  for(auto&[stem,pair]:pairs) {
   if(!pair.down&&!pair.up)continue;
   require(pair.down&&pair.up,"incomplete LoRA pair: "+stem);
   const auto& down=*pair.down;const auto& up=*pair.up;
   require(down.ndim()==2&&up.ndim()==2&&down.shape(0)==up.shape(1),"invalid LoRA rank geometry: "+stem);
   float scale=adapter.strength;if(pair.alpha){require(pair.alpha->size()==1,"LoRA alpha must be scalar: "+stem);mx::eval(*pair.alpha);scale*=pair.alpha->item<float>()/float(down.shape(0));}
   auto targets=lora_targets(stem);int offset=0;
   for(const auto& target:targets) {
    auto key=target.ends_with(".weight")?target:target+".weight";auto found=values_.find(key);
    if(found==values_.end())continue;
    const auto& base=found->second;require(base.ndim()==2&&base.shape(1)==down.shape(1),"LoRA input does not match "+key);
    int rows=base.shape(0);require(offset+rows<=up.shape(0),"LoRA output does not match "+key);
    auto selected=targets.size()==1?up:slice_axis(up,0,offset,offset+rows);offset+=rows;
    require(selected.shape(0)==rows&&selected.shape(1)==down.shape(0),"LoRA rank does not match "+key);
    auto delta=mx::matmul(mx::astype(selected,mx::float32),mx::astype(down,mx::float32))*Tensor(scale,mx::float32);
    auto merged=mx::astype(mx::astype(base,mx::float32)+delta,base.dtype());mx::eval(merged);found->second=merged;++adapter_applied;
   }
  }
  require(adapter_applied>0,"LoRA did not match any "+role+" weights: "+adapter.path);applied+=adapter_applied;
  event("load_lora",int(adapter_index+1),int(adapters.size()));
 }
 return applied;
}
void Weights::clear(){values_.clear();}
size_t Weights::bytes()const{size_t n=0;for(auto&[k,v]:values_)n+=v.nbytes();return n;}
Tensor linear(const Tensor& x,const Weights&w,const std::string& p){auto wt=mx::transpose(w.at(p+".weight"));return w.has(p+".bias")?mx::addmm(w.at(p+".bias"),x,wt):mx::matmul(x,wt);}
Tensor silu(const Tensor& x){static auto compiled=mx::compile([](const std::vector<Tensor>& a){return std::vector<Tensor>{a[0]*mx::sigmoid(a[0])};},true);return compiled({x})[0];}
Tensor rms(const Tensor& x,const Tensor&w,float eps){auto f=mx::astype(x,mx::float32);return mx::astype(f*mx::rsqrt(mx::mean(mx::square(f),-1,true)+eps)*mx::astype(w,mx::float32),x.dtype());}
Tensor norm(const Tensor& x){return mx::fast::layer_norm(x,{}, {},1e-6f);}
Tensor slice_axis(const Tensor& x,int axis,int start,int stop){if(axis<0)axis+=x.ndim();mx::Shape a(x.ndim(),0),b=x.shape();a[axis]=start;b[axis]=stop;return mx::slice(x,a,b);}
Tensor heads(const Tensor& x,int n,int d){return mx::transpose(mx::reshape(x,{1,x.shape(1),n,d}),{0,2,1,3});}
Tensor attend(const Tensor&q,const Tensor&k,const Tensor&v,bool f32,const std::optional<Tensor>&mask){auto dtype=q.dtype();auto a=mx::fast::scaled_dot_product_attention(f32?mx::astype(q,mx::float32):q,f32?mx::astype(k,mx::float32):k,f32?mx::astype(v,mx::float32):v,1.f/std::sqrt(float(q.shape(-1))),"",mask);if(f32)a=mx::astype(a,dtype);return mx::reshape(mx::transpose(a,{0,2,1,3}),{1,q.shape(2),q.shape(1)*q.shape(3)});}
Tensor rope_pairs(const Tensor&x,const Tensor&cos,const Tensor&sin){auto f=mx::reshape(mx::astype(x,mx::float32),{1,x.shape(1),x.shape(2),64,2});auto parts=mx::split(f,2,-1);auto a=mx::squeeze(parts[0],-1),b=mx::squeeze(parts[1],-1);auto c=mx::reshape(cos,{1,1,x.shape(2),64}),s=mx::reshape(sin,{1,1,x.shape(2),64});return mx::astype(mx::reshape(mx::stack({a*c-b*s,b*c+a*s},-1),x.shape()),x.dtype());}
std::vector<float> flux_gpu_sigmas(int tokens,int steps) {
 double m200=.00016927*tokens+.45666666,mu=m200;
 if(tokens<=4300){double m10=8.73809524e-5*tokens+1.89833333;double a=(m200-m10)/190.;mu=a*steps+(m200-200.*a);}
 auto t=mx::linspace(1.,1./steps,steps,mx::float32);
 auto e=mx::exp(Tensor(float(mu)));
 auto values=e/(e+mx::power(1.f/t-1.f,Tensor(1.f)));mx::eval(values);
 std::vector<float> result(values.data<float>(),values.data<float>()+steps);result.push_back(0);return result;
}
Tensor euler_step(const Tensor&x,const Tensor&noise,float dt){
 if(x.dtype()==mx::float32){static auto step=mx::compile([](const std::vector<Tensor>& a){return std::vector<Tensor>{a[0]+a[2]*a[1]};},true);return step({x,noise,Tensor(dt)})[0];}
 // Custom Metal kernel participates in MLX's dependency/allocator system.
 // Keep the reference BF16 multiply rounding before the residual addition.
 static auto kernel=mx::fast::metal_kernel("tc_euler",{"x","noise","dt","count"},{"out"},"uint i = thread_position_in_grid.x; if(i < uint(count)) { T product = T(noise[i] * dt); out[i] = T(x[i] + product); }");
 auto d=mx::astype(Tensor(dt),x.dtype());
 return kernel({x,noise,d,Tensor(int(x.size()))},{x.shape()},{x.dtype()},{int(x.size()),1,1},{256,1,1},{{"T",x.dtype()}},{},false,{})[0];
}
}

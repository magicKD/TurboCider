#include "runtime.hpp"
#include "../backends/coreml.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
namespace tc {
namespace {
static std::string sha256_file(const std::filesystem::path& path) {
 std::ifstream stream(path,std::ios::binary);if(!stream.good())return {};
 CC_SHA256_CTX context;if(CC_SHA256_Init(&context)!=1)return {};
 std::array<char,1<<20> buffer{};
 while(stream.good()){
  stream.read(buffer.data(),static_cast<std::streamsize>(buffer.size()));
  auto count=stream.gcount();
  if(count>0&&CC_SHA256_Update(&context,buffer.data(),static_cast<CC_LONG>(count))!=1)return {};
 }
 if(!stream.eof())return {};
 unsigned char digest[CC_SHA256_DIGEST_LENGTH];if(CC_SHA256_Final(digest,&context)!=1)return {};
 static constexpr char hex[]="0123456789abcdef";std::string result(CC_SHA256_DIGEST_LENGTH*2,'0');
 for(size_t i=0;i<CC_SHA256_DIGEST_LENGTH;++i){result[i*2]=hex[digest[i]>>4];result[i*2+1]=hex[digest[i]&15];}
 return result;
}
}
Flux::Flux(const std::filesystem::path&root,std::string model_id):root_(root),model_id_(std::move(model_id)),tokenizer_(root/"tokenizer"){
 auto t=read_json(root/"transformer/config.json"),q=read_json(root/"text_encoder/config.json"),v=read_json(root/"vae/config.json");
 heads_=[t[@"num_attention_heads"] intValue];hidden_=heads_*[t[@"attention_head_dim"] intValue];dual_layers_=[t[@"num_layers"] intValue];single_layers_=[t[@"num_single_layers"] intValue];
 require([t[@"attention_head_dim"] intValue]==128&&[t[@"in_channels"] intValue]==128&&![t[@"guidance_embeds"] boolValue],"unsupported FLUX.2 configuration");
 require((model_id_=="flux2-klein-4b"&&heads_==24&&dual_layers_==5&&single_layers_==20)||(model_id_=="flux2-klein-9b"&&heads_==32&&dual_layers_==8&&single_layers_==24),"FLUX config does not match selected module");
 text_hidden_=[q[@"hidden_size"] intValue];text_heads_=[q[@"num_attention_heads"] intValue];text_kv_heads_=[q[@"num_key_value_heads"] intValue];text_layers_=[q[@"num_hidden_layers"] intValue];
 require(text_hidden_>0&&text_heads_>0&&text_kv_heads_>0&&text_layers_==36&&[t[@"joint_attention_dim"] intValue]==text_hidden_*3,"unsupported Qwen3 configuration");
 require([v[@"latent_channels"] intValue]==32,"unsupported Flux VAE");
}
Flux::~Flux()=default;
NSDictionary *Flux::generate(const Request&r,const Event&event,std::atomic<bool>&cancelled){
 auto begin=Clock::now();auto plan=make_plan(r);
 require(r.model==model_id_,"model executor unavailable; see static acceptance plan");
 require(!r.prompt.empty()&&!r.output.empty(),"prompt and output are required");
 require(std::filesystem::path(r.output).extension()==".png","native image output must be .png");
 auto physical=[NSProcessInfo processInfo].physicalMemory;
 require(physical>=[plan[@"memory_estimate_bytes"] unsignedLongLongValue]+(4ull<<30),"insufficient physical memory for the conservative BF16 plan");
 require(!r.memory_budget_bytes||r.memory_budget_bytes>=[plan[@"memory_estimate_bytes"] unsignedLongLongValue],"profile memory budget is below the BF16 plan estimate");
 mx::reset_peak_memory();mx::set_cache_limit(r.allocator_cache_bytes);
 auto tokens=tokenizer_.prompt(r.prompt,r.dynamic_text);
 auto identify_lora=[&](const LoRAAsset& asset,LoRAAsset& normalized){
  auto requested=std::filesystem::path(asset.path);
  require(std::filesystem::is_regular_file(requested),"LoRA file missing: "+asset.path);
  std::error_code error;
  auto canonical=std::filesystem::canonical(requested,error);
  require(!error,"cannot canonicalize LoRA path: "+asset.path);
  normalized=asset;normalized.path=canonical.string();
  auto bytes=std::filesystem::file_size(canonical,error);
  require(!error,"cannot inspect LoRA size: "+canonical.string());
  auto mtime=std::filesystem::last_write_time(canonical,error);
  require(!error,"cannot inspect LoRA timestamp: "+canonical.string());
  auto key=canonical.string();
  auto found=lora_hash_cache_.find(key);
  if(found==lora_hash_cache_.end()||found->second.bytes!=bytes||found->second.mtime!=mtime){
   auto digest=sha256_file(canonical);
   require(!digest.empty(),"cannot hash LoRA file: "+canonical.string());
   std::error_code verify_error;
   auto final_bytes=std::filesystem::file_size(canonical,verify_error);
   require(!verify_error&&final_bytes==bytes,"LoRA changed while it was being hashed: "+canonical.string());
   auto final_mtime=std::filesystem::last_write_time(canonical,verify_error);
   require(!verify_error&&final_mtime==mtime,"LoRA changed while it was being hashed: "+canonical.string());
   lora_hash_cache_[key]=LoRAFileHash{bytes,mtime,std::move(digest)};
   found=lora_hash_cache_.find(key);
  }
  const auto& cached=found->second;
  return std::to_string(asset.role.size())+":"+asset.role+":"+
   std::to_string(key.size())+":"+key+":"+std::to_string(bytes)+":"+
   std::to_string(static_cast<long long>(mtime.time_since_epoch().count()))+":"+
   cached.sha256+":"+std::to_string(std::bit_cast<uint32_t>(asset.strength))+";";
 };
 std::string lora_identity;std::vector<LoRAAsset> normalized_loras;
 normalized_loras.reserve(r.loras.size());
 for(const auto& l:r.loras){normalized_loras.emplace_back();lora_identity+=identify_lora(l,normalized_loras.back());}
 if(lora_identity!=cached_lora_identity_){cached_lora_identity_=lora_identity;cached_conditioning_.reset();transformer_.clear();vae_.clear();active_loras_=std::move(normalized_loras);mx::clear_cache();}
 if(r.execution!="gpu_ane")hybrid_.reset();
 auto dump=[&](const std::string&name,const Tensor&a){if(!r.dump.empty()){std::filesystem::create_directories(r.dump);mx::save_safetensors((std::filesystem::path(r.dump)/(name+".safetensors")).string(),{{"tensor",a}});}};
 bool prompt_hit=cached_conditioning_.has_value()&&cached_prompt_==r.prompt&&cached_dynamic_==r.dynamic_text;
 auto text_start=Clock::now();
 if(!prompt_hit){
  // A different prompt must not hold the resident DiT while encoding Qwen.
  hybrid_.reset();transformer_.clear();vae_.clear();cached_conditioning_.reset();mx::clear_cache();
  cached_conditioning_=encode(tokens,event,cancelled);cached_prompt_=r.prompt;cached_dynamic_=r.dynamic_text;mx::clear_cache();
 } else event("text_cache_hit",1,1);
 double text_s=std::chrono::duration<double>(Clock::now()-text_start).count();
 std::optional<Tensor> reference_latents, clean_latents;
 std::vector<float> reference_ids;
 auto image_start=Clock::now();
 if(!r.inputs.empty()) {
  vae_.load(root_/"vae",event,cancelled);
  for(size_t index=0;index<r.inputs.size();++index) {
   checkpoint(cancelled);
   auto image=load_image_tensor(r.inputs[index].path,r.width,r.height,r.operation=="image.edit");
   dump("input_image_"+std::to_string(index),image);
   auto encoded=encode_image(image,event,cancelled);dump("image_latent_"+std::to_string(index),encoded);
   if(r.operation=="image.transform")clean_latents=encoded;
   else {
    reference_latents=reference_latents?mx::concatenate({*reference_latents,encoded},1):encoded;
    int h=image.shape(1)/16,w=image.shape(2)/16;
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){reference_ids.push_back(float(10+10*index));reference_ids.push_back(float(y));reference_ids.push_back(float(x));reference_ids.push_back(0);}
   }
  }
 }
 double image_s=std::chrono::duration<double>(Clock::now()-image_start).count();
 int actual_tokens=int(tokens.ids.size())+(r.width/16)*(r.height/16)+(reference_latents?reference_latents->shape(1):0);
 require(actual_tokens<=20000,"request exceeds native token workspace budget");
 auto hybrid_start=Clock::now();
 if(r.execution=="gpu_ane"&&(!hybrid_||hybrid_->manifest!=r.ane_manifest))hybrid_=std::make_unique<HybridSession>(r.ane_manifest,root_,actual_tokens,event,cancelled,r.warmup_iterations);
 if(hybrid_)require(actual_tokens<=hybrid_->rows,"cached hybrid bucket too small");
 double hybrid_s=std::chrono::duration<double>(Clock::now()-hybrid_start).count();
 auto text=*cached_conditioning_;dump("conditioning",text);
 checkpoint(cancelled);bool transformer_cold=transformer_.bytes()==0;transformer_.load(root_/"transformer",event,cancelled);
 if(transformer_cold&&!active_loras_.empty())transformer_.apply_loras(active_loras_,"transformer",event,cancelled);
 auto z=mx::astype(mx::random::normal({1,128,r.height/16,r.width/16},mx::float32,0,1,mx::random::key(r.seed)),mx::bfloat16);
 z=mx::transpose(mx::reshape(z,{1,128,(r.height/16)*(r.width/16)}),{0,2,1});mx::eval(z);dump("initial_latent",z);
 auto sigmas=flux_gpu_sigmas(z.shape(1),r.steps);
 int start_step=0;
 if(clean_latents && r.inputs[0].strength>0) {
  start_step=std::max(1,int(r.steps*r.inputs[0].strength));
  auto sigma=Tensor(sigmas[start_step]);
  z=(Tensor(1.f)-sigma)*(*clean_latents)+sigma*z;mx::eval(z);dump("conditioned_initial_latent",z);
 }
 auto dit_start=Clock::now();
 for(int i=start_step;i<r.steps;++i){checkpoint(cancelled);event("denoise",i,r.steps);
  auto model_input=reference_latents?mx::concatenate({z,*reference_latents},1):z;
  auto noise=denoise(model_input,text,sigmas[i],r.height,r.width,event,cancelled,reference_ids);
  if(reference_latents)noise=slice_axis(noise,1,0,z.shape(1));mx::eval(noise);dump("noise_"+std::to_string(i),noise);
  z=euler_step(z,noise,sigmas[i+1]-sigmas[i]);mx::eval(z);dump("latent_"+std::to_string(i),z);
  require(mx::all(mx::isfinite(z)).item<bool>(),"nonfinite latent");event("denoise",i+1,r.steps);
 }
 double dit_s=std::chrono::duration<double>(Clock::now()-dit_start).count();
 checkpoint(cancelled);if(r.residency=="component_staged"){transformer_.clear();hybrid_.reset();mx::clear_cache();}
 vae_.load(root_/"vae",event,cancelled);
 auto decode_start=Clock::now();
 auto pixels=decode(z,r.height,r.width,event,cancelled,r.dump);dump("pixels_nhwc",pixels);
 require(mx::all(mx::isfinite(pixels)).item<bool>(),"nonfinite decoded pixels");
 double decode_s=std::chrono::duration<double>(Clock::now()-decode_start).count();
 checkpoint(cancelled);event("export",0,1);checkpoint(cancelled);save_png(pixels,r.output);event("export",1,1);
 auto hybrid_metrics=hybrid_?hybrid_->metrics():@{};
 if(r.residency=="component_staged"){vae_.clear();mx::clear_cache();}
 double seconds=std::chrono::duration<double>(Clock::now()-begin).count();
 return @{@"schema_version":@1,@"model":@(r.model.c_str()),@"output":@(r.output.c_str()),@"width":@(r.width),@"height":@(r.height),@"seed":@(r.seed),@"steps":@(r.steps),@"operation":@(r.operation.c_str()),@"reference_tokens":@(reference_latents?reference_latents->shape(1):0),@"actual_denoise_steps":@(r.steps-start_step),@"text_tokens":@(tokens.ids.size()),@"valid_text_tokens":@(tokens.valid),@"prompt_cache_hit":@(prompt_hit),@"plan":plan,@"timings_seconds":@{@"request_wall":@(seconds),@"text_encode":@(text_s),@"image_encode":@(image_s),@"hybrid_setup":@(hybrid_s),@"denoise":@(dit_s),@"vae_decode":@(decode_s)},@"memory":@{@"mlx_peak_bytes":@(mx::get_peak_memory()),@"mlx_active_bytes":@(mx::get_active_memory()),@"scope":@"MLX allocator; excludes Core ML/OS/file cache"},@"hybrid":hybrid_metrics,@"validation":@"candidate; consult recorded parity suite"};
}
}

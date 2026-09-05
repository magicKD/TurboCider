#include "runtime.hpp"
#include "../backends/coreml.hpp"
#include <cmath>
namespace tc {
Flux::Flux(const std::filesystem::path&root):root_(root),tokenizer_(root/"tokenizer"){
 auto t=read_json(root/"transformer/config.json"),q=read_json(root/"text_encoder/config.json"),v=read_json(root/"vae/config.json");
 require([t[@"num_attention_heads"] intValue]==24&&[t[@"attention_head_dim"] intValue]==128&&[t[@"num_layers"] intValue]==5&&[t[@"num_single_layers"] intValue]==20&&[t[@"joint_attention_dim"] intValue]==7680&&[t[@"in_channels"] intValue]==128&&![t[@"guidance_embeds"] boolValue],"only official FLUX.2 Klein 4B configuration supported");
 require([q[@"hidden_size"] intValue]==2560&&[q[@"num_hidden_layers"] intValue]==36&&[q[@"num_attention_heads"] intValue]==32&&[q[@"num_key_value_heads"] intValue]==8,"unsupported Qwen3 configuration");
 require([v[@"latent_channels"] intValue]==32,"unsupported Flux VAE");
}
Flux::~Flux()=default;
NSDictionary *Flux::generate(const Request&r,const Event&event,std::atomic<bool>&cancelled){
 auto begin=Clock::now();auto plan=make_plan(r);
 require(r.model=="flux2-klein-4b","model executor unavailable; see static acceptance plan");
 require(!r.prompt.empty()&&!r.output.empty(),"prompt and output are required");
 require(std::filesystem::path(r.output).extension()==".png","native image output must be .png");
 auto physical=[NSProcessInfo processInfo].physicalMemory;
 require(physical>=[plan[@"memory_estimate_bytes"] unsignedLongLongValue]+(4ull<<30),"insufficient physical memory for the conservative BF16 plan");
 require(!r.memory_budget_bytes||r.memory_budget_bytes>=[plan[@"memory_estimate_bytes"] unsignedLongLongValue],"profile memory budget is below the BF16 plan estimate");
 mx::reset_peak_memory();mx::set_cache_limit(r.allocator_cache_bytes);
 auto tokens=tokenizer_.prompt(r.prompt,r.dynamic_text);
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
 checkpoint(cancelled);transformer_.load(root_/"transformer",event,cancelled);
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

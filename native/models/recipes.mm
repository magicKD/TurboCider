#include "runtime.hpp"
#include <set>
#include <cmath>
#include <algorithm>
namespace tc {
void validate_recipe(const Recipe& r) {
 std::set<std::string> done;
 for(auto& s:r.stages) {
  require(!s.id.empty() && s.iterations>0,"invalid recipe stage");
  require(!done.count(s.id),"duplicate recipe stage: "+s.id);
  for(auto& dep:s.dependencies) require(done.count(dep),"unsatisfied recipe dependency: "+dep);
  done.insert(s.id);
 }
 require(!done.empty(),"empty recipe");
}
std::vector<float> flux_sigmas(int tokens,int steps) {
 require(tokens>0 && steps>=1 && steps<=50,"invalid Flux schedule");
 double mu;
 double m200=.00016927*tokens+.45666666;
 if(tokens>4300) mu=m200;
 else { double m10=8.73809524e-5*tokens+1.89833333; double a=(m200-m10)/190.; mu=a*steps+m200-200*a; }
 std::vector<float> s;
 for(int i=0;i<steps;++i) {double t=1.-double(i)/steps;s.push_back(float(std::exp(mu)/(std::exp(mu)+1./t-1.)));}
 s.push_back(0);return s;
}
NSDictionary *make_plan(const Request& r) {
 auto recipe=model_recipe(r.model); validate_recipe(recipe);
 require(r.width>=64&&r.height>=64&&r.width<=2048&&r.height<=2048,"dimensions must be 64...2048");
 require(r.steps>=1&&r.steps<=50,"steps must be 1...50");
 require(r.execution=="gpu"||r.execution=="auto"||r.execution=="gpu_ane","unknown execution mode");
 bool hybrid=r.execution=="gpu_ane";
 std::string ane_identity;
 if(hybrid) {
  require(r.model!="flux2-klein-9b","FLUX.2 Klein 9B GPU+ANE partition is not validated; use gpu");
  require(!r.ane_manifest.empty(),"gpu_ane requires an explicit ANE manifest or partition directory");
  if(r.model=="ltx-2.5-distilled")
   ane_identity=validate_ltx_ane_profile(r.ane_manifest,r.width,r.height,r.frames,r.fps);
  if(r.model=="fastmetal-1.3b-qad")
   ane_identity=validate_fastmetal_ane_manifest(r.ane_manifest);
  require(!ane_identity.empty(),"invalid GPU+ANE artifact profile");
 }
 int dw=r.width,dh=r.height;
 module_for(r.model).validate(r);
 if(r.model=="ltx-2.5-distilled"&&!r.audio) {
  recipe.stages.erase(std::remove_if(recipe.stages.begin(),recipe.stages.end(),
      [](const Stage& stage) {
          return stage.id=="audio_vae_vocoder" || stage.id=="mux";
      }),recipe.stages.end());
  recipe.stages.push_back({"export",{"video_vae"}});
 }
 if(r.model.starts_with("flux2-klein-")&&!r.inputs.empty()) {
  recipe.stages.insert(recipe.stages.begin()+1,{"image_encode",{}});
  for(auto& stage:recipe.stages)if(stage.id=="denoise")stage.dependencies.push_back("image_encode");
 }
 if(r.model=="ltx-2.5-distilled"&&r.operation=="video.image") {
  recipe.stages.insert(recipe.stages.begin(),{"first_frame_vae_encode",{}});
  for(auto& stage:recipe.stages)if(stage.id=="av_stage1")stage.dependencies.push_back("first_frame_vae_encode");
 }
 validate_recipe(recipe);
 bool executable=recipe.executable;
 if(r.model=="ltx-2.5-distilled" &&
    (r.audio || r.operation!="video.generate")) executable=false;
 NSMutableArray *stages=[NSMutableArray array];
 for(auto& s:recipe.stages) {NSMutableArray *deps=[NSMutableArray array];for(auto& d:s.dependencies)[deps addObject:@(d.c_str())];[stages addObject:@{@"id":@(s.id.c_str()),@"dependencies":deps,@"iterations":@(s.id=="denoise"?r.steps:s.iterations)}];}
 uint64_t memory=0;
 if(r.model=="flux2-klein-4b")memory=uint64_t((hybrid?16ull:12ull)<<30)+uint64_t(r.width)*r.height*8192;
 else if(r.model=="flux2-klein-9b")memory=uint64_t(28ull<<30)+uint64_t(r.width)*r.height*12288;
 else if(r.model=="ltx-2.5-distilled")memory=uint64_t(36ull<<30)+uint64_t(r.width)*r.height*r.frames*128;
 else if(r.model=="minimax-h3-turbo")memory=uint64_t(32ull<<30)+uint64_t(r.width)*r.height*r.frames*64;
 else if(r.model=="fastmetal-1.3b-qad")memory=uint64_t((hybrid?12ull:10ull)<<30)+uint64_t(r.width)*r.height*r.frames*48;
 NSString *validation=r.model.starts_with("flux2")?@"native_candidate":r.model=="minimax-h3-turbo"?@"manifest_verified_native":r.model=="fastmetal-1.3b-qad"?@"manifest_verified_python_runtime":r.model=="ltx-2.5-distilled"?(executable?@"native_video_executor":@"native_capability_gated"):@"weights_pending";
 NSString *weight_validation=r.model.starts_with("flux2")?@"see parity evidence":r.model=="minimax-h3-turbo"&&!r.loras.empty()?@"runtime-cache-or-sidecar-verified-at-execution":r.model=="minimax-h3-turbo"?@"manifest-verified":r.model=="fastmetal-1.3b-qad"&&!r.loras.empty()?@"premerged-manifest-verified-at-execution":r.model=="fastmetal-1.3b-qad"?@"checkpoint-and-ane-identity-verified-at-load":r.model=="ltx-2.5-distilled"&&!r.loras.empty()?@"runtime-cache-or-sidecar-verified-at-execution":r.model=="ltx-2.5-distilled"?@"checkpoint-validated-at-load":@"pending";
 NSString *lora_fusion=r.loras.empty()?@"none":(r.model=="minimax-h3-turbo"||r.model=="ltx-2.5-distilled")?@"runtime_bake_cache":(r.model=="fastmetal-1.3b-qad"?@"premerged_manifest_verified":@"load_time_baked");
 bool ltx_parallel=ane_identity.ends_with(":parallel");
 NSString *backend=hybrid?(r.model=="fastmetal-1.3b-qad"?@"fastmetal-mlx+ane_parallel":r.model=="ltx-2.5-distilled"?(ltx_parallel?@"gpu+ane_parallel":@"gpu+ane_serial"):@"gpu+ane_parallel"):r.model=="minimax-h3-turbo"?@"h3-metal-mps":r.model=="ltx-2.5-distilled"?@"ltx-metal-mps":r.model=="fastmetal-1.3b-qad"?@"fastmetal-mlx":@"mlx_cpp_metal";
 NSString *audio_capability=r.model=="ltx-2.5-distilled"?(r.audio?@"latent_to_48khz_aac_candidate":@"video_only_native"):@"not_applicable";
 return @{@"schema_version":@1,@"model":@(r.model.c_str()),@"executable":@(executable),@"validation":validation,@"backend":backend,@"execution":hybrid?@"gpu_ane":@"gpu",@"precision":hybrid?@"bf16_gpu+int8_ane_fp16_join":@"bf16",@"algorithm_approximations":hybrid?@[@"explicit_partition_quantization"]:@[],@"requested_shape":@[@(r.width),@(r.height),@(r.frames)],@"decoded_shape":@[@(dw),@(dh),@(r.frames)],@"stages":stages,@"memory_estimate_bytes":memory?@(memory):[NSNull null],@"memory_estimate_kind":@"conservative_heuristic_not_hard_limit",@"operation":@(r.operation.c_str()),@"residency":@(r.residency.c_str()),@"profile_identity":@(r.profile_identity.c_str()),@"ane_profile_identity":ane_identity.empty()?(id)[NSNull null]:@(ane_identity.c_str()),@"lora_count":@(r.loras.size()),@"lora_fusion":lora_fusion,@"weight_validation":weight_validation,@"audio":@(r.audio),@"audio_capability":audio_capability,@"limitation":executable?@"capabilities depend on model artifacts and configured hardware":(r.model=="ltx-2.5-distilled"?@"operation remains capability-gated":@"backend artifact required")};
}
}

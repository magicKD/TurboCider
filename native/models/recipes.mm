#include "runtime.hpp"
#include <set>
#include <cmath>
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
 if(hybrid)require(r.model=="flux2-klein-4b"&&r.allow_approximation&&!r.ane_manifest.empty(),"experimental hybrid requires FLUX, allow_approximation=true and ane_manifest");
 int dw=r.width,dh=r.height;
 module_for(r.model).validate(r);
 if(r.model=="flux2-klein-4b"&&!r.inputs.empty()) {
  recipe.stages.insert(recipe.stages.begin()+1,{"image_encode",{}});
  for(auto& stage:recipe.stages)if(stage.id=="denoise")stage.dependencies.push_back("image_encode");
 }
 validate_recipe(recipe);
 NSMutableArray *stages=[NSMutableArray array];
 for(auto& s:recipe.stages) {NSMutableArray *deps=[NSMutableArray array];for(auto& d:s.dependencies)[deps addObject:@(d.c_str())];[stages addObject:@{@"id":@(s.id.c_str()),@"dependencies":deps,@"iterations":@(s.id=="denoise"?r.steps:s.iterations)}];}
 return @{@"selection_pending":@(r.execution=="auto"),@"requested_execution":@(r.execution.c_str()),@"schema_version":@1,@"model":@(r.model.c_str()),@"executable":@(recipe.executable),@"validation":r.model=="flux2-klein-4b"?@"native_candidate":@"weights_pending",@"backend":hybrid?@"mlx_cpp_metal+coreml":r.model=="minimax-h3-turbo"?@"metal_mps_native":@"mlx_cpp_metal",@"execution":hybrid?@"gpu_ane_experimental":@"gpu",@"gpu_graph":r.compile_gpu?@"compiled_single_blocks":@"eager_blocks",@"precision":hybrid?@"bf16_gpu+int8_mlp_fp16_io":@"bf16",@"algorithm_approximations":hybrid?@[@"single_block_mlp_int8_per_channel"]:@[],@"requested_shape":@[@(r.width),@(r.height),@(r.frames)],@"decoded_shape":@[@(dw),@(dh),@(r.frames)],@"stages":stages,@"memory_estimate_bytes":r.model=="flux2-klein-4b"?@(uint64_t((hybrid?16ull:12ull)<<30)+uint64_t(r.width)*r.height*8192):[NSNull null],@"memory_estimate_kind":@"conservative_heuristic_not_hard_limit",@"operation":@(r.operation.c_str()),@"residency":@(r.residency.c_str()),@"profile_identity":@(r.profile_identity.c_str()),@"weight_validation":r.model=="flux2-klein-4b"?@"see parity evidence":@"pending",@"limitation":recipe.executable?@"capabilities depend on model artifacts and configured hardware":@"native executor migration incomplete"};
}
}

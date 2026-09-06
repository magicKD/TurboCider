#include "runtime.hpp"
#include "../models/ltx_runtime/ltx_gemma_tokenizer.h"
#include "../models/ltx_runtime/ltx_weights.h"
#import <Metal/Metal.h>
#include <mlx/version.h>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
namespace {std::mutex execution_mutex;
struct DeviceLease {
 int fd=-1;
 DeviceLease(){
  auto path=std::string("/private/tmp/turbocider-gpu-")+std::to_string(geteuid())+".lock";
  fd=open(path.c_str(),O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600);
  if(fd<0)throw std::runtime_error("cannot open GPU coordination lock");
  struct stat info{};
  if(fstat(fd,&info)!=0||!S_ISREG(info.st_mode)||info.st_uid!=geteuid()||flock(fd,LOCK_EX|LOCK_NB)!=0){close(fd);fd=-1;throw std::runtime_error("GPU is busy in another TurboCider process; submit through the shared service queue");}
 }
 ~DeviceLease(){if(fd>=0){flock(fd,LOCK_UN);close(fd);}}
};
// DispatchQueue is serial but does not guarantee a stable OS thread. Persistent
// tensors need streams valid across callers; execution_mutex serializes access.
void configure_streams(){
 using namespace tc;
 static auto cpu=mx::new_thread_unsafe_stream(mx::Device(mx::Device::cpu));
 static auto gpu=mx::new_thread_unsafe_stream(mx::Device(mx::Device::gpu));
 mx::set_default_stream(cpu);mx::set_default_stream(gpu);
 mx::set_default_device(mx::Device(mx::Device::gpu));
}
char *copy(const std::string&s){auto p=strdup(s.c_str());if(!p)throw std::bad_alloc();return p;}
int fail(char**error,const std::exception&e){if(error)*error=strdup(e.what());return dynamic_cast<const tc::Cancelled*>(&e)?2:1;}
}
namespace tc {
NSDictionary *system_info(){id<MTLDevice>d=MTLCreateSystemDefaultDevice();return @{@"abi":@1,@"engine_version":@"0.2.0-native-dev",@"gpu_available":@(d!=nil),@"gpu":d.name?:@"unavailable",@"physical_memory_bytes":@([NSProcessInfo processInfo].physicalMemory),@"recommended_working_set_bytes":@(d?d.recommendedMaxWorkingSetSize:0),@"os":[NSProcessInfo processInfo].operatingSystemVersionString,@"mlx_version":@(mx::version()),@"runtime_dependencies":@[@"libmlx",@"Metal",@"Foundation",@"ImageIO"],@"python_runtime_required":@NO,@"optional_python_models":@[@"fastmetal-1.3b-qad"]};}
}
uint32_t tc_abi_version(void){return 1;}
void tc_string_free(char*s){free(s);}
char *tc_system_json(void){@autoreleasepool{try{return copy(tc::json(tc::system_info()));}catch(...){return strdup("{\"error\":\"system probe failed\"}");}}}
char *tc_models_json(void){@autoreleasepool{try{return copy(tc::json(tc::describe_modules()));}catch(...){return strdup("{}");}}}
int tc_plan_json(const char*r,char**out,char**error){if(out)*out=nullptr;if(error)*error=nullptr;@autoreleasepool{try{tc::require(out,"missing output pointer");*out=copy(tc::json(tc::make_plan(tc::request_from_json(tc::parse_json(r)))));return 0;}catch(const std::exception&e){return fail(error,e);}catch(...){if(error)*error=strdup("unknown native error");return 1;}}}
int tc_engine_create_model(const char*id,const char*path,tc_engine**engine,char**error){if(engine)*engine=nullptr;if(error)*error=nullptr;@autoreleasepool{try{tc::require(id&&path&&engine,"missing model id/path or output handle");auto&module=tc::module_for(id);tc::require(bool(module.create),"model executor unavailable");NSDictionary *description=module.describe();tc::require([description[@"executor"] boolValue],"model executor unavailable");auto e=std::make_unique<tc_engine>();e->session=module.create(path);*engine=e.release();return 0;}catch(const std::exception&e){return fail(error,e);}catch(...){if(error)*error=strdup("unknown native error");return 1;}}}
extern "C" int tc_engine_create_model_candidate(const char*id,const char*path,tc_engine**engine,char**error){if(engine)*engine=nullptr;if(error)*error=nullptr;@autoreleasepool{try{tc::require(id&&path&&engine,"missing model id/path or output handle");tc::require(std::strcmp(id,"ltx-2.5-distilled")==0,"candidate executor is restricted to LTX");auto&module=tc::module_for(id);tc::require(bool(module.create),"model candidate unavailable");auto e=std::make_unique<tc_engine>();e->session=module.create(path);*engine=e.release();return 0;}catch(const std::exception&e){return fail(error,e);}catch(...){if(error)*error=strdup("unknown candidate error");return 1;}}}
int tc_engine_create(const char*path,tc_engine**engine,char**error){return tc_engine_create_model("flux2-klein-4b",path,engine,error);}
void tc_engine_cancel(tc_engine*e){if(e)e->cancelled.store(true);}
void tc_engine_free(tc_engine*e){delete e;}
int tc_engine_generate(tc_engine*e,const char*r,tc_event_callback cb,void*ctx,char**out,char**error){if(out)*out=nullptr;if(error)*error=nullptr;@autoreleasepool{try{
 tc::require(e&&out,"missing engine or output");std::unique_lock<std::mutex>local(e->mutex,std::try_to_lock);tc::require(local.owns_lock(),"engine busy");
 // MLX uses process-global device/allocation policy. Serialize all embeddings.
 std::unique_lock<std::mutex>global(execution_mutex,std::try_to_lock);tc::require(global.owns_lock(),"native GPU runtime busy");
 DeviceLease device_lease;
 e->cancelled.store(false);bool parent_mlx=e->session->uses_parent_mlx();
 if(parent_mlx){tc::require(tc::mx::is_available(tc::mx::Device(tc::mx::Device::gpu)),"Metal GPU unavailable");configure_streams();}
 else tc::require(MTLCreateSystemDefaultDevice()!=nil,"Metal GPU unavailable");
 struct Drain {bool active;~Drain(){if(active)try{tc::mx::synchronize();}catch(...){}}} drain{parent_mlx};
 auto request=tc::request_from_json(tc::parse_json(r));uint64_t seq=0;auto begin=tc::Clock::now();
 tc::Event event=[&](const std::string&phase,int current,int total){if(!(phase=="export"&&current==total))tc::checkpoint(e->cancelled);if(cb){@autoreleasepool{auto s=tc::json(@{@"schema_version":@1,@"sequence":@(++seq),@"phase":@(phase.c_str()),@"completed":@(current),@"total":@(total),@"elapsed_seconds":@(std::chrono::duration<double>(tc::Clock::now()-begin).count())});cb(s.c_str(),ctx);}}};
 auto result=e->session->generate(request,event,e->cancelled);*out=copy(tc::json(result));return 0;
 }catch(const std::exception&ex){return fail(error,ex);}catch(...){if(error)*error=strdup("unknown native error");return 1;}}}
int tc_tokenize_json(const char*path,const char*prompt,char**out,char**error){if(out)*out=nullptr;if(error)*error=nullptr;@autoreleasepool{try{tc::require(path&&prompt&&out,"missing tokenizer input");tc::Tokenizer t(std::filesystem::path(path)/"tokenizer");auto ids=t.prompt(prompt);NSMutableArray*a=[NSMutableArray array];for(int i:ids.ids)[a addObject:@(i)];*out=copy(tc::json(@{@"ids":a,@"valid":@(ids.valid)}));return 0;}catch(const std::exception&e){return fail(error,e);}catch(...){return 1;}}}
int tc_ltx_gemma_tokenize_json(const char *tokenizer_json, const char *prompt,
                               uint32_t max_length, char **out, char **error) {
    if (out) *out = nullptr;
    if (error) *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(tokenizer_json && prompt && out,
                        "missing Gemma tokenizer input");
            char detail[1024] = {};
            ltx_gemma_tokenizer *tokenizer =
                ltx_gemma_tokenizer_load(tokenizer_json, detail, sizeof(detail));
            tc::require(tokenizer != nullptr, detail);
            uint32_t *ids = nullptr;
            uint8_t *mask = nullptr;
            size_t count = 0;
            int ok = ltx_gemma_tokenizer_encode(
                tokenizer, prompt, max_length, &ids, &mask, &count,
                detail, sizeof(detail));
            ltx_gemma_tokenizer_free(tokenizer);
            tc::require(ok, detail);
            NSMutableArray *idArray = [NSMutableArray arrayWithCapacity:count];
            NSMutableArray *maskArray = [NSMutableArray arrayWithCapacity:count];
            for (size_t index = 0; index < count; index++) {
                [idArray addObject:@(ids[index])];
                [maskArray addObject:@(mask[index])];
            }
            ltx_gemma_tokenizer_ids_free(ids, mask);
            *out = copy(tc::json(@{@"ids": idArray,
                                   @"mask": maskArray,
                                   @"count": @(count)}));
            return 0;
        } catch (const std::exception &e) {
            return fail(error, e);
        } catch (...) {
            if (error) *error = strdup("unknown native error");
            return 1;
        }
    }
}
int tc_ltx_gemma_inspect_json(const char *checkpoint, char **out, char **error) {
    if (out) *out = nullptr;
    if (error) *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(checkpoint && out, "missing Gemma checkpoint input");
            char detail[1024] = {};
            ltx_gemma_checkpoint_info info{};
            tc::require(ltx_gemma_checkpoint_inspect(
                            checkpoint, &info, detail, sizeof(detail)), detail);
            tc::require(ltx_gemma_checkpoint_validate(
                            &info, detail, sizeof(detail)), detail);
            *out = copy(tc::json(@{
                @"tensor_count": @(info.tensor_count),
                @"vocab_size": @(info.vocab_size),
                @"hidden_size": @(info.hidden_size),
                @"intermediate_size": @(info.intermediate_size),
                @"layers": @(info.num_layers),
                @"attention_heads": @(info.attention_heads),
                @"sliding_kv_heads": @(info.sliding_kv_heads),
                @"full_kv_heads": @(info.full_kv_heads),
                @"sliding_head_dim": @(info.sliding_head_dim),
                @"full_head_dim": @(info.full_head_dim),
                @"projection_input_dim": @(info.projection_input_dim),
                @"projection_video_dim": @(info.projection_video_dim),
                @"projection_audio_dim": @(info.projection_audio_dim),
                @"attention_k_eq_v": @(info.attention_k_eq_v != 0),
                @"quantization": @"int8_convrot_256",
                @"validated": @YES,
            }));
            return 0;
        } catch (const std::exception &e) {
            return fail(error, e);
        } catch (...) {
            if (error) *error = strdup("unknown native error");
            return 1;
        }
    }
}
int tc_ltx_lora_preflight_json(const char *model_path, const char *lora_path,
                               float strength, char **out, char **error) {
    if (out) *out = nullptr;
    if (error) *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(model_path && lora_path && out,
                        "missing LTX LoRA preflight input");
            tc::LoRAAsset lora;
            lora.path = lora_path;
            lora.strength = strength;
            lora.role = "transformer";
            tc::require(std::isfinite(strength) && strength > 0.0f &&
                            strength <= 4.0f,
                        "LTX LoRA strength must be in (0, 4]");
            *out = copy(tc::json(tc::preflight_ltx_lora(model_path, lora)));
            return 0;
        } catch (const std::exception &e) {
            return fail(error, e);
        } catch (...) {
            if (error) *error = strdup("unknown native error");
            return 1;
        }
    }
}
int tc_fastmetal_lora_preflight_json(const char *model_path, const char *lora_path,
                                     float strength, char **out, char **error) {
    if (out) *out = nullptr;
    if (error) *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(model_path && lora_path && out,
                        "missing FastMetal LoRA preflight input");
            tc::LoRAAsset lora;
            lora.path = lora_path;
            lora.strength = strength;
            lora.role = "transformer";
            tc::require(std::isfinite(strength) && strength > 0.0f &&
                            strength <= 4.0f,
                        "FastMetal LoRA strength must be in (0, 4]");
            *out = copy(tc::json(tc::preflight_fastmetal_lora(model_path, lora)));
            return 0;
        } catch (const std::exception &e) {
            return fail(error, e);
        } catch (...) {
            if (error) *error = strdup("unknown native error");
            return 1;
        }
    }
}
int tc_ltx_audio_preflight_json(const char *model_path, char **out,
                                char **error) {
    if (out) *out = nullptr;
    if (error) *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(model_path && out,
                        "missing LTX audio preflight input");
            *out = copy(tc::json(tc::preflight_ltx_audio(model_path)));
            return 0;
        } catch (const std::exception &e) {
            return fail(error, e);
        } catch (...) {
            if (error) *error = strdup("unknown native error");
            return 1;
        }
    }
}
int tc_native_self_test(char**out,char**error){if(out)*out=nullptr;if(error)*error=nullptr;@autoreleasepool{try{
 tc::require(out,"missing output");std::lock_guard<std::mutex>lock(execution_mutex);using namespace tc;
 for(auto&s:{"flux2-klein-4b","ltx-2.5-distilled","minimax-h3-turbo"})validate_recipe(model_recipe(s));
 bool rejected=false;try{validate_recipe({"bad",{{"a",{"b"}}},false});}catch(...){rejected=true;}require(rejected,"graph cycle/missing dependency accepted");
 auto sigmas=flux_sigmas(1024,4);require(sigmas.front()==1&&sigmas.back()==0&&sigmas[1]>sigmas[2],"bad schedule");
 configure_streams();auto x=mx::astype(mx::random::normal({4097},mx::float32,0.f,1.f,mx::random::key(17)),mx::bfloat16);auto noise=mx::astype(mx::sin(mx::astype(x,mx::float32)),mx::bfloat16);
 auto compiled_step=mx::compile([](const std::vector<Tensor>& a){return std::vector<Tensor>{a[0]+a[2]*a[1]};},true);
 for(float dt:{-.25f,-.04191464f,-.71749657f,.0317f}){auto step=euler_step(x,noise,dt);auto ref=compiled_step({x,noise,mx::astype(Tensor(dt),x.dtype())})[0];require(mx::all(step==ref).item<bool>(),"custom Metal Euler mismatch for dt="+std::to_string(dt));}
 auto y=norm(mx::reshape(mx::arange(64,mx::float32),{4,16}));require(mx::max(mx::abs(mx::mean(y,-1))).item<float>()<1e-5,"normalization mismatch");
 *out=copy(json(@{@"passed":@YES,@"checks":@[@"three_model_recipes",@"dependency_rejection",@"flux_schedule",@"custom_metal_euler_bf16",@"layer_norm"],@"system":system_info()}));return 0;
 }catch(const std::exception&e){return fail(error,e);}catch(...){return 1;}}}

int tc_compile_coreml_json(const char*source,const char*cache,char**out,char**error){if(out)*out=nullptr;if(error)*error=nullptr;@autoreleasepool{try{tc::require(source&&cache&&out,"missing compile input/output");*out=copy(tc::json(tc::compile_artifact(source,cache)));return 0;}catch(const std::exception&e){return fail(error,e);}catch(...){if(error)*error=strdup("unknown compile error");return 1;}}}

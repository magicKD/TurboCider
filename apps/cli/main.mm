#import <Foundation/Foundation.h>
#include "turbocider/turbocider.h"
#include <iostream>
#include <csignal>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mach-o/dyld.h>
#include <limits.h>
#include <filesystem>
#include <vector>
#include <unistd.h>
#include "query_worker.hpp"
#include "startup_gate.hpp"
int tc_service_main(const char*,const char*,const char*);
int tc_rpc_main(const char*,const char*);
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
extern "C" int tc_engine_test_set_streaming_catalog_json(
    tc_engine*,const char*,char**);
#endif
static std::string executable_path(const char *fallback);
static NSString *registered_model_path(const char *alias,NSString *expected_model) {
 auto helper=std::filesystem::path(executable_path("turbocider")).parent_path()/"turbocider-library";
 NSTask *task=[NSTask new];task.executableURL=[NSURL fileURLWithPath:@(helper.c_str())];
 task.arguments=@[@"resolve",@(alias+1)];NSPipe *pipe=[NSPipe pipe];task.standardOutput=pipe;
 task.standardError=[NSFileHandle fileHandleWithStandardError];NSError *failure=nil;
 if(![task launchAndReturnError:&failure])return nil;
 NSData *data=[pipe.fileHandleForReading readDataToEndOfFile];[task waitUntilExit];
 if(task.terminationStatus!=0)return nil;
 id envelope=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
 if(![envelope isKindOfClass:NSDictionary.class])return nil;
 id result=envelope[@"result"];
 if(![result isKindOfClass:NSDictionary.class]||![result[@"modelID"] isEqual:expected_model]||![result[@"path"] isKindOfClass:NSString.class])return nil;
 return result[@"path"];
}
static int create_for(const char *path,NSString *request,tc_engine **engine,char **error) {
 NSDictionary *value=[NSJSONSerialization JSONObjectWithData:[request dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
 if(![value isKindOfClass:NSDictionary.class]){*error=strdup("request must be a JSON object");return 1;}
 id model=value[@"model"]?:@"flux2-klein-4b";
 if(![model isKindOfClass:NSString.class]){*error=strdup("model must be a string");return 1;}
 if(path&&path[0]=='@'){
  NSString *resolved=registered_model_path(path,model);
  if(!resolved){*error=strdup("cannot resolve registered model or alias does not match request.model; use turbocider library list");return 1;}
  return tc_engine_create_model_worker([model UTF8String],resolved.UTF8String,engine,error);
 }
 return tc_engine_create_model_worker([model UTF8String],path,engine,error);
}
static tc_engine *active=nullptr;
static bool resource_mode=false;
static volatile std::sig_atomic_t interrupted=0;
static void stop(int){interrupted=1;}
static void event(const char*s,void*){if(interrupted){if(resource_mode)tc_coreml_resources_cancel();else tc_engine_cancel(active);}std::cerr<<s<<std::endl;}
static std::string executable_path(const char *fallback) {
 uint32_t size=PATH_MAX;std::vector<char> value(size);
 if(_NSGetExecutablePath(value.data(),&size)!=0){value.resize(size);if(_NSGetExecutablePath(value.data(),&size)!=0)return fallback;}
 char resolved[PATH_MAX];return realpath(value.data(),resolved)?resolved:fallback;
}
static bool ltx_exec_finalizer_plan(NSString *request) {
 char *text=nullptr,*failure=nullptr;
 int status=tc_plan_json(request.UTF8String,&text,&failure);
 if(failure)tc_string_free(failure);
 if(status||!text)return false;
 NSData *data=[NSData dataWithBytes:text length:strlen(text)];
 tc_string_free(text);
 id value=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
 return [value isKindOfClass:NSDictionary.class]&&
        [value[@"model"] isEqual:@"ltx-2.5-distilled"]&&
        ([value[@"operation"] isEqual:@"video.generate"]||
         [value[@"operation"] isEqual:@"video.image"])&&
        [value[@"residency"] isEqual:@"component_staged"];
}
static void configure_ltx_cli_environment(NSString *request) {
 if(!ltx_exec_finalizer_plan(request))return;
 /* Keep the connected Gemma tensors across one-shot CLI invocations too.
  * The cache identity includes the model/checkpoint/tokenizer/prompt, so a
  * shared temporary root cannot mix incompatible conditioning. */
 setenv("TURBOCIDER_LTX_EXEC_FINALIZER","1",1);
 if(!std::getenv("TURBOCIDER_LTX_CONDITIONING_CACHE_DIR")){
  NSURL *base=[NSFileManager.defaultManager URLForDirectory:NSCachesDirectory
    inDomain:NSUserDomainMask appropriateForURL:nil create:YES error:nil];
  NSString *path=[[base URLByAppendingPathComponent:@"TurboCider"
    isDirectory:YES] URLByAppendingPathComponent:@"ltx-conditioning"
    isDirectory:YES].path;
  if(path.length)setenv("TURBOCIDER_LTX_CONDITIONING_CACHE_DIR",path.UTF8String,0);
 }
}
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
static int install_test_streaming_catalog(tc_engine *engine,char **error) {
 const char *path=std::getenv("TURBOCIDER_TEST_STREAMING_CATALOG");
 if(!path||!path[0])return 0;
 NSString *catalog=[NSString stringWithContentsOfFile:@(path)
   encoding:NSUTF8StringEncoding error:nil];
 if(!catalog){if(error)*error=strdup("cannot read test streaming catalog");return 1;}
 return tc_engine_test_set_streaming_catalog_json(
   engine,catalog.UTF8String,error);
}
#endif
static NSString *request_with_acceleration(NSString *request,const char *manifest_path,
                                           const char *hybrid_mode,std::string &failure) {
 if(!request){failure="cannot read request";return nil;}
 if(!manifest_path&&!hybrid_mode)return request;
 std::error_code ec;
 std::filesystem::path path;
 if(manifest_path){
  path=std::filesystem::absolute(manifest_path,ec);
  if(ec||!std::filesystem::is_regular_file(path,ec)||ec){
   failure="--ane-manifest must name an existing manifest JSON file";return nil;
  }
  path=path.lexically_normal();
 }
 NSError *error=nil;
 id raw=[NSJSONSerialization JSONObjectWithData:[request dataUsingEncoding:NSUTF8StringEncoding]
                                          options:NSJSONReadingMutableContainers error:&error];
 if(![raw isKindOfClass:NSMutableDictionary.class]){
  failure="request must be a JSON object";return nil;
 }
 NSMutableDictionary *value=raw;
 int version=[value[@"schema_version"] respondsToSelector:@selector(intValue)]
   ? [value[@"schema_version"] intValue] : 1;
 if(version!=1&&version!=2){failure="unsupported request schema_version";return nil;}
 id placement=version==1?value:value[@"execution"];
 if(version==2){
  if(placement&&![placement isKindOfClass:NSDictionary.class]){
   failure="execution must be an object for schema 2";return nil;
  }
  placement=placement?[placement mutableCopy]:[NSMutableDictionary dictionary];
 }
 if(![placement isKindOfClass:NSMutableDictionary.class]){
  failure="invalid ANE request execution";return nil;
 }
 NSMutableDictionary *execution=placement;
 NSString *key=version==1?@"execution":@"policy";
 if(manifest_path){
  id existing=execution[@"ane_manifest"];
  if(existing){
   if(![existing isKindOfClass:NSString.class]){
    failure="request ane_manifest must be a string";return nil;
   }
   std::error_code prior_error;
   auto prior=std::filesystem::absolute(std::string([(NSString *)existing UTF8String]),prior_error);
   if(prior_error||prior.lexically_normal()!=path){
    failure="--ane-manifest conflicts with request ane_manifest";return nil;
   }
  }
  execution[key]=@"gpu_ane";
  execution[@"ane_manifest"]=@(path.string().c_str());
 }
 if(hybrid_mode){
  std::string_view mode(hybrid_mode);
  if(mode!="auto"&&mode!="base_fused"&&mode!="lora_suffix"&&
     mode!="lora_gate_up"&&mode!="lora_fused"&&mode!="lora_merged"&&mode!="runtime"){
   failure="--hybrid-mode must be auto, base_fused, lora_suffix, lora_gate_up, lora_fused, lora_merged or runtime";return nil;
  }
  id existing=execution[@"hybrid_mlp_mode"];
  if(existing&&(![existing isKindOfClass:NSString.class]||
                ![existing isEqualToString:@(hybrid_mode)])){
   failure="--hybrid-mode conflicts with request hybrid_mlp_mode";return nil;
  }
  execution[@"hybrid_mlp_mode"]=@(hybrid_mode);
 }
 // Selecting an algorithm must not silently waive the caller's numerical
 // accuracy policy. The manifest override retains its original opt-in.
 if(manifest_path)execution[@"allow_approximation"]=@YES;
 if(version==2)value[@"execution"]=execution;
 NSData *json=[NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingSortedKeys error:&error];
 if(!json){failure="could not encode ANE request";return nil;}
 return [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding];
}
static NSString *cli_request(const char *file,const char *manifest_path,
                             const char *hybrid_mode,std::string &failure) {
 NSString *request=[NSString stringWithContentsOfFile:@(file) encoding:NSUTF8StringEncoding error:nil];
 return request_with_acceleration(request,manifest_path,hybrid_mode,failure);
}
static int library_main(int argc,char **argv) {
 auto helper=std::filesystem::path(executable_path(argv[0])).parent_path()/"turbocider-library";
 if(!std::filesystem::is_regular_file(helper)){
  std::cerr<<"Model-library helper is missing. Rebuild or restore the complete CLI bundle.\n";return 1;
 }
 NSTask *task=[NSTask new];task.executableURL=[NSURL fileURLWithPath:@(helper.c_str())];
 NSMutableArray<NSString*> *arguments=[NSMutableArray array];
 if(std::string(argv[1])=="cache")[arguments addObject:@"cache"];
 for(int index=2;index<argc;index++)[arguments addObject:@(argv[index])];
 if(argc==2)[arguments addObject:@"help"];
 task.arguments=arguments;
 task.standardOutput=[NSFileHandle fileHandleWithStandardOutput];
 task.standardError=[NSFileHandle fileHandleWithStandardError];
 NSError *error=nil;
 if(![task launchAndReturnError:&error]){std::cerr<<"Cannot launch model library: "<<error.localizedDescription.UTF8String<<"\n";return 1;}
 [task waitUntilExit];return task.terminationStatus;
}
int main(int argc,char**argv){@autoreleasepool{
 if(argc<2){std::cerr<<"turbocider library help | cache help | serve SOCKET STATE | rpc SOCKET REQUEST.json | worker-query INPUT.json | worker-generate INPUT.json | doctor|models|self-test|plan REQUEST.json [--ane-manifest MANIFEST.json] [--hybrid-mode MODE]|tokenize MODEL PROMPT|generate MODEL REQUEST.json [--ane-manifest MANIFEST.json] [--hybrid-mode MODE] | batch MODEL REQUEST1.json REQUEST2.json ... [--ane-manifest MANIFEST.json] [--hybrid-mode MODE] | prepare-lora MODEL BASE LORA OUTPUT [options]\n";return 1;}
 std::string cmd=argv[1];char*out=nullptr,*err=nullptr;int code=0;
 const bool allows_ane=cmd=="plan"||cmd=="generate"||cmd=="batch";
 const char *ane_manifest=nullptr,*hybrid_mode=nullptr;
 int request_argc=argc;
 if(allows_ane){
  while(request_argc>=4){
   const std::string flag=argv[request_argc-2];
   if(flag=="--ane-manifest"&&!ane_manifest)ane_manifest=argv[request_argc-1];
   else if(flag=="--hybrid-mode"&&!hybrid_mode)hybrid_mode=argv[request_argc-1];
   else break;
   request_argc-=2;
  }
 }
 if(allows_ane){
  for(int i=2;i<request_argc;++i)if(std::string(argv[i])=="--ane-manifest"||
                                      std::string(argv[i])=="--hybrid-mode"){
   std::cerr<<"--ane-manifest and --hybrid-mode must be trailing argument pairs\n";return 1;
  }
 }
 if((cmd=="worker-query" || cmd=="worker-generate") &&
    (argc==3 || (argc==4 && std::string(argv[3])=="--supervised"))) {
   if(argc==4 && !tc_worker::await_admission()) {std::cerr<<"worker_start_not_admitted\n";return 1;}
   return cmd=="worker-query"?tc_worker::query(argv[2]):tc_worker::generate(argv[2]);
 }
 if(cmd=="library"||cmd=="cache")return library_main(argc,argv);
 if(cmd=="prepare-lora"){std::cerr<<"LoRA preparation is offline-only; run python3 tools/native/prepare_lora.py MODEL BASE LORA OUTPUT [options] in the development environment\n";return 1;}
 if(cmd=="serve"&&argc==4){auto executable=executable_path(argv[0]);return tc_service_main(argv[2],argv[3],executable.c_str());}
 if(cmd=="rpc"&&argc==4)return tc_rpc_main(argv[2],argv[3]);
 if(cmd=="coreml"&&argc==3){
  NSString*request=[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil];
  if(!request){std::cerr<<"cannot read resource request\n";return 1;}
  id resource=[NSJSONSerialization JSONObjectWithData:[request dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
  bool inspection=[resource isKindOfClass:NSDictionary.class] && [resource[@"action"] isEqual:@"inventory"];
  // A helper must also stop if its UI parent exits during a blocked file-provider read.
  if(inspection)alarm(6);
  resource_mode=true;std::signal(SIGINT,stop);code=tc_coreml_resources_json(request.UTF8String,event,nullptr,&out,&err);std::signal(SIGINT,SIG_DFL);if(inspection)alarm(0);
 }
 else if(cmd=="doctor")out=tc_system_json();
 else if(cmd=="models")out=tc_models_json();
 else if(cmd=="self-test")code=tc_native_self_test(&out,&err);
 else if(cmd=="compile-coreml"&&argc==4)code=tc_compile_coreml_json(argv[2],argv[3],&out,&err);
 else if(cmd=="tokenize"&&argc==4)code=tc_tokenize_json(argv[2],argv[3],&out,&err);
 else if(cmd=="batch"&&request_argc>=4){
  std::string failure;
  NSString *first=cli_request(argv[3],ane_manifest,hybrid_mode,failure);
  if(!first){std::cerr<<failure<<"\n";return 1;}
  code=create_for(argv[2],first,&active,&err);
  if(!code){std::signal(SIGINT,stop);for(int i=3;i<request_argc;++i){
    NSString*request=i==3?first:cli_request(argv[i],ane_manifest,hybrid_mode,failure);
    if(!request){std::cerr<<failure<<"\n";code=1;break;}
    code=tc_engine_generate(active,request.UTF8String,event,nullptr,&out,&err);
    if(out){std::cout<<out<<std::endl;tc_string_free(out);out=nullptr;}
    if(code)break;
  }std::signal(SIGINT,SIG_DFL);}tc_engine_free(active);
 }
 else if((cmd=="plan"&&request_argc==3)||(cmd=="generate"&&request_argc==4)||
         (cmd=="ltx-worker"&&argc==4)){
  std::string failure;
  NSString*request=cli_request(argv[request_argc-1],ane_manifest,hybrid_mode,failure);
  if(!request){std::cerr<<failure<<"\n";return 1;}
  if(cmd=="plan")code=tc_plan_json(request.UTF8String,&out,&err);
  else {
   if(cmd=="generate"||cmd=="ltx-worker")
    configure_ltx_cli_environment(request);
   code=create_for(argv[2],request,&active,&err);
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
   if(!code)code=install_test_streaming_catalog(active,&err);
#endif
   if(!code){std::signal(SIGINT,stop);std::signal(SIGTERM,stop);code=tc_engine_generate(active,request.UTF8String,event,nullptr,&out,&err);std::signal(SIGINT,SIG_DFL);std::signal(SIGTERM,SIG_DFL);}
   tc_engine_free(active);
  }
 }else{std::cerr<<"invalid command or arguments\n";return 1;}
 if(out){std::cout<<out<<std::endl;tc_string_free(out);}if(err){std::cerr<<err<<std::endl;tc_string_free(err);}return code;
}}

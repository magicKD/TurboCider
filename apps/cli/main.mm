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
int tc_service_main(const char*,const char*,const char*);
int tc_rpc_main(const char*,const char*);
static int create_for(const char *path,NSString *request,tc_engine **engine,char **error) {
 NSDictionary *value=[NSJSONSerialization JSONObjectWithData:[request dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
 if(![value isKindOfClass:NSDictionary.class]){*error=strdup("request must be a JSON object");return 1;}
 id model=value[@"model"]?:@"flux2-klein-4b";
 if(![model isKindOfClass:NSString.class]){*error=strdup("model must be a string");return 1;}
 return tc_engine_create_model([model UTF8String],path,engine,error);
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
        [value[@"operation"] isEqual:@"video.generate"]&&
        [value[@"residency"] isEqual:@"component_staged"]&&
        ![value[@"audio"] boolValue];
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
int main(int argc,char**argv){@autoreleasepool{
 if(argc<2){std::cerr<<"turbocider coreml REQUEST.json | doctor|models|self-test|plan REQUEST.json|tokenize MODEL PROMPT|generate MODEL REQUEST.json | batch MODEL REQUEST1.json REQUEST2.json ...\n";return 1;}
 std::string cmd=argv[1];char*out=nullptr,*err=nullptr;int code=0;
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
 else if(cmd=="batch"&&argc>=4){
  code=create_for(argv[2],[NSString stringWithContentsOfFile:@(argv[3]) encoding:NSUTF8StringEncoding error:nil],&active,&err);
  if(!code){std::signal(SIGINT,stop);for(int i=3;i<argc;++i){
    NSString*request=[NSString stringWithContentsOfFile:@(argv[i]) encoding:NSUTF8StringEncoding error:nil];
    if(!request){code=1;break;}
    code=tc_engine_generate(active,request.UTF8String,event,nullptr,&out,&err);
    if(out){std::cout<<out<<std::endl;tc_string_free(out);out=nullptr;}
    if(code)break;
  }std::signal(SIGINT,SIG_DFL);}tc_engine_free(active);
 }
 else if((cmd=="plan"&&argc==3)||(cmd=="generate"&&argc==4)||
         (cmd=="ltx-worker"&&argc==4)){
  NSString*request=[NSString stringWithContentsOfFile:@(argv[argc-1]) encoding:NSUTF8StringEncoding error:nil];
  if(!request){std::cerr<<"cannot read request\n";return 1;}
  if(cmd=="plan")code=tc_plan_json(request.UTF8String,&out,&err);
  else {
   if(cmd=="generate"||cmd=="ltx-worker")
    configure_ltx_cli_environment(request);
   code=create_for(argv[2],[NSString stringWithContentsOfFile:@(argv[3]) encoding:NSUTF8StringEncoding error:nil],&active,&err);
   if(!code){std::signal(SIGINT,stop);std::signal(SIGTERM,stop);code=tc_engine_generate(active,request.UTF8String,event,nullptr,&out,&err);std::signal(SIGINT,SIG_DFL);std::signal(SIGTERM,SIG_DFL);}
   tc_engine_free(active);
  }
 }else{std::cerr<<"invalid command or arguments\n";return 1;}
 if(out){std::cout<<out<<std::endl;tc_string_free(out);}if(err){std::cerr<<err<<std::endl;tc_string_free(err);}return code;
}}

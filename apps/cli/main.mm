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
  return tc_engine_create_model([model UTF8String],resolved.UTF8String,engine,error);
 }
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
static std::filesystem::path lora_prepare_script(const char *executable) {
 std::error_code error;
 auto executable_path_value=std::filesystem::absolute(executable,error);
 if(!error){
  auto adjacent=executable_path_value.parent_path()/"prepare_lora.py";
  if(std::filesystem::is_regular_file(adjacent))return adjacent;
  auto root=executable_path_value.parent_path().parent_path().parent_path();
  auto source=root/"tools/native/prepare_lora.py";
  if(std::filesystem::is_regular_file(source))return source;
 }
 return {};
}
static int prepare_lora_main(int argc,char **argv,const char *executable) {
 if(argc<6){
  std::cerr<<"usage: turbocider prepare-lora MODEL BASE LORA OUTPUT [options]\n";
  return 1;
 }
 auto script=lora_prepare_script(executable);
 if(script.empty()){
  std::cerr<<"cannot locate packaged tools/native/prepare_lora.py\n";
  return 1;
 }
 NSTask *task=[NSTask new];
 NSMutableArray<NSString*> *arguments=[NSMutableArray array];
 if(const char *configured=std::getenv("TURBOCIDER_PREPARE_PYTHON")){
  task.launchPath=@(configured);
 }else{
  auto source_root=script.parent_path().parent_path().parent_path();
  auto bundled=source_root/"Python/bin/python";
  if(std::filesystem::is_regular_file(bundled))task.launchPath=@(bundled.c_str());
  else {task.launchPath=@"/usr/bin/env";[arguments addObject:@"python3"];}
 }
 [arguments addObject:@(script.c_str())];
 for(int index=2;index<argc;index++) [arguments addObject:@(argv[index])];
 task.arguments=arguments;
 task.standardOutput=[NSFileHandle fileHandleWithStandardOutput];
 task.standardError=[NSFileHandle fileHandleWithStandardError];
 @try {[task launch];[task waitUntilExit];return task.terminationStatus;}
 @catch(NSException *exception){
  std::cerr<<"cannot launch LoRA preparation tool: "<<exception.reason.UTF8String<<"\n";
  return 1;
 }
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
 if(argc<2){std::cerr<<"turbocider library help | cache help | serve SOCKET STATE | rpc SOCKET REQUEST.json | coreml REQUEST.json | doctor|models|self-test|plan REQUEST.json|tokenize MODEL PROMPT|generate MODEL REQUEST.json | batch MODEL REQUEST1.json REQUEST2.json ... | prepare-lora MODEL BASE LORA OUTPUT [options]\n";return 1;}
 std::string cmd=argv[1];char*out=nullptr,*err=nullptr;int code=0;
 if(cmd=="library"||cmd=="cache")return library_main(argc,argv);
 if(cmd=="prepare-lora")return prepare_lora_main(argc,argv,argv[0]);
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

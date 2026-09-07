#import <Foundation/Foundation.h>
#include "turbocider/turbocider.h"
#include <iostream>
#include <csignal>
#include <atomic>
#include <unistd.h>
int tc_service_main(const char*,const char*);
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
int main(int argc,char**argv){@autoreleasepool{
 if(argc<2){std::cerr<<"turbocider coreml REQUEST.json | doctor|models|self-test|plan REQUEST.json|tokenize MODEL PROMPT|generate MODEL REQUEST.json | batch MODEL REQUEST1.json REQUEST2.json ...\n";return 1;}
 std::string cmd=argv[1];char*out=nullptr,*err=nullptr;int code=0;
 if(cmd=="serve"&&argc==4)return tc_service_main(argv[2],argv[3]);
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
 else if((cmd=="plan"&&argc==3)||(cmd=="generate"&&argc==4)){
  NSString*request=[NSString stringWithContentsOfFile:@(argv[argc-1]) encoding:NSUTF8StringEncoding error:nil];
  if(!request){std::cerr<<"cannot read request\n";return 1;}
  if(cmd=="plan")code=tc_plan_json(request.UTF8String,&out,&err);
  else {code=create_for(argv[2],[NSString stringWithContentsOfFile:@(argv[3]) encoding:NSUTF8StringEncoding error:nil],&active,&err);if(!code){std::signal(SIGINT,stop);code=tc_engine_generate(active,request.UTF8String,event,nullptr,&out,&err);std::signal(SIGINT,SIG_DFL);}tc_engine_free(active);}
 }else{std::cerr<<"invalid command or arguments\n";return 1;}
 if(out){std::cout<<out<<std::endl;tc_string_free(out);}if(err){std::cerr<<err<<std::endl;tc_string_free(err);}return code;
}}

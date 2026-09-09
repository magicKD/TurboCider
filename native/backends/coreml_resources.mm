#include "../platform/apple/bridge.hpp"
#include "../platform/apple/platform.hpp"
#import <CommonCrypto/CommonDigest.h>
#include <sys/stat.h>
namespace tc {
namespace fs=std::filesystem;
static fs::path support(){return fs::path(NSHomeDirectory().UTF8String)/"Library/Application Support/TurboCiderNative";}
static std::string digest(const std::string&s){unsigned char bytes[CC_SHA256_DIGEST_LENGTH];CC_SHA256(s.data(),CC_LONG(s.size()),bytes);std::string out;for(auto b:bytes){char pair[3];snprintf(pair,3,"%02x",b);out+=pair;}return out;}
struct Usage {uint64_t bytes=0,allocated=0,files=0;std::string stamp;};
static Usage usage(const fs::path&root,bool strict=false){
 Usage result;std::error_code error;
 bool exists=fs::exists(root,error);require(!error,"cannot access resource: "+root.string());
 if(!exists){require(!strict,"resource missing: "+root.string());return result;}
 require(!fs::is_symlink(root),"resource root must not be a symlink");
 auto add=[&](const fs::path&p){struct stat st;require(lstat(p.c_str(),&st)==0,"cannot inspect resource: "+p.string());require(!S_ISLNK(st.st_mode),"resource contains a symlink");if(S_ISREG(st.st_mode)){if(strict)require(p.extension()!=".safetensors","refusing to delete source weights");result.bytes+=st.st_size;result.allocated+=st.st_blocks*512;result.files++;result.stamp+=p.string()+":"+std::to_string(st.st_size)+":"+std::to_string(st.st_mtimespec.tv_sec)+":"+std::to_string(st.st_mtimespec.tv_nsec)+"\n";}};
 if(fs::is_directory(root)){std::vector<fs::path> paths;for(auto&item:fs::recursive_directory_iterator(root))paths.push_back(item.path());std::sort(paths.begin(),paths.end());for(auto&p:paths)add(p);}else add(root);
 return result;
}
static std::vector<fs::path> runtime_paths(){std::vector<fs::path> paths;for(auto name:{"org.turbocider.native","TurboCiderNativeApp","turbocider","turbocider-native"})paths.push_back(fs::path(NSHomeDirectory().UTF8String)/"Library/Caches"/name/"com.apple.e5rt.e5bundlecache");return paths;}
static std::vector<fs::path> artifacts(const fs::path&file,const std::string&kind){
 auto d=read_json(file);require([d[@"artifacts"] isKindOfClass:NSDictionary.class],"manifest artifacts missing");
 auto parent=fs::weakly_canonical(file.parent_path());std::vector<fs::path> paths;
 int count=int([d[@"artifacts"] count]);require(count>0&&count<=64,"manifest must contain 1...64 partitions");
 for(int i=0;i<count;++i){auto name=string_value(d[@"artifacts"][[NSString stringWithFormat:@"%d",i]],@"int8_pc");require(!name.empty(),"missing contiguous partition");auto raw=(parent/name).lexically_normal();auto path=fs::weakly_canonical(raw);require(path==raw&&!fs::is_symlink(path)&&path.string().starts_with(parent.string()+"/"),"partition path escapes manifest or contains symlink");require(kind=="source"?path.extension()==".mlpackage":path.extension()==".mlmodelc","partition kind does not match manifest");paths.push_back(path);}
 std::sort(paths.begin(),paths.end());paths.erase(std::unique(paths.begin(),paths.end()),paths.end());return paths;
}
NSDictionary *coreml_resources(NSDictionary*request,const Event&event,std::atomic<bool>&cancelled){
 for(NSString*k in request)require([@[@"action",@"cache",@"manifest",@"source_manifest",@"storage",@"profile",@"model",@"model_root",@"kind",@"apply",@"plan_token"] containsObject:k],"unknown Core ML resource request field");
 auto action=string_value(request,@"action");
 require(action!="export", "Core ML export is offline-only; run tools/coreml/export_flux2.py or export_z_image.py, then import and compile the resulting manifest");
 require(action=="inventory"||action=="delete_artifacts"||action=="clear_runtime"||action=="clear_compiled"||action=="compile","unknown Core ML resource action");
 auto absolute=[](const std::string&s){return fs::absolute(s).lexically_normal();};
 auto model_id=string_value(request,@"model","flux2-klein-4b");
 require(model_id=="flux2-klein-4b"||model_id=="z-image-turbo"||
             model_id=="z-image-turbo-gguf",
         "Core ML resource model must be flux2-klein-4b, z-image-turbo or z-image-turbo-gguf");
 bool z_image=model_id=="z-image-turbo"||model_id=="z-image-turbo-gguf";
 int bucket=z_image?4128:1088,ane_mlp_limit=z_image?10239:9216;
 fs::path storage=support()/("coreml/"+model_id+"/m"+std::to_string(bucket)),cache=support()/"cache/coreml";
 std::string manifest=string_value(request,@"manifest"),source=string_value(request,@"source_manifest");
 auto profile=string_value(request,@"profile");
 if(!profile.empty()){
  auto file=absolute(profile);auto d=read_json(file);require([d[@"schema_version"] isKindOfClass:NSNumber.class]&&[d[@"schema_version"] intValue]==1,"profile schema must be 1");require([d[@"models"] isKindOfClass:NSDictionary.class],"profile models missing");NSDictionary *model=d[@"models"][@(model_id.c_str())];require([model isKindOfClass:NSDictionary.class],"requested model profile missing");
  if(manifest.empty()){auto p=string_value(model,@"ane_manifest");if(!p.empty())manifest=(file.parent_path()/p).lexically_normal().string();}
  id config=model[@"coreml_export"];
  if(config){require([config isKindOfClass:NSDictionary.class],"coreml_export must be object");for(NSString*k in config)require([@[@"bucket",@"ane_mlp_width",@"variant",@"output_dir",@"cache_dir",@"python",@"python_path"] containsObject:k],"unknown coreml_export field");require([config[@"bucket"] isKindOfClass:NSNumber.class],"export bucket required");double n=[config[@"bucket"] doubleValue];require(n==int(n)&&n>=64&&n<=8192,"invalid fixed bucket");bucket=int(n);if(config[@"ane_mlp_width"]){require([config[@"ane_mlp_width"] isKindOfClass:NSNumber.class],"ane_mlp_width must be numeric");double width=[config[@"ane_mlp_width"] doubleValue];require(width==int(width)&&width>0&&width<=ane_mlp_limit,"invalid ANE MLP width");}storage=support()/("coreml/"+model_id+"/m"+std::to_string(bucket));require(string_value(config,@"variant","int8_pc")=="int8_pc","native hybrid supports int8_pc only");auto output=string_value(config,@"output_dir");if(!output.empty())storage=absolute((file.parent_path()/output).string());auto c=string_value(config,@"cache_dir");if(!c.empty())cache=absolute((file.parent_path()/c).string());}
 }
 if(!string_value(request,@"storage").empty())storage=absolute(string_value(request,@"storage"));
 if(!string_value(request,@"cache").empty())cache=absolute(string_value(request,@"cache"));
 if(source.empty()&&!manifest.empty()&&fs::is_regular_file(manifest)){auto d=read_json(manifest);source=string_value(d,@"source_manifest");}
 if(source.empty()&&fs::is_regular_file(storage/"manifest.json"))source=(storage/"manifest.json").string();
 if(action=="compile")return manage_coreml_cache(@{@"action":@"compile_manifest",@"cache":@(cache.c_str()),@"source":@(source.c_str())},event,cancelled);
 if(action=="inventory"){
  NSMutableArray*rows=[NSMutableArray array];std::vector<fs::path> counted;uint64_t total=0,allocated=0;
  auto add=[&](fs::path path,const char*category,bool removable){path=absolute(path.string());for(auto&p:counted)if(path==p||path.string().starts_with(p.string()+"/"))return;try{auto u=usage(path);[rows addObject:@{@"path":@(path.c_str()),@"category":@(category),@"bytes":@(u.bytes),@"allocated_bytes":@(u.allocated),@"files":@(u.files),@"exists":@(fs::exists(path)),@"removable":@(removable)}];counted.push_back(path);total+=u.bytes;allocated+=u.allocated;}catch(const std::exception&e){[rows addObject:@{@"path":@(path.c_str()),@"category":@(category),@"error":@(e.what()),@"removable":@NO}];}};
  add(cache,"managed_compiled",true);
  for(auto&pair:std::vector<std::pair<std::string,std::string>>{{manifest,"compiled"},{source,"source"}})if(!pair.first.empty())try{for(auto&p:artifacts(absolute(pair.first),pair.second))add(p,pair.second.c_str(),true);}catch(const std::exception&e){[rows addObject:@{@"path":@(pair.first.c_str()),@"category":@(pair.second.c_str()),@"error":@(e.what()),@"removable":@NO}];}
  for(auto&p:runtime_paths())add(p,"runtime_specialization",true);
  return @{@"action":@"inventory",@"entries":rows,@"bytes":@(total),@"allocated_bytes":@(allocated),@"cache":@(cache.c_str()),@"storage":@(storage.c_str()),@"manifest":@(manifest.c_str()),@"source_manifest":@(source.c_str()),@"scope":@"selected Core ML artifacts and TurboCider process caches; excludes other apps and global OS caches; APFS clones may share physical blocks"};
 }
 bool apply=false;if(request[@"apply"]){require(CFGetTypeID((__bridge CFTypeRef)request[@"apply"])==CFBooleanGetTypeID(),"apply must be boolean");apply=[request[@"apply"] boolValue];}
 std::vector<fs::path> paths;
 if(action=="clear_runtime"){for(auto&p:runtime_paths())if(fs::exists(p))paths.push_back(p);}
 else if(action=="clear_compiled"){
  if(fs::exists(cache)){
   auto marker=cache/".turbocider-cache.json";
   if(fs::is_empty(cache)){}else require(fs::is_regular_file(marker)&&!fs::is_symlink(marker)&&[read_json(marker)[@"owner"] isEqual:@"turbocider.coreml.v1"],"not a managed compilation cache");
   auto key=[](std::string s){return s.size()==64&&s.find_first_not_of("0123456789abcdef")==std::string::npos;};
   for(auto&entry:fs::directory_iterator(cache)){
    auto name=entry.path().filename().string();require(!entry.is_symlink(),"cache contains symlink");
    if(key(name)&&entry.is_directory()&&fs::is_regular_file(entry.path()/"identity.json"))paths.push_back(entry.path());
    if(entry.is_regular_file()&&name.starts_with("manifest-")&&name.ends_with(".json")&&key(name.substr(9,name.size()-14)))paths.push_back(entry.path());
   }
  }
 }
 else {auto kind=string_value(request,@"kind");require(kind=="source"||kind=="compiled","delete kind must be source or compiled");auto file=absolute(kind=="source"?source:manifest);paths=artifacts(file,kind);paths.push_back(file);}
 std::sort(paths.begin(),paths.end());
 NSMutableArray*entries=[NSMutableArray array];uint64_t bytes=0;std::string identity;
 for(auto&p:paths){checkpoint(cancelled);auto u=usage(p,true);bytes+=u.bytes;identity+=p.string()+u.stamp;[entries addObject:@{@"path":@(p.c_str()),@"bytes":@(u.bytes)}];}
 auto token=digest(action+identity);if(apply){require(string_value(request,@"plan_token")==token,"cleanup plan changed; inspect a fresh deletion preview");if(action=="clear_compiled")manage_coreml_cache(@{@"action":@"clear",@"cache":@(cache.c_str())},event,cancelled);else for(auto&p:paths)fs::remove_all(p);}
 return @{@"action":@(action.c_str()),@"applied":@(apply),@"entries":entries,@"bytes":@(bytes),@"plan_token":@(token.c_str()),@"scope":@"explicit listed artifacts only; source safetensors are never deleted"};
}
}

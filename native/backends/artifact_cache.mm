#include "runtime.hpp"
#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <fstream>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/sysctl.h>
namespace tc {
static std::string hex_digest(const unsigned char *value,size_t count) {
    std::string result;for(size_t i=0;i<count;++i){char pair[3];snprintf(pair,sizeof(pair),"%02x",value[i]);result+=pair;}return result;
}
static std::string tree_digest(const std::filesystem::path& root) {
    CC_SHA256_CTX context;CC_SHA256_Init(&context);
    std::vector<std::filesystem::path> files;
    if(std::filesystem::is_regular_file(root))files.push_back(root);
    else for(auto& entry:std::filesystem::recursive_directory_iterator(root)) {
        require(!entry.is_symlink(),"compiled artifact source must not contain symlinks");
        if(entry.is_regular_file())files.push_back(entry.path());
    }
    std::sort(files.begin(),files.end());require(!files.empty(),"empty artifact source");
    std::vector<char> buffer(1<<20);
    for(auto& file:files){
        auto name=std::filesystem::relative(file,root).generic_string();uint64_t length=name.size(),bytes=std::filesystem::file_size(file);
        CC_SHA256_Update(&context,&length,sizeof(length));CC_SHA256_Update(&context,name.data(),CC_LONG(name.size()));CC_SHA256_Update(&context,&bytes,sizeof(bytes));
        std::ifstream stream(file,std::ios::binary);require(bool(stream),"cannot read artifact: "+file.string());
        uint64_t read=0;while(stream){stream.read(buffer.data(),buffer.size());auto n=stream.gcount();if(n){CC_SHA256_Update(&context,buffer.data(),CC_LONG(n));read+=n;}}
        require(read==bytes,"artifact changed while hashing");
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(digest,&context);return hex_digest(digest,sizeof(digest));
}
NSDictionary *compile_artifact(const std::filesystem::path& source,const std::filesystem::path& cache) {
    require(source.extension()==".mlpackage"||source.extension()==".mlmodel","source must be .mlpackage or .mlmodel");
    require(std::filesystem::exists(source),"Core ML source missing");
    auto begin=Clock::now();auto hash=tree_digest(source);char build[128]={};size_t length=sizeof(build);sysctlbyname("kern.osversion",build,&length,nullptr,0);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    auto identity=@{@"schema_version":@1,@"source_sha256":@(hash.c_str()),@"os_build":@(build),@"gpu":device.name?:@"unknown",@"architecture":@"arm64",@"compiler":@"CoreML.compileModel",@"abi":@1};
    auto encoded=json(identity);unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(encoded.data(),CC_LONG(encoded.size()),digest);auto key=hex_digest(digest,sizeof(digest));
    std::filesystem::create_directories(cache);auto lock_path=cache/(key+".lock");
    int fd=open(lock_path.c_str(),O_CREAT|O_RDWR|O_CLOEXEC,0600);require(fd>=0,"cannot open compile lock");
    struct Lock {int fd;~Lock(){flock(fd,LOCK_UN);close(fd);}} lock{fd};require(flock(fd,LOCK_EX)==0,"cannot acquire compile lock");
    auto directory=cache/key;auto artifact=directory/"model.mlmodelc";bool hit=std::filesystem::is_directory(artifact)&&std::filesystem::is_regular_file(directory/"identity.json");
    if(hit)require([read_json(directory/"identity.json") isEqual:identity],"cache identity mismatch");
    if(!hit){
        NSError *error=nil;NSURL *compiled=[MLModel compileModelAtURL:[NSURL fileURLWithPath:@(source.c_str())] error:&error];
        require(compiled!=nil,"Core ML compile failed: "+std::string(error?error.localizedDescription.UTF8String:"unknown"));
        auto temporary=cache/(key+"."+std::string(NSUUID.UUID.UUIDString.UTF8String)+".partial");
        struct Cleanup {std::filesystem::path path;~Cleanup(){std::error_code ignored;std::filesystem::remove_all(path,ignored);}} cleanup{temporary},compiled_cleanup{compiled.path.UTF8String};
        require(tree_digest(source)==hash,"artifact source changed during compilation");
        std::filesystem::create_directory(temporary);
        std::filesystem::copy(compiled.path.UTF8String,temporary/"model.mlmodelc",std::filesystem::copy_options::recursive);
        auto text=json(identity);NSData *data=[NSData dataWithBytes:text.data() length:text.size()];
        require([data writeToFile:@((temporary/"identity.json").c_str()) options:NSDataWritingAtomic error:&error],"cannot commit compile identity");
        // A key is immutable; incomplete entries can only originate from an
        // interrupted writer holding this same lock.
        if(std::filesystem::exists(directory))std::filesystem::remove_all(directory);
        std::filesystem::rename(temporary,directory);
    }
    return @{@"schema_version":@1,@"key":@(key.c_str()),@"artifact":@(std::filesystem::absolute(artifact).c_str()),@"cache_hit":@(hit),@"identity":identity,@"seconds":@(std::chrono::duration<double>(Clock::now()-begin).count())};
}
}

namespace tc {
NSDictionary *manage_coreml_cache(NSDictionary*request,const Event&event,std::atomic<bool>&cancelled){
 for(NSString*key in request)require([@[@"action",@"cache",@"source"] containsObject:key],"unknown cache request field");
 auto action=string_value(request,@"action"),root_text=string_value(request,@"cache");
 require(action=="inspect"||action=="compile_manifest"||action=="clear","unknown cache action");
 require(!root_text.empty(),"cache directory required");auto root=std::filesystem::absolute(root_text).lexically_normal();
 require(!std::filesystem::is_symlink(root)&&root!=root.root_path(),"invalid managed cache root");
 std::filesystem::create_directories(root);
 int fd=open((root/".management.lock").c_str(),O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600);require(fd>=0,"cannot open cache management lock");
 struct Guard {int fd;~Guard(){flock(fd,LOCK_UN);close(fd);}} guard{fd};require(flock(fd,LOCK_EX|LOCK_NB)==0,"cache maintenance already running");
 auto marker=root/".turbocider-cache.json";
 if(!std::filesystem::exists(marker)){
  if(action=="clear")for(auto& entry:std::filesystem::directory_iterator(root))require(entry.path().filename()==".management.lock","directory is not a TurboCider managed cache");
  auto data=[@"{\"owner\":\"turbocider.coreml.v1\"}" dataUsingEncoding:NSUTF8StringEncoding];
  require([data writeToFile:@(marker.c_str()) options:NSDataWritingAtomic error:nil],"cannot initialize cache identity");
 }
 require(!std::filesystem::is_symlink(marker)&&[read_json(marker)[@"owner"] isEqual:@"turbocider.coreml.v1"],"wrong cache ownership");
 auto is_key=[](const std::string&s){return s.size()==64&&s.find_first_not_of("0123456789abcdef")==std::string::npos;};
 NSMutableArray *entries=[NSMutableArray array];uint64_t bytes=0;int removed=0;
 if(action=="compile_manifest"){
  auto source=std::filesystem::absolute(string_value(request,@"source"));auto d=read_json(source);
  require([d[@"shape"] isKindOfClass:NSDictionary.class]&&[d[@"artifacts"] isKindOfClass:NSDictionary.class]&&[d[@"source"] isKindOfClass:NSDictionary.class],"invalid source partition manifest");
  auto artifacts=[NSMutableDictionary dictionary];int hits=0;
  // Validate all input paths before compiling any partition.
  std::vector<std::filesystem::path> paths;auto base=std::filesystem::weakly_canonical(source.parent_path());
  for(int i=0;i<20;++i){NSString*k=[NSString stringWithFormat:@"%d",i];auto entry=d[@"artifacts"][k];require([entry isKindOfClass:NSDictionary.class],"missing partition");auto path=std::filesystem::weakly_canonical(base/string_value(entry,@"int8_pc"));require(path.string().starts_with(base.string()+"/")&&(path.extension()==".mlpackage"||path.extension()==".mlmodel"),"select the source manifest containing .mlpackage partitions");paths.push_back(path);}
  for(int i=0;i<20;++i){checkpoint(cancelled);event("coreml_compile",i,20);auto result=compile_artifact(paths[i],root);if([result[@"cache_hit"] boolValue])++hits;NSString*k=[NSString stringWithFormat:@"%d",i];artifacts[k]=@{@"int8_pc":[NSString stringWithFormat:@"%@/model.mlmodelc",result[@"key"]]};}
  checkpoint(cancelled);NSMutableDictionary *compiled=[d mutableCopy];compiled[@"schema_version"]=@2;compiled[@"artifacts"]=artifacts;compiled[@"source_manifest"]=@(source.c_str());
  auto content=json(compiled);unsigned char hash[CC_SHA256_DIGEST_LENGTH];CC_SHA256(content.data(),CC_LONG(content.size()),hash);
  auto target=root/("manifest-"+hex_digest(hash,sizeof(hash))+".json");
  require([[NSData dataWithBytes:content.data() length:content.size()] writeToFile:@(target.c_str()) options:NSDataWritingAtomic error:nil],"cannot publish compiled manifest");
  event("coreml_compile",20,20);
  return @{@"action":@"compile_manifest",@"manifest":@(target.c_str()),@"partitions":@20,@"cache_hits":@(hits),@"cache":@(root.c_str())};
 }
 for(auto& entry:std::filesystem::directory_iterator(root)){
  auto name=entry.path().filename().string();if(entry.is_symlink())continue;
  if(is_key(name)&&entry.is_directory()&&std::filesystem::is_regular_file(entry.path()/"identity.json")){
   auto identity=read_json(entry.path()/"identity.json");require([identity[@"compiler"] isEqual:@"CoreML.compileModel"],"unrecognized cache entry");
   uint64_t size=0;for(auto& file:std::filesystem::recursive_directory_iterator(entry.path())){require(!file.is_symlink(),"cache entry contains symlink");if(file.is_regular_file())size+=file.file_size();}
   bytes+=size;[entries addObject:@{@"key":@(name.c_str()),@"bytes":@(size)}];
   if(action=="clear"){
    checkpoint(cancelled);int lockfd=open((root/(name+".lock")).c_str(),O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600);require(lockfd>=0,"cannot lock cache entry");Guard keyguard{lockfd};require(flock(lockfd,LOCK_EX|LOCK_NB)==0,"compiled cache entry currently in use by compiler");
    std::filesystem::remove_all(entry.path());++removed;
   }
  }else if(action=="clear"&&entry.is_regular_file()&&name.starts_with("manifest-")&&name.ends_with(".json")&&is_key(name.substr(9,name.size()-14))){std::filesystem::remove(entry.path());}
 }
 return @{@"action":@(action.c_str()),@"entries":entries,@"bytes":@(bytes),@"removed_entries":@(removed),@"cache":@(root.c_str()),@"scope":@"TurboCider-managed artifacts only; source weights and system Core ML caches are untouched"};
}
}

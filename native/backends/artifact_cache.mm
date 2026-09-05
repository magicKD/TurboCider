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

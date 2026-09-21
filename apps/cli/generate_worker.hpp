#pragma once
#include "worker_protocol.hpp"
#include <array>
#include <algorithm>
#include <atomic>
#include <limits.h>

namespace tc_worker {
inline NSDictionary *object(id value) {
    require([value isKindOfClass:NSDictionary.class],"worker_native_result_invalid");return value;
}
inline bool equal(id left,id right) {return left && right && left!=NSNull.null && [left isEqual:right];}
inline bool text_value(id value) {return [value isKindOfClass:NSString.class] && [value length]>0;}
inline bool boolean(id value,bool expected) {
    return [value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value)==CFBooleanGetTypeID() && [value boolValue]==expected;
}
// This freezes native's exact selector, never reconstructs authority. The C API
// revalidates catalog/source and verifies the actual execution receipt itself.
inline NSDictionary *bind_generation(NSDictionary *request,NSDictionary *resolution) {
    NSDictionary *exact=object(resolution[@"exact_selector"]),*selection=object(resolution[@"selection"]);
    require(integer(resolution[@"schema_version"],1) && integer(exact[@"schema_version"],2) &&
        boolean(exact[@"enabled"],true) && equal(exact[@"selection"],@"preset") && equal(exact[@"retention"],@"request") &&
        equal(exact[@"preset_id"],selection[@"preset_id"]) && equal(exact[@"preset_revision"],selection[@"preset_revision"]) &&
        equal(exact[@"catalog_revision"],resolution[@"catalog_revision"]) &&
        equal(exact[@"expected_resolution_digest"],resolution[@"resolution_digest"]) &&
        equal(exact[@"target_request_memory_bytes"],selection[@"target_request_memory_bytes"]) &&
        equal(request[@"execution"][@"streaming"][@"target_request_memory_bytes"],selection[@"target_request_memory_bytes"]) &&
        text_value(resolution[@"resolution_digest"]) && text_value(selection[@"record_digest"]),"worker_resolution_mismatch");
    NSMutableDictionary *bound=[request mutableCopy],*execution=[request[@"execution"] mutableCopy];
    execution[@"streaming"]=exact;bound[@"execution"]=execution;return bound;
}
inline NSDictionary *generation_output(NSDictionary *request) {
    id outputs=request[@"outputs"];
    require([outputs isKindOfClass:NSArray.class] && [outputs count]==1,"worker_output_invalid");
    NSDictionary *output=object(outputs[0]);id path=output[@"path"];
    require(equal(output[@"kind"],@"image") && text_value(path) && [path isAbsolutePath] &&
        ![[path pathComponents] containsObject:@"."] && ![[path pathComponents] containsObject:@".."] &&
        [path rangeOfString:@"//"].location==NSNotFound && [path rangeOfString:@"\0"].location==NSNotFound &&
        [[path pathExtension] isEqual:@"png"],"worker_output_invalid");
    return output;
}
inline void require_fresh_output(NSDictionary *request) {
    NSString *path=generation_output(request)[@"path"],*parent=[path stringByDeletingLastPathComponent];
    char physical[PATH_MAX];
    require(realpath(parent.fileSystemRepresentation,physical)!=nullptr && [parent isEqual:@(physical)],"worker_output_parent_invalid");
    struct stat info{};
    require(lstat(path.fileSystemRepresentation,&info)!=0 && errno==ENOENT,"worker_output_exists");
}
inline NSDictionary *verify_generation(NSDictionary *result,NSDictionary *request,NSDictionary *resolution) {
    NSDictionary *summary=object(result[@"public_streaming"]),*selection=object(resolution[@"selection"]),*identity=object(resolution[@"identity"]);
    NSDictionary *output=generation_output(request),*sampling=object(request[@"sampling"]);
    require(integer(result[@"schema_version"],1) && boolean(result[@"warmup"],false) &&
        equal(result[@"model"],request[@"model"]) && equal(result[@"operation"],request[@"operation"]) &&
        equal(result[@"output"],output[@"path"]) && equal(result[@"width"],output[@"width"]) && equal(result[@"height"],output[@"height"]) &&
        equal(result[@"seed"],sampling[@"seed"]) && equal(result[@"steps"],sampling[@"steps"]) &&
        integer(summary[@"schema_version"],1) && boolean(summary[@"actual_plan_verified"],true),"worker_result_mismatch");
    for(NSString *key in @[@"target_request_memory_bytes",@"calibrated_request_bytes",@"preset_id",@"preset_revision",@"record_digest",@"component_policy_revision",@"execution_container",@"memory_scope"])
        require(equal(summary[key],selection[key]),"worker_result_mismatch");
    for(NSString *key in @[@"catalog_revision",@"resolution_digest"])
        require(equal(summary[key],resolution[key]),"worker_result_mismatch");
    for(NSString *key in @[@"source_digest",@"runtime_digest",@"device_digest"])
        require(equal(summary[key],identity[key]),"worker_result_mismatch");
    require(equal(summary[@"workload_digest"],resolution[@"request_digest"]) &&
        equal(summary[@"authorized_layout_digest"],selection[@"layout_digest"]) &&
        equal(summary[@"actual_layout_digest"],selection[@"layout_digest"]) &&
        (integer(summary[@"receipt_schema_version"],2) || integer(summary[@"receipt_schema_version"],3)) &&
        [summary[@"receipt_source_generation"] isKindOfClass:NSNumber.class] &&
        CFGetTypeID((__bridge CFTypeRef)summary[@"receipt_source_generation"])!=CFBooleanGetTypeID() &&
        [summary[@"receipt_source_generation"] doubleValue]>0 &&
        text_value(summary[@"receipt_digest"]) && text_value(summary[@"receipt_verifier_revision"]),"worker_result_mismatch");
    return summary;
}
// Parent still validates PNG framing/decode/dimensions and owns publication.
// Hash the opened regular file and reject replacement/change during the read.
inline NSDictionary *generation_artifact(NSDictionary *request,const std::atomic<bool> &stop) {
    NSString *path=generation_output(request)[@"path"];
    int fd=open(path.fileSystemRepresentation,O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);
    require(fd>=0,"worker_artifact_open_failed");
    struct Close {int fd;~Close(){close(fd);}} close_fd{fd};
    struct stat before{},after{},named{};
    require(fstat(fd,&before)==0 && S_ISREG(before.st_mode) && before.st_nlink==1 && before.st_size>0,"worker_artifact_invalid");
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    CC_SHA256_CTX hash;CC_SHA256_Init(&hash);std::array<unsigned char,1024*1024> buffer{};
    off_t count=0;
    while(count<before.st_size) {
        require(!stop.load(std::memory_order_relaxed),"worker_cancelled");
        ssize_t got=read(fd,buffer.data(),size_t(std::min<off_t>(buffer.size(),before.st_size-count)));
        if(got<0 && errno==EINTR)continue;
        require(got>0,"worker_artifact_read_failed");CC_SHA256_Update(&hash,buffer.data(),CC_LONG(got));count+=got;
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(digest,&hash);
#pragma clang diagnostic pop
    require(fstat(fd,&after)==0 && lstat(path.fileSystemRepresentation,&named)==0 &&
        S_ISREG(named.st_mode) && named.st_dev==before.st_dev && named.st_ino==before.st_ino &&
        after.st_nlink==1 && after.st_size==before.st_size && named.st_size==after.st_size &&
        before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec && before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec &&
        before.st_ctimespec.tv_sec==after.st_ctimespec.tv_sec && before.st_ctimespec.tv_nsec==after.st_ctimespec.tv_nsec,"worker_artifact_changed");
    NSMutableString *hex=[NSMutableString stringWithCapacity:64];for(unsigned char value:digest)[hex appendFormat:@"%02x",value];
    return @{@"path":path,@"size":@(before.st_size),@"sha256":hex};
}
}

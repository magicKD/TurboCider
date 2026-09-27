#pragma once
#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tc_worker {
inline void require(bool value, const char *message) {
    if (!value) throw std::invalid_argument(message);
}
inline NSData *canonical_request(NSDictionary *request) {
    NSError *error=nil;
    NSData *data=[NSJSONSerialization dataWithJSONObject:request
        options:NSJSONWritingSortedKeys|NSJSONWritingWithoutEscapingSlashes error:&error];
    require(data!=nil,"worker_request_not_json"); return data;
}
inline NSString *request_digest(NSDictionary *request) {
    NSMutableData *bytes=[NSMutableData dataWithData:[@"tc-worker-request-v1\n" dataUsingEncoding:NSUTF8StringEncoding]];
    [bytes appendData:canonical_request(request)];
    require(bytes.length<=UINT32_MAX,"worker_request_too_large");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(bytes.bytes,CC_LONG(bytes.length),digest);
    NSMutableString *hex=[NSMutableString stringWithCapacity:64];
    for(unsigned char value:digest)[hex appendFormat:@"%02x",value]; return hex;
}
inline bool uuid(id value) {
    return [value isKindOfClass:NSString.class] && [value length]==36 &&
        [[[NSUUID alloc] initWithUUIDString:value].UUIDString.lowercaseString isEqual:value];
}
inline bool integer(id value, int expected) {
    return [value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value)!=CFBooleanGetTypeID() &&
        [value doubleValue]==expected;
}
inline NSDictionary *validate_input(id input) {
    require([input isKindOfClass:NSDictionary.class],"worker_input_not_object");
    NSSet *keys=[NSSet setWithArray:@[@"protocol_version",@"job_id",@"request_id",@"request_digest",@"model_installation_ref",@"native_request_v2"]];
    require([[NSSet setWithArray:[input allKeys]] isEqual:keys],"worker_input_fields_invalid");
    require(integer(input[@"protocol_version"],1),"worker_protocol_unsupported");
    require(uuid(input[@"job_id"]) && uuid(input[@"request_id"]),"worker_ids_invalid");
    id path=input[@"model_installation_ref"];
    require([path isKindOfClass:NSString.class] && [path isAbsolutePath] &&
        [path isEqual:[path stringByStandardizingPath]] && [path rangeOfString:@"\0"].location==NSNotFound,
        "worker_installation_ref_invalid");
    id request=input[@"native_request_v2"];
    require([request isKindOfClass:NSDictionary.class] && integer(request[@"schema_version"],2),"worker_request_schema_invalid");
    require([request[@"model"] isEqual:@"z-image-turbo"] || [request[@"model"] isEqual:@"flux2-klein-4b"],"worker_model_unsupported");
    require([request[@"operation"] isEqual:@"image.generate"],"worker_operation_unsupported");
    id execution=request[@"execution"];
    require([execution isKindOfClass:NSDictionary.class] && [execution[@"streaming"] isKindOfClass:NSDictionary.class],"worker_public_selector_required");
    id selector=execution[@"streaming"];
    require(integer(selector[@"schema_version"],2) && [selector[@"enabled"] isKindOfClass:NSNumber.class] &&
        CFGetTypeID((__bridge CFTypeRef)selector[@"enabled"])==CFBooleanGetTypeID() && [selector[@"enabled"] boolValue],"worker_public_selector_required");
    require([input[@"request_digest"] isKindOfClass:NSString.class] &&
        [input[@"request_digest"] isEqual:request_digest(request)],"worker_request_digest_mismatch");
    return input;
}
inline NSDictionary *read_input(const char *path) {
    int fd=open(path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);
    require(fd>=0,"worker_input_open_failed");
    struct Close { int fd; ~Close(){close(fd);} } close_fd{fd};
    struct stat before{},after{};
    require(fstat(fd,&before)==0 && S_ISREG(before.st_mode) && before.st_size>0 && before.st_size<=1<<20,"worker_input_size_invalid");
    NSMutableData *data=[NSMutableData dataWithLength:size_t(before.st_size)];
    size_t offset=0;
    while(offset<data.length) {
        ssize_t count=read(fd,static_cast<char *>(data.mutableBytes)+offset,data.length-offset);
        if(count<0 && errno==EINTR)continue;
        require(count>0,"worker_input_read_failed");offset+=size_t(count);
    }
    require(fstat(fd,&after)==0 && before.st_size==after.st_size &&
        before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec && before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec &&
        before.st_ctimespec.tv_sec==after.st_ctimespec.tv_sec && before.st_ctimespec.tv_nsec==after.st_ctimespec.tv_nsec,"worker_input_changed");
    return validate_input([NSJSONSerialization JSONObjectWithData:data options:0 error:nil]);
}
inline NSMutableDictionary *terminal(NSDictionary *input, const char *runtime) {
    return [@{@"protocol_version":@1,@"job_id":input[@"job_id"],@"request_id":input[@"request_id"],
        @"request_digest":input[@"request_digest"],@"status":@"error",@"actual_container":@"cli_worker",
        @"runtime_fingerprint":@(runtime),@"resolution_digest":[NSNull null],@"record_digest":[NSNull null],
        @"layout_digest":[NSNull null],@"public_streaming_summary":[NSNull null],@"artifact":[NSNull null],
        @"error":[NSNull null]} mutableCopy];
}
}

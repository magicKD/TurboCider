#import "h3_ane_bridge.h"

#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct h3_ane_model {
    void *model;
    void *request;
    char *staging_directory;
    double compile_seconds;
    bool cache_hit;
    bool loaded;
};

static void bridge_fail(char *error, size_t error_size, const char *format,
                        ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double bridge_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

int h3_ane_bridge_available(void) {
    static int state = -1;
    if (state >= 0) return state;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
           "AppleNeuralEngine", RTLD_NOW);
    state = NSClassFromString(@"_ANEInMemoryModelDescriptor") != nil &&
            NSClassFromString(@"_ANEInMemoryModel") != nil &&
            NSClassFromString(@"_ANERequest") != nil &&
            NSClassFromString(@"_ANEIOSurfaceObject") != nil;
    return state;
}

IOSurfaceRef h3_ane_bridge_surface(size_t bytes) {
    size_t aligned = (bytes + 16383u) & ~(size_t)16383u;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(aligned),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(aligned),
        (id)kIOSurfaceAllocSize: @(aligned),
        (id)kIOSurfacePixelFormat: @0});
}

bool h3_ane_cache_enabled(void) {
    const char *env = getenv("H3_ANE_CACHE");
    return !env || atoi(env) != 0;
}

static bool bridge_blob_cache_enabled(void) {
    const char *env = getenv("H3_ANE_BLOB_CACHE");
    return !env || atoi(env) != 0;
}

static bool bridge_blob_kind_valid(const char *kind) {
    if (!kind || !*kind) return false;
    for (const unsigned char *cursor = (const unsigned char *)kind;
         *cursor; cursor++) {
        if ((*cursor >= 'a' && *cursor <= 'z') ||
            (*cursor >= 'A' && *cursor <= 'Z') ||
            (*cursor >= '0' && *cursor <= '9') || *cursor == '-' ||
            *cursor == '_') continue;
        return false;
    }
    return true;
}

static void bridge_blob_hash(uint64_t *key, const void *bytes, size_t length) {
    if (!key || (!bytes && length)) return;
    const uint8_t *values = bytes;
    for (size_t index = 0; index < length; index++) {
        *key ^= values[index];
        *key *= UINT64_C(1099511628211);
    }
}

uint64_t h3_ane_blob_cache_key_begin(const char *kind) {
    uint64_t key = UINT64_C(14695981039346656037);
    static const char version[] = "h3-ane-blob-cache-v1";
    bridge_blob_hash(&key, version, sizeof(version));
    if (kind) bridge_blob_hash(&key, kind, strlen(kind) + 1u);
    return key;
}

void h3_ane_blob_cache_key_bytes(uint64_t *key, const void *bytes,
                                 size_t length) {
    bridge_blob_hash(key, bytes, length);
}

int h3_ane_blob_cache_key_file_range(uint64_t *key, const char *path,
                                     uint64_t offset, uint64_t bytes,
                                     char *error, size_t error_size) {
    struct stat status;
    if (!key || !path || !*path || stat(path, &status) != 0 ||
        status.st_size < 0 || offset > UINT64_MAX - bytes ||
        offset + bytes > (uint64_t)status.st_size) {
        bridge_fail(error, error_size,
                    "invalid ANE blob-cache source range %s [%llu, %llu)",
                    path ? path : "?", (unsigned long long)offset,
                    (unsigned long long)(offset <= UINT64_MAX - bytes ?
                                         offset + bytes : UINT64_MAX));
        return 0;
    }
    struct {
        uint64_t device;
        uint64_t inode;
        uint64_t size;
        int64_t modified_seconds;
        int64_t modified_nanoseconds;
        uint64_t offset;
        uint64_t bytes;
    } identity = {
        (uint64_t)status.st_dev,
        (uint64_t)status.st_ino,
        (uint64_t)status.st_size,
        (int64_t)status.st_mtimespec.tv_sec,
        (int64_t)status.st_mtimespec.tv_nsec,
        offset,
        bytes
    };
    bridge_blob_hash(key, &identity, sizeof(identity));
    bridge_blob_hash(key, path, strlen(path) + 1u);
    return 1;
}

static NSString *bridge_blob_cache_path(const char *kind, uint64_t key,
                                        size_t bytes) {
    if (!bridge_blob_kind_valid(kind)) return nil;
    NSString *directory = [[[NSTemporaryDirectory()
        stringByAppendingPathComponent:@"h3-ane-blob-cache"]
        stringByAppendingPathComponent:@(kind)] stringByStandardizingPath];
    return [directory stringByAppendingPathComponent:
        [NSString stringWithFormat:@"%016llx-%zu.bin",
         (unsigned long long)key, bytes]];
}

void *h3_ane_blob_cache_load(const char *kind, uint64_t key,
                             size_t expected_bytes, int *cache_hit) {
    if (cache_hit) *cache_hit = 0;
    if (!bridge_blob_cache_enabled() || !expected_bytes) return NULL;
    @autoreleasepool {
        NSString *path = bridge_blob_cache_path(kind, key, expected_bytes);
        if (!path) return NULL;
        const char *file = path.fileSystemRepresentation;
        struct stat status;
        int descriptor = stat(file, &status) == 0 && status.st_size >= 0 &&
                         (uint64_t)status.st_size == (uint64_t)expected_bytes ?
                         open(file, O_RDONLY) : -1;
        void *result = descriptor >= 0 ? malloc(expected_bytes) : NULL;
        size_t done = 0;
        while (result && done < expected_bytes) {
            ssize_t count = read(descriptor, (uint8_t *)result + done,
                                 expected_bytes - done);
            if (count <= 0) break;
            done += (size_t)count;
        }
        int close_status = descriptor >= 0 ? close(descriptor) : -1;
        if (!result || done != expected_bytes || close_status != 0) {
            free(result);
            return NULL;
        }
        if (cache_hit) *cache_hit = 1;
        return result;
    }
}

void h3_ane_blob_cache_store(const char *kind, uint64_t key,
                             const void *bytes, size_t length) {
    if (!bridge_blob_cache_enabled() || !bytes || !length) return;
    @autoreleasepool {
        NSString *path = bridge_blob_cache_path(kind, key, length);
        if (!path) return;
        NSFileManager *files = [NSFileManager defaultManager];
        NSString *directory = path.stringByDeletingLastPathComponent;
        if (![files createDirectoryAtPath:directory
               withIntermediateDirectories:YES attributes:nil error:nil])
            return;
        NSDictionary *attributes = [files attributesOfItemAtPath:path
                                                            error:nil];
        if ([attributes[NSFileSize] unsignedLongLongValue] ==
            (unsigned long long)length) return;
        NSString *temporary = [path stringByAppendingFormat:@".%d.tmp",
                                                       (int)getpid()];
        [files removeItemAtPath:temporary error:nil];
        int descriptor = open(temporary.fileSystemRepresentation,
                              O_WRONLY | O_CREAT | O_EXCL, 0600);
        size_t done = 0;
        while (descriptor >= 0 && done < length) {
            ssize_t count = write(descriptor, (const uint8_t *)bytes + done,
                                  length - done);
            if (count <= 0) break;
            done += (size_t)count;
        }
        int close_status = descriptor >= 0 ? close(descriptor) : -1;
        if (done == length && close_status == 0) {
            [files removeItemAtPath:path error:nil];
            if (rename(temporary.fileSystemRepresentation,
                       path.fileSystemRepresentation) == 0) return;
        }
        [files removeItemAtPath:temporary error:nil];
    }
}

static void bridge_write_sources(NSString *directory, NSData *program,
                                 NSData *weights) {
    NSFileManager *files = [NSFileManager defaultManager];
    [files createDirectoryAtPath:
        [directory stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [program writeToFile:
        [directory stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [weights writeToFile:
        [directory stringByAppendingPathComponent:@"weights/weight.bin"]
        atomically:YES];
}

static NSString *bridge_cache_entry(NSString *identifier) {
    return [[NSTemporaryDirectory()
        stringByAppendingPathComponent:@"h3-ane-cache"]
        stringByAppendingPathComponent:identifier];
}

/* Clone a directory tree with per-file hardlinks on the same volume. macOS
 * cannot hardlink a directory itself; falling back to copyItemAtPath for the
 * whole tree duplicated every compiled program during restore. */
static bool bridge_clone_item(NSFileManager *files, NSString *from,
                              NSString *to) {
    BOOL directory = NO;
    if (![files fileExistsAtPath:from isDirectory:&directory]) return false;
    [files removeItemAtPath:to error:nil];
    if (!directory) {
        if ([files linkItemAtPath:from toPath:to error:nil]) return true;
        [files removeItemAtPath:to error:nil];
        return [files copyItemAtPath:from toPath:to error:nil];
    }
    if (![files createDirectoryAtPath:to withIntermediateDirectories:YES
                           attributes:nil error:nil]) return false;
    for (NSString *name in
         [files contentsOfDirectoryAtPath:from error:nil]) {
        if (!bridge_clone_item(
                files, [from stringByAppendingPathComponent:name],
                [to stringByAppendingPathComponent:name])) return false;
    }
    return true;
}

static bool bridge_mirror(NSString *from, NSString *to) {
    NSFileManager *files = [NSFileManager defaultManager];
    [files removeItemAtPath:to error:nil];
    [files createDirectoryAtPath:[to stringByDeletingLastPathComponent]
        withIntermediateDirectories:YES attributes:nil error:nil];
    return bridge_clone_item(files, from, to);
}

/* The model unload deletes its staging directory, so compiled artifacts are
 * preserved in a content-addressed entry next to it and restored on reuse. */
static bool bridge_cache_restore(NSString *identifier, NSString *directory) {
    NSString *entry = bridge_cache_entry(identifier);
    NSFileManager *files = [NSFileManager defaultManager];
    if (![files fileExistsAtPath:
            [entry stringByAppendingPathComponent:@"compiled.ok"]])
        return false;
    if (!bridge_mirror(entry, directory)) return false;
    [files removeItemAtPath:
        [directory stringByAppendingPathComponent:@"compiled.ok"] error:nil];
    return true;
}

/* FP16 conv constants are embedded in the compiled `data`. Programs with
 * constexpr dequantization may still consult the source weight mapping while
 * loading in a later process, so retain their MIL/weights as hardlinks. */
static void bridge_cache_store(NSString *identifier, NSString *directory,
                               bool retain_sources) {
    NSString *entry = bridge_cache_entry(identifier);
    NSFileManager *files = [NSFileManager defaultManager];
    [files removeItemAtPath:entry error:nil];
    if (![files createDirectoryAtPath:entry withIntermediateDirectories:YES
                           attributes:nil error:nil]) return;
    for (NSString *name in
         [files contentsOfDirectoryAtPath:directory error:nil]) {
        if (!retain_sources &&
            ([name isEqualToString:@"weights"] ||
             [name isEqualToString:@"model.mil"])) continue;
        NSString *from = [directory stringByAppendingPathComponent:name];
        NSString *to = [entry stringByAppendingPathComponent:name];
        if (!bridge_clone_item(files, from, to)) return;
    }
    [[NSData data] writeToFile:
        [entry stringByAppendingPathComponent:@"compiled.ok"] atomically:YES];
}

static void bridge_cache_evict(const char *identifier) {
    if (!identifier) return;
    [[NSFileManager defaultManager]
        removeItemAtPath:bridge_cache_entry(@(identifier)) error:nil];
}

h3_ane_model *h3_ane_model_create(const char *name, const char *mil,
                                  void *weight_bytes_owned, size_t weight_bytes,
                                  IOSurfaceRef *input_surfaces,
                                  uint32_t input_count, IOSurfaceRef output,
                                  char *error, size_t error_size) {
    if (!h3_ane_bridge_available()) {
        free(weight_bytes_owned);
        bridge_fail(error, error_size, "the Neural Engine bridge is "
                    "unavailable");
        return NULL;
    }
    h3_ane_model *handle = calloc(1, sizeof(*handle));
    if (!handle) {
        free(weight_bytes_owned);
        bridge_fail(error, error_size, "out of memory creating ANE %s", name);
        return NULL;
    }
    @autoreleasepool {
        NSError *failure = nil;
        NSData *weights = [NSData dataWithBytesNoCopy:weight_bytes_owned
                                               length:weight_bytes
                                         freeWhenDone:YES];
        NSData *program = [NSData dataWithBytes:mil length:strlen(mil)];
        Class descriptorClass =
            NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:),
            program, @{@"@model_path/weights/weight.bin":
                       @{@"offset": @0, @"data": weights}}, nil);
        if (!descriptor) {
            bridge_fail(error, error_size, "ANE %s descriptor rejected", name);
            free(handle);
            return NULL;
        }
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
        if (!model) {
            bridge_fail(error, error_size, "ANE %s model rejected", name);
            free(handle);
            return NULL;
        }
        NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(
            model, @selector(hexStringIdentifier));
        NSString *directory = [NSTemporaryDirectory()
            stringByAppendingPathComponent:identifier];
        NSFileManager *files = [NSFileManager defaultManager];
        bool cache = h3_ane_cache_enabled();
        bool retain_sources = strstr(mil, "constexpr_affine_dequantize") != NULL;
        bool cached = cache && bridge_cache_restore(identifier, directory);
        if (getenv("H3_ANE_CACHE_DEBUG"))
            fprintf(stderr,
                    "h3: ANE cache lookup name=%s id=%s restored=%d "
                    "retain-sources=%d\n",
                    name, identifier.UTF8String, cached, retain_sources);
        if (!cached) bridge_write_sources(directory, program, weights);
        handle->staging_directory = strdup(directory.UTF8String);
        double started = bridge_seconds();
        bool loaded = cached &&
            ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{},
                &failure);
        if (cached && !loaded) {
            if (getenv("H3_ANE_CACHE_DEBUG"))
                fprintf(stderr, "h3: ANE cache restore rejected for %s: %s\n",
                        name, failure ?
                        failure.localizedDescription.UTF8String : "?");
            failure = nil;
            bridge_write_sources(directory, program, weights);
        }
        if (!loaded) {
            if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                    model, @selector(compileWithQoS:options:error:), 21, @{},
                    &failure)) {
                bridge_fail(error, error_size, "ANE %s compile failed: %s",
                            name, failure ?
                            failure.localizedDescription.UTF8String : "?");
                [files removeItemAtPath:directory error:nil];
                free(handle->staging_directory);
                free(handle);
                return NULL;
            }
            if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                    model, @selector(loadWithQoS:options:error:), 21, @{},
                    &failure)) {
                bridge_fail(error, error_size, "ANE %s load failed: %s", name,
                            failure ?
                            failure.localizedDescription.UTF8String : "?");
                [files removeItemAtPath:directory error:nil];
                free(handle->staging_directory);
                free(handle);
                return NULL;
            }
            if (cache)
                bridge_cache_store(identifier, directory, retain_sources);
        }
        handle->cache_hit = loaded;
        handle->loaded = true;
        handle->compile_seconds = bridge_seconds() - started;
        NSMutableArray *inputs = [NSMutableArray array];
        NSMutableArray *indices = [NSMutableArray array];
        for (uint32_t index = 0; index < input_count; index++) {
            [inputs addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:),
                input_surfaces[index])];
            [indices addObject:@(index)];
        }
        id wrapped = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), output);
        id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))
                      objc_msgSend)(
            requestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                      weightsBuffer:perfStats:procedureIndex:),
            inputs, indices, @[wrapped], @[@0], nil, nil, @0);
        if (!request) {
            bridge_fail(error, error_size, "ANE %s request rejected", name);
            h3_ane_model_free(handle);
            return NULL;
        }
        handle->model = (__bridge_retained void *)model;
        handle->request = (__bridge_retained void *)request;
    }
    return handle;
}

int h3_ane_model_eval(h3_ane_model *handle, char *error, size_t error_size) {
    if (!handle || !handle->model || !handle->request || !handle->loaded) {
        bridge_fail(error, error_size, "the ANE model is not loaded");
        return 0;
    }
    int ok = 0;
    @autoreleasepool {
        NSError *failure = nil;
        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            (__bridge id)handle->model,
            @selector(evaluateWithQoS:options:request:error:), 21, @{},
            (__bridge id)handle->request, &failure) ? 1 : 0;
        if (!ok)
            bridge_fail(error, error_size, "ANE evaluation failed: %s",
                        failure ?
                        failure.localizedDescription.UTF8String : "?");
    }
    return ok;
}

int h3_ane_model_unload(h3_ane_model *handle, char *error, size_t error_size) {
    if (!handle || !handle->model) {
        bridge_fail(error, error_size, "no ANE model to unload");
        return 0;
    }
    if (!handle->loaded) return 1;
    @autoreleasepool {
        NSError *failure = nil;
        if (!((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                (__bridge id)handle->model, @selector(unloadWithQoS:error:), 21,
                &failure)) {
            bridge_fail(error, error_size, "ANE unload failed: %s",
                        failure ?
                        failure.localizedDescription.UTF8String : "?");
            return 0;
        }
    }
    handle->loaded = false;
    return 1;
}

int h3_ane_model_reload(h3_ane_model *handle, char *error, size_t error_size) {
    if (!handle || !handle->model || !handle->staging_directory) {
        bridge_fail(error, error_size, "no ANE model to reload");
        return 0;
    }
    if (handle->loaded) return 1;
    @autoreleasepool {
        NSString *directory = @(handle->staging_directory);
        bridge_cache_restore(directory.lastPathComponent, directory);
        NSError *failure = nil;
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                (__bridge id)handle->model,
                @selector(loadWithQoS:options:error:), 21, @{}, &failure)) {
            bridge_fail(error, error_size, "ANE reload failed: %s",
                        failure ?
                        failure.localizedDescription.UTF8String : "?");
            return 0;
        }
    }
    handle->loaded = true;
    return 1;
}

void h3_ane_model_free(h3_ane_model *handle) {
    if (!handle) return;
    @autoreleasepool {
        if (handle->model) {
            id model = (__bridge_transfer id)handle->model;
            if (handle->loaded) {
                NSError *failure = nil;
                ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                    model, @selector(unloadWithQoS:error:), 21, &failure);
            }
        }
        if (handle->request) {
            id request = (__bridge_transfer id)handle->request;
            (void)request;
        }
        if (handle->staging_directory) {
            [[NSFileManager defaultManager]
                removeItemAtPath:@(handle->staging_directory) error:nil];
            if (!h3_ane_cache_enabled())
                bridge_cache_evict(strrchr(handle->staging_directory, '/') ?
                    strrchr(handle->staging_directory, '/') + 1 :
                    handle->staging_directory);
            free(handle->staging_directory);
        }
    }
    free(handle);
}

double h3_ane_model_compile_seconds(const h3_ane_model *handle) {
    return handle ? handle->compile_seconds : 0.0;
}

bool h3_ane_model_cache_hit(const h3_ane_model *handle) {
    return handle ? handle->cache_hit : false;
}

int h3_ane_model_is_loaded(const h3_ane_model *handle) {
    return handle && handle->loaded;
}

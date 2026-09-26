#include "ltx_gemma_ane_mlp.h"
#include "ltx_gpu_internal.h"

#import <CommonCrypto/CommonDigest.h>
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

struct ltx_gemma_ane_mlp {
    ltx_gemma_ane_mlp_shape shape;
    ltx_gpu_buffer *gate_weight;
    ltx_gpu_buffer *gate_scale;
    ltx_gpu_buffer *up_weight;
    ltx_gpu_buffer *up_scale;
    ltx_gpu_buffer *down_weight;
    ltx_gpu_buffer *down_scale;
    ltx_gpu_buffer *gpu_partial;
    ltx_gpu_buffer *ane_input;
    ltx_gpu_buffer *ane_output;
    MLModel *model;
    MLMultiArray *input_array;
    MLMultiArray *output_array;
    MLDictionaryFeatureProvider *provider;
    MLPredictionOptions *options;
    pthread_t worker_thread;
    pthread_mutex_t worker_mutex;
    pthread_cond_t worker_condition;
    int worker_started;
    int worker_pending;
    int worker_stop;
    int inflight;
    int async_ok;
    int output_backing_used;
    double async_ms;
    char async_error[1024];
};

static void gm_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double gm_now_ms(void) {
    struct timespec value = {};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return ((double)value.tv_sec + (double)value.tv_nsec * 1e-9) * 1000.0;
}

static NSDictionary *gm_dict(NSDictionary *root, NSString *key) {
    id value = root[key];
    return [value isKindOfClass:[NSDictionary class]] ? value : nil;
}

static int gm_u32(NSDictionary *root, NSString *key, int allow_zero,
                  uint32_t *result) {
    id value = root[key];
    if (![value isKindOfClass:[NSNumber class]]) return 0;
    unsigned long long parsed = [value unsignedLongLongValue];
    if ((!allow_zero && !parsed) || parsed > UINT32_MAX) return 0;
    *result = (uint32_t)parsed;
    return 1;
}

static int gm_constraint_supports_shape(MLMultiArrayConstraint *constraint,
                                        NSArray<NSNumber *> *shape) {
    if (!constraint || !shape) return 0;
    if ([constraint.shape isEqual:shape]) return 1;
    MLMultiArrayShapeConstraint *flexible = constraint.shapeConstraint;
    if (flexible.type == MLMultiArrayShapeConstraintTypeEnumerated)
        return [flexible.enumeratedShapes containsObject:shape];
    if (flexible.type != MLMultiArrayShapeConstraintTypeRange ||
        flexible.sizeRangeForDimension.count != shape.count) return 0;
    for (NSUInteger index = 0; index < shape.count; index++)
        if (!NSLocationInRange(shape[index].unsignedIntegerValue,
                               [flexible.sizeRangeForDimension[index]
                                   rangeValue])) return 0;
    return 1;
}

static NSString *gm_sha256_data(NSData *data) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString *result = [NSMutableString stringWithCapacity:64];
    for (unsigned index = 0; index < CC_SHA256_DIGEST_LENGTH; index++)
        [result appendFormat:@"%02x", digest[index]];
    return result;
}

static NSData *gm_data(NSURL *directory, NSDictionary *files, NSString *name,
                       size_t bytes,
                       char *error, size_t error_size) {
    NSDictionary *metadata = gm_dict(files, name);
    NSNumber *manifest_bytes = metadata[@"bytes"];
    NSString *manifest_sha = metadata[@"sha256"];
    if (!metadata || ![manifest_bytes isKindOfClass:[NSNumber class]] ||
        ![manifest_sha isKindOfClass:[NSString class]] ||
        manifest_bytes.unsignedLongLongValue != bytes) {
        gm_fail(error, error_size, "invalid metadata for %s", name.UTF8String);
        return nil;
    }
    NSURL *url = [directory URLByAppendingPathComponent:name];
    NSData *data = [NSData dataWithContentsOfURL:url
                                          options:NSDataReadingMappedIfSafe
                                            error:nil];
    if (!data || data.length != bytes) {
        gm_fail(error, error_size, "%s has %llu bytes, expected %zu",
                url.path.UTF8String,
                (unsigned long long)(data ? data.length : 0), bytes);
        return nil;
    }
    if (![[gm_sha256_data(data) lowercaseString]
            isEqualToString:[manifest_sha lowercaseString]]) {
        gm_fail(error, error_size, "%s SHA-256 mismatch",
                url.path.UTF8String);
        return nil;
    }
    return data;
}

static ltx_gpu_buffer *gm_load(ltx_gpu *gpu, NSURL *directory,
                               NSDictionary *files, NSString *name,
                               size_t bytes, char *error, size_t error_size) {
    NSData *data = gm_data(directory, files, name, bytes, error, error_size);
    return data ? ltx_gpu_buffer_new_copy(gpu, data.bytes, data.length,
                                          error, error_size) : NULL;
}

static NSString *gm_sha256_uncached(NSString *path) {
    NSFileHandle *handle = [NSFileHandle fileHandleForReadingAtPath:path];
    if (!handle) return nil;
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    while (true) {
        NSData *data = [handle readDataOfLength:(8u << 20)];
        if (!data.length) break;
        CC_SHA256_Update(&context, data.bytes, (CC_LONG)data.length);
    }
    [handle closeFile];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    NSMutableString *result = [NSMutableString stringWithCapacity:64];
    for (unsigned index = 0; index < CC_SHA256_DIGEST_LENGTH; index++)
        [result appendFormat:@"%02x", digest[index]];
    return result;
}

/* A Gemma checkpoint is roughly 14 GiB.  Per-layer or per-encode hashing
 * would dominate the actual prefill, so cache the verified digest against a
 * canonical path plus the same size/mtime identity used by the manifest.
 * The mutex also prevents concurrent encoders from hashing the same file
 * twice. */
static NSString *gm_sha256(NSString *path) {
    static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    static char cached_path[PATH_MAX] = {};
    static char cached_digest[CC_SHA256_DIGEST_LENGTH * 2u + 1u] = {};
    static off_t cached_bytes = 0;
    static struct timespec cached_mtime = {};
    char canonical[PATH_MAX];
    struct stat status = {};
    if (!path || !realpath(path.fileSystemRepresentation, canonical) ||
        stat(canonical, &status) != 0) return nil;
    pthread_mutex_lock(&mutex);
    if (cached_digest[0] && strcmp(cached_path, canonical) == 0 &&
        cached_bytes == status.st_size &&
        cached_mtime.tv_sec == status.st_mtimespec.tv_sec &&
        cached_mtime.tv_nsec == status.st_mtimespec.tv_nsec) {
        NSString *result = [NSString stringWithUTF8String:cached_digest];
        pthread_mutex_unlock(&mutex);
        return result;
    }
    NSString *canonical_path = [NSString stringWithUTF8String:canonical];
    NSString *result = gm_sha256_uncached(canonical_path);
    if (result) {
        snprintf(cached_path, sizeof(cached_path), "%s", canonical);
        snprintf(cached_digest, sizeof(cached_digest), "%s",
                 result.UTF8String);
        cached_bytes = status.st_size;
        cached_mtime = status.st_mtimespec;
    }
    pthread_mutex_unlock(&mutex);
    return result;
}

static int gm_paths_equal(NSString *left, NSString *right) {
    char left_path[PATH_MAX], right_path[PATH_MAX];
    return realpath(left.fileSystemRepresentation, left_path) &&
           realpath(right.fileSystemRepresentation, right_path) &&
           strcmp(left_path, right_path) == 0;
}

static int gm_bind_io(ltx_gemma_ane_mlp *mlp, ltx_gpu *gpu,
                      char *error, size_t error_size) {
    size_t elements = (size_t)mlp->shape.rows * mlp->shape.hidden;
    size_t bytes = elements * sizeof(uint16_t);
    mlp->ane_input = ltx_gpu_buffer_new(gpu, bytes, error, error_size);
    mlp->ane_output = ltx_gpu_buffer_new(gpu, bytes, error, error_size);
    if (!mlp->ane_input || !mlp->ane_output) return 0;
    NSError *failure = nil;
    id<MTLBuffer> input_buffer =
        (__bridge id<MTLBuffer>)ltx_gpu_buffer_native(mlp->ane_input);
    id<MTLBuffer> output_buffer =
        (__bridge id<MTLBuffer>)ltx_gpu_buffer_native(mlp->ane_output);
    NSArray<NSNumber *> *shape = @[@1, @(mlp->shape.hidden), @1,
                                   @(mlp->shape.rows)];
    NSArray<NSNumber *> *strides = @[@(elements), @1, @(elements),
                                     @(mlp->shape.hidden)];
    mlp->input_array = [[MLMultiArray alloc]
        initWithDataPointer:input_buffer.contents shape:shape
        dataType:MLMultiArrayDataTypeFloat16 strides:strides
        deallocator:^(void *pointer) { (void)pointer; } error:&failure];
    mlp->output_array = [[MLMultiArray alloc]
        initWithDataPointer:output_buffer.contents shape:shape
        dataType:MLMultiArrayDataTypeFloat16 strides:strides
        deallocator:^(void *pointer) { (void)pointer; } error:&failure];
    if (!mlp->input_array || !mlp->output_array) {
        gm_fail(error, error_size, "Gemma ANE Core ML array setup failed: %s",
                failure.localizedDescription.UTF8String ?: "unknown error");
        return 0;
    }
    mlp->provider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"x" :
            [MLFeatureValue featureValueWithMultiArray:mlp->input_array] }
        error:&failure];
    mlp->options = [MLPredictionOptions new];
    mlp->options.outputBackings = @{ @"y" : mlp->output_array };
    if (!mlp->provider) {
        gm_fail(error, error_size, "Gemma ANE Core ML provider setup failed: %s",
                failure.localizedDescription.UTF8String ?: "unknown error");
        return 0;
    }
    return 1;
}

static void *gm_prediction_thread(void *opaque) {
    ltx_gemma_ane_mlp *mlp = opaque;
    double started = gm_now_ms();
    mlp->async_ok = 0;
    mlp->output_backing_used = 0;
    mlp->async_error[0] = '\0';
    @autoreleasepool {
        @try {
            NSError *failure = nil;
            id<MLFeatureProvider> result =
                [mlp->model predictionFromFeatures:mlp->provider
                                           options:mlp->options error:&failure];
            if (!result) {
                snprintf(mlp->async_error, sizeof(mlp->async_error),
                         "%s", failure.localizedDescription.UTF8String ?
                         failure.localizedDescription.UTF8String :
                         "unknown Core ML prediction failure");
            } else {
                MLMultiArray *returned =
                    [result featureValueForName:@"y"].multiArrayValue;
                mlp->output_backing_used =
                    returned == mlp->output_array &&
                    returned.dataPointer == mlp->output_array.dataPointer;
                if (!mlp->output_backing_used) {
                    snprintf(mlp->async_error, sizeof(mlp->async_error),
                             "Gemma ANE rejected caller-owned output backing");
                } else {
                    mlp->async_ok = 1;
                }
            }
        } @catch (NSException *exception) {
            snprintf(mlp->async_error, sizeof(mlp->async_error), "%s",
                     exception.reason.UTF8String ? exception.reason.UTF8String :
                     "unknown Core ML exception");
        }
    }
    mlp->async_ms = gm_now_ms() - started;
    return NULL;
}

static void *gm_persistent_worker(void *opaque) {
    ltx_gemma_ane_mlp *mlp = opaque;
    pthread_mutex_lock(&mlp->worker_mutex);
    for (;;) {
        while (!mlp->worker_pending && !mlp->worker_stop)
            pthread_cond_wait(&mlp->worker_condition, &mlp->worker_mutex);
        if (mlp->worker_stop) break;
        pthread_mutex_unlock(&mlp->worker_mutex);
        gm_prediction_thread(mlp);
        pthread_mutex_lock(&mlp->worker_mutex);
        mlp->worker_pending = 0;
        pthread_cond_broadcast(&mlp->worker_condition);
    }
    pthread_mutex_unlock(&mlp->worker_mutex);
    return NULL;
}

static int gm_worker_initialize(ltx_gemma_ane_mlp *mlp,
                                char *error, size_t error_size) {
    if (pthread_mutex_init(&mlp->worker_mutex, NULL) != 0) {
        gm_fail(error, error_size,
                "cannot initialize Gemma ANE worker mutex");
        return 0;
    }
    if (pthread_cond_init(&mlp->worker_condition, NULL) != 0) {
        pthread_mutex_destroy(&mlp->worker_mutex);
        gm_fail(error, error_size,
                "cannot initialize Gemma ANE worker condition");
        return 0;
    }
    if (pthread_create(&mlp->worker_thread, NULL, gm_persistent_worker,
                       mlp) != 0) {
        pthread_cond_destroy(&mlp->worker_condition);
        pthread_mutex_destroy(&mlp->worker_mutex);
        gm_fail(error, error_size,
                "cannot create persistent Gemma ANE prediction worker");
        return 0;
    }
    mlp->worker_started = 1;
    return 1;
}

static int gm_prediction_start(ltx_gemma_ane_mlp *mlp,
                               char *error, size_t error_size) {
    if (!mlp || !mlp->worker_started) {
        gm_fail(error, error_size,
                "Gemma ANE prediction worker is missing");
        return 0;
    }
    pthread_mutex_lock(&mlp->worker_mutex);
    if (mlp->inflight || mlp->worker_pending || mlp->worker_stop) {
        pthread_mutex_unlock(&mlp->worker_mutex);
        gm_fail(error, error_size,
                "Gemma ANE prediction worker is unavailable or already active");
        return 0;
    }
    mlp->async_ok = 0;
    mlp->async_ms = 0.0;
    mlp->inflight = 1;
    mlp->worker_pending = 1;
    pthread_cond_signal(&mlp->worker_condition);
    pthread_mutex_unlock(&mlp->worker_mutex);
    return 1;
}

static int gm_prediction_wait(ltx_gemma_ane_mlp *mlp,
                              char *error, size_t error_size) {
    if (!mlp || !mlp->worker_started) {
        gm_fail(error, error_size, "Gemma ANE prediction was not started");
        return 0;
    }
    pthread_mutex_lock(&mlp->worker_mutex);
    if (!mlp->inflight) {
        pthread_mutex_unlock(&mlp->worker_mutex);
        gm_fail(error, error_size, "Gemma ANE prediction was not started");
        return 0;
    }
    while (mlp->worker_pending)
        pthread_cond_wait(&mlp->worker_condition, &mlp->worker_mutex);
    int ok = mlp->async_ok;
    mlp->inflight = 0;
    pthread_mutex_unlock(&mlp->worker_mutex);
    if (!ok)
        gm_fail(error, error_size, "Gemma ANE prediction failed: %s",
                mlp->async_error[0] ? mlp->async_error : "unknown error");
    return ok;
}

static int gm_plan_rows(NSDictionary *shape, uint32_t requested_rows,
                        uint32_t *execution_rows,
                        uint32_t *minimum_profitable_rows,
                        char *error, size_t error_size) {
    uint32_t manifest_rows = 0;
    NSArray *supported = shape[@"supported_rows"];
    if (!gm_u32(shape, @"rows", 0, &manifest_rows) ||
        ![supported isKindOfClass:[NSArray class]] ||
        supported.count == 0u || supported.count > 5u) {
        gm_fail(error, error_size, "invalid Gemma ANE supported rows");
        return 0;
    }
    uint32_t row_values[5] = {};
    uint32_t selected = 0;
    uint32_t previous = 0;
    int default_supported = 0;
    for (NSUInteger index = 0; index < supported.count; ++index) {
        id value = supported[index];
        NSNumber *number = [value isKindOfClass:[NSNumber class]] ? value : nil;
        if (!number || number.doubleValue != number.unsignedIntValue ||
            number.unsignedIntValue == 0u ||
            number.unsignedIntValue > 1024u ||
            number.unsignedIntValue <= previous) {
            gm_fail(error, error_size, "invalid Gemma ANE supported row");
            return 0;
        }
        const uint32_t row = number.unsignedIntValue;
        row_values[index] = row;
        previous = row;
        if (row == manifest_rows) default_supported = 1;
        if (!selected && row >= requested_rows) selected = row;
    }
    if (!default_supported || !selected) {
        gm_fail(error, error_size,
                "Gemma ANE manifest does not support %u rows", requested_rows);
        return 0;
    }

    uint32_t minimum = selected;
    id policy_value = shape[@"minimum_profitable_rows"];
    if (policy_value != nil) {
        if (![policy_value isKindOfClass:[NSDictionary class]] ||
            [(NSDictionary *)policy_value count] > supported.count) {
            gm_fail(error, error_size,
                    "Gemma ANE minimum-profitable-row policy is invalid");
            return 0;
        }
        NSDictionary *policy = policy_value;
        for (id key in policy) {
            if (![key isKindOfClass:[NSString class]]) {
                gm_fail(error, error_size,
                        "Gemma ANE profitability bucket is invalid");
                return 0;
            }
            uint32_t policy_bucket = 0;
            for (NSUInteger index = 0; index < supported.count; ++index) {
                NSString *expected = [NSString stringWithFormat:@"%u",
                                      row_values[index]];
                if ([(NSString *)key isEqualToString:expected]) {
                    policy_bucket = row_values[index];
                    break;
                }
            }
            NSNumber *threshold = policy[key];
            if (!policy_bucket ||
                ![threshold isKindOfClass:[NSNumber class]] ||
                threshold.doubleValue != threshold.unsignedIntValue ||
                threshold.unsignedIntValue == 0u ||
                threshold.unsignedIntValue > policy_bucket) {
                gm_fail(error, error_size,
                        "Gemma ANE minimum-profitable-row threshold is invalid");
                return 0;
            }
            if (policy_bucket == selected)
                minimum = threshold.unsignedIntValue;
        }
    }
    *execution_rows = selected;
    *minimum_profitable_rows = minimum;
    return 1;
}

int ltx_gemma_ane_mlp_plan(
        const char *manifest_path, uint32_t requested_rows,
        uint32_t *execution_rows, uint32_t *minimum_profitable_rows,
        char *error, size_t error_size) {
    if (!manifest_path || !*manifest_path || !requested_rows ||
        !execution_rows || !minimum_profitable_rows) {
        gm_fail(error, error_size, "invalid Gemma ANE row-plan arguments");
        return 0;
    }
    @autoreleasepool {
        NSString *manifest_name = [NSString stringWithUTF8String:manifest_path];
        NSData *encoded = [NSData dataWithContentsOfFile:manifest_name];
        NSError *failure = nil;
        id decoded = encoded ?
            [NSJSONSerialization JSONObjectWithData:encoded options:0 error:&failure]
            : nil;
        if (![decoded isKindOfClass:[NSDictionary class]] ||
            ![decoded[@"schema"] isEqualToString:@"ltx-gemma-ane-mlp-v1"]) {
            gm_fail(error, error_size, "invalid Gemma ANE row-plan manifest");
            return 0;
        }
        NSDictionary *shape = gm_dict(decoded, @"shape");
        if (!shape) {
            gm_fail(error, error_size,
                    "Gemma ANE row-plan manifest has no shape");
            return 0;
        }
        return gm_plan_rows(shape, requested_rows, execution_rows,
                            minimum_profitable_rows, error, error_size);
    }
}

ltx_gemma_ane_mlp *ltx_gemma_ane_mlp_create(
    ltx_gpu *gpu, const char *manifest_path, const char *checkpoint,
    uint32_t layer, uint32_t rows, char *error, size_t error_size) {
    if (!gpu || !manifest_path || !checkpoint || !*manifest_path || !*checkpoint)
        return (gm_fail(error, error_size, "Gemma ANE requires GPU, manifest and checkpoint"), NULL);
    @autoreleasepool {
        NSString *manifest_name = [NSString stringWithUTF8String:manifest_path];
        NSData *encoded = [NSData dataWithContentsOfFile:manifest_name];
        NSError *failure = nil;
        id decoded = encoded ?
            [NSJSONSerialization JSONObjectWithData:encoded options:0 error:&failure]
            : nil;
        if (![decoded isKindOfClass:[NSDictionary class]]) {
            gm_fail(error, error_size, "invalid Gemma ANE manifest: %s",
                    failure.localizedDescription.UTF8String ?: "missing file");
            return NULL;
        }
        NSDictionary *root = decoded;
        if (![root[@"schema"] isEqualToString:@"ltx-gemma-ane-mlp-v1"]) {
            gm_fail(error, error_size, "unsupported Gemma ANE manifest schema");
            return NULL;
        }
        NSDictionary *shape = gm_dict(root, @"shape");
        NSDictionary *partition = gm_dict(root, @"partition");
        NSDictionary *ane = gm_dict(partition, @"ane");
        NSDictionary *gpu_part = gm_dict(partition, @"gpu");
        uint32_t manifest_layer = 0, manifest_rows = 0, hidden = 0;
        uint32_t intermediate = 0, ane_start = 0, ane_width = 0;
        uint32_t gpu_start = 0, gpu_width = 0;
        if (!shape || !ane || !gpu_part || !gm_u32(root, @"block_index", 1,
                                                     &manifest_layer) ||
            !gm_u32(shape, @"rows", 0, &manifest_rows) ||
            !gm_u32(shape, @"hidden", 0, &hidden) ||
            !gm_u32(shape, @"intermediate", 0, &intermediate) ||
            !gm_u32(ane, @"start", 1, &ane_start) ||
            !gm_u32(ane, @"width", 0, &ane_width) ||
            !gm_u32(gpu_part, @"start", 1, &gpu_start) ||
            !gm_u32(gpu_part, @"width", 0, &gpu_width) ||
            manifest_layer >= 48u || manifest_layer != layer || hidden != 3840u ||
            intermediate != 15360u || ane_width + gpu_width != intermediate ||
            ane_start != 0u || gpu_start != ane_width ||
            !ane_width || !gpu_width || ane_width % 256u || gpu_width % 256u) {
            gm_fail(error, error_size, "invalid Gemma ANE manifest geometry");
            return NULL;
        }
        uint32_t execution_rows = 0;
        uint32_t minimum_profitable_rows = 0;
        if (!gm_plan_rows(shape, rows, &execution_rows,
                          &minimum_profitable_rows, error, error_size))
            return NULL;
        NSDictionary *source = gm_dict(root, @"source");
        NSString *source_path = source[@"path"];
        NSNumber *source_bytes = source[@"bytes"];
        NSString *source_sha = source[@"sha256"];
        NSString *requested = [NSString stringWithUTF8String:checkpoint];
        NSString *requested_resolved = requested.stringByResolvingSymlinksInPath;
        NSDictionary *compiled = gm_dict(root, @"compiled_artifacts");
        NSString *artifact_name = compiled[@"fp16"];
        NSDictionary *convrot = gm_dict(root, @"convrot");
        uint32_t convrot_group = 0;
        if (![root[@"activation"] isEqualToString:@"gelu_tanh_gated"] ||
            !convrot || !gm_u32(convrot, @"group_size", 0, &convrot_group) ||
            convrot_group != 256u ||
            ![convrot[@"ane_weights"] isEqualToString:@"absorbed_fp16"] ||
            ![convrot[@"gpu_weights"] isEqualToString:@"checkpoint_int8"]) {
            gm_fail(error, error_size, "invalid Gemma ANE operation contract");
            return NULL;
        }
        NSDictionary *source_attributes = [[NSFileManager defaultManager]
            attributesOfItemAtPath:requested_resolved error:nil];
        NSNumber *requested_bytes = source_attributes[NSFileSize];
        if (![requested isKindOfClass:[NSString class]] ||
            ![source_path isKindOfClass:[NSString class]] ||
            ![source_bytes isKindOfClass:[NSNumber class]] ||
            ![source_sha isKindOfClass:[NSString class]] ||
            !gm_paths_equal(source_path, requested) ||
            ![requested_bytes isKindOfClass:[NSNumber class]] ||
            source_bytes.unsignedLongLongValue !=
                requested_bytes.unsignedLongLongValue ||
            ![[gm_sha256(requested_resolved) lowercaseString]
                isEqualToString:[source_sha lowercaseString]] ||
            ![artifact_name isKindOfClass:[NSString class]] ||
            ![artifact_name.lastPathComponent isEqualToString:artifact_name]) {
            gm_fail(error, error_size, "Gemma ANE checkpoint provenance mismatch");
            return NULL;
        }
        NSURL *directory = [[NSURL fileURLWithPath:manifest_name]
            URLByDeletingLastPathComponent];
        NSURL *artifact = [directory URLByAppendingPathComponent:artifact_name];
        BOOL artifact_is_directory = NO;
        if (![artifact.pathExtension isEqualToString:@"mlmodelc"] ||
            ![[NSFileManager defaultManager] fileExistsAtPath:artifact.path
                                                  isDirectory:&artifact_is_directory] ||
            !artifact_is_directory) {
            gm_fail(error, error_size, "Gemma ANE requires compiled .mlmodelc artifact");
            return NULL;
        }
        ltx_gemma_ane_mlp *mlp = calloc(1, sizeof(*mlp));
        if (!mlp) return (gm_fail(error, error_size, "out of memory"), NULL);
        mlp->shape = (ltx_gemma_ane_mlp_shape){
            layer, execution_rows, hidden, intermediate, ane_width, gpu_width,
            minimum_profitable_rows};
        size_t input_weights = (size_t)gpu_width * hidden;
        size_t input_scales = (size_t)gpu_width * sizeof(float);
        size_t down_weights = (size_t)hidden * gpu_width;
        size_t down_scales = (size_t)hidden * sizeof(float);
        NSDictionary *files = gm_dict(root, @"files");
        mlp->gate_weight = gm_load(gpu, directory, files,
                                   @"gpu_gate.weight.i8",
                                   input_weights, error, error_size);
        mlp->gate_scale = gm_load(gpu, directory, files,
                                  @"gpu_gate.scale.f32",
                                  input_scales, error, error_size);
        mlp->up_weight = gm_load(gpu, directory, files,
                                 @"gpu_up.weight.i8",
                                 input_weights, error, error_size);
        mlp->up_scale = gm_load(gpu, directory, files,
                                @"gpu_up.scale.f32",
                                input_scales, error, error_size);
        mlp->down_weight = gm_load(gpu, directory, files,
                                   @"gpu_down.weight.i8",
                                   down_weights, error, error_size);
        mlp->down_scale = gm_load(gpu, directory, files,
                                  @"gpu_down.scale.f32",
                                  down_scales, error, error_size);
        size_t hidden_bytes = (size_t)execution_rows * hidden * sizeof(uint16_t);
        mlp->gpu_partial = ltx_gpu_buffer_new(gpu, hidden_bytes, error, error_size);
        if (!mlp->gate_weight || !mlp->gate_scale || !mlp->up_weight ||
            !mlp->up_scale || !mlp->down_weight || !mlp->down_scale ||
            !mlp->gpu_partial ||
            !gm_bind_io(mlp, gpu, error, error_size)) {
            ltx_gemma_ane_mlp_free(mlp);
            return NULL;
        }
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        mlp->model = [MLModel modelWithContentsOfURL:artifact
                                      configuration:configuration
                                              error:&failure];
        if (!mlp->model) {
            gm_fail(error, error_size, "load Gemma ANE model: %s",
                    failure.localizedDescription.UTF8String ?: "unknown error");
            ltx_gemma_ane_mlp_free(mlp);
            return NULL;
        }
        MLFeatureDescription *input_description =
            mlp->model.modelDescription.inputDescriptionsByName[@"x"];
        MLFeatureDescription *output_description =
            mlp->model.modelDescription.outputDescriptionsByName[@"y"];
        NSArray<NSNumber *> *expected_shape =
            @[@1, @(hidden), @1, @(execution_rows)];
        MLMultiArrayConstraint *input_constraint =
            input_description.multiArrayConstraint;
        MLMultiArrayConstraint *output_constraint =
            output_description.multiArrayConstraint;
        /* Core ML 7 records the enumerated input shapes but leaves the
         * shape of an output inferred from that flexible dimension empty in
         * MLModelDescription.  The prediction is still required to use our
         * caller-owned, exactly shaped output backing below, so an empty
         * output constraint is a valid dynamic-output declaration rather
         * than an unchecked geometry. */
        int output_shape_supported = output_constraint &&
            (output_constraint.shape.count == 0u ||
             gm_constraint_supports_shape(output_constraint, expected_shape));
        if (!input_constraint || !output_constraint ||
            input_constraint.dataType != MLMultiArrayDataTypeFloat16 ||
            output_constraint.dataType != MLMultiArrayDataTypeFloat16 ||
            !gm_constraint_supports_shape(input_constraint, expected_shape) ||
            !output_shape_supported) {
            gm_fail(error, error_size,
                    "Gemma ANE model has invalid x/y interface: input=%s; output=%s",
                    input_description.description.UTF8String ?: "missing",
                    output_description.description.UTF8String ?: "missing");
            ltx_gemma_ane_mlp_free(mlp);
            return NULL;
        }
        if (!gm_worker_initialize(mlp, error, error_size)) {
            ltx_gemma_ane_mlp_free(mlp);
            return NULL;
        }
        return mlp;
    }
}

void ltx_gemma_ane_mlp_free(ltx_gemma_ane_mlp *mlp) {
    if (!mlp) return;
    if (mlp->inflight) gm_prediction_wait(mlp, NULL, 0);
    if (mlp->worker_started) {
        pthread_mutex_lock(&mlp->worker_mutex);
        mlp->worker_stop = 1;
        pthread_cond_signal(&mlp->worker_condition);
        pthread_mutex_unlock(&mlp->worker_mutex);
        pthread_join(mlp->worker_thread, NULL);
        pthread_cond_destroy(&mlp->worker_condition);
        pthread_mutex_destroy(&mlp->worker_mutex);
    }
    @autoreleasepool {
        mlp->model = nil;
        mlp->provider = nil;
        mlp->options = nil;
        mlp->input_array = nil;
        mlp->output_array = nil;
    }
    ltx_gpu_buffer_free(mlp->gate_weight);
    ltx_gpu_buffer_free(mlp->gate_scale);
    ltx_gpu_buffer_free(mlp->up_weight);
    ltx_gpu_buffer_free(mlp->up_scale);
    ltx_gpu_buffer_free(mlp->down_weight);
    ltx_gpu_buffer_free(mlp->down_scale);
    ltx_gpu_buffer_free(mlp->gpu_partial);
    ltx_gpu_buffer_free(mlp->ane_input);
    ltx_gpu_buffer_free(mlp->ane_output);
    free(mlp);
}

const ltx_gemma_ane_mlp_shape *ltx_gemma_ane_mlp_get_shape(
    const ltx_gemma_ane_mlp *mlp) { return mlp ? &mlp->shape : NULL; }

uint64_t ltx_gemma_ane_mlp_resident_weight_bytes(
    const ltx_gemma_ane_mlp *mlp) {
    if (!mlp) return 0;
    return ltx_gpu_buffer_bytes(mlp->gate_weight) +
           ltx_gpu_buffer_bytes(mlp->gate_scale) +
           ltx_gpu_buffer_bytes(mlp->up_weight) +
           ltx_gpu_buffer_bytes(mlp->up_scale) +
           ltx_gpu_buffer_bytes(mlp->down_weight) +
           ltx_gpu_buffer_bytes(mlp->down_scale);
}

uint64_t ltx_gemma_ane_mlp_workspace_bytes(
    const ltx_gemma_ane_mlp *mlp) {
    if (!mlp) return 0;
    return ltx_gpu_buffer_bytes(mlp->gpu_partial) +
           ltx_gpu_buffer_bytes(mlp->ane_input) +
           ltx_gpu_buffer_bytes(mlp->ane_output);
}

int ltx_gemma_ane_mlp_eval(
    ltx_gemma_ane_mlp *mlp, ltx_gpu *gpu, ltx_gpu_buffer *output,
    const ltx_gpu_buffer *input, ltx_gemma_ane_mlp_timing *timing,
    char *error, size_t error_size) {
    if (!mlp || !gpu || !output || !input) {
        gm_fail(error, error_size, "invalid Gemma ANE evaluation arguments");
        return 0;
    }
    ltx_gemma_ane_mlp_timing measured = {};
    uint32_t rows = mlp->shape.rows;
    uint32_t hidden = mlp->shape.hidden;
    uint32_t suffix = mlp->shape.gpu_intermediate;
    uint32_t elements = rows * hidden;
    double total = gm_now_ms();
    double started = gm_now_ms();
    if (!ltx_gpu_cast_bf16_f16(gpu, mlp->ane_input, input, elements,
                               error, error_size)) return 0;
    measured.pack_ms = gm_now_ms() - started;
    if (!gm_prediction_start(mlp, error, error_size)) return 0;
    started = gm_now_ms();
    if (!ltx_gpu_batch_begin(gpu, error, error_size)) {
        gm_prediction_wait(mlp, NULL, 0);
        return 0;
    }
    int ok = ltx_gpu_gated_mlp_int8_convrot_mps_bf16(
        gpu, mlp->gpu_partial, input, mlp->gate_weight, mlp->gate_scale,
        mlp->up_weight, mlp->up_scale, mlp->down_weight, mlp->down_scale,
        rows, hidden, suffix, hidden, 256u, error, error_size);
    if (!ltx_gpu_batch_end(gpu, error, error_size)) ok = 0;
    measured.gpu_ms = gm_now_ms() - started;
    int prediction_ok = gm_prediction_wait(
        mlp, ok ? error : NULL, ok ? error_size : 0);
    measured.ane_ms = mlp->async_ms;
    if (!prediction_ok) ok = 0;
    if (!ok) return 0;
    measured.overlap_ms = fmax(measured.ane_ms, measured.gpu_ms);
    started = gm_now_ms();
    if (!ltx_gpu_join_bf16_f16(gpu, output, mlp->gpu_partial,
                               mlp->ane_output, elements,
                               error, error_size)) return 0;
    measured.join_ms = gm_now_ms() - started;
    measured.total_ms = gm_now_ms() - total;
    measured.ane_output_backing_used = mlp->output_backing_used;
    if (timing) *timing = measured;
    return 1;
}

#include "ltx_ane_kv.h"
#include "ltx_gpu_internal.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct ltx_ane_kv_io {
    ltx_gpu *gpu;
    uint32_t text_rows;
    uint32_t hidden;
    unsigned references;
    ltx_gpu_buffer *input;
    ltx_gpu_buffer *key_output;
    ltx_gpu_buffer *value_output;
    struct ltx_ane_kv_io *next;
} ltx_ane_kv_io;

static pthread_mutex_t kv_io_mutex = PTHREAD_MUTEX_INITIALIZER;
static ltx_ane_kv_io *kv_io_list;

struct ltx_ane_kv {
    ltx_ane_kv_shape shape;
    ltx_ane_kv_io *shared_io;
    MLModel *model;
    MLMultiArray *input_array;
    MLMultiArray *key_array;
    MLMultiArray *value_array;
    MLDictionaryFeatureProvider *provider;
    MLPredictionOptions *options;
    NSURL *temporary_compiled;
    pthread_t thread;
    int inflight;
    int async_ok;
    int key_output_backing_used;
    int value_output_backing_used;
    double started_ms;
    double pack_ms;
    double async_ms;
    char async_error[1024];
};

static double kv_now_ms(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return ((double)value.tv_sec + (double)value.tv_nsec * 1e-9) * 1000.0;
}

static void kv_fail(char *error, size_t error_size,
                    const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static NSDictionary *kv_dictionary(NSDictionary *root, NSString *key) {
    id value = root[key];
    return [value isKindOfClass:[NSDictionary class]] ? value : nil;
}

static int kv_u32(NSDictionary *root, NSString *key, int allow_zero,
                  uint32_t *value) {
    id number = root[key];
    if (![number isKindOfClass:[NSNumber class]]) return 0;
    unsigned long long parsed = [number unsignedLongLongValue];
    if ((!allow_zero && !parsed) || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static ltx_ane_kv_io *kv_io_acquire(
        ltx_gpu *gpu, uint32_t text_rows, uint32_t hidden,
        char *error, size_t error_size) {
    pthread_mutex_lock(&kv_io_mutex);
    for (ltx_ane_kv_io *io = kv_io_list; io; io = io->next)
        if (io->gpu == gpu && io->text_rows == text_rows &&
            io->hidden == hidden) {
            io->references++;
            pthread_mutex_unlock(&kv_io_mutex);
            return io;
        }
    pthread_mutex_unlock(&kv_io_mutex);

    if ((size_t)text_rows > SIZE_MAX / hidden / sizeof(uint16_t)) {
        kv_fail(error, error_size, "ANE K/V IO size overflow");
        return NULL;
    }
    size_t bytes = (size_t)text_rows * hidden * sizeof(uint16_t);
    ltx_ane_kv_io *io = calloc(1, sizeof(*io));
    if (!io) {
        kv_fail(error, error_size, "out of memory creating ANE K/V IO");
        return NULL;
    }
    io->gpu = gpu;
    io->text_rows = text_rows;
    io->hidden = hidden;
    io->references = 1u;
    io->input = ltx_gpu_buffer_new(gpu, bytes, error, error_size);
    io->key_output = ltx_gpu_buffer_new(gpu, bytes, error, error_size);
    io->value_output = ltx_gpu_buffer_new(gpu, bytes, error, error_size);
    if (!io->input || !io->key_output || !io->value_output) {
        ltx_gpu_buffer_free(io->input);
        ltx_gpu_buffer_free(io->key_output);
        ltx_gpu_buffer_free(io->value_output);
        free(io);
        return NULL;
    }
    pthread_mutex_lock(&kv_io_mutex);
    io->next = kv_io_list;
    kv_io_list = io;
    pthread_mutex_unlock(&kv_io_mutex);
    return io;
}

static void kv_io_release(ltx_ane_kv_io *io) {
    if (!io) return;
    int destroy = 0;
    pthread_mutex_lock(&kv_io_mutex);
    if (io->references) io->references--;
    if (!io->references) {
        ltx_ane_kv_io **cursor = &kv_io_list;
        while (*cursor && *cursor != io) cursor = &(*cursor)->next;
        if (*cursor == io) *cursor = io->next;
        destroy = 1;
    }
    pthread_mutex_unlock(&kv_io_mutex);
    if (!destroy) return;
    ltx_gpu_buffer_free(io->input);
    ltx_gpu_buffer_free(io->key_output);
    ltx_gpu_buffer_free(io->value_output);
    free(io);
}

static void *kv_prediction_thread(void *opaque) {
    ltx_ane_kv *kv = opaque;
    kv->async_ok = 0;
    kv->async_error[0] = '\0';
    double start = kv_now_ms();
    @autoreleasepool {
        @try {
            NSError *failure = nil;
            id<MLFeatureProvider> result =
                [kv->model predictionFromFeatures:kv->provider
                                         options:kv->options
                                           error:&failure];
            if (!result) {
                const char *message = failure.localizedDescription.UTF8String;
                snprintf(kv->async_error, sizeof(kv->async_error), "%s",
                         message ? message : "Core ML K/V prediction failed");
            } else {
                MLMultiArray *returned_key =
                    [result featureValueForName:@"key"].multiArrayValue;
                MLMultiArray *returned_value =
                    [result featureValueForName:@"value"].multiArrayValue;
                kv->key_output_backing_used =
                    returned_key == kv->key_array &&
                    returned_key.dataPointer == kv->key_array.dataPointer;
                kv->value_output_backing_used =
                    returned_value == kv->value_array &&
                    returned_value.dataPointer == kv->value_array.dataPointer;
                if (!kv->key_output_backing_used ||
                    !kv->value_output_backing_used) {
                    snprintf(kv->async_error, sizeof(kv->async_error),
                             "Core ML rejected K/V output backing");
                } else {
                    kv->async_ok = 1;
                }
            }
        } @catch (NSException *exception) {
            const char *message = exception.reason.UTF8String;
            snprintf(kv->async_error, sizeof(kv->async_error), "%s",
                     message ? message : "Core ML K/V exception");
        }
    }
    kv->async_ms = kv_now_ms() - start;
    return NULL;
}

ltx_ane_kv *ltx_ane_kv_create(ltx_gpu *gpu,
                              const char *manifest_path,
                              const char *variant,
                              char *error, size_t error_size) {
    if (!gpu || !manifest_path || !manifest_path[0] ||
        !variant || !variant[0]) {
        kv_fail(error, error_size,
                "GPU, ANE K/V manifest and variant are required");
        return NULL;
    }
    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:manifest_path];
        NSData *encoded = [NSData dataWithContentsOfFile:path];
        NSError *failure = nil;
        id decoded = encoded ?
            [NSJSONSerialization JSONObjectWithData:encoded options:0
                                               error:&failure] : nil;
        if (![decoded isKindOfClass:[NSDictionary class]]) {
            const char *message = failure ?
                failure.localizedDescription.UTF8String : "missing file";
            kv_fail(error, error_size, "cannot parse ANE K/V manifest: %s",
                    message ? message : "invalid JSON");
            return NULL;
        }
        NSDictionary *root = decoded;
        NSDictionary *shape = kv_dictionary(root, @"shape");
        ltx_ane_kv_shape parsed = {0};
        if (![root[@"schema"] isEqualToString:@"ltx-ane-text-kv-v1"] ||
            !shape ||
            !kv_u32(root, @"block_index", 1, &parsed.block_index) ||
            !kv_u32(shape, @"text_rows", 0, &parsed.text_rows) ||
            !kv_u32(shape, @"hidden", 0, &parsed.hidden) ||
            !kv_u32(shape, @"heads", 0, &parsed.heads) ||
            !kv_u32(shape, @"head_dim", 0, &parsed.head_dim) ||
            parsed.hidden != 4096u || parsed.heads != 32u ||
            parsed.head_dim != 128u) {
            kv_fail(error, error_size, "invalid ANE K/V manifest geometry");
            return NULL;
        }
        NSDictionary *artifacts = kv_dictionary(root, @"artifacts");
        NSDictionary *compiled =
            kv_dictionary(root, @"compiled_artifacts");
        NSString *variant_key = [NSString stringWithUTF8String:variant];
        NSString *model_name = nil;
        id compiled_name = compiled[variant_key];
        if ([compiled_name isKindOfClass:[NSString class]])
            model_name = compiled_name;
        if (!model_name) {
            id package_name = artifacts[variant_key];
            if ([package_name isKindOfClass:[NSString class]])
                model_name = package_name;
        }
        if (!model_name) {
            kv_fail(error, error_size,
                    "ANE K/V manifest lacks variant %s", variant);
            return NULL;
        }

        NSURL *directory = [[NSURL fileURLWithPath:path]
            URLByDeletingLastPathComponent];
        NSURL *model_url = [directory URLByAppendingPathComponent:model_name];
        ltx_ane_kv *kv = calloc(1, sizeof(*kv));
        if (!kv) {
            kv_fail(error, error_size,
                    "out of memory creating ANE K/V runtime");
            return NULL;
        }
        kv->shape = parsed;
        kv->shared_io = kv_io_acquire(
            gpu, parsed.text_rows, parsed.hidden, error, error_size);
        if (!kv->shared_io) {
            ltx_ane_kv_free(kv);
            return NULL;
        }

        NSURL *load_url = model_url;
        if (![model_url.pathExtension isEqualToString:@"mlmodelc"]) {
            load_url = [MLModel compileModelAtURL:model_url error:&failure];
            if (!load_url) {
                kv_fail(error, error_size, "compile ANE K/V model: %s",
                        failure.localizedDescription.UTF8String);
                ltx_ane_kv_free(kv);
                return NULL;
            }
            kv->temporary_compiled = load_url;
        }
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        kv->model = [MLModel modelWithContentsOfURL:load_url
                                      configuration:configuration
                                              error:&failure];
        if (!kv->model) {
            kv_fail(error, error_size, "load ANE K/V model: %s",
                    failure.localizedDescription.UTF8String);
            ltx_ane_kv_free(kv);
            return NULL;
        }

        size_t elements = (size_t)parsed.text_rows * parsed.hidden;
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)
            ltx_gpu_buffer_native(kv->shared_io->input);
        id<MTLBuffer> key_buffer = (__bridge id<MTLBuffer>)
            ltx_gpu_buffer_native(kv->shared_io->key_output);
        id<MTLBuffer> value_buffer = (__bridge id<MTLBuffer>)
            ltx_gpu_buffer_native(kv->shared_io->value_output);
        NSArray<NSNumber *> *input_shape =
            @[@1, @(parsed.hidden), @1, @(parsed.text_rows)];
        NSArray<NSNumber *> *input_strides =
            @[@(elements), @1, @(elements), @(parsed.hidden)];
        NSArray<NSNumber *> *output_shape =
            @[@1, @(parsed.heads), @(parsed.text_rows), @(parsed.head_dim)];
        NSArray<NSNumber *> *output_strides = @[
            @(elements), @(parsed.text_rows * parsed.head_dim),
            @(parsed.head_dim), @1
        ];
        kv->input_array = [[MLMultiArray alloc]
            initWithDataPointer:input_buffer.contents
                          shape:input_shape
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:input_strides
                     deallocator:^(void *pointer) { (void)pointer; }
                          error:&failure];
        kv->key_array = [[MLMultiArray alloc]
            initWithDataPointer:key_buffer.contents
                          shape:output_shape
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:output_strides
                     deallocator:^(void *pointer) { (void)pointer; }
                          error:&failure];
        kv->value_array = [[MLMultiArray alloc]
            initWithDataPointer:value_buffer.contents
                          shape:output_shape
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:output_strides
                     deallocator:^(void *pointer) { (void)pointer; }
                          error:&failure];
        if (!kv->input_array || !kv->key_array || !kv->value_array) {
            kv_fail(error, error_size, "create ANE K/V arrays: %s",
                    failure.localizedDescription.UTF8String);
            ltx_ane_kv_free(kv);
            return NULL;
        }
        kv->provider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{
                @"x": [MLFeatureValue featureValueWithMultiArray:
                        kv->input_array],
            } error:&failure];
        kv->options = [MLPredictionOptions new];
        kv->options.outputBackings = @{
            @"key": kv->key_array,
            @"value": kv->value_array,
        };
        if (!kv->provider) {
            kv_fail(error, error_size, "create ANE K/V provider: %s",
                    failure.localizedDescription.UTF8String);
            ltx_ane_kv_free(kv);
            return NULL;
        }
        return kv;
    }
}

void ltx_ane_kv_free(ltx_ane_kv *kv) {
    if (!kv) return;
    if (kv->inflight) pthread_join(kv->thread, NULL);
    @autoreleasepool {
        kv->model = nil;
        kv->provider = nil;
        kv->options = nil;
        kv->input_array = nil;
        kv->key_array = nil;
        kv->value_array = nil;
        if (kv->temporary_compiled)
            [[NSFileManager defaultManager]
                removeItemAtURL:kv->temporary_compiled error:nil];
        kv->temporary_compiled = nil;
    }
    kv_io_release(kv->shared_io);
    free(kv);
}

const ltx_ane_kv_shape *ltx_ane_kv_get_shape(const ltx_ane_kv *kv) {
    return kv ? &kv->shape : NULL;
}

int ltx_ane_kv_start(ltx_ane_kv *kv, ltx_gpu *gpu,
                     const ltx_gpu_buffer *scaled_text,
                     char *error, size_t error_size) {
    if (!kv || !gpu || !scaled_text || kv->inflight) {
        kv_fail(error, error_size, "invalid ANE K/V start arguments");
        return 0;
    }
    uint32_t elements = kv->shape.text_rows * kv->shape.hidden;
    kv->started_ms = kv_now_ms();
    double pack_start = kv->started_ms;
    if (!ltx_gpu_cast_bf16_f16(
            gpu, kv->shared_io->input, scaled_text, elements,
            error, error_size)) return 0;
    kv->pack_ms = kv_now_ms() - pack_start;
    kv->async_ok = 0;
    kv->async_ms = 0.0;
    kv->key_output_backing_used = 0;
    kv->value_output_backing_used = 0;
    kv->inflight = 1;
    if (pthread_create(&kv->thread, NULL, kv_prediction_thread, kv) != 0) {
        kv->inflight = 0;
        kv_fail(error, error_size,
                "cannot create ANE K/V prediction thread");
        return 0;
    }
    return 1;
}

int ltx_ane_kv_wait(ltx_ane_kv *kv, ltx_gpu *gpu,
                    ltx_gpu_buffer *key_output,
                    ltx_gpu_buffer *value_output,
                    ltx_ane_kv_timing *timing,
                    char *error, size_t error_size) {
    if (!kv || !gpu || !key_output || !value_output || !kv->inflight) {
        kv_fail(error, error_size, "invalid ANE K/V wait arguments");
        return 0;
    }
    int join_ok = pthread_join(kv->thread, NULL) == 0;
    kv->inflight = 0;
    if (!join_ok) {
        kv_fail(error, error_size, "cannot join ANE K/V thread");
        return 0;
    }
    if (!kv->async_ok) {
        kv_fail(error, error_size, "%s", kv->async_error[0] ?
                kv->async_error : "ANE K/V prediction failed");
        return 0;
    }
    double unpack_start = kv_now_ms();
    uint32_t elements = kv->shape.text_rows * kv->shape.hidden;
    if (!ltx_gpu_cast_f16_bf16(
            gpu, key_output, kv->shared_io->key_output, elements,
            error, error_size) ||
        !ltx_gpu_cast_f16_bf16(
            gpu, value_output, kv->shared_io->value_output, elements,
            error, error_size)) return 0;
    ltx_ane_kv_timing measured = {
        .pack_ms = kv->pack_ms,
        .ane_ms = kv->async_ms,
        .unpack_ms = kv_now_ms() - unpack_start,
        .total_ms = kv_now_ms() - kv->started_ms,
        .key_output_backing_used = kv->key_output_backing_used,
        .value_output_backing_used = kv->value_output_backing_used,
    };
    if (timing) *timing = measured;
    return 1;
}

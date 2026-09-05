#include "ltx_ane_v2a.h"
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

typedef struct ltx_ane_v2a_io {
    ltx_gpu *gpu;
    uint32_t video_rows;
    unsigned references;
    ltx_gpu_buffer *audio_input;
    ltx_gpu_buffer *video_input;
    ltx_gpu_buffer *output;
    struct ltx_ane_v2a_io *next;
} ltx_ane_v2a_io;

static pthread_mutex_t v2a_io_mutex = PTHREAD_MUTEX_INITIALIZER;
static ltx_ane_v2a_io *v2a_io_list;

struct ltx_ane_v2a {
    ltx_ane_v2a_shape shape;
    ltx_ane_v2a_io *shared_io;
    MLModel *model;
    MLMultiArray *audio_array;
    MLMultiArray *video_array;
    MLMultiArray *output_array;
    MLDictionaryFeatureProvider *provider;
    MLPredictionOptions *options;
    NSURL *temporary_compiled;
    pthread_t thread;
    int inflight;
    int async_ok;
    int output_backing_used;
    double started_ms;
    double pack_ms;
    double async_ms;
    char async_error[1024];
};

static double v2a_now_ms(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return ((double)value.tv_sec + (double)value.tv_nsec * 1e-9) * 1000.0;
}

static void v2a_fail(char *error, size_t error_size,
                     const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static NSDictionary *v2a_dictionary(NSDictionary *root, NSString *key) {
    id value = root[key];
    return [value isKindOfClass:[NSDictionary class]] ? value : nil;
}

static int v2a_u32(NSDictionary *root, NSString *key, int allow_zero,
                   uint32_t *value) {
    id number = root[key];
    if (![number isKindOfClass:[NSNumber class]]) return 0;
    unsigned long long parsed = [number unsignedLongLongValue];
    if ((!allow_zero && !parsed) || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static ltx_ane_v2a_io *v2a_io_acquire(
        ltx_gpu *gpu, uint32_t video_rows,
        char *error, size_t error_size) {
    pthread_mutex_lock(&v2a_io_mutex);
    for (ltx_ane_v2a_io *io = v2a_io_list; io; io = io->next)
        if (io->gpu == gpu && io->video_rows == video_rows) {
            io->references++;
            pthread_mutex_unlock(&v2a_io_mutex);
            return io;
        }
    pthread_mutex_unlock(&v2a_io_mutex);

    size_t audio_bytes =
        (size_t)101u * 2048u * sizeof(uint16_t);
    size_t video_bytes =
        (size_t)video_rows * 4096u * sizeof(uint16_t);
    ltx_ane_v2a_io *io = calloc(1, sizeof(*io));
    if (!io) {
        v2a_fail(error, error_size, "out of memory creating V-to-A IO");
        return NULL;
    }
    io->gpu = gpu;
    io->video_rows = video_rows;
    io->references = 1u;
    io->audio_input = ltx_gpu_buffer_new(
        gpu, audio_bytes, error, error_size);
    io->video_input = ltx_gpu_buffer_new(
        gpu, video_bytes, error, error_size);
    io->output = ltx_gpu_buffer_new(
        gpu, audio_bytes, error, error_size);
    if (!io->audio_input || !io->video_input || !io->output) {
        ltx_gpu_buffer_free(io->audio_input);
        ltx_gpu_buffer_free(io->video_input);
        ltx_gpu_buffer_free(io->output);
        free(io);
        return NULL;
    }
    pthread_mutex_lock(&v2a_io_mutex);
    io->next = v2a_io_list;
    v2a_io_list = io;
    pthread_mutex_unlock(&v2a_io_mutex);
    return io;
}

static void v2a_io_release(ltx_ane_v2a_io *io) {
    if (!io) return;
    int destroy = 0;
    pthread_mutex_lock(&v2a_io_mutex);
    if (io->references) io->references--;
    if (!io->references) {
        ltx_ane_v2a_io **cursor = &v2a_io_list;
        while (*cursor && *cursor != io) cursor = &(*cursor)->next;
        if (*cursor == io) *cursor = io->next;
        destroy = 1;
    }
    pthread_mutex_unlock(&v2a_io_mutex);
    if (!destroy) return;
    ltx_gpu_buffer_free(io->audio_input);
    ltx_gpu_buffer_free(io->video_input);
    ltx_gpu_buffer_free(io->output);
    free(io);
}

static void *v2a_prediction_thread(void *opaque) {
    ltx_ane_v2a *attention = opaque;
    attention->async_ok = 0;
    attention->async_error[0] = '\0';
    double start = v2a_now_ms();
    @autoreleasepool {
        @try {
            NSError *failure = nil;
            id<MLFeatureProvider> result =
                [attention->model predictionFromFeatures:attention->provider
                                                  options:attention->options
                                                    error:&failure];
            if (!result) {
                const char *message = failure.localizedDescription.UTF8String;
                snprintf(attention->async_error,
                         sizeof(attention->async_error), "%s",
                         message ? message : "Core ML V-to-A failed");
            } else {
                MLMultiArray *returned =
                    [result featureValueForName:@"y"].multiArrayValue;
                attention->output_backing_used =
                    returned == attention->output_array &&
                    returned.dataPointer ==
                        attention->output_array.dataPointer;
                if (!attention->output_backing_used) {
                    snprintf(attention->async_error,
                             sizeof(attention->async_error),
                             "Core ML rejected V-to-A output backing");
                } else {
                    attention->async_ok = 1;
                }
            }
        } @catch (NSException *exception) {
            const char *message = exception.reason.UTF8String;
            snprintf(attention->async_error,
                     sizeof(attention->async_error), "%s",
                     message ? message : "Core ML V-to-A exception");
        }
    }
    attention->async_ms = v2a_now_ms() - start;
    return NULL;
}

ltx_ane_v2a *ltx_ane_v2a_create(ltx_gpu *gpu,
                                const char *manifest_path,
                                const char *variant,
                                char *error, size_t error_size) {
    if (!gpu || !manifest_path || !*manifest_path ||
        !variant || !*variant) {
        v2a_fail(error, error_size,
                 "GPU, V-to-A manifest and variant are required");
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
            v2a_fail(error, error_size, "cannot parse V-to-A manifest: %s",
                     message ? message : "invalid JSON");
            return NULL;
        }
        NSDictionary *root = decoded;
        NSDictionary *shape = v2a_dictionary(root, @"shape");
        ltx_ane_v2a_shape parsed = {0};
        if (![root[@"schema"] isEqualToString:@"ltx-ane-v2a-v1"] ||
            !shape ||
            !v2a_u32(root, @"block_index", 1, &parsed.block_index) ||
            !v2a_u32(shape, @"audio_rows", 0, &parsed.audio_rows) ||
            !v2a_u32(shape, @"audio_dim", 0, &parsed.audio_dim) ||
            !v2a_u32(shape, @"video_rows", 0, &parsed.video_rows) ||
            !v2a_u32(shape, @"video_dim", 0, &parsed.video_dim) ||
            !v2a_u32(shape, @"heads", 0, &parsed.heads) ||
            !v2a_u32(shape, @"head_dim", 0, &parsed.head_dim) ||
            parsed.audio_rows != 101u || parsed.audio_dim != 2048u ||
            (parsed.video_rows != 1001u && parsed.video_rows != 4004u) ||
            parsed.video_dim != 4096u || parsed.heads != 32u ||
            parsed.head_dim != 64u) {
            v2a_fail(error, error_size, "invalid V-to-A manifest geometry");
            return NULL;
        }
        NSDictionary *artifacts = v2a_dictionary(root, @"artifacts");
        NSDictionary *compiled =
            v2a_dictionary(root, @"compiled_artifacts");
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
            v2a_fail(error, error_size,
                     "V-to-A manifest lacks Core ML variant %s", variant);
            return NULL;
        }

        NSURL *directory = [[NSURL fileURLWithPath:path]
            URLByDeletingLastPathComponent];
        NSURL *model_url = [directory URLByAppendingPathComponent:model_name];
        ltx_ane_v2a *attention = calloc(1, sizeof(*attention));
        if (!attention) {
            v2a_fail(error, error_size,
                     "out of memory creating V-to-A runtime");
            return NULL;
        }
        attention->shape = parsed;
        attention->shared_io = v2a_io_acquire(
            gpu, parsed.video_rows, error, error_size);
        if (!attention->shared_io) {
            ltx_ane_v2a_free(attention);
            return NULL;
        }

        NSURL *load_url = model_url;
        if (![model_url.pathExtension isEqualToString:@"mlmodelc"]) {
            load_url = [MLModel compileModelAtURL:model_url error:&failure];
            if (!load_url) {
                v2a_fail(error, error_size, "compile V-to-A Core ML: %s",
                         failure.localizedDescription.UTF8String);
                ltx_ane_v2a_free(attention);
                return NULL;
            }
            attention->temporary_compiled = load_url;
        }
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        attention->model = [MLModel modelWithContentsOfURL:load_url
                                            configuration:configuration
                                                    error:&failure];
        if (!attention->model) {
            v2a_fail(error, error_size, "load V-to-A Core ML: %s",
                     failure.localizedDescription.UTF8String);
            ltx_ane_v2a_free(attention);
            return NULL;
        }

        size_t audio_elements = (size_t)parsed.audio_rows * parsed.audio_dim;
        size_t video_elements = (size_t)parsed.video_rows * parsed.video_dim;
        id<MTLBuffer> audio_buffer = (__bridge id<MTLBuffer>)
            ltx_gpu_buffer_native(attention->shared_io->audio_input);
        id<MTLBuffer> video_buffer = (__bridge id<MTLBuffer>)
            ltx_gpu_buffer_native(attention->shared_io->video_input);
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)
            ltx_gpu_buffer_native(attention->shared_io->output);
        NSArray<NSNumber *> *audio_shape =
            @[@1, @(parsed.audio_dim), @1, @(parsed.audio_rows)];
        NSArray<NSNumber *> *audio_strides =
            @[@(audio_elements), @1, @(audio_elements), @(parsed.audio_dim)];
        NSArray<NSNumber *> *video_shape =
            @[@1, @(parsed.video_dim), @1, @(parsed.video_rows)];
        NSArray<NSNumber *> *video_strides =
            @[@(video_elements), @1, @(video_elements), @(parsed.video_dim)];
        attention->audio_array = [[MLMultiArray alloc]
            initWithDataPointer:audio_buffer.contents
                          shape:audio_shape
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:audio_strides
                     deallocator:^(void *pointer) { (void)pointer; }
                          error:&failure];
        attention->video_array = [[MLMultiArray alloc]
            initWithDataPointer:video_buffer.contents
                          shape:video_shape
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:video_strides
                     deallocator:^(void *pointer) { (void)pointer; }
                          error:&failure];
        attention->output_array = [[MLMultiArray alloc]
            initWithDataPointer:output_buffer.contents
                          shape:audio_shape
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:audio_strides
                     deallocator:^(void *pointer) { (void)pointer; }
                          error:&failure];
        if (!attention->audio_array || !attention->video_array ||
            !attention->output_array) {
            v2a_fail(error, error_size, "create V-to-A Core ML arrays: %s",
                     failure.localizedDescription.UTF8String);
            ltx_ane_v2a_free(attention);
            return NULL;
        }
        attention->provider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{
                @"audio": [MLFeatureValue featureValueWithMultiArray:
                            attention->audio_array],
                @"video": [MLFeatureValue featureValueWithMultiArray:
                            attention->video_array],
            } error:&failure];
        attention->options = [MLPredictionOptions new];
        attention->options.outputBackings = @{
            @"y": attention->output_array
        };
        if (!attention->provider) {
            v2a_fail(error, error_size,
                     "create V-to-A feature provider: %s",
                     failure.localizedDescription.UTF8String);
            ltx_ane_v2a_free(attention);
            return NULL;
        }
        return attention;
    }
}

void ltx_ane_v2a_free(ltx_ane_v2a *attention) {
    if (!attention) return;
    if (attention->inflight) pthread_join(attention->thread, NULL);
    @autoreleasepool {
        attention->model = nil;
        attention->provider = nil;
        attention->options = nil;
        attention->audio_array = nil;
        attention->video_array = nil;
        attention->output_array = nil;
        if (attention->temporary_compiled)
            [[NSFileManager defaultManager]
                removeItemAtURL:attention->temporary_compiled error:nil];
        attention->temporary_compiled = nil;
    }
    v2a_io_release(attention->shared_io);
    free(attention);
}

const ltx_ane_v2a_shape *ltx_ane_v2a_get_shape(
        const ltx_ane_v2a *attention) {
    return attention ? &attention->shape : NULL;
}

int ltx_ane_v2a_start(ltx_ane_v2a *attention, ltx_gpu *gpu,
                      const ltx_gpu_buffer *audio_input,
                      const ltx_gpu_buffer *video_input,
                      char *error, size_t error_size) {
    if (!attention || !gpu || !audio_input || !video_input ||
        attention->inflight) {
        v2a_fail(error, error_size, "invalid V-to-A start arguments");
        return 0;
    }
    uint32_t audio_elements =
        attention->shape.audio_rows * attention->shape.audio_dim;
    uint32_t video_elements =
        attention->shape.video_rows * attention->shape.video_dim;
    attention->started_ms = v2a_now_ms();
    double pack_start = attention->started_ms;
    if (!ltx_gpu_cast_bf16_f16(
            gpu, attention->shared_io->audio_input, audio_input,
            audio_elements, error, error_size) ||
        !ltx_gpu_cast_bf16_f16(
            gpu, attention->shared_io->video_input, video_input,
            video_elements, error, error_size)) return 0;
    attention->pack_ms = v2a_now_ms() - pack_start;
    attention->async_ok = 0;
    attention->async_ms = 0.0;
    attention->output_backing_used = 0;
    attention->inflight = 1;
    if (pthread_create(
            &attention->thread, NULL, v2a_prediction_thread,
            attention) != 0) {
        attention->inflight = 0;
        v2a_fail(error, error_size,
                 "cannot create V-to-A prediction thread");
        return 0;
    }
    return 1;
}

int ltx_ane_v2a_wait(ltx_ane_v2a *attention, ltx_gpu *gpu,
                     ltx_gpu_buffer *output,
                     ltx_ane_v2a_timing *timing,
                     char *error, size_t error_size) {
    if (!attention || !gpu || !output || !attention->inflight) {
        v2a_fail(error, error_size, "invalid V-to-A wait arguments");
        return 0;
    }
    int join_ok = pthread_join(attention->thread, NULL) == 0;
    attention->inflight = 0;
    if (!join_ok) {
        v2a_fail(error, error_size,
                 "cannot join V-to-A prediction thread");
        return 0;
    }
    if (!attention->async_ok) {
        v2a_fail(error, error_size, "%s",
                 attention->async_error[0] ? attention->async_error :
                 "V-to-A Core ML prediction failed");
        return 0;
    }
    double unpack_start = v2a_now_ms();
    uint32_t elements =
        attention->shape.audio_rows * attention->shape.audio_dim;
    if (!ltx_gpu_cast_f16_bf16(
            gpu, output, attention->shared_io->output, elements,
            error, error_size)) return 0;
    ltx_ane_v2a_timing measured = {
        .pack_ms = attention->pack_ms,
        .ane_ms = attention->async_ms,
        .unpack_ms = v2a_now_ms() - unpack_start,
        .total_ms = v2a_now_ms() - attention->started_ms,
        .output_backing_used = attention->output_backing_used,
    };
    if (timing) *timing = measured;
    return 1;
}

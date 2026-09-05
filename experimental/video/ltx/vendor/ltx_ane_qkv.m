#include "ltx_ane_qkv.h"
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

typedef struct ltx_ane_qkv_io {
    ltx_gpu *gpu;
    uint32_t rows;
    uint32_t hidden;
    unsigned references;
    ltx_gpu_buffer *input;
    ltx_gpu_buffer *query_output;
    ltx_gpu_buffer *key_output;
    ltx_gpu_buffer *value_output;
    struct ltx_ane_qkv_io *next;
} ltx_ane_qkv_io;

static pthread_mutex_t qkv_io_mutex = PTHREAD_MUTEX_INITIALIZER;
static ltx_ane_qkv_io *qkv_io_list;

struct ltx_ane_qkv {
    ltx_ane_qkv_shape shape;
    ltx_ane_qkv_io *shared_io;
    MLModel *model;
    MLMultiArray *input_array;
    MLMultiArray *query_array;
    MLMultiArray *key_array;
    MLMultiArray *value_array;
    MLDictionaryFeatureProvider *provider;
    MLPredictionOptions *options;
    NSURL *temporary_compiled;
    pthread_t thread;
    int inflight;
    int async_ok;
    int query_output_backing_used;
    int key_output_backing_used;
    int value_output_backing_used;
    double started_ms;
    double pack_ms;
    double async_ms;
    char async_error[1024];
};

static double qkv_now_ms(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return ((double)value.tv_sec + (double)value.tv_nsec * 1e-9) * 1000.0;
}

static void qkv_fail(char *error, size_t error_size,
                     const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static NSDictionary *qkv_dictionary(NSDictionary *root, NSString *key) {
    id value = root[key];
    return [value isKindOfClass:[NSDictionary class]] ? value : nil;
}

static int qkv_u32(NSDictionary *root, NSString *key, int allow_zero,
                   uint32_t *value) {
    id number = root[key];
    if (![number isKindOfClass:[NSNumber class]]) return 0;
    unsigned long long parsed = [number unsignedLongLongValue];
    if ((!allow_zero && !parsed) || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static ltx_ane_qkv_io *qkv_io_acquire(
        ltx_gpu *gpu, uint32_t rows, uint32_t hidden,
        char *error, size_t error_size) {
    pthread_mutex_lock(&qkv_io_mutex);
    for (ltx_ane_qkv_io *io = qkv_io_list; io; io = io->next)
        if (io->gpu == gpu && io->rows == rows && io->hidden == hidden) {
            io->references++;
            pthread_mutex_unlock(&qkv_io_mutex);
            return io;
        }
    pthread_mutex_unlock(&qkv_io_mutex);

    if ((size_t)rows > SIZE_MAX / hidden / sizeof(uint16_t)) {
        qkv_fail(error, error_size, "ANE QKV IO size overflow");
        return NULL;
    }
    size_t bytes = (size_t)rows * hidden * sizeof(uint16_t);
    ltx_ane_qkv_io *io = calloc(1, sizeof(*io));
    if (!io) {
        qkv_fail(error, error_size, "out of memory creating ANE QKV IO");
        return NULL;
    }
    io->gpu = gpu;
    io->rows = rows;
    io->hidden = hidden;
    io->references = 1u;
    io->input = ltx_gpu_buffer_new(gpu, bytes, error, error_size);
    io->query_output = ltx_gpu_buffer_new(gpu, bytes, error, error_size);
    io->key_output = ltx_gpu_buffer_new(gpu, bytes, error, error_size);
    io->value_output = ltx_gpu_buffer_new(gpu, bytes, error, error_size);
    if (!io->input || !io->query_output || !io->key_output ||
        !io->value_output) {
        ltx_gpu_buffer_free(io->input);
        ltx_gpu_buffer_free(io->query_output);
        ltx_gpu_buffer_free(io->key_output);
        ltx_gpu_buffer_free(io->value_output);
        free(io);
        return NULL;
    }
    pthread_mutex_lock(&qkv_io_mutex);
    io->next = qkv_io_list;
    qkv_io_list = io;
    pthread_mutex_unlock(&qkv_io_mutex);
    return io;
}

static void qkv_io_release(ltx_ane_qkv_io *io) {
    if (!io) return;
    int destroy = 0;
    pthread_mutex_lock(&qkv_io_mutex);
    if (io->references) io->references--;
    if (!io->references) {
        ltx_ane_qkv_io **cursor = &qkv_io_list;
        while (*cursor && *cursor != io) cursor = &(*cursor)->next;
        if (*cursor == io) *cursor = io->next;
        destroy = 1;
    }
    pthread_mutex_unlock(&qkv_io_mutex);
    if (!destroy) return;
    ltx_gpu_buffer_free(io->input);
    ltx_gpu_buffer_free(io->query_output);
    ltx_gpu_buffer_free(io->key_output);
    ltx_gpu_buffer_free(io->value_output);
    free(io);
}

static void *qkv_prediction_thread(void *opaque) {
    ltx_ane_qkv *qkv = opaque;
    qkv->async_ok = 0;
    qkv->async_error[0] = '\0';
    double start = qkv_now_ms();
    @autoreleasepool {
        @try {
            NSError *failure = nil;
            id<MLFeatureProvider> result =
                [qkv->model predictionFromFeatures:qkv->provider
                                           options:qkv->options
                                             error:&failure];
            if (!result) {
                const char *message = failure.localizedDescription.UTF8String;
                snprintf(qkv->async_error, sizeof(qkv->async_error), "%s",
                         message ? message : "Core ML QKV prediction failed");
            } else {
                MLMultiArray *returned_query =
                    [result featureValueForName:@"query"].multiArrayValue;
                MLMultiArray *returned_key =
                    [result featureValueForName:@"key"].multiArrayValue;
                MLMultiArray *returned_value =
                    [result featureValueForName:@"value"].multiArrayValue;
                qkv->query_output_backing_used =
                    returned_query == qkv->query_array &&
                    returned_query.dataPointer == qkv->query_array.dataPointer;
                qkv->key_output_backing_used =
                    returned_key == qkv->key_array &&
                    returned_key.dataPointer == qkv->key_array.dataPointer;
                qkv->value_output_backing_used =
                    returned_value == qkv->value_array &&
                    returned_value.dataPointer == qkv->value_array.dataPointer;
                if (!qkv->query_output_backing_used ||
                    !qkv->key_output_backing_used ||
                    !qkv->value_output_backing_used) {
                    snprintf(qkv->async_error, sizeof(qkv->async_error),
                             "Core ML rejected QKV output backing");
                } else {
                    qkv->async_ok = 1;
                }
            }
        } @catch (NSException *exception) {
            const char *message = exception.reason.UTF8String;
            snprintf(qkv->async_error, sizeof(qkv->async_error), "%s",
                     message ? message : "Core ML QKV exception");
        }
    }
    qkv->async_ms = qkv_now_ms() - start;
    return NULL;
}

ltx_ane_qkv *ltx_ane_qkv_create(ltx_gpu *gpu,
                                const char *manifest_path,
                                const char *variant,
                                char *error, size_t error_size) {
    if (!gpu || !manifest_path || !manifest_path[0] ||
        !variant || !variant[0]) {
        qkv_fail(error, error_size,
                 "GPU, ANE QKV manifest and variant are required");
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
            qkv_fail(error, error_size, "cannot parse ANE QKV manifest: %s",
                     message ? message : "invalid JSON");
            return NULL;
        }
        NSDictionary *root = decoded;
        NSDictionary *shape = qkv_dictionary(root, @"shape");
        ltx_ane_qkv_shape parsed = {0};
        if (![root[@"schema"] isEqualToString:@"ltx-ane-qkv-sequence-v1"] ||
            !shape ||
            !qkv_u32(root, @"block_index", 1, &parsed.block_index) ||
            !qkv_u32(shape, @"rows", 0, &parsed.rows) ||
            !qkv_u32(shape, @"hidden", 0, &parsed.hidden) ||
            parsed.hidden != 4096u) {
            qkv_fail(error, error_size, "invalid ANE QKV manifest geometry");
            return NULL;
        }
        NSDictionary *artifacts = qkv_dictionary(root, @"artifacts");
        NSDictionary *compiled = qkv_dictionary(root, @"compiled_artifacts");
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
            qkv_fail(error, error_size,
                     "ANE QKV manifest lacks variant %s", variant);
            return NULL;
        }

        NSURL *directory = [[NSURL fileURLWithPath:path]
            URLByDeletingLastPathComponent];
        NSURL *model_url = [directory URLByAppendingPathComponent:model_name];
        ltx_ane_qkv *qkv = calloc(1, sizeof(*qkv));
        if (!qkv) {
            qkv_fail(error, error_size,
                     "out of memory creating ANE QKV runtime");
            return NULL;
        }
        qkv->shape = parsed;
        qkv->shared_io = qkv_io_acquire(
            gpu, parsed.rows, parsed.hidden, error, error_size);
        if (!qkv->shared_io) {
            ltx_ane_qkv_free(qkv);
            return NULL;
        }
        NSURL *load_url = model_url;
        if (![model_url.pathExtension isEqualToString:@"mlmodelc"]) {
            load_url = [MLModel compileModelAtURL:model_url error:&failure];
            if (!load_url) {
                qkv_fail(error, error_size, "compile ANE QKV model: %s",
                         failure.localizedDescription.UTF8String);
                ltx_ane_qkv_free(qkv);
                return NULL;
            }
            qkv->temporary_compiled = load_url;
        }
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        qkv->model = [MLModel modelWithContentsOfURL:load_url
                                       configuration:configuration
                                               error:&failure];
        if (!qkv->model) {
            qkv_fail(error, error_size, "load ANE QKV model: %s",
                     failure.localizedDescription.UTF8String);
            ltx_ane_qkv_free(qkv);
            return NULL;
        }

        size_t elements = (size_t)parsed.rows * parsed.hidden;
        NSArray<NSNumber *> *array_shape =
            @[@1, @(parsed.hidden), @1, @(parsed.rows)];
        NSArray<NSNumber *> *array_strides =
            @[@(elements), @1, @(elements), @(parsed.hidden)];
#define LTX_QKV_ARRAY(PROPERTY, BUFFER) \
        do { \
            id<MTLBuffer> native = (__bridge id<MTLBuffer>) \
                ltx_gpu_buffer_native((BUFFER)); \
            qkv->PROPERTY = [[MLMultiArray alloc] \
                initWithDataPointer:native.contents shape:array_shape \
                dataType:MLMultiArrayDataTypeFloat16 strides:array_strides \
                deallocator:^(void *pointer) { (void)pointer; } \
                error:&failure]; \
        } while (0)
        LTX_QKV_ARRAY(input_array, qkv->shared_io->input);
        LTX_QKV_ARRAY(query_array, qkv->shared_io->query_output);
        LTX_QKV_ARRAY(key_array, qkv->shared_io->key_output);
        LTX_QKV_ARRAY(value_array, qkv->shared_io->value_output);
#undef LTX_QKV_ARRAY
        if (!qkv->input_array || !qkv->query_array || !qkv->key_array ||
            !qkv->value_array) {
            qkv_fail(error, error_size, "create ANE QKV arrays: %s",
                     failure.localizedDescription.UTF8String);
            ltx_ane_qkv_free(qkv);
            return NULL;
        }
        qkv->provider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{
                @"x": [MLFeatureValue featureValueWithMultiArray:
                        qkv->input_array],
            } error:&failure];
        qkv->options = [MLPredictionOptions new];
        qkv->options.outputBackings = @{
            @"query": qkv->query_array,
            @"key": qkv->key_array,
            @"value": qkv->value_array,
        };
        if (!qkv->provider) {
            qkv_fail(error, error_size, "create ANE QKV provider: %s",
                     failure.localizedDescription.UTF8String);
            ltx_ane_qkv_free(qkv);
            return NULL;
        }
        return qkv;
    }
}

void ltx_ane_qkv_free(ltx_ane_qkv *qkv) {
    if (!qkv) return;
    if (qkv->inflight) pthread_join(qkv->thread, NULL);
    @autoreleasepool {
        qkv->model = nil;
        qkv->provider = nil;
        qkv->options = nil;
        qkv->input_array = nil;
        qkv->query_array = nil;
        qkv->key_array = nil;
        qkv->value_array = nil;
        if (qkv->temporary_compiled)
            [[NSFileManager defaultManager]
                removeItemAtURL:qkv->temporary_compiled error:nil];
        qkv->temporary_compiled = nil;
    }
    qkv_io_release(qkv->shared_io);
    free(qkv);
}

const ltx_ane_qkv_shape *ltx_ane_qkv_get_shape(const ltx_ane_qkv *qkv) {
    return qkv ? &qkv->shape : NULL;
}

int ltx_ane_qkv_start(ltx_ane_qkv *qkv, ltx_gpu *gpu,
                      const ltx_gpu_buffer *input,
                      uint32_t input_rows, uint32_t start_row,
                      char *error, size_t error_size) {
    if (!qkv || !gpu || !input || qkv->inflight ||
        start_row >= input_rows || qkv->shape.rows > input_rows - start_row) {
        qkv_fail(error, error_size, "invalid ANE QKV start arguments");
        return 0;
    }
    qkv->started_ms = qkv_now_ms();
    double pack_start = qkv->started_ms;
    if (!ltx_gpu_slice_rows_bf16_f16(
            gpu, qkv->shared_io->input, input,
            input_rows, qkv->shape.hidden, start_row, qkv->shape.rows,
            error, error_size)) return 0;
    qkv->pack_ms = qkv_now_ms() - pack_start;
    qkv->async_ok = 0;
    qkv->async_ms = 0.0;
    qkv->query_output_backing_used = 0;
    qkv->key_output_backing_used = 0;
    qkv->value_output_backing_used = 0;
    qkv->inflight = 1;
    if (pthread_create(&qkv->thread, NULL, qkv_prediction_thread, qkv) != 0) {
        qkv->inflight = 0;
        qkv_fail(error, error_size,
                 "cannot create ANE QKV prediction thread");
        return 0;
    }
    return 1;
}

int ltx_ane_qkv_wait(ltx_ane_qkv *qkv, ltx_gpu *gpu,
                     ltx_gpu_buffer *query_output,
                     ltx_gpu_buffer *key_output,
                     ltx_gpu_buffer *value_output,
                     const ltx_gpu_buffer *query_prefix,
                     const ltx_gpu_buffer *key_prefix,
                     const ltx_gpu_buffer *value_prefix,
                     uint32_t prefix_rows,
                     ltx_ane_qkv_timing *timing,
                     char *error, size_t error_size) {
    if (!qkv || !gpu || !query_output || !key_output || !value_output ||
        !query_prefix || !key_prefix || !value_prefix || !prefix_rows ||
        !qkv->inflight) {
        qkv_fail(error, error_size, "invalid ANE QKV wait arguments");
        return 0;
    }
    int join_ok = pthread_join(qkv->thread, NULL) == 0;
    qkv->inflight = 0;
    if (!join_ok) {
        qkv_fail(error, error_size, "cannot join ANE QKV thread");
        return 0;
    }
    if (!qkv->async_ok) {
        qkv_fail(error, error_size, "%s", qkv->async_error[0] ?
                 qkv->async_error : "ANE QKV prediction failed");
        return 0;
    }
    double unpack_start = qkv_now_ms();
    if (!ltx_gpu_batch_begin(gpu, error, error_size)) return 0;
    int ok = ltx_gpu_concat_rows_bf16_f16(
            gpu, query_output, query_prefix, qkv->shared_io->query_output,
            prefix_rows, qkv->shape.rows, qkv->shape.hidden,
            error, error_size) &&
        ltx_gpu_concat_rows_bf16_f16(
            gpu, key_output, key_prefix, qkv->shared_io->key_output,
            prefix_rows, qkv->shape.rows, qkv->shape.hidden,
            error, error_size) &&
        ltx_gpu_concat_rows_bf16_f16(
            gpu, value_output, value_prefix, qkv->shared_io->value_output,
            prefix_rows, qkv->shape.rows, qkv->shape.hidden,
            error, error_size);
    char batch_error[1024] = {0};
    int batch_ok = ltx_gpu_batch_end(
        gpu, batch_error, sizeof(batch_error));
    if (!batch_ok && ok)
        snprintf(error, error_size, "%s",
                 batch_error[0] ? batch_error : "ANE QKV concat failed");
    if (!ok || !batch_ok) return 0;
    ltx_ane_qkv_timing measured = {
        .pack_ms = qkv->pack_ms,
        .ane_ms = qkv->async_ms,
        .unpack_ms = qkv_now_ms() - unpack_start,
        .total_ms = qkv_now_ms() - qkv->started_ms,
        .query_output_backing_used = qkv->query_output_backing_used,
        .key_output_backing_used = qkv->key_output_backing_used,
        .value_output_backing_used = qkv->value_output_backing_used,
    };
    if (timing) *timing = measured;
    return 1;
}

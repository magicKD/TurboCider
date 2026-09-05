#include "ltx_ane_mlp.h"
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

typedef struct ltx_ane_mlp_io {
    ltx_gpu *gpu;
    uint32_t rows;
    uint32_t hidden;
    unsigned references;
    ltx_gpu_buffer *gpu_partial;
    ltx_gpu_buffer *ane_input;
    ltx_gpu_buffer *ane_output;
    struct ltx_ane_mlp_io *next;
} ltx_ane_mlp_io;

static pthread_mutex_t ane_io_mutex = PTHREAD_MUTEX_INITIALIZER;
static ltx_ane_mlp_io *ane_io_list;

static void ane_fail(char *error, size_t error_size, const char *format, ...);

struct ltx_ane_mlp {
    ltx_ane_mlp_shape shape;
    ltx_gpu_buffer *gpu_fc1_weight;
    ltx_gpu_buffer *gpu_fc1_scale;
    ltx_gpu_buffer *gpu_fc2_weight;
    ltx_gpu_buffer *gpu_fc2_scale;
    ltx_gpu_buffer *gpu_partial;
    ltx_gpu_buffer *ane_input;
    ltx_gpu_buffer *ane_output;
    ltx_ane_mlp_io *shared_io;
    MLModel *model;
    MLMultiArray *input_array;
    MLMultiArray *output_array;
    MLDictionaryFeatureProvider *provider;
    MLPredictionOptions *options;
    NSURL *temporary_compiled;
    pthread_t thread;
    int inflight;
    int async_ok;
    int output_backing_used;
    double async_ms;
    char async_error[1024];
};

static ltx_ane_mlp_io *ane_io_acquire(
        ltx_gpu *gpu, uint32_t rows, uint32_t hidden,
        char *error, size_t error_size) {
    pthread_mutex_lock(&ane_io_mutex);
    for (ltx_ane_mlp_io *io = ane_io_list; io; io = io->next)
        if (io->gpu == gpu && io->rows == rows && io->hidden == hidden) {
            io->references++;
            pthread_mutex_unlock(&ane_io_mutex);
            return io;
        }
    pthread_mutex_unlock(&ane_io_mutex);
    if ((size_t)rows > SIZE_MAX / hidden / sizeof(uint16_t)) {
        ane_fail(error, error_size, "ANE shared IO size overflow");
        return NULL;
    }
    size_t bytes = (size_t)rows * hidden * sizeof(uint16_t);
    ltx_ane_mlp_io *io = calloc(1, sizeof(*io));
    if (!io) {
        ane_fail(error, error_size, "out of memory creating ANE shared IO");
        return NULL;
    }
    io->gpu = gpu;
    io->rows = rows;
    io->hidden = hidden;
    io->references = 1u;
    io->gpu_partial = ltx_gpu_buffer_new(gpu, bytes, error, error_size);
    io->ane_input = ltx_gpu_buffer_new(gpu, bytes, error, error_size);
    io->ane_output = ltx_gpu_buffer_new(gpu, bytes, error, error_size);
    if (!io->gpu_partial || !io->ane_input || !io->ane_output) {
        ltx_gpu_buffer_free(io->gpu_partial);
        ltx_gpu_buffer_free(io->ane_input);
        ltx_gpu_buffer_free(io->ane_output);
        free(io);
        return NULL;
    }
    pthread_mutex_lock(&ane_io_mutex);
    io->next = ane_io_list;
    ane_io_list = io;
    pthread_mutex_unlock(&ane_io_mutex);
    return io;
}

static void ane_io_release(ltx_ane_mlp_io *io) {
    if (!io) return;
    int destroy = 0;
    pthread_mutex_lock(&ane_io_mutex);
    if (io->references > 0u) io->references--;
    if (io->references == 0u) {
        ltx_ane_mlp_io **cursor = &ane_io_list;
        while (*cursor && *cursor != io) cursor = &(*cursor)->next;
        if (*cursor == io) *cursor = io->next;
        destroy = 1;
    }
    pthread_mutex_unlock(&ane_io_mutex);
    if (!destroy) return;
    ltx_gpu_buffer_free(io->gpu_partial);
    ltx_gpu_buffer_free(io->ane_input);
    ltx_gpu_buffer_free(io->ane_output);
    free(io);
}

static double ane_now_ms(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return ((double)value.tv_sec + (double)value.tv_nsec * 1e-9) * 1000.0;
}

static void ane_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static NSDictionary *ane_dictionary(NSDictionary *root, NSString *key) {
    id value = root[key];
    return [value isKindOfClass:[NSDictionary class]] ? value : nil;
}

static int ane_u32_value(NSDictionary *root, NSString *key, int allow_zero,
                         uint32_t *value) {
    id number = root[key];
    if (![number isKindOfClass:[NSNumber class]]) return 0;
    unsigned long long parsed = [number unsignedLongLongValue];
    if ((!allow_zero && !parsed) || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static NSData *ane_read_file(NSURL *directory, NSString *name,
                             size_t expected, char *error,
                             size_t error_size) {
    NSURL *url = [directory URLByAppendingPathComponent:name];
    NSData *data = [NSData dataWithContentsOfURL:url
                                        options:NSDataReadingMappedIfSafe
                                          error:nil];
    if (!data || data.length != expected) {
        ane_fail(error, error_size, "%s has %llu bytes, expected %zu",
                 url.path.UTF8String,
                 (unsigned long long)(data ? data.length : 0), expected);
        return nil;
    }
    return data;
}

static ltx_gpu_buffer *ane_load_buffer(ltx_gpu *gpu, NSURL *directory,
                                       NSString *name, size_t expected,
                                       char *error, size_t error_size) {
    NSData *data = ane_read_file(directory, name, expected,
                                 error, error_size);
    return data ? ltx_gpu_buffer_new_copy(
        gpu, data.bytes, data.length, error, error_size) : NULL;
}

static int ane_shape_supports_rows(const ltx_ane_mlp_shape *shape,
                                   uint32_t rows) {
    if (!shape || !rows) return 0;
    for (uint32_t index = 0; index < shape->supported_row_count; index++)
        if (shape->supported_rows[index] == rows) return 1;
    return 0;
}

static int ane_bind_io(ltx_ane_mlp *mlp, ltx_gpu *gpu, uint32_t rows,
                       char *error, size_t error_size) {
    if (!mlp || !gpu || mlp->inflight ||
        !ane_shape_supports_rows(&mlp->shape, rows)) {
        ane_fail(error, error_size, "unsupported or busy ANE MLP row shape");
        return 0;
    }
    if (mlp->shape.rows == rows && mlp->shared_io && mlp->provider)
        return 1;
    ltx_ane_mlp_io *next_io = ane_io_acquire(
        gpu, rows, mlp->shape.hidden, error, error_size);
    if (!next_io) return 0;
    size_t output_elements = (size_t)rows * mlp->shape.hidden;
    __block MLMultiArray *input_array = nil;
    __block MLMultiArray *output_array = nil;
    __block MLDictionaryFeatureProvider *provider = nil;
    __block MLPredictionOptions *options = nil;
    @autoreleasepool {
        NSError *failure = nil;
        id<MTLBuffer> input_buffer =
            (__bridge id<MTLBuffer>)ltx_gpu_buffer_native(next_io->ane_input);
        id<MTLBuffer> output_buffer =
            (__bridge id<MTLBuffer>)ltx_gpu_buffer_native(next_io->ane_output);
        NSArray<NSNumber *> *array_shape =
            @[@1, @(mlp->shape.hidden), @1, @(rows)];
        NSArray<NSNumber *> *strides = @[
            @(output_elements), @1, @(output_elements), @(mlp->shape.hidden)
        ];
        input_array = [[MLMultiArray alloc]
            initWithDataPointer:input_buffer.contents
                          shape:array_shape
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:strides
                     deallocator:^(void *pointer) { (void)pointer; }
                          error:&failure];
        output_array = [[MLMultiArray alloc]
            initWithDataPointer:output_buffer.contents
                          shape:array_shape
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:strides
                     deallocator:^(void *pointer) { (void)pointer; }
                          error:&failure];
        if (input_array && output_array) {
            provider = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{
                    @"x": [MLFeatureValue featureValueWithMultiArray:
                            input_array]
                } error:&failure];
            options = [MLPredictionOptions new];
            options.outputBackings = @{ @"y": output_array };
        }
        if (!input_array || !output_array || !provider) {
            const char *message = failure.localizedDescription.UTF8String;
            ane_fail(error, error_size, "bind Core ML row shape %u: %s",
                     rows, message ? message : "array/provider creation failed");
        }
    }
    if (!input_array || !output_array || !provider) {
        ane_io_release(next_io);
        return 0;
    }
    ane_io_release(mlp->shared_io);
    mlp->shared_io = next_io;
    mlp->gpu_partial = next_io->gpu_partial;
    mlp->ane_input = next_io->ane_input;
    mlp->ane_output = next_io->ane_output;
    mlp->input_array = input_array;
    mlp->output_array = output_array;
    mlp->provider = provider;
    mlp->options = options;
    mlp->shape.rows = rows;
    return 1;
}

static void *ane_prediction_thread(void *opaque) {
    ltx_ane_mlp *mlp = opaque;
    mlp->async_error[0] = '\0';
    mlp->async_ok = 0;
    double started = ane_now_ms();
    @autoreleasepool {
        @try {
            NSError *failure = nil;
            id<MLFeatureProvider> result =
                [mlp->model predictionFromFeatures:mlp->provider
                                           options:mlp->options
                                             error:&failure];
            if (!result) {
                const char *message = failure.localizedDescription.UTF8String;
                snprintf(mlp->async_error, sizeof(mlp->async_error), "%s",
                         message ? message : "Core ML prediction failed");
            } else {
                MLMultiArray *returned =
                    [result featureValueForName:@"y"].multiArrayValue;
                mlp->output_backing_used =
                    returned == mlp->output_array &&
                    returned.dataPointer == mlp->output_array.dataPointer;
                if (!mlp->output_backing_used) {
                    snprintf(mlp->async_error, sizeof(mlp->async_error),
                             "Core ML rejected caller-owned output backing");
                } else {
                    mlp->async_ok = 1;
                }
            }
        } @catch (NSException *exception) {
            const char *message = exception.reason.UTF8String;
            snprintf(mlp->async_error, sizeof(mlp->async_error), "%s",
                     message ? message : "Core ML exception");
        }
    }
    mlp->async_ms = ane_now_ms() - started;
    return NULL;
}

static int ane_start(ltx_ane_mlp *mlp, char *error, size_t error_size) {
    if (!mlp || mlp->inflight) {
        ane_fail(error, error_size, "ANE MLP is missing or already active");
        return 0;
    }
    mlp->async_ok = 0;
    mlp->async_ms = 0.0;
    mlp->inflight = 1;
    if (pthread_create(&mlp->thread, NULL, ane_prediction_thread, mlp) != 0) {
        mlp->inflight = 0;
        ane_fail(error, error_size, "cannot create ANE prediction thread");
        return 0;
    }
    return 1;
}

static int ane_wait(ltx_ane_mlp *mlp, char *error, size_t error_size) {
    if (!mlp || !mlp->inflight) {
        ane_fail(error, error_size, "ANE MLP was not started");
        return 0;
    }
    pthread_join(mlp->thread, NULL);
    mlp->inflight = 0;
    if (!mlp->async_ok) {
        ane_fail(error, error_size, "%s", mlp->async_error[0] ?
                 mlp->async_error : "ANE prediction failed");
        return 0;
    }
    return 1;
}

ltx_ane_mlp *ltx_ane_mlp_create(ltx_gpu *gpu, const char *manifest_path,
                                const char *variant,
                                char *error, size_t error_size) {
    if (!gpu || !manifest_path || !*manifest_path ||
        !variant || !*variant) {
        ane_fail(error, error_size, "GPU, manifest and variant are required");
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
            ane_fail(error, error_size, "cannot parse ANE manifest: %s",
                     message ? message : "invalid JSON");
            return NULL;
        }
        NSDictionary *root = decoded;
        if (![root[@"schema"] isEqualToString:@"ltx-ane-mlp-v1"]) {
            ane_fail(error, error_size, "unsupported ANE manifest schema");
            return NULL;
        }
        NSDictionary *shape = ane_dictionary(root, @"shape");
        NSDictionary *partition = ane_dictionary(root, @"partition");
        NSDictionary *gpu_part = ane_dictionary(partition, @"gpu");
        NSDictionary *ane_part = ane_dictionary(partition, @"ane");
        ltx_ane_mlp_shape parsed = {0};
        if (!shape || !partition || !gpu_part || !ane_part ||
            !ane_u32_value(root, @"block_index", 1, &parsed.block_index) ||
            !ane_u32_value(shape, @"rows", 0, &parsed.rows) ||
            !ane_u32_value(shape, @"hidden", 0, &parsed.hidden) ||
            !ane_u32_value(shape, @"intermediate", 0,
                           &parsed.full_intermediate) ||
            !ane_u32_value(gpu_part, @"width", 0,
                           &parsed.gpu_intermediate) ||
            !ane_u32_value(ane_part, @"width", 0,
                           &parsed.ane_intermediate) ||
            parsed.gpu_intermediate + parsed.ane_intermediate !=
                parsed.full_intermediate ||
            parsed.hidden != 4096u || parsed.full_intermediate != 16384u ||
            parsed.gpu_intermediate % 256u ||
            parsed.ane_intermediate % 256u) {
            ane_fail(error, error_size, "invalid LTX ANE MLP geometry");
            return NULL;
        }
        parsed.supported_rows[0] = parsed.rows;
        parsed.supported_row_count = 1u;
        id supported_value = shape[@"supported_rows"];
        if (supported_value) {
            if (![supported_value isKindOfClass:[NSArray class]] ||
                [(NSArray *)supported_value count] == 0u ||
                [(NSArray *)supported_value count] > 2u) {
                ane_fail(error, error_size,
                         "invalid LTX ANE MLP supported_rows");
                return NULL;
            }
            parsed.supported_row_count = 0u;
            for (id value in (NSArray *)supported_value) {
                if (![value isKindOfClass:[NSNumber class]]) {
                    ane_fail(error, error_size,
                             "invalid LTX ANE MLP supported row");
                    return NULL;
                }
                unsigned long long row = [value unsignedLongLongValue];
                if (!row || row > UINT32_MAX ||
                    (parsed.supported_row_count > 0u &&
                     parsed.supported_rows[0] == (uint32_t)row)) {
                    ane_fail(error, error_size,
                             "invalid LTX ANE MLP supported row value");
                    return NULL;
                }
                parsed.supported_rows[parsed.supported_row_count++] =
                    (uint32_t)row;
            }
            if (!ane_shape_supports_rows(&parsed, parsed.rows)) {
                ane_fail(error, error_size,
                         "default ANE MLP rows are not supported");
                return NULL;
            }
        }
        NSDictionary *artifacts = ane_dictionary(root, @"artifacts");
        NSDictionary *compiled = ane_dictionary(root, @"compiled_artifacts");
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
            ane_fail(error, error_size, "manifest lacks Core ML variant %s",
                     variant);
            return NULL;
        }
        NSURL *directory = [[NSURL fileURLWithPath:path]
            URLByDeletingLastPathComponent];
        NSURL *model_url = [directory URLByAppendingPathComponent:model_name];
        ltx_ane_mlp *mlp = calloc(1, sizeof(*mlp));
        if (!mlp) {
            ane_fail(error, error_size, "out of memory creating ANE MLP");
            return NULL;
        }
        mlp->shape = parsed;
        size_t fc1_weight_bytes =
            (size_t)parsed.gpu_intermediate * parsed.hidden;
        size_t fc1_scale_bytes =
            (size_t)parsed.gpu_intermediate * sizeof(float);
        size_t fc2_weight_bytes =
            (size_t)parsed.hidden * parsed.gpu_intermediate;
        size_t fc2_scale_bytes = (size_t)parsed.hidden * sizeof(float);
        mlp->gpu_fc1_weight = ane_load_buffer(
            gpu, directory, @"gpu_fc1.weight.i8", fc1_weight_bytes,
            error, error_size);
        mlp->gpu_fc1_scale = ane_load_buffer(
            gpu, directory, @"gpu_fc1.scale.f32", fc1_scale_bytes,
            error, error_size);
        mlp->gpu_fc2_weight = ane_load_buffer(
            gpu, directory, @"gpu_fc2.weight.i8", fc2_weight_bytes,
            error, error_size);
        mlp->gpu_fc2_scale = ane_load_buffer(
            gpu, directory, @"gpu_fc2.scale.f32", fc2_scale_bytes,
            error, error_size);
        if (!mlp->gpu_fc1_weight || !mlp->gpu_fc1_scale ||
            !mlp->gpu_fc2_weight || !mlp->gpu_fc2_scale) {
            ltx_ane_mlp_free(mlp);
            return NULL;
        }

        NSURL *load_url = model_url;
        if (![model_url.pathExtension isEqualToString:@"mlmodelc"]) {
            load_url = [MLModel compileModelAtURL:model_url error:&failure];
            if (!load_url) {
                ane_fail(error, error_size, "compile Core ML model: %s",
                         failure.localizedDescription.UTF8String);
                ltx_ane_mlp_free(mlp);
                return NULL;
            }
            mlp->temporary_compiled = load_url;
        }
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        mlp->model = [MLModel modelWithContentsOfURL:load_url
                                      configuration:configuration
                                              error:&failure];
        if (!mlp->model) {
            ane_fail(error, error_size, "load Core ML model: %s",
                     failure.localizedDescription.UTF8String);
            ltx_ane_mlp_free(mlp);
            return NULL;
        }
        if (!ane_bind_io(mlp, gpu, parsed.rows, error, error_size)) {
            ltx_ane_mlp_free(mlp);
            return NULL;
        }
        return mlp;
    }
}

void ltx_ane_mlp_free(ltx_ane_mlp *mlp) {
    if (!mlp) return;
    if (mlp->inflight) pthread_join(mlp->thread, NULL);
    @autoreleasepool {
        mlp->model = nil;
        mlp->provider = nil;
        mlp->options = nil;
        mlp->input_array = nil;
        mlp->output_array = nil;
        if (mlp->temporary_compiled)
            [[NSFileManager defaultManager]
                removeItemAtURL:mlp->temporary_compiled error:nil];
        mlp->temporary_compiled = nil;
    }
    ltx_gpu_buffer_free(mlp->gpu_fc1_weight);
    ltx_gpu_buffer_free(mlp->gpu_fc1_scale);
    ltx_gpu_buffer_free(mlp->gpu_fc2_weight);
    ltx_gpu_buffer_free(mlp->gpu_fc2_scale);
    ane_io_release(mlp->shared_io);
    free(mlp);
}

const ltx_ane_mlp_shape *ltx_ane_mlp_get_shape(const ltx_ane_mlp *mlp) {
    return mlp ? &mlp->shape : NULL;
}

int ltx_ane_mlp_supports_rows(const ltx_ane_mlp *mlp, uint32_t rows) {
    return mlp ? ane_shape_supports_rows(&mlp->shape, rows) : 0;
}

int ltx_ane_mlp_set_rows(ltx_ane_mlp *mlp, ltx_gpu *gpu, uint32_t rows,
                         char *error, size_t error_size) {
    return ane_bind_io(mlp, gpu, rows, error, error_size);
}

int ltx_ane_mlp_eval(ltx_ane_mlp *mlp, ltx_gpu *gpu,
                     ltx_gpu_buffer *output,
                     const ltx_gpu_buffer *input,
                     ltx_ane_mlp_timing *timing,
                     char *error, size_t error_size) {
    if (!mlp || !gpu || !output || !input) {
        ane_fail(error, error_size, "invalid ANE MLP evaluation arguments");
        return 0;
    }
    ltx_ane_mlp_timing measured = {0};
    uint32_t elements = mlp->shape.rows * mlp->shape.hidden;
    double total_started = ane_now_ms();
    double started = ane_now_ms();
    if (!ltx_gpu_cast_bf16_f16(
            gpu, mlp->ane_input, input, elements, error, error_size))
        return 0;
    measured.pack_ms = ane_now_ms() - started;

    started = ane_now_ms();
    if (!ane_start(mlp, error, error_size)) return 0;
    double gpu_started = ane_now_ms();
    int gpu_ok = ltx_gpu_mlp_int8_convrot_mps_bf16(
        gpu, mlp->gpu_partial, input,
        mlp->gpu_fc1_weight, mlp->gpu_fc1_scale, NULL,
        mlp->gpu_fc2_weight, mlp->gpu_fc2_scale, NULL,
        mlp->shape.rows, mlp->shape.hidden, mlp->shape.gpu_intermediate,
        mlp->shape.hidden, 256u, error, error_size);
    measured.gpu_ms = ane_now_ms() - gpu_started;
    int ane_ok = ane_wait(mlp, gpu_ok ? error : NULL,
                          gpu_ok ? error_size : 0);
    measured.ane_ms = mlp->async_ms;
    measured.overlap_ms = ane_now_ms() - started;
    if (!gpu_ok || !ane_ok) return 0;

    started = ane_now_ms();
    if (!ltx_gpu_join_bf16_f16(
            gpu, output, mlp->gpu_partial, mlp->ane_output, elements,
            error, error_size)) return 0;
    measured.join_ms = ane_now_ms() - started;
    measured.total_ms = ane_now_ms() - total_started;
    measured.ane_output_backing_used = mlp->output_backing_used;
    if (timing) *timing = measured;
    return 1;
}

int ltx_ane_mlp_eval_adaln(ltx_ane_mlp *mlp, ltx_gpu *gpu,
                     ltx_gpu_buffer *output,
                     ltx_gpu_buffer *normalized,
                     const ltx_gpu_buffer *input,
                     const ltx_gpu_buffer *scale,
                     const ltx_gpu_buffer *shift,
                     uint32_t rows, uint32_t columns,
                     uint32_t parameter_rows, float epsilon,
                     ltx_ane_mlp_timing *timing,
                     char *error, size_t error_size) {
    if (!mlp || !gpu || !output || !normalized || !input || !scale ||
        !shift || rows != mlp->shape.rows || columns != mlp->shape.hidden ||
        (parameter_rows != 1u && parameter_rows != rows)) {
        ane_fail(error, error_size,
                 "invalid ANE MLP AdaLN evaluation arguments");
        return 0;
    }
    ltx_ane_mlp_timing measured = {0};
    uint32_t elements = rows * columns;
    double total_started = ane_now_ms();
    double started = ane_now_ms();
    if (!ltx_gpu_adaln_bf16_f16(
            gpu, normalized, mlp->ane_input, input, scale, shift,
            rows, columns, parameter_rows, epsilon, error, error_size))
        return 0;
    measured.pack_ms = ane_now_ms() - started;

    started = ane_now_ms();
    if (!ane_start(mlp, error, error_size)) return 0;
    double gpu_started = ane_now_ms();
    int gpu_ok = ltx_gpu_mlp_int8_convrot_mps_bf16(
        gpu, mlp->gpu_partial, normalized,
        mlp->gpu_fc1_weight, mlp->gpu_fc1_scale, NULL,
        mlp->gpu_fc2_weight, mlp->gpu_fc2_scale, NULL,
        rows, columns, mlp->shape.gpu_intermediate,
        columns, 256u, error, error_size);
    measured.gpu_ms = ane_now_ms() - gpu_started;
    int ane_ok = ane_wait(mlp, gpu_ok ? error : NULL,
                          gpu_ok ? error_size : 0);
    measured.ane_ms = mlp->async_ms;
    measured.overlap_ms = ane_now_ms() - started;
    if (!gpu_ok || !ane_ok) return 0;

    started = ane_now_ms();
    if (!ltx_gpu_join_bf16_f16(
            gpu, output, mlp->gpu_partial, mlp->ane_output, elements,
            error, error_size)) return 0;
    measured.join_ms = ane_now_ms() - started;
    measured.total_ms = ane_now_ms() - total_started;
    measured.ane_output_backing_used = mlp->output_backing_used;
    if (timing) *timing = measured;
    return 1;
}

int ltx_ane_mlp_eval_residual(ltx_ane_mlp *mlp, ltx_gpu *gpu,
                     ltx_gpu_buffer *output,
                     const ltx_gpu_buffer *residual,
                     const ltx_gpu_buffer *input,
                     const ltx_gpu_buffer *gate,
                     uint32_t rows, uint32_t columns,
                     uint32_t gate_rows,
                     ltx_ane_mlp_timing *timing,
                     char *error, size_t error_size) {
    if (!mlp || !gpu || !output || !residual || !input || !gate ||
        rows != mlp->shape.rows || columns != mlp->shape.hidden ||
        (gate_rows != 1u && gate_rows != rows)) {
        ane_fail(error, error_size,
                 "invalid ANE MLP residual evaluation arguments");
        return 0;
    }
    ltx_ane_mlp_timing measured = {0};
    uint32_t elements = rows * columns;
    double total_started = ane_now_ms();
    double started = ane_now_ms();
    if (!ltx_gpu_cast_bf16_f16(
            gpu, mlp->ane_input, input, elements, error, error_size))
        return 0;
    measured.pack_ms = ane_now_ms() - started;

    started = ane_now_ms();
    if (!ane_start(mlp, error, error_size)) return 0;
    double gpu_started = ane_now_ms();
    int gpu_ok = ltx_gpu_mlp_int8_convrot_mps_bf16(
        gpu, mlp->gpu_partial, input,
        mlp->gpu_fc1_weight, mlp->gpu_fc1_scale, NULL,
        mlp->gpu_fc2_weight, mlp->gpu_fc2_scale, NULL,
        rows, columns, mlp->shape.gpu_intermediate,
        columns, 256u, error, error_size);
    measured.gpu_ms = ane_now_ms() - gpu_started;
    int ane_ok = ane_wait(mlp, gpu_ok ? error : NULL,
                          gpu_ok ? error_size : 0);
    measured.ane_ms = mlp->async_ms;
    measured.overlap_ms = ane_now_ms() - started;
    if (!gpu_ok || !ane_ok) return 0;

    started = ane_now_ms();
    if (!ltx_gpu_join_residual_gate_bf16_f16(
            gpu, output, residual, mlp->gpu_partial, mlp->ane_output, gate,
            rows, columns, gate_rows, error, error_size)) return 0;
    measured.join_ms = ane_now_ms() - started;
    measured.total_ms = ane_now_ms() - total_started;
    measured.ane_output_backing_used = mlp->output_backing_used;
    if (timing) *timing = measured;
    return 1;
}

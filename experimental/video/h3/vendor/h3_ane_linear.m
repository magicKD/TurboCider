#import "h3_ane_linear.h"

#import "h3_ane_bridge.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { H3_ANE_LINEAR_ROW_MULTIPLE = 16 };

struct h3_ane_linear_io {
    uint32_t rows;
    uint32_t plane_rows;
    uint32_t input_width;
    uint32_t output_width;
    IOSurfaceRef input_surface;
    IOSurfaceRef output_surface;
    h3_gpu_tensor *input;
    h3_gpu_tensor *output;
};

struct h3_ane_linear {
    h3_ane_model *model;
    h3_ane_linear_io *io;
    uint64_t weight_bytes;
    int blob_cache_hit;
    pthread_t thread;
    int inflight;
    int async_ok;
    char async_error[512];
};

typedef struct {
    uint8_t *data;
    size_t length;
    size_t cursor;
} linear_blob;

static void linear_fail(char *error, size_t error_size, const char *format,
                        ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

int h3_ane_linear_recipe_set(h3_ane_linear_recipe *recipe,
                             const char *weight_path, uint64_t file_offset,
                             char *error, size_t error_size) {
    if (!recipe || !weight_path || !*weight_path) {
        linear_fail(error, error_size,
                    "invalid private ANE linear weight recipe");
        return 0;
    }
    char *retained = strdup(weight_path);
    if (!retained) {
        linear_fail(error, error_size,
                    "out of memory retaining private ANE linear recipe");
        return 0;
    }
    free(recipe->weight_path);
    recipe->weight_path = retained;
    recipe->file_offset = file_offset;
    return 1;
}

int h3_ane_linear_recipe_present(const h3_ane_linear_recipe *recipe) {
    return recipe && recipe->weight_path && *recipe->weight_path;
}

void h3_ane_linear_recipe_clear(h3_ane_linear_recipe *recipe) {
    if (!recipe) return;
    free(recipe->weight_path);
    recipe->weight_path = NULL;
    recipe->file_offset = 0;
}

int h3_ane_linear_transient_requested(const char *value) {
    return value && *value && strcmp(value, "0") != 0;
}

int h3_ane_linear_available(void) {
    return h3_ane_bridge_available();
}

h3_ane_linear_io *h3_ane_linear_io_create(h3_gpu *gpu, uint32_t rows,
                                          uint32_t input_width,
                                          uint32_t output_width,
                                          char *error, size_t error_size) {
    if (!gpu || !rows || !input_width || !output_width) {
        linear_fail(error, error_size, "invalid private ANE linear IO shape");
        return NULL;
    }
    uint32_t plane_rows =
        (rows + H3_ANE_LINEAR_ROW_MULTIPLE - 1u) /
        H3_ANE_LINEAR_ROW_MULTIPLE * H3_ANE_LINEAR_ROW_MULTIPLE;
    if ((size_t)plane_rows > SIZE_MAX / input_width ||
        (size_t)plane_rows * input_width > SIZE_MAX / sizeof(float) ||
        (size_t)plane_rows > SIZE_MAX / output_width ||
        (size_t)plane_rows * output_width > SIZE_MAX / sizeof(float)) {
        linear_fail(error, error_size, "private ANE linear IO size overflow");
        return NULL;
    }
    h3_ane_linear_io *io = calloc(1, sizeof(*io));
    if (!io) {
        linear_fail(error, error_size,
                    "out of memory creating private ANE linear IO");
        return NULL;
    }
    io->rows = rows;
    io->plane_rows = plane_rows;
    io->input_width = input_width;
    io->output_width = output_width;
    size_t input_elements = (size_t)plane_rows * input_width;
    size_t output_elements = (size_t)plane_rows * output_width;
    io->input_surface = h3_ane_bridge_surface(
        input_elements * sizeof(float));
    io->output_surface = h3_ane_bridge_surface(
        output_elements * sizeof(float));
    if (!io->input_surface || !io->output_surface) {
        linear_fail(error, error_size,
                    "cannot allocate private ANE linear IOSurfaces");
        h3_ane_linear_io_free(io);
        return NULL;
    }
    void *input_base = IOSurfaceGetBaseAddress(io->input_surface);
    void *output_base = IOSurfaceGetBaseAddress(io->output_surface);
    memset(input_base, 0, input_elements * sizeof(float));
    memset(output_base, 0, output_elements * sizeof(float));
    io->input = h3_gpu_tensor_wrap_f32(gpu, input_base, input_elements);
    io->output = h3_gpu_tensor_wrap_f32(gpu, output_base, output_elements);
    if (!io->input || !io->output) {
        linear_fail(error, error_size,
                    "cannot share private ANE linear IOSurfaces: %s",
                    h3_gpu_error(gpu));
        h3_ane_linear_io_free(io);
        return NULL;
    }
    return io;
}

void h3_ane_linear_io_free(h3_ane_linear_io *io) {
    if (!io) return;
    h3_gpu_tensor_free(io->input);
    h3_gpu_tensor_free(io->output);
    if (io->input_surface) CFRelease(io->input_surface);
    if (io->output_surface) CFRelease(io->output_surface);
    free(io);
}

uint32_t h3_ane_linear_io_rows(const h3_ane_linear_io *io) {
    return io ? io->rows : 0;
}

uint32_t h3_ane_linear_io_plane_rows(const h3_ane_linear_io *io) {
    return io ? io->plane_rows : 0;
}

uint32_t h3_ane_linear_io_input_width(const h3_ane_linear_io *io) {
    return io ? io->input_width : 0;
}

uint32_t h3_ane_linear_io_output_width(const h3_ane_linear_io *io) {
    return io ? io->output_width : 0;
}

h3_gpu_tensor *h3_ane_linear_io_output(h3_ane_linear_io *io) {
    return io ? io->output : NULL;
}

static size_t blob_add(linear_blob *blob, size_t bytes) {
    size_t padded = (bytes + 63u) & ~(size_t)63u;
    if (blob->cursor > blob->length || bytes > UINT32_MAX ||
        blob->length - blob->cursor < 64u ||
        padded > blob->length - blob->cursor - 64u) return SIZE_MAX;
    size_t header = blob->cursor;
    uint8_t *chunk = blob->data + header;
    chunk[0] = 0xef;
    chunk[1] = 0xbe;
    chunk[2] = 0xad;
    chunk[3] = 0xde;
    chunk[4] = 0x01;
    uint32_t size32 = (uint32_t)bytes;
    uint32_t offset32 = (uint32_t)(header + 64u);
    memcpy(chunk + 8, &size32, sizeof(size32));
    memcpy(chunk + 16, &offset32, sizeof(offset32));
    blob->cursor = header + 64u + padded;
    return header;
}

static uint16_t *read_bf16_file(const char *path, uint64_t file_offset,
                                size_t elements, int exact_file,
                                char *error, size_t error_size) {
    if (!path || elements > SIZE_MAX / sizeof(uint16_t) ||
        file_offset > INT64_MAX) {
        linear_fail(error, error_size, "invalid BF16 linear weight size");
        return NULL;
    }
    size_t bytes = elements * sizeof(uint16_t);
    if ((uint64_t)bytes > UINT64_MAX - file_offset) {
        linear_fail(error, error_size, "BF16 linear weight range overflow");
        return NULL;
    }
    uint64_t end = file_offset + (uint64_t)bytes;
    struct stat status;
    if (stat(path, &status) != 0 || status.st_size < 0 ||
        (exact_file ? (uint64_t)status.st_size != end :
                      (uint64_t)status.st_size < end)) {
        linear_fail(error, error_size,
                    "private ANE linear weight %s does not contain byte "
                    "range [%llu, %llu)", path,
                    (unsigned long long)file_offset,
                    (unsigned long long)end);
        return NULL;
    }
    uint16_t *values = malloc(bytes);
    int descriptor = values ? open(path, O_RDONLY) : -1;
    size_t done = 0;
    while (descriptor >= 0 && done < bytes) {
        ssize_t count = pread(descriptor, (uint8_t *)values + done,
                              bytes - done,
                              (off_t)(file_offset + (uint64_t)done));
        if (count <= 0) break;
        done += (size_t)count;
    }
    int close_status = descriptor >= 0 ? close(descriptor) : -1;
    if (done != bytes || close_status != 0) {
        free(values);
        linear_fail(error, error_size,
                    "cannot read private ANE linear weight range from %s",
                    path);
        return NULL;
    }
    return values;
}

static float bf16_value(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static NSString *linear_program(uint32_t input_width, uint32_t output_width,
                                uint32_t rows, size_t weight_header) {
    NSMutableString *text = [NSMutableString string];
    [text appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\""
        "coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", "
        "\"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [text appendFormat:
        @"    func main<ios18>(tensor<fp32, [1,%u,1,%u]> x) {\n",
        input_width, rows];
    [text appendString:
        @"        string f16t = const()[name=string(\"f16t\"), val=string(\"fp16\")];\n"
         "        string f32t = const()[name=string(\"f32t\"), val=string(\"fp32\")];\n"
         "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
         "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
         "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"];
    [text appendFormat:
        @"        tensor<fp16, [1,%u,1,%u]> h = cast(dtype=f16t, x=x)[name=string(\"h\")];\n"
         "        tensor<fp16, [%u,%u,1,1]> w = const()[name=string(\"w\"), val=tensor<fp16, [%u,%u,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu)))];\n"
         "        tensor<fp16, [1,%u,1,%u]> y16 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=h)[name=string(\"y16\")];\n"
         "        tensor<fp32, [1,%u,1,%u]> y = cast(dtype=f32t, x=y16)[name=string(\"y\")];\n"
         "    } -> (y);\n}\n",
        input_width, rows, output_width, input_width,
        output_width, input_width, weight_header,
        output_width, rows, output_width, rows];
    return text;
}

static NSString *linear_program_int8(uint32_t input_width,
                                     uint32_t output_width, uint32_t rows,
                                     size_t weight_header,
                                     size_t scale_header) {
    NSMutableString *text = [NSMutableString string];
    [text appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\""
        "coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", "
        "\"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [text appendFormat:
        @"    func main<ios18>(tensor<fp32, [1,%u,1,%u]> x) {\n",
        input_width, rows];
    [text appendString:
        @"        string f16t = const()[name=string(\"f16t\"), val=string(\"fp16\")];\n"
         "        string f32t = const()[name=string(\"f32t\"), val=string(\"fp32\")];\n"
         "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
         "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
         "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"];
    [text appendFormat:
        @"        tensor<fp16, [1,%u,1,%u]> h = cast(dtype=f16t, x=x)[name=string(\"h\")];\n"
         "        tensor<fp16, [%u,%u,1,1]> w = constexpr_affine_dequantize()[axis=int32(0), name=string(\"w\"), quantized_data=tensor<int8, [%u,%u,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu))), scale=tensor<fp16, [%u]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu))), zero_point=int8(0)];\n"
         "        tensor<fp16, [1,%u,1,%u]> y16 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=h)[name=string(\"y16\")];\n"
         "        tensor<fp32, [1,%u,1,%u]> y = cast(dtype=f32t, x=y16)[name=string(\"y\")];\n"
         "    } -> (y);\n}\n",
        input_width, rows, output_width, input_width,
        output_width, input_width, weight_header, output_width, scale_header,
        output_width, rows, output_width, rows];
    return text;
}

static h3_ane_linear *create_linear(const char *name,
                                    const char *weight_path,
                                    uint64_t file_offset, int exact_file,
                                    h3_ane_linear_io *io, int int8_weights,
                                    char *error, size_t error_size) {
    if (!name || !*name || !weight_path || !io) {
        linear_fail(error, error_size, "invalid private ANE linear arguments");
        return NULL;
    }
    if ((size_t)io->output_width > SIZE_MAX / io->input_width) {
        linear_fail(error, error_size,
                    "private ANE linear weight size overflow");
        return NULL;
    }
    size_t elements = (size_t)io->output_width * io->input_width;
    size_t weight_bytes = int8_weights ? elements :
        elements * sizeof(uint16_t);
    size_t scale_bytes = int8_weights ?
        (size_t)io->output_width * sizeof(uint16_t) : 0;
    size_t total = 64u + 64u + ((weight_bytes + 63u) & ~(size_t)63u) +
        (int8_weights ? 64u + ((scale_bytes + 63u) & ~(size_t)63u) : 0u);
    size_t weight_header = 64u;
    size_t scale_header = int8_weights ?
        weight_header + 64u + ((weight_bytes + 63u) & ~(size_t)63u) : 0u;
    struct {
        uint32_t format_version;
        uint32_t input_width;
        uint32_t output_width;
        uint32_t int8_weights;
    } parameters = {1u, io->input_width, io->output_width,
                    int8_weights ? 1u : 0u};
    uint64_t key = h3_ane_blob_cache_key_begin("linear");
    h3_ane_blob_cache_key_bytes(&key, &parameters, sizeof(parameters));
    if (!h3_ane_blob_cache_key_file_range(
            &key, weight_path, file_offset,
            (uint64_t)elements * sizeof(uint16_t), error, error_size))
        return NULL;
    int blob_cache_hit = 0;
    linear_blob blob = {
        h3_ane_blob_cache_load("linear", key, total, &blob_cache_hit),
        total, total
    };
    if (blob.data && (blob.data[0] != 0x01 || blob.data[4] != 0x02)) {
        free(blob.data);
        blob.data = NULL;
        blob_cache_hit = 0;
    }
    uint16_t *source = NULL;
    if (!blob.data) {
        source = read_bf16_file(
            weight_path, file_offset, elements, exact_file, error, error_size);
        if (!source) return NULL;
        blob.data = calloc(total, 1);
        blob.cursor = 64u;
    }
    if (!blob.data) {
        free(source);
        linear_fail(error, error_size,
                    "out of memory building private ANE linear blob");
        return NULL;
    }
    if (!blob_cache_hit) {
        blob.data[0] = 0x01;
        blob.data[4] = 0x02;
        size_t built_weight_header = blob_add(&blob, weight_bytes);
        size_t built_scale_header = int8_weights ?
            blob_add(&blob, scale_bytes) : 0;
        if (built_weight_header != weight_header ||
            (int8_weights && built_scale_header != scale_header) ||
            blob.cursor != total) {
            free(source);
            free(blob.data);
            linear_fail(error, error_size,
                        "private ANE linear blob layout mismatch");
            return NULL;
        }
    }

    if (!blob_cache_hit && int8_weights) {
        int8_t *quantized = (int8_t *)(void *)(
            blob.data + weight_header + 64u);
        __fp16 *scales = (__fp16 *)(void *)(
            blob.data + scale_header + 64u);
        for (uint32_t row = 0; row < io->output_width; row++) {
            const uint16_t *source_row =
                source + (size_t)row * io->input_width;
            float peak = 0.0f;
            for (uint32_t column = 0; column < io->input_width; column++) {
                float value = fabsf(bf16_value(source_row[column]));
                if (value > peak) peak = value;
            }
            float scale = peak > 0.0f ? peak / 127.0f : 1.0f;
            scales[row] = (__fp16)scale;
            float stored_scale = (float)scales[row];
            if (!isfinite(stored_scale) || stored_scale <= 0.0f) {
                free(source);
                free(blob.data);
                linear_fail(error, error_size,
                            "private ANE INT8 linear scale overflow at row %u",
                            row);
                return NULL;
            }
            int8_t *destination =
                quantized + (size_t)row * io->input_width;
            for (uint32_t column = 0; column < io->input_width; column++) {
                long rounded = lroundf(
                    bf16_value(source_row[column]) / stored_scale);
                if (rounded > 127) rounded = 127;
                if (rounded < -127) rounded = -127;
                destination[column] = (int8_t)rounded;
            }
        }
    } else if (!blob_cache_hit) {
        __fp16 *destination = (__fp16 *)(void *)(
            blob.data + weight_header + 64u);
        for (size_t index = 0; index < elements; index++) {
            destination[index] = (__fp16)bf16_value(source[index]);
            if (!isfinite((float)destination[index])) {
                free(source);
                free(blob.data);
                linear_fail(error, error_size,
                            "private ANE linear FP16 overflow at %zu", index);
                return NULL;
            }
        }
    }
    free(source);
    if (!blob_cache_hit)
        h3_ane_blob_cache_store("linear", key, blob.data, blob.cursor);

    h3_ane_linear *linear = calloc(1, sizeof(*linear));
    if (!linear) {
        free(blob.data);
        linear_fail(error, error_size,
                    "out of memory creating private ANE linear");
        return NULL;
    }
    linear->io = io;
    linear->weight_bytes = blob.cursor;
    linear->blob_cache_hit = blob_cache_hit;
    @autoreleasepool {
        NSString *program = int8_weights ?
            linear_program_int8(io->input_width, io->output_width,
                                io->plane_rows, weight_header, scale_header) :
            linear_program(io->input_width, io->output_width,
                           io->plane_rows, weight_header);
        IOSurfaceRef inputs[] = {io->input_surface};
        linear->model = h3_ane_model_create(
            name, program.UTF8String, blob.data, blob.cursor,
            inputs, 1, io->output_surface, error, error_size);
    }
    if (!linear->model) {
        free(linear);
        return NULL;
    }
    return linear;
}

h3_ane_linear *h3_ane_linear_create_bf16_file(
                                          const char *name,
                                          const char *weight_path,
                                          h3_ane_linear_io *io,
                                          char *error, size_t error_size) {
    return create_linear(name, weight_path, 0, 1, io, 0,
                         error, error_size);
}

h3_ane_linear *h3_ane_linear_create_bf16_file_range(
                                          const char *name,
                                          const char *weight_path,
                                          uint64_t file_offset,
                                          h3_ane_linear_io *io,
                                          char *error, size_t error_size) {
    return create_linear(name, weight_path, file_offset, 0, io, 0,
                         error, error_size);
}

h3_ane_linear *h3_ane_linear_create_int8_file(
                                          const char *name,
                                          const char *weight_path,
                                          h3_ane_linear_io *io,
                                          char *error, size_t error_size) {
    return create_linear(name, weight_path, 0, 1, io, 1,
                         error, error_size);
}

h3_ane_linear *h3_ane_linear_create_int8_file_range(
                                          const char *name,
                                          const char *weight_path,
                                          uint64_t file_offset,
                                          h3_ane_linear_io *io,
                                          char *error, size_t error_size) {
    return create_linear(name, weight_path, file_offset, 0, io, 1,
                         error, error_size);
}

static void *linear_eval_worker(void *opaque) {
    h3_ane_linear *linear = opaque;
    linear->async_error[0] = '\0';
    linear->async_ok = h3_ane_model_eval(
        linear->model, linear->async_error, sizeof(linear->async_error));
    return NULL;
}

void h3_ane_linear_free(h3_ane_linear *linear) {
    if (!linear) return;
    if (linear->inflight) pthread_join(linear->thread, NULL);
    h3_ane_model_free(linear->model);
    free(linear);
}

int h3_ane_linear_start(h3_ane_linear *linear,
                        char *error, size_t error_size) {
    if (!linear || linear->inflight) {
        linear_fail(error, error_size,
                    "private ANE linear is missing or already in flight");
        return 0;
    }
    linear->async_ok = 0;
    linear->async_error[0] = '\0';
    linear->inflight = 1;
    if (pthread_create(&linear->thread, NULL, linear_eval_worker, linear)) {
        linear->inflight = 0;
        linear_fail(error, error_size,
                    "cannot start private ANE linear evaluation thread");
        return 0;
    }
    return 1;
}

int h3_ane_linear_wait(h3_ane_linear *linear,
                       char *error, size_t error_size) {
    if (!linear || !linear->inflight) {
        linear_fail(error, error_size,
                    "private ANE linear evaluation was not started");
        return 0;
    }
    pthread_join(linear->thread, NULL);
    linear->inflight = 0;
    if (!linear->async_ok) {
        linear_fail(error, error_size, "%s",
                    linear->async_error[0] ? linear->async_error :
                    "private ANE linear evaluation failed");
        return 0;
    }
    return 1;
}

int h3_ane_linear_inflight(const h3_ane_linear *linear) {
    return linear && linear->inflight;
}

int h3_ane_linear_unload(h3_ane_linear *linear,
                         char *error, size_t error_size) {
    if (!linear || linear->inflight) {
        linear_fail(error, error_size,
                    "cannot unload an active private ANE linear");
        return 0;
    }
    return h3_ane_model_unload(linear->model, error, error_size);
}

int h3_ane_linear_reload(h3_ane_linear *linear,
                         char *error, size_t error_size) {
    if (!linear || linear->inflight) {
        linear_fail(error, error_size,
                    "cannot reload an active private ANE linear");
        return 0;
    }
    return h3_ane_model_reload(linear->model, error, error_size);
}

int h3_ane_linear_is_loaded(const h3_ane_linear *linear) {
    return linear && h3_ane_model_is_loaded(linear->model);
}

double h3_ane_linear_compile_seconds(const h3_ane_linear *linear) {
    return linear ? h3_ane_model_compile_seconds(linear->model) : 0.0;
}

int h3_ane_linear_cache_hit(const h3_ane_linear *linear) {
    return linear && h3_ane_model_cache_hit(linear->model);
}

int h3_ane_linear_blob_cache_hit(const h3_ane_linear *linear) {
    return linear && linear->blob_cache_hit;
}

uint64_t h3_ane_linear_weight_bytes(const h3_ane_linear *linear) {
    return linear ? linear->weight_bytes : 0;
}

int h3_ane_linear_pack(h3_ane_linear *linear, h3_gpu *gpu,
                       const h3_gpu_tensor *input,
                       char *error, size_t error_size) {
    if (!linear || !gpu || !input ||
        !h3_gpu_pack_ane_input_bf16(
            gpu, linear->io->input, input, linear->io->rows,
            linear->io->input_width, 0, linear->io->input_width,
            linear->io->plane_rows) || !h3_gpu_submit(gpu)) {
        linear_fail(error, error_size,
                    "cannot pack private ANE linear input: %s",
                    gpu ? h3_gpu_error(gpu) : "missing GPU");
        return 0;
    }
    return 1;
}

#import "h3_ane_mlp.h"

#import "h3_ane_bridge.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum {
    H3_ANE_MLP_ROW_MULTIPLE = 16,
    H3_ANE_MLP_RANGE_WIDTH = 16
};

struct h3_ane_mlp_io {
    uint32_t rows;
    uint32_t plane_rows;
    uint32_t hidden;
    IOSurfaceRef input_surface;
    IOSurfaceRef range_surface;
    IOSurfaceRef output_surface;
    h3_gpu_tensor *input;
    h3_gpu_tensor *output;
    uint32_t range_channels;
    float range_inverse;
};

struct h3_ane_mlp {
    h3_ane_model *model;
    h3_ane_mlp_io *io;
    uint32_t intermediate;
    uint32_t fc2_chunks;
    float output_scale;
    float runtime_scale;
    uint64_t weight_bytes;
    int blob_cache_hit;
    pthread_t thread;
    int inflight;
    int async_ok;
    char async_error[512];
};

static void mlp_fail(char *error, size_t error_size, const char *format,
                     ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int mlp_fc2_chunk_count(uint32_t intermediate, uint32_t *chunks,
                               char *error, size_t error_size) {
    const char *value = getenv("H3_PRIVATE_ANE_MLP_FC2_CHUNKS");
    if (!value || !*value) value = getenv("H3_ANE_MLP_FC2_CHUNKS");
    unsigned long parsed = 1;
    if (value && *value) {
        char *end = NULL;
        parsed = strtoul(value, &end, 10);
        if (!end || *end || (parsed != 1 && parsed != 2 && parsed != 4)) {
            mlp_fail(error, error_size,
                     "H3_PRIVATE_ANE_MLP_FC2_CHUNKS must be 1, 2, or 4");
            return 0;
        }
    }
    if (intermediate % parsed) {
        mlp_fail(error, error_size,
                 "ANE MLP intermediate %u is not divisible by %lu FC2 chunks",
                 intermediate, parsed);
        return 0;
    }
    *chunks = (uint32_t)parsed;
    return 1;
}

int h3_ane_mlp_available(void) {
    return h3_ane_bridge_available();
}

static int plan_u32(NSDictionary *manifest, NSString *key, int allow_zero,
                    uint32_t *value) {
    id number = manifest[key];
    if (![number isKindOfClass:[NSNumber class]]) return 0;
    unsigned long long parsed = [number unsignedLongLongValue];
    if ((!allow_zero && !parsed) || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static int plan_file_size(NSString *path, size_t expected) {
    NSDictionary *attributes = [[NSFileManager defaultManager]
        attributesOfItemAtPath:path error:nil];
    NSNumber *size = attributes[NSFileSize];
    return size && size.unsignedLongLongValue == (unsigned long long)expected;
}

h3_ane_mlp_io *h3_ane_mlp_io_create(h3_gpu *gpu, uint32_t rows,
                                    uint32_t hidden,
                                    char *error, size_t error_size) {
    if (!gpu || !rows || !hidden) {
        mlp_fail(error, error_size, "invalid private ANE MLP IO shape");
        return NULL;
    }
    uint32_t plane_rows = (rows + H3_ANE_MLP_ROW_MULTIPLE - 1u) /
        H3_ANE_MLP_ROW_MULTIPLE * H3_ANE_MLP_ROW_MULTIPLE;
    if ((size_t)plane_rows > SIZE_MAX / hidden ||
        (size_t)plane_rows * hidden > SIZE_MAX / sizeof(float)) {
        mlp_fail(error, error_size, "private ANE MLP IO size overflow");
        return NULL;
    }
    h3_ane_mlp_io *io = calloc(1, sizeof(*io));
    if (!io) {
        mlp_fail(error, error_size, "out of memory creating ANE MLP IO");
        return NULL;
    }
    io->rows = rows;
    io->plane_rows = plane_rows;
    io->hidden = hidden;
    size_t elements = (size_t)plane_rows * hidden;
    size_t bytes = elements * sizeof(float);
    io->input_surface = h3_ane_bridge_surface(bytes);
    io->output_surface = h3_ane_bridge_surface(bytes);
    if (!io->input_surface || !io->output_surface) {
        mlp_fail(error, error_size, "cannot allocate ANE MLP IOSurfaces");
        h3_ane_mlp_io_free(io);
        return NULL;
    }
    void *input_base = IOSurfaceGetBaseAddress(io->input_surface);
    void *output_base = IOSurfaceGetBaseAddress(io->output_surface);
    memset(input_base, 0, bytes);
    memset(output_base, 0, bytes);
    io->input = h3_gpu_tensor_wrap_f32(gpu, input_base, elements);
    io->output = h3_gpu_tensor_wrap_f32(gpu, output_base, elements);
    if (!io->input || !io->output) {
        mlp_fail(error, error_size, "cannot share ANE MLP IOSurfaces: %s",
                 h3_gpu_error(gpu));
        h3_ane_mlp_io_free(io);
        return NULL;
    }
    return io;
}

void h3_ane_mlp_io_free(h3_ane_mlp_io *io) {
    if (!io) return;
    h3_gpu_tensor_free(io->input);
    h3_gpu_tensor_free(io->output);
    if (io->input_surface) CFRelease(io->input_surface);
    if (io->range_surface) CFRelease(io->range_surface);
    if (io->output_surface) CFRelease(io->output_surface);
    free(io);
}

uint32_t h3_ane_mlp_io_rows(const h3_ane_mlp_io *io) {
    return io ? io->rows : 0;
}

uint32_t h3_ane_mlp_io_plane_rows(const h3_ane_mlp_io *io) {
    return io ? io->plane_rows : 0;
}

uint32_t h3_ane_mlp_io_hidden(const h3_ane_mlp_io *io) {
    return io ? io->hidden : 0;
}

h3_gpu_tensor *h3_ane_mlp_io_input(h3_ane_mlp_io *io) {
    return io ? io->input : NULL;
}

h3_gpu_tensor *h3_ane_mlp_io_output(h3_ane_mlp_io *io) {
    return io ? io->output : NULL;
}

typedef struct {
    uint8_t *data;
    size_t length;
    size_t cursor;
} mlp_blob;

static size_t blob_add(mlp_blob *blob, size_t bytes) {
    size_t padded = (bytes + 63u) & ~(size_t)63u;
    if (blob->cursor > blob->length || bytes > UINT32_MAX ||
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

static uint16_t *read_bf16_file(const char *path, size_t elements,
                                char *error, size_t error_size) {
    if (!path || elements > SIZE_MAX / sizeof(uint16_t)) {
        mlp_fail(error, error_size, "invalid BF16 weight path or size");
        return NULL;
    }
    struct stat status;
    size_t bytes = elements * sizeof(uint16_t);
    if (stat(path, &status) != 0 || status.st_size < 0 ||
        (uint64_t)status.st_size != (uint64_t)bytes) {
        mlp_fail(error, error_size,
                 "ANE BF16 weight %s does not have %zu bytes", path, bytes);
        return NULL;
    }
    uint16_t *values = malloc(bytes);
    FILE *stream = values ? fopen(path, "rb") : NULL;
    if (!stream || fread(values, sizeof(*values), elements, stream) !=
                       elements || fclose(stream) != 0) {
        if (stream) fclose(stream);
        free(values);
        mlp_fail(error, error_size, "cannot read ANE BF16 weight %s", path);
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

static int pack_weights(mlp_blob *blob, const uint16_t *fc1,
                        const uint16_t *fc2, uint32_t hidden,
                        uint32_t intermediate, float up_scale,
                        float output_scale, size_t *fc1_header,
                        size_t *fc2_header,
                        char *error, size_t error_size) {
    size_t fc1_elements = (size_t)intermediate * 2u * hidden;
    size_t fc2_elements = (size_t)hidden * intermediate;
    *fc1_header = blob_add(blob, fc1_elements * sizeof(uint16_t));
    *fc2_header = blob_add(blob, fc2_elements * sizeof(uint16_t));
    if (*fc1_header == SIZE_MAX || *fc2_header == SIZE_MAX) {
        mlp_fail(error, error_size, "ANE MLP weight blob overflow");
        return 0;
    }
    __fp16 *fc1_out = (__fp16 *)(void *)(blob->data + *fc1_header + 64u);
    __fp16 *fc2_out = (__fp16 *)(void *)(blob->data + *fc2_header + 64u);
    size_t gate_elements = (size_t)intermediate * hidden;
    for (size_t index = 0; index < fc1_elements; index++) {
        float scale = index >= gate_elements ? 1.0f / up_scale : 1.0f;
        float value = bf16_value(fc1[index]) * scale;
        fc1_out[index] = (__fp16)value;
        if (!isfinite((float)fc1_out[index])) {
            mlp_fail(error, error_size,
                     "ANE MLP FC1 FP16 conversion overflow at %zu", index);
            return 0;
        }
    }
    float down_scale = up_scale / output_scale;
    for (size_t index = 0; index < fc2_elements; index++) {
        float value = bf16_value(fc2[index]) * down_scale;
        fc2_out[index] = (__fp16)value;
        if (!isfinite((float)fc2_out[index])) {
            mlp_fail(error, error_size,
                     "ANE MLP FC2 FP16 conversion overflow at %zu", index);
            return 0;
        }
    }
    return 1;
}

static int pack_int8_matrix(mlp_blob *blob, const uint16_t *source,
                            uint32_t rows, uint32_t columns,
                            uint32_t split_row, float first_scale,
                            float second_scale, size_t *weight_header,
                            size_t *scale_header,
                            char *error, size_t error_size) {
    size_t elements = (size_t)rows * columns;
    *weight_header = blob_add(blob, elements);
    *scale_header = blob_add(blob, (size_t)rows * sizeof(uint16_t));
    if (*weight_header == SIZE_MAX || *scale_header == SIZE_MAX) {
        mlp_fail(error, error_size, "ANE INT8 MLP weight blob overflow");
        return 0;
    }
    int8_t *quantized = (int8_t *)(void *)(
        blob->data + *weight_header + 64u);
    __fp16 *scales = (__fp16 *)(void *)(
        blob->data + *scale_header + 64u);
    for (uint32_t row = 0; row < rows; row++) {
        float algebraic_scale = row < split_row ? first_scale : second_scale;
        float peak = 0.0f;
        const uint16_t *source_row = source + (size_t)row * columns;
        for (uint32_t column = 0; column < columns; column++) {
            float value = fabsf(bf16_value(source_row[column]) *
                                algebraic_scale);
            if (value > peak) peak = value;
        }
        float scale = peak > 0.0f ? peak / 127.0f : 1.0f;
        scales[row] = (__fp16)scale;
        float stored_scale = (float)scales[row];
        if (!isfinite(stored_scale) || stored_scale <= 0.0f) {
            mlp_fail(error, error_size,
                     "ANE INT8 MLP scale overflow at row %u", row);
            return 0;
        }
        int8_t *destination = quantized + (size_t)row * columns;
        for (uint32_t column = 0; column < columns; column++) {
            float value = bf16_value(source_row[column]) * algebraic_scale;
            long rounded = lroundf(value / stored_scale);
            if (rounded > 127) rounded = 127;
            if (rounded < -127) rounded = -127;
            destination[column] = (int8_t)rounded;
        }
    }
    return 1;
}

static int pack_int8_weights(mlp_blob *blob, const uint16_t *fc1,
                             const uint16_t *fc2, uint32_t hidden,
                             uint32_t intermediate, float up_scale,
                             float output_scale, size_t *fc1_header,
                             size_t *fc1_scale_header, size_t *fc2_header,
                             size_t *fc2_scale_header,
                             char *error, size_t error_size) {
    return pack_int8_matrix(
               blob, fc1, intermediate * 2u, hidden, intermediate,
               1.0f, 1.0f / up_scale, fc1_header, fc1_scale_header,
               error, error_size) &&
           pack_int8_matrix(
               blob, fc2, hidden, intermediate, hidden,
               up_scale / output_scale, up_scale / output_scale,
               fc2_header, fc2_scale_header, error, error_size);
}

static int mlp_io_prepare_range(h3_ane_mlp_io *io, uint32_t channels,
                                char *error, size_t error_size) {
    if (io->range_surface) {
        if (io->range_channels != channels) {
            mlp_fail(error, error_size,
                     "shared ANE MLP range width is %u, requested %u",
                     io->range_channels, channels);
            return 0;
        }
        return 1;
    }
    if ((size_t)channels > SIZE_MAX / H3_ANE_MLP_RANGE_WIDTH /
                               sizeof(float)) {
        mlp_fail(error, error_size, "ANE MLP range surface size overflow");
        return 0;
    }
    size_t elements = (size_t)channels * H3_ANE_MLP_RANGE_WIDTH;
    io->range_surface = h3_ane_bridge_surface(elements * sizeof(float));
    if (!io->range_surface) {
        mlp_fail(error, error_size, "cannot allocate ANE MLP range surface");
        return 0;
    }
    float *range = IOSurfaceGetBaseAddress(io->range_surface);
    for (size_t index = 0; index < elements; index++) range[index] = 1.0f;
    io->range_channels = channels;
    io->range_inverse = 1.0f;
    return 1;
}

static NSString *mlp_program(uint32_t hidden, uint32_t intermediate,
                             uint32_t rows, size_t fc1_header,
                             size_t fc2_header, uint32_t fc2_chunks) {
    NSMutableString *text = [NSMutableString string];
    [text appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\""
        "coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", "
        "\"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [text appendFormat:@"    func main<ios18>(tensor<fp32, [1,%u,1,%u]> i0_x, tensor<fp32, [1,%u,1,%u]> i1_range) {\n",
                       hidden, rows, intermediate,
                       (uint32_t)H3_ANE_MLP_RANGE_WIDTH];
    [text appendString:
        @"        string f16t = const()[name=string(\"f16t\"), val=string(\"fp16\")];\n"
         "        string f32t = const()[name=string(\"f32t\"), val=string(\"fp32\")];\n"
         "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
         "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
         "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"];
    [text appendFormat:
        @"        tensor<fp16, [1,%u,1,%u]> h = cast(dtype=f16t, x=i0_x)[name=string(\"h\")];\n"
         "        tensor<fp16, [1,%u,1,%u]> inv_rt = cast(dtype=f16t, x=i1_range)[name=string(\"inv_rt\")];\n"
         "        tensor<int32, [4]> rb = const()[name=string(\"rb\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [4]> rs = const()[name=string(\"rs\"), val=tensor<int32, [4]>([1,%u,1,1])];\n"
         "        tensor<fp16, [1,%u,1,1]> inv_r = slice_by_size(x=inv_rt, begin=rb, size=rs)[name=string(\"inv_r\")];\n"
         "        tensor<fp16, [%u,%u,1,1]> w1 = const()[name=string(\"w1\"), val=tensor<fp16, [%u,%u,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu)))];\n"
         "        tensor<fp16, [1,%u,1,%u]> gu = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w1, x=h)[name=string(\"gu\")];\n",
        hidden, rows, intermediate, (uint32_t)H3_ANE_MLP_RANGE_WIDTH,
        intermediate, intermediate, intermediate * 2u, hidden,
        intermediate * 2u, hidden,
        fc1_header, intermediate * 2u, rows];
    [text appendString:
        @"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [text appendFormat:
        @"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%u,0,0])];\n"
         "        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%u,1,%u])];\n"
         "        tensor<fp16, [1,%u,1,%u]> gate = slice_by_size(x=gu, begin=b0, size=sz)[name=string(\"gate\")];\n"
         "        tensor<fp16, [1,%u,1,%u]> up = slice_by_size(x=gu, begin=b1, size=sz)[name=string(\"up\")];\n"
         "        tensor<fp16, [1,%u,1,%u]> sig = sigmoid(x=gate)[name=string(\"sig\")];\n"
         "        tensor<fp16, [1,%u,1,%u]> silu = mul(x=gate, y=sig)[name=string(\"silu\")];\n"
         "        tensor<fp16, [1,%u,1,%u]> up_r = mul(x=up, y=inv_r)[name=string(\"up_r\")];\n"
         "        tensor<fp16, [1,%u,1,%u]> act = mul(x=silu, y=up_r)[name=string(\"act\")];\n",
        intermediate, intermediate, rows, intermediate, rows,
        intermediate, rows, intermediate, rows, intermediate, rows,
        intermediate, rows, intermediate, rows];
    [text appendFormat:
        @"        tensor<fp16, [%u,%u,1,1]> w2 = const()[name=string(\"w2\"), val=tensor<fp16, [%u,%u,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu)))];\n",
        hidden, intermediate, hidden, intermediate, fc2_header];
    if (fc2_chunks == 1) {
        [text appendFormat:
            @"        tensor<fp16, [1,%u,1,%u]> y16 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w2, x=act)[name=string(\"y16\")];\n"
             "        tensor<fp32, [1,%u,1,%u]> y = cast(dtype=f32t, x=y16)[name=string(\"y\")];\n",
            hidden, rows, hidden, rows];
    } else {
        uint32_t chunk = intermediate / fc2_chunks;
        for (uint32_t index = 0; index < fc2_chunks; index++) {
            uint32_t offset = index * chunk;
            [text appendFormat:
                @"        tensor<int32, [4]> a_b%u = const()[name=string(\"a_b%u\"), val=tensor<int32, [4]>([0,%u,0,0])];\n"
                 "        tensor<int32, [4]> a_s%u = const()[name=string(\"a_s%u\"), val=tensor<int32, [4]>([1,%u,1,%u])];\n"
                 "        tensor<int32, [4]> w_b%u = const()[name=string(\"w_b%u\"), val=tensor<int32, [4]>([0,%u,0,0])];\n"
                 "        tensor<int32, [4]> w_s%u = const()[name=string(\"w_s%u\"), val=tensor<int32, [4]>([%u,%u,1,1])];\n"
                 "        tensor<fp16, [1,%u,1,%u]> a%u = slice_by_size(x=act, begin=a_b%u, size=a_s%u)[name=string(\"a%u\")];\n"
                 "        tensor<fp16, [%u,%u,1,1]> w2_%u = slice_by_size(x=w2, begin=w_b%u, size=w_s%u)[name=string(\"w2_%u\")];\n"
                 "        tensor<fp16, [1,%u,1,%u]> y16_%u = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w2_%u, x=a%u)[name=string(\"y16_%u\")];\n",
                index, index, offset,
                index, index, chunk, rows,
                index, index, offset,
                index, index, hidden, chunk,
                chunk, rows, index, index, index, index,
                hidden, chunk, index, index, index, index,
                hidden, rows, index, index, index, index];
        }
        if (fc2_chunks == 2) {
            [text appendFormat:
                @"        tensor<fp16, [1,%u,1,%u]> y16 = add(x=y16_0, y=y16_1)[name=string(\"y16\")];\n",
                hidden, rows];
        } else {
            [text appendFormat:
                @"        tensor<fp16, [1,%u,1,%u]> y16_01 = add(x=y16_0, y=y16_1)[name=string(\"y16_01\")];\n"
                 "        tensor<fp16, [1,%u,1,%u]> y16_23 = add(x=y16_2, y=y16_3)[name=string(\"y16_23\")];\n"
                 "        tensor<fp16, [1,%u,1,%u]> y16 = add(x=y16_01, y=y16_23)[name=string(\"y16\")];\n",
                hidden, rows, hidden, rows, hidden, rows];
        }
        [text appendFormat:
            @"        tensor<fp32, [1,%u,1,%u]> y = cast(dtype=f32t, x=y16)[name=string(\"y\")];\n",
            hidden, rows];
    }
    [text appendString:@"    } -> (y);\n}\n"];
    return text;
}

static NSString *mlp_program_int8(uint32_t hidden, uint32_t intermediate,
                                  uint32_t rows, size_t fc1_header,
                                  size_t fc1_scale_header,
                                  size_t fc2_header,
                                  size_t fc2_scale_header) {
    NSMutableString *text = [NSMutableString string];
    [text appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\""
        "coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", "
        "\"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [text appendFormat:@"    func main<ios18>(tensor<fp32, [1,%u,1,%u]> i0_x, tensor<fp32, [1,%u,1,%u]> i1_range) {\n",
                       hidden, rows, intermediate,
                       (uint32_t)H3_ANE_MLP_RANGE_WIDTH];
    [text appendString:
        @"        string f16t = const()[name=string(\"f16t\"), val=string(\"fp16\")];\n"
         "        string f32t = const()[name=string(\"f32t\"), val=string(\"fp32\")];\n"
         "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
         "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
         "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"];
    [text appendFormat:
        @"        tensor<fp16, [1,%u,1,%u]> h = cast(dtype=f16t, x=i0_x)[name=string(\"h\")];\n"
         "        tensor<fp16, [1,%u,1,%u]> inv_rt = cast(dtype=f16t, x=i1_range)[name=string(\"inv_rt\")];\n"
         "        tensor<int32, [4]> rb = const()[name=string(\"rb\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [4]> rs = const()[name=string(\"rs\"), val=tensor<int32, [4]>([1,%u,1,1])];\n"
         "        tensor<fp16, [1,%u,1,1]> inv_r = slice_by_size(x=inv_rt, begin=rb, size=rs)[name=string(\"inv_r\")];\n"
         "        tensor<fp16, [%u,%u,1,1]> w1 = constexpr_affine_dequantize()[axis=int32(0), name=string(\"w1\"), quantized_data=tensor<int8, [%u,%u,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu))), scale=tensor<fp16, [%u]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu))), zero_point=int8(0)];\n"
         "        tensor<fp16, [1,%u,1,%u]> gu = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w1, x=h)[name=string(\"gu\")];\n",
        hidden, rows, intermediate, (uint32_t)H3_ANE_MLP_RANGE_WIDTH,
        intermediate, intermediate, intermediate * 2u, hidden,
        intermediate * 2u, hidden,
        fc1_header, intermediate * 2u, fc1_scale_header,
        intermediate * 2u, rows];
    [text appendString:
        @"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [text appendFormat:
        @"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%u,0,0])];\n"
         "        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%u,1,%u])];\n"
         "        tensor<fp16, [1,%u,1,%u]> gate = slice_by_size(x=gu, begin=b0, size=sz)[name=string(\"gate\")];\n"
         "        tensor<fp16, [1,%u,1,%u]> up = slice_by_size(x=gu, begin=b1, size=sz)[name=string(\"up\")];\n"
         "        tensor<fp16, [1,%u,1,%u]> sig = sigmoid(x=gate)[name=string(\"sig\")];\n"
         "        tensor<fp16, [1,%u,1,%u]> silu = mul(x=gate, y=sig)[name=string(\"silu\")];\n"
         "        tensor<fp16, [1,%u,1,%u]> up_r = mul(x=up, y=inv_r)[name=string(\"up_r\")];\n"
         "        tensor<fp16, [1,%u,1,%u]> act = mul(x=silu, y=up_r)[name=string(\"act\")];\n",
        intermediate, intermediate, rows, intermediate, rows,
        intermediate, rows, intermediate, rows, intermediate, rows,
        intermediate, rows, intermediate, rows];
    [text appendFormat:
        @"        tensor<fp16, [%u,%u,1,1]> w2 = constexpr_affine_dequantize()[axis=int32(0), name=string(\"w2\"), quantized_data=tensor<int8, [%u,%u,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu))), scale=tensor<fp16, [%u]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu))), zero_point=int8(0)];\n"
         "        tensor<fp16, [1,%u,1,%u]> y16 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w2, x=act)[name=string(\"y16\")];\n"
         "        tensor<fp32, [1,%u,1,%u]> y = cast(dtype=f32t, x=y16)[name=string(\"y\")];\n"
         "    } -> (y);\n}\n",
        hidden, intermediate, hidden, intermediate, fc2_header, hidden,
        fc2_scale_header, hidden, rows, hidden, rows];
    return text;
}

h3_ane_mlp *h3_ane_mlp_create_bf16_files(
                                    const char *name,
                                    const char *fc1_path,
                                    const char *fc2_path,
                                    uint32_t intermediate,
                                    float up_scale, float output_scale,
                                    h3_ane_mlp_io *io,
                                    char *error, size_t error_size) {
    if (!name || !*name || !io || !intermediate ||
        !isfinite(up_scale) || up_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f) {
        mlp_fail(error, error_size, "invalid private ANE MLP arguments");
        return NULL;
    }
    uint32_t hidden = io->hidden;
    uint32_t fc2_chunks = 1;
    if (!mlp_fc2_chunk_count(intermediate, &fc2_chunks,
                             error, error_size)) return NULL;
    if ((size_t)intermediate > SIZE_MAX / 2u / hidden ||
        (size_t)hidden > SIZE_MAX / intermediate) {
        mlp_fail(error, error_size, "private ANE MLP weight size overflow");
        return NULL;
    }
    size_t fc1_elements = (size_t)intermediate * 2u * hidden;
    size_t fc2_elements = (size_t)hidden * intermediate;
    size_t fc1_bytes = fc1_elements * sizeof(uint16_t);
    size_t fc2_bytes = fc2_elements * sizeof(uint16_t);
    size_t total = 64u + 64u + ((fc1_bytes + 63u) & ~(size_t)63u) +
        64u + ((fc2_bytes + 63u) & ~(size_t)63u);
    size_t fc1_header = 64u;
    size_t fc2_header =
        fc1_header + 64u + ((fc1_bytes + 63u) & ~(size_t)63u);
    struct {
        uint32_t format_version;
        uint32_t hidden;
        uint32_t intermediate;
        uint32_t int8_weights;
        float up_scale;
        float output_scale;
    } parameters = {1u, hidden, intermediate, 0u, up_scale, output_scale};
    uint64_t key = h3_ane_blob_cache_key_begin("mlp");
    h3_ane_blob_cache_key_bytes(&key, &parameters, sizeof(parameters));
    if (!h3_ane_blob_cache_key_file_range(
            &key, fc1_path, 0, fc1_bytes, error, error_size) ||
        !h3_ane_blob_cache_key_file_range(
            &key, fc2_path, 0, fc2_bytes, error, error_size)) return NULL;
    int blob_cache_hit = 0;
    mlp_blob blob = {
        h3_ane_blob_cache_load("mlp", key, total, &blob_cache_hit),
        total, total
    };
    if (blob.data && (blob.data[0] != 0x01 || blob.data[4] != 0x02)) {
        free(blob.data);
        blob.data = NULL;
        blob_cache_hit = 0;
    }
    uint16_t *fc1 = NULL;
    uint16_t *fc2 = NULL;
    if (!blob.data) {
        fc1 = read_bf16_file(fc1_path, fc1_elements, error, error_size);
        fc2 = fc1 ? read_bf16_file(fc2_path, fc2_elements,
                                   error, error_size) : NULL;
        if (!fc1 || !fc2) {
            free(fc1);
            free(fc2);
            return NULL;
        }
        blob.data = calloc(total, 1);
        blob.cursor = 64u;
    }
    if (!blob.data) {
        free(fc1);
        free(fc2);
        mlp_fail(error, error_size, "out of memory building ANE MLP blob");
        return NULL;
    }
    int packed = 1;
    if (!blob_cache_hit) {
        blob.data[0] = 0x01;
        blob.data[4] = 0x02;
        size_t built_fc1_header = 0, built_fc2_header = 0;
        packed = pack_weights(&blob, fc1, fc2, hidden, intermediate,
                              up_scale, output_scale,
                              &built_fc1_header, &built_fc2_header,
                              error, error_size);
        if (packed && (built_fc1_header != fc1_header ||
                       built_fc2_header != fc2_header ||
                       blob.cursor != total)) {
            mlp_fail(error, error_size, "ANE MLP blob layout mismatch");
            packed = 0;
        }
    }
    free(fc1);
    free(fc2);
    if (!packed) {
        free(blob.data);
        return NULL;
    }
    if (!blob_cache_hit)
        h3_ane_blob_cache_store("mlp", key, blob.data, blob.cursor);
    if (!mlp_io_prepare_range(io, intermediate, error, error_size)) {
        free(blob.data);
        return NULL;
    }
    h3_ane_mlp *mlp = calloc(1, sizeof(*mlp));
    if (!mlp) {
        free(blob.data);
        mlp_fail(error, error_size, "out of memory creating ANE MLP");
        return NULL;
    }
    mlp->io = io;
    mlp->intermediate = intermediate;
    mlp->fc2_chunks = fc2_chunks;
    mlp->output_scale = output_scale;
    mlp->runtime_scale = 1.0f;
    mlp->weight_bytes = blob.cursor;
    mlp->blob_cache_hit = blob_cache_hit;
    @autoreleasepool {
        NSString *program = mlp_program(hidden, intermediate, io->plane_rows,
                                        fc1_header, fc2_header, fc2_chunks);
        IOSurfaceRef inputs[] = {io->input_surface, io->range_surface};
        mlp->model = h3_ane_model_create(
            name, program.UTF8String, blob.data, blob.cursor,
            inputs, 2, io->output_surface, error, error_size);
    }
    if (!mlp->model) {
        free(mlp);
        return NULL;
    }
    return mlp;
}

h3_ane_mlp *h3_ane_mlp_create_int8_files(
                                    const char *name,
                                    const char *fc1_path,
                                    const char *fc2_path,
                                    uint32_t intermediate,
                                    float up_scale, float output_scale,
                                    h3_ane_mlp_io *io,
                                    char *error, size_t error_size) {
    if (!name || !*name || !io || !intermediate ||
        !isfinite(up_scale) || up_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f) {
        mlp_fail(error, error_size, "invalid private ANE INT8 MLP arguments");
        return NULL;
    }
    uint32_t hidden = io->hidden;
    if ((size_t)intermediate > SIZE_MAX / 2u / hidden ||
        (size_t)hidden > SIZE_MAX / intermediate) {
        mlp_fail(error, error_size,
                 "private ANE INT8 MLP weight size overflow");
        return NULL;
    }
    size_t fc1_elements = (size_t)intermediate * 2u * hidden;
    size_t fc2_elements = (size_t)hidden * intermediate;
    size_t fc1_bytes = fc1_elements * sizeof(uint16_t);
    size_t fc2_bytes = fc2_elements * sizeof(uint16_t);
    size_t fc1_scale_bytes = (size_t)intermediate * 2u * sizeof(uint16_t);
    size_t fc2_scale_bytes = (size_t)hidden * sizeof(uint16_t);
    size_t total = 64u +
        64u + ((fc1_elements + 63u) & ~(size_t)63u) +
        64u + ((fc1_scale_bytes + 63u) & ~(size_t)63u) +
        64u + ((fc2_elements + 63u) & ~(size_t)63u) +
        64u + ((fc2_scale_bytes + 63u) & ~(size_t)63u);
    size_t fc1_header = 64u;
    size_t fc1_scale_header =
        fc1_header + 64u + ((fc1_elements + 63u) & ~(size_t)63u);
    size_t fc2_header = fc1_scale_header + 64u +
        ((fc1_scale_bytes + 63u) & ~(size_t)63u);
    size_t fc2_scale_header = fc2_header + 64u +
        ((fc2_elements + 63u) & ~(size_t)63u);
    struct {
        uint32_t format_version;
        uint32_t hidden;
        uint32_t intermediate;
        uint32_t int8_weights;
        float up_scale;
        float output_scale;
    } parameters = {1u, hidden, intermediate, 1u, up_scale, output_scale};
    uint64_t key = h3_ane_blob_cache_key_begin("mlp");
    h3_ane_blob_cache_key_bytes(&key, &parameters, sizeof(parameters));
    if (!h3_ane_blob_cache_key_file_range(
            &key, fc1_path, 0, fc1_bytes, error, error_size) ||
        !h3_ane_blob_cache_key_file_range(
            &key, fc2_path, 0, fc2_bytes, error, error_size)) return NULL;
    int blob_cache_hit = 0;
    mlp_blob blob = {
        h3_ane_blob_cache_load("mlp", key, total, &blob_cache_hit),
        total, total
    };
    if (blob.data && (blob.data[0] != 0x01 || blob.data[4] != 0x02)) {
        free(blob.data);
        blob.data = NULL;
        blob_cache_hit = 0;
    }
    uint16_t *fc1 = NULL;
    uint16_t *fc2 = NULL;
    if (!blob.data) {
        fc1 = read_bf16_file(fc1_path, fc1_elements, error, error_size);
        fc2 = fc1 ? read_bf16_file(fc2_path, fc2_elements,
                                   error, error_size) : NULL;
        if (!fc1 || !fc2) {
            free(fc1);
            free(fc2);
            return NULL;
        }
        blob.data = calloc(total, 1);
        blob.cursor = 64u;
    }
    if (!blob.data) {
        free(fc1);
        free(fc2);
        mlp_fail(error, error_size,
                 "out of memory building ANE INT8 MLP blob");
        return NULL;
    }
    int packed = 1;
    if (!blob_cache_hit) {
        blob.data[0] = 0x01;
        blob.data[4] = 0x02;
        size_t built_fc1_header = 0, built_fc1_scale_header = 0;
        size_t built_fc2_header = 0, built_fc2_scale_header = 0;
        packed = pack_int8_weights(
            &blob, fc1, fc2, hidden, intermediate, up_scale, output_scale,
            &built_fc1_header, &built_fc1_scale_header,
            &built_fc2_header, &built_fc2_scale_header,
            error, error_size);
        if (packed && (built_fc1_header != fc1_header ||
                       built_fc1_scale_header != fc1_scale_header ||
                       built_fc2_header != fc2_header ||
                       built_fc2_scale_header != fc2_scale_header ||
                       blob.cursor != total)) {
            mlp_fail(error, error_size, "ANE INT8 MLP blob layout mismatch");
            packed = 0;
        }
    }
    free(fc1);
    free(fc2);
    if (!packed) {
        free(blob.data);
        return NULL;
    }
    if (!blob_cache_hit)
        h3_ane_blob_cache_store("mlp", key, blob.data, blob.cursor);
    if (!mlp_io_prepare_range(io, intermediate, error, error_size)) {
        free(blob.data);
        return NULL;
    }
    h3_ane_mlp *mlp = calloc(1, sizeof(*mlp));
    if (!mlp) {
        free(blob.data);
        mlp_fail(error, error_size, "out of memory creating ANE INT8 MLP");
        return NULL;
    }
    mlp->io = io;
    mlp->intermediate = intermediate;
    mlp->fc2_chunks = 1;
    mlp->output_scale = output_scale;
    mlp->runtime_scale = 1.0f;
    mlp->weight_bytes = blob.cursor;
    mlp->blob_cache_hit = blob_cache_hit;
    @autoreleasepool {
        NSString *program = mlp_program_int8(
            hidden, intermediate, io->plane_rows, fc1_header,
            fc1_scale_header, fc2_header, fc2_scale_header);
        IOSurfaceRef inputs[] = {io->input_surface, io->range_surface};
        mlp->model = h3_ane_model_create(
            name, program.UTF8String, blob.data, blob.cursor,
            inputs, 2, io->output_surface, error, error_size);
    }
    if (!mlp->model) {
        free(mlp);
        return NULL;
    }
    return mlp;
}

static h3_ane_mlp *h3_ane_mlp_create_plan(
                                    const char *directory_path,
                                    uint32_t expected_block,
                                    uint32_t expected_rows,
                                    uint32_t expected_hidden,
                                    h3_ane_mlp_io *io,
                                    h3_ane_mlp_plan *plan,
                                    int int8_weights,
                                    char *error, size_t error_size) {
    if (!directory_path || !*directory_path || !io || !plan) {
        mlp_fail(error, error_size, "invalid private ANE plan arguments");
        return NULL;
    }
    @autoreleasepool {
        NSString *directory = @(directory_path);
        NSString *manifest_path = [directory
            stringByAppendingPathComponent:@"manifest.json"];
        NSData *data = [NSData dataWithContentsOfFile:manifest_path];
        NSError *failure = nil;
        NSDictionary *manifest = data ?
            [NSJSONSerialization JSONObjectWithData:data options:0
                                              error:&failure] : nil;
        if (![manifest isKindOfClass:[NSDictionary class]] ||
            ![manifest[@"schema"] isEqual:@"h3-private-ane-resplit-v1"]) {
            mlp_fail(error, error_size, "cannot read private ANE manifest %s%s%s",
                     manifest_path.fileSystemRepresentation,
                     failure ? ": " : "",
                     failure ? failure.localizedDescription.UTF8String : "");
            return NULL;
        }
        h3_ane_mlp_plan parsed = {0};
        if (!plan_u32(manifest, @"block_index", 1, &parsed.block_index) ||
            !plan_u32(manifest, @"rows", 0, &parsed.rows) ||
            !plan_u32(manifest, @"hidden", 0, &parsed.hidden) ||
            !plan_u32(manifest, @"ane_intermediate", 0,
                      &parsed.ane_intermediate) ||
            !plan_u32(manifest, @"gpu_intermediate", 1,
                      &parsed.gpu_intermediate) ||
            !plan_u32(manifest, @"full_intermediate", 0,
                      &parsed.full_intermediate)) {
            mlp_fail(error, error_size,
                     "private ANE manifest has invalid integer fields");
            return NULL;
        }
        parsed.up_scale = [manifest[@"up_scale"] floatValue];
        parsed.output_scale = [manifest[@"output_scale"] floatValue];
        if (parsed.block_index != expected_block ||
            parsed.rows != expected_rows ||
            parsed.hidden != expected_hidden ||
            parsed.full_intermediate !=
                parsed.ane_intermediate + parsed.gpu_intermediate ||
            !isfinite(parsed.up_scale) || parsed.up_scale <= 0.0f ||
            !isfinite(parsed.output_scale) || parsed.output_scale <= 0.0f ||
            io->rows != expected_rows || io->hidden != expected_hidden) {
            mlp_fail(error, error_size,
                     "private ANE manifest geometry/scale mismatch for block %u",
                     expected_block);
            return NULL;
        }
        NSString *ane_fc1 = [directory
            stringByAppendingPathComponent:@"ane_fc1.bf16"];
        NSString *ane_fc2 = [directory
            stringByAppendingPathComponent:@"ane_fc2.bf16"];
        NSString *gpu_fc1 = [directory
            stringByAppendingPathComponent:@"gpu_fc1.bf16"];
        NSString *gpu_fc2 = [directory
            stringByAppendingPathComponent:@"gpu_fc2.bf16"];
        size_t ane_fc1_bytes =
            (size_t)parsed.ane_intermediate * 2u * parsed.hidden * 2u;
        size_t ane_fc2_bytes =
            (size_t)parsed.hidden * parsed.ane_intermediate * 2u;
        size_t gpu_fc1_bytes =
            (size_t)parsed.gpu_intermediate * 2u * parsed.hidden * 2u;
        size_t gpu_fc2_bytes =
            (size_t)parsed.hidden * parsed.gpu_intermediate * 2u;
        if (!plan_file_size(ane_fc1, ane_fc1_bytes) ||
            !plan_file_size(ane_fc2, ane_fc2_bytes) ||
            !plan_file_size(gpu_fc1, gpu_fc1_bytes) ||
            !plan_file_size(gpu_fc2, gpu_fc2_bytes)) {
            mlp_fail(error, error_size,
                     "private ANE plan weight byte mismatch for block %u",
                     expected_block);
            return NULL;
        }
        char name[64];
        snprintf(name, sizeof(name), "h3-private-ane-block-%u", expected_block);
        h3_ane_mlp *model = int8_weights ?
            h3_ane_mlp_create_int8_files(
                name, ane_fc1.fileSystemRepresentation,
                ane_fc2.fileSystemRepresentation, parsed.ane_intermediate,
                parsed.up_scale, parsed.output_scale, io, error, error_size) :
            h3_ane_mlp_create_bf16_files(
                name, ane_fc1.fileSystemRepresentation,
                ane_fc2.fileSystemRepresentation, parsed.ane_intermediate,
                parsed.up_scale, parsed.output_scale, io, error, error_size);
        if (!model) return NULL;
        *plan = parsed;
        return model;
    }
}

h3_ane_mlp *h3_ane_mlp_create_int8_plan(
                                    const char *directory_path,
                                    uint32_t expected_block,
                                    uint32_t expected_rows,
                                    uint32_t expected_hidden,
                                    h3_ane_mlp_io *io,
                                    h3_ane_mlp_plan *plan,
                                    char *error, size_t error_size) {
    return h3_ane_mlp_create_plan(
        directory_path, expected_block, expected_rows, expected_hidden,
        io, plan, 1, error, error_size);
}

h3_ane_mlp *h3_ane_mlp_create_bf16_plan(
                                    const char *directory_path,
                                    uint32_t expected_block,
                                    uint32_t expected_rows,
                                    uint32_t expected_hidden,
                                    h3_ane_mlp_io *io,
                                    h3_ane_mlp_plan *plan,
                                    char *error, size_t error_size) {
    return h3_ane_mlp_create_plan(
        directory_path, expected_block, expected_rows, expected_hidden,
        io, plan, 0, error, error_size);
}

static void *mlp_eval_worker(void *opaque) {
    h3_ane_mlp *mlp = opaque;
    mlp->async_error[0] = '\0';
    mlp->async_ok = h3_ane_model_eval(mlp->model, mlp->async_error,
                                      sizeof(mlp->async_error));
    return NULL;
}

void h3_ane_mlp_free(h3_ane_mlp *mlp) {
    if (!mlp) return;
    if (mlp->inflight) pthread_join(mlp->thread, NULL);
    h3_ane_model_free(mlp->model);
    free(mlp);
}

int h3_ane_mlp_eval(h3_ane_mlp *mlp, char *error, size_t error_size) {
    if (!mlp || mlp->inflight) {
        mlp_fail(error, error_size, "ANE MLP is missing or already in flight");
        return 0;
    }
    return h3_ane_model_eval(mlp->model, error, error_size);
}

int h3_ane_mlp_start(h3_ane_mlp *mlp, char *error, size_t error_size) {
    if (!mlp || mlp->inflight) {
        mlp_fail(error, error_size, "ANE MLP is missing or already in flight");
        return 0;
    }
    mlp->async_ok = 0;
    mlp->async_error[0] = '\0';
    mlp->inflight = 1;
    if (pthread_create(&mlp->thread, NULL, mlp_eval_worker, mlp) != 0) {
        mlp->inflight = 0;
        mlp_fail(error, error_size, "cannot start ANE MLP evaluation thread");
        return 0;
    }
    return 1;
}

int h3_ane_mlp_wait(h3_ane_mlp *mlp, char *error, size_t error_size) {
    if (!mlp || !mlp->inflight) {
        mlp_fail(error, error_size, "ANE MLP evaluation was not started");
        return 0;
    }
    pthread_join(mlp->thread, NULL);
    mlp->inflight = 0;
    if (!mlp->async_ok) {
        mlp_fail(error, error_size, "%s",
                 mlp->async_error[0] ? mlp->async_error :
                 "ANE MLP evaluation failed");
        return 0;
    }
    return 1;
}

int h3_ane_mlp_inflight(const h3_ane_mlp *mlp) {
    return mlp && mlp->inflight;
}

int h3_ane_mlp_unload(h3_ane_mlp *mlp, char *error, size_t error_size) {
    if (!mlp || mlp->inflight) {
        mlp_fail(error, error_size, "cannot unload an active ANE MLP");
        return 0;
    }
    return h3_ane_model_unload(mlp->model, error, error_size);
}

int h3_ane_mlp_reload(h3_ane_mlp *mlp, char *error, size_t error_size) {
    if (!mlp || mlp->inflight) {
        mlp_fail(error, error_size, "cannot reload an active ANE MLP");
        return 0;
    }
    return h3_ane_model_reload(mlp->model, error, error_size);
}

int h3_ane_mlp_is_loaded(const h3_ane_mlp *mlp) {
    return mlp && h3_ane_model_is_loaded(mlp->model);
}

double h3_ane_mlp_compile_seconds(const h3_ane_mlp *mlp) {
    return mlp ? h3_ane_model_compile_seconds(mlp->model) : 0.0;
}

int h3_ane_mlp_cache_hit(const h3_ane_mlp *mlp) {
    return mlp && h3_ane_model_cache_hit(mlp->model);
}

int h3_ane_mlp_blob_cache_hit(const h3_ane_mlp *mlp) {
    return mlp && mlp->blob_cache_hit;
}

uint64_t h3_ane_mlp_weight_bytes(const h3_ane_mlp *mlp) {
    return mlp ? mlp->weight_bytes : 0;
}

uint32_t h3_ane_mlp_intermediate(const h3_ane_mlp *mlp) {
    return mlp ? mlp->intermediate : 0;
}

uint32_t h3_ane_mlp_fc2_chunks(const h3_ane_mlp *mlp) {
    return mlp ? mlp->fc2_chunks : 0;
}

float h3_ane_mlp_output_scale(const h3_ane_mlp *mlp) {
    return mlp ? mlp->output_scale : 0.0f;
}

float h3_ane_mlp_runtime_scale(const h3_ane_mlp *mlp) {
    return mlp ? mlp->runtime_scale : 0.0f;
}

float h3_ane_mlp_effective_output_scale(const h3_ane_mlp *mlp) {
    return mlp ? mlp->output_scale * mlp->runtime_scale : 0.0f;
}

static int mlp_write_runtime_scale(h3_ane_mlp *mlp,
                                   char *error, size_t error_size) {
    float effective = h3_ane_mlp_effective_output_scale(mlp);
    float inverse = 1.0f / mlp->runtime_scale;
    if (!isfinite(effective) || effective <= 0.0f ||
        !isfinite(inverse) || inverse <= 0.0f) {
        mlp_fail(error, error_size,
                 "ANE MLP runtime scale %.9g overflows output scale %.9g",
                 mlp->runtime_scale, mlp->output_scale);
        return 0;
    }
    if (mlp->io->range_inverse == inverse) return 1;
    float *surface = IOSurfaceGetBaseAddress(mlp->io->range_surface);
    if (!surface) {
        mlp_fail(error, error_size, "ANE MLP range surface is unavailable");
        return 0;
    }
    size_t elements =
        (size_t)mlp->intermediate * H3_ANE_MLP_RANGE_WIDTH;
    for (size_t index = 0; index < elements; index++) surface[index] = inverse;
    mlp->io->range_inverse = inverse;
    return 1;
}

int h3_ane_mlp_set_runtime_scale(h3_ane_mlp *mlp, float scale,
                                 char *error, size_t error_size) {
    int exponent = 0;
    float mantissa = frexpf(scale, &exponent);
    if (!mlp || mlp->inflight || !isfinite(scale) || scale < 1.0f ||
        mantissa != 0.5f) {
        mlp_fail(error, error_size,
                 "ANE MLP runtime scale must be a power of two >= 1 and the model must be idle");
        return 0;
    }
    float previous = mlp->runtime_scale;
    mlp->runtime_scale = scale;
    if (!mlp_write_runtime_scale(mlp, error, error_size)) {
        mlp->runtime_scale = previous;
        return 0;
    }
    return 1;
}

int h3_ane_mlp_pack(h3_ane_mlp *mlp, h3_gpu *gpu,
                    const h3_gpu_tensor *input,
                    char *error, size_t error_size) {
    return h3_ane_mlp_pack_rows(
        mlp, gpu, input, 0, error, error_size);
}

int h3_ane_mlp_pack_rows(h3_ane_mlp *mlp, h3_gpu *gpu,
                         const h3_gpu_tensor *input,
                         uint32_t input_row_offset,
                         char *error, size_t error_size) {
    if (!mlp || !gpu || !input ||
        !mlp_write_runtime_scale(mlp, error, error_size) ||
        !h3_gpu_pack_ane_input_bf16_rows(
            gpu, mlp->io->input, input, input_row_offset, mlp->io->rows,
            mlp->io->hidden, 0, mlp->io->hidden, mlp->io->plane_rows) ||
        !h3_gpu_submit(gpu)) {
        mlp_fail(error, error_size, "cannot pack ANE MLP input: %s",
                 gpu ? h3_gpu_error(gpu) : "missing GPU");
        return 0;
    }
    return 1;
}

int h3_ane_mlp_unpack(h3_ane_mlp *mlp, h3_gpu *gpu,
                      h3_gpu_tensor *output,
                      char *error, size_t error_size) {
    if (!mlp || !gpu || !output ||
        !h3_gpu_unpack_ane_output_bf16(
            gpu, output, mlp->io->output, mlp->io->rows, mlp->io->hidden,
            mlp->io->plane_rows,
            h3_ane_mlp_effective_output_scale(mlp))) {
        mlp_fail(error, error_size, "cannot unpack ANE MLP output: %s",
                 gpu ? h3_gpu_error(gpu) : "missing GPU");
        return 0;
    }
    return 1;
}

int h3_ane_mlp_unpack_rows_checked(h3_ane_mlp *mlp, h3_gpu *gpu,
                                   h3_gpu_tensor *output,
                                   h3_gpu_tensor *nonfinite_flag,
                                   uint32_t output_row_offset,
                                   uint32_t block_index,
                                   char *error, size_t error_size) {
    if (!mlp || !gpu || !output || !nonfinite_flag ||
        !h3_gpu_unpack_ane_output_bf16_rows_checked(
            gpu, output, mlp->io->output, nonfinite_flag,
            output_row_offset, mlp->io->rows, mlp->io->hidden,
            mlp->io->plane_rows,
            h3_ane_mlp_effective_output_scale(mlp), block_index)) {
        mlp_fail(error, error_size,
                 "cannot unpack row-split ANE MLP output: %s",
                 gpu ? h3_gpu_error(gpu) : "missing GPU");
        return 0;
    }
    return 1;
}

int h3_ane_mlp_unpack_rows_checked_range(h3_ane_mlp *mlp, h3_gpu *gpu,
                                   h3_gpu_tensor *output,
                                   h3_gpu_tensor *range_stats,
                                   uint32_t output_row_offset,
                                   uint32_t block_index,
                                   char *error, size_t error_size) {
    if (!mlp || !gpu || !output || !range_stats ||
        !h3_gpu_unpack_ane_output_bf16_rows_checked_range(
            gpu, output, mlp->io->output, range_stats,
            output_row_offset, mlp->io->rows, mlp->io->hidden,
            mlp->io->plane_rows,
            h3_ane_mlp_effective_output_scale(mlp), block_index)) {
        mlp_fail(error, error_size,
                 "cannot range-check row-split ANE MLP output: %s",
                 gpu ? h3_gpu_error(gpu) : "missing GPU");
        return 0;
    }
    return 1;
}

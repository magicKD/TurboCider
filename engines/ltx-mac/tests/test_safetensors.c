#include "ltx_safetensors.h"
#include "ltx_weights.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "test_safetensors: check failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int write_all(int descriptor, const void *data, size_t bytes) {
    const unsigned char *source = data;
    size_t completed = 0;
    while (completed < bytes) {
        ssize_t count = write(descriptor, source + completed,
                              bytes - completed);
        if (count <= 0) return 0;
        completed += (size_t)count;
    }
    return 1;
}

int main(void) {
    char path[] = "/tmp/ltx-safetensors-test-XXXXXX";
    int descriptor = mkstemp(path);
    CHECK(descriptor >= 0);
    const char *quant =
        "{\"format\":\"int8_tensorwise\",\"convrot\":true,"
        "\"convrot_groupsize\":256}";
    size_t quant_bytes = strlen(quant);
    char json[4096];
    int json_length = snprintf(
        json, sizeof(json),
        "{\"__metadata__\":{\"config\":\"{\\\"transformer\\\":"
        "{\\\"_class_name\\\":\\\"AVTransformer3DModel\\\","
        "\\\"num_layers\\\":48}}\"},"
        "\"tensor\":{\"dtype\":\"F32\",\"shape\":[2],"
        "\"data_offsets\":[0,8]},"
        "\"linear.weight\":{\"dtype\":\"I8\",\"shape\":[3,256],"
        "\"data_offsets\":[8,776]},"
        "\"linear.weight_scale\":{\"dtype\":\"F32\","
        "\"shape\":[3,1],\"data_offsets\":[776,788]},"
        "\"linear.bias\":{\"dtype\":\"BF16\",\"shape\":[3],"
        "\"data_offsets\":[788,794]},"
        "\"linear.comfy_quant\":{\"dtype\":\"U8\","
        "\"shape\":[%zu],\"data_offsets\":[794,%zu]}}",
        quant_bytes, 794u + quant_bytes);
    CHECK(json_length > 0 && (size_t)json_length < sizeof(json));
    uint64_t length = (uint64_t)json_length;
    unsigned char prefix[8];
    for (unsigned index = 0; index < 8; index++)
        prefix[index] = (unsigned char)((length >> (index * 8u)) & 0xffu);
    float payload[2] = {1.25f, -2.5f};
    int8_t linear_weight[3 * 256];
    for (unsigned index = 0; index < sizeof(linear_weight); index++)
        linear_weight[index] = (int8_t)((int)(index % 251u) - 125);
    float linear_scale[3] = {0.01f, 0.02f, 0.03f};
    uint16_t linear_bias[3] = {0x3f00u, 0xbf00u, 0x0000u};
    CHECK(write_all(descriptor, prefix, sizeof(prefix)));
    CHECK(write_all(descriptor, json, (size_t)length));
    CHECK(write_all(descriptor, payload, sizeof(payload)));
    CHECK(write_all(descriptor, linear_weight, sizeof(linear_weight)));
    CHECK(write_all(descriptor, linear_scale, sizeof(linear_scale)));
    CHECK(write_all(descriptor, linear_bias, sizeof(linear_bias)));
    CHECK(write_all(descriptor, quant, quant_bytes));
    close(descriptor);

    ltx_st_header header;
    char error[512];
    CHECK(ltx_st_read_header(path, &header, error, sizeof(error)));
    CHECK(header.tensor_count == 5u);
    CHECK(header.metadata_config != NULL);
    const ltx_st_tensor *tensor = ltx_st_find(&header, "tensor");
    CHECK(tensor != NULL);
    CHECK(tensor->dtype == LTX_DTYPE_F32);
    CHECK(tensor->ndim == 1u && tensor->shape[0] == 2u);
    float readback[2] = {0};
    CHECK(ltx_st_read_data(&header, tensor, readback, sizeof(readback),
                           error, sizeof(error)));
    CHECK(readback[0] == payload[0] && readback[1] == payload[1]);
    ltx_st_mapping mapping;
    CHECK(ltx_st_map_open(&header, &mapping, error, sizeof(error)));
    size_t mapped_bytes = 0;
    const float *mapped = ltx_st_map_tensor(
        &mapping, tensor, &mapped_bytes, error, sizeof(error));
    CHECK(mapped != NULL);
    CHECK(mapped_bytes == sizeof(payload));
    CHECK(mapped[0] == payload[0] && mapped[1] == payload[1]);
    ltx_linear_weight_info linear;
    CHECK(ltx_linear_weight_resolve(&header, &mapping, "linear", &linear,
                                    error, sizeof(error)));
    CHECK(linear.weight != NULL && linear.weight_scale != NULL);
    CHECK(linear.bias != NULL && linear.quant_metadata != NULL);
    CHECK(linear.input_dim == 256u && linear.output_dim == 3u);
    CHECK(linear.quantized_int8 && linear.convrot);
    CHECK(linear.convrot_group_size == 256u);
    ltx_st_map_close(&mapping);
    ltx_st_free_header(&header);
    unlink(path);
    puts("test_safetensors: PASS");
    return 0;
}

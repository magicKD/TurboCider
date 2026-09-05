#ifndef LTX_SAFETENSORS_H
#define LTX_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    LTX_DTYPE_UNKNOWN = 0,
    LTX_DTYPE_BOOL,
    LTX_DTYPE_I8,
    LTX_DTYPE_U8,
    LTX_DTYPE_I16,
    LTX_DTYPE_U16,
    LTX_DTYPE_F16,
    LTX_DTYPE_BF16,
    LTX_DTYPE_I32,
    LTX_DTYPE_U32,
    LTX_DTYPE_F32,
    LTX_DTYPE_I64,
    LTX_DTYPE_U64,
    LTX_DTYPE_F64
} ltx_dtype;

typedef struct {
    char *name;
    ltx_dtype dtype;
    uint32_t ndim;
    uint64_t shape[8];
    uint64_t data_begin;
    uint64_t data_end;
    uint64_t file_offset;
} ltx_st_tensor;

typedef struct {
    char *path;
    uint64_t file_size;
    uint64_t header_size;
    ltx_st_tensor *tensors;
    size_t tensor_count;
    char *metadata_config;
} ltx_st_header;

typedef struct {
    void *address;
    size_t bytes;
} ltx_st_mapping;

int ltx_st_read_header(const char *path, ltx_st_header *header,
                       char *error, size_t error_size);
void ltx_st_free_header(ltx_st_header *header);
const ltx_st_tensor *ltx_st_find(const ltx_st_header *header,
                                 const char *name);
uint64_t ltx_st_tensor_elements(const ltx_st_tensor *tensor);
size_t ltx_dtype_size(ltx_dtype dtype);
const char *ltx_dtype_name(ltx_dtype dtype);
int ltx_st_read_data(const ltx_st_header *header,
                     const ltx_st_tensor *tensor, void *data, size_t bytes,
                     char *error, size_t error_size);
int ltx_st_map_open(const ltx_st_header *header, ltx_st_mapping *mapping,
                    char *error, size_t error_size);
/* Drop clean file-backed pages after tensors have been copied to device
 * buffers. The mapping remains valid and pages will be faulted back if a
 * later tensor access is required. */
int ltx_st_map_discard(const ltx_st_mapping *mapping,
                       char *error, size_t error_size);
void ltx_st_map_close(ltx_st_mapping *mapping);
const void *ltx_st_map_tensor(const ltx_st_mapping *mapping,
                              const ltx_st_tensor *tensor,
                              size_t *bytes,
                              char *error, size_t error_size);

#endif

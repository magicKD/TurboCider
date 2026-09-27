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
    char *metadata_gemma_config;
    /* Internal cold metadata. Only read_header_fd captures this snapshot;
     * the legacy path does not add probes. This is NOT a content hash. */
    struct {
        uint32_t valid;
        uint64_t device, inode, bytes;
        int64_t modified_seconds, changed_seconds;
        uint32_t modified_nanoseconds, changed_nanoseconds;
    } snapshot;
} ltx_st_header;

typedef struct {
    void *address;
    size_t bytes;
    int descriptor;
    int descriptor_open;
} ltx_st_mapping;

int ltx_st_read_header(const char *path, ltx_st_header *header,
                       char *error, size_t error_size);
/* Parse from the supplied open file without reopening path. The fd is borrowed;
 * path is only a diagnostic label retained in the returned header. */
int ltx_st_read_header_fd(int descriptor, const char *path, ltx_st_header *header,
                         char *error, size_t error_size);
/* Check the captured identity of an fd-parsed header. Optional path must name
 * the same unchanged file. Does not reopen it or prove immutable contents. */
int ltx_st_validate_snapshot_fd(const ltx_st_header *, int descriptor,
                               const char *path, char *error, size_t error_size);
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
/* Requires a read_header_fd snapshot; validates its identity, then owns a
 * duplicate fd and mapping of the SAME open file. Close with map_close. */
int ltx_st_map_fd(const ltx_st_header *, int descriptor, ltx_st_mapping *,
                  char *error, size_t error_size);
/* Drop clean file-backed pages after tensors have been copied to device
 * buffers. The mapping remains valid and pages will be faulted back if a
 * later tensor access is required. */
int ltx_st_map_discard(const ltx_st_mapping *mapping,
                       char *error, size_t error_size);
int ltx_st_read_mapped_data(const ltx_st_mapping *mapping,
                            const ltx_st_tensor *tensor,
                            void *data, size_t bytes,
                            char *error, size_t error_size);
void ltx_st_map_close(ltx_st_mapping *mapping);
const void *ltx_st_map_tensor(const ltx_st_mapping *mapping,
                              const ltx_st_tensor *tensor,
                              size_t *bytes,
                              char *error, size_t error_size);

#endif

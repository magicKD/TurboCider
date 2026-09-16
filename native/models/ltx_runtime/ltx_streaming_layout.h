#ifndef LTX_STREAMING_LAYOUT_H
#define LTX_STREAMING_LAYOUT_H

#include "ltx_weights.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Internal construction ABI. All source pointers borrow the immutable header;
 * keep it and its opened file alive until every fill has joined. No GPU or
 * full-weight allocation is performed by describe. */
#define LTX_STREAM_BLOCKS 48u
#define LTX_STREAM_LINEAR_COUNT 28u
#define LTX_STREAM_ATTENTION_COUNT 6u
#define LTX_STREAM_TABLE_COUNT 8u
#define LTX_STREAM_MAX_FIELDS 192u
#define LTX_STREAM_READER_REVISION 1u

typedef enum {
    LTX_STREAM_LINEAR = 1,       /* object: linear 0..27; part: W/scale/bias */
    LTX_STREAM_ATTENTION_AUX,    /* object: attention 0..5; part: QN/KN/GW/GB */
    LTX_STREAM_TABLE_BASE,       /* object: table 0..7; CPU BF16 base */
    LTX_STREAM_TABLE_ROW         /* object: table 0..7; part: GPU BF16 row */
} ltx_stream_field_kind;

typedef struct {
    const ltx_st_tensor *source;
    uint64_t bytes;              /* destination capacity, not source bytes */
    uint32_t kind, object, part;
} ltx_stream_field;

typedef struct {
    uint32_t query_dim, key_value_dim, inner_dim, output_dim, heads, head_dim;
} ltx_stream_attention_geometry;

typedef struct {
    uint32_t block, field_count;
    ltx_linear_weight_info linears[LTX_STREAM_LINEAR_COUNT];
    ltx_stream_attention_geometry attentions[LTX_STREAM_ATTENTION_COUNT];
    uint32_t table_rows[LTX_STREAM_TABLE_COUNT];
    uint32_t table_columns[LTX_STREAM_TABLE_COUNT];
    ltx_stream_field fields[LTX_STREAM_MAX_FIELDS];
    uint64_t gpu_bytes, cpu_bytes, source_read_bytes, metadata_read_bytes;
    uint64_t scratch_bytes;
} ltx_stream_block_layout;

int ltx_stream_describe_block(const ltx_st_header *, const ltx_st_mapping *,
                             uint32_t block, ltx_stream_block_layout *,
                             char *error, size_t error_size);
/* Same binding layout and geometry, not necessarily the same source contents. */
int ltx_stream_blocks_compatible(const ltx_stream_block_layout *,
                                 const ltx_stream_block_layout *);

typedef int (*ltx_stream_cancel_query)(const void *);
/* All destinations are caller-owned shared/CPU writable spans, one per field.
 * TABLE_ROW copies from this fill's converted TABLE_BASE, not another read.
 * scratch is exclusive to this job (per slot in the initial C adapter).
 * No heap allocation, metadata parsing, GPU context operation, or global seek.
 * Success returns actual destination and file-read byte counts separately.
 * A failed/partial fill is not usable and must never be encoded. */
int ltx_stream_fill_block(const ltx_stream_block_layout *, const ltx_st_mapping *,
                         void *const *destinations, const size_t *capacities,
                         void *scratch, size_t scratch_bytes,
                         ltx_stream_cancel_query, const void *cancel_user,
                         uint64_t *content_bytes, uint64_t *source_read_bytes,
                         char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif

#ifndef LTX_STREAMING_SLOT_H
#define LTX_STREAMING_SLOT_H
#include "ltx_streaming_layout.h"
#include "ltx_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Owner allocates/releases; only a framework-authorized fill writer may use
 * destinations. This class owns storage, not reader safety or scheduling.
 * All Metal buffers use the existing shared-storage LTX allocator. */
typedef struct {
    const ltx_stream_block_layout *construction;
    ltx_gpu_buffer *buffers[LTX_STREAM_MAX_FIELDS];
    void *destinations[LTX_STREAM_MAX_FIELDS];
    size_t capacities[LTX_STREAM_MAX_FIELDS];
    void *scratch;
    size_t scratch_bytes;
} ltx_stream_slot;

/* Destination must be zero-initialized and empty. On failure it is empty;
 * previously allocated live slots cannot be overwritten by create. */
int ltx_stream_slot_create(ltx_gpu *, const ltx_stream_block_layout *,
                           ltx_stream_slot *, char *, size_t);
/* Caller must first join fill workers and prove all GPU readers completed. */
void ltx_stream_slot_destroy(ltx_stream_slot *);
int ltx_stream_slot_fill(ltx_stream_slot *, const ltx_stream_block_layout *,
                         const ltx_st_mapping *, ltx_stream_cancel_query,
                         const void *cancel_user, uint64_t *content_bytes,
                         uint64_t *source_read_bytes, char *, size_t);

#ifdef __cplusplus
}
#endif
#endif

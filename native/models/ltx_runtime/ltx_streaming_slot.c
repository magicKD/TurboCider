#include "ltx_streaming_slot.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ltx_stream_slot_destroy(ltx_stream_slot *slot) {
    if (!slot) return;
    for (uint32_t i = 0; i < LTX_STREAM_MAX_FIELDS; ++i) {
        if (slot->buffers[i]) ltx_gpu_buffer_free(slot->buffers[i]);
        else free(slot->destinations[i]);
    }
    free(slot->scratch);
    memset(slot, 0, sizeof(*slot));
}

int ltx_stream_slot_create(ltx_gpu *gpu, const ltx_stream_block_layout *layout,
                           ltx_stream_slot *slot, char *error, size_t size) {
    if (!gpu || !layout || !slot || slot->construction || slot->scratch ||
        !layout->field_count || layout->field_count > LTX_STREAM_MAX_FIELDS ||
        !layout->scratch_bytes || layout->scratch_bytes > SIZE_MAX) {
        if (error && size) snprintf(error, size, "invalid LTX slot construction");
        return 0;
    }
    for (uint32_t i = 0; i < LTX_STREAM_MAX_FIELDS; ++i)
        if (slot->destinations[i] || slot->buffers[i] || slot->capacities[i]) {
            if (error && size) snprintf(error, size, "LTX slot must be empty before construction");
            return 0;
        }
    slot->construction = layout;
    slot->scratch_bytes = (size_t)layout->scratch_bytes;
    slot->scratch = malloc(slot->scratch_bytes);
    if (!slot->scratch) goto failed;
    for (uint32_t i = 0; i < layout->field_count; ++i) {
        const ltx_stream_field *field = &layout->fields[i];
        if (!field->bytes || field->bytes > SIZE_MAX) goto failed;
        slot->capacities[i] = (size_t)field->bytes;
        if (field->kind == LTX_STREAM_TABLE_BASE) slot->destinations[i] = malloc((size_t)field->bytes);
        else {
            slot->buffers[i] = ltx_gpu_buffer_new_classified(
                gpu, (size_t)field->bytes, LTX_GPU_MEMORY_WEIGHTS, "ltx_stream_slot_field", error, size);
            if (slot->buffers[i]) slot->destinations[i] = ltx_gpu_buffer_contents(slot->buffers[i]);
        }
        if (!slot->destinations[i]) goto failed;
    }
    return 1;
failed:
    if (error && size && !error[0]) snprintf(error, size, "LTX slot allocation failed");
    ltx_stream_slot_destroy(slot);
    return 0;
}

int ltx_stream_slot_fill(ltx_stream_slot *slot, const ltx_stream_block_layout *layout,
                         const ltx_st_mapping *mapping, ltx_stream_cancel_query cancelled,
                         const void *user, uint64_t *content_bytes, uint64_t *read_bytes,
                         char *error, size_t size) {
    if (content_bytes) *content_bytes = 0;
    if (read_bytes) *read_bytes = 0;
    if (!slot || !ltx_stream_blocks_compatible(slot->construction, layout)) {
        if (error && size) snprintf(error, size, "LTX fill does not match fixed slot geometry");
        return 0;
    }
    return ltx_stream_fill_block(layout, mapping, slot->destinations, slot->capacities,
        slot->scratch, slot->scratch_bytes, cancelled, user, content_bytes, read_bytes, error, size);
}

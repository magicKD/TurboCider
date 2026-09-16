#include "ltx_streaming_slot.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    assert(argc == 3);
    char error[1024] = {0};
    ltx_gpu *gpu = ltx_gpu_create(argv[2], error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "Metal setup failed (not a PASS): %s\n", error);
        return 2;
    }
    ltx_st_header header = {0}; ltx_st_mapping mapping = {0};
    ltx_stream_block_layout a = {0}, b = {0}; ltx_stream_slot slot = {0};
    assert(ltx_st_read_header(argv[1], &header, error, sizeof(error)));
    assert(ltx_st_map_open(&header, &mapping, error, sizeof(error)));
    assert(ltx_stream_describe_block(&header, &mapping, 0, &a, error, sizeof(error)));
    assert(ltx_stream_describe_block(&header, &mapping, 1, &b, error, sizeof(error)));
    assert(ltx_stream_slot_create(gpu, &a, &slot, error, sizeof(error)));
    void *original[LTX_STREAM_MAX_FIELDS]; memcpy(original, slot.destinations, sizeof(original));
    assert(!ltx_stream_slot_create(gpu, &a, &slot, error, sizeof(error)));
    for (unsigned pass = 0; pass < 4; ++pass) {
        const ltx_stream_block_layout *layout = pass % 2 ? &b : &a;
        uint64_t content, read;
        assert(ltx_stream_slot_fill(&slot, layout, &mapping, NULL, NULL, &content, &read, error, sizeof(error)));
        assert(content == a.gpu_bytes + a.cpu_bytes && read == layout->source_read_bytes);
        for (unsigned i = 0; i < a.field_count; ++i) {
            assert(original[i] == slot.destinations[i]);
            if (slot.buffers[i]) {
                size_t bytes = (size_t)a.fields[i].bytes;
                unsigned char *readback = malloc(bytes); assert(readback);
                assert(ltx_gpu_buffer_read(slot.buffers[i], readback, bytes, error, sizeof(error)));
                assert(memcmp(readback, slot.destinations[i], bytes) == 0); free(readback);
            }
        }
    }
    assert(ltx_gpu_drain(gpu, error, sizeof(error)));
    ltx_stream_slot_destroy(&slot); ltx_stream_slot_destroy(&slot);
    assert(!slot.construction && !slot.scratch);
    ltx_st_map_close(&mapping); ltx_st_free_header(&header); ltx_gpu_free(gpu);
    puts("PASS Metal LTX fixed slot: real shared buffers, exact capacities, repeated refill without reallocation, cleanup");
    return 0;
}

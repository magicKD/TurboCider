#include "ltx_streaming_layout.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int never_cancel(const void *p) { (void)p; return 0; }
static int cancel_now(const void *p) { (void)p; return 1; }
typedef struct { unsigned calls, threshold; } cancellation;
static int cancel_later(const void *p) {
    cancellation *c = (cancellation *)p;
    return ++c->calls >= c->threshold;
}

static void verify(const ltx_stream_block_layout *layout, const ltx_st_mapping *mapping,
                   void *const *destinations) {
    char error[1024] = {0};
    for (uint32_t i = 0; i < layout->field_count; ++i) {
        const ltx_stream_field *f = &layout->fields[i];
        size_t n = (size_t)(f->source->data_end - f->source->data_begin);
        unsigned char *source = malloc(n);
        assert(source && ltx_st_read_mapped_data(mapping, f->source, source, n, error, sizeof(error)));
        if (f->kind == LTX_STREAM_TABLE_BASE || f->kind == LTX_STREAM_TABLE_ROW) {
            size_t offset = f->kind == LTX_STREAM_TABLE_ROW ? (size_t)f->part * (size_t)f->bytes * 2u : 0;
            for (size_t j = 0; j < f->bytes / 2u; ++j) {
                uint32_t bits;
                memcpy(&bits, source + offset + j * 4u, 4u);
                uint32_t upper = bits >> 16u, lower = bits & 0xffffu;
                if (lower > 0x8000u || (lower == 0x8000u && (upper & 1u))) ++upper;
                uint16_t got;
                memcpy(&got, (const unsigned char *)destinations[i] + j * 2u, 2u);
                assert(got == (uint16_t)upper);
            }
        } else assert(memcmp(source, destinations[i], n) == 0);
        free(source);
    }
}

int main(int argc, char **argv) {
    assert(argc == 2 || argc == 3);
    const int inspect = argc == 3 && strcmp(argv[2], "--inspect") == 0;
    char error[1024] = {0};
    ltx_st_header header = {0}; ltx_st_mapping mapping = {0};
    assert(ltx_st_read_header(argv[1], &header, error, sizeof(error)));
    assert(ltx_st_map_open(&header, &mapping, error, sizeof(error)));
    ltx_stream_block_layout a = {0}, b = {0};
    if (!ltx_stream_describe_block(&header, &mapping, 0, &a, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    uint64_t total_read = a.source_read_bytes, metadata_read = a.metadata_read_bytes;
    for (uint32_t block = 1; block < (inspect ? 48u : 2u); ++block) {
        if (!ltx_stream_describe_block(&header, &mapping, block, &b, error, sizeof(error))) {
            fprintf(stderr, "block %u: %s\n", block, error); return 1;
        }
        assert(ltx_stream_blocks_compatible(&a, &b));
        total_read += b.source_read_bytes; metadata_read += b.metadata_read_bytes;
    }
    if (inspect) {
        printf("PASS real LTX metadata: blocks=48 fields_per_block=%u gpu_bytes_per_slot=%llu cpu_bytes_per_slot=%llu scratch_bytes_per_slot=%llu all_block_read_bytes=%llu quant_metadata_read_bytes=%llu\n",
            a.field_count, (unsigned long long)a.gpu_bytes, (unsigned long long)a.cpu_bytes,
            (unsigned long long)a.scratch_bytes, (unsigned long long)total_read, (unsigned long long)metadata_read);
        ltx_st_map_close(&mapping); ltx_st_free_header(&header); return 0;
    }
    assert(a.field_count == 162u);
    assert(a.cpu_bytes == 46u * 256u * 2u && a.scratch_bytes == 9u * 256u * 4u);
    /* Duplicate source tables must have distinct CPU/GPU destination fields. */
    const ltx_st_tensor *base0 = NULL, *base1 = NULL;
    for (uint32_t i = 0; i < a.field_count; ++i) {
        if (a.fields[i].kind == LTX_STREAM_TABLE_BASE && a.fields[i].object == 0) base0 = a.fields[i].source;
        if (a.fields[i].kind == LTX_STREAM_TABLE_BASE && a.fields[i].object == 1) base1 = a.fields[i].source;
    }
    assert(base0 && base0 == base1);
    void *destinations[LTX_STREAM_MAX_FIELDS] = {0};
    size_t capacities[LTX_STREAM_MAX_FIELDS] = {0};
    for (uint32_t i = 0; i < a.field_count; ++i) {
        capacities[i] = (size_t)a.fields[i].bytes;
        destinations[i] = malloc(capacities[i]); assert(destinations[i]);
    }
    void *scratch = malloc((size_t)a.scratch_bytes); assert(scratch);
    uint64_t content = 0, read = 0;
    for (uint32_t pass = 0; pass < 4; ++pass) {
        const ltx_stream_block_layout *layout = pass % 2 ? &b : &a;
        assert(ltx_stream_fill_block(layout, &mapping, destinations, capacities, scratch,
            (size_t)a.scratch_bytes, never_cancel, NULL, &content, &read, error, sizeof(error)));
        assert(content == a.gpu_bytes + a.cpu_bytes && read == layout->source_read_bytes);
        verify(layout, &mapping, destinations);
    }
    capacities[a.field_count - 1]--;
    memset(destinations[0], 0xab, capacities[0]);
    assert(!ltx_stream_fill_block(&a, &mapping, destinations, capacities, scratch,
        (size_t)a.scratch_bytes, NULL, NULL, &content, &read, error, sizeof(error)));
    assert(content == 0 && read == 0 && ((unsigned char *)destinations[0])[0] == 0xab);
    capacities[a.field_count - 1]++;
    assert(!ltx_stream_fill_block(&a, &mapping, destinations, capacities, scratch,
        (size_t)a.scratch_bytes, cancel_now, NULL, &content, &read, error, sizeof(error)));
    assert(content == 0 && read == 0);
    cancellation cancel = {0, 5};
    assert(!ltx_stream_fill_block(&a, &mapping, destinations, capacities, scratch,
        (size_t)a.scratch_bytes, cancel_later, &cancel, &content, &read, error, sizeof(error)));
    assert(content == 0 && read > 0 && strstr(error, "cancelled"));
    FILE *empty = tmpfile(); assert(empty);
    ltx_st_mapping short_file = mapping; short_file.descriptor = fileno(empty);
    assert(!ltx_stream_fill_block(&a, &short_file, destinations, capacities, scratch,
        (size_t)a.scratch_bytes, NULL, NULL, &content, &read, error, sizeof(error)));
    assert(strstr(error, "EOF") && content == 0); fclose(empty);
    b.attentions[0].heads++;
    assert(!ltx_stream_blocks_compatible(&a, &b));
    ltx_st_tensor *weight = (ltx_st_tensor *)a.linears[0].weight;
    uint64_t old = weight->file_offset; weight->file_offset = UINT64_MAX;
    assert(!ltx_stream_describe_block(&header, &mapping, 0, &b, error, sizeof(error)) && b.field_count == 0);
    weight->file_offset = old;
    old = weight->shape[0]; weight->shape[0] = UINT64_MAX;
    assert(!ltx_stream_describe_block(&header, &mapping, 0, &b, error, sizeof(error)) && b.field_count == 0);
    weight->shape[0] = old;
    assert(!ltx_stream_describe_block(&header, &mapping, 48, &b, error, sizeof(error)) && b.field_count == 0);
    for (uint32_t i = 0; i < a.field_count; ++i) free(destinations[i]);
    free(scratch); ltx_st_map_close(&mapping); ltx_st_free_header(&header);
    puts("PASS LTX metadata and fixed-span fill: geometry, copies, BF16 parity, refill, cancellation, short read, range/shape failures");
    return 0;
}

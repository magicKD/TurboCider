/* Included after the native buffer implementation. Queue attachment is weak;
 * block_workspace owns this object and its Metal storage for one stage. */
struct ltx_gpu_stage_workspace {
    ltx_gpu *owner;
    void *heap;
    ltx_gpu_buffer *hidden;
    ltx_gpu_buffer *query;
    ltx_gpu_buffer *key;
    ltx_gpu_buffer *value;
    ltx_gpu_buffer *core;
    ltx_gpu_buffer *codes;
    ltx_gpu_buffer *scales;
    uint32_t rows;
    size_t allocated_bytes;
};

#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
static unsigned ltx_stage_failure_at, ltx_stage_allocation_index;
void ltx_gpu_stage_workspace_test_fail_at(unsigned allocation) {
    ltx_stage_failure_at = allocation;
    ltx_stage_allocation_index = 0;
}
static int ltx_stage_allocation_allowed(void) {
    return !ltx_stage_failure_at || ++ltx_stage_allocation_index != ltx_stage_failure_at;
}
#else
#define ltx_stage_allocation_allowed() 1
#endif

static int ltx_gpu_buffers_overlap(const ltx_gpu_buffer *left,
                                    const ltx_gpu_buffer *right) {
    if (!left || !right) return 0;
    if (left == right) return 1;
    id<MTLBuffer> a = ltx_buffer(left), b = ltx_buffer(right);
    if (a == b) return 1;
    id<MTLHeap> heap = a.heap;
    if (!heap || heap != b.heap) return 0;
    const NSUInteger x = a.heapOffset, y = b.heapOffset;
    return x <= y ? y - x < left->bytes : x - y < right->bytes;
}

static void ltx_gpu_stage_workspace_detach_gpu(ltx_gpu *gpu) {
    if (!gpu || !gpu->stage_workspace) return;
    gpu->stage_workspace->owner = NULL;
    gpu->stage_workspace = NULL;
}

static ltx_gpu_buffer *ltx_stage_wrap_heap_buffer(ltx_gpu *gpu,
                                                  id<MTLBuffer> buffer) {
    if (!buffer) return NULL;
    if (!ltx_stage_allocation_allowed()) return NULL;
    ltx_gpu_buffer *result = calloc(1, sizeof(*result));
    if (!result) return NULL;
    result->buffer = (__bridge_retained void *)buffer;
    result->owner = gpu;
    result->bytes = buffer.length;
    result->references = 1u;
    result->allocator_domain = gpu->allocator_domain;
    result->generation = gpu->generation;
    result->handle = ++gpu->next_buffer_handle;
    return result;
}

void ltx_gpu_stage_workspace_free(ltx_gpu_stage_workspace *workspace) {
    if (!workspace) return;
    ltx_gpu *gpu = workspace->owner;
    if (gpu && gpu->stage_workspace == workspace) {
        char drain_error[512] = {0};
        (void)ltx_gpu_drain(gpu, drain_error, sizeof(drain_error));
        gpu->stage_workspace = NULL;
    }
    workspace->owner = NULL;
    ltx_gpu_buffer_free(workspace->hidden);
    ltx_gpu_buffer_free(workspace->query);
    ltx_gpu_buffer_free(workspace->key);
    ltx_gpu_buffer_free(workspace->value);
    ltx_gpu_buffer_free(workspace->core);
    ltx_gpu_buffer_free(workspace->codes);
    ltx_gpu_buffer_free(workspace->scales);
    if (workspace->heap) (void)CFBridgingRelease(workspace->heap);
    free(workspace);
}

ltx_gpu_stage_workspace *ltx_gpu_stage_workspace_create(
        ltx_gpu *gpu, uint32_t rows, char *error, size_t error_size) {
    if (!gpu || !rows || rows > 4096u || !gpu->convrot_w8a8_enabled ||
        gpu->memory_hooks.reserve || gpu->stage_workspace ||
        !ltx_gpu_streaming_boundary_completed(gpu)) {
        ltx_gpu_fail(error, error_size,
            "stage workspace requires an idle W8A8 queue without memory hooks or another workspace");
        return NULL;
    }
    ltx_gpu_stage_workspace *workspace = ltx_stage_allocation_allowed() ?
        calloc(1, sizeof(*workspace)) : NULL;
    if (!workspace) {
        ltx_gpu_fail(error, error_size, "out of memory creating stage workspace");
        return NULL;
    }
    workspace->owner = gpu;
    workspace->rows = rows;
    @autoreleasepool {
        const size_t view_bytes = (size_t)rows * 4096u * 2u;
        const size_t hidden_bytes = view_bytes * 4u;
        const uint32_t bm = rows <= 64u ? 32u : 128u;
        const uint32_t padded_rows = (rows + bm - 1u) / bm * bm;
        const MTLResourceOptions options = MTLResourceStorageModeShared |
            MTLResourceHazardTrackingModeTracked;
        MTLSizeAndAlign view = [ltx_device(gpu)
            heapBufferSizeAndAlignWithLength:view_bytes options:options];
        MTLSizeAndAlign full = [ltx_device(gpu)
            heapBufferSizeAndAlignWithLength:hidden_bytes options:options];
        if (!view.align || !full.align || view.size < view_bytes ||
            full.size < hidden_bytes) {
            ltx_gpu_fail(error, error_size, "invalid stage heap alignment");
            goto failed;
        }
        const size_t stride = (view.size + view.align - 1u) / view.align * view.align;
        const size_t heap_bytes = MAX(full.size, 4u * stride);
        if (heap_bytes > (128u << 20)) {
            ltx_gpu_fail(error, error_size, "stage heap exceeds 128 MiB cap");
            goto failed;
        }
        MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
        descriptor.type = MTLHeapTypePlacement;
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
        descriptor.size = heap_bytes;
        id<MTLHeap> heap = ltx_stage_allocation_allowed() ?
            [ltx_device(gpu) newHeapWithDescriptor:descriptor] : nil;
        if (!heap) {
            ltx_gpu_fail(error, error_size, "shared stage heap unavailable");
            goto failed;
        }
        workspace->heap = (__bridge_retained void *)heap;
        workspace->hidden = ltx_stage_wrap_heap_buffer(gpu,
            [heap newBufferWithLength:hidden_bytes options:options offset:0]);
        ltx_gpu_buffer **views[] = {&workspace->query, &workspace->key,
                                   &workspace->value, &workspace->core};
        for (unsigned index = 0; index < 4u; ++index)
            *views[index] = ltx_stage_wrap_heap_buffer(gpu,
                [heap newBufferWithLength:view_bytes options:options offset:index * stride]);
        workspace->codes = ltx_stage_allocation_allowed() ? ltx_gpu_buffer_new_classified(gpu,
            (size_t)padded_rows * 16384u, LTX_GPU_MEMORY_CONVERSION_SCRATCH,
            "ltx_w8a8_codes", error, error_size) : NULL;
        workspace->scales = ltx_stage_allocation_allowed() ? ltx_gpu_buffer_new_classified(gpu,
            (size_t)padded_rows * 4u, LTX_GPU_MEMORY_CONVERSION_SCRATCH,
            "ltx_w8a8_token_scales", error, error_size) : NULL;
        if (!workspace->hidden || !workspace->query || !workspace->key ||
            !workspace->value || !workspace->core ||
            !workspace->codes || !workspace->scales) {
            if (error && error_size && !error[0])
                ltx_gpu_fail(error, error_size, "stage workspace buffer allocation failed");
            goto failed;
        }
        workspace->allocated_bytes = heap.size +
            MAX(ltx_buffer(workspace->codes).allocatedSize, workspace->codes->bytes) +
            MAX(ltx_buffer(workspace->scales).allocatedSize, workspace->scales->bytes);
    }
    gpu->stage_workspace = workspace;
    return workspace;
failed:
    ltx_gpu_stage_workspace_free(workspace);
    return NULL;
}

ltx_gpu_buffer *ltx_gpu_stage_workspace_hidden(const ltx_gpu_stage_workspace *workspace) {
    return workspace ? workspace->hidden : NULL;
}

size_t ltx_gpu_stage_workspace_bytes(const ltx_gpu_stage_workspace *workspace) {
    return workspace ? workspace->allocated_bytes : 0;
}

uint64_t ltx_gpu_stage_workspace_reuses(const ltx_gpu *gpu) {
    return gpu ? gpu->stage_workspace_reuses : 0;
}

static ltx_gpu_buffer *ltx_stage_borrow(ltx_gpu *gpu, ltx_gpu_buffer *buffer) {
    ltx_gpu_buffer *result = ltx_gpu_buffer_retain(buffer);
    if (result) gpu->stage_workspace_reuses++;
    return result;
}

static int ltx_stage_avoids_buffers(const ltx_gpu_buffer *scratch,
                                    const ltx_gpu_buffer *const *buffers,
                                    size_t count) {
    for (size_t index = 0; index < count; ++index)
        if (ltx_gpu_buffers_overlap(scratch, buffers[index])) return 0;
    return 1;
}

static int ltx_stage_avoids_heap(const ltx_gpu_stage_workspace *workspace,
                                 const ltx_gpu_buffer *const *buffers,
                                 size_t count) {
    for (size_t index = 0; index < count; ++index)
        if (buffers[index] && ltx_buffer(buffers[index]).heap ==
            (__bridge id<MTLHeap>)workspace->heap) return 0;
    return 1;
}

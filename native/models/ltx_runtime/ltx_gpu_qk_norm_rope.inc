/* Private outputs for an ordered Q/K/V projection. Q and K share the stage's
 * raw core view; each is transformed before the next GEMM overwrites it. */
typedef struct {
    ltx_gpu_buffer *output[2];
    const ltx_gpu_buffer *weight[2];
    const ltx_gpu_buffer *cosine;
    const ltx_gpu_buffer *sine;
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    float epsilon;
} ltx_qk_norm_rope_outputs;

uint64_t ltx_gpu_qk_norm_rope_dispatches(const ltx_gpu *gpu) {
    return gpu ? gpu->qk_norm_rope_dispatches : 0;
}

static void ltx_encode_qk_norm_rope(
        ltx_gpu *gpu, id<MTLComputeCommandEncoder> encoder,
        const ltx_qk_norm_rope_outputs *outputs,
        const ltx_gpu_buffer *raw, unsigned index) {
    struct {
        uint32_t rows, heads, head_dim;
        float epsilon;
    } args = {outputs->rows, outputs->heads, outputs->head_dim, outputs->epsilon};
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:ltx_pipeline(gpu, LTX_PIPELINE_QK_NORM_ROPE_BF16)];
    [encoder setBuffer:ltx_buffer(outputs->output[index]) offset:0 atIndex:0];
    [encoder setBuffer:ltx_buffer(raw) offset:0 atIndex:1];
    [encoder setBuffer:ltx_buffer(outputs->weight[index]) offset:0 atIndex:2];
    [encoder setBuffer:ltx_buffer(outputs->cosine) offset:0 atIndex:3];
    [encoder setBuffer:ltx_buffer(outputs->sine) offset:0 atIndex:4];
    [encoder setBytes:&args length:sizeof(args) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(outputs->rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

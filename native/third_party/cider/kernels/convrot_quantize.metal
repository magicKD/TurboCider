// TurboCider integration: FP32 absmax/RNE activation quantization after the
// original Comfy H256 transform. Source BF16 is never narrowed to FP16.
#include <metal_stdlib>
using namespace metal;

template<typename T>
inline void tc_convrot_quantize(const device T* x, device char* codes,
    device float* scales, constant uint& rows, constant uint& columns,
    uint row, uint lane, uint simd_lane, uint simd_group, uint threads,
    threadgroup float* maxima, threadgroup uint* invalid) {
    float peak = 0.0f;
    uint bad = 0;
    if (row < rows) {
        for (uint k = lane; k < columns; k += threads) {
            float v = float(x[ulong(row)*columns+k]);
            bad |= !isfinite(v);
            peak = max(peak, abs(v));
        }
    }
    peak = simd_max(peak);
    bad = simd_sum(bad);
    if (simd_lane == 0) { maxima[simd_group] = peak; invalid[simd_group] = bad; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        peak = simd_max(simd_lane < threads/32 ? maxima[simd_lane] : 0.0f);
        bad = simd_sum(simd_lane < threads/32 ? invalid[simd_lane] : 0u);
        if (simd_lane == 0) {
            // Invalid inputs must remain visible in the result, not turn into
            // a plausible zero projection. Zero rows have an exact zero dot.
            maxima[0] = bad ? NAN : (peak == 0.0f ? 1.0f : peak/127.0f);
            invalid[0] = bad;
            scales[row] = maxima[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = maxima[0];
    for (uint k = lane; k < columns; k += threads) {
        float value = row < rows && !invalid[0] ? float(x[ulong(row)*columns+k])/scale : 0.0f;
        codes[ulong(row)*columns+k] = char(clamp(rint(value), -127.0f, 127.0f));
    }
}

#define TC_CONVROT_QUANTIZE(NAME, TYPE) \
kernel void NAME(const device TYPE* x [[buffer(0)]], device char* codes [[buffer(1)]], \
    device float* scales [[buffer(2)]], constant uint& rows [[buffer(3)]], \
    constant uint& columns [[buffer(4)]], uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) { \
    threadgroup float maxima[8]; threadgroup uint invalid[8]; \
    tc_convrot_quantize(x,codes,scales,rows,columns,row,lane,simd_lane,simd_group,threads,maxima,invalid); \
}
TC_CONVROT_QUANTIZE(tc_convrot_quantize_bf16, bfloat)
TC_CONVROT_QUANTIZE(tc_convrot_quantize_f16, half)
TC_CONVROT_QUANTIZE(tc_convrot_quantize_f32, float)

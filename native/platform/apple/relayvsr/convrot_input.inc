// H256 radix-4 register transform adapted from TurboCider (MIT).
// The App bundles TurboCider-LICENSE.
@R"metal(
#include <metal_stdlib>
using namespace metal;

inline float hadamard4(float a, float b, float c, float d, uint digit) {
  switch (digit) {
    case 0: return a + b + c - d;
    case 1: return a + b - c + d;
    case 2: return a - b + c + d;
    default: return -a + b + c + d;
  }
}

inline void rotate_h256(device const bfloat* input, uint lane, thread float (&output)[8]) {
  float value[8], next[8];
  for (uint j = 0; j < 8; ++j) value[j] = float(input[lane + j * 32]);
  for (uint stride = 1; stride <= 4; stride *= 4) {
    uint digit = (lane / stride) & 3, first = lane - digit * stride;
    for (uint j = 0; j < 8; ++j) {
      next[j] = hadamard4(simd_shuffle(value[j], first),
                         simd_shuffle(value[j], first + stride),
                         simd_shuffle(value[j], first + 2 * stride),
                         simd_shuffle(value[j], first + 3 * stride), digit);
    }
    for (uint j = 0; j < 8; ++j) value[j] = next[j];
  }
  for (uint j = 0; j < 8; ++j) {
    uint even = j & ~1u, first = lane & 15u;
    uint digit = lane / 16 + (j & 1u) * 2;
    next[j] = hadamard4(simd_shuffle(value[even], first),
                       simd_shuffle(value[even], first + 16),
                       simd_shuffle(value[even + 1], first),
                       simd_shuffle(value[even + 1], first + 16), digit);
  }
  for (uint j = 0; j < 8; ++j) {
    uint first = j & 1u;
    output[j] = hadamard4(next[first], next[first + 2], next[first + 4], next[first + 6], j / 2) * 0.0625f;
  }
}

// One threadgroup owns an entire token. Rotated BF16 values stay in
// threadgroup memory while the token's scale is reduced across H256 groups.
kernel void relay_rotate_quantize(
    device const bfloat* input [[buffer(0)]], device char* codes [[buffer(1)]],
    device float* scales [[buffer(2)]], constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]], threadgroup bfloat* scratch [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint groups = 8, threads = groups * 32;
  const ulong base = ulong(row) * columns;
  if (row >= rows) {
    for (uint i = tid; i < columns; i += threads) codes[base + i] = 0;
    if (tid == 0) scales[row] = 1.0f;
    return;
  }
  float peak = 0.0f;
  uint bad = 0;
  for (uint begin = group * 256; begin < columns; begin += groups * 256) {
    float values[8];
    rotate_h256(input + base + begin, lane, values);
    for (uint j = 0; j < 8; ++j) {
      bfloat rounded = bfloat(values[j]);
      scratch[begin + j * 32 + lane] = rounded;
      float v = float(rounded);
      bad |= !isfinite(v);
      peak = max(peak, abs(v));
    }
  }
  threadgroup float peaks[groups];
  threadgroup uint invalid[groups];
  peak = simd_max(peak);
  bad = simd_max(bad);
  if (lane == 0) {
    peaks[group] = peak;
    invalid[group] = bad;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (group == 0) {
    float max_abs = simd_max(lane < groups ? peaks[lane] : 0.0f);
    uint any_bad = simd_max(lane < groups ? invalid[lane] : 0u);
    if (lane == 0) {
      peaks[0] = any_bad ? NAN : max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
      invalid[0] = any_bad;
      scales[row] = peaks[0];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < columns; i += threads) {
    float x = invalid[0] ? 0.0f : float(scratch[i]) / peaks[0];
    codes[base + i] = char(clamp(rint(x), -127.0f, 127.0f));
  }
}
)metal"

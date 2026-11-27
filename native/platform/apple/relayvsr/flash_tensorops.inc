@R"metal(
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;

struct FlashGeometry { uint h, w, planes, query_heads, kv_heads, condition; };

inline half4 flash_half(ushort4 bits) {
  return half4(as_type<float4>(uint4(bits) << 16));
}

inline ushort flash_bfloat(float x) {
  uint bits = as_type<uint>(x);
  return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// Each threadgroup owns a 4x8 query tile. Newly gathered K/V tiles cover the
// union of the queries' windows; softmax restores each exact per-query mask.
kernel void flash_tensorops_window(
    device const ushort* q [[buffer(0)]], device ushort* y [[buffer(1)]],
    constant FlashGeometry& g [[buffer(2)]],
    device const ushort* k0 [[buffer(3)]], device const ushort* k1 [[buffer(4)]],
    device const ushort* k2 [[buffer(5)]], device const ushort* k3 [[buffer(6)]],
    device const ushort* v0 [[buffer(7)]], device const ushort* v1 [[buffer(8)]],
    device const ushort* v2 [[buffer(9)]], device const ushort* v3 [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr uint GroupRows = FLASH_QUERY_ROWS, GroupCols = 8, BK = FLASH_KEY_ROWS;
  constexpr uint BQ = GroupRows * GroupCols, D = 64, Vectors = D / 4;
  constexpr uint Groups = FLASH_SIMDGROUPS, Threads = Groups * 32, KeysPerLane = BK / 32;
  threadgroup half4 q_tile[BQ * Vectors], k_tile[BK * Vectors], v_tile[BK * Vectors];
  threadgroup float logits[BQ * BK];
  threadgroup float maxima[BQ], sums[BQ], correction[BQ];
  // Probabilities reuse the first half of each row of FP32 score storage.
  auto probabilities = reinterpret_cast<threadgroup half*>(logits);
  const uint tiles_x = (g.w + GroupCols - 1) / GroupCols;
  const uint top = (group.x / tiles_x) * GroupRows;
  const uint left = (group.x % tiles_x) * GroupCols;
  const uint n = g.h * g.w;
  const uint q_head = (group.z * g.query_heads + group.y) * n * D;
  const uint kv_group = group.y / (g.query_heads / g.kv_heads);
  const uint kv_head = (group.z * g.kv_heads + kv_group) * n * D;

  for (uint index = tid; index < BQ * Vectors; index += Threads) {
    const uint row = index / Vectors, d = index % Vectors;
    const uint qy = min(top + row / GroupCols, g.h - 1);
    const uint qx = min(left + row % GroupCols, g.w - 1);
    const auto address = q_head + (qy * g.w + qx) * D + d * 4;
    q_tile[index] = flash_half(*reinterpret_cast<device const ushort4*>(q + address));
  }
  if (tid < BQ) {
    maxima[tid] = -INFINITY;
    sums[tid] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto qk_descriptor = matmul2d_descriptor(
      BQ, BK, D, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto pv_descriptor = matmul2d_descriptor(
      BQ, D, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<Groups>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<Groups>> pv;
  auto queries = tensor(reinterpret_cast<threadgroup half*>(q_tile), dextents<int32_t, 2>(D, BQ));
  auto keys = tensor(reinterpret_cast<threadgroup half*>(k_tile), dextents<int32_t, 2>(D, BK));
  auto values = tensor(reinterpret_cast<threadgroup half*>(v_tile), dextents<int32_t, 2>(D, BK));
  auto weights = tensor(probabilities, dextents<int32_t, 2>(BK, BQ),
                        array<int, 2>{1, 2 * BK});
  auto output = pv.template get_destination_cooperative_tensor<decltype(weights), decltype(values), float>();
  for (ushort e = 0; e < output.get_capacity(); ++e)
    if (output.is_valid_element(e)) output[e] = 0.0f;

  for (uint plane = 0; plane < g.planes; ++plane) {
    device const ushort* key = plane == 0 ? k0 : plane == 1 ? k1 : plane == 2 ? k2 : k3;
    device const ushort* value = plane == 0 ? v0 : plane == 1 ? v1 : plane == 2 ? v2 : v3;
    const int radius = (g.condition || plane < 2) ? 24 : 12;
    const int ymin = max(0, int(top) - radius);
    const int ymax = min(int(g.h), int(min(top + GroupRows - 1, g.h - 1)) + radius);
    const int xmin = max(0, int(left) - radius);
    const int xmax = min(int(g.w), int(min(left + GroupCols - 1, g.w - 1)) + radius);
    const uint width = uint(xmax - xmin), count = width * uint(ymax - ymin);

    for (uint base = 0; base < count; base += BK) {
      for (uint index = tid; index < BK * Vectors; index += Threads) {
        const uint token = min(base + index / Vectors, count - 1);
        const uint pos = (uint(ymin) + token / width) * g.w + uint(xmin) + token % width;
        const uint address = kv_head + pos * D + index % Vectors * 4;
        k_tile[index] = flash_half(*reinterpret_cast<device const ushort4*>(key + address));
        v_tile[index] = flash_half(*reinterpret_cast<device const ushort4*>(value + address));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      auto scores = qk.template get_destination_cooperative_tensor<decltype(queries), decltype(keys), float>();
      for (ushort e = 0; e < scores.get_capacity(); ++e)
        if (scores.is_valid_element(e)) scores[e] = 0.0f;
      qk.run(queries, keys, scores);
      for (ushort e = 0; e < scores.get_capacity(); ++e) {
        if (!scores.is_valid_element(e)) continue;
        auto index = scores.get_multidimensional_index(e);
        logits[uint(index[1]) * BK + uint(index[0])] = scores[e] * 0.125f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (uint row = simd; row < BQ; row += Groups) {
        const int qy = int(min(top + row / GroupCols, g.h - 1));
        const int qx = int(min(left + row % GroupCols, g.w - 1));
        float selected[KeysPerLane], maximum = -INFINITY;
        for (uint j = 0; j < KeysPerLane; ++j) {
          const uint kcol = lane + j * 32, token = base + kcol;
          const int ky = ymin + int(token / width), kx = xmin + int(token % width);
          const bool within = token < count && ky >= qy - radius && ky < qy + radius &&
                              kx >= qx - radius && kx < qx + radius;
          selected[j] = within ? logits[row * BK + kcol] : -INFINITY;
          maximum = max(maximum, selected[j]);
        }
        const float tile_max = simd_max(maximum);
        const float old_max = maxima[row], next_max = max(old_max, tile_max);
        const float alpha = tile_max == -INFINITY ? 1.0f :
                            old_max == -INFINITY ? 0.0f : fast::exp(old_max - next_max);
        float probs[KeysPerLane], partial = 0.0f;
        for (uint j = 0; j < KeysPerLane; ++j) {
          probs[j] = selected[j] == -INFINITY ? 0.0f : fast::exp(selected[j] - next_max);
          partial += probs[j];
        }
        const float sum = simd_sum(partial);
        if (lane == 0) {
          maxima[row] = next_max;
          sums[row] = fma(sums[row], alpha, sum);
          correction[row] = alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < KeysPerLane; ++j)
          probabilities[row * (2 * BK) + lane + j * 32] = half(probs[j]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (ushort e = 0; e < output.get_capacity(); ++e) {
        if (output.is_valid_element(e)) {
          auto index = output.get_multidimensional_index(e);
          output[e] *= correction[uint(index[1])];
        }
      }
      pv.run(weights, values, output);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (ushort e = 0; e < output.get_capacity(); ++e) {
    if (!output.is_valid_element(e)) continue;
    auto index = output.get_multidimensional_index(e);
    const uint row = uint(index[1]), d = uint(index[0]);
    const uint qy = top + row / GroupCols, qx = left + row % GroupCols;
    if (qy < g.h && qx < g.w)
      y[q_head + (qy * g.w + qx) * D + d] = flash_bfloat(output[e] / sums[row]);
  }
}
)metal"

#pragma once

namespace tc::h3_mlx {

inline constexpr const char *kVDNMmaHeader = R"METAL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define TC_VDN_BQ 32
#define TC_VDN_BK 16
#define TC_VDN_SG 4
#define TC_VDN_THREADS (TC_VDN_SG * 32)
#define TC_VDN_D 128
)METAL";

inline constexpr const char *kVDNMmaBody = R"METAL(
  const int tile = (int)threadgroup_position_in_grid.z;
  const int head = (int)threadgroup_position_in_grid.y;
  const int q0 = tile * TC_VDN_BQ;
  const int qrows = min(TC_VDN_BQ, n_q - q0);
  const device T *q_head = q + ((int64_t)head * n_q + q0) * TC_VDN_D;
  const device T *k_head = k + (int64_t)head * n_q * TC_VDN_D;
  const device T *v_head = v + (int64_t)head * n_q * TC_VDN_D;

  threadgroup T score[TC_VDN_BQ * TC_VDN_BK];
  threadgroup float output[TC_VDN_BQ * TC_VDN_D];
  threadgroup float row_max[TC_VDN_BQ];
  threadgroup float row_norm[TC_VDN_BQ];
  threadgroup float rescale[TC_VDN_BQ];
  const uint lid = thread_index_in_threadgroup;
  for (int index = (int)lid; index < TC_VDN_BQ * TC_VDN_D;
       index += TC_VDN_THREADS) output[index] = 0.0f;
  for (int index = (int)lid; index < TC_VDN_BQ; index += TC_VDN_THREADS) {
    row_max[index] = -INFINITY;
    row_norm[index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  using QTile = tensor<device T, dextents<int32_t, 2>, tensor_inline>;
  using STile = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>;
  using FTile = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;
  constexpr auto qk_desc = matmul2d_descriptor(
      TC_VDN_BQ, TC_VDN_BK, static_cast<int>(dynamic_extent), false, true, false);
  constexpr auto pv_desc = matmul2d_descriptor(
      TC_VDN_BQ, TC_VDN_D, TC_VDN_BK, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<TC_VDN_SG>> qk_op;
  matmul2d<pv_desc, execution_simdgroups<TC_VDN_SG>> pv_op;
  QTile q_tile(const_cast<device T *>(q_head),
               dextents<int32_t, 2>(TC_VDN_D, qrows));

  for (int span = block_off[tile]; span < block_off[tile + 1]; ++span) {
    const int block = block_ids[span];
    const int key0 = block * TC_VDN_BK;
    const int keys = min(TC_VDN_BK, n_q - key0);
    QTile k_tile(const_cast<device T *>(k_head + (int64_t)key0 * TC_VDN_D),
                 dextents<int32_t, 2>(TC_VDN_D, keys));
    auto qk_result = qk_op.get_destination_cooperative_tensor<QTile, QTile, T>();
    qk_op.run(q_tile, k_tile, qk_result);
    STile score_tile(score, dextents<int32_t, 2>(TC_VDN_BK, TC_VDN_BQ));
    qk_result.store(score_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int row = (int)lid; row < TC_VDN_BQ; row += TC_VDN_THREADS) {
      const int query = q0 + row;
      const bool query_ok = query < n_q;
      const bool query_video = query >= video_start && query < video_end;
      const int query_frame = query_video
          ? min(num_frames - 1, max(0, (query - video_start) / tokens_per_frame))
          : 0;
      const bool row_anchor = query_video &&
          (query_frame == 0 || query_frame == num_frames - 1);
      float maximum = row_max[row];
      for (int column = 0; column < TC_VDN_BK; ++column) {
        const int key = key0 + column;
        bool allowed = query_ok && column < keys;
        if (allowed && query_video && !row_anchor &&
            key >= video_start && key < video_end) {
          const int key_frame = (key - video_start) / tokens_per_frame;
          const int lower = max(0, bounds[query_frame * 2]);
          const int upper = min(num_frames - 1, bounds[query_frame * 2 + 1]);
          const bool in_window = key_frame >= lower && key_frame <= upper;
          const bool anchor_column = key_frame == 0 || key_frame == num_frames - 1;
          allowed = in_window || anchor_column;
        }
        const float value = allowed
            ? float(score[row * TC_VDN_BK + column]) * scale : -INFINITY;
        maximum = max(maximum, value);
        score[row * TC_VDN_BK + column] = T(value);
      }
      const float correction = exp(row_max[row] - maximum);
      float norm = row_norm[row] * correction;
      for (int column = 0; column < TC_VDN_BK; ++column) {
        const float value = float(score[row * TC_VDN_BK + column]);
        const float probability = value == -INFINITY ? 0.0f : exp(value - maximum);
        score[row * TC_VDN_BK + column] = T(probability);
        norm += probability;
      }
      row_max[row] = maximum;
      row_norm[row] = norm;
      rescale[row] = correction;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int index = (int)lid; index < TC_VDN_BQ * TC_VDN_D;
         index += TC_VDN_THREADS)
      output[index] *= rescale[index / TC_VDN_D];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STile probability(score, dextents<int32_t, 2>(TC_VDN_BK, TC_VDN_BQ));
    QTile value_tile(const_cast<device T *>(v_head + (int64_t)key0 * TC_VDN_D),
                     dextents<int32_t, 2>(TC_VDN_D, keys));
    FTile output_tile(output, dextents<int32_t, 2>(TC_VDN_D, TC_VDN_BQ));
    pv_op.run(probability, value_tile, output_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  device T *out_head = out + ((int64_t)head * n_q + q0) * TC_VDN_D;
  for (int index = (int)lid; index < TC_VDN_BQ * TC_VDN_D;
       index += TC_VDN_THREADS) {
    const int row = index / TC_VDN_D;
    const int column = index % TC_VDN_D;
    if (row < qrows) {
      const float inverse = row_norm[row] > 0.0f ? 1.0f / row_norm[row] : 0.0f;
      out_head[row * TC_VDN_D + column] = T(output[index] * inverse);
    }
  }
)METAL";

} // namespace tc::h3_mlx

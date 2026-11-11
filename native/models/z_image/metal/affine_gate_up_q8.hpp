#pragma once

#include <mlx/fast.h>
#include <mlx/mlx.h>
#include <stdexcept>
#include <vector>

namespace tc::z_metal {
namespace mx = mlx::core;

// Joint affine-Q8 projection for an FFN's W1/W3 pair.  Each threadgroup
// loads one [32, 32] BF16 activation tile once, decodes the corresponding
// affine-Q8 tile from each weight bank, and keeps two FP32 accumulators.
//
// This deliberately returns the two projections rather than fusing SwiGLU:
// W1/W3's BF16 rounding boundary remains identical in placement to the MLX
// reference, and callers retain their existing LoRA/bias epilogues.
inline std::vector<mx::array> affine_gate_up_q8(const mx::array &x,
                                                 const mx::array &gate_words,
                                                 const mx::array &gate_scales,
                                                 const mx::array &gate_biases,
                                                 const mx::array &up_words,
                                                 const mx::array &up_scales,
                                                 const mx::array &up_biases) {
    if (x.ndim() != 3 || x.shape(0) != 1 || x.dtype() != mx::bfloat16 ||
        !x.flags().row_contiguous || gate_words.ndim() != 2 ||
        up_words.shape() != gate_words.shape() || gate_words.dtype() != mx::uint32 ||
        !gate_words.flags().row_contiguous || !up_words.flags().row_contiguous ||
        gate_scales.dtype() != mx::bfloat16 || gate_biases.dtype() != mx::bfloat16 ||
        up_scales.dtype() != mx::bfloat16 || up_biases.dtype() != mx::bfloat16 ||
        gate_scales.shape() != gate_biases.shape() || up_scales.shape() != up_biases.shape() ||
        gate_scales.shape() != up_scales.shape() || gate_scales.ndim() != 2 ||
        gate_scales.shape(0) != gate_words.shape(0) ||
        gate_words.shape(0) != up_words.shape(0) ||
        gate_words.shape(1) * 4 != x.shape(2) ||
        gate_scales.shape(1) * 32 != x.shape(2) || x.shape(1) <= 0)
        throw std::invalid_argument("Z-Image joint affine-Q8 W1/W3 geometry mismatch");

    const int m = x.shape(1), n = gate_words.shape(0), k = x.shape(2);
    // The current ConvRot FFN is 3840 -> 10240, but N need only be 32-aligned.
    if (n % 32 || k % 32)
        throw std::invalid_argument("Z-Image joint affine-Q8 W1/W3 requires N/K multiples of 32");
    static auto kernel = mx::fast::metal_kernel(
        "tc_z_affine_q8_gate_up_joint", {"x", "gw", "gs", "gb", "uw", "us", "ub"},
        {"gate", "up"}, R"metal(
        using namespace mpp::tensor_ops;
        constexpr uint BM = 32, BN = 32, BK = 32;
        uint row = threadgroup_position_in_grid.y * BM;
        uint col = threadgroup_position_in_grid.x * BN;
        uint tid = thread_index_in_threadgroup;
        threadgroup T xs[BM * BK];
        threadgroup T wg[BN * BK];
        threadgroup T wu[BN * BK];
        auto a = tensor(xs, dextents<int,2>{BK,BM}, array<int,2>{1,BK});
        auto bg = tensor(wg, dextents<int,2>{BK,BN}, array<int,2>{1,BK});
        auto bu = tensor(wu, dextents<int,2>{BK,BN}, array<int,2>{1,BK});
        matmul2d<matmul2d_descriptor(BM,BN,BK,false,true,false), execution_simdgroups<4>> op;
        auto ag = op.template get_destination_cooperative_tensor<decltype(a),decltype(bg),float>();
        auto au = op.template get_destination_cooperative_tensor<decltype(a),decltype(bu),float>();
        for (uint i = 0; i < ag.get_capacity(); ++i)
            if (ag.is_valid_element(i)) { ag[i] = 0; au[i] = 0; }
        for (uint kb = 0; kb < K; kb += BK) {
            for (uint q = tid; q < BM * BK; q += 128) {
                uint r = q / BK, c = q % BK;
                xs[r * BK + c] = row + r < M ? x[(row + r) * K + kb + c] : T(0);
            }
            for (uint q = tid; q < BN * BK; q += 128) {
                uint r = q / BK, c = q % BK;
                uint out_row = col + r, logical_k = kb + c;
                uint word_index = out_row * (K / 4) + logical_k / 4;
                uint shift = (logical_k & 3u) * 8u;
                uint meta = out_row * (K / 32) + logical_k / 32;
                // Matches MLX QuantizedBlockLoader::dequantize<T,4,8>:
                // perform scale*code+bias in T, never widen metadata first.
                wg[r*BK+c] = gs[meta] * T((gw[word_index] >> shift) & 255u) + gb[meta];
                wu[r*BK+c] = us[meta] * T((uw[word_index] >> shift) & 255u) + ub[meta];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            op.run(a,bg,ag); op.run(a,bu,au);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i = 0; i < ag.get_capacity(); ++i) {
            if (!ag.is_valid_element(i)) continue;
            auto coord = ag.get_multidimensional_index(i);
            if (row+coord[1] < M && col+coord[0] < N) {
                uint offset = (row+coord[1])*N+col+coord[0];
                gate[offset]=T(ag[i]); up[offset]=T(au[i]);
            }
        }
        )metal", "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    return kernel({x, gate_words, gate_scales, gate_biases, up_words, up_scales, up_biases},
                  {{1, m, n}, {1, m, n}}, {mx::bfloat16, mx::bfloat16},
                  {(n / 32) * 128, (m + 31) / 32, 1}, {128, 1, 1},
                  {{"T", x.dtype()}, {"M", m}, {"N", n}, {"K", k}}, {}, false, {});
}

} // namespace tc::z_metal

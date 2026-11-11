#pragma once

#include <mlx/mlx.h>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <variant>

namespace ltx_prunavaed {
namespace mx = mlx::core;

inline bool tensorops_available() {
    static const bool supported = [] {
#if defined(__APPLE__)
        // Cooperative BF16 inputs need a newer compiler/runtime than the
        // general MLX deployment target. Unsupported systems retain MLX Conv3D.
        if (__builtin_available(macOS 26.4, *)) {
            const mx::Device gpu(mx::Device::gpu);
            if (!mx::is_available(gpu)) return false;
            const auto& info = mx::device_info(gpu);
            const auto found = info.find("architecture");
            if (found == info.end()) return false;
            const auto* name = std::get_if<std::string>(&found->second);
            unsigned generation = 0;
            return name && std::sscanf(name->c_str(), "applegpu_g%u", &generation) == 1 && generation >= 17;
        }
#endif
        return false;
    }();
    return supported;
}

// Noncausal replicate-time/zero-space padding, with original BF16 coefficients
// and FP32 TensorOps accumulation. No im2col or padded activation is allocated.
// The reduction order differs from MLX Conv3D; this is not a byte-exact route.
inline bool can_convolve_tensorops(const mx::array& input, const mx::array& weight,
                                   const mx::array& bias) {
    if (input.ndim() != 5 || input.shape(0) != 1 || weight.ndim() != 5 ||
        bias.ndim() != 1 || input.dtype() != mx::bfloat16 || weight.dtype() != mx::bfloat16 ||
        bias.dtype() != mx::bfloat16 || !input.size() ||
        input.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
    const int channels = input.shape(4);
    if (channels != 64 && channels != 128 && channels != 256 && channels != 384 &&
        channels != 512 && channels != 1024)
        return false;
    if (weight.shape(1) != 3 || weight.shape(2) != 3 || weight.shape(3) != 3 ||
        weight.shape(4) != channels || weight.shape(0) != bias.shape(0) ||
        weight.shape(0) <= 0 || weight.shape(0) > 8192 || input.size() / channels < 1024)
        return false;
    return tensorops_available();
}

inline bool optimized_input_loads_admitted(const mx::array& input, const mx::array& weight) {
    // Expanding projections use the scalar route: CI=1024/CO=4096 regressed
    // in paired operator tests. Alignment is checked on the GPU because lazy
    // dense views may acquire their byte offset only during evaluation.
    return input.ndim() == 5 && weight.ndim() == 5 &&
        weight.shape(0) <= input.shape(4);
}

inline mx::array convolution_tensorops(const mx::array& input, const mx::array& weight,
                                      const mx::array& bias, bool full_channel_tiles = true,
                                      bool optimized_input_loads = false) {
    if (!can_convolve_tensorops(input, weight, bias))
        throw std::invalid_argument("unsupported PrunaVAED TensorOps convolution shape");
    static auto kernel = mx::fast::metal_kernel(
        "tc_prunavaed_conv3d_tensorops_v7", {"inp", "weight", "bias"}, {"out"}, R"MSL(

const bool input_aligned = (reinterpret_cast<ulong>(inp) & 7ul) == 0;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const uint tg_m = threadgroup_position_in_grid.x, tg_n = threadgroup_position_in_grid.y;
const short qid = lane >> 2;
const short sy = (qid & 4) | ((lane >> 1) & 3);
const short sx = ((qid & 2) | (lane & 1)) * 4;
const uint m0 = tg_m * BM + (sg / WN) * 32;
const uint n0 = tg_n * BN + (sg % WN) * 32;
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
auto b = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
auto c = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
float accum[2][2][8];
for (uint mm = 0; mm < 2; ++mm)
    for (uint nn = 0; nn < 2; ++nn)
        for (uint j = 0; j < 8; ++j) accum[mm][nn][j] = 0;
for (uint tap = 0; tap < 27; ++tap) {
    int base[2][2];
    for (uint mm = 0; mm < 2; ++mm)
        for (uint i = 0; i < 2; ++i)
            base[mm][i] = pruna_virtual_base<CIN, TT, HH, WW>(inp, m0 + mm * 16 + sy + i * 8, tap);
    #pragma clang loop unroll_count(2)
    for (uint ci = 0; ci < CIN; ci += 16) {
        const uint kb = tap * CIN + ci;
        bfloat av[2][8], bv[2][8];
        if constexpr (OPTIMIZED_INPUT_LOADS && CIN == 1024 && COUT == 1024) {
            // Group the four scalar loads under one padding branch. This is
            // faster than explicit vectors for the 1024-channel residuals.
            for (uint mm = 0; mm < 2; ++mm)
                for (uint i = 0; i < 2; ++i) {
                    if (base[mm][i] < 0) {
                        for (uint j = 0; j < 4; ++j) av[mm][i * 4 + j] = bfloat(0);
                    } else {
                        for (uint j = 0; j < 4; ++j)
                            av[mm][i * 4 + j] = inp[size_t(base[mm][i]) + ci + sx + j];
                    }
                }
        } else if constexpr (OPTIMIZED_INPUT_LOADS) {
            if (input_aligned) {
                for (uint mm = 0; mm < 2; ++mm)
                    for (uint i = 0; i < 2; ++i) {
                        const vec<bfloat, 4> values = base[mm][i] < 0 ? vec<bfloat, 4>(bfloat(0)) :
                            *(const device vec<bfloat, 4> *)(inp + size_t(base[mm][i]) + ci + sx);
                        for (uint j = 0; j < 4; ++j) av[mm][i * 4 + j] = values[j];
                    }
            } else {
                for (uint mm = 0; mm < 2; ++mm)
                    for (uint i = 0; i < 2; ++i)
                        for (uint j = 0; j < 4; ++j)
                            av[mm][i * 4 + j] = base[mm][i] < 0 ? bfloat(0) : inp[size_t(base[mm][i]) + ci + sx + j];
            }
        } else {
            for (uint mm = 0; mm < 2; ++mm)
                for (uint i = 0; i < 2; ++i)
                    for (uint j = 0; j < 4; ++j)
                        av[mm][i * 4 + j] = base[mm][i] < 0 ? bfloat(0) : inp[size_t(base[mm][i]) + ci + sx + j];
        }
        for (uint nn = 0; nn < 2; ++nn)
            for (uint i = 0; i < 2; ++i)
                for (uint j = 0; j < 4; ++j) {
                    const uint n = n0 + nn * 16 + sy + i * 8;
                    // The host launches exactly ceil(COUT / BN) channel tiles.
                    // FULL_CHANNEL_TILES backs the last tile with zero-padded
                    // coefficients. Retained columns use the original bytes.
                    bv[nn][i * 4 + j] = (FULL_CHANNEL_TILES || n < COUT)
                        ? weight[size_t(n) * CIN * 27 + kb + sx + j] : bfloat(0);
                }
        for (uint mm = 0; mm < 2; ++mm) {
            for (uint j = 0; j < 8; ++j) {
                a[j] = av[mm][j]; b[j] = bv[0][j]; b[8 + j] = bv[1][j];
                c[j] = accum[mm][0][j]; c[8 + j] = accum[mm][1][j];
            }
            op.run(a, b, c);
            for (uint j = 0; j < 8; ++j) {
                accum[mm][0][j] = c[j]; accum[mm][1][j] = c[8 + j];
            }
        }
    }
}

for (uint mm = 0; mm < 2; ++mm)
    for (uint nn = 0; nn < 2; ++nn)
        for (uint i = 0; i < 2; ++i)
            for (uint j = 0; j < 4; ++j) {
                const uint m = m0 + mm * 16 + sy + i * 8;
                const uint n = n0 + nn * 16 + sx + j;
                if (m < TT * HH * WW && ((FULL_CHANNEL_TILES && COUT % BN == 0) || n < COUT)) {
                    const bfloat rounded = bfloat(accum[mm][nn][i * 4 + j]);
                    out[size_t(m) * COUT + n] = bfloat(float(rounded) + float(bias[n]));
                }
            }
)MSL", R"MSL(
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

template<int IC, int TT, int HH, int WW>
inline int pruna_virtual_base(const device bfloat* x, uint row, uint tap) {
    if (row >= TT * HH * WW) return -1;
    const int t = row / (HH * WW), y = (row / WW) % HH, z = row % WW;
    const int it = clamp(t + int(tap / 9) - 1, 0, TT - 1);
    const int iy = y + int((tap / 3) % 3) - 1, iz = z + int(tap % 3) - 1;
    if (iy < 0 || iy >= HH || iz < 0 || iz >= WW) return -1;
    return ((it * HH + iy) * WW + iz) * IC;
}

)MSL");
    const int frames = input.shape(1), height = input.shape(2), width = input.shape(3);
    const int channels = input.shape(4), output_channels = weight.shape(0);
    const int rows = static_cast<int>(input.size() / channels);
    auto prepared_weight = weight;
    if (full_channel_tiles && output_channels % 64 != 0) {
        prepared_weight = mx::pad(weight,
            {{0, 64 - output_channels % 64}, {0, 0}, {0, 0}, {0, 0}, {0, 0}});
    }
    return kernel({input, prepared_weight, bias}, {{1, frames, height, width, output_channels}}, {mx::bfloat16},
                  {((rows + 63) / 64) * 128, (output_channels + 63) / 64, 1}, {128, 1, 1},
                  {{"CIN", channels}, {"COUT", output_channels}, {"TT", frames}, {"HH", height},
                   {"WW", width}, {"BM", 64}, {"BN", 64}, {"WN", 2},
                   {"FULL_CHANNEL_TILES", full_channel_tiles},
                   {"OPTIMIZED_INPUT_LOADS", optimized_input_loads &&
                       optimized_input_loads_admitted(input, weight)}}, std::nullopt, false, {})[0];
}
} // namespace ltx_prunavaed

#pragma once
#include "../model.hpp"
#include <algorithm>
#include <limits>

namespace tc::seedvr2::metal {
// Exact causal/spatial stencil -> bounded typed columns -> FP32-accumulating
// MLX GEMM. Histories are read virtually: no replicated/concatenated temporal
// halo or spatial padding tensor is necessary. This is not spatial VAE tiling;
// it changes only how each convolution's original operands are scheduled.
inline Tensor conv_tiled(const Tensor &input, const Tensor &history, int halo,
                         bool repeat_history, const Tensor &weight, int stride_t,
                         int stride_s, int padding, bool pad_tail, int row_limit) {
    require(input.ndim() == 5 && history.ndim() == 5 && weight.ndim() == 5 &&
            input.dtype() == mx::float16 && history.dtype() == input.dtype() &&
            (weight.dtype() == input.dtype() || weight.dtype() == mx::float32),
            "SeedVR2 tiled convolution requires FP16 NTHWC inputs and FP16/FP32 filters");
    const int b = input.shape(0), t = input.shape(1), h = input.shape(2), w = input.shape(3), c = input.shape(4);
    const int kt = weight.shape(1), kh = weight.shape(2), kw = weight.shape(3), out_c = weight.shape(0);
    require(c > 0 && weight.shape(4) == c && history.shape(0) == b && history.shape(2) == h &&
            history.shape(3) == w && history.shape(4) == c && halo >= 0 &&
            (!halo || history.shape(1) == (repeat_history ? 1 : halo)) &&
            stride_t > 0 && stride_s > 0 && padding >= 0 && row_limit >= 128 && row_limit <= 65536,
            "invalid SeedVR2 tiled convolution geometry");
    const int ot = (t + halo - kt) / stride_t + 1;
    const int oh = (h + int(pad_tail) + 2 * padding - kh) / stride_s + 1;
    const int ow = (w + int(pad_tail) + 2 * padding - kw) / stride_s + 1;
    require(t + halo >= kt && h + int(pad_tail) + 2 * padding >= kh &&
            w + int(pad_tail) + 2 * padding >= kw && ot > 0 && oh > 0 && ow > 0,
            "empty SeedVR2 convolution output");
    const int k = kt * kh * kw * c;
    const int64_t total = int64_t(b) * ot * oh * ow;
    require(total <= std::numeric_limits<int>::max(), "SeedVR2 convolution row count exceeds native limits");
    constexpr size_t workspace_bytes = 256ull << 20;
    const int tile = std::min(row_limit, int(workspace_bytes / (size_t(k) * weight.itemsize()))) / 128 * 128;
    require(tile >= 128, "SeedVR2 convolution stencil exceeds bounded workspace");
    static auto kernel = mx::fast::metal_kernel(
        "tc_seedvr2_causal_im2col_fp16", {"x", "hist", "params"}, {"columns"}, R"metal(
        uint base = thread_position_in_grid.x * 4;
        uint count = uint(params[1]) * K;
        if (C % 4 == 0) {
            if (base >= count) return;
            uint r = base / K + uint(params[0]);
            uint column = base % K;
            uint channel = column % C; column /= C;
            uint kx = column % KW; column /= KW;
            uint ky = column % KH; uint time = column / KH;
            uint ox = r % OW; r /= OW;
            uint oy = r % OH; r /= OH;
            uint ot = r % OT; uint batch = r / OT;
            int ix = int(ox * SS + kx) - Pad;
            int iy = int(oy * SS + ky) - Pad;
            time += ot * ST;
            half4 value = half4(0);
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                if (time < Halo) {
                    uint ht = Repeat ? 0 : time;
                    uint index = (((batch * HT + ht) * H + uint(iy)) * W + uint(ix)) * C + channel;
                    value = *((device const half4 *)(hist + index));
                } else {
                    uint it = time - Halo;
                    uint index = (((batch * T + it) * H + uint(iy)) * W + uint(ix)) * C + channel;
                    value = *((device const half4 *)(x + index));
                }
            }
            using Output4 = metal::vec<OutputT, 4>;
            *((device Output4 *)(columns + base)) = Output4(value);
            return;
        }
        for (uint lane = 0; lane < 4; ++lane) {
            uint dst = base + lane;
            if (dst >= count) continue;
            uint r = dst / K + uint(params[0]);
            uint column = dst % K;
            uint channel = column % C; column /= C;
            uint kx = column % KW; column /= KW;
            uint ky = column % KH; uint time = column / KH;
            uint ox = r % OW; r /= OW;
            uint oy = r % OH; r /= OH;
            uint ot = r % OT; uint batch = r / OT;
            int ix = int(ox * SS + kx) - Pad;
            int iy = int(oy * SS + ky) - Pad;
            time += ot * ST;
            half value = half(0);
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                if (time < Halo) {
                    uint ht = Repeat ? 0 : time;
                    value = hist[(((batch * HT + ht) * H + uint(iy)) * W + uint(ix)) * C + channel];
                } else {
                    uint it = time - Halo;
                    value = x[(((batch * T + it) * H + uint(iy)) * W + uint(ix)) * C + channel];
                }
            }
            columns[dst] = OutputT(value);
        }
        )metal");
    mx::eval({input, history, weight});
    auto matrix = mx::transpose(mx::reshape(weight, {out_c, k}));
    std::vector<Tensor> chunks;
    std::vector<Tensor> pending;
    for (int first = 0; first < int(total); first += tile) {
        const int count = std::min(tile, int(total) - first);
        const std::array<int32_t, 2> parameters{first, count};
        auto columns = kernel({input, history, Tensor(parameters.data(), {2}, mx::int32)},
            {{count, k}}, {weight.dtype()}, {(count * k + 3) / 4, 1, 1}, {256, 1, 1},
            {{"OutputT", weight.dtype()}, {"K", k}, {"C", c}, {"T", t}, {"HT", history.shape(1)}, {"H", h}, {"W", w},
             {"KT", kt}, {"KH", kh}, {"KW", kw}, {"OT", ot}, {"OH", oh}, {"OW", ow},
             {"ST", stride_t}, {"SS", stride_s}, {"Pad", padding}, {"Halo", halo}, {"Repeat", repeat_history}},
            {}, false, {})[0];
        auto output = mx::astype(mx::matmul(columns, matrix), input.dtype());
        // Two outstanding tiles bound the stencil workspaces while allowing
        // CPU submission to overlap GPU execution. Do not queue the full
        // frame's column buffers through an unbounded lazy concatenation.
        mx::async_eval(output);
        pending.push_back(output);
        if (pending.size() == 2 || first + count == int(total)) {
            mx::eval(pending);
            pending.clear();
        }
        chunks.push_back(output);
    }
    return mx::reshape(mx::concatenate(chunks, 0), {b, ot, oh, ow, out_c});
}
} // namespace tc::seedvr2::metal

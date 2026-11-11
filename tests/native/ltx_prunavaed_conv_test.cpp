#include "ltx_prunavaed_conv.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace mx = mlx::core;

int main() {
    mx::set_default_device(mx::Device(mx::Device::gpu));
    if (!ltx_prunavaed::tensorops_available()) {
        std::puts("SKIP PrunaVAED TensorOps convolution requires M5+");
        return 0;
    }
    const int configurations[][5] = {
        {64, 48, 3, 19, 19}, {64, 73, 1, 33, 33},
        {128, 512, 3, 19, 19}, {384, 128, 3, 19, 19}, {512, 384, 3, 19, 19},
        {64, 64, 4, 16, 16}, {128, 128, 4, 16, 16}, {256, 256, 4, 16, 16},
        {1024, 73, 3, 19, 19}, {1024, 1024, 4, 16, 16},
    };
    int cases = 0, witnesses = 0;
    for (const auto& config : configurations) {
        const int ci = config[0], co = config[1], frames = config[2], h = config[3], w = config[4];
        const int rows = frames * h * w, k = ci * 27;
        std::vector<mx::bfloat16_t> input_host(static_cast<size_t>(rows) * ci);
        std::vector<mx::bfloat16_t> weight_host(static_cast<size_t>(co) * k), bias_host(co);
        for (size_t i = 0; i < input_host.size(); ++i)
            input_host[i] = mx::bfloat16_t(static_cast<float>(std::sin(i * .071) * .7));
        for (size_t i = 0; i < weight_host.size(); ++i)
            weight_host[i] = mx::bfloat16_t(static_cast<float>(std::cos(i * .013) * .015));
        for (int i = 0; i < co; ++i) bias_host[i] = mx::bfloat16_t((i - co / 2) * .001f);
        auto weight = mx::array(weight_host.data(), {co, 3, 3, 3, ci});
        auto bias = mx::array(bias_host.data(), {co});
        for (int layout : {0, 1, 2}) {
            const bool strided = layout == 1;
            auto input = mx::array(input_host.data(), {1, frames, h, w, ci});
            if (strided) input = mx::transpose(input, {0, 1, 3, 2, 4});
            if (layout == 2) {
                auto flat = mx::concatenate({mx::zeros({1}, mx::bfloat16),
                                             mx::reshape(input, {-1})});
                input = mx::reshape(mx::slice(flat, {1}, {static_cast<int>(input.size()) + 1}),
                                    {1, frames, h, w, ci});
            }
            auto first = mx::slice(input, {0, 0, 0, 0, 0}, {1, 1, h, w, ci});
            auto last = mx::slice(input, {0, frames - 1, 0, 0, 0}, {1, frames, h, w, ci});
            auto padded = mx::pad(mx::concatenate({first, input, last}, 1),
                                 {{0, 0}, {0, 0}, {1, 1}, {1, 1}, {0, 0}});
            auto expected = mx::conv3d(padded, weight) + bias;
            auto actual = ltx_prunavaed::convolution_tensorops(input, weight, bias);
            auto checked = ltx_prunavaed::convolution_tensorops(input, weight, bias, false);
            auto packed = ltx_prunavaed::convolution_tensorops(input, weight, bias, true, true);
            auto packed_checked = ltx_prunavaed::convolution_tensorops(input, weight, bias, false, true);
            auto logical_input = mx::contiguous(input);
            mx::eval({expected, actual, checked, packed, packed_checked, logical_input});
            if (layout == 2 && (input.offset() % 8 != 2 || !input.flags().row_contiguous)) {
                std::fprintf(stderr, "FAIL dense BF16 offset fixture\n");
                return 1;
            }
            if (std::memcmp(actual.data<mx::bfloat16_t>(), packed.data<mx::bfloat16_t>(), actual.nbytes()) ||
                std::memcmp(actual.data<mx::bfloat16_t>(), packed_checked.data<mx::bfloat16_t>(), actual.nbytes())) {
                std::fprintf(stderr, "FAIL packed BF16 input ci=%d co=%d layout=%d\n", ci, co, layout);
                return 1;
            }
            if (ltx_prunavaed::optimized_input_loads_admitted(input, weight) != (co <= ci)) return 1;
            if (std::memcmp(actual.data<mx::bfloat16_t>(), checked.data<mx::bfloat16_t>(),
                            actual.nbytes()) != 0) {
                std::fprintf(stderr, "FAIL full channel tile predicate ci=%d co=%d strided=%d\n", ci, co, strided);
                return 1;
            }
            const auto* a = expected.data<mx::bfloat16_t>();
            const auto* b = actual.data<mx::bfloat16_t>();
            const auto* x = logical_input.data<mx::bfloat16_t>();
            double error = 0, energy = 0, maximum = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                const double reference = static_cast<float>(a[i]), value = static_cast<float>(b[i]);
                if (!std::isfinite(reference) || !std::isfinite(value)) return 1;
                const double delta = value - reference;
                error += delta * delta; energy += reference * reference;
                maximum = std::max(maximum, std::abs(delta));
            }
            const double relative = std::sqrt(error / std::max(energy, 1e-30));
            if (relative > .001) {
                std::fprintf(stderr, "FAIL Conv3D ci=%d co=%d strided=%d relL2=%g max=%g\n", ci, co, strided, relative, maximum);
                return 1;
            }
            // Independent FP64 dot products at temporal/spatial borders and
            // tail rows/columns; no MLX convolution is used by this witness.
            for (int row : {0, w - 1, h * w - 1, rows / 2, rows - 1}) {
                const int t = row / (h * w), y = (row / w) % h, z = row % w;
                for (int column : {0, co / 2, co - 1}) {
                    double sum = 0;
                    for (int dt = 0; dt < 3; ++dt)
                        for (int dy = 0; dy < 3; ++dy)
                            for (int dz = 0; dz < 3; ++dz) {
                                const int it = std::clamp(t + dt - 1, 0, frames - 1);
                                const int iy = y + dy - 1, iz = z + dz - 1;
                                if (iy < 0 || iy >= h || iz < 0 || iz >= w) continue;
                                for (int channel = 0; channel < ci; ++channel) {
                                    const size_t xi = ((static_cast<size_t>(it) * h + iy) * w + iz) * ci + channel;
                                    const size_t wi = (((static_cast<size_t>(column) * 3 + dt) * 3 + dy) * 3 + dz) * ci + channel;
                                    sum += static_cast<double>(static_cast<float>(x[xi])) *
                                        static_cast<float>(weight_host[wi]);
                                }
                            }
                    const auto rounded = mx::bfloat16_t(static_cast<float>(sum));
                    const float reference = static_cast<float>(mx::bfloat16_t(
                        static_cast<float>(rounded) + static_cast<float>(bias_host[column])));
                    const float value = static_cast<float>(b[static_cast<size_t>(row) * co + column]);
                    if (std::abs(value - reference) > std::max(.002f, .01f * std::abs(reference))) {
                        std::fprintf(stderr, "FAIL FP64 witness row=%d col=%d expected=%g actual=%g\n", row, column, reference, value);
                        return 1;
                    }
                    witnesses++;
                }
            }
            cases++;
        }
    }
    std::printf("PASS PrunaVAED Conv3D: %d channel/tail/strided/offset cases, optimized loads and full-channel predicate byte-exact, expansion fallback, %d independent FP64 boundary witnesses\n", cases, witnesses);
    return 0;
}

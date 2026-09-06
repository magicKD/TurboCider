#pragma once
#include "../../backends/mlx.hpp"
namespace tc {
struct FluxVaeOps {
    Weights &weights;
    Tensor conv(const Tensor &a, const std::string &p, int stride = 1, int padding = -1) {
        auto weight = mx::transpose(weights.at(p + ".weight"), {0, 2, 3, 1});
        int pad = padding < 0 ? weight.shape(1) / 2 : padding;
        return mx::conv2d(a, weight, {stride, stride}, {pad, pad}) + weights.at(p + ".bias");
    }
    Tensor group_norm(const Tensor &a, const std::string &p) {
        int n = a.shape(1) * a.shape(2), c = a.shape(3);
        auto f = mx::reshape(mx::astype(a, mx::float32), {1, n, 32, c / 32});
        f = mx::reshape(mx::transpose(f, {0, 2, 1, 3}), {1, 32, n * c / 32});
        f = mx::fast::layer_norm(f, {}, {}, 1e-6f);
        auto z =
            mx::reshape(mx::transpose(mx::reshape(f, {1, 32, n, c / 32}), {0, 2, 1, 3}), a.shape());
        return mx::astype(z * mx::astype(weights.at(p + ".weight"), mx::float32) +
                              mx::astype(weights.at(p + ".bias"), mx::float32),
                          mx::bfloat16);
    }
    Tensor residual(const Tensor &a, const std::string &p) {
        auto z = conv(silu(group_norm(a, p + ".norm1")), p + ".conv1");
        z = conv(silu(group_norm(z, p + ".norm2")), p + ".conv2");
        return z + (weights.has(p + ".conv_shortcut.weight") ? conv(a, p + ".conv_shortcut") : a);
    }
    Tensor attention(const Tensor &x, const std::string &p) {
        int n = x.shape(1) * x.shape(2), c = x.shape(3);
        auto z = group_norm(x, p + ".group_norm");
        auto q = heads(mx::reshape(linear(z, weights, p + ".to_q"), {1, n, c}), 1, c);
        auto k = heads(mx::reshape(linear(z, weights, p + ".to_k"), {1, n, c}), 1, c);
        auto v = heads(mx::reshape(linear(z, weights, p + ".to_v"), {1, n, c}), 1, c);
        return x + linear(mx::reshape(attend(q, k, v), x.shape()), weights, p + ".to_out.0");
    }
};
} // namespace tc

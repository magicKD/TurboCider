#include "common.h"
#include "../../platform/apple/relayvsr/convrot.h"

#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <algorithm>

namespace rv {
namespace {

Tensor rotate_convrot(const Tensor& x) {
  static const auto transform = [] {
    const auto h4 = at::tensor({1.f, 1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f,
                               1.f, -1.f, 1.f, 1.f, -1.f, 1.f, 1.f, 1.f},
                              at::kFloat).reshape({4, 4});
    auto h256 = h4;
    for (int digit = 1; digit < 4; ++digit) h256 = at::kron(h256, h4);
    return h256.mul_(1.f / 16.f);
  }();
  // The W8A16 fallback caches the H256 basis and shares rotation across Q/K/V.
  const auto& basis = x.is_mps() ? []() -> const Tensor& {
    static const auto device_transform = transform.to(at::kMPS);
    return device_transform;
  }() : transform;
  return at::matmul(x.reshape({-1, 256}).to(at::kFloat), basis)
      .reshape(x.sizes()).to(x.scalar_type());
}

std::vector<Tensor> convrot_linears(const Tensor& x, const Weights& w,
                                   std::initializer_list<std::string> prefixes, bool gelu) {
  auto shape = x.sizes().vec();
  auto matrix = x.reshape({-1, shape.back()}).contiguous();
  std::vector<Tensor> outputs;
  outputs.reserve(prefixes.size());
  const bool use_a8 = w.tensorops() && matrix.scalar_type() == at::kBFloat16 &&
      convrot_int8_fits(matrix.size(0), matrix.size(1));
  ConvRotRows input{};
  if (use_a8) {
    input = prepare_convrot_int8(matrix);
    w.record_int8(prefixes.size(), gelu);
  } else {
    matrix = rotate_convrot(matrix);
  }
  for (const auto& p : prefixes) {
    auto weight = w.get(p + ".weight");
    auto scale = w.get(p + ".weight_scale").reshape({-1});
    auto bias = w.get(p + ".bias");
    Tensor output;
    if (use_a8) {
      output = project_int8(input, weight, scale, bias, gelu);
    } else {
      output = at::_weight_int8pack_mm(matrix, weight, scale.to(matrix.scalar_type())) + bias;
      if (gelu) output = at::gelu(output, "tanh");
    }
    shape.back() = weight.size(0);
    outputs.push_back(output.reshape(shape));
  }
  return outputs;
}

bool is_convrot_projection(const Weights& w, const std::string& p) {
  return w.convrot() && w.contains(p + ".comfy_quant");
}

}  // namespace

Tensor linear(Tensor x, const Weights& w, const std::string& p) {
  if (is_convrot_projection(w, p)) return convrot_linears(x, w, {p}, false).front();
  if (w.contains(p + ".weight_scale")) {
    auto weight = w.get(p + ".weight");
    auto shape = x.sizes().vec();
    auto matrix = x.reshape({-1, shape.back()}).contiguous();
    auto y = at::_weight_int8pack_mm(matrix, weight, w.get(p + ".weight_scale"));
    if (w.contains(p + ".bias")) y = y + w.get(p + ".bias");
    shape.back() = weight.size(0);
    return y.reshape(shape);
  }
  return at::linear(x, w.get(p + ".weight"),
                    w.contains(p + ".bias") ? w.get(p + ".bias") : Tensor());
}

Tensor linear_gelu(Tensor x, const Weights& w, const std::string& p) {
  if (is_convrot_projection(w, p)) return convrot_linears(x, w, {p}, true).front();
  return at::gelu(linear(x, w, p), "tanh");
}

std::vector<Tensor> linear_shared(Tensor x, const Weights& w,
                                  std::initializer_list<std::string> prefixes) {
  if (w.convrot() && std::ranges::all_of(prefixes, [&](const auto& p) { return is_convrot_projection(w, p); }))
    return convrot_linears(x, w, prefixes, false);
  std::vector<Tensor> outputs;
  outputs.reserve(prefixes.size());
  for (const auto& p : prefixes) outputs.push_back(linear(x, w, p));
  return outputs;
}
Tensor rms(Tensor x, Tensor weight, double eps) {
  return at::rms_norm(x, {x.size(-1)}, weight, eps);
}
Tensor layernorm(Tensor x, Tensor weight, Tensor bias, double eps) {
  return at::layer_norm(
      x, {x.size(-1)}, weight.defined() ? weight.to(x.scalar_type()) : Tensor(),
      bias.defined() ? bias.to(x.scalar_type()) : Tensor(), eps, false);
}
Tensor attention(Tensor q, Tensor k, Tensor v,
                 std::function<Tensor(int64_t, int64_t)> mask, int64_t chunk) {
  TORCH_CHECK(q.size(1) % k.size(1) == 0,
              "Invalid grouped heads / 注意力分组无效");
  if (q.size(1) != k.size(1)) {
    auto groups = q.size(1) / k.size(1);
    k = k.repeat_interleave(groups, 1);
    v = v.repeat_interleave(groups, 1);
  }
  if (!mask)
    return at::scaled_dot_product_attention(q, k, v, std::nullopt, 0., false);
  auto out = at::empty_like(q);
  for (int64_t start = 0; start < q.size(2); start += chunk) {
    auto end = std::min(start + chunk, q.size(2));
    auto m = mask(start, end).to(q.device());
    auto y = at::scaled_dot_product_attention(q.slice(2, start, end), k, v, m,
                                              0., false);
    out.slice(2, start, end).copy_(y);
  }
  return out;
}

}  // namespace rv

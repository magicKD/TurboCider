#include "flash.h"


#include "../../platform/apple/relayvsr/metal_attention.h"

#include <ATen/ATen.h>

#include <cmath>
#include <vector>

namespace rv {
namespace {
constexpr double norm_eps = 1e-5;
using Rope = std::pair<Tensor, Tensor>;

Tensor conv(const Tensor& x, const Weights& w, const std::string& name,
            int stride = 1, int padding = 0) {
  return at::conv2d(x, w.get(name + ".weight"), w.get(name + ".bias"),
                    {stride, stride}, {padding, padding}, {1, 1}, 1);
}
Tensor norm(const Tensor& x, const Weights& w, const std::string& name) {
  // Native RMSNorm is the operation used by the official Flash model.
  return at::rms_norm(x, {x.size(-1)}, w.get(name + ".weight"), norm_eps);
}
Rope compute_rope(const Tensor& like, int h, int w, const std::vector<int>& times) {
  auto opts = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  std::vector<Tensor> cosines, sines;
  const int dims[] = {16, 24, 24};
  const int64_t count = int64_t(h) * w * times.size();
  auto positions = at::empty({3, count}, opts);
  auto p = positions.accessor<float, 2>();
  int64_t index = 0;
  for (int t : times)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x, ++index) {
        p[0][index] = t;
        p[1][index] = y;
        p[2][index] = x;
      }
  for (int axis = 0; axis < 3; ++axis) {
    auto inv = at::reciprocal(
        at::pow(10000.0, at::arange(0, dims[axis], 2, opts) / dims[axis]));
    auto angles = positions[axis].unsqueeze(-1) * inv;
    cosines.push_back(at::repeat_interleave(at::cos(angles), 2, -1));
    sines.push_back(at::repeat_interleave(at::sin(angles), 2, -1));
  }
  return {at::cat(cosines, -1).to(like.options()),
          at::cat(sines, -1).to(like.options())};
}
Rope rope(const Tensor& like, int h, int w, const std::vector<int>& times) {
  return compute_rope(like, h, w, times);
}
Tensor rotate(const Tensor& x, const Rope& r) {
  auto even = x.slice(-1, 0, x.size(-1), 2);
  auto odd = x.slice(-1, 1, x.size(-1), 2);
  auto c = r.first.slice(-1, 0, x.size(-1), 2).unsqueeze(0).unsqueeze(0);
  auto s = r.second.slice(-1, 0, x.size(-1), 2).unsqueeze(0).unsqueeze(0);
  // Preserve BF16 multiply/add boundaries of the eager reference.
  return at::stack({even * c - odd * s, even * s + odd * c}, -1).flatten(-2);
}
Tensor attend(const Tensor& query, const Tensor& key, const Tensor& value,
              const Rope& qr, const Rope& kr, int h, int w, bool condition_only) {
  return metal_local_attention(rotate(query, qr), rotate(key, kr), value, h, w, condition_only);
}
struct Projected {
  Tensor query, key, value;
};
Projected project(const Tensor& x, const Weights& w, const std::string& p,
                  bool condition) {
  const auto b = x.size(0), n = x.size(1);
  auto q = linear(x, w, p + ".query").view({b, n, 8, 64}).transpose(1, 2);
  auto k = linear(x, w, p + (condition ? ".condition_key" : ".key"))
               .view({b, n, 2, 64})
               .transpose(1, 2);
  auto v = linear(x, w, p + (condition ? ".condition_value" : ".value"))
               .view({b, n, 2, 64})
               .transpose(1, 2);
  return {q, norm(k, w, p + ".key_norm"), norm(v, w, p + ".value_norm")};
}
Tensor finish_block(Tensor x, Tensor attended, const Weights& w,
                    const std::string& p) {
  attended = attended.transpose(1, 2).reshape({x.size(0), x.size(1), 512});
  x = x + linear(attended, w, p + ".attention.output");
  auto normalized = norm(x, w, p + ".feed_forward_norm");
  auto gu = linear(normalized, w,
                   p + ".feed_forward.gate_up_proj")
                .chunk(2, -1);
  return x + linear(at::silu(gu[0]) * gu[1], w, p + ".feed_forward.down");
}
}  // namespace

Flash::Flash(Weights& weights, bool prefer_tensorops)
    : weights_(weights), tensorops_(prefer_tensorops && flash_tensorops_available()) {}
std::shared_ptr<FlashCondition> Flash::prepare(const Tensor& latent) {
  TORCH_CHECK(latent.dim() == 4 && latent.size(1) == 48,
              "Flash端点须为[B,48,H,W] / Expected Flash endpoint [B,48,H,W]");
  auto result = std::make_shared<FlashCondition>();
  result->batch_size = latent.size(0);
  result->height = latent.size(2);
  result->width = latent.size(3);
  result->owner = this;
  auto x = conv(latent, weights_, "condition_encoder.projection")
               .flatten(2)
               .transpose(1, 2);
  x = norm(x, weights_, "condition_encoder.norm") +
      weights_.get("condition_type_embedding");
  auto r = rope(x, result->height, result->width, {0});
  for (int i = 0; i < 10; ++i) {
    auto p = "backbone.blocks." + std::to_string(i);
    auto qkv = project(norm(x, weights_, p + ".attention_norm"), weights_,
                       p + ".attention", true);
    if (tensorops_) {
      auto& kv = result->caches[i];
      kv = {rotate(qkv.key, r).contiguous(), qkv.value.contiguous()};
      // The last layer contributes its K/V, but no subsequent layer reads x.
      if (i + 1 == static_cast<int>(result->caches.size())) break;
      auto a = tensorops_window_attention(rotate(qkv.query, r), std::span(&kv, 1),
                                         result->height, result->width, true);
      x = finish_block(x, a, weights_, p);
      continue;
    }
    result->caches[i] = {qkv.key, qkv.value};
    auto a = attend(qkv.query, qkv.key, qkv.value, r, r, result->height,
                    result->width, true);
    x = finish_block(x, a, weights_, p);
  }
  return result;
}
FlashState Flash::begin(std::shared_ptr<FlashCondition> start,
                        std::shared_ptr<FlashCondition> end, int total_frames,
                        int height, int width) {
  TORCH_CHECK(start && end && start->owner == this && end->owner == this &&
                  total_frames >= 1,
              "Flash条件或流长度无效 / Invalid Flash prepared conditions or "
              "stream length");
  constexpr int stride = 8;
  TORCH_CHECK(height > 0 && width > 0 && start->height == (height + stride - 1) / stride &&
                  start->width == (width + stride - 1) / stride &&
                  end->height == start->height && end->width == start->width &&
                  end->batch_size == start->batch_size,
              "Flash端点与低分辨率流尺寸不符 / Flash endpoint and LR stream "
              "shapes do not match");
  FlashState s;
  s.owner = this;
  s.total_frames = total_frames;
  s.height = height;
  s.width = width;
  s.batch_size = start->batch_size;
  if (tensorops_) {
    const auto& like = start->caches.front().key;
    s.rope_now = rope(like, start->height, start->width, {0});
    s.rope_previous = rope(like, start->height, start->width, {-1});
    s.start = std::move(start);
    s.end = std::move(end);
    return s;
  }
  for (int i = 0; i < 10; ++i) {
    s.condition[i] = {
        at::cat({start->caches[i].key, end->caches[i].key}, 2),
        at::cat({start->caches[i].value, end->caches[i].value}, 2)};
  }
  return s;
}
Tensor Flash::decode(const Tensor& input, FlashState& state) {
  TORCH_CHECK(state.owner == this && state.frame_index < state.total_frames,
              "Flash流无效或已完成 / Invalid/completed Flash stream");
  TORCH_CHECK(input.dim() == 4 && input.size(0) == state.batch_size &&
                  input.size(1) == 3 && input.size(2) == state.height &&
                  input.size(3) == state.width,
              "Flash低分辨率流尺寸改变 / Flash LR stream shape changed");
  constexpr int stride = 8;
  int ph = (stride - state.height % stride) % stride, pw = (stride - state.width % stride) % stride;
  auto lr = (ph || pw) ? at::replication_pad2d(input, {0, pw, 0, ph}) : input;
  int h = lr.size(2) / stride, w = lr.size(3) / stride;
  int64_t n = int64_t(h) * w;
  auto x = lr;
  for (int i = 0; i < 3; ++i)
    x = at::silu(conv(x, weights_, "lr_encoder.layers." + std::to_string(i * 2),
                      i < 2 ? 2 : stride / 4, 1));
  x = conv(x, weights_, "lr_encoder.layers.6").flatten(2).transpose(1, 2);
  x = norm(x, weights_, "lr_encoder.norm") +
      weights_.get("video_type_embedding");
  bool previous = state.frame_index > 0;
  const auto qr = tensorops_ ? state.rope_now : rope(x, h, w, {0});
  const auto kr = tensorops_ ? Rope{} : rope(
      x, h, w,
      previous ? std::vector<int>{0, 0, -1, 0} : std::vector<int>{0, 0, 0});
  for (int i = 0; i < 10; ++i) {
    auto p = "backbone.blocks." + std::to_string(i);
    auto qkv = project(norm(x, weights_, p + ".attention_norm"), weights_,
                       p + ".attention", false);
    Tensor a;
    if (tensorops_) {
      std::array<FlashKV, 4> planes;
      size_t count = 2;
      planes[0] = state.start->caches[i];
      planes[1] = state.end->caches[i];
      if (previous) planes[count++] = std::move(state.video[i]);
      auto value = qkv.value.contiguous();
      planes[count++] = {rotate(qkv.key, qr).contiguous(), value};
      a = tensorops_window_attention(rotate(qkv.query, qr), std::span(planes).first(count), h, w, false);
      if (state.frame_index + 1 < state.total_frames)
        state.video[i] = {rotate(qkv.key, state.rope_previous).contiguous(), std::move(value)};
    } else {
      auto vk = qkv.key, vv = qkv.value;
      if (previous) {
        auto& old = state.video[i];
        vk = at::cat({old.key.slice(2, old.key.size(2) - n), vk}, 2);
        vv = at::cat({old.value.slice(2, old.value.size(2) - n), vv}, 2);
      }
      auto key = at::cat({state.condition[i].key, vk}, 2);
      auto value = at::cat({state.condition[i].value, vv}, 2);
      a = attend(qkv.query, key, value, qr, kr, h, w, false);
      state.video[i] = {vk, vv};
    }
    x = finish_block(x, a, weights_, p);
  }
  auto features = at::silu(linear(x, weights_, "spatial_head.0"));
  auto pixels = linear(features, weights_, "residual_projection")
                    .transpose(1, 2)
                    .reshape({input.size(0), 768, h, w});
  auto residual = at::pixel_shuffle(pixels, 16);
  auto bicubic = at::upsample_bicubic2d(lr, {lr.size(2) * 2, lr.size(3) * 2},
                                        false, std::nullopt, std::nullopt);
  // Preserve model dtype: the official writer performs its RGB arithmetic in
  // BF16.
  auto output = (bicubic + residual)
                    .slice(2, 0, state.height * 2)
                    .slice(3, 0, state.width * 2);
  ++state.frame_index;
  return output;
}
}  // namespace rv

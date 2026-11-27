// Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
// Adapted from RelayVSR's Apache-2.0 Wan transformer and LQ projection.
// Licensed under Apache-2.0; see native/licenses/RelayVSR-LICENSE.txt.
#include "wan.h"
#include <map>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace rv {
namespace {
Tensor heads(Tensor x) {
  return x.reshape({x.size(0), x.size(1), 24, 128}).transpose(1, 2);
}
Tensor merge(Tensor x) { return x.transpose(1, 2).flatten(2, 3); }
Tensor conv(Tensor x, const Weights& w, const std::string& p, int stride = 1,
            int pad = 0) {
  return at::conv2d(x, w.get(p + ".weight"), w.get(p + ".bias"),
                    {stride, stride}, {pad, pad}, {1, 1}, 1);
}
Tensor rms2(Tensor x, Tensor weight) {
  auto y = x.to(at::kFloat);
  if (weight.dim() == 1) weight = weight.view({1, -1, 1, 1});
  return (y * at::rsqrt(y.square().mean(1, true) + 1e-6) * weight)
      .to(x.scalar_type());
}
// Compute RoPE on CPU in double as on CUDA, then transfer float frequencies.
// Keeping this source in float avoids BF16 quantization of position
// frequencies.
std::pair<Tensor, Tensor> rope(int64_t h, int64_t w,
                               const std::vector<int64_t>& positions,
                               const Tensor& like) {
  auto n = static_cast<int64_t>(positions.size()) * h * w;
  auto c = at::empty({1, 1, n, 64}, at::TensorOptions().dtype(at::kFloat));
  auto s = at::empty_like(c);
  auto cp = c.data_ptr<float>();
  auto sp = s.data_ptr<float>();
  for (int64_t f = 0; f < (int64_t)positions.size(); ++f)
    for (int64_t y = 0; y < h; ++y)
      for (int64_t x = 0; x < w; ++x) {
        auto base = ((f * h + y) * w + x) * 64;
        for (int d = 0; d < 64; ++d) {
          int dim = d < 22 ? 44 : 42,
              local = d < 22 ? d : (d < 43 ? d - 22 : d - 43);
          int64_t pos = d < 22 ? positions[f] : (d < 43 ? y : x);
          double phase = pos * (1.0 / std::pow(10000.0, 2.0 * local / dim));
          cp[base + d] = std::cos(phase);
          sp[base + d] = std::sin(phase);
        }
      }
  return {c.to(like.device()), s.to(like.device())};
}
Tensor rotate(Tensor x, const std::pair<Tensor, Tensor>& r) {
  auto pairs = x.reshape({x.size(0), x.size(1), x.size(2), 64, 2});
  auto a = pairs.select(-1, 0), b = pairs.select(-1, 1);
  // Official implementation assigns FP32 intermediates into empty_like Q/K.
  return at::stack({a * r.first - b * r.second, a * r.second + b * r.first}, -1)
      .flatten(-2, -1)
      .to(x.scalar_type());
}

Tensor run_block(Weights& weights, int i, Tensor x, const Tensor& time_proj,
                 const Tensor& cross_k, const Tensor& cross_v, Tensor& cache_k,
                 Tensor& cache_v, int64_t frame, int64_t h, int64_t w,
                 const std::pair<Tensor, Tensor>& qr,
                 const std::pair<Tensor, Tensor>& kr,
                 const std::function<Tensor(int64_t, int64_t)>& mask) {
  const int64_t N = h * w;
  auto B = x.size(0);
  auto dtype = x.scalar_type();
  auto valid = std::min<int64_t>(frame + 1, 4);
  auto p = "blocks." + std::to_string(i);
  auto mod = weights.get(p + ".scale_shift_table").unsqueeze(0) +
             time_proj.to(at::kFloat);
  auto shift = mod.select(2, 0), scale = mod.select(2, 1),
       gate = mod.select(2, 2);
  auto normalized =
      (layernorm(x.to(at::kFloat)) * (1 + scale) + shift).to(dtype);
  auto qkv = linear_shared(normalized, weights, {p + ".attn1.to_q", p + ".attn1.to_k", p + ".attn1.to_v"});
  auto q = heads(rms(std::move(qkv[0]),
                     weights.get(p + ".attn1.norm_q.weight"), 1e-6));
  auto k = heads(rms(std::move(qkv[1]),
                     weights.get(p + ".attn1.norm_k.weight"), 1e-6));
  auto v = heads(std::move(qkv[2]));
  if (!cache_k.defined()) {
    cache_k = at::empty({B, 24, 4 * N, 128}, k.options());
    cache_v = at::empty_like(cache_k);
  }
  auto slot = (frame % 4) * N;
  cache_k.narrow(2, slot, N).copy_(k);
  cache_v.narrow(2, slot, N).copy_(v);
  q = rotate(q, qr);
  k = rotate(cache_k.narrow(2, 0, valid * N), kr);
  v = cache_v.narrow(2, 0, valid * N);
  auto self = attention(q, k, v, mask);
  auto a = linear(merge(self), weights, p + ".attn1.to_out.0");
  x = (x.to(at::kFloat) + a * gate).to(dtype);
  normalized = layernorm(x.to(at::kFloat),
                         weights.get(p + ".norm2.weight").to(at::kFloat),
                         weights.get(p + ".norm2.bias").to(at::kFloat))
                   .to(dtype);
  q = heads(rms(linear(normalized, weights, p + ".attn2.to_q"),
                weights.get(p + ".attn2.norm_q.weight"), 1e-6));
  x = x + linear(merge(attention(q, cross_k, cross_v)), weights,
                 p + ".attn2.to_out.0");
  normalized =
      (layernorm(x.to(at::kFloat)) * (1 + mod.select(2, 4)) + mod.select(2, 3))
          .to(dtype);
  auto ff = linear(linear_gelu(normalized, weights, p + ".ffn.net.0.proj"),
                   weights, p + ".ffn.net.2");
  x = (x.to(at::kFloat) + ff.to(at::kFloat) * mod.select(2, 5)).to(dtype);

  return x;
}
}  // namespace
Wan::Wan(Weights& weights) : w_(weights) {
  auto prompt = w_.get("prompt_embeds");
  prompt = prompt.unsqueeze(0);
  auto freq = at::arange(
      128, at::TensorOptions().dtype(at::kFloat).device(prompt.device()));
  auto phase = at::exp((freq * (-std::log(10000.0))) / 128.0) * 1000.0;
  auto t = at::cat({phase.cos(), phase.sin()})
               .reshape({1, 1, 256})
               .to(prompt.device(), prompt.scalar_type());
  temb_ = linear(
      at::silu(linear(t, w_, "condition_embedder.time_embedder.linear_1")), w_,
      "condition_embedder.time_embedder.linear_2");
  time_proj_ = linear(at::silu(temb_), w_, "condition_embedder.time_proj")
                   .reshape({1, 1, 6, 3072});
  auto text = linear(
      at::gelu(linear(prompt, w_, "condition_embedder.text_embedder.linear_1"),
               "tanh"),
      w_, "condition_embedder.text_embedder.linear_2");
  for (int i = 0; i < 30; ++i) {
    w_.with_wan_block(i, WanBlockUse::text, [&] {
      auto p = "blocks." + std::to_string(i) + ".attn2";
      auto kv = linear_shared(text, w_, {p + ".to_k", p + ".to_v"});
      cross_keys_[i] = heads(rms(std::move(kv[0]),
                                 w_.get(p + ".norm_k.weight"), 1e-6));
      cross_values_[i] = heads(std::move(kv[1]));
    });
  }
  w_.release_wan_conditioner();
}
Tensor Wan::step(const Tensor& lq, const Tensor& noise) {
  at::NoGradGuard guard;
  TORCH_CHECK(lq.dim() == 4 && noise.dim() == 5 && noise.size(2) == 1,
              "Wan 条件帧须为 BCHW，噪声须为 BC1HW / Wan expects BCHW "
              "conditioning and BC1HW noise");
  constexpr int condition_stride = 16;
  TORCH_CHECK(lq.size(2) % condition_stride == 0 && lq.size(3) % condition_stride == 0,
              "Invalid Wan conditioning dimensions / Wan 条件帧尺寸无效");
  auto B = noise.size(0), H = noise.size(3) / 2, W = noise.size(4) / 2,
       N = H * W;
  TORCH_CHECK(
      lq.size(2) / condition_stride == H && lq.size(3) / condition_stride == W,
      "Wan 潜变量与条件帧网格不匹配 / Wan latent/conditioning grid mismatch");
  if (frame_ == 0) {
    height_ = H;
    width_ = W;
  }
  TORCH_CHECK(height_ == H && width_ == W,
              "Wan 流式网格已改变，请创建新的状态 / Wan streaming grid "
              "changed; create a fresh Wan state");
  auto dtype = lq.scalar_type();
  // Conv3d kernel has temporal extent one, so its exact operation is Conv2d.
  auto x = at::conv2d(noise.squeeze(2).to(dtype),
                      w_.get("patch_embedding.weight").squeeze(2),
                      w_.get("patch_embedding.bias"), {2, 2},
                      at::IntArrayRef{0, 0}, {1, 1}, 1)
               .flatten(2)
               .transpose(1, 2);
  auto low = at::pixel_unshuffle(lq, condition_stride / 2);
  low = at::silu(
      rms2(conv(low, w_, "lq_proj.conv1"), w_.get("lq_proj.norm1.weight")));
  low = at::silu(rms2(conv(low, w_, "lq_proj.conv2", 2, 1),
                      w_.get("lq_proj.norm2.weight")));
  x = x + linear(low.flatten(2).transpose(1, 2), w_, "lq_proj.out_proj");
  low = Tensor();
  int64_t valid = std::min<int64_t>(frame_ + 1, 4),
          origin = std::max<int64_t>(0, frame_ - 7);
  std::vector<int64_t> positions;
  for (int64_t slot = 0; slot < valid; ++slot)
    positions.push_back(frame_ - ((frame_ - slot) % 4) - origin);
  auto qr = rope(H, W, {frame_ - origin}, x), kr = rope(H, W, positions, x);
  auto idx = at::arange(
      valid * N, at::TensorOptions().dtype(at::kLong).device(x.device()));
  auto ky = at::floor_divide(idx.remainder(N), W).unsqueeze(0),
       kx = idx.remainder(W).unsqueeze(0);
  int64_t wh = std::min<int64_t>(22, H), ww = std::min<int64_t>(40, W);
  std::map<std::pair<int64_t, int64_t>, Tensor> window_masks;
  auto mask = [&](int64_t start, int64_t end) {
    const auto range = std::make_pair(start, end);
    if (auto existing = window_masks.find(range); existing != window_masks.end())
      return existing->second;
    auto qi = at::arange(start, end, idx.options());
    auto top = (at::floor_divide(qi, W) - wh / 2).clamp(0, H - wh).unsqueeze(1);
    auto left = (qi.remainder(W) - ww / 2).clamp(0, W - ww).unsqueeze(1);
    auto visible = (ky >= top) & (ky < top + wh) & (kx >= left) & (kx < left + ww);
    window_masks.emplace(range, visible);
    return visible;
  };
  for (int i = 0; i < 30; ++i) {
    w_.with_wan_block(i, WanBlockUse::endpoint, [&] {
      x = run_block(w_, i, x, time_proj_, cross_keys_[i], cross_values_[i],
                    keys_[i], values_[i], frame_, H, W, qr, kr, mask);
    });
  }
  // Unlike block modulation, the output scale/shift addition occurs in BF16.
  auto mod = w_.get("scale_shift_table").unsqueeze(0) + temb_.unsqueeze(2);
  x = (layernorm(x.to(at::kFloat)) * (1 + mod.select(2, 1)) + mod.select(2, 0))
          .to(dtype);
  x = linear(x, w_, "proj_out");
  auto pred = x.reshape({B, 1, H, W, 1, 2, 2, 48})
                  .permute({0, 7, 1, 4, 2, 5, 3, 6})
                  .reshape({B, 48, 1, H * 2, W * 2});
  ++frame_;
  return (noise - pred.to(at::kFloat)).squeeze(2).to(dtype);
}
}  // namespace rv

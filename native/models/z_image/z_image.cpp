#include "z_image.hpp"

#include "../../media/image.hpp"
#include "../../platform/apple/platform.hpp"
#include "../../runtime/residency.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <regex>

namespace tc {
namespace {

constexpr int kTextHeads = 32;
constexpr int kTextKVHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kHeads = 30;
constexpr float kVaeScale = 0.3611f;
constexpr float kVaeShift = 0.1159f;

std::string z_diffusers_transformer_key(std::string key) {
    for (const auto *prefix : {"transformer.", "diffusion_model."})
        if (key.starts_with(prefix)) {
            key.erase(0, std::strlen(prefix));
            break;
        }
    if (key.starts_with("all_x_embedder.2-1."))
        key.replace(0, std::strlen("all_x_embedder.2-1."), "x_embedder.");
    else if (key.starts_with("all_final_layer.2-1."))
        key.replace(0, std::strlen("all_final_layer.2-1."), "final_layer.");
    key = std::regex_replace(key, std::regex("\\.attention\\.to_out\\.0\\."),
                             ".attention.out.");
    key = std::regex_replace(key, std::regex("\\.attention\\.norm_q\\."),
                             ".attention.q_norm.");
    key = std::regex_replace(key, std::regex("\\.attention\\.norm_k\\."),
                             ".attention.k_norm.");
    return key;
}

std::string z_diffusers_vae_key(std::string key) {
    key = std::regex_replace(key, std::regex("^vae\\."), "");
    key = std::regex_replace(key, std::regex("^decoder\\.mid_block\\.resnets\\.0\\."),
                             "decoder.mid.block_1.");
    key = std::regex_replace(key, std::regex("^decoder\\.mid_block\\.resnets\\.1\\."),
                             "decoder.mid.block_2.");
    key = std::regex_replace(key, std::regex("^decoder\\.mid_block\\.attentions\\.0\\.group_norm\\."),
                             "decoder.mid.attn_1.norm.");
    key = std::regex_replace(key, std::regex("^decoder\\.mid_block\\.attentions\\.0\\.to_q\\."),
                             "decoder.mid.attn_1.q.");
    key = std::regex_replace(key, std::regex("^decoder\\.mid_block\\.attentions\\.0\\.to_k\\."),
                             "decoder.mid.attn_1.k.");
    key = std::regex_replace(key, std::regex("^decoder\\.mid_block\\.attentions\\.0\\.to_v\\."),
                             "decoder.mid.attn_1.v.");
    key = std::regex_replace(key, std::regex("^decoder\\.mid_block\\.attentions\\.0\\.to_out\\.0\\."),
                             "decoder.mid.attn_1.proj_out.");
    std::smatch match;
    if (std::regex_match(key, match,
                         std::regex("^decoder\\.up_blocks\\.([0-9]+)\\.(.*)$"))) {
        const int stage = 3 - std::stoi(match[1]);
        auto tail = std::string(match[2]);
        tail = std::regex_replace(tail, std::regex("^resnets\\.([0-9]+)\\.conv_shortcut\\."),
                                  "block.$1.nin_shortcut.");
        tail = std::regex_replace(tail, std::regex("^resnets\\.([0-9]+)\\."),
                                  "block.$1.");
        tail = std::regex_replace(tail, std::regex("^upsamplers\\.0\\.conv\\."),
                                  "upsample.conv.");
        key = "decoder.up." + std::to_string(stage) + "." + tail;
    }
    key = std::regex_replace(key, std::regex("^decoder\\.conv_norm_out\\."),
                             "decoder.norm_out.");
    return key;
}

void normalize_z_diffusers_transformer(Weights &weights) {
    if (weights.has("x_embedder.weight"))
        return;
    weights.remap_keys(z_diffusers_transformer_key);
    for (const char *group : {"noise_refiner", "context_refiner", "layers"}) {
        const int count = std::string(group) == "layers" ? 30 : 2;
        for (int i = 0; i < count; ++i) {
            const auto prefix = std::string(group) + "." + std::to_string(i) + ".attention.";
            weights.fuse_keys(prefix + "qkv.weight",
                              {prefix + "to_q.weight", prefix + "to_k.weight",
                               prefix + "to_v.weight"}, 0);
        }
    }
}

void normalize_z_diffusers_vae(Weights &weights) {
    if (!weights.has("decoder.mid.block_1.conv1.weight"))
        weights.remap_keys(z_diffusers_vae_key);
}

bool has_safetensors(const std::filesystem::path &directory) {
    if (!std::filesystem::is_directory(directory))
        return false;
    for (const auto &entry : std::filesystem::directory_iterator(directory))
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors")
            return true;
    return false;
}

void load_z_component(Weights &weights, const std::filesystem::path &path,
                      const Event &event, std::atomic<bool> &cancelled) {
    if (std::filesystem::is_directory(path))
        weights.load(path, event, cancelled);
    else
        weights.load_file(path);
}

Tensor linear_compat(const Tensor &x, const Weights &w, const std::string &prefix) {
    auto weight = w.at(prefix + ".weight");
    if (weight.ndim() != 2)
        weight = mx::reshape(weight, {weight.shape(0), int(weight.size() / weight.shape(0))});
    auto output = mx::matmul(x, mx::transpose(weight));
    if (w.has(prefix + ".bias"))
        output = output + w.at(prefix + ".bias");
    return output;
}

Tensor rope_text(const Tensor &x, const Tensor &cos, const Tensor &sin) {
    // Qwen3 uses rotate-half RoPE, not the interleaved complex pairs used by
    // the Z-Image DiT below.
    auto halves = mx::split(x, 2, -1);
    auto rotated = mx::concatenate({-halves[1], halves[0]}, -1);
    return x * mx::expand_dims(cos, 1) + rotated * mx::expand_dims(sin, 1);
}

Tensor qwen_attention(const Tensor &x, const Weights &w, const std::string &prefix,
                      const Tensor &cos, const Tensor &sin, const Tensor &mask) {
    auto q = heads(linear_compat(x, w, prefix + ".q_proj"), kTextHeads, kHeadDim);
    auto k = heads(linear_compat(x, w, prefix + ".k_proj"), kTextKVHeads, kHeadDim);
    auto v = heads(linear_compat(x, w, prefix + ".v_proj"), kTextKVHeads, kHeadDim);
    q = rope_text(rms(q, w.at(prefix + ".q_norm.weight"), 1e-6f), cos, sin);
    k = rope_text(rms(k, w.at(prefix + ".k_norm.weight"), 1e-6f), cos, sin);
    k = mx::repeat(k, kTextHeads / kTextKVHeads, 1);
    v = mx::repeat(v, kTextHeads / kTextKVHeads, 1);
    return linear_compat(attend(q, k, v, true, mask), w, prefix + ".o_proj");
}

Tensor qwen_encode(const Tensor &ids, const Weights &w, int valid, const Event &event,
                   std::atomic<bool> &cancelled) {
    const int n = ids.shape(1);
    auto x = mx::astype(mx::take(w.at("model.embed_tokens.weight"), ids, 0), mx::float32);
    auto freq = 1.f / mx::power(Tensor(1000000.f), mx::arange(0, kHeadDim, 2, mx::float32) /
                                                       float(kHeadDim));
    auto angle = mx::reshape(mx::arange(n, mx::float32), {1, n, 1}) *
                 mx::reshape(freq, {1, 1, kHeadDim / 2});
    auto doubled = mx::concatenate({angle, angle}, -1);
    auto cos = mx::cos(doubled);
    auto sin = mx::sin(doubled);
    auto qi = mx::reshape(mx::arange(n, mx::int32), {n, 1});
    auto ki = mx::reshape(mx::arange(n, mx::int32), {1, n});
    auto forbidden = mx::logical_or(ki > qi, ki >= Tensor(valid));
    auto mask = mx::reshape(mx::where(forbidden, Tensor(-INFINITY, mx::float32),
                                      Tensor(0.f, mx::float32)),
                            {1, 1, n, n});
    // Qwen3Model returns all_hidden_states[-2].  That value is the output of
    // layer 34, so the final (36th) layer and final RMSNorm are dead work.
    for (int i = 0; i < 35; ++i) {
        checkpoint(cancelled);
        event("z_image_text_encode", i, 35);
        auto p = "model.layers." + std::to_string(i);
        auto residual = x;
        x = residual + qwen_attention(rms(x, w.at(p + ".input_layernorm.weight"), 1e-6f), w,
                                      p + ".self_attn", cos, sin, mask);
        residual = x;
        auto a = rms(x, w.at(p + ".post_attention_layernorm.weight"), 1e-6f);
        x = residual + linear_compat(
                            silu(linear_compat(a, w, p + ".mlp.gate_proj")) *
                                linear_compat(a, w, p + ".mlp.up_proj"),
                            w, p + ".mlp.down_proj");
        mx::eval(x);
    }
    event("z_image_text_encode", 35, 35);
    return mx::astype(x, mx::bfloat16);
}

Tensor z_group_norm(const Tensor &x, const Weights &w, const std::string &prefix, int groups) {
    const int b = x.shape(0), c = x.shape(1), h = x.shape(2), width = x.shape(3);
    require(c % groups == 0, "Z-Image VAE group count does not divide channels");
    auto nhwc = mx::transpose(x, {0, 2, 3, 1});
    auto grouped = mx::reshape(nhwc, {b, h, width, groups, c / groups});
    grouped = mx::transpose(grouped, {0, 3, 1, 2, 4});
    auto flat = mx::reshape(grouped, {b, groups, h * width * (c / groups)});
    auto mean = mx::mean(flat, -1, true);
    auto variance = mx::mean(mx::square(flat - mean), -1, true);
    auto normalized = mx::reshape((flat - mean) * mx::rsqrt(variance + 1e-6f),
                                  {b, groups, h, width, c / groups});
    normalized = mx::transpose(normalized, {0, 2, 3, 1, 4});
    normalized = mx::reshape(normalized, {b, h, width, c});
    auto gamma = mx::reshape(w.at(prefix + ".weight"), {1, 1, 1, c});
    auto beta = mx::reshape(w.at(prefix + ".bias"), {1, 1, 1, c});
    return mx::astype(mx::transpose(normalized * gamma + beta, {0, 3, 1, 2}), x.dtype());
}

Tensor z_conv(const Tensor &x, const Weights &w, const std::string &prefix, int padding = 1) {
    auto weight = w.at(prefix + ".weight");
    require(weight.ndim() == 4, "Z-Image VAE convolution must be rank four");
    weight = mx::transpose(weight, {0, 2, 3, 1});
    auto nhwc = mx::transpose(x, {0, 2, 3, 1});
    auto y = mx::conv2d(nhwc, weight, {1, 1}, {padding, padding});
    if (w.has(prefix + ".bias"))
        y = y + mx::reshape(w.at(prefix + ".bias"), {1, 1, 1, y.shape(3)});
    return mx::transpose(y, {0, 3, 1, 2});
}

Tensor z_resnet(const Tensor &input, const Weights &w, const std::string &prefix) {
    auto value = z_conv(silu(z_group_norm(input, w, prefix + ".norm1", 32)),
                        w, prefix + ".conv1");
    value = z_conv(silu(z_group_norm(value, w, prefix + ".norm2", 32)),
                   w, prefix + ".conv2");
    auto skip = input;
    if (w.has(prefix + ".nin_shortcut.weight"))
        skip = z_conv(input, w, prefix + ".nin_shortcut", 0);
    return value + skip;
}

Tensor z_vae_attention(const Tensor &input, const Weights &w, const std::string &prefix) {
    auto normalized = z_group_norm(input, w, prefix + ".norm", 32);
    const int b = normalized.shape(0), h = normalized.shape(2), width = normalized.shape(3),
              c = normalized.shape(1), n = h * width;
    auto seq = mx::reshape(mx::transpose(normalized, {0, 2, 3, 1}), {b, n, c});
    auto projection = [&](const std::string &name) {
        auto value = w.at(prefix + "." + name + ".weight");
        value = mx::reshape(value, {value.shape(0), int(value.size() / value.shape(0))});
        auto output = mx::matmul(seq, mx::transpose(value));
        return output + w.at(prefix + "." + name + ".bias");
    };
    auto q = mx::transpose(mx::reshape(projection("q"), {b, n, 1, c}), {0, 2, 1, 3});
    auto k = mx::transpose(mx::reshape(projection("k"), {b, n, 1, c}), {0, 2, 1, 3});
    auto v = mx::transpose(mx::reshape(projection("v"), {b, n, 1, c}), {0, 2, 1, 3});
    auto result = attend(q, k, v, true);
    auto out_weight = w.at(prefix + ".proj_out.weight");
    out_weight = mx::reshape(out_weight,
                             {out_weight.shape(0), int(out_weight.size() / out_weight.shape(0))});
    result = mx::matmul(result, mx::transpose(out_weight)) + w.at(prefix + ".proj_out.bias");
    result = mx::reshape(result, {b, h, width, c});
    return input + mx::transpose(result, {0, 3, 1, 2});
}

Tensor z_vae_decode(const Tensor &latent, const Weights &w) {
    require(latent.ndim() == 4 && latent.shape(0) == 16 && latent.shape(1) == 1,
            "Z-Image latent must be [16, 1, height, width]");
    auto x = mx::reshape(latent, {1, latent.shape(0), latent.shape(2), latent.shape(3)});
    x = x / Tensor(kVaeScale, x.dtype()) + Tensor(kVaeShift, x.dtype());
    x = z_conv(x, w, "decoder.conv_in");
    x = z_resnet(x, w, "decoder.mid.block_1");
    x = z_vae_attention(x, w, "decoder.mid.attn_1");
    x = z_resnet(x, w, "decoder.mid.block_2");
    // The Comfy single-file VAE stores the high-resolution stages in reverse
    // order relative to the diffusers module list: up.3 -> up.2 -> up.1 -> up.0.
    for (int stage = 3; stage >= 0; --stage) {
        const int in_blocks = 3;
        for (int block = 0; block < in_blocks; ++block) {
            x = z_resnet(x, w, "decoder.up." + std::to_string(stage) + ".block." +
                                   std::to_string(block));
            mx::eval(x);
        }
        if (stage > 0) {
            x = mx::repeat(mx::repeat(x, 2, 2), 2, 3);
            x = z_conv(x, w, "decoder.up." + std::to_string(stage) + ".upsample.conv");
        }
    }
    x = silu(z_group_norm(x, w, "decoder.norm_out", 32));
    return z_conv(x, w, "decoder.conv_out");
}

Tensor z_apply_rope(const Tensor &x, const Tensor &freqs) {
    // x: [1, heads, sequence, 128], freqs: [sequence, 64, 2].
    auto pair = mx::reshape(mx::astype(x, mx::float32),
                            {x.shape(0), x.shape(1), x.shape(2), x.shape(3) / 2, 2});
    auto f = mx::reshape(freqs, {1, 1, x.shape(2), x.shape(3) / 2, 2});
    auto a = mx::squeeze(slice_axis(pair, -1, 0, 1), -1);
    auto b = mx::squeeze(slice_axis(pair, -1, 1, 2), -1);
    auto c = mx::squeeze(slice_axis(f, -1, 0, 1), -1);
    auto s = mx::squeeze(slice_axis(f, -1, 1, 2), -1);
    return mx::astype(mx::reshape(mx::stack({a * c - b * s, a * s + b * c}, -1), x.shape()),
                      x.dtype());
}

Tensor z_rope(const Tensor &ids) {
    const int axes[] = {32, 48, 48};
    const int limits[] = {1536, 512, 512};
    std::vector<Tensor> parts;
    for (int axis = 0; axis < 3; ++axis) {
        auto d = axes[axis];
        auto inv = 1.f / mx::power(Tensor(256.f), mx::arange(0, d, 2, mx::float32) / float(d));
        auto positions = mx::take(mx::arange(limits[axis], mx::float32),
                                   mx::astype(slice_axis(ids, 1, axis, axis + 1), mx::int32), 0);
        auto angle = positions * mx::reshape(inv, {1, d / 2});
        parts.push_back(mx::stack({mx::cos(angle), mx::sin(angle)}, -1));
    }
    return mx::concatenate(parts, 1);
}

Tensor z_attention(const Tensor &x, const Weights &w, const std::string &prefix,
                   const Tensor &freqs) {
    auto qkv = linear_compat(x, w, prefix + ".attention.qkv");
    auto chunks = mx::split(qkv, 3, -1);
    auto q = z_apply_rope(rms(heads(chunks[0], kHeads, kHeadDim),
                              w.at(prefix + ".attention.q_norm.weight"), 1e-5f), freqs);
    auto k = z_apply_rope(rms(heads(chunks[1], kHeads, kHeadDim),
                              w.at(prefix + ".attention.k_norm.weight"), 1e-5f), freqs);
    auto v = heads(chunks[2], kHeads, kHeadDim);
    auto result = attend(q, k, v, false);
    return linear_compat(result, w, prefix + ".attention.out");
}

Tensor z_ffn(const Tensor &x, const Weights &w, const std::string &prefix) {
    return linear_compat(silu(linear_compat(x, w, prefix + ".w1")) *
                             linear_compat(x, w, prefix + ".w3"),
                         w, prefix + ".w2");
}

Tensor z_context_block(const Tensor &x, const Weights &w, const std::string &prefix,
                       const Tensor &freqs) {
    auto attention = z_attention(rms(x, w.at(prefix + ".attention_norm1.weight"), 1e-5f),
                                 w, prefix, freqs);
    auto value = x + rms(attention, w.at(prefix + ".attention_norm2.weight"), 1e-5f);
    auto feed = z_ffn(rms(value, w.at(prefix + ".ffn_norm1.weight"), 1e-5f), w,
                      prefix + ".feed_forward");
    return value + rms(feed, w.at(prefix + ".ffn_norm2.weight"), 1e-5f);
}

Tensor z_block(const Tensor &x, const Weights &w, const std::string &prefix,
               const Tensor &freqs, const Tensor &temb) {
    auto modulation = mx::expand_dims(linear_compat(temb, w, prefix + ".adaLN_modulation.0"), 1);
    auto parts = mx::split(modulation, 4, -1);
    auto scale_msa = Tensor(1.f, parts[0].dtype()) + parts[0];
    auto gate_msa = mx::tanh(parts[1]);
    auto scale_mlp = Tensor(1.f, parts[2].dtype()) + parts[2];
    auto gate_mlp = mx::tanh(parts[3]);
    auto attention = z_attention(rms(x, w.at(prefix + ".attention_norm1.weight"), 1e-5f) *
                                     scale_msa,
                                 w, prefix, freqs);
    auto value = x + gate_msa * rms(attention, w.at(prefix + ".attention_norm2.weight"), 1e-5f);
    auto feed = z_ffn(rms(value, w.at(prefix + ".ffn_norm1.weight"), 1e-5f) * scale_mlp,
                      w, prefix + ".feed_forward");
    return value + gate_mlp * rms(feed, w.at(prefix + ".ffn_norm2.weight"), 1e-5f);
}

Tensor z_timestep(float timestep, const Weights &w) {
    const int n = 256;
    auto half = n / 2;
    auto freq = mx::exp(-std::log(10000.f) * mx::arange(0, half, mx::float32) / float(half));
    auto args = Tensor(timestep) * freq;
    auto embedding = mx::reshape(mx::concatenate({mx::cos(args), mx::sin(args)}, -1), {1, n});
    return linear_compat(silu(linear_compat(embedding, w, "t_embedder.mlp.0")), w,
                         "t_embedder.mlp.2");
}

struct ZPatch {
    Tensor image;
    Tensor caption;
    Tensor image_ids;
    Tensor caption_ids;
    int image_length = 0;
    int caption_length = 0;
    int image_h = 0;
    int image_w = 0;
};

ZPatch z_patchify(const Tensor &latent, const Tensor &caption) {
    const int c = latent.shape(0), frames = latent.shape(1), h = latent.shape(2), width = latent.shape(3);
    const int ph = h / 2, pw = width / 2;
    auto image = mx::reshape(latent, {c, frames, ph, 2, pw, 2});
    image = mx::transpose(image, {1, 2, 4, 3, 5, 0});
    image = mx::reshape(image, {frames * ph * pw, 4 * c});
    const int image_length = image.shape(0);
    const int image_pad = (32 - image_length % 32) % 32;
    if (image_pad)
        image = mx::concatenate({image,
                                 mx::repeat(slice_axis(image, 0, image_length - 1, image_length),
                                            image_pad, 0)}, 0);

    int cap_length = caption.shape(0);
    const int cap_pad = (32 - cap_length % 32) % 32;
    auto cap = caption;
    if (cap_pad)
        cap = mx::concatenate({cap, mx::repeat(slice_axis(cap, 0, cap_length - 1, cap_length),
                                               cap_pad, 0)}, 0);

    std::vector<int32_t> cap_pos((cap_length + cap_pad) * 3, 0);
    for (int i = 0; i < cap_length + cap_pad; ++i)
        cap_pos[i * 3] = i + 1;
    std::vector<int32_t> image_pos(image.shape(0) * 3, 0);
    const int start = cap_length + cap_pad + 1;
    int i = 0;
    for (int f = 0; f < frames; ++f)
        for (int y = 0; y < ph; ++y)
            for (int x = 0; x < pw; ++x, ++i) {
                image_pos[i * 3] = start + f;
                image_pos[i * 3 + 1] = y;
                image_pos[i * 3 + 2] = x;
            }
    // Padded image coordinates are ignored by the pad token but stay inside
    // the legal zero coordinate range of the precomputed RoPE table.
    for (; i < int(image.shape(0)); ++i)
        image_pos[i * 3] = 0;
    return {image, cap, Tensor(image_pos.data(), {int(image.shape(0)), 3}, mx::int32),
            Tensor(cap_pos.data(), {int(cap.shape(0)), 3}, mx::int32), image_length,
            cap_length, ph, pw};
}

Tensor z_transformer(const Tensor &latent, const Tensor &caption, float sigma, int width,
                     int height, const Weights &w, const Event &event,
                     std::atomic<bool> &cancelled) {
    auto patch = z_patchify(latent, caption);
    auto image = linear_compat(patch.image, w, "x_embedder");
    auto caption_emb = linear_compat(
        rms(patch.caption, w.at("cap_embedder.0.weight"), 1e-5f), w, "cap_embedder.1");
    if (image.shape(0) > patch.image_length)
        image = mx::concatenate(
            {slice_axis(image, 0, 0, patch.image_length),
             mx::repeat(w.at("x_pad_token"), image.shape(0) - patch.image_length, 0)}, 0);
    if (caption_emb.shape(0) > patch.caption_length)
        caption_emb = mx::concatenate(
            {slice_axis(caption_emb, 0, 0, patch.caption_length),
             mx::repeat(w.at("cap_pad_token"), caption_emb.shape(0) - patch.caption_length, 0)}, 0);
    auto temb = z_timestep((1.f - sigma) * 1000.f, w);
    auto image_freqs = z_rope(patch.image_ids);
    auto caption_freqs = z_rope(patch.caption_ids);
    image = mx::expand_dims(image, 0);
    caption_emb = mx::expand_dims(caption_emb, 0);
    for (int i = 0; i < 2; ++i) {
        checkpoint(cancelled);
        image = z_block(image, w, "noise_refiner." + std::to_string(i), image_freqs, temb);
        caption_emb = z_context_block(caption_emb, w,
                                      "context_refiner." + std::to_string(i), caption_freqs);
    }
    auto unified = mx::concatenate({image, caption_emb}, 1);
    auto unified_freqs = mx::concatenate({image_freqs, caption_freqs}, 0);
    for (int i = 0; i < 30; ++i) {
        checkpoint(cancelled);
        event("z_image_denoise_block", i, 30);
        unified = z_block(unified, w, "layers." + std::to_string(i), unified_freqs, temb);
        mx::eval(unified);
    }
    auto final_scale = Tensor(1.f, temb.dtype()) +
                       linear_compat(silu(temb), w, "final_layer.adaLN_modulation.1");
    auto final = linear_compat(mx::fast::layer_norm(unified, {}, {}, 1e-6f) * final_scale,
                               w, "final_layer.linear");
    final = slice_axis(final, 1, 0, patch.image_length);
    auto output = mx::reshape(final, {1, patch.image_h, patch.image_w, 1, 2, 2,
                                      latent.shape(0)});
    output = mx::transpose(output, {6, 0, 3, 1, 4, 2, 5});
    output = mx::reshape(output, latent.shape());
    event("z_image_denoise_block", 30, 30);
    return -output;
}

std::vector<float> z_sigmas(int width, int height, int steps) {
    // ComfyUI registers Z-Image as ModelSamplingDiscreteFlow with shift=3.0
    // and the official workflow uses its `simple` scheduler.  This is the
    // fixed-shift flow schedule, not FLUX/MFLUX's resolution-dependent
    // exponential time shift.  Keeping it here makes the native sampler
    // numerically compatible with ComfyUI at 1024² and at other supported
    // dimensions while leaving the execution path entirely native.
    (void)width;
    (void)height;
    constexpr float shift = 3.f;
    constexpr int training_steps = 1000;
    std::vector<float> result;
    result.reserve(size_t(steps) + 1);
    for (int i = 0; i < steps; ++i) {
        // ComfyUI's `simple` scheduler indexes the 1000-entry training sigma
        // table with `-(1 + int(i * 1000 / steps))`; preserve that discrete
        // rounding instead of approximating it with the continuous fraction.
        const int table_step = training_steps - (i * training_steps / steps);
        const float t = float(table_step) / float(training_steps);
        result.push_back(shift * t / (1.f + (shift - 1.f) * t));
    }
    result.push_back(0.f);
    return result;
}

Tensor z_initial_noise(const Request &r, int height, int width) {
    if (r.noise_path.empty()) {
        return mx::random::normal({16, 1, height, width}, mx::float32, 0.f, 1.f,
                                  mx::random::key(r.seed));
    }
    require(std::filesystem::is_regular_file(r.noise_path),
            "initial noise file missing: " + r.noise_path);
    auto loaded = mx::load_safetensors(r.noise_path);
    require(!loaded.first.empty(), "initial noise file contains no tensors");
    auto found = loaded.first.find("noise");
    if (found == loaded.first.end())
        found = loaded.first.find("tensor");
    if (found == loaded.first.end())
        found = loaded.first.find("latent_tensor");
    require(found != loaded.first.end(),
            "initial noise file must contain noise, tensor, or latent_tensor");
    auto noise = found->second;
    require(noise.ndim() == 4, "initial noise must be rank four");
    if (noise.shape(0) == 1 && noise.shape(1) == 16)
        noise = mx::transpose(noise, {1, 0, 2, 3});
    require(noise.shape(0) == 16 && noise.shape(1) == 1 &&
                noise.shape(2) == height && noise.shape(3) == width,
            "initial noise shape must be [1,16,H,W] or [16,1,H,W]");
    // ComfyUI keeps the Euler sampler state in FP32 and casts only the model
    // input to BF16.  Keeping the shared noise and every update in FP32 avoids
    // accumulating a BF16 residual-rounding difference at all nine steps.
    return mx::astype(noise, mx::float32);
}

} // namespace

ZImage::ZImage(const std::filesystem::path &root)
    : root_(root), tokenizer_(root / "tokenizer") {
    auto comfy_text = root / "split_files/text_encoders/qwen_3_4b.safetensors";
    auto comfy_transformer =
        root / "split_files/diffusion_models/z_image_turbo_bf16.safetensors";
    auto comfy_vae = root / "split_files/vae/ae.safetensors";
    if (std::filesystem::is_regular_file(comfy_text) &&
        std::filesystem::is_regular_file(comfy_transformer) &&
        std::filesystem::is_regular_file(comfy_vae)) {
        text_path_ = std::move(comfy_text);
        transformer_path_ = std::move(comfy_transformer);
        vae_path_ = std::move(comfy_vae);
        return;
    }
    diffusers_layout_ = true;
    text_path_ = root / "text_encoder";
    transformer_path_ = root / "transformer";
    vae_path_ = root / "vae";
    require(has_safetensors(text_path_),
            "missing Z-Image Qwen3 safetensors in text_encoder/");
    require(has_safetensors(transformer_path_),
            "missing Z-Image DiT safetensors in transformer/");
    require(has_safetensors(vae_path_), "missing Z-Image VAE safetensors in vae/");
}

void ZImage::select_loras(const Request &request) {
    std::string identity;
    std::vector<LoRAAsset> normalized;
    for (const auto &adapter : request.loras) {
        std::error_code error;
        auto path = std::filesystem::canonical(adapter.path, error);
        require(!error && std::filesystem::is_regular_file(path), "LoRA file missing: " + adapter.path);
        auto bytes = std::filesystem::file_size(path, error);
        require(!error, "cannot inspect LoRA size: " + path.string());
        std::filesystem::last_write_time(path, error);
        require(!error, "cannot inspect LoRA timestamp: " + path.string());
        auto digest = sha256_file(path);
        identity += path.string() + ":" + std::to_string(bytes) + ":" + digest + ":" +
                    adapter.role + ":" + std::to_string(std::bit_cast<uint32_t>(adapter.strength)) + ";";
        auto selected = adapter;
        selected.path = path.string();
        normalized.push_back(std::move(selected));
    }
    if (identity == cached_lora_identity_)
        return;
    cached_lora_identity_ = std::move(identity);
    active_loras_ = std::move(normalized);
    cached_conditioning_.reset();
    cached_prompt_.clear();
    transformer_.clear();
    lora_applied_projections_ = 0;
    mx::clear_cache();
}

LoadResult ZImage::load(const Event &event, std::atomic<bool> &cancelled) {
    const bool transformer_cold = transformer_.bytes() == 0;
    if (transformer_cold) {
        checkpoint(cancelled);
        event("load_z_image_transformer", 0, 1);
        load_z_component(transformer_, transformer_path_, event, cancelled);
        if (diffusers_layout_)
            normalize_z_diffusers_transformer(transformer_);
        event("load_z_image_transformer", 1, 1);
    }
    if (vae_.bytes() == 0) {
        checkpoint(cancelled);
        event("load_z_image_vae", 0, 1);
        load_z_component(vae_, vae_path_, event, cancelled);
        if (diffusers_layout_)
            normalize_z_diffusers_vae(vae_);
        event("load_z_image_vae", 1, 1);
    }
    if (transformer_cold && !active_loras_.empty())
        lora_applied_projections_ =
            transformer_.apply_loras(active_loras_, "transformer", event, cancelled);
    transformer_.materialize();
    vae_.materialize();
    return {uint64_t(transformer_.bytes() + vae_.bytes()), mx::get_active_memory()};
}

void ZImage::unload() {
    cached_conditioning_.reset();
    cached_prompt_.clear();
    text_encoder_.clear();
    transformer_.clear();
    vae_.clear();
    mx::clear_cache();
}

Tensor ZImage::encode_text(const Tokens &tokens, const Event &event, std::atomic<bool> &cancelled) {
    if (text_encoder_.bytes() == 0)
        load_z_component(text_encoder_, text_path_, event, cancelled);
    auto ids = Tensor(tokens.ids.data(), {1, int(tokens.ids.size())}, mx::int32);
    auto result = qwen_encode(ids, text_encoder_, tokens.valid, event, cancelled);
    result = slice_axis(mx::squeeze(result, 0), 0, 0, tokens.valid);
    text_encoder_.clear();
    mx::clear_cache();
    return result;
}

bool ZImage::conditioning(const Request &r, const Event &event, std::atomic<bool> &cancelled) {
    if (cached_conditioning_ && cached_prompt_ == r.prompt && cached_dynamic_ == r.dynamic_text) {
        event("z_image_text_cache_hit", 1, 1);
        return true;
    }
    cached_conditioning_ = encode_text(tokenizer_.z_image_prompt(r.prompt, r.dynamic_text), event, cancelled);
    cached_prompt_ = r.prompt;
    cached_dynamic_ = r.dynamic_text;
    return false;
}

std::string ZImage::select_acceleration(Request &r, const Event &, std::atomic<bool> &) {
    if (r.execution == "auto") {
        r.execution = "gpu";
        return "gpu: Z-Image ANE partition not validated for this checkpoint";
    }
    require(r.execution == "gpu", "Z-Image GPU+ANE is not enabled without a validated profile");
    return "gpu: native MLX single-stream S3-DiT";
}

RunResult ZImage::prepare(const Request &requested, bool warmup, const Event &event,
                          std::atomic<bool> &cancelled) {
    return run(requested, event, cancelled, warmup);
}

RunResult ZImage::generate(const Request &r, const Event &event, std::atomic<bool> &cancelled) {
    return run(r, event, cancelled, false);
}

RunResult ZImage::run(const Request &requested, const Event &event, std::atomic<bool> &cancelled,
                      bool warmup) {
    auto r = requested;
    auto begin = Clock::now();
    require(r.model == "z-image-turbo", "Z-Image session received a different model id");
    auto plan = make_plan(r);
    require(!r.prompt.empty() && (warmup || !r.output.empty()), "prompt and output are required");
    require(warmup || std::filesystem::path(r.output).extension() == ".png",
            "Z-Image output must be .png");
    require(r.inputs.empty(), "Z-Image-Turbo currently supports text-to-image only");
    require(r.width % 16 == 0 && r.height % 16 == 0, "Z-Image dimensions must be multiples of 16");
    select_loras(r);
    auto selection = select_acceleration(r, event, cancelled);
    if (plan.request.execution != r.execution)
        plan = make_plan(r);
    auto text_start = Clock::now();
    bool prompt_hit = conditioning(r, event, cancelled);
    const double text_seconds = std::chrono::duration<double>(Clock::now() - text_start).count();
    load(event, cancelled);
    const int latent_h = r.height / 8, latent_w = r.width / 8;
    auto z = z_initial_noise(r, latent_h, latent_w);
    mx::eval(z);
    auto dump = [&](const std::string &name, const Tensor &value) {
        if (r.dump.empty())
            return;
        std::filesystem::create_directories(r.dump);
        mx::save_safetensors((std::filesystem::path(r.dump) / (name + ".safetensors")).string(),
                             {{"tensor", value}});
    };
    dump("z_latent_initial", z);
    auto sigmas = z_sigmas(r.width, r.height, r.steps);
    const auto caption = *cached_conditioning_;
    auto dit_start = Clock::now();
    for (int i = 0; i < r.steps; ++i) {
        checkpoint(cancelled);
        event("denoise", i, r.steps);
        auto noise = denoise(z, caption, sigmas[i], float(r.width), r.height, i, event, cancelled);
        z = euler_step(z, noise, sigmas[i + 1] - sigmas[i]);
        mx::eval(z);
        dump("z_latent_step_" + std::to_string(i + 1), z);
        require(mx::all(mx::isfinite(z)).item<bool>(), "nonfinite Z-Image latent");
        event("denoise", i + 1, r.steps);
    }
    const double denoise_seconds = std::chrono::duration<double>(Clock::now() - dit_start).count();
    checkpoint(cancelled);
    auto decode_start = Clock::now();
    auto decoded = decode(z, r.width, r.height, event, cancelled);
    dump("z_latent_final", z);
    dump("z_decoded", decoded);
    const double decode_seconds = std::chrono::duration<double>(Clock::now() - decode_start).count();
    require(mx::all(mx::isfinite(decoded)).item<bool>(), "nonfinite Z-Image pixels");
    auto pixels = mx::transpose(decoded, {0, 2, 3, 1});
    if (!warmup) {
        event("export", 0, 1);
        checkpoint(cancelled);
        save_png(pixels, r.output);
        event("export", 1, 1);
    }
    RunResult result;
    result.request = r;
    result.plan = std::move(plan);
    result.selection = selection;
    result.warmup = warmup;
    result.prompt_cache_hit = prompt_hit;
    auto reported_tokens = tokenizer_.z_image_prompt(r.prompt, r.dynamic_text);
    result.text_tokens = int(reported_tokens.ids.size());
    result.valid_text_tokens = reported_tokens.valid;
    result.actual_steps = r.steps;
    result.lora_applied_projections = lora_applied_projections_;
    result.timings.wall = std::chrono::duration<double>(Clock::now() - begin).count();
    result.timings.text = text_seconds;
    result.timings.denoise = denoise_seconds;
    result.timings.decode = decode_seconds;
    result.peak_bytes = mx::get_peak_memory();
    result.active_bytes = mx::get_active_memory();
    return result;
}

Tensor ZImage::denoise(const Tensor &latent, const Tensor &caption, float sigma, float width,
                       int height, int, const Event &event, std::atomic<bool> &cancelled) {
    auto model_input = mx::astype(latent, mx::bfloat16);
    return mx::astype(
        z_transformer(model_input, caption, sigma, int(width), height, transformer_, event,
                      cancelled),
        mx::float32);
}

Tensor ZImage::decode(const Tensor &latent, int, int, const Event &, std::atomic<bool> &) {
    return z_vae_decode(mx::astype(latent, mx::bfloat16), vae_);
}

} // namespace tc

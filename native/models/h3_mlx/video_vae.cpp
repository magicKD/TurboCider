#include "video_vae.hpp"

#include <algorithm>
#include <cmath>

namespace tc::h3_mlx {
namespace {
Tensor ct(const Tensor &value, const std::vector<int> &axes) {
    return mx::contiguous(mx::transpose(value, axes));
}

Tensor silu(const Tensor &value) { return value * mx::sigmoid(value); }

Tensor reflect_pad_axis(const Tensor &value, int axis, int left, int right) {
    if (!left && !right) return value;
    std::vector<Tensor> pieces;
    if (left) {
        std::vector<int32_t> indices;
        for (int index = left; index > 0; --index) indices.push_back(index);
        pieces.push_back(mx::take(value,
                                  Tensor(indices.data(), {int(indices.size())}, mx::int32),
                                  axis));
    }
    pieces.push_back(value);
    if (right) {
        std::vector<int32_t> indices;
        for (int index = 2; index < 2 + right; ++index)
            indices.push_back(value.shape(axis) - index);
        pieces.push_back(mx::take(value,
                                  Tensor(indices.data(), {int(indices.size())}, mx::int32),
                                  axis));
    }
    return mx::contiguous(mx::concatenate(pieces, axis));
}

Tensor linear(const Tensor &value, const Tensor &weight,
              const std::optional<Tensor> &bias = {}) {
    auto output = mx::matmul(value, mx::transpose(weight));
    if (bias) output = output + *bias;
    return output;
}

Tensor rms_no_affine(const Tensor &value, float eps) {
    auto fp32 = mx::astype(value, mx::float32);
    return mx::astype(fp32 * mx::rsqrt(mx::mean(mx::square(fp32), -1, true) + eps),
                      value.dtype());
}

int positive_mod(int value, int divisor) {
    require(divisor > 0, "H3 video VAE modulo divisor must be positive");
    return ((value % divisor) + divisor) % divisor;
}
} // namespace

void VideoVAE::load(const std::filesystem::path &root, const Event &event,
                    std::atomic<bool> &cancelled) {
    config_ = load_video_vae_config(root / "config.json");
    weights_.load(root, {"post_quant_conv.", "decoder."}, mx::float32, true,
                  event, cancelled);
    for (const auto &key : {"post_quant_conv.weight", "post_quant_conv.bias",
                            "decoder.proj_in.weight", "decoder.norm_out.weight",
                            "decoder.norm_out.bias", "decoder.proj_out.weight",
                            "decoder.proj_out.bias", "decoder.register_tokens"})
        require(weights_.has(key), "H3 video VAE is missing tensor: " +
                                    std::string(key));
}

void VideoVAE::unload() { weights_.clear(); }

Tensor VideoVAE::conv3d(const Tensor &input, const std::string &prefix,
                        int temporal_padding, int spatial_padding,
                        const std::array<int, 3> &stride) const {
    auto value = ct(input, {0, 2, 3, 4, 1});
    if (spatial_padding) {
        value = reflect_pad_axis(value, 2, spatial_padding, spatial_padding);
        value = reflect_pad_axis(value, 3, spatial_padding, spatial_padding);
    }
    if (temporal_padding)
        value = mx::pad(value,
                        {{0, 0}, {temporal_padding, 0}, {0, 0}, {0, 0}, {0, 0}},
                        Tensor(0.f, value.dtype()));
    auto output = mx::conv3d(value, weights_.at(prefix + ".weight"),
                             std::tuple<int, int, int>{stride[0], stride[1], stride[2]},
                             std::tuple<int, int, int>{0, 0, 0});
    output = output + mx::reshape(weights_.at(prefix + ".bias"),
                                  {1, 1, 1, 1, output.shape(-1)});
    return ct(output, {0, 4, 1, 2, 3});
}

Tensor VideoVAE::rms_affine(const Tensor &value, const std::string &prefix,
                            float eps) const {
    auto fp32 = mx::astype(value, mx::float32);
    auto normalized = fp32 * mx::rsqrt(mx::mean(mx::square(fp32), -1, true) + eps);
    return normalized * weights_.at(prefix + ".weight");
}

Tensor VideoVAE::layer_norm(const Tensor &value, const std::string &weight,
                            const std::string &bias, float eps) const {
    auto mean = mx::mean(value, -1, true);
    auto centered = value - mean;
    auto variance = mx::mean(mx::square(centered), -1, true);
    return centered * mx::rsqrt(variance + eps) * weights_.at(weight) +
           weights_.at(bias);
}

std::pair<Tensor, Tensor> VideoVAE::rotary(const Tensor &position_ids) const {
    const int rotary_dim = int(config_.head_dim * config_.rope_ratio);
    const int frequency_count = rotary_dim / 6;
    auto indices = mx::arange(0, frequency_count, mx::float32);
    auto inv = 1.f / mx::power(Tensor(config_.rope_theta, mx::float32),
                               indices * (6.f / rotary_dim));
    std::vector<Tensor> axes;
    for (int axis = 0; axis < 3; ++axis)
        axes.push_back(slice_axis(position_ids, 1, axis, axis + 1) * inv);
    auto angles = mx::concatenate(axes, -1);
    angles = 2.f * float(M_PI) * angles;
    angles = mx::concatenate({angles, angles}, -1);
    angles = mx::expand_dims(angles, 1);
    return {mx::cos(angles), mx::sin(angles)};
}

Tensor VideoVAE::attention(const Tensor &input, const std::string &prefix,
                           const Tensor &cosine, const Tensor &sine) const {
    const int sequence = input.shape(0);
    const int batch = input.shape(1);
    auto q = mx::reshape(linear(input, weights_.at(prefix + ".attn.to_q.weight"),
                                weights_.has(prefix + ".attn.to_q.bias")
                                    ? std::optional<Tensor>(weights_.at(prefix + ".attn.to_q.bias"))
                                    : std::nullopt),
                         {sequence, batch, config_.num_heads, config_.head_dim});
    auto k = mx::reshape(linear(input, weights_.at(prefix + ".attn.to_k.weight"),
                                weights_.has(prefix + ".attn.to_k.bias")
                                    ? std::optional<Tensor>(weights_.at(prefix + ".attn.to_k.bias"))
                                    : std::nullopt),
                         {sequence, batch, config_.num_heads, config_.head_dim});
    auto v = mx::reshape(linear(input, weights_.at(prefix + ".attn.to_v.weight"),
                                weights_.has(prefix + ".attn.to_v.bias")
                                    ? std::optional<Tensor>(weights_.at(prefix + ".attn.to_v.bias"))
                                    : std::nullopt),
                         {sequence, batch, config_.num_heads, config_.head_dim});
    q = rms_no_affine(q, config_.norm_epsilon);
    k = rms_no_affine(k, config_.norm_epsilon);
    const int rotary_dim = cosine.shape(-1);
    auto rotate = [&](const Tensor &value, const Tensor &cos, const Tensor &sin) {
        auto rot = mx::astype(slice_axis(value, -1, 0, rotary_dim), mx::float32);
        auto pass = slice_axis(value, -1, rotary_dim, value.shape(-1));
        auto halves = mx::split(rot, 2, -1);
        auto rotated = mx::concatenate({-halves[1], halves[0]}, -1);
        auto c = mx::expand_dims(cos, 1);
        auto s = mx::expand_dims(sin, 1);
        return mx::concatenate({mx::astype(rot * c + rotated * s, value.dtype()), pass}, -1);
    };
    q = rotate(q, cosine, sine);
    k = rotate(k, cosine, sine);
    q = mx::transpose(q, {1, 2, 0, 3});
    k = mx::transpose(k, {1, 2, 0, 3});
    v = mx::transpose(v, {1, 2, 0, 3});
    auto attended = mx::fast::scaled_dot_product_attention(
        q, k, v, 1.f / std::sqrt(float(config_.head_dim)));
    attended = mx::transpose(attended, {2, 0, 1, 3});
    attended = mx::reshape(attended,
                           {sequence, batch, config_.num_heads * config_.head_dim});
    return linear(attended, weights_.at(prefix + ".attn.to_out.0.weight"),
                  weights_.has(prefix + ".attn.to_out.0.bias")
                      ? std::optional<Tensor>(weights_.at(prefix + ".attn.to_out.0.bias"))
                      : std::nullopt);
}

Tensor VideoVAE::feed_forward(const Tensor &input,
                              const std::string &prefix) const {
    auto hidden = linear(input, weights_.at(prefix + ".ff.net.0.proj.weight"),
                         weights_.has(prefix + ".ff.net.0.proj.bias")
                             ? std::optional<Tensor>(weights_.at(prefix + ".ff.net.0.proj.bias"))
                             : std::nullopt);
    auto halves = mx::split(hidden, 2, -1);
    return linear(halves[0] * silu(halves[1]),
                  weights_.at(prefix + ".ff.net.2.weight"),
                  weights_.has(prefix + ".ff.net.2.bias")
                      ? std::optional<Tensor>(weights_.at(prefix + ".ff.net.2.bias"))
                      : std::nullopt);
}

Tensor VideoVAE::decode_clip(const Tensor &latents, const Event &event,
                             std::atomic<bool> &cancelled) const {
    auto z = conv3d(latents, "post_quant_conv", 0);
    const int batch = z.shape(0), frames = z.shape(2), height = z.shape(3), width = z.shape(4);
    require(batch == 1, "H3 video VAE decoder currently requires batch size one");
    auto tokens = ct(z, {0, 2, 3, 4, 1});
    tokens = mx::reshape(tokens, {-1, tokens.shape(-1)});
    const int patches = tokens.shape(0);
    tokens = linear(tokens, weights_.at("decoder.proj_in.weight"),
                    weights_.at("decoder.proj_in.bias"));
    auto registers = mx::broadcast_to(weights_.at("decoder.register_tokens"),
                                      {batch, config_.register_tokens,
                                       config_.num_heads * config_.head_dim});
    registers = mx::reshape(registers, {-1, registers.shape(-1)});
    auto cls = mx::zeros({batch, config_.num_heads * config_.head_dim}, tokens.dtype());
    auto hidden = mx::expand_dims(mx::concatenate({tokens, registers, cls}, 0), 1);
    std::vector<float> positions;
    positions.reserve(size_t(frames * height * width + config_.register_tokens + 1) * 3);
    auto grid = [](int size) {
        std::vector<float> result(size);
        for (int index = 0; index < size; ++index)
            result[index] = 2.f * ((float(index) + .5f) / float(size)) - 1.f;
        return result;
    };
    auto ts = grid(frames), hs = grid(height), ws = grid(width);
    for (float t : ts) for (float h : hs) for (float w : ws) {
        positions.push_back(t); positions.push_back(h); positions.push_back(w);
    }
    const int suffix = config_.register_tokens + 1;
    for (int index = 0; index < suffix; ++index)
        positions.insert(positions.end(), {0.f, 0.f, 0.f});
    auto position_ids = Tensor(positions.data(), {int(positions.size() / 3), 3}, mx::float32);
    auto [cosine, sine] = rotary(position_ids);
    for (int index = 0; index < config_.decoder_layers; ++index) {
        checkpoint(cancelled);
        event("h3_mlx_video_decode", index, config_.decoder_layers);
        const auto prefix = "decoder.transformer_blocks." + std::to_string(index);
        auto normalized = rms_affine(hidden, prefix + ".norm1", config_.norm_epsilon);
        auto attended = attention(normalized, prefix, cosine, sine);
        hidden = hidden + attended * weights_.at(prefix + ".scale1");
        normalized = rms_affine(hidden, prefix + ".norm2", config_.norm_epsilon);
        hidden = hidden + feed_forward(normalized, prefix) * weights_.at(prefix + ".scale2");
        mx::eval(hidden);
    }
    hidden = layer_norm(hidden, "decoder.norm_out.weight", "decoder.norm_out.bias",
                        config_.norm_epsilon);
    auto output = linear(hidden, weights_.at("decoder.proj_out.weight"),
                         weights_.at("decoder.proj_out.bias"));
    output = slice_axis(output, 0, 0, patches);
    output = mx::reshape(output, {patches, 3, config_.temporal_ratio,
                                  config_.spatial_ratio, config_.spatial_ratio});
    output = mx::reshape(output, {batch, frames, height, width, 3,
                                  config_.temporal_ratio, config_.spatial_ratio,
                                  config_.spatial_ratio});
    output = ct(output, {0, 4, 1, 5, 2, 6, 3, 7});
    return mx::reshape(output, {batch, 3, frames * config_.temporal_ratio,
                                height * config_.spatial_ratio,
                                width * config_.spatial_ratio});
}

std::vector<int> VideoVAE::split_tiles(int length, int tile_size,
                                       int min_overlap, int compression,
                                       std::vector<int> &overlaps) {
    if (tile_size >= length) { overlaps.clear(); return {0}; }
    int count = int(std::ceil(double(length) / tile_size));
    while (tile_size * count - min_overlap * (count - 1) - length < 0) ++count;
    overlaps.assign(count - 1, min_overlap);
    int remaining = tile_size * count - min_overlap * (count - 1) - length;
    for (int index = 0; index < remaining / compression; ++index)
        overlaps[index % (count - 1)] += compression;
    std::vector<int> starts{0};
    for (int index = 0; index < count - 1; ++index)
        starts.push_back(starts.back() + tile_size - overlaps[index]);
    return starts;
}

Tensor VideoVAE::blend(const Tensor &a, const Tensor &b, int extent, int axis) {
    extent = std::min({a.shape(axis), b.shape(axis), extent});
    if (extent <= 0) return b;
    auto positions = mx::arange(extent, b.dtype());
    mx::Shape shape(a.ndim(), 1); shape[axis < 0 ? axis + a.ndim() : axis] = extent;
    auto wa = mx::reshape(1.f - positions / float(extent), shape);
    auto wb = mx::reshape(positions / float(extent), shape);
    int normalized_axis = axis < 0 ? axis + a.ndim() : axis;
    auto av = slice_axis(a, normalized_axis, a.shape(axis) - extent, a.shape(axis));
    auto bv = slice_axis(b, normalized_axis, 0, extent);
    auto mixed = av * wa + bv * wb;
    if (extent == b.shape(axis)) return mixed;
    return mx::concatenate({mixed, slice_axis(b, normalized_axis, extent, b.shape(axis))}, axis);
}

Tensor VideoVAE::decode_clip_tiled(const Tensor &latents, int tile_height,
                                   int tile_width, int overlap_height,
                                   int overlap_width, const Event &event,
                                   std::atomic<bool> &cancelled) const {
    std::vector<int> h_overlaps, w_overlaps;
    const int height = latents.shape(-2) * config_.spatial_ratio;
    const int width = latents.shape(-1) * config_.spatial_ratio;
    auto h_starts = split_tiles(height, tile_height, overlap_height,
                                 config_.spatial_ratio, h_overlaps);
    auto w_starts = split_tiles(width, tile_width, overlap_width,
                                config_.spatial_ratio, w_overlaps);
    std::vector<std::vector<Tensor>> rows;
    for (size_t row = 0; row < h_starts.size(); ++row) {
        std::vector<Tensor> values;
        for (size_t col = 0; col < w_starts.size(); ++col) {
            const int h = std::min(tile_height, height - h_starts[row]);
            const int w = std::min(tile_width, width - w_starts[col]);
            auto tile = slice_axis(slice_axis(latents, -2,
                                               h_starts[row] / config_.spatial_ratio,
                                               h_starts[row] / config_.spatial_ratio + h / config_.spatial_ratio),
                                   -1, w_starts[col] / config_.spatial_ratio,
                                   w_starts[col] / config_.spatial_ratio + w / config_.spatial_ratio);
            values.push_back(decode_clip(tile, event, cancelled));
        }
        rows.push_back(std::move(values));
    }
    std::vector<Tensor> stitched_rows;
    for (size_t row = 0; row < rows.size(); ++row) {
        std::vector<Tensor> stitched;
        for (size_t col = 0; col < rows[row].size(); ++col) {
            auto value = rows[row][col];
            if (row) value = blend(rows[row - 1][col], value, h_overlaps[row - 1], -2);
            if (col) value = blend(rows[row][col - 1], value, w_overlaps[col - 1], -1);
            if (row + 1 < rows.size()) value = slice_axis(value, -2, 0, value.shape(-2) - h_overlaps[row]);
            if (col + 1 < rows[row].size()) value = slice_axis(value, -1, 0, value.shape(-1) - w_overlaps[col]);
            stitched.push_back(value);
        }
        stitched_rows.push_back(mx::concatenate(stitched, -1));
    }
    return mx::concatenate(stitched_rows, -2);
}

Tensor VideoVAE::denormalize_latents(const Tensor &latents) const {
    auto mean = Tensor(config_.latent_mean.data(), {1, config_.latent_channels, 1, 1, 1}, mx::float32);
    auto std = Tensor(config_.latent_std.data(), {1, config_.latent_channels, 1, 1, 1}, mx::float32);
    return latents * std + mean;
}

Tensor VideoVAE::denormalize_pixels(const Tensor &pixels) const {
    static constexpr float mean[] = {.485f, .456f, .406f};
    static constexpr float std[] = {.229f, .224f, .225f};
    auto m = Tensor(mean, {1, 3, 1, 1, 1}, mx::float32);
    auto s = Tensor(std, {1, 3, 1, 1, 1}, mx::float32);
    return pixels * s + m;
}

Tensor VideoVAE::decode(const Tensor &latents, int target_frames,
                        int target_height, int target_width, bool tiled,
                        const Event &event, std::atomic<bool> &cancelled) const {
    require(loaded(), "H3 video VAE is not loaded");
    require(latents.ndim() == 5 && latents.shape(1) == config_.latent_channels,
            "invalid H3 video VAE latent geometry");
    const int tokens_chunk = (config_.clip_length + config_.temporal_ratio - 1) /
                             config_.temporal_ratio;
    const int token_overlap = positive_mod(-config_.token_drop, tokens_chunk);
    const int frame_pre_padding =
        positive_mod(-config_.clip_length, config_.temporal_ratio);
    const int frame_overlap = std::max(
        token_overlap * config_.temporal_ratio - frame_pre_padding, 0);
    int num_tokens = latents.shape(2) + config_.token_drop;
    int pad_tokens = positive_mod(-num_tokens, tokens_chunk);
    auto padded = latents;
    if (pad_tokens)
        padded = mx::concatenate({padded,
                                  mx::repeat(slice_axis(padded, 2, padded.shape(2) - 1, padded.shape(2)),
                                             pad_tokens, 2)}, 2);
    const int chunks = (num_tokens + pad_tokens) / tokens_chunk -
                       int(config_.token_drop > 0);
    std::vector<Tensor> decoded;
    std::optional<Tensor> overlap;
    const int min_h = std::min(target_height, 256);
    const int min_w = std::min(target_width, 256);
    for (int index = 0; index < chunks; ++index) {
        checkpoint(cancelled);
        auto clip = slice_axis(padded, 2, index * tokens_chunk,
                               index * tokens_chunk + tokens_chunk + token_overlap);
        auto pixels = tiled ? decode_clip_tiled(clip, min_h, min_w, 64, 64,
                                                event, cancelled)
                            : decode_clip(clip, event, cancelled);
        const int chunk_frames = tokens_chunk * config_.temporal_ratio;
        for (int overlap_index = 0; overlap_index < (config_.token_drop > 0 ? 2 : 1); ++overlap_index) {
            int start = overlap_index * chunk_frames;
            auto part = slice_axis(pixels, 2, start, start + chunk_frames);
            part = slice_axis(part, 2, frame_pre_padding, part.shape(2));
            if (!overlap_index) {
                if (overlap) part = blend(*overlap, part, frame_overlap, 2);
                mx::eval(part);
                decoded.push_back(part);
            } else {
                overlap = part;
                mx::eval(*overlap);
            }
        }
    }
    if (overlap) decoded.push_back(*overlap);
    require(!decoded.empty(), "H3 video VAE produced no chunks");
    auto output = mx::concatenate(decoded, 2);
    if (pad_tokens) {
        int intra_tail = config_.clip_length % config_.temporal_ratio;
        int before = padded.shape(2) - pad_tokens;
        int trim = 0;
        for (int offset = 0; offset < pad_tokens; ++offset)
            trim += intra_tail && ((before + offset) % tokens_chunk == 0)
                        ? intra_tail : config_.temporal_ratio;
        output = slice_axis(output, 2, 0, output.shape(2) - trim);
    }
    output = slice_axis(output, 2, 0, std::min(output.shape(2), target_frames));
    mx::eval(output);
    return output;
}

} // namespace tc::h3_mlx

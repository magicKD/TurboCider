#include "z_image.hpp"

#include "../../media/image.hpp"
#include "../../platform/apple/platform.hpp"
#include "../../runtime/acceleration.hpp"
#include "../../runtime/residency.hpp"
#include "../../components/text/qwen3.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <regex>

namespace tc {
namespace {

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
        if ((entry.is_regular_file() || entry.is_symlink()) &&
            entry.path().extension() == ".safetensors")
            return true;
    return false;
}

void load_z_component(Weights &weights, const std::filesystem::path &path,
                      const Event &event, std::atomic<bool> &cancelled) {
    if (std::filesystem::is_directory(path)) {
        // A few diffusers snapshots expose a convenience symlink such as
        // text_encoder/model.safetensors -> a shared ComfyUI file.  Generic
        // directory loading intentionally ignores symlinks so an index folder
        // cannot load the same checkpoint twice; for a component directory
        // containing only that symlink, load the explicit file once instead.
        bool has_regular = false;
        std::filesystem::path linked;
        for (const auto &entry : std::filesystem::directory_iterator(path)) {
            if (!entry.is_symlink() && entry.is_regular_file() &&
                entry.path().extension() == ".safetensors")
                has_regular = true;
            else if (entry.is_symlink() && entry.path().extension() == ".safetensors" &&
                     linked.empty())
                linked = entry.path();
        }
        if (!has_regular && !linked.empty())
            weights.load_file(linked);
        else
            weights.load(path, event, cancelled);
    }
    else
        weights.load_file(path);
}

Tensor linear_compat(const Tensor &x, const Weights &w, const std::string &prefix) {
    return w.project(x, prefix);
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

Tensor z_apply_rope_reference(const Tensor &x, const Tensor &freqs) {
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

std::vector<Tensor> z_apply_rope_pair(const Tensor &q, const Tensor &k,
                                      const Tensor &freqs) {
    require(q.shape() == k.shape() && q.ndim() == 4 && q.shape(-1) == kHeadDim,
            "Z-Image Q/K RoPE geometry mismatch");
    if (std::getenv("TURBOCIDER_Z_EAGER_ROPE"))
        return {z_apply_rope_reference(q, freqs), z_apply_rope_reference(k, freqs)};

    // One native MLX Metal dispatch rotates Q and K together.  The frequency
    // table is shared across batch/head rows, while each pair is accumulated
    // in FP32 and stored in the original activation dtype.  This removes the
    // eager cast/reshape/slice/stack chain from every attention block without
    // introducing a PyTorch or third-party runtime dependency.
    static auto kernel = mx::fast::metal_kernel(
        "tc_z_image_rope_qk", {"q", "k", "freqs", "pair_count", "frequency_span"},
        {"q_out", "k_out"},
        "uint pair = thread_position_in_grid.x; "
        "if (pair < uint(pair_count)) { "
        "  uint fi = pair % uint(frequency_span); "
        "  uint base = pair * 2; uint fbase = fi * 2; "
        "  float c = float(freqs[fbase]); float s = float(freqs[fbase + 1]); "
        "  float qa = float(q[base]); float qb = float(q[base + 1]); "
        "  float ka = float(k[base]); float kb = float(k[base + 1]); "
        "  q_out[base] = T(qa * c - qb * s); "
        "  q_out[base + 1] = T(qa * s + qb * c); "
        "  k_out[base] = T(ka * c - kb * s); "
        "  k_out[base + 1] = T(ka * s + kb * c); "
        "}");
    const int pairs = int(q.size() / 2);
    const int frequency_span = q.shape(2) * (q.shape(3) / 2);
    return kernel({q, k, freqs, Tensor(pairs), Tensor(frequency_span)},
                  {q.shape(), k.shape()}, {q.dtype(), k.dtype()}, {pairs, 1, 1},
                  {256, 1, 1}, {{"T", q.dtype()}}, {}, false, {});
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
    if (std::getenv("TURBOCIDER_Z_CONVROT_DEBUG")) {
        mx::eval({qkv});
        auto qkv32 = mx::astype(qkv, mx::float32);
        std::fprintf(stderr, "convrot_debug attention=%s qkv_finite=%d qkv_max=%g\n",
                     prefix.c_str(), mx::all(mx::isfinite(qkv32)).item<bool>(),
                     mx::max(mx::abs(qkv32)).item<float>());
    }
    auto chunks = mx::split(qkv, 3, -1);
    auto q = rms(heads(chunks[0], kHeads, kHeadDim),
                 w.at(prefix + ".attention.q_norm.weight"), 1e-5f);
    auto k = rms(heads(chunks[1], kHeads, kHeadDim),
                 w.at(prefix + ".attention.k_norm.weight"), 1e-5f);
    auto rotated = z_apply_rope_pair(q, k, freqs);
    auto v = heads(chunks[2], kHeads, kHeadDim);
    auto result = attend(rotated[0], rotated[1], v, false, {},
                         !std::getenv("TURBOCIDER_Z_DISABLE_FUSED_SDPA"));
    if (std::getenv("TURBOCIDER_Z_CONVROT_DEBUG")) {
        mx::eval({rotated[0], rotated[1], v, result});
        auto q32 = mx::astype(rotated[0], mx::float32);
        auto k32 = mx::astype(rotated[1], mx::float32);
        auto v32 = mx::astype(v, mx::float32);
        auto result32 = mx::astype(result, mx::float32);
        std::fprintf(stderr,
                     "convrot_debug attention=%s q_finite=%d q_max=%g k_finite=%d k_max=%g v_finite=%d v_max=%g sdpa_finite=%d sdpa_max=%g\n",
                     prefix.c_str(), mx::all(mx::isfinite(q32)).item<bool>(),
                     mx::max(mx::abs(q32)).item<float>(),
                     mx::all(mx::isfinite(k32)).item<bool>(), mx::max(mx::abs(k32)).item<float>(),
                     mx::all(mx::isfinite(v32)).item<bool>(), mx::max(mx::abs(v32)).item<float>(),
                     mx::all(mx::isfinite(result32)).item<bool>(),
                     mx::max(mx::abs(result32)).item<float>());
    }
    auto output = linear_compat(result, w, prefix + ".attention.out");
    if (std::getenv("TURBOCIDER_Z_CONVROT_DEBUG")) {
        mx::eval({output});
        auto output32 = mx::astype(output, mx::float32);
        std::fprintf(stderr, "convrot_debug attention=%s out_finite=%d out_max=%g\n",
                     prefix.c_str(), mx::all(mx::isfinite(output32)).item<bool>(),
                     mx::max(mx::abs(output32)).item<float>());
    }
    return output;
}

Tensor z_ffn(const Tensor &x, const Weights &w, const std::string &prefix) {
    return linear_compat(silu(linear_compat(x, w, prefix + ".w1")) *
                             linear_compat(x, w, prefix + ".w3"),
                         w, prefix + ".w2");
}

float z_hybrid_output_scale(const HybridSession *hybrid) {
    return hybrid ? hybrid->output_scale : 1.f;
}

std::function<std::vector<Tensor>(const std::vector<Tensor> &)>
make_z_hybrid_gpu_graph(int hidden, int mlp_width, int gpu_mlp_start) {
    require(hidden == 3840 && mlp_width == 10240 && gpu_mlp_start > 0 &&
                gpu_mlp_start < mlp_width,
            "unsupported Z-Image hybrid FFN geometry");
    return mx::compile(
        [hidden, mlp_width, gpu_mlp_start](const std::vector<Tensor> &args) {
            auto w1 = slice_axis(args[1], 0, gpu_mlp_start, mlp_width);
            auto w3 = slice_axis(args[2], 0, gpu_mlp_start, mlp_width);
            auto w2 = slice_axis(args[3], 1, gpu_mlp_start, mlp_width);
            auto gate = mx::matmul(args[0], mx::transpose(w1));
            auto up = mx::matmul(args[0], mx::transpose(w3));
            auto value = mx::matmul(silu(gate) * up, mx::transpose(w2));
            require(value.shape(-1) == hidden, "Z-Image hybrid FFN output mismatch");
            return std::vector<Tensor>{value};
    });
}

struct ZQuantizedGeometry {
    int group_size = 0;
    int bits = 0;
};

ZQuantizedGeometry z_quantized_geometry(const Tensor &weight, const Tensor &scales,
                                        int logical_input) {
    require(weight.ndim() == 2 && scales.ndim() == 2 && logical_input > 0 &&
                logical_input % scales.shape(1) == 0 &&
                (weight.shape(1) * 32) % logical_input == 0,
            "invalid Z-Image GGUF affine geometry");
    const int group_size = logical_input / scales.shape(1);
    const int bits = weight.shape(1) * 32 / logical_input;
    require(group_size == 32 && (bits == 4 || bits == 8),
            "Z-Image hybrid requires native Q4_0/Q4_1/Q8_0 affine weights");
    return {group_size, bits};
}

Tensor z_hybrid_output_range(const Tensor &x, const Weights &w,
                             const std::string &prefix, int start, int end) {
    if (w.convrot(prefix)) {
        const auto &weight = w.at(prefix + ".weight");
        const int logical_input = weight.dtype() == mx::uint32
                                      ? weight.shape(1) * 4
                                      : weight.shape(1);
        return w.project_range(x, prefix, start, end, 0, logical_input);
    }
    if (!w.quantized(prefix)) {
        auto weight = slice_axis(w.at(prefix + ".weight"), 0, start, end);
        return mx::matmul(x, mx::transpose(weight));
    }
    const auto &full_weight = w.at(prefix + ".weight");
    const auto &full_scales = w.at(prefix + ".scales");
    auto geometry = z_quantized_geometry(full_weight, full_scales, x.shape(-1));
    auto weight = slice_axis(full_weight, 0, start, end);
    auto scales = slice_axis(full_scales, 0, start, end);
    std::optional<Tensor> biases;
    if (w.has(prefix + ".biases"))
        biases = slice_axis(w.at(prefix + ".biases"), 0, start, end);
    return mx::quantized_matmul(x, weight, scales, biases, true,
                                geometry.group_size, geometry.bits, "affine");
}

Tensor z_hybrid_input_range(const Tensor &x, const Weights &w,
                            const std::string &prefix, int full_input,
                            int start, int end) {
    if (w.convrot(prefix))
        return w.project_range(x, prefix, 0, w.at(prefix + ".weight").shape(0), start, end);
    if (!w.quantized(prefix)) {
        auto weight = slice_axis(w.at(prefix + ".weight"), 1, start, end);
        return mx::matmul(x, mx::transpose(weight));
    }
    const auto &full_weight = w.at(prefix + ".weight");
    const auto &full_scales = w.at(prefix + ".scales");
    auto geometry = z_quantized_geometry(full_weight, full_scales, full_input);
    require(start % geometry.group_size == 0 && end % geometry.group_size == 0 &&
                (start * geometry.bits) % 32 == 0 && (end * geometry.bits) % 32 == 0,
            "Z-Image hybrid split must align to the GGUF quantization group");
    auto weight = slice_axis(full_weight, 1, start * geometry.bits / 32,
                             end * geometry.bits / 32);
    auto scales = slice_axis(full_scales, 1, start / geometry.group_size,
                             end / geometry.group_size);
    std::optional<Tensor> biases;
    if (w.has(prefix + ".biases"))
        biases = slice_axis(w.at(prefix + ".biases"), 1,
                            start / geometry.group_size, end / geometry.group_size);
    return mx::quantized_matmul(x, weight, scales, biases, true,
                                geometry.group_size, geometry.bits, "affine");
}

Tensor z_hybrid_gpu_suffix(const Tensor &x, const Weights &w,
                           const std::string &prefix, int start, int end) {
    auto gate = z_hybrid_output_range(x, w, prefix + ".w1", start, end);
    auto up = z_hybrid_output_range(x, w, prefix + ".w3", start, end);
    return z_hybrid_input_range(silu(gate) * up, w, prefix + ".w2", end,
                                start, end);
}

std::function<std::vector<Tensor>(const std::vector<Tensor> &)> &z_gpu_block_graph() {
    static auto graph = mx::compile([](const std::vector<Tensor> &args) {
        require(args.size() == 16, "invalid Z-Image compiled block inputs");
        auto fast_rms = [](const Tensor &x, const Tensor &weight) {
            return mx::astype(
                mx::fast::rms_norm(mx::astype(x, mx::float32),
                                   mx::astype(weight, mx::float32), 1e-5f),
                x.dtype());
        };
        auto modulation = mx::expand_dims(
            mx::matmul(args[2], mx::transpose(args[3])) + args[4], 1);
        auto mod = mx::split(modulation, 4, -1);
        auto attention_input =
            fast_rms(args[0], args[5]) * (Tensor(1.f, mod[0].dtype()) + mod[0]);
        auto qkv = mx::matmul(attention_input, mx::transpose(args[6]));
        auto qkv_parts = mx::split(qkv, 3, -1);
        auto q = fast_rms(heads(qkv_parts[0], kHeads, kHeadDim), args[7]);
        auto k = fast_rms(heads(qkv_parts[1], kHeads, kHeadDim), args[8]);
        auto rotated = z_apply_rope_pair(q, k, args[1]);
        auto attention = mx::matmul(
            attend(rotated[0], rotated[1], heads(qkv_parts[2], kHeads, kHeadDim), false, {},
                   !std::getenv("TURBOCIDER_Z_DISABLE_FUSED_SDPA")),
            mx::transpose(args[9]));
        auto value = args[0] + mx::tanh(mod[1]) * fast_rms(attention, args[10]);
        auto feed_input =
            fast_rms(value, args[11]) * (Tensor(1.f, mod[2].dtype()) + mod[2]);
        auto gate = mx::matmul(feed_input, mx::transpose(args[12]));
        auto up = mx::matmul(feed_input, mx::transpose(args[13]));
        auto feed = mx::matmul((gate * mx::sigmoid(gate)) * up,
                               mx::transpose(args[14]));
        return std::vector<Tensor>{
            value + mx::tanh(mod[3]) * fast_rms(feed, args[15])};
    });
    return graph;
}

Tensor z_compiled_gpu_block(const Tensor &x, const Weights &w, const std::string &prefix,
                            const Tensor &freqs, const Tensor &temb) {
    return z_gpu_block_graph()(
        {x, freqs, temb,
         w.at(prefix + ".adaLN_modulation.0.weight"),
         w.at(prefix + ".adaLN_modulation.0.bias"),
         w.at(prefix + ".attention_norm1.weight"),
         w.at(prefix + ".attention.qkv.weight"),
         w.at(prefix + ".attention.q_norm.weight"),
         w.at(prefix + ".attention.k_norm.weight"),
         w.at(prefix + ".attention.out.weight"),
         w.at(prefix + ".attention_norm2.weight"),
         w.at(prefix + ".ffn_norm1.weight"),
         w.at(prefix + ".feed_forward.w1.weight"),
         w.at(prefix + ".feed_forward.w3.weight"),
         w.at(prefix + ".feed_forward.w2.weight"),
         w.at(prefix + ".ffn_norm2.weight")})[0];
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
               const Tensor &freqs, const Tensor &temb, HybridSession *hybrid,
               int hybrid_block,
               const std::function<std::vector<Tensor>(const std::vector<Tensor> &)> *gpu_graph) {
    // A LoRA can dequantize only the projections it touches.  Do not infer
    // that the whole block is dense from QKV/w1 alone: Q8 GGUF modulation or
    // the remaining attention/FFN weights may still be packed affine tensors.
    // The compiled graph accepts ordinary dense matrices only; mixed
    // dense/quantized blocks stay on linear_compat below.
    const bool fully_dense =
        !w.quantized(prefix + ".adaLN_modulation.0") &&
        !w.convrot(prefix + ".adaLN_modulation.0") &&
        !w.quantized(prefix + ".attention.qkv") &&
        !w.convrot(prefix + ".attention.qkv") &&
        !w.quantized(prefix + ".attention.out") &&
        !w.convrot(prefix + ".attention.out") &&
        !w.quantized(prefix + ".feed_forward.w1") &&
        !w.convrot(prefix + ".feed_forward.w1") &&
        !w.quantized(prefix + ".feed_forward.w2") &&
        !w.convrot(prefix + ".feed_forward.w2") &&
        !w.quantized(prefix + ".feed_forward.w3") &&
        !w.convrot(prefix + ".feed_forward.w3");
    if (!hybrid && !std::getenv("TURBOCIDER_Z_EAGER_BLOCKS") && fully_dense &&
        !w.has_runtime_loras())
        return z_compiled_gpu_block(x, w, prefix, freqs, temb);
    auto modulation = mx::expand_dims(linear_compat(temb, w, prefix + ".adaLN_modulation.0"), 1);
    if (std::getenv("TURBOCIDER_Z_CONVROT_DEBUG")) {
        mx::eval({temb, modulation});
        std::fprintf(stderr,
                     "convrot_debug block=%s temb_finite=%d temb_max=%g modulation_finite=%d modulation_max=%g\n",
                     prefix.c_str(), mx::all(mx::isfinite(temb)).item<bool>(),
                     mx::max(mx::abs(temb)).item<float>(),
                     mx::all(mx::isfinite(modulation)).item<bool>(),
                     mx::max(mx::abs(modulation)).item<float>());
    }
    auto parts = mx::split(modulation, 4, -1);
    auto scale_msa = Tensor(1.f, parts[0].dtype()) + parts[0];
    auto gate_msa = mx::tanh(parts[1]);
    auto scale_mlp = Tensor(1.f, parts[2].dtype()) + parts[2];
    auto gate_mlp = mx::tanh(parts[3]);
    auto attention = z_attention(rms(x, w.at(prefix + ".attention_norm1.weight"), 1e-5f) *
                                     scale_msa,
                                 w, prefix, freqs);
    auto value = x + gate_msa * rms(attention, w.at(prefix + ".attention_norm2.weight"), 1e-5f);
    auto feed_input = rms(value, w.at(prefix + ".ffn_norm1.weight"), 1e-5f) * scale_mlp;
    if (std::getenv("TURBOCIDER_Z_CONVROT_DEBUG")) {
        mx::eval({attention, value, feed_input});
        std::fprintf(stderr,
                     "convrot_debug block=%s attention_finite=%d attention_max=%g value_finite=%d value_max=%g feed_input_finite=%d feed_input_max=%g\n",
                     prefix.c_str(), mx::all(mx::isfinite(attention)).item<bool>(),
                     mx::max(mx::abs(attention)).item<float>(),
                     mx::all(mx::isfinite(value)).item<bool>(), mx::max(mx::abs(value)).item<float>(),
                     mx::all(mx::isfinite(feed_input)).item<bool>(),
                     mx::max(mx::abs(feed_input)).item<float>());
    }
    Tensor feed = feed_input;
    if (hybrid && gpu_graph) {
        auto packed = mx::astype(feed_input, mx::float16);
        const int actual_rows = packed.shape(1);
        if (actual_rows < hybrid->rows)
            packed = mx::concatenate(
                {packed, mx::zeros({1, hybrid->rows - actual_rows, 3840}, mx::float16)}, 1);
        mx::eval({feed_input, packed});
        const auto ffn = prefix + ".feed_forward";
        const bool dense = !w.quantized(ffn + ".w1") && !w.convrot(ffn + ".w1") &&
                           !w.quantized(ffn + ".w2") && !w.convrot(ffn + ".w2") &&
                           !w.quantized(ffn + ".w3") && !w.convrot(ffn + ".w3");
        auto gpu = dense
            ? (*gpu_graph)({feed_input, w.at(ffn + ".w1.weight"),
                            w.at(ffn + ".w3.weight"), w.at(ffn + ".w2.weight")})[0]
            : z_hybrid_gpu_suffix(feed_input, w, ffn, hybrid->ane_mlp_end,
                                  hybrid->mlp_width);
        mx::async_eval({gpu});
        auto ane = slice_axis(hybrid->predict(hybrid_block, packed), 1, 0, actual_rows);
        auto ane_scaled = mx::astype(ane, gpu.dtype()) * Tensor(z_hybrid_output_scale(hybrid), gpu.dtype());
        feed = gpu + ane_scaled;
        if (std::getenv("TURBOCIDER_Z_HYBRID_VALIDATE")) {
            auto reference = z_ffn(feed_input, w, prefix + ".feed_forward");
            mx::eval({gpu, ane, feed, reference});
            const bool gpu_finite = mx::all(mx::isfinite(gpu)).item<bool>();
            const bool ane_finite = mx::all(mx::isfinite(ane)).item<bool>();
            const bool ane_scaled_finite = mx::all(mx::isfinite(ane_scaled)).item<bool>();
            const bool feed_finite = mx::all(mx::isfinite(feed)).item<bool>();
            const float gpu_max = mx::max(mx::abs(gpu)).item<float>();
            const float ane_max = mx::max(mx::abs(ane_scaled)).item<float>();
            const float reference_max = mx::max(mx::abs(reference)).item<float>();
            const float mae = mx::mean(mx::abs(feed - reference)).item<float>();
            if (const char *directory = std::getenv("TURBOCIDER_Z_HYBRID_DUMP");
                directory && hybrid_block == 0) {
                std::filesystem::create_directories(directory);
                auto root = std::filesystem::path(directory);
                mx::save_safetensors((root / "input.safetensors").string(), {{"x", packed}});
                mx::save_safetensors((root / "ane.safetensors").string(), {{"y", ane}});
                mx::save_safetensors((root / "reference.safetensors").string(),
                                     {{"y", reference}});
            }
            std::fprintf(stderr,
                         "z_hybrid block=%d gpu_finite=%d ane_finite=%d ane_scaled_finite=%d feed_finite=%d "
                         "gpu_max=%g ane_max=%g reference_max=%g mae=%g\n",
                         hybrid_block, gpu_finite, ane_finite, ane_scaled_finite, feed_finite, gpu_max, ane_max,
                         reference_max, mae);
            require(gpu_finite && ane_finite && ane_scaled_finite && feed_finite,
                    "nonfinite Z-Image hybrid FFN at block " + std::to_string(hybrid_block));
        }
    } else {
        feed = z_ffn(feed_input, w, prefix + ".feed_forward");
    }
    if (std::getenv("TURBOCIDER_Z_CONVROT_DEBUG")) {
        mx::eval({feed});
        std::fprintf(stderr, "convrot_debug block=%s feed_finite=%d feed_max=%g\n",
                     prefix.c_str(), mx::all(mx::isfinite(feed)).item<bool>(),
                     mx::max(mx::abs(feed)).item<float>());
    }
    auto result = value + gate_mlp * rms(feed, w.at(prefix + ".ffn_norm2.weight"), 1e-5f);
    if (std::getenv("TURBOCIDER_Z_CONVROT_DEBUG")) {
        mx::eval({result});
        const bool finite = mx::all(mx::isfinite(result)).item<bool>();
        const float max_abs = mx::max(mx::abs(result)).item<float>();
        std::fprintf(stderr, "convrot_debug block=%s finite=%d max=%g\n",
                     prefix.c_str(), finite, max_abs);
        require(finite, "nonfinite ConvRot block output: " + prefix);
    }
    return result;
}

Tensor z_timestep(float timestep, const Weights &w) {
    const int n = 256;
    auto half = n / 2;
    auto freq = mx::exp(-std::log(10000.f) * mx::arange(0, half, mx::float32) / float(half));
    auto args = Tensor(timestep) * freq;
    // Keep the tiny timestep MLP in FP32 for parity with the established
    // ComfyUI oracle, then cast its 256-value result to the model dtype.  The
    // output cast is the important performance boundary: without it every
    // block's scale/gate arithmetic promotes the full activation to FP32.
    auto embedding =
        mx::reshape(mx::concatenate({mx::cos(args), mx::sin(args)}, -1), {1, n});
    return mx::astype(
        linear_compat(silu(linear_compat(embedding, w, "t_embedder.mlp.0")), w,
                      "t_embedder.mlp.2"),
        mx::bfloat16);
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
                     std::atomic<bool> &cancelled, HybridSession *hybrid,
                     const std::function<std::vector<Tensor>(const std::vector<Tensor> &)> *gpu_graph) {
    auto patch = z_patchify(latent, caption);
    auto image = linear_compat(patch.image, w, "x_embedder");
    auto caption_emb = linear_compat(
        rms(patch.caption, w.at("cap_embedder.0.weight"), 1e-5f), w, "cap_embedder.1");
    if (std::getenv("TURBOCIDER_Z_CONVROT_DEBUG")) {
        mx::eval({image, caption_emb});
        std::fprintf(stderr, "convrot_debug inputs image_finite=%d image_max=%g caption_finite=%d caption_max=%g\n",
                     mx::all(mx::isfinite(image)).item<bool>(), mx::max(mx::abs(image)).item<float>(),
                     mx::all(mx::isfinite(caption_emb)).item<bool>(), mx::max(mx::abs(caption_emb)).item<float>());
    }
    if (image.shape(0) > patch.image_length)
        image = mx::concatenate(
            {slice_axis(image, 0, 0, patch.image_length),
             mx::repeat(w.at("x_pad_token"), image.shape(0) - patch.image_length, 0)}, 0);
    if (caption_emb.shape(0) > patch.caption_length)
        caption_emb = mx::concatenate(
            {slice_axis(caption_emb, 0, 0, patch.caption_length),
             mx::repeat(w.at("cap_pad_token"), caption_emb.shape(0) - patch.caption_length, 0)}, 0);
    // The ConvRot checkpoint stores unquantized tensors as FP32, while Comfy
    // executes the transformer with BF16 manual-cast semantics.  Pin this
    // boundary after embedding and padding so those storage dtypes cannot
    // promote every residual and modulation tensor to FP32.
    image = mx::astype(image, mx::bfloat16);
    caption_emb = mx::astype(caption_emb, mx::bfloat16);
    auto temb = z_timestep((1.f - sigma) * 1000.f, w);
    auto image_freqs = z_rope(patch.image_ids);
    auto caption_freqs = z_rope(patch.caption_ids);
    image = mx::expand_dims(image, 0);
    caption_emb = mx::expand_dims(caption_emb, 0);
    for (int i = 0; i < 2; ++i) {
        checkpoint(cancelled);
        image = z_block(image, w, "noise_refiner." + std::to_string(i), image_freqs, temb,
                        hybrid, i, gpu_graph);
        caption_emb = z_context_block(caption_emb, w,
                                      "context_refiner." + std::to_string(i), caption_freqs);
    }
    auto unified = mx::concatenate({image, caption_emb}, 1);
    auto unified_freqs = mx::concatenate({image_freqs, caption_freqs}, 0);
    for (int i = 0; i < 30; ++i) {
        checkpoint(cancelled);
        event("z_image_denoise_block", i, 30);
        unified = z_block(unified, w, "layers." + std::to_string(i), unified_freqs, temb,
                          hybrid, 2 + i, gpu_graph);
        // Compiled blocks retain the allocator dependency chain, so pure GPU
        // execution does not need a host synchronization after every one of
        // the 270 main blocks in a 9-step request. The sampler synchronizes at
        // the end of every denoise step, which remains a bounded cancellation
        // point. Hybrid execution must synchronize around its Core ML calls;
        // the eager compatibility path keeps its former per-block behavior.
        const bool eager = std::getenv("TURBOCIDER_Z_EAGER_BLOCKS");
        if (eager || hybrid) {
            mx::eval(unified);
            checkpoint(cancelled);
        }
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
    : ZImage(root, "z-image-turbo", {}) {}

ZImage::ZImage(const std::filesystem::path &root, std::string model_id,
               const std::filesystem::path &transformer_checkpoint)
    : root_(root), model_id_(std::move(model_id)), tokenizer_(root / "tokenizer") {
    auto comfy_text = root / "split_files/text_encoders/qwen_3_4b.safetensors";
    auto comfy_transformer =
        root / "split_files/diffusion_models/z_image_turbo_bf16.safetensors";
    if (const char *override_path = std::getenv("TURBOCIDER_Z_IMAGE_TRANSFORMER");
        override_path && *override_path)
        comfy_transformer = std::filesystem::canonical(override_path);
    convrot_transformer_ = comfy_transformer.filename().string().find("convrot") !=
                           std::string::npos;
    auto comfy_vae = root / "split_files/vae/ae.safetensors";
    if (!transformer_checkpoint.empty()) {
        require(std::filesystem::is_regular_file(transformer_checkpoint) &&
                    transformer_checkpoint.extension() == ".gguf",
                "native Z-Image GGUF transformer checkpoint is invalid");
        transformer_path_ = std::filesystem::canonical(transformer_checkpoint);
        transformer_checkpoint_ = transformer_path_;
        gguf_transformer_ = true;
        if (std::filesystem::is_regular_file(comfy_text) &&
            std::filesystem::is_regular_file(comfy_vae)) {
            text_path_ = std::move(comfy_text);
            vae_path_ = std::move(comfy_vae);
            return;
        }
        text_path_ = root / "text_encoder";
        vae_path_ = root / "vae";
        require(has_safetensors(text_path_),
                "missing Z-Image Qwen3 safetensors in text_encoder/");
        require(has_safetensors(vae_path_), "missing Z-Image VAE safetensors in vae/");
        return;
    }
    if (std::filesystem::is_regular_file(comfy_transformer) &&
        std::filesystem::is_regular_file(comfy_vae)) {
        // Comfy checkpoints may share a sharded Qwen3 encoder through the
        // App's text_encoder/ directory binding instead of a single file.
        text_path_ = std::filesystem::is_regular_file(comfy_text)
                         ? comfy_text : root / "text_encoder";
        require(std::filesystem::is_regular_file(text_path_) || has_safetensors(text_path_),
                "missing Z-Image Qwen3 weights; select a shared text model in the App");
        transformer_path_ = std::move(comfy_transformer);
        transformer_checkpoint_ = transformer_path_;
        vae_path_ = std::move(comfy_vae);
        return;
    }
    diffusers_layout_ = true;
    text_path_ = root / "text_encoder";
    transformer_path_ = root / "transformer";
    // Indexed diffusers checkpoints stay sharded. Core ML provenance binds the
    // index plus every shard instead of requiring a merged 24 GB file.
    auto diffusers_checkpoint = transformer_path_ / "diffusion_pytorch_model.safetensors";
    auto diffusers_index = transformer_path_ / "diffusion_pytorch_model.safetensors.index.json";
    if (std::filesystem::is_regular_file(diffusers_index))
        transformer_checkpoint_ = std::move(diffusers_index);
    else if (std::filesystem::is_regular_file(diffusers_checkpoint))
        transformer_checkpoint_ = std::move(diffusers_checkpoint);
    vae_path_ = root / "vae";
    require(has_safetensors(text_path_),
            "missing Z-Image Qwen3 safetensors in text_encoder/");
    require(has_safetensors(transformer_path_),
            "missing Z-Image DiT safetensors in transformer/");
    require(has_safetensors(vae_path_), "missing Z-Image VAE safetensors in vae/");
}

void ZImage::select_loras(const Request &request) {
    const auto strategy = effective_lora_strategy(request);
    std::string identity = "strategy:" + strategy + ";";
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
    active_lora_strategy_ = strategy;
    hybrid_.reset();
    hybrid_gpu_graph_ = {};
    hybrid_gpu_mlp_start_ = -1;
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
        if (gguf_transformer_)
            transformer_.load_gguf_file(transformer_path_);
        else
            load_z_component(transformer_, transformer_path_, event, cancelled);
        if (diffusers_layout_)
            normalize_z_diffusers_transformer(transformer_);
        convrot_transformer_ = transformer_.convrot("layers.0.attention.qkv");
        if (convrot_transformer_)
            transformer_.cast_unquantized_float32(mx::bfloat16);
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
            transformer_.apply_loras(active_loras_, "transformer", event, cancelled,
                                     active_lora_strategy_ == "inference_time");
    if (transformer_cold && convrot_transformer_) {
        // Apply LoRA against the original ConvRot representation first. The
        // touched projections are deliberately materialized as dense BF16;
        // only untouched projections take the packed affine-Q8 fast path.
        const auto packed = transformer_.pack_convrot_q8();
        if (active_loras_.empty())
            require(packed > 0, "ConvRot checkpoint contains no packable INT8 projections");
        event("pack_z_image_convrot_q8", int(packed), int(packed));
    }
    transformer_.materialize();
    vae_.materialize();
    return {uint64_t(transformer_.bytes() + vae_.bytes()), mx::get_active_memory()};
}

void ZImage::unload() {
    hybrid_.reset();
    hybrid_gpu_graph_ = {};
    hybrid_gpu_mlp_start_ = -1;
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
    auto result = components::qwen3_conditioning(
        tokens, text_encoder_, components::Qwen3Conditioning::z_image(), event, cancelled);
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

std::string ZImage::select_acceleration(Request &r, int rows, const Event &event,
                                        std::atomic<bool> &cancelled) {
    const bool automatic = r.execution == "auto";
    const AccelerationCase *matched = nullptr;
    if (automatic) {
        r.execution = "gpu";
        if (!r.allow_approximation || r.ane_manifest.empty()) {
            hybrid_.reset();
            return "gpu: no opted-in compatible local Z-Image partition";
        }
        if (!active_loras_.empty()) {
            hybrid_.reset();
            return "gpu: automatic Z-Image LoRA hybrid is not performance-qualified";
        }
        auto system = device_info();
        if (system.gpu != "Apple M4 Max" || system.physical_memory != (64ull << 30)) {
            hybrid_.reset();
            return "gpu: automatic Z-Image hybrid is not validated on this hardware";
        }
        matched = hybrid_case(r, rows, system.gpu, system.physical_memory);
        if (!matched) {
            hybrid_.reset();
            return "gpu: no measured Z-Image hybrid case for operation, dimensions, steps and token bucket";
        }
        const uint64_t estimate = (30ull << 30) + uint64_t(r.width) * r.height * 12288;
        if (system.physical_memory < estimate + (4ull << 30) ||
            (r.memory_budget_bytes && r.memory_budget_bytes < estimate)) {
            hybrid_.reset();
            return "gpu: Z-Image hybrid memory budget unavailable";
        }
        r.execution = "gpu_ane";
    }
    if (r.execution != "gpu_ane") {
        hybrid_.reset();
        return "gpu: native MLX single-stream S3-DiT";
    }
    try {
        require(std::filesystem::is_regular_file(transformer_checkpoint_),
                "Z-Image hybrid requires a safetensors file or index");
        if (automatic && hybrid_ && hybrid_->rows != matched->bucket)
            hybrid_.reset();
        if (!hybrid_ || hybrid_->manifest != r.ane_manifest)
            hybrid_ = std::make_unique<HybridSession>(
                r.ane_manifest, root_, rows, event, cancelled, r.warmup_iterations,
                transformer_checkpoint_, active_loras_, matched ? matched->bucket : 0);
        hybrid_->set_tokens(rows);
        require(hybrid_->hidden == 3840 &&
                    hybrid_->block_count == 32 && hybrid_->mlp_width == 10240 &&
                    hybrid_->ane_mlp_start == 0 && hybrid_->ane_mlp_end < 10240,
                "Z-Image Core ML FFN partition geometry mismatch");
        if (automatic)
            require(hybrid_->ane_mlp_end == 4096,
                    "M4 Max automatic Z-Image profile requires the validated 4096-channel ANE prefix");
        if (!hybrid_gpu_graph_ || hybrid_gpu_mlp_start_ != hybrid_->ane_mlp_end) {
            hybrid_gpu_graph_ = make_z_hybrid_gpu_graph(
                hybrid_->hidden, hybrid_->mlp_width, hybrid_->ane_mlp_end);
            hybrid_gpu_mlp_start_ = hybrid_->ane_mlp_end;
        }
        return automatic ? std::string("gpu_ane: measured case ") + matched->id
                         : "gpu_ane: explicitly selected Z-Image FFN partition";
    } catch (const Cancelled &) {
        throw;
    } catch (const std::exception &error) {
        // A failed shape rebinding must not leave partially rebound branches
        // available to the next request in this persistent session.
        hybrid_.reset();
        hybrid_gpu_graph_ = {};
        hybrid_gpu_mlp_start_ = -1;
        mx::clear_cache();
        if (!automatic)
            throw;
        r.execution = "gpu";
        event("acceleration_gpu_fallback", 1, 1);
        return std::string("gpu: ") + error.what();
    }
}

RunResult ZImage::prepare(const Request &requested, bool warmup, const Event &event,
                          std::atomic<bool> &cancelled) {
    return run(requested, event, cancelled, warmup, !warmup);
}

RunResult ZImage::generate(const Request &r, const Event &event, std::atomic<bool> &cancelled) {
    return run(r, event, cancelled, false, false);
}

RunResult ZImage::run(const Request &requested, const Event &event, std::atomic<bool> &cancelled,
                      bool warmup, bool load_only) {
    auto r = requested;
    auto begin = Clock::now();
    require(r.model == model_id_, "Z-Image session received a different model id");
    auto plan = make_plan(r);
    require(!r.prompt.empty() && (warmup || load_only || !r.output.empty()),
            "prompt and output are required");
    require(warmup || load_only || std::filesystem::path(r.output).extension() == ".png",
            "Z-Image output must be .png");
    require(r.inputs.empty(), "Z-Image-Turbo currently supports text-to-image only");
    require(r.width % 16 == 0 && r.height % 16 == 0, "Z-Image dimensions must be multiples of 16");
    select_loras(r);
    auto text_start = Clock::now();
    bool prompt_hit = conditioning(r, event, cancelled);
    const double text_seconds = std::chrono::duration<double>(Clock::now() - text_start).count();
    const int image_rows = ((r.height / 16) * (r.width / 16) + 31) / 32 * 32;
    const int caption_rows = (cached_conditioning_->shape(0) + 31) / 32 * 32;
    auto selection = select_acceleration(r, image_rows + caption_rows, event, cancelled);
    event(r.execution == "gpu_ane" ? "route_gpu_ane" : "route_gpu", 1, 1);
    r.compile_gpu = r.execution == "gpu" && !gguf_transformer_ && !convrot_transformer_ &&
                    active_lora_strategy_ != "inference_time" &&
                    !std::getenv("TURBOCIDER_Z_EAGER_BLOCKS");
    if (plan.request.execution != r.execution || plan.request.compile_gpu != r.compile_gpu)
        plan = make_plan(r);
    load(event, cancelled);
    if (load_only) {
        RunResult result;
        result.prepared = true;
        result.request = r;
        result.plan = std::move(plan);
        result.selection = selection;
        result.prompt_cache_hit = prompt_hit;
        auto reported_tokens = tokenizer_.z_image_prompt(r.prompt, r.dynamic_text);
        result.text_tokens = int(reported_tokens.ids.size());
        result.valid_text_tokens = reported_tokens.valid;
        result.total_tokens = image_rows + caption_rows;
        result.lora_applied_projections = lora_applied_projections_;
        if (gguf_transformer_) {
            result.backend = hybrid_ ? "mlx_cpp_metal_gguf+coreml" : "mlx_cpp_metal_gguf";
            result.precision = "gguf_native:" + r.model_variant;
            result.checkpoint = transformer_checkpoint_.filename().string();
        } else if (convrot_transformer_) {
            result.backend = "mlx_cpp_metal_convrot_packed_q8";
            result.precision = "int8_tensorwise_convrot_g256";
            result.checkpoint = transformer_checkpoint_.filename().string();
        } else {
            result.backend = hybrid_ ? "mlx_cpp_metal+coreml" : "mlx_cpp_metal";
            result.precision = hybrid_ ? "bf16_gpu+int8_mlp_fp16_io" : "bf16";
        }
        if (hybrid_)
            result.hybrid = hybrid_->metrics();
        result.timings.wall =
            std::chrono::duration<double>(Clock::now() - begin).count();
        result.timings.text = text_seconds;
        result.peak_bytes = mx::get_peak_memory();
        result.active_bytes = mx::get_active_memory();
        return result;
    }
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
    if (gguf_transformer_) {
        result.backend = hybrid_ ? "mlx_cpp_metal_gguf+coreml" : "mlx_cpp_metal_gguf";
        result.precision = "gguf_native:" + r.model_variant;
        result.checkpoint = transformer_checkpoint_.filename().string();
    } else if (convrot_transformer_) {
        result.backend = "mlx_cpp_metal_convrot_packed_q8";
        result.precision = "int8_tensorwise_convrot_g256";
        result.checkpoint = transformer_checkpoint_.filename().string();
    }
    if (hybrid_)
        result.hybrid = hybrid_->metrics();
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
                      cancelled, hybrid_.get(), hybrid_ ? &hybrid_gpu_graph_ : nullptr),
        mx::float32);
}

Tensor ZImage::decode(const Tensor &latent, int, int, const Event &, std::atomic<bool> &) {
    return z_vae_decode(mx::astype(latent, mx::bfloat16), vae_);
}

} // namespace tc

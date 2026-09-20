#include "../../runtime/build_identity.hpp"
#include "z_image.hpp"
#include "block_profile.hpp"
#include "hybrid_math.hpp"
#include "hybrid_stream.hpp"
#include "vae.hpp"

#include "../../media/image.hpp"
#include "../../platform/apple/platform.hpp"
#include "../../runtime/acceleration.hpp"
#include "../../runtime/residency.hpp"
#include "../../runtime/streaming/canonical_encoding.hpp"
#include "../../runtime/streaming/context.hpp"
#include "../../runtime/streaming/resolved_request.hpp"
#include "streaming_descriptor.hpp"
#include "../../components/text/qwen3.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <regex>

namespace tc {

class ZImageExactStream {
  public:
    ZImageExactStream(const std::filesystem::path &checkpoint,
                      const StreamingConfig &config,
                      const z_image::StreamingWorkload &workload,
                      uint64_t budget, uint64_t activation_reserve,
                      Weights &fixed, const Event &event,
                      std::atomic<bool> &cancelled,
                      uint64_t request_generation);
    ZImageExactStream(
        std::shared_ptr<const streaming::SourceLease> lease,
        const StreamingConfig &config,
        const z_image::StreamingWorkload &workload,
        uint64_t budget, uint64_t activation_reserve,
        Weights &fixed, const Event &event,
        std::atomic<bool> &cancelled, uint64_t request_generation);
    ~ZImageExactStream();
    ZImageExactStream(const ZImageExactStream &) = delete;
    ZImageExactStream &operator=(const ZImageExactStream &) = delete;

    void start();
    bool drain_safely() noexcept;
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
    void test_set_drain_failure(bool);
#endif
    void run_pass(uint32_t pass, uint32_t step, Tensor &unified,
                  const Tensor &freqs, const Tensor &temb);
    void finish();
    void enable_receipt(streaming::ExecutionReceiptOptions);
    std::shared_ptr<const streaming::ActualStageReceipt> receipt() const;
    const z_image::StreamingPlanView &plan() const;
    const BlockResidencyMetrics &metrics() const;
    streaming::ExecutionCounters counters() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

namespace {

constexpr int kHeadDim = 128;
constexpr int kHeads = 30;
constexpr float kVaeScale = 0.3611f;
constexpr float kVaeShift = 0.1159f;
constexpr const char *kZImageKernelRevision =
    "z-image-mlx-compiled-dense-block-v1";
constexpr const char *kZImagePublicImplementation =
    "generic_stage_executor_v2";
constexpr const char *kZImagePublicComponentPolicy =
    "zimage-components-v2-all-sources-request-cache";

uint32_t padded_z_image_rows(uint32_t rows) {
    require(rows && rows <= UINT32_MAX - 31,
            "streaming_workload_invalid: Z-Image token rows overflow");
    return (rows + 31) / 32 * 32;
}

streaming::PresetSourceIdentity z_image_public_source_identity(
        const streaming::SourceLease &lease) {
    if (lease.has_verified_content())
        return {"z-image-turbo-comfy-bf16", "comfy-bf16-single-file",
                std::string(lease.artifact_digest()), "", 2};
    streaming::CanonicalEncoder manifest(
        "z-image-public-artifact-manifest-v1");
    manifest.string_field("transformer_snapshot", lease.digest());
    manifest.unsigned_field("artifact_count", lease.file_count());
    return {
        "z-image-turbo-comfy-bf16",
        "comfy-bf16-single-file",
        manifest.sha256(),
        std::string(lease.digest()),
    };
}

streaming::PresetRuntimeIdentity z_image_public_runtime_identity() {
    return {
        tc::runtime_build_identity(),
        "public-streaming-runtime-v2",
        "z-image-public-adapter-v3-all-component-lease",
        "z-image-pread-bf16-v2-fd-lease",
        kZImageKernelRevision,
        "mlx-request-cache-policy-v2-k1-zero-cache",
    };
}

std::string z_image_public_feature_digest(
        const Request &request, uint32_t caption_rows) {
    streaming::CanonicalEncoder feature(
        "z-image-public-workload-features-v1");
    feature.boolean_field("inputs_empty", request.inputs.empty());
    feature.boolean_field("loras_empty", request.loras.empty());
    feature.boolean_field("ane_disabled", request.ane_manifest.empty());
    feature.boolean_field(
        "encoder_ane_disabled", request.encoder_ane_manifest.empty());
    feature.boolean_field("compile_gpu", request.compile_gpu);
    feature.boolean_field("dynamic_text", request.dynamic_text);
    feature.unsigned_field("caption_rows", caption_rows);
    return feature.sha256();
}

// Small unified-memory machines also need a bounded cache in resident mode.
// Restore the process-wide setting before another request or model starts.
struct RequestCacheLimit {
    std::optional<size_t> previous;
    RequestCacheLimit(bool enabled, size_t limit) {
        if (enabled) {
            previous = mx::set_cache_limit(limit);
            mx::clear_cache();
        }
    }
    ~RequestCacheLimit() {
        if (previous) {
            mx::clear_cache();
            mx::set_cache_limit(*previous);
        }
    }
};

bool z_image_exact_streaming_requested(const Request &request) {
    if (!request.streaming.active()) return false;
    const auto stage = request.streaming.stages.find("denoiser");
    return stage != request.streaming.stages.end() &&
        stage->second.residency &&
        *stage->second.residency == "streamed";
}

uint32_t z_image_exact_slot_count(const Request &request) {
    const auto stage = request.streaming.stages.find("denoiser");
    if (stage == request.streaming.stages.end() ||
        !stage->second.slot_count)
        return 0;
    return *stage->second.slot_count;
}

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
    // Streamed ConvRot MLPs retain only the GPU suffix. Rotation groups and
    // quantization groups align at the partition boundary, so local columns
    // have the same meaning as the selected columns of the resident matrix.
    const int rows = w.at(prefix + ".w1.weight").shape(0);
    require(rows == end || rows == end - start, "invalid Z-Image GPU suffix width");
    if (rows != end) { end = rows; start = 0; }
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

std::function<std::vector<Tensor>(const std::vector<Tensor> &)> &z_hybrid_pre_graph() {
    // Core ML is a graph boundary. Compile the GPU work up to that boundary
    // together, retaining the BF16 input for the suffix and FP16 for Core ML.
    // Every weight is an argument, so cached graphs cannot retain another layer
    // or a previous stream slot's weights.
    static auto graph = mx::compile([](const std::vector<Tensor> &a) {
        require(a.size() == 12, "invalid Z-Image hybrid pre-MLP inputs");
        auto fast_rms = [](const Tensor &x, const Tensor &weight) {
            return mx::astype(mx::fast::rms_norm(mx::astype(x, mx::float32),
                                                mx::astype(weight, mx::float32), 1e-5f),
                              x.dtype());
        };
        auto mod = mx::split(mx::expand_dims(mx::matmul(a[2], mx::transpose(a[3])) + a[4], 1), 4, -1);
        auto input = fast_rms(a[0], a[5]) * (Tensor(1.f, mod[0].dtype()) + mod[0]);
        auto qkv = mx::split(mx::matmul(input, mx::transpose(a[6])), 3, -1);
        auto q = fast_rms(heads(qkv[0], kHeads, kHeadDim), a[7]);
        auto k = fast_rms(heads(qkv[1], kHeads, kHeadDim), a[8]);
        auto rotated = z_apply_rope_pair(q, k, a[1]);
        auto attention = mx::matmul(
            attend(rotated[0], rotated[1], heads(qkv[2], kHeads, kHeadDim), false, {},
                   !std::getenv("TURBOCIDER_Z_DISABLE_FUSED_SDPA")), mx::transpose(a[9]));
        auto value = a[0] + mx::tanh(mod[1]) * fast_rms(attention, a[10]);
        auto feed = fast_rms(value, a[11]) * (Tensor(1.f, mod[2].dtype()) + mod[2]);
        return std::vector<Tensor>{value, feed, mx::tanh(mod[3]), mx::astype(feed, mx::float16)};
    });
    return graph;
}

std::function<std::vector<Tensor>(const std::vector<Tensor> &)> &z_hybrid_post_graph() {
    static auto graph = mx::compile([](const std::vector<Tensor> &a) {
        require(a.size() == 6, "invalid Z-Image hybrid post-MLP inputs");
        // Match the existing order and dtypes, including BF16 rounding before
        // normalization. The Core ML output remains in its shared FP16 backing.
        auto feed = z_image::join_hybrid_ffn(a[0], a[1], a[5]);
        auto normalized = mx::astype(
            mx::fast::rms_norm(mx::astype(feed, mx::float32),
                               mx::astype(a[4], mx::float32), 1e-5f), feed.dtype());
        return std::vector<Tensor>{a[2] + a[3] * normalized};
    });
    return graph;
}

Tensor z_compiled_hybrid_block(const Tensor &x, const Weights &w, const std::string &prefix,
                               const Tensor &freqs, const Tensor &temb, HybridSession *hybrid,
                               int block,
                               const std::function<std::vector<Tensor>(const std::vector<Tensor> &)> &gpu_graph,
                               ZBlockProfile &profile) {
    auto pre = z_hybrid_pre_graph()({x, freqs, temb,
        w.at(prefix + ".adaLN_modulation.0.weight"), w.at(prefix + ".adaLN_modulation.0.bias"),
        w.at(prefix + ".attention_norm1.weight"), w.at(prefix + ".attention.qkv.weight"),
        w.at(prefix + ".attention.q_norm.weight"), w.at(prefix + ".attention.k_norm.weight"),
        w.at(prefix + ".attention.out.weight"), w.at(prefix + ".attention_norm2.weight"),
        w.at(prefix + ".ffn_norm1.weight")});
    if (profile) profile.pre_done({pre[0], pre[2], pre[1]});
    const int actual_rows = x.shape(1);
    auto packed = pre[3];
    if (actual_rows < hybrid->rows)
        packed = mx::concatenate(
            {packed, mx::zeros({1, hybrid->rows - actual_rows, 3840}, mx::float16)}, 1);
    mx::eval({pre[0], pre[1], pre[2], packed});
    profile.packed();
    const auto ffn = prefix + ".feed_forward";
    auto gpu = gpu_graph({pre[1], w.at(ffn + ".w1.weight"),
                         w.at(ffn + ".w3.weight"), w.at(ffn + ".w2.weight")})[0];
    mx::async_eval({gpu});
    profile.submitted(gpu);
    auto ane = slice_axis(hybrid->predict(block, packed), 1, 0, actual_rows);
    profile.predicted(gpu);
    auto result = z_hybrid_post_graph()({gpu, ane, pre[0], pre[2],
        w.at(prefix + ".ffn_norm2.weight"), Tensor(z_hybrid_output_scale(hybrid), gpu.dtype())})[0];
    profile.finish(result);
    return result;
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
                const std::function<std::vector<Tensor>(const std::vector<Tensor> &)> *gpu_graph,
                bool compile_hybrid_segments = false, std::vector<Tensor> *keepalive = nullptr) {
    ZBlockProfile profile(prefix, hybrid != nullptr);
    // A LoRA can dequantize only the projections it touches.  Do not infer
    // that the whole block is dense from QKV/w1 alone: Q8 GGUF modulation or
    // the remaining attention/FFN weights may still be packed affine tensors.
    // The compiled graph accepts ordinary dense matrices only; mixed
    // dense/quantized blocks stay on linear_compat below.
    const bool fully_dense =
        !w.nvfp4(prefix + ".adaLN_modulation.0") &&
        !w.nvfp4(prefix + ".attention.qkv") &&
        !w.nvfp4(prefix + ".attention.out") &&
        !w.nvfp4(prefix + ".feed_forward.w1") &&
        !w.nvfp4(prefix + ".feed_forward.w2") &&
        !w.nvfp4(prefix + ".feed_forward.w3") &&
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
    if (profile.split_gpu())
        require(fully_dense && !w.has_runtime_loras(),
                "Z-Image gpu_split profiling requires dense BF16 weights without runtime LoRA");
    if (compile_hybrid_segments && hybrid && gpu_graph && fully_dense && x.dtype() == mx::bfloat16 &&
        !w.has_runtime_loras() && w.has(prefix + ".adaLN_modulation.0.bias") &&
        !std::getenv("TURBOCIDER_Z_HYBRID_EAGER_SEGMENTS") &&
        !std::getenv("TURBOCIDER_Z_EAGER_BLOCKS") &&
        !std::getenv("TURBOCIDER_DISABLE_FUSED_RMSNORM") &&
        !std::getenv("TURBOCIDER_Z_CONVROT_DEBUG") &&
        !std::getenv("TURBOCIDER_Z_HYBRID_VALIDATE"))
        return z_compiled_hybrid_block(x, w, prefix, freqs, temb, hybrid, hybrid_block,
                                      *gpu_graph, profile);
    if (!hybrid && !std::getenv("TURBOCIDER_Z_EAGER_BLOCKS") && fully_dense &&
        !w.has_runtime_loras() && !profile.split_gpu()) {
        auto result = z_compiled_gpu_block(x, w, prefix, freqs, temb);
        profile.finish(result, true);
        return result;
    }
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
    if (profile) profile.pre_done({value, gate_mlp, feed_input});
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
        if (keepalive) keepalive->insert(keepalive->end(), {value, gate_mlp, feed_input, packed});
        mx::eval({feed_input, packed});
        profile.packed();
        const auto ffn = prefix + ".feed_forward";
        const bool dense = !w.quantized(ffn + ".w1") && !w.convrot(ffn + ".w1") &&
                           !w.quantized(ffn + ".w2") && !w.convrot(ffn + ".w2") &&
                           !w.quantized(ffn + ".w3") && !w.convrot(ffn + ".w3");
        auto gpu = dense
            ? (*gpu_graph)({feed_input, w.at(ffn + ".w1.weight"),
                            w.at(ffn + ".w3.weight"), w.at(ffn + ".w2.weight")})[0]
            : z_hybrid_gpu_suffix(feed_input, w, ffn, hybrid->ane_mlp_end,
                                  hybrid->mlp_width);
        if (keepalive) keepalive->push_back(gpu);
        mx::async_eval({gpu});
        profile.submitted(gpu);
        auto ane = slice_axis(hybrid->predict(hybrid_block, packed), 1, 0, actual_rows);
        if (keepalive) keepalive->push_back(ane);
        profile.predicted(gpu);
        feed = z_image::join_hybrid_ffn(gpu, ane, Tensor(z_hybrid_output_scale(hybrid), gpu.dtype()));
        if (std::getenv("TURBOCIDER_Z_HYBRID_VALIDATE")) {
            auto ane_scaled = mx::astype(ane, gpu.dtype()) * Tensor(z_hybrid_output_scale(hybrid), gpu.dtype());
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
    } else if (profile.split_gpu()) {
        // Diagnostic counterpart to the compiled hybrid suffix, with all MLP
        // channels. This intentionally bypasses the production full-block graph.
        static auto full_mlp = mx::compile([](const std::vector<Tensor> &a) {
            auto gate = mx::matmul(a[0], mx::transpose(a[1]));
            auto up = mx::matmul(a[0], mx::transpose(a[2]));
            return std::vector<Tensor>{mx::matmul(silu(gate) * up, mx::transpose(a[3]))};
        });
        const auto ffn = prefix + ".feed_forward";
        feed = full_mlp({feed_input, w.at(ffn + ".w1.weight"),
                        w.at(ffn + ".w3.weight"), w.at(ffn + ".w2.weight")})[0];
        profile.full_mlp(feed);
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
    if (keepalive) keepalive->push_back(result);
    profile.finish(result);
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
                     const std::function<std::vector<Tensor>(const std::vector<Tensor> &)> *gpu_graph,
                     ZImageWeightStream *weight_stream,
                     ZImageExactStream *exact_stream, uint32_t pass,
                     bool compile_hybrid_segments, ZImageHybridStream *hybrid_stream = nullptr) {
    require(!hybrid_stream || (!weight_stream && !exact_stream), "hybrid/exact stream conflict");
    require(!(weight_stream && exact_stream),
            "Z-Image legacy and exact streaming cannot run together");
    if (weight_stream) weight_stream->begin_pass();
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
        image = hybrid_stream ? hybrid_stream->encode_noise(uint32_t(i), image, image_freqs, temb)
            : z_block(image, w, "noise_refiner." + std::to_string(i), image_freqs, temb,
                      hybrid, i, gpu_graph, compile_hybrid_segments);
        caption_emb = z_context_block(caption_emb, w,
                                      "context_refiner." + std::to_string(i), caption_freqs);
    }
    auto unified = mx::concatenate({image, caption_emb}, 1);
    auto unified_freqs = mx::concatenate({image_freqs, caption_freqs}, 0);
    if (hybrid_stream) {
        hybrid_stream->run_main(pass, unified, unified_freqs, temb);
    } else if (exact_stream) {
        exact_stream->run_pass(pass, pass, unified, unified_freqs, temb);
    } else {
        for (int i = 0; i < 30; ++i) {
            checkpoint(cancelled);
            event("z_image_denoise_block", i, 30);
            auto streamed = weight_stream ? weight_stream->acquire(i) : Weights{};
            unified = z_block(unified, weight_stream ? streamed : w,
                              "layers." + std::to_string(i), unified_freqs, temb,
                              hybrid, 2 + i, gpu_graph, compile_hybrid_segments);
            const bool eager = std::getenv("TURBOCIDER_Z_EAGER_BLOCKS");
            if (eager || hybrid || weight_stream) {
                mx::eval(unified);
                checkpoint(cancelled);
            }
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

Tensor z_image::decode_vae(const Tensor &latent, const Weights &weights) {
    return z_vae_decode(mx::astype(latent, mx::bfloat16), weights);
}

namespace {

struct ZHybridExecution {
    HybridSession session;
    z_image::HybridGpuGraph graph;
    std::vector<Tensor> pending;
    std::vector<ZHybridBranchCompletion> completed;
    uint32_t step = 0;
    ZHybridExecution(std::shared_ptr<const z_image::VerifiedCoreMLBundleLease> bundle,
                     const Event &event, std::atomic<bool> &cancel)
        : session(bundle, event, cancel),
          graph(z_image::make_hybrid_gpu_graph(3840, 10240, int(bundle->partition().ane_end))) {
        pending.reserve(16); completed.reserve(9 * 32);
    }
    void record(uint32_t branch, uint32_t rows) {
        require(step < 9 && branch < 32 && rows == (branch < 2 ? 1024u : 1088u), "hybrid branch shape mismatch");
        require(completed.size() == size_t(step) * 32 + branch, "hybrid branch order mismatch");
        const auto calls = session.metrics().runtime_calls;
        require(calls == completed.size() + 1, "hybrid prediction count mismatch");
        completed.push_back({step, branch, rows, calls});
        pending.clear(); // Only called after the block GPU consumer completes.
    }
};

class ZImageStageAdapter final : public streaming::ModelSlotAdapter {
    struct Job {
        ZImageStageAdapter *owner = nullptr;
        uint32_t slot = 0;
        uint32_t block = 0;
        std::array<char, 512> error{};
    };

    ZImageWeightStream &source_;
    ZHybridExecution *hybrid_ = nullptr;
    uint32_t prefix_ = 0;
    uint32_t slot_count_ = 0;
    Event event_;
    std::atomic<bool> &cancelled_;
    std::array<Job, 2> jobs_{};
    Weights current_;
    struct PassStorage { Tensor unified, freqs, temb; };
    std::optional<PassStorage> pass_storage_;
    Tensor *unified_ = nullptr;
    const Tensor *freqs_ = nullptr;
    const Tensor *temb_ = nullptr;
    uint32_t pass_ = 0;
    uint32_t step_ = 0;
    uint64_t reader_sequence_ = 0;

    void require_context() const {
        require(unified_ && freqs_ && temb_,
                "Z-Image exact adapter has no active pass context");
    }

  public:
    ZImageStageAdapter(ZImageWeightStream &source, uint32_t prefix,
                       uint32_t slot_count,
                       const Event &event, std::atomic<bool> &cancelled, ZHybridExecution *hybrid = nullptr)
        : source_(source), hybrid_(hybrid), prefix_(prefix), slot_count_(slot_count),
          event_(event),
          cancelled_(cancelled) {
        require(slot_count_ >= 1 && slot_count_ <= jobs_.size(),
                "Z-Image exact adapter requires one or two slots");
        for (uint32_t slot = 0; slot < jobs_.size(); ++slot) {
            jobs_[slot].owner = this;
            jobs_[slot].slot = slot;
        }
    }

    void bind_pass(uint32_t pass, uint32_t step, Tensor &unified,
                   const Tensor &freqs, const Tensor &temb) {
        require(!unified_, "Z-Image exact pass context is already bound");
        pass_ = pass;
        step_ = step;
        pass_storage_.emplace(PassStorage{unified, freqs, temb});
        unified_ = &pass_storage_->unified;
        freqs_ = &pass_storage_->freqs;
        temb_ = &pass_storage_->temb;
    }

    void copy_pass_result(Tensor &unified) const {
        require_context();
        unified = *unified_;
    }

    void unbind_pass(bool safe = true) noexcept {
        if (safe) {
            current_.clear();
            pass_storage_.reset();
        } else {
            // Joined I/O cannot call back into a returned API stack.
            event_ = {};
        }
        unified_ = nullptr;
        freqs_ = nullptr;
        temb_ = nullptr;
    }

    std::string fill_error() const {
        for (const auto &job : jobs_)
            if (job.error[0]) return job.error.data();
        return {};
    }

    void create_pool(const streaming::PoolLayout &pool) override {
        require(pool.id == 0 && pool.slots.size() == slot_count_,
                "Z-Image exact adapter requires one compiled K1/K2 pool");
        const uint64_t capacity = pool.slots.front().capacity_bytes;
        for (const auto &slot : pool.slots)
            require(slot.capacity_bytes == capacity,
                    "Z-Image exact slot capacities differ");
        source_.create_exact_pool(uint32_t(pool.slots.size()), capacity);
    }

    streaming::FillJob make_fill_job(
            const streaming::Group &group,
            const tc_stream_slot_ticket_v1 &ticket) override {
        require(group.blocks.size() == 1 && ticket.slot < slot_count_,
                "Z-Image exact fill requires one block and a valid slot");
        auto &job = jobs_[ticket.slot];
        job.block = group.blocks.front();
        job.error[0] = 0;
        return {ticket, &job,
                [](void *raw, const tc_stream_slot_ticket_v1 *,
                   const std::atomic<bool> *worker_cancel,
                   uint64_t *bytes) -> int {
                    auto &job = *static_cast<Job *>(raw);
                    try {
                        *bytes = job.owner->source_.fill_exact(
                            job.slot, job.block, worker_cancel);
                        return 0;
                    } catch (const std::exception &error) {
                        std::snprintf(job.error.data(), job.error.size(),
                                      "%s", error.what());
                        return -1;
                    } catch (...) {
                        std::snprintf(job.error.data(), job.error.size(),
                                      "%s", "unknown Z-Image fill failure");
                        return -1;
                    }
                }};
    }

    void encode_prefix(uint32_t pass) override {
        require_context();
        require(pass == pass_, "Z-Image exact prefix pass mismatch");
        source_.check_unchanged();
        for (uint32_t block = 0; block < prefix_; ++block) {
            checkpoint(cancelled_);
            event_("z_image_denoise_block", int(block), 30);
            *unified_ = z_block(
                *unified_, source_.prefix_weights(block),
                "layers." + std::to_string(block), *freqs_, *temb_,
                hybrid_ ? &hybrid_->session : nullptr, int(2 + block),
                hybrid_ ? &hybrid_->graph : nullptr, false, hybrid_ ? &hybrid_->pending : nullptr);
            mx::eval(*unified_);
            if (hybrid_) hybrid_->record(2 + block, uint32_t(unified_->shape(1)));
            checkpoint(cancelled_);
        }
    }

    void prepare_group(const streaming::Group &group,
                       const tc_stream_slot_ticket_v1 &ticket) override {
        require_context();
        require(group.blocks.size() == 1 && ticket.item.pass == pass_ &&
                    ticket.item.step == step_ &&
                    ticket.item.group == group.id &&
                    ticket.slot == group.slot,
                "Z-Image exact prepare ticket mismatch");
        current_ = source_.bind_exact(ticket.slot, group.blocks.front());
    }

    bool overlap_next_fill_after_claim() const noexcept override {
        return slot_count_ > 1;
    }

    streaming::ReaderSet encode_group(
            const streaming::Group &group,
            const tc_stream_slot_ticket_v1 &ticket,
            streaming::CompletionMailbox &) override {
        require_context();
        require(group.blocks.size() == 1 && ticket.item.pass == pass_ &&
                    ticket.item.step == step_,
                "Z-Image exact encode ticket mismatch");
        const uint32_t block = group.blocks.front();
        checkpoint(cancelled_);
        event_("z_image_denoise_block", int(block), 30);
        *unified_ = z_block(
            *unified_, current_, "layers." + std::to_string(block),
            *freqs_, *temb_, hybrid_ ? &hybrid_->session : nullptr, int(2 + block),
            hybrid_ ? &hybrid_->graph : nullptr, false, hybrid_ ? &hybrid_->pending : nullptr);
        // The executor has already started the following vacant slot's fill
        // after claiming this content. This synchronous completion therefore
        // matches the specialized pager's ordering without an extra
        // async_eval call per block.
        mx::eval(*unified_);
        if (hybrid_) hybrid_->record(2 + block, uint32_t(unified_->shape(1)));
        checkpoint(cancelled_);
        require(reader_sequence_ != std::numeric_limits<uint64_t>::max(),
                "Z-Image exact reader sequence overflow");
        const tc_stream_reader_fence_v1 fence{1, ++reader_sequence_};
        streaming::ReaderSet readers;
        readers.count = 1;
        readers.fences[0] = fence;
        readers.already_complete = true;
        current_.clear();
        return readers;
    }

#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
    bool test_fail_drain = false;
#endif
    bool drain() noexcept override {
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
        if (test_fail_drain) return false;
#endif
        try {
            mx::synchronize();
            return true;
        } catch (...) {
            return false;
        }
    }

    void destroy_pool() noexcept override {
        current_.clear();
        source_.destroy_exact_pool();
    }
};

} // namespace

struct ZImageHybridStream::Impl {
    std::shared_ptr<const streaming::SourceLease> parent;
    std::shared_ptr<const z_image::VerifiedCoreMLBundleLease> bundle;
    std::shared_ptr<std::atomic<bool>> cancelled;
    Event event;
    std::thread::id owner = std::this_thread::get_id();
    z_image::StreamingWorkload workload;
    std::unique_ptr<z_image::StreamingMetadata> metadata;
    z_image::HybridStreamingPlan plan;
    std::shared_ptr<const z_image::GpuSuffixSource> derived;
    Weights fixed;
    std::unique_ptr<ZImageWeightStream> source;
    std::unique_ptr<ZHybridExecution> hybrid;
    std::shared_ptr<ZImageStageAdapter> adapter;
    std::unique_ptr<streaming::StageExecutor> executor;
    struct Inputs { Tensor latent, caption; };
    std::optional<Inputs> inputs;
    uint32_t next_pass = 0;
    bool active = false, failed = false, finished = false;
    void check_owner() const {
        require(owner == std::this_thread::get_id(), "hybrid stream owner thread mismatch");
    }
    Impl(std::shared_ptr<const streaming::SourceLease> p,
         std::shared_ptr<const z_image::VerifiedCoreMLBundleLease> b,
         const StreamingConfig &config, const z_image::StreamingWorkload &work,
         Event e, std::shared_ptr<std::atomic<bool>> c)
        : parent(std::move(p)), bundle(std::move(b)), cancelled(std::move(c)), event(std::move(e)), workload(work) {
        require(parent && bundle && cancelled, "hybrid stream requires owned verified sources and cancellation");
        if (!event) event = [](const std::string &, int, int) {};
        checkpoint(*cancelled);
        metadata = std::make_unique<z_image::StreamingMetadata>(parent);
        plan = z_image::describe_hybrid_streaming(*metadata, *bundle, config, workload);
        require(config.active() && !plan.layout.stages[0].resident && !plan.layout.stages[0].groups.empty(),
                "hybrid stream requires a streamed main stage");
        require(!std::getenv("TURBOCIDER_Z_HYBRID_VALIDATE"),
                "legacy full-weight hybrid validator cannot consume suffix-only weights");
    }
    void initialize(uint64_t budget, uint64_t activation, uint64_t request) {
        require(request != 0, "hybrid request generation must be nonzero");
        derived = metadata->materialize_gpu_suffix(workload, bundle->partition().ane_end, *cancelled, event);
        require(derived->plan().recipe_digest == plan.gpu.recipe_digest, "hybrid materialization recipe changed");
        const auto &stage = plan.layout.stages[0];
        source = std::make_unique<ZImageWeightStream>(derived, stage.prefix, stage.slot_count,
                                                    budget, activation, fixed, event, *cancelled);
        hybrid = std::make_unique<ZHybridExecution>(bundle, event, *cancelled);
        adapter = std::make_shared<ZImageStageAdapter>(*source, stage.prefix, stage.slot_count,
                                                      event, *cancelled, hybrid.get());
        executor = std::make_unique<streaming::StageExecutor>(0, request, adapter);
        executor->begin(stage);
        executor->enable_receipt({plan.layout.digest, "z-image-verified-hybrid-stage-v1", parent->generation()});
        bundle->revalidate(); derived->check_unchanged();
    }
};

ZImageHybridStream::ZImageHybridStream(std::shared_ptr<const streaming::SourceLease> parent,
        std::shared_ptr<const z_image::VerifiedCoreMLBundleLease> bundle,
        const StreamingConfig &config, const z_image::StreamingWorkload &workload,
        uint64_t budget, uint64_t activation, Event event,
        std::shared_ptr<std::atomic<bool>> cancel, uint64_t request)
    : impl_(std::make_unique<Impl>(std::move(parent), std::move(bundle), config, workload,
                                  std::move(event), std::move(cancel))) {
    try { impl_->initialize(budget, activation, request); }
    catch (...) {
        if (!drain_safely()) (void)impl_.release();
        throw;
    }
}
ZImageHybridStream::~ZImageHybridStream() {
    if (impl_ && !drain_safely()) (void)impl_.release();
}
bool ZImageHybridStream::drain_safely() noexcept {
    if (!impl_) return true;
    if (impl_->owner != std::this_thread::get_id()) return false;
    if (!impl_->finished) impl_->failed = true;
    bool safe = true;
    try {
        if (impl_->executor) safe = impl_->executor->retry_drain();
        // Fixed noise, embeddings and final projection live outside the pool.
        if (safe) mx::synchronize();
    } catch (...) { safe = false; }
    if (impl_->adapter) impl_->adapter->unbind_pass(safe);
    if (safe) {
        if (impl_->hybrid) impl_->hybrid->pending.clear();
        impl_->inputs.reset();
    }
    impl_->active = false;
    return safe;
}
bool ZImageHybridStream::failed() const noexcept { return !impl_ || impl_->failed; }
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
void ZImageHybridStream::test_set_drain_failure(bool value) {
    impl_->check_owner(); impl_->adapter->test_fail_drain = value;
}
#endif
const streaming::Layout &ZImageHybridStream::layout() const { return impl_->plan.layout; }
streaming::ExecutionCounters ZImageHybridStream::counters() const {
    impl_->check_owner(); return impl_->executor->counters();
}
HybridMetrics ZImageHybridStream::hybrid_metrics() const {
    impl_->check_owner(); return impl_->hybrid->session.metrics();
}
const std::vector<ZHybridBranchCompletion> &ZImageHybridStream::branches() const { return impl_->hybrid->completed; }
std::shared_ptr<const streaming::ActualStageReceipt> ZImageHybridStream::receipt() const {
    impl_->check_owner(); require(impl_->finished, "hybrid receipt is not finalized"); return impl_->executor->receipt();
}
std::vector<float> ZImageHybridStream::sigmas() const { return z_sigmas(512, 512, 9); }
Tensor ZImageHybridStream::encode_noise(uint32_t branch, const Tensor &image, const Tensor &freqs, const Tensor &temb) {
    impl_->check_owner();
    require(impl_->active && !impl_->failed && branch < 2, "noise hook requires active hybrid transform");
    checkpoint(*impl_->cancelled);
    auto &execution = *impl_->hybrid;
    execution.pending = {image, freqs, temb};
    auto result = z_block(image, impl_->fixed, "noise_refiner." + std::to_string(branch), freqs, temb,
                          &execution.session, int(branch), &execution.graph, false, &execution.pending);
    mx::eval(result);
    execution.record(branch, uint32_t(result.shape(1)));
    checkpoint(*impl_->cancelled);
    return result;
}
void ZImageHybridStream::run_main(uint32_t pass, Tensor &unified, const Tensor &freqs, const Tensor &temb) {
    impl_->check_owner();
    require(impl_->active && !impl_->failed && pass == impl_->next_pass &&
            impl_->hybrid->completed.size() == size_t(pass) * 32 + 2, "main hook requires completed noise refiners");
    impl_->adapter->bind_pass(pass, pass, unified, freqs, temb);
    impl_->executor->run_pass(pass, pass, *impl_->cancelled);
    impl_->adapter->copy_pass_result(unified);
    impl_->adapter->unbind_pass();
}
Tensor ZImageHybridStream::transform(const Tensor &latent, const Tensor &caption, float sigma, uint32_t step) {
    impl_->check_owner();
    require(!impl_->active && !impl_->failed && !impl_->finished && step == impl_->next_pass && step < 9,
            "hybrid stream unavailable or pass out of order");
    require(latent.shape() == mx::Shape{16, 1, 64, 64} && caption.ndim() == 2 &&
            caption.shape(0) > 32 && caption.shape(0) <= 64 && caption.shape(1) == 2560 &&
            std::isfinite(sigma) && sigma >= 0 && sigma <= 1, "hybrid transform input scope mismatch");
    try {
        checkpoint(*impl_->cancelled);
        impl_->bundle->revalidate(); impl_->derived->check_unchanged();
        impl_->inputs.emplace(Impl::Inputs{mx::astype(latent, mx::bfloat16), caption});
        impl_->active = true; impl_->hybrid->step = step;
        auto output = mx::astype(z_transformer(impl_->inputs->latent, impl_->inputs->caption, sigma, 512, 512,
            impl_->fixed, impl_->event, *impl_->cancelled, &impl_->hybrid->session, &impl_->hybrid->graph,
            nullptr, nullptr, step, false, this), mx::float32);
        impl_->hybrid->pending.push_back(output);
        mx::eval(output);
        impl_->bundle->revalidate(); impl_->derived->check_unchanged();
        require(impl_->hybrid->completed.size() == size_t(step + 1) * 32, "incomplete hybrid transformer pass");
        impl_->hybrid->pending.clear(); impl_->inputs.reset(); impl_->active = false; ++impl_->next_pass;
        checkpoint(*impl_->cancelled);
        return output;
    } catch (const std::exception &error) {
        impl_->failed = true; drain_safely();
        if (std::strcmp(error.what(), "streaming_cancelled") == 0) throw Cancelled();
        throw;
    } catch (...) { impl_->failed = true; drain_safely(); throw; }
}
void ZImageHybridStream::finish() {
    impl_->check_owner();
    require(!impl_->active && !impl_->failed && !impl_->finished && impl_->next_pass == 9,
            "hybrid execution incomplete");
    try {
        impl_->executor->finish();
        impl_->bundle->revalidate(); impl_->derived->check_unchanged();
        require(impl_->hybrid->completed.size() == 288 && impl_->hybrid->session.metrics().runtime_calls == 288,
                "hybrid branch totals mismatch");
        impl_->finished = true;
    } catch (...) { impl_->failed = true; drain_safely(); throw; }
}

struct ZImageExactStream::Impl {
    z_image::StreamingPlanView plan;
    ZImageWeightStream source;
    std::shared_ptr<ZImageStageAdapter> adapter;
    std::unique_ptr<streaming::StageExecutor> executor;
    std::atomic<bool> &cancelled;
    streaming::ExecutionCounters final_counters{};
    bool finished = false;
    bool started = false;

    Impl(const std::filesystem::path &checkpoint,
         const StreamingConfig &config,
         const z_image::StreamingWorkload &workload,
         uint64_t budget, uint64_t activation_reserve, Weights &fixed,
         const Event &event, std::atomic<bool> &cancelled,
         uint64_t request_generation)
        : plan(checkpoint.string(), config, workload),
          source(checkpoint, plan.layout().stages.front().prefix,
                 plan.layout().stages.front().slot_count, budget,
                 activation_reserve, fixed, event, cancelled),
          adapter(std::make_shared<ZImageStageAdapter>(
              source, plan.layout().stages.front().prefix,
              plan.layout().stages.front().slot_count, event, cancelled)),
          executor(std::make_unique<streaming::StageExecutor>(
              0, request_generation, adapter)),
          cancelled(cancelled) {
        plan.metadata().check_unchanged();
    }

    Impl(std::shared_ptr<const streaming::SourceLease> lease,
         const StreamingConfig &config,
         const z_image::StreamingWorkload &workload,
         uint64_t budget, uint64_t activation_reserve, Weights &fixed,
         const Event &event, std::atomic<bool> &cancelled,
         uint64_t request_generation)
        : plan(std::move(lease), config, workload),
          source(plan.lease_ptr(), plan.layout().stages.front().prefix,
                 plan.layout().stages.front().slot_count, budget,
                 activation_reserve, fixed, event, cancelled),
          adapter(std::make_shared<ZImageStageAdapter>(
              source, plan.layout().stages.front().prefix,
              plan.layout().stages.front().slot_count, event, cancelled)),
          executor(std::make_unique<streaming::StageExecutor>(
              0, request_generation, adapter)),
          cancelled(cancelled) {
        plan.metadata().check_unchanged();
    }
};

ZImageExactStream::ZImageExactStream(
        const std::filesystem::path &checkpoint,
        const StreamingConfig &config,
        const z_image::StreamingWorkload &workload,
        uint64_t budget, uint64_t activation_reserve, Weights &fixed,
        const Event &event, std::atomic<bool> &cancelled,
        uint64_t request_generation)
    : impl_(std::make_unique<Impl>(
          checkpoint, config, workload, budget, activation_reserve, fixed,
          event, cancelled, request_generation)) {}

ZImageExactStream::ZImageExactStream(
        std::shared_ptr<const streaming::SourceLease> lease,
        const StreamingConfig &config,
        const z_image::StreamingWorkload &workload,
        uint64_t budget, uint64_t activation_reserve, Weights &fixed,
        const Event &event, std::atomic<bool> &cancelled,
        uint64_t request_generation)
    : impl_(std::make_unique<Impl>(
          std::move(lease), config, workload, budget, activation_reserve,
          fixed, event, cancelled, request_generation)) {}

ZImageExactStream::~ZImageExactStream() {
    if (impl_ && !drain_safely()) (void)impl_.release();
}

void ZImageExactStream::start() {
    require(impl_ && !impl_->finished, "Z-Image exact executor unavailable");
    if (impl_->started) return;
    impl_->started = true;
    impl_->executor->begin(impl_->plan.layout().stages.front());
}

bool ZImageExactStream::drain_safely() noexcept {
    if (!impl_) return true;
    const bool safe = impl_->executor->retry_drain();
    impl_->adapter->unbind_pass(safe);
    return safe;
}

#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
void ZImageExactStream::test_set_drain_failure(bool value) {
    impl_->adapter->test_fail_drain = value;
}
#endif

void ZImageExactStream::run_pass(
        uint32_t pass, uint32_t step, Tensor &unified,
        const Tensor &freqs, const Tensor &temb) {
    require(impl_ && !impl_->finished,
            "Z-Image exact executor is unavailable");
    impl_->adapter->bind_pass(pass, step, unified, freqs, temb);
    try {
        start();
        impl_->executor->run_pass(pass, step, impl_->cancelled);
    } catch (const Cancelled &) {
        // A cooperative refill cancellation is cleanup detail, not a new
        // primary runtime failure. Keep the API's typed cancellation result.
        drain_safely();
        throw;
    } catch (const std::exception &primary) {
        drain_safely();
        if (std::strcmp(primary.what(), "streaming_cancelled") == 0)
            throw Cancelled();
        const auto detail = impl_->adapter->fill_error();
        if (!detail.empty())
            throw std::runtime_error(std::string(primary.what()) + " ; Z-Image exact fill: " + detail);
        throw;
    } catch (...) {
        drain_safely();
        throw;
    }
    impl_->adapter->copy_pass_result(unified);
    impl_->adapter->unbind_pass();
}

void ZImageExactStream::finish() {
    require(impl_ && !impl_->finished,
            "Z-Image exact executor is already finished");
    impl_->final_counters = impl_->executor->finish();
    impl_->finished = true;
}

void ZImageExactStream::enable_receipt(
        streaming::ExecutionReceiptOptions options) {
    require(impl_ && !impl_->finished,
            "Z-Image exact receipt is unavailable");
    start();
    impl_->executor->enable_receipt(std::move(options));
}

std::shared_ptr<const streaming::ActualStageReceipt>
ZImageExactStream::receipt() const {
    require(impl_ && impl_->finished,
            "Z-Image exact receipt is not finalized");
    return impl_->executor->receipt();
}

const z_image::StreamingPlanView &ZImageExactStream::plan() const {
    require(impl_ != nullptr, "Z-Image exact plan is unavailable");
    return impl_->plan;
}

const BlockResidencyMetrics &ZImageExactStream::metrics() const {
    require(impl_ != nullptr, "Z-Image exact metrics are unavailable");
    return impl_->source.metrics();
}

streaming::ExecutionCounters ZImageExactStream::counters() const {
    require(impl_ != nullptr, "Z-Image exact counters are unavailable");
    return impl_->finished ? impl_->final_counters : impl_->executor->counters();
}

ZImage::ZImage(const std::filesystem::path &root)
    : ZImage(root, "z-image-turbo", {}) {}

ZImage::ZImage(const std::filesystem::path &root, std::string model_id,
               const std::filesystem::path &transformer_checkpoint)
    : root_(root), model_id_(std::move(model_id)), tokenizer_(root / "tokenizer") {
    optimizations_ = device_info().optimizations();
    auto comfy_text = root / "split_files/text_encoders/qwen_3_4b.safetensors";
    auto comfy_transformer =
        root / "split_files/diffusion_models/z_image_turbo_bf16.safetensors";
    if (!std::filesystem::is_regular_file(comfy_transformer)) {
        auto int8 = root / "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors";
        if (std::filesystem::is_regular_file(int8)) comfy_transformer = int8;
        else comfy_transformer = root / "split_files/diffusion_models/z_image_turbo_nvfp4.safetensors";
    }
    if (const char *override_path = std::getenv("TURBOCIDER_Z_IMAGE_TRANSFORMER");
        override_path && *override_path)
        comfy_transformer = std::filesystem::canonical(override_path);
    convrot_transformer_ = comfy_transformer.filename().string().find("convrot") !=
                           std::string::npos;
    nvfp4_transformer_ = std::filesystem::is_regular_file(comfy_transformer) &&
                         comfy_transformer.filename().string().find("nvfp4") != std::string::npos;
    auto comfy_vae = root / "split_files/vae/ae.safetensors";
    if (!transformer_checkpoint.empty()) {
        require(std::filesystem::is_regular_file(transformer_checkpoint) &&
                    transformer_checkpoint.extension() == ".gguf",
                "native Z-Image GGUF transformer checkpoint is invalid");
        transformer_path_ = std::filesystem::canonical(transformer_checkpoint);
        transformer_checkpoint_ = transformer_path_;
        gguf_transformer_ = true;
        nvfp4_transformer_ = false;
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
        text_path_ = has_safetensors(root / "text_encoder")
                         ? root / "text_encoder" : comfy_text;
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

ZImage::~ZImage() = default;

std::vector<streaming::SourceFileIdentity> ZImage::streaming_source_files() const {
    require(!diffusers_layout_ && !gguf_transformer_ && !convrot_transformer_ &&
                !nvfp4_transformer_, "streaming_artifact_verification_unsupported");
    streaming::SourceFileIdentity transformer_file;
    transformer_file.logical_id = "transformer";
    transformer_file.path = transformer_path_;
    streaming::SourceFileIdentity text_file;
    text_file.logical_id = "text_encoder";
    text_file.path = text_path_;
    streaming::SourceFileIdentity vae_file;
    vae_file.logical_id = "vae";
    vae_file.path = vae_path_;
    streaming::SourceFileIdentity tokenizer_file;
    tokenizer_file.logical_id = "tokenizer";
    tokenizer_file.path = root_ / "tokenizer/tokenizer.json";
    return {std::move(transformer_file), std::move(text_file),
            std::move(vae_file), std::move(tokenizer_file)};
}

std::shared_ptr<const streaming::SourceLease>
ZImage::verify_streaming_sources(std::atomic<bool> &cancelled) {
    // Once requested, a failed/changed proof must not silently return to v1.
    streaming_content_identity_ = true;
    return streaming::SourceLease::capture_verified(streaming_source_files(), &cancelled);
}

std::shared_ptr<const streaming::ModelStreamingProbe>
ZImage::probe_public_streaming(
        const streaming::PublicResolveInput &input) const {
    const auto &request = input.request;
    require(request.model == model_id_,
            "streaming_engine_model_mismatch");
    require(request.operation == "image.generate" && request.frames == 1 &&
                request.inputs.empty() && !request.audio,
            "streaming_route_unsupported: Z-Image public card is text-to-image only");
    require(request.execution == "gpu" && request.ane_manifest.empty() &&
                request.encoder_ane_manifest.empty(),
            "streaming_route_unsupported: Z-Image public card is GPU-only");
    require(!request.allow_approximation && !request.compile_gpu &&
                request.loras.empty() && !diffusers_layout_ &&
                !gguf_transformer_ && !convrot_transformer_ &&
                !nvfp4_transformer_,
            "streaming_route_unsupported: Z-Image public card requires Comfy BF16 eager GPU without LoRA/quantization");
    require(std::filesystem::is_regular_file(transformer_path_) &&
                transformer_path_.extension() == ".safetensors" &&
                std::filesystem::is_regular_file(text_path_) &&
                text_path_.extension() == ".safetensors" &&
                std::filesystem::is_regular_file(vae_path_) &&
                vae_path_.extension() == ".safetensors",
            "streaming_route_unsupported: Z-Image public card requires single-file transformer/text/VAE artifacts");

    auto files = streaming_source_files();
    auto lease = streaming::SourceLease::capture_for_query(
        std::move(files), streaming_content_identity_);
    auto tokenizer_fd = lease->duplicate_fd("tokenizer");
    Tokenizer tokenizer(tokenizer_fd.get(), lease->file("tokenizer").bytes);
    const auto tokens = tokenizer.z_image_prompt(request.prompt, request.dynamic_text);
    lease->revalidate_open_files();
    lease->revalidate_paths();
    const uint32_t caption_rows = padded_z_image_rows(
        static_cast<uint32_t>(tokens.ids.size()));

    streaming::PresetWorkload workload;
    workload.model = model_id_;
    workload.operation = request.operation;
    workload.execution = request.execution;
    workload.device_class = input.device.device_class;
    workload.execution_container = input.execution_container;
    workload.width = static_cast<uint32_t>(request.width);
    workload.height = static_cast<uint32_t>(request.height);
    workload.frames = 1;
    workload.fps = 0;
    workload.steps = static_cast<uint32_t>(request.steps);
    workload.batch = 1;
    workload.audio = false;
    workload.dynamic_text = request.dynamic_text;
    workload.approximation = false;
    workload.conditioning_revision = "qwen3-simple-flow-shift3-v1";
    workload.vae_policy_revision = "z-image-vae-v1";
    workload.feature_digest = z_image_public_feature_digest(
        request, caption_rows);
    workload.token_shapes.push_back({
        "qwen3", "qwen3-z-image-v1", "z-image-template-v1",
        static_cast<uint32_t>(tokens.valid), caption_rows, caption_rows});

    return std::make_shared<streaming::ValueModelStreamingProbe>(
        streaming::ValueModelStreamingProbe::Values{
            model_id_, z_image_public_source_identity(*lease),
            std::move(workload), z_image_public_runtime_identity(),
            kZImagePublicComponentPolicy, std::move(lease)});
}

std::shared_ptr<const streaming::ModelStreamingSnapshot>
ZImage::compile_public_streaming(
        std::shared_ptr<const streaming::ModelStreamingProbe> probe,
        const streaming::StreamingPresetRecord &record) const {
    auto value_probe =
        std::dynamic_pointer_cast<const streaming::ValueModelStreamingProbe>(
            probe);
    require(value_probe != nullptr,
            "streaming_public_probe_type_mismatch");
    require(value_probe->model_id() == model_id_,
            "streaming_probe_identity_mismatch");
    require(value_probe->component_policy_revision() ==
                record.plan.component_policy_revision,
            "streaming_probe_identity_mismatch");
    require(record.source == value_probe->source_identity() &&
                record.workload == value_probe->workload_identity() &&
                record.runtime == value_probe->runtime_identity(),
            "streaming_record_identity_mismatch");
    const auto &workload = value_probe->workload_identity();
    require(workload.token_shapes.size() == 1,
            "streaming_workload_invalid: Z-Image token shape count");
    z_image::StreamingWorkload descriptor_workload{
        workload.width, workload.height,
        workload.token_shapes.front().padded_rows,
        workload.steps};
    auto plan = std::make_shared<z_image::StreamingPlanView>(
        value_probe->lease_ptr(), record.plan.canonical_config,
        descriptor_workload);
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
    if (!record.plan.layout_digest.empty())
#endif
        require(plan->layout().digest == record.plan.layout_digest,
                "streaming_layout_digest_mismatch");
    return std::make_shared<streaming::ValueModelStreamingSnapshot>(
        streaming::ValueModelStreamingSnapshot::Values{
            model_id_, value_probe->source_identity(),
            value_probe->runtime_identity(), plan->descriptor(),
            plan->layout(), std::string(value_probe->component_policy_revision()),
            value_probe->lease_ptr()});
}

RunResult ZImage::generate_resolved(
        std::shared_ptr<const streaming::ResolvedRequestExecution> execution,
        const Event &event, std::atomic<bool> &cancelled) {
    require(execution != nullptr,
            "streaming_authority_mismatch");
    require(execution->model_snapshot != nullptr &&
                execution->probe != nullptr,
            "streaming_authority_mismatch");
    require(execution->model_snapshot->source_lease() != nullptr,
            "streaming_source_lease_required");
    require(execution->request.streaming.active(),
            "streaming_actual_plan_mismatch");
    require(!public_stream_lease_,
            "streaming_public_request_reentrant");
    auto value_probe =
        std::dynamic_pointer_cast<const streaming::ValueModelStreamingProbe>(
            execution->probe);
    require(value_probe != nullptr,
            "streaming_public_probe_type_mismatch");
    auto lease = value_probe->lease_ptr();
    require(lease != nullptr &&
                execution->model_snapshot->source_lease() == lease.get() &&
                execution->probe->source_lease() == lease.get(),
            "streaming_source_lease_mismatch");
    require(execution->model_snapshot->model_id() == model_id_ &&
                execution->selection.record.source ==
                    execution->model_snapshot->source_identity() &&
                execution->selection.record.runtime ==
                    execution->model_snapshot->runtime_identity() &&
                execution->selection.record.plan.layout_digest ==
                    execution->model_snapshot->layout().digest &&
                execution->selection.record.plan.component_policy_revision ==
                    execution->model_snapshot
                        ->component_policy_revision(),
            "streaming_authority_mismatch");
    const auto target = execution->selection.exact_selector
                            .target_request_memory_bytes;
    require(target && streaming::supported_streaming_target(*target),
            "streaming_target_unsupported");
    const auto previous_target = public_stream_target_bytes_;
    public_stream_lease_ = std::move(lease);
    public_stream_target_bytes_ = *target;
    try {
        auto tokenizer_fd = public_stream_lease_->duplicate_fd("tokenizer");
        public_stream_tokenizer_ = std::make_unique<Tokenizer>(
            tokenizer_fd.get(), public_stream_lease_->file("tokenizer").bytes);
        public_stream_lease_->revalidate_open_files();
        public_stream_lease_->revalidate_paths();
        auto result = run(execution->request, event, cancelled, false, false);
        public_stream_target_bytes_ = previous_target;
        public_stream_tokenizer_.reset();
        public_stream_lease_.reset();
        return result;
    } catch (...) {
        public_stream_target_bytes_ = previous_target;
        public_stream_tokenizer_.reset();
        public_stream_lease_.reset();
        throw;
    }
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
    encoder_hybrid_.reset();
    hybrid_gpu_graph_ = {};
    hybrid_gpu_mlp_start_ = -1;
    cached_conditioning_.reset();
    cached_encoder_hybrid_metrics_.reset();
    cached_prompt_.clear();
    cached_encoder_manifest_.clear();
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
        nvfp4_transformer_ = transformer_.nvfp4("layers.0.attention.qkv");
        if (nvfp4_transformer_) {
            require(active_loras_.empty() && !hybrid_, "NVFP4 currently supports GPU without LoRA/ANE only");
            transformer_.pack_comfy_nvfp4();
        }
        if (convrot_transformer_)
            transformer_.cast_unquantized_float32(mx::bfloat16);
        event("load_z_image_transformer", 1, 1);
    }
    load_vae(event, cancelled);
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

void ZImage::load_vae(
        const Event &event, std::atomic<bool> &cancelled) {
    if (vae_.bytes() != 0)
        return;
    checkpoint(cancelled);
    event("load_z_image_vae", 0, 1);
    if (public_stream_lease_)
        vae_.load_lease(public_stream_lease_, {"vae"}, event, cancelled);
    else
        load_z_component(vae_, vae_path_, event, cancelled);
    if (diffusers_layout_)
        normalize_z_diffusers_vae(vae_);
    event("load_z_image_vae", 1, 1);
}

void ZImage::unload() {
    exact_stream_.reset();
    weight_stream_.reset();
    stream_configuration_.clear();
    hybrid_.reset();
    encoder_hybrid_.reset();
    hybrid_gpu_graph_ = {};
    hybrid_gpu_mlp_start_ = -1;
    cached_conditioning_.reset();
    cached_encoder_hybrid_metrics_.reset();
    cached_prompt_.clear();
    cached_encoder_manifest_.clear();
    text_encoder_.clear();
    transformer_.clear();
    vae_.clear();
    public_component_cache_ = false;
    mx::clear_cache();
}

Tensor ZImage::encode_text(const Tokens &tokens, const Event &event, std::atomic<bool> &cancelled) try {
    if (text_encoder_.bytes() == 0) {
        if (public_stream_lease_)
            text_encoder_.load_lease(public_stream_lease_, {"text_encoder"}, event, cancelled);
        else
            load_z_component(text_encoder_, text_path_, event, cancelled);
    }
    auto result = components::qwen3_conditioning(
        tokens, text_encoder_, components::Qwen3Conditioning::z_image(), event, cancelled,
        encoder_hybrid_.get());
    result = slice_axis(mx::squeeze(result, 0), 0, 0, tokens.valid);
    text_encoder_.clear();
    mx::clear_cache();
    return result;
} catch (...) {
    // An interrupted prompt must not retain Qwen3 alongside the next denoiser.
    if (optimizations_.z_image_memory_lifecycle) {
        text_encoder_.clear();
        mx::clear_cache();
    }
    throw;
}

bool ZImage::conditioning(const Request &r, const Event &event, std::atomic<bool> &cancelled) {
    if (cached_conditioning_ && cached_prompt_ == r.prompt && cached_dynamic_ == r.dynamic_text &&
        cached_encoder_manifest_ == r.encoder_ane_manifest) {
        event("z_image_text_cache_hit", 1, 1);
        return true;
    }
    auto tokens = (public_stream_tokenizer_ ? *public_stream_tokenizer_ : tokenizer_)
                      .z_image_prompt(r.prompt, r.dynamic_text);
    if (!r.encoder_ane_manifest.empty()) {
        const auto prefill = components::qwen3_prefill_plan(
            r.encoder_ane_manifest, int(tokens.ids.size()));
        for (const auto &lora : active_loras_)
            require(lora.role != "text_encoder",
                    "Qwen3 encoder hybrid does not yet support text-encoder LoRA");
        if (prefill.use_hybrid) {
            if (!encoder_hybrid_ || encoder_hybrid_->manifest != r.encoder_ane_manifest)
                encoder_hybrid_ = std::make_unique<HybridSession>(
                    r.encoder_ane_manifest, text_path_, int(tokens.ids.size()), event,
                    cancelled, r.warmup_iterations,
                    components::qwen3_checkpoint_path(text_path_),
                    std::vector<LoRAAsset>{}, 0, 35, true);
            encoder_hybrid_->set_tokens(int(tokens.ids.size()));
            require(encoder_hybrid_->rows == prefill.compute_tokens,
                    "Qwen3 encoder manifest bucket changed during prefill setup");
        } else {
            encoder_hybrid_.reset();
            event("qwen3_encoder_gpu_" + prefill.reason, 1, 1);
        }
    } else {
        encoder_hybrid_.reset();
    }
    auto encoded = encode_text(tokens, event, cancelled);
    auto encoder_metrics = encoder_hybrid_
        ? std::optional<HybridMetrics>(encoder_hybrid_->metrics()) : std::nullopt;
    cached_conditioning_ = std::move(encoded);
    cached_encoder_hybrid_metrics_ = std::move(encoder_metrics);
    cached_prompt_ = r.prompt;
    cached_dynamic_ = r.dynamic_text;
    cached_encoder_manifest_ = r.encoder_ane_manifest;
    return false;
}

std::string ZImage::select_acceleration(Request &r, int rows, const Event &event,
                                        std::atomic<bool> &cancelled) {
    if (nvfp4_transformer_) {
        require(active_loras_.empty() && r.execution != "gpu_ane", "NVFP4 currently supports GPU without LoRA/ANE only");
        r.execution = "gpu";
        hybrid_.reset();
        return "gpu: Comfy NVFP4 weights via MLX W4A16; no activation quantization";
    }
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
            hybrid_gpu_graph_ = z_image::make_hybrid_gpu_graph(
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
                      bool warmup, bool load_only) try {
    require(!streaming_quarantined_, "streaming_process_quarantined: restart the process");
    auto r = requested;
    ZProfileRequest profile(r);
    auto begin = Clock::now();
    require(r.model == model_id_, "Z-Image session received a different model id");
    auto plan = make_plan(r);
    require(!r.prompt.empty() && (warmup || load_only || !r.output.empty()),
            "prompt and output are required");
    require(warmup || load_only || std::filesystem::path(r.output).extension() == ".png",
            "Z-Image output must be .png");
    require(r.inputs.empty(), "Z-Image-Turbo currently supports text-to-image only");
    require(r.width % 16 == 0 && r.height % 16 == 0, "Z-Image dimensions must be multiples of 16");
    const bool exact_streaming = z_image_exact_streaming_requested(r);
    const uint32_t exact_slot_count = exact_streaming
        ? z_image_exact_slot_count(r) : 0;
    const bool tight_exact = exact_streaming && exact_slot_count == 1;
    const bool legacy_streamed = r.residency == "streamed";
    const bool streamed = exact_streaming || legacy_streamed;
    const bool constrained_memory =
        optimizations_.z_image_memory_lifecycle &&
        !ResidencyPolicy::for_request(
            r, device_info().physical_memory).retain_images_during_text;
    const auto budget = public_stream_target_bytes_
        ? public_stream_target_bytes_
        : (r.memory_budget_bytes
               ? r.memory_budget_bytes
               : std::min<uint64_t>(10ull << 30,
                                     device_info().physical_memory / 2));
    const unsigned prefetch_layers = legacy_streamed ? optimizations_.z_image_stream_prefetch(
        convrot_transformer_, r.execution == "gpu_ane", budget,
        std::getenv("TURBOCIDER_Z_STREAM_PREFETCH")) : 1;
    // Reserve VAE, temporary activations and allocator cache. This is a planning
    // estimate for the denoiser, not an OS-enforced process memory limit.
    const uint64_t reserve = (3ull << 30) + uint64_t(r.width) * r.height * 2048;
    const auto configuration = streamed ?
        std::to_string(budget) + ":" + std::to_string(reserve) +
        ":prefetch=" + std::to_string(prefetch_layers) +
        (optimizations_.z_image_suffix_streaming
             ? ":" + r.execution + ":" + r.ane_manifest : "") : "";
    if (public_stream_lease_ || public_component_cache_) {
        // Neither incoming nor outgoing public requests may inherit unkeyed
        // component caches. Keep the marker on failure until the next request
        // safely synchronizes; do not release possibly live arrays in a catch.
        mx::synchronize();
        cached_conditioning_.reset();
        cached_encoder_hybrid_metrics_.reset();
        cached_prompt_.clear();
        cached_encoder_manifest_.clear();
        text_encoder_.clear();
        vae_.clear();
        public_component_cache_ = bool(public_stream_lease_);
        mx::clear_cache();
    }
    const bool prompt_changed = !cached_conditioning_ ||
        cached_prompt_ != r.prompt || cached_dynamic_ != r.dynamic_text ||
        cached_encoder_manifest_ != r.encoder_ane_manifest;
    if (exact_streaming) {
        // Exact retention is request-scoped. Never inherit resident weights,
        // a legacy prefetcher, or a previous exact executor into this request.
        mx::synchronize();
        exact_stream_.reset();
        weight_stream_.reset();
        transformer_.clear();
        if (tight_exact)
            vae_.clear();
        mx::clear_cache();
        stream_configuration_.clear();
    } else if (configuration != stream_configuration_ ||
               ((legacy_streamed || constrained_memory) && prompt_changed)) {
        mx::synchronize();
        weight_stream_.reset();
        // Do not overlap a previous denoiser/Core ML working set with Qwen3
        // when recomputing conditioning on a small unified-memory machine.
        if (optimizations_.z_image_memory_lifecycle) {
            hybrid_.reset();
            hybrid_gpu_graph_ = {};
            hybrid_gpu_mlp_start_ = -1;
        }
        transformer_.clear();
        vae_.clear();
        mx::clear_cache();
        stream_configuration_ = configuration;
    }
    RequestCacheLimit cache_limit(
        streamed || constrained_memory,
        tight_exact ? 0 : r.allocator_cache_bytes);
    mx::reset_peak_memory();
    select_loras(r);
    auto text_start = Clock::now();
    bool prompt_hit = conditioning(r, event, cancelled);
    if (legacy_streamed && encoder_hybrid_) {
        // Finish every consumer of the shared Core ML output backing before
        // releasing the encoder. Preserve provenance on cached conditioning,
        // so a later cache hit is still reported as hybrid conditioning.
        mx::eval(*cached_conditioning_);
        mx::synchronize();
        encoder_hybrid_.reset();
        if (cached_encoder_hybrid_metrics_)
            cached_encoder_hybrid_metrics_->session_released_after_encoding = true;
        mx::clear_cache();
        event("qwen3_encoder_session_released", 1, 1);
    }
    const double text_seconds = std::chrono::duration<double>(Clock::now() - text_start).count();
    const int image_rows = ((r.height / 16) * (r.width / 16) + 31) / 32 * 32;
    const int caption_rows = (cached_conditioning_->shape(0) + 31) / 32 * 32;
    auto selection = select_acceleration(r, image_rows + caption_rows, event, cancelled);
    event(r.execution == "gpu_ane" ? "route_gpu_ane" : "route_gpu", 1, 1);
    r.compile_gpu = !exact_streaming && r.execution == "gpu" &&
                    !gguf_transformer_ && !convrot_transformer_ && !nvfp4_transformer_ &&
                    active_lora_strategy_ != "inference_time" &&
                    !std::getenv("TURBOCIDER_Z_EAGER_BLOCKS");
    if (plan.request.execution != r.execution || plan.request.compile_gpu != r.compile_gpu)
        plan = make_plan(r);
    if (exact_streaming) {
        require(!load_only && r.execution == "gpu" && !hybrid_ &&
                    active_loras_.empty() && !gguf_transformer_ &&
                    !convrot_transformer_ && !nvfp4_transformer_ &&
                    !diffusers_layout_,
                "streaming_route_unsupported: the Z-Image exact candidate "
                "supports generated Comfy BF16 GPU requests without LoRA, "
                "ANE, quantization, Diffusers shards, or prepare-only mode");
        require(exact_stream_generation_ != UINT64_MAX,
                "Z-Image exact request generation overflow");
        ++exact_stream_generation_;
        const z_image::StreamingWorkload workload{
            uint32_t(r.width), uint32_t(r.height), uint32_t(caption_rows),
            uint32_t(r.steps)};
        if (public_stream_lease_) {
            exact_stream_ = std::make_unique<ZImageExactStream>(
                public_stream_lease_, r.streaming, workload, budget, reserve,
                transformer_, event, cancelled, exact_stream_generation_);
            exact_stream_->enable_receipt({
                exact_stream_->plan().layout().digest,
                kZImagePublicImplementation,
                public_stream_lease_->generation()});
        } else {
            exact_stream_ = std::make_unique<ZImageExactStream>(
                transformer_path_, r.streaming, workload, budget, reserve,
                transformer_, event, cancelled, exact_stream_generation_);
        }
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
        exact_stream_->test_set_drain_failure(test_fail_drain_);
#endif
        exact_stream_->start();
    } else if (legacy_streamed) {
        require(!gguf_transformer_ && !nvfp4_transformer_ && !diffusers_layout_,
                "Z-Image streaming requires Comfy BF16 or INT8 ConvRot weights");
        require(!hybrid_ || !std::getenv("TURBOCIDER_Z_HYBRID_VALIDATE"),
                "full-MLP hybrid validation requires resident weights");
        if (!weight_stream_)
            weight_stream_ = std::make_unique<ZImageWeightStream>(
                transformer_path_, budget, reserve, transformer_, event, cancelled,
                hybrid_ && optimizations_.z_image_suffix_streaming ? hybrid_->ane_mlp_end : 0,
                prefetch_layers);
        else weight_stream_->reset_metrics();
    }
    if (tight_exact) {
        // K1 is the minimum-memory exact layout.  Its fixed/prefix tensors are
        // already materialized by ZImageWeightStream; loading the VAE here
        // would keep another 320+ MiB resident throughout all denoise passes.
        transformer_.materialize();
    } else {
        load(event, cancelled);
    }
    if (load_only) {
        RunResult result;
        result.prepared = true;
        result.request = r;
        result.plan = std::move(plan);
        result.selection = selection;
        result.prompt_cache_hit = prompt_hit;
        auto reported_tokens = (public_stream_tokenizer_ ? *public_stream_tokenizer_ : tokenizer_)
                                   .z_image_prompt(r.prompt, r.dynamic_text);
        result.text_tokens = int(reported_tokens.ids.size());
        result.valid_text_tokens = reported_tokens.valid;
        result.total_tokens = image_rows + caption_rows;
        result.lora_applied_projections = lora_applied_projections_;
        if (gguf_transformer_) {
            result.backend = hybrid_ ? "mlx_cpp_metal_gguf+coreml" : "mlx_cpp_metal_gguf";
            result.precision = "gguf_native:" + r.model_variant;
            result.checkpoint = transformer_checkpoint_.filename().string();
        } else if (nvfp4_transformer_) {
            result.backend = "mlx_cpp_metal_nvfp4_w4a16";
            result.precision = "nvfp4_weights_bf16_activations";
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
        result.encoder_hybrid = cached_encoder_hybrid_metrics_;
        result.timings.wall =
            std::chrono::duration<double>(Clock::now() - begin).count();
        result.timings.text = text_seconds;
        result.peak_bytes = mx::get_peak_memory();
        result.active_bytes = mx::get_active_memory();
        if (weight_stream_) result.block_residency = weight_stream_->metrics();
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
    profile.phase("denoise_begin");
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
    std::optional<BlockResidencyMetrics> exact_metrics;
    std::optional<StreamingRuntimeMetrics> exact_runtime;
    std::shared_ptr<const streaming::ActualExecutionReceipt> exact_receipt;
    streaming::ExecutionCounters exact_counters{};
    auto finish_exact_stream = [&] {
        require(exact_streaming && exact_stream_ != nullptr,
                "Z-Image exact executor is unavailable at drain");
        exact_stream_->finish();
        exact_metrics = exact_stream_->metrics();
        exact_counters = exact_stream_->counters();
        exact_metrics->request_wait_seconds = exact_counters.wait_seconds;
        const auto &layout = exact_stream_->plan().layout();
        const auto &stage = layout.stages.front();
        StreamingRuntimeMetrics runtime;
        runtime.implementation = public_stream_lease_
            ? kZImagePublicImplementation : "generic_stage_executor_v1";
        runtime.layout_digest = layout.digest;
        runtime.resident_prefix_blocks = stage.prefix;
        runtime.block_group_size = stage.group_size;
        runtime.slot_count = stage.slot_count;
        runtime.prefetch_distance = stage.distance;
        runtime.io_workers = stage.workers;
        runtime.group_count = uint32_t(stage.groups.size());
        runtime.pass_count = stage.pass_count;
        runtime.startup_policy = "prefetch_window_before_prefix";
        runtime.pass_transition = "reload";
        runtime.retention = "request";
        runtime.reader_revision = 1;
        runtime.weight_format = "comfy-bf16-single-file";
        runtime.kernel_revision = kZImageKernelRevision;
        runtime.conditioning_recipe = "qwen3-simple-flow-shift3-v1";
        runtime.upsample_boundary = (public_stream_lease_ || tight_exact)
            ? "no-upsample;denoiser-pool-drained-before-vae"
            : "no-upsample;denoiser-pool-retained-through-vae";
        runtime.component_policy_revision = public_stream_lease_
            ? kZImagePublicComponentPolicy : "";
        runtime.multi_pool_policy =
            stage.multi_pool_policy == streaming::MultiPoolPolicy::serial
                ? "serial" : "retain_all";
        runtime.pool_count = uint32_t(stage.pools.size());
        runtime.slot_bundle_count = exact_counters.slot_bundles;
        runtime.refill_worker_count = stage.workers;
        runtime.drained = true;
        if (public_stream_lease_) {
            const auto stage_receipt = exact_stream_->receipt();
            require(stage_receipt != nullptr,
                    "streaming_actual_receipt_missing");
            exact_receipt = std::make_shared<
                const streaming::ActualExecutionReceipt>(
                    streaming::make_actual_execution_receipt(
                        kZImagePublicImplementation, layout.digest,
                        kZImagePublicComponentPolicy,
                        std::vector<streaming::ActualStageReceipt>{
                            *stage_receipt}));
        }
        exact_runtime = std::move(runtime);
        require(exact_counters.slot_bundles == stage.slot_count &&
                    exact_counters.fills ==
                        uint64_t(exact_metrics->streamed_blocks) *
                            uint64_t(r.steps) &&
                    exact_counters.groups_submitted == exact_counters.fills &&
                    exact_metrics->request_slot_fills ==
                        exact_counters.fills,
                "Z-Image exact counters differ from the compiled layout");
        // The exact slot pool was released by finish(). Prefix/fixed weights
        // also have request retention and must not leak into VAE or a later
        // request. Public execution drains here to reduce the full-request
        // peak; the private candidate retains its historical post-VAE drain.
        exact_stream_.reset();
        transformer_.clear();
    };
    if (exact_streaming && (public_stream_lease_ || tight_exact)) {
        finish_exact_stream();
        mx::clear_cache();
    }
    profile.phase("denoise_end");
    checkpoint(cancelled);
    if (tight_exact) {
        load_vae(event, cancelled);
        vae_.materialize();
    }
    auto decode_start = Clock::now();
    auto decoded = decode(z, r.width, r.height, event, cancelled);
    dump("z_latent_final", z);
    dump("z_decoded", decoded);
    const double decode_seconds = std::chrono::duration<double>(Clock::now() - decode_start).count();
    profile.phase("decode_end");
    require(mx::all(mx::isfinite(decoded)).item<bool>(), "nonfinite Z-Image pixels");
    auto pixels = mx::transpose(decoded, {0, 2, 3, 1});
    if (!warmup) {
        event("export", 0, 1);
        checkpoint(cancelled);
        save_png(pixels, r.output);
        event("export", 1, 1);
    }
    if (exact_streaming && exact_stream_)
        finish_exact_stream();
    RunResult result;
    result.request = r;
    result.plan = std::move(plan);
    result.selection = selection;
    result.warmup = warmup;
    result.prompt_cache_hit = prompt_hit;
    auto reported_tokens = (public_stream_tokenizer_ ? *public_stream_tokenizer_ : tokenizer_)
                                   .z_image_prompt(r.prompt, r.dynamic_text);
    result.text_tokens = int(reported_tokens.ids.size());
    result.valid_text_tokens = reported_tokens.valid;
    result.actual_steps = r.steps;
    result.lora_applied_projections = lora_applied_projections_;
    if (gguf_transformer_) {
        result.backend = hybrid_ ? "mlx_cpp_metal_gguf+coreml" : "mlx_cpp_metal_gguf";
        result.precision = "gguf_native:" + r.model_variant;
        result.checkpoint = transformer_checkpoint_.filename().string();
    } else if (nvfp4_transformer_) {
        result.backend = "mlx_cpp_metal_nvfp4_w4a16";
        result.precision = "nvfp4_weights_bf16_activations";
        result.checkpoint = transformer_checkpoint_.filename().string();
    } else if (convrot_transformer_) {
        result.backend = "mlx_cpp_metal_convrot_packed_q8";
        result.precision = "int8_tensorwise_convrot_g256";
        result.checkpoint = transformer_checkpoint_.filename().string();
    }
    if (!gguf_transformer_ && !nvfp4_transformer_ &&
        !convrot_transformer_ && !hybrid_) {
        result.backend = "mlx_cpp_metal";
        result.precision = "bf16";
    }
    if (hybrid_)
        result.hybrid = hybrid_->metrics();
    result.encoder_hybrid = cached_encoder_hybrid_metrics_;
    result.timings.wall = std::chrono::duration<double>(Clock::now() - begin).count();
    result.timings.text = text_seconds;
    result.timings.denoise = denoise_seconds;
    result.timings.decode = decode_seconds;
    result.peak_bytes = mx::get_peak_memory();
    result.active_bytes = mx::get_active_memory();
    if (exact_metrics) {
        result.block_residency = *exact_metrics;
        result.streaming_runtime = *exact_runtime;
        result.streaming_receipt = std::move(exact_receipt);
    } else if (weight_stream_) {
        const auto metrics = weight_stream_->metrics();
        result.block_residency = metrics;
        StreamingRuntimeMetrics runtime;
        runtime.implementation = "mlx_specialized_double_buffer_v1";
        runtime.resident_prefix_blocks = metrics.pinned_blocks;
        runtime.block_group_size = 1;
        runtime.slot_count = metrics.refill_slots;
        runtime.prefetch_distance = 0;
        runtime.io_workers = 1;
        runtime.group_count = metrics.streamed_blocks;
        runtime.pass_count = uint32_t(r.steps);
        runtime.startup_policy = "prefetch_window_before_prefix";
        runtime.pass_transition = "reload";
        runtime.retention = "engine";
        runtime.reader_revision = 1;
        runtime.weight_format = "comfy-bf16-single-file";
        runtime.kernel_revision = kZImageKernelRevision;
        runtime.conditioning_recipe = "qwen3-simple-flow-shift3-v1";
        runtime.upsample_boundary =
            "no-upsample;denoiser-pool-retained-through-vae";
        result.streaming_runtime = std::move(runtime);
    }
    return result;
} catch (...) {
    if (exact_stream_ && !exact_stream_->drain_safely()) {
        streaming_quarantined_ = true;
        throw;
    }
    if (exact_stream_ || weight_stream_ || !stream_configuration_.empty()) {
        // Cancellation can leave a prefetch outstanding. Join it before a retry
        // resets the cancellation flag or reuses any of its destination buffers.
        try { mx::synchronize(); } catch (...) {}
        exact_stream_.reset();
        weight_stream_.reset();
        transformer_.clear();
        vae_.clear();
        stream_configuration_.clear();
        mx::clear_cache();
    }
    throw;
}

Tensor ZImage::denoise(const Tensor &latent, const Tensor &caption, float sigma, float width,
                       int height, int step, const Event &event,
                       std::atomic<bool> &cancelled) {
    auto model_input = mx::astype(latent, mx::bfloat16);
    return mx::astype(
        z_transformer(model_input, caption, sigma, int(width), height, transformer_, event,
                      cancelled, hybrid_.get(), hybrid_ ? &hybrid_gpu_graph_ : nullptr,
                      weight_stream_.get(), exact_stream_.get(), uint32_t(step),
                      optimizations_.z_image_hybrid_segments),
        mx::float32);
}

Tensor ZImage::decode(const Tensor &latent, int, int, const Event &, std::atomic<bool> &) {
    return z_image::decode_vae(latent, vae_);
}

} // namespace tc

#include "../../runtime/build_identity.hpp"
#include "z_image.hpp"
#include "block_profile.hpp"
#include "hybrid_math.hpp"
#include "ffn.hpp"
#include "padding.hpp"
#include "hybrid_stream.hpp"
#include "vae.hpp"
#include "metal_kernels.hpp"

#include "../../media/image.hpp"
#include "../../platform/apple/platform.hpp"
#include "../../runtime/acceleration.hpp"
#include "../../runtime/residency.hpp"
#include "../../runtime/streaming/canonical_encoding.hpp"
#include "../../runtime/streaming/context.hpp"
#include "../../runtime/streaming/gguf_packed_bank.hpp"
#include "../../runtime/streaming/resolved_request.hpp"
#include "streaming_descriptor.hpp"
#include "gguf_execution.hpp"
#include "../../components/text/qwen3.hpp"
#include "../../components/text/qwen3_gguf.hpp"

#include <array>
#include <bit>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <fcntl.h>
#include <regex>
#include <sstream>
#include <unistd.h>

namespace tc {

using ZImageGpuGraph = std::function<std::vector<Tensor>(const std::vector<Tensor> &)>;

static const char *z_qwen3_gguf_path() {
    const char *path = std::getenv("TURBOCIDER_Z_QWEN3_GGUF");
    if (!path) return nullptr;
#ifndef TURBOCIDER_ENABLE_QUANTIZED_EXECUTION_EXPERIMENTS
    throw std::runtime_error("qe_capability_unqualified: Qwen3 GGUF requires an explicit experimental build");
#else
    require(*path && std::filesystem::path(path).extension() == ".gguf", "qe_config_conflict: invalid Qwen3 GGUF path");
    return path;
#endif
}
static uint64_t z_qwen3_gguf_integer(const char *name, uint64_t fallback, uint64_t maximum) {
    const char *raw = std::getenv(name);
    if (!raw) return fallback;
    const std::string value(raw);
    require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
            std::string("qe_config_conflict: invalid ") + name);
    const uint64_t result = std::stoull(value);
    require(result <= maximum, std::string("qe_config_conflict: out of range ") + name); return result;
}

class ZImageGgufStream {
  public:
    ZImageGgufStream(const std::filesystem::path &, uint32_t prefetch,
                    uint32_t width, uint32_t height, uint32_t caption, uint32_t steps,
                    uint64_t managed_budget, Weights &fixed, const Event &, std::atomic<bool> &, const std::string &profile,
                    const std::string &residency);
    ~ZImageGgufStream();
    void run_pass(uint32_t, Tensor &, const Tensor &, const Tensor &);
    bool streams_refiners() const noexcept;
    void run_refiners(uint32_t, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &);
    void finish();
    bool drain_safely() noexcept;
    QuantizedExecutionMetrics metrics() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

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
    void bind_hybrid(HybridSession *, ZImageGpuGraph *, bool compile_segments);
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
    "zimage-components-v3-shared-text-request-cache";

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
        tc::catalog_runtime_identity(),
        "public-streaming-runtime-v2",
        "z-image-public-adapter-v6-text-capacity",
        "z-image-pread-bf16-v2-fd-lease",
        kZImageKernelRevision,
        "mlx-request-cache-policy-v2-k1-zero-cache",
    };
}

std::string z_image_public_feature_digest(const Request &request) {
    streaming::CanonicalEncoder feature(
        "z-image-public-workload-features-v2");
    feature.boolean_field("inputs_empty", request.inputs.empty());
    feature.boolean_field("loras_empty", request.loras.empty());
    feature.boolean_field("ane_disabled", request.ane_manifest.empty());
    feature.boolean_field(
        "encoder_ane_disabled", request.encoder_ane_manifest.empty());
    feature.boolean_field("compile_gpu", request.compile_gpu);
    feature.boolean_field("dynamic_text", request.dynamic_text);
    return feature.sha256();
}

// Research-only A8 layer ablation. Skipped blocks retain complete BF16 GPU
// weights; participating blocks may still pack only their GPU W8 suffix.
bool z_hybrid_bf16_block(int ordinal) {
    static const auto configured = [] {
        std::array<bool, 32> selected{};
        const char *raw = std::getenv("TURBOCIDER_Z_HYBRID_BF16_BLOCKS");
        if (!raw) return selected;
        require(*raw, "hybrid BF16 block list must not be empty");
        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ',')) {
            size_t consumed = 0;
            int block = -1;
            try { block = std::stoi(token, &consumed); }
            catch (const std::exception &) {
                throw std::runtime_error("invalid hybrid BF16 block index: " + token);
            }
            require(consumed == token.size() && block >= 0 && block < 32 &&
                        !selected[block], "invalid or repeated hybrid BF16 block index");
            selected[block] = true;
        }
        require(raw[std::strlen(raw) - 1] != ',', "hybrid BF16 block list has trailing comma");
        return selected;
    }();
    return configured[ordinal];
}

// Opt-in calibration capture of the *actual* normalized/modulated FFN input.
// A separate file per call covers multiple denoising timesteps and prompts;
// normal requests never materialize or copy anything for this experiment.
void z_capture_ffn_input(const Tensor &input, int block) {
    const char *directory = std::getenv("TURBOCIDER_Z_FFN_CAPTURE_DIR");
    if (!directory || !*directory) return;
    require(block >= 0 && block < 32 && input.shape(0) == 1 && input.shape(2) == 3840,
            "invalid Z-Image FFN calibration capture geometry");
    if (const char *selected = std::getenv("TURBOCIDER_Z_FFN_CAPTURE_BLOCK")) {
        const std::string raw(selected);
        size_t consumed = 0;
        int selected_block = -1;
        try { selected_block = std::stoi(raw, &consumed); }
        catch (const std::exception &) {
            throw std::runtime_error("invalid Z-Image FFN calibration capture block");
        }
        require(consumed == raw.size() && selected_block >= 0 && selected_block < 32,
                "invalid Z-Image FFN calibration capture block");
        if (selected_block != block) return;
    }
    auto sample = mx::contiguous(mx::astype(input, mx::float16));
    mx::eval(sample);
    static std::atomic<uint64_t> serial{0};
    const auto folder = std::filesystem::path(directory) / ("block" + std::to_string(block));
    std::filesystem::create_directories(folder);
    const auto id = std::to_string(::getpid()) + "-" +
        std::to_string(Clock::now().time_since_epoch().count()) + "-" +
        std::to_string(serial.fetch_add(1));
    const auto path = folder / ("sample-" + id + ".npy");
    const auto partial = folder / ("sample-" + id + ".partial");
    std::string header = "{'descr': '<f2', 'fortran_order': False, 'shape': (" +
        std::to_string(input.shape(1)) + ", 3840), }";
    header.append((16 - (10 + header.size() + 1) % 16) % 16, ' ');
    header += '\n';
    require(header.size() <= UINT16_MAX, "Z-Image calibration header too large");
    std::string prefix("\x93NUMPY", 6);
    prefix.push_back(1);
    prefix.push_back(0);
    prefix.push_back(char(header.size() & 255));
    prefix.push_back(char(header.size() >> 8));
    int fd = ::open(partial.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    require(fd >= 0, "cannot create Z-Image calibration sample");
    auto write_all = [&](const char *data, size_t bytes) {
        while (bytes) {
            auto n = ::write(fd, data, bytes);
            if (n < 0 && errno == EINTR) continue;
            require(n > 0, "cannot write Z-Image calibration sample (check free disk space)");
            data += n;
            bytes -= size_t(n);
        }
    };
    try {
        write_all(prefix.data(), prefix.size());
        write_all(header.data(), header.size());
        write_all(reinterpret_cast<const char *>(sample.data<mx::float16_t>()),
                  sample.size() * sizeof(mx::float16_t));
        require(::close(fd) == 0, "cannot close Z-Image calibration sample");
        fd = -1;
        std::filesystem::rename(partial, path);
    } catch (...) {
        if (fd >= 0) ::close(fd);
        std::filesystem::remove(partial);
        throw;
    }
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

std::vector<Tensor> z_prepare_qkv(const Tensor &qkv, const Tensor &qw,
                                  const Tensor &kw, const Tensor &freqs) {
    // Exact-shape Metal path is the production default after parity. Keep a
    // disable switch for bisecting old binaries and device-specific issues.
    if (!std::getenv("TURBOCIDER_Z_DISABLE_FUSED_QKV") &&
        !std::getenv("TURBOCIDER_Z_EAGER_ROPE"))
        return z_metal::prepare_qkv(qkv, qw, kw, freqs);
    auto parts = mx::split(qkv, 3, -1);
    auto q = rms(heads(parts[0], kHeads, kHeadDim), qw, 1e-5f);
    auto k = rms(heads(parts[1], kHeads, kHeadDim), kw, 1e-5f);
    auto result = z_apply_rope_pair(q, k, freqs);
    result.push_back(heads(parts[2], kHeads, kHeadDim));
    return result;
}

Tensor z_modulate_norm(const Tensor &x, const Tensor &weight, const Tensor &mod,
                       const Tensor &residual, bool gated) {
    // Smaller-thread experiments retain their scalar kernel and reduction
    // geometry. The vector kernel preserves the default 960-thread ordering.
    const char *norm_threads = std::getenv("TURBOCIDER_Z_NORM_THREADS");
    const bool vector_norm = !std::getenv("TURBOCIDER_Z_DISABLE_VECTOR_NORM") &&
        (!norm_threads || std::string(norm_threads) == "960");
    if (const char *virtual_threads = std::getenv("TURBOCIDER_Z_VIRTUAL_NORM_THREADS")) {
        const std::string threads(virtual_threads);
        require(vector_norm && !std::getenv("TURBOCIDER_Z_DISABLE_FUSED_MOD"),
                "virtual norm threads require the fused vector norm path");
        require(threads == "128" || threads == "256" || threads == "512",
                "virtual norm threads must be 128, 256 or 512");
        return z_metal::norm_mod_virtual(x,weight,mod,residual,gated,
            x.dtype() != mx::float32 && !std::getenv("TURBOCIDER_Z_INLINE_GATE_TANH"),
            std::stoi(threads));
    }
    // Qualified large shapes only: image refiner and one padded text block.
    // Keep512, other prompt lengths/devices/dtypes on their original path.
    if (vector_norm && !norm_threads &&
        !std::getenv("TURBOCIDER_Z_DISABLE_FUSED_MOD") &&
        !std::getenv("TURBOCIDER_Z_DISABLE_VIRTUAL_NORM") &&
        x.dtype() == mx::bfloat16 && x.ndim() == 3 && x.shape(0) == 1 &&
        x.shape(2) == 3840 && (x.shape(1) == 4096 || x.shape(1) == 4128) &&
        z_image_virtual_norm_default())
        return z_metal::norm_mod_virtual(x,weight,mod,residual,gated,
            !std::getenv("TURBOCIDER_Z_INLINE_GATE_TANH"),256);
    if (!std::getenv("TURBOCIDER_Z_DISABLE_FUSED_MOD"))
        return z_metal::norm_mod(x, weight, mod, residual, gated,
            x.dtype() != mx::float32 && !std::getenv("TURBOCIDER_Z_INLINE_GATE_TANH"),
            vector_norm);
    auto normalized = rms(x, weight, 1e-5f);
    if (gated)
        return residual + mx::tanh(mod) * normalized;
    return normalized * (Tensor(1.f, mod.dtype()) + mod);
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
    auto rotated = z_prepare_qkv(qkv, w.at(prefix + ".attention.q_norm.weight"),
                                w.at(prefix + ".attention.k_norm.weight"), freqs);
    auto v = rotated[2];
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
    const char *shared = std::getenv("TURBOCIDER_Z_CONVROT_SHARED_GATE_UP");
    return z_image::feed_forward(x, w, prefix, shared && std::strcmp(shared, "1") == 0);
}

float z_hybrid_output_scale(const HybridSession *hybrid) {
    return hybrid ? hybrid->output_scale : 1.f;
}


struct ZQuantizedGeometry {
    int group_size = 0;
    int bits = 0;
};

// Research-only packed-W8 GEMM variants. The packed tensors remain resident;
// dequantized matrices are expression-local and their cost is on the path.
enum class ZHybridDequantMode { direct, bf16, gate_up_fp16, scaled_all_fp16 };

ZHybridDequantMode z_hybrid_dequant_mode() {
    const char *mode = std::getenv("TURBOCIDER_Z_HYBRID_W8_DEQUANT_GEMM");
    if (!mode) return ZHybridDequantMode::direct;
    if (std::strcmp(mode, "1") == 0 || std::strcmp(mode, "bf16") == 0)
        return ZHybridDequantMode::bf16;
    if (std::strcmp(mode, "gate_up_fp16") == 0)
        return ZHybridDequantMode::gate_up_fp16;
    if (std::strcmp(mode, "scaled_all_fp16") == 0)
        return ZHybridDequantMode::scaled_all_fp16;
    throw std::runtime_error(
        "hybrid W8 dequant GEMM mode must be bf16, gate_up_fp16 or scaled_all_fp16");
}

std::string z_hybrid_dequant_precision() {
    switch (z_hybrid_dequant_mode()) {
        case ZHybridDequantMode::direct: return "";
        case ZHybridDequantMode::bf16: return "+on_demand_dequant_bf16_gemm";
        case ZHybridDequantMode::gate_up_fp16:
            return "+on_demand_gate_up_fp16_direct_w8_down";
        case ZHybridDequantMode::scaled_all_fp16:
            return "+on_demand_scaled_all_fp16_gemm";
    }
    throw std::runtime_error("invalid hybrid dequant mode");
}

ZQuantizedGeometry z_quantized_geometry(const Tensor &weight, const Tensor &scales,
                                        int logical_input) {
    require(weight.ndim() == 2 && scales.ndim() == 2 && logical_input > 0 &&
                scales.shape(1) > 0 && weight.shape(0) == scales.shape(0) &&
                logical_input % scales.shape(1) == 0 &&
                (weight.shape(1) * 32) % logical_input == 0,
            "invalid Z-Image GGUF affine geometry");
    const int group_size = logical_input / scales.shape(1);
    const int bits = weight.shape(1) * 32 / logical_input;
    require((group_size == 32 || (bits == 8 && (group_size == 64 || group_size == 128))) &&
                (bits == 4 || bits == 8),
            "Z-Image hybrid requires Q4 g32 or Q8 g32/g64/g128 affine weights");
    return {group_size, bits};
}

Tensor z_hybrid_output_range(const Tensor &x, const Weights &w,
                             const std::string &prefix, int start, int end) {
    // Both the complete shared-base route and the lossy suffix-only diagnostic
    // need runtime LoRA on every GPU suffix projection. Only the latter leaves
    // the Core ML prefix unchanged by the adapter.
    if (w.has_runtime_loras() && !w.convrot(prefix))
        return w.project_slice(x, prefix, start, end, 0, x.shape(-1), false);
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
    const auto mode = z_hybrid_dequant_mode();
    if (geometry.bits == 8 && (mode == ZHybridDequantMode::gate_up_fp16 ||
                               mode == ZHybridDequantMode::scaled_all_fp16)) {
        // Round gate/up back to BF16 before SwiGLU. The opt-in scaled-all
        // mode also handles the down GEMM separately below.
        auto dense = mx::dequantize(weight, scales, biases, geometry.group_size,
                                    8, "affine", std::nullopt, mx::float16);
        return mx::astype(mx::matmul(mx::astype(x, mx::float16), mx::transpose(dense)),
                          mx::bfloat16);
    }
    if (geometry.bits == 8 && mode == ZHybridDequantMode::bf16) {
        // Experimental wide-GEMM path: retain only packed W8 in Weights,
        // expand this shard for one BF16 matmul, then release the expression.
        // Never use this path as evidence of pre-dequantized W8 residence.
        auto dense = mx::dequantize(weight, scales, biases, geometry.group_size,
                                     8, "affine", std::nullopt, mx::bfloat16);
        return mx::matmul(x, mx::transpose(dense));
    }
    return mx::quantized_matmul(x, weight, scales, biases, true,
                                geometry.group_size, geometry.bits, "affine");
}

Tensor z_hybrid_input_range(const Tensor &x, const Weights &w,
                            const std::string &prefix, int full_input,
                            int start, int end) {
    if (w.has_runtime_loras() && !w.convrot(prefix))
        return w.project_slice(x, prefix, 0, w.at(prefix + ".weight").shape(0),
                               start, end, false);
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
    if (geometry.bits == 8 && z_hybrid_dequant_mode() == ZHybridDequantMode::bf16) {
        auto dense = mx::dequantize(weight, scales, biases, geometry.group_size,
                                     8, "affine", std::nullopt, mx::bfloat16);
        return mx::matmul(x, mx::transpose(dense));
    }
    if (geometry.bits == 8 &&
        z_hybrid_dequant_mode() == ZHybridDequantMode::scaled_all_fp16) {
        // Expression-local FP16 dequantization keeps the packed W8 matrices
        // resident. Scale the potentially huge SwiGLU hidden before the
        // FP16 down GEMM and restore its BF16 result afterward. This is a
        // research mode; whole-image quality and finite outputs are mandatory.
        constexpr float hidden_scale = 64.0f;
        auto dense = mx::dequantize(weight, scales, biases, geometry.group_size,
                                    8, "affine", std::nullopt, mx::float16);
        auto scaled = mx::astype(x / hidden_scale, mx::float16);
        return mx::astype(mx::matmul(scaled, mx::transpose(dense)),
                          mx::bfloat16) * hidden_scale;
    }
    return mx::quantized_matmul(x, weight, scales, biases, true,
                                geometry.group_size, geometry.bits, "affine");
}

Tensor z_hybrid_gpu_suffix(const Tensor &x, const Weights &w,
                           const std::string &prefix, int start, int end) {
    // Compact streamed weights already exclude the ANE prefix.
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
        auto projection = [](const Tensor &x, const Tensor &w) {
            const char *mode = std::getenv("TURBOCIDER_Z_MPP_PROJECTIONS");
            const bool qualified_default = !std::getenv("TURBOCIDER_Z_DISABLE_MPP_PROJECTIONS") &&
                z_image_small_shape_metal_default() && x.shape(1) <= 1056;
            if (mode || qualified_default) {
                if (!mode || std::string(mode) != "attention_out" ||
                    w.shape() == mx::Shape{3840,3840})
                    return z_metal::projection(x, w);
            }
            return mx::matmul(x, mx::transpose(w));
        };
        auto modulation = mx::expand_dims(
            mx::matmul(args[2], mx::transpose(args[3])) + args[4], 1);
        auto mod = mx::split(modulation, 4, -1);
        auto attention_input = z_modulate_norm(args[0], args[5], mod[0], args[0], false);
        const bool fused_qkv_projection =
            !std::getenv("TURBOCIDER_Z_DISABLE_MPP_QKV_PREPARE") &&
            (std::getenv("TURBOCIDER_Z_MPP_QKV_PREPARE") ||
             (z_image_small_shape_metal_default() && attention_input.shape(1) <= 1056));
        auto rotated = fused_qkv_projection
            ? z_metal::project_prepare_qkv(attention_input,args[6],args[7],args[8],args[1])
            : z_prepare_qkv(projection(attention_input,args[6]),args[7],args[8],args[1]);
        auto attention = projection(
            attend(rotated[0], rotated[1], rotated[2], false, {},
                   !std::getenv("TURBOCIDER_Z_DISABLE_FUSED_SDPA")),
            args[9]);
        auto residual_and_feed = [&] {
            const char *setting = std::getenv("TURBOCIDER_Z_GATE_NORM_VIRTUAL_THREADS");
            const bool qualified_default = !std::getenv("TURBOCIDER_Z_DISABLE_GATE_NORM") &&
                z_image_small_shape_metal_default() && attention.shape(1) <= 1056;
            if (setting || qualified_default) {
                const std::string threads(setting ? setting : "128");
                require(threads == "128" || threads == "256" || threads == "512",
                        "virtual gate norm threads must be 128, 256 or 512");
                require(!std::getenv("TURBOCIDER_Z_DISABLE_FUSED_MOD") &&
                        !std::getenv("TURBOCIDER_Z_FUSED_GATE_NORM"),
                        "virtual gate norm conflicts with disabled modulation or scalar gate fusion");
                if (attention.shape(1) <= 1056)
                    return z_metal::gate_norm_virtual(attention,args[0],args[10],mod[1],
                        args[11],mod[2],std::stoi(threads));
            }
            if (std::getenv("TURBOCIDER_Z_FUSED_GATE_NORM"))
                return z_metal::gate_norm(attention,args[0],args[10],mod[1],args[11],mod[2]);
            auto value = z_modulate_norm(attention, args[10], mod[1], args[0], true);
            return std::vector<Tensor>{value,
                z_modulate_norm(value, args[11], mod[2], value, false)};
        }();
        auto value = residual_and_feed[0], feed_input = residual_and_feed[1];
        auto activation = [&] {
            if (std::getenv("TURBOCIDER_Z_MPP_SWIGLU_DUAL"))
                return z_metal::swiglu_dual_gemm(feed_input, args[12], args[13]);
            auto up = projection(feed_input, args[13]);
            if (!std::getenv("TURBOCIDER_Z_DISABLE_MPP_SWIGLU") &&
                (std::getenv("TURBOCIDER_Z_MPP_SWIGLU") || z_image_mpp_swiglu_default()) &&
                feed_input.dtype() == mx::bfloat16 && args[12].dtype() == mx::bfloat16 &&
                up.dtype() == mx::bfloat16)
                return z_metal::swiglu_gemm(feed_input, args[12], up);
            auto gate = projection(feed_input, args[12]);
            return (gate * mx::sigmoid(gate)) * up;
        }();
        auto feed = projection(activation, args[14]);
        return std::vector<Tensor>{
            z_modulate_norm(feed, args[15], mod[3], value, true)};
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

std::function<std::vector<Tensor>(const std::vector<Tensor> &)> &z_hybrid_pre_graph(
        bool fused_qkv, bool fused_gate_norm) {
    // Core ML is a graph boundary. Compile the GPU work up to that boundary
    // together, retaining the BF16 input for the suffix and FP16 for Core ML.
    // Every weight is an argument, so cached graphs cannot retain another layer
    // or a previous stream slot's weights.
    auto make = [](bool use_fused_qkv, bool use_fused_gate_norm) {
        return mx::compile([use_fused_qkv, use_fused_gate_norm](const std::vector<Tensor> &a) {
        require(a.size() == 12, "invalid Z-Image hybrid pre-MLP inputs");
        auto fast_rms = [](const Tensor &x, const Tensor &weight) {
            return mx::astype(mx::fast::rms_norm(mx::astype(x, mx::float32),
                                                mx::astype(weight, mx::float32), 1e-5f),
                              x.dtype());
        };
        auto mod = mx::split(mx::expand_dims(mx::matmul(a[2], mx::transpose(a[3])) + a[4], 1), 4, -1);
        auto input = fast_rms(a[0], a[5]) * (Tensor(1.f, mod[0].dtype()) + mod[0]);
        auto rotated = [&] {
            if (use_fused_qkv)
                return z_metal::project_prepare_qkv(input, a[6], a[7], a[8], a[1]);
            auto qkv = mx::matmul(input, mx::transpose(a[6]));
            return z_prepare_qkv(qkv, a[7], a[8], a[1]);
        }();
        auto attention = mx::matmul(
            attend(rotated[0], rotated[1], rotated[2], false, {},
                   !std::getenv("TURBOCIDER_Z_DISABLE_FUSED_SDPA")), mx::transpose(a[9]));
        auto value_and_feed = [&] {
            if (use_fused_gate_norm)
                return z_metal::gate_norm_virtual(attention, a[0], a[10], mod[1],
                                                  a[11], mod[2], 128);
            auto value = a[0] + mx::tanh(mod[1]) * fast_rms(attention, a[10]);
            auto feed = fast_rms(value, a[11]) * (Tensor(1.f, mod[2].dtype()) + mod[2]);
            return std::vector<Tensor>{value, feed};
        }();
        auto value = value_and_feed[0], feed = value_and_feed[1];
        return std::vector<Tensor>{value, feed, mx::tanh(mod[3]), mx::astype(feed, mx::float16)};
        });
    };
    // Construct only the selected compiled graph. MLX's compiler cache can
    // finalize before a DSO-local mx::compile closure at process exit; keep
    // the bounded, process-lifetime graph objects alive until teardown.
    if (fused_gate_norm) {
        if (fused_qkv) {
            static auto *qkv_gate_graph = new decltype(make(true, true))(make(true, true));
            return *qkv_gate_graph;
        }
        static auto *gate_graph = new decltype(make(false, true))(make(false, true));
        return *gate_graph;
    }
    if (fused_qkv) {
        static auto *qkv_graph = new decltype(make(true, false))(make(true, false));
        return *qkv_graph;
    }
    static auto *graph = new decltype(make(false, false))(make(false, false));
    return *graph;
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
    // The image-only, short-row M4 Max profile saves a GPU pre-FFN launch by
    // using the same QKV+prepare kernel as the GPU-only path. Keep the
    // explicit probe and opt-out for matched comparisons and other machines.
    const bool qualified_fused_qkv = z_image_small_shape_metal_default() &&
        x.shape(1) <= 1056 && !std::getenv("TURBOCIDER_Z_HYBRID_DISABLE_FUSED_QKV");
    const bool fused_qkv = hybrid->image_only_token_rows == 1024 &&
        (qualified_fused_qkv || std::getenv("TURBOCIDER_Z_HYBRID_FUSED_QKV"));
    // The qualified short-row M4 Max hybrid also benefits from the existing
    // two-reduction GPU kernel; longer captions/other devices stay unchanged.
    const bool qualified_fused_gate_norm = z_image_small_shape_metal_default() &&
        x.shape(1) <= 1056 && !std::getenv("TURBOCIDER_Z_HYBRID_DISABLE_FUSED_GATE_NORM");
    const bool fused_gate_norm = hybrid->image_only_token_rows == 1024 &&
        (qualified_fused_gate_norm || std::getenv("TURBOCIDER_Z_HYBRID_FUSED_GATE_NORM"));
    auto pre = z_hybrid_pre_graph(fused_qkv, fused_gate_norm)({x, freqs, temb,
        w.at(prefix + ".adaLN_modulation.0.weight"), w.at(prefix + ".adaLN_modulation.0.bias"),
        w.at(prefix + ".attention_norm1.weight"), w.at(prefix + ".attention.qkv.weight"),
        w.at(prefix + ".attention.q_norm.weight"), w.at(prefix + ".attention.k_norm.weight"),
        w.at(prefix + ".attention.out.weight"), w.at(prefix + ".attention_norm2.weight"),
        w.at(prefix + ".ffn_norm1.weight")});
    if (profile) profile.pre_done({pre[0], pre[2], pre[1]});
    const int actual_rows = x.shape(1);
    const int ane_rows = hybrid->image_only_token_rows
        ? hybrid->image_only_token_rows : actual_rows;
    require(ane_rows <= actual_rows,
            "image-only Z-Image FFN needs at least 1024 image rows");
    auto packed = pre[3];
    if (hybrid->image_only_token_rows)
        packed = slice_axis(packed, 1, 0, ane_rows);
    else if (actual_rows < hybrid->rows)
        packed = mx::concatenate(
            {packed, mx::zeros({1, hybrid->rows - actual_rows, 3840}, mx::float16)}, 1);
    mx::eval({pre[0], pre[1], pre[2], packed});
    profile.packed();
    const auto ffn = prefix + ".feed_forward";
    auto image_input = hybrid->image_only_token_rows
        ? slice_axis(pre[1], 1, 0, ane_rows) : pre[1];
    auto gpu = gpu_graph({image_input, w.at(ffn + ".w1.weight"),
                         w.at(ffn + ".w3.weight"), w.at(ffn + ".w2.weight")})[0];
    if (hybrid->image_only_token_rows && actual_rows > ane_rows) {
        auto caption = slice_axis(pre[1], 1, ane_rows, actual_rows);
        // The GPU complement has no ANE-prefix channels. Compute those
        // caption rows from the original, full resident BF16 FFN instead.
        auto full_caption = z_ffn(caption, w, ffn);
        gpu = mx::concatenate({gpu, full_caption}, 1);
    }
    mx::async_eval({gpu});
    profile.submitted(gpu);
    auto ane = slice_axis(hybrid->predict(block, packed), 1, 0, ane_rows);
    if (ane_rows < actual_rows)
        ane = mx::concatenate({ane,
            mx::zeros({1, actual_rows - ane_rows, hybrid->hidden}, ane.dtype())}, 1);
    profile.predicted(gpu);
    auto result = z_hybrid_post_graph()({gpu, ane, pre[0], pre[2],
        w.at(prefix + ".ffn_norm2.weight"), Tensor(z_hybrid_output_scale(hybrid), gpu.dtype())})[0];
    profile.finish(result);
    return result;
}

Tensor z_block(const Tensor &x, const Weights &w, const std::string &prefix,
               const Tensor &freqs, const Tensor &temb, HybridSession *hybrid,
               int hybrid_block, const ZImageGpuGraph *gpu_graph,
               bool compile_hybrid_segments, std::vector<Tensor> *keepalive, bool allow_dense_compile = true);

Tensor z_runtime_block(const Tensor &x, const Weights &w, const std::string &prefix,
                       const Tensor &freqs, const Tensor &temb, ane::HybridFfn &runtime,
                       int block, std::atomic<bool> &cancelled, bool gguf_compatibility) {
    const auto plan = runtime.plan_block(block, x.shape(1));
    // Compare identical complete block windows. A lazy preceding GPU block
    // must not inflate this block's sample. Stable hybrid blocks return owned
    // FFN output and can leave their final residual lazy, like the GPU route.
    if (plan.measured()) mx::eval(x);
    const auto start = plan.measured() ? Clock::now() : Clock::time_point{};
    auto complete = [&](Tensor output) {
        if (plan.measured()) {
            mx::eval(output);
            checkpoint(cancelled);
            runtime.observe_block(block, x.shape(1),
                std::chrono::duration<double>(Clock::now() - start).count());
        }
        return output;
    };
    if (!plan.split()) {
        // Use the existing full GPU implementation, including its BF16
        // compiled graph or GGUF/LoRA projections. No staging, split FFN
        // callback, or extra timing fence on an unmeasured disabled block.
        return complete(z_block(x, w, prefix, freqs, temb, nullptr, block,
                                nullptr, false, nullptr));
    }
    const auto ffn = prefix + ".feed_forward";
    if (gguf_compatibility || w.has_runtime_loras()) {
        // GGUF can mix packed affine and floating projections, including
        // modulation/attention. Keep the baseline's native GPU projections
        // and dtype promotion; never feed packed uint32 into a dense graph.
        auto source = [&](const std::string &name, int input_width) -> ane::FfnWeight {
            require(!w.convrot(name) && !w.nvfp4(name) && !w.has(name + ".bias"),
                    "runtime GGUF FFN requires bias-free dense or affine projections: " + name);
            const auto &weight = w.at(name + ".weight");
            if (!w.quantized(name)) return {weight, std::nullopt, std::nullopt};
            const auto &scales = w.at(name + ".scales");
            const auto geometry = z_quantized_geometry(weight, scales, input_width);
            return {weight, scales, w.has(name + ".biases")
                ? std::optional<Tensor>(w.at(name + ".biases")) : std::nullopt,
                geometry.group_size, geometry.bits};
        };
        runtime.stage_weights(block, x.shape(1),
            {source(ffn + ".w1", 3840), source(ffn + ".w3", 3840), source(ffn + ".w2", 10240)});
        auto modulation = mx::expand_dims(linear_compat(temb, w, prefix + ".adaLN_modulation.0"), 1);
        auto parts = mx::split(modulation, 4, -1);
        auto attention = z_attention(rms(x, w.at(prefix + ".attention_norm1.weight"), 1e-5f) *
            (Tensor(1.f, parts[0].dtype()) + parts[0]), w, prefix, freqs);
        auto value = x + mx::tanh(parts[1]) * rms(attention, w.at(prefix + ".attention_norm2.weight"), 1e-5f);
        auto feed_input = rms(value, w.at(prefix + ".ffn_norm1.weight"), 1e-5f) *
            (Tensor(1.f, parts[2].dtype()) + parts[2]);
        ane::HybridFfn::Adapter adapter{
            [&](const Tensor &input) {
                return std::make_pair(w.lora_delta_slice(input, ffn + ".w1", 0, 10240, 0, 3840),
                                      w.lora_delta_slice(input, ffn + ".w3", 0, 10240, 0, 3840));
            },
            [&](const Tensor &hidden, const Tensor &base) {
                auto delta = w.lora_delta_slice(hidden, ffn + ".w2", 0, 3840, 0, 10240);
                return mx::astype(mx::astype(base, mx::float32) + mx::astype(delta, mx::float32), base.dtype());
            }};
        auto feed = runtime.run(block, feed_input,
            [&](const Tensor &input) { return z_ffn(input, w, ffn); }, cancelled,
            w.has_runtime_loras() ? &adapter : nullptr);
        auto output = value + mx::tanh(parts[3]) * rms(feed, w.at(prefix + ".ffn_norm2.weight"), 1e-5f);
        return complete(output);
    }
    std::vector<Tensor> weights{w.at(ffn + ".w1.weight"), w.at(ffn + ".w3.weight"),
                                w.at(ffn + ".w2.weight")};
    runtime.stage(block, x.shape(1), weights);
    const bool small = z_image_small_shape_metal_default() && x.shape(1) <= 1056;
    auto pre = z_hybrid_pre_graph(small, small)({x, freqs, temb,
        w.at(prefix + ".adaLN_modulation.0.weight"), w.at(prefix + ".adaLN_modulation.0.bias"),
        w.at(prefix + ".attention_norm1.weight"), w.at(prefix + ".attention.qkv.weight"),
        w.at(prefix + ".attention.q_norm.weight"), w.at(prefix + ".attention.k_norm.weight"),
        w.at(prefix + ".attention.out.weight"), w.at(prefix + ".attention_norm2.weight"),
        w.at(prefix + ".ffn_norm1.weight")});
    // Use the same tuned short-row projections and fused SwiGLU as the base
    // GPU block. Weights are arguments, never captured from another layer.
    static auto *gpu = new ZImageGpuGraph(mx::compile([](const std::vector<Tensor> &a) {
        auto project = [](const Tensor &v, const Tensor &weight) {
            const char *mode = std::getenv("TURBOCIDER_Z_MPP_PROJECTIONS");
            const bool tuned = !std::getenv("TURBOCIDER_Z_DISABLE_MPP_PROJECTIONS") &&
                z_image_small_shape_metal_default() && v.shape(1) <= 1056;
            if ((mode && std::string(mode) != "attention_out") || (!mode && tuned))
                return z_metal::projection(v, weight);
            return mx::matmul(v, mx::transpose(weight));
        };
        auto hidden = [&] {
            if (std::getenv("TURBOCIDER_Z_MPP_SWIGLU_DUAL"))
                return z_metal::swiglu_dual_gemm(a[0], a[1], a[2]);
            auto up = project(a[0], a[2]);
            if (!std::getenv("TURBOCIDER_Z_DISABLE_MPP_SWIGLU") &&
                (std::getenv("TURBOCIDER_Z_MPP_SWIGLU") || z_image_mpp_swiglu_default()))
                return z_metal::swiglu_gemm(a[0], a[1], up);
            return silu(project(a[0], a[1])) * up;
        }();
        return std::vector<Tensor>{project(hidden, a[3])};
    }));
    auto feed = runtime.run(block, pre[1], [&](const Tensor &input) {
        return (*gpu)({input, weights[0], weights[1], weights[2]})[0];
    }, cancelled);
    static auto *post = new ZImageGpuGraph(mx::compile([](const std::vector<Tensor> &a) {
        auto normalized = mx::astype(mx::fast::rms_norm(mx::astype(a[0], mx::float32),
                                      mx::astype(a[3], mx::float32), 1e-5f), a[0].dtype());
        return std::vector<Tensor>{a[1] + a[2] * normalized};
    }));
    auto output = (*post)({feed, pre[0], pre[2], w.at(prefix + ".ffn_norm2.weight")})[0];
    return complete(output);
}

Tensor z_context_block(const Tensor &x, const Weights &w, const std::string &prefix,
                       const Tensor &freqs) {
    auto attention = z_attention(rms(x, w.at(prefix + ".attention_norm1.weight"), 1e-5f),
                                 w, prefix, freqs);
    auto value = x + rms(attention, w.at(prefix + ".attention_norm2.weight"), 1e-5f);
    auto feed_input = rms(value, w.at(prefix + ".ffn_norm1.weight"), 1e-5f);
    auto feed = z_ffn(feed_input, w,
                      prefix + ".feed_forward");
    return value + rms(feed, w.at(prefix + ".ffn_norm2.weight"), 1e-5f);
}

Tensor z_block(const Tensor &x, const Weights &w, const std::string &prefix,
               const Tensor &freqs, const Tensor &temb, HybridSession *hybrid,
               int hybrid_block,
                const std::function<std::vector<Tensor>(const std::vector<Tensor> &)> *gpu_graph,
                bool compile_hybrid_segments = false, std::vector<Tensor> *keepalive = nullptr,
                bool allow_dense_compile) {
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
    // The tuned whole-block Metal graph is qualified for homogeneous BF16.
    // GGUF refiners may have BF16 matrices but FP32 norms/modulation biases;
    // those promote activations and do not satisfy fused norm/MPP contracts.
    // Preserve their native arithmetic via linear_compat, without recasting
    // the checkpoint or silently changing the GPU comparison's precision.
    const bool bf16_graph = fully_dense && x.dtype() == mx::bfloat16 &&
        temb.dtype() == mx::bfloat16 && [&] {
            for (const auto *suffix : {".adaLN_modulation.0.weight", ".adaLN_modulation.0.bias",
                 ".attention_norm1.weight", ".attention.qkv.weight", ".attention.q_norm.weight",
                 ".attention.k_norm.weight", ".attention.out.weight", ".attention_norm2.weight",
                 ".ffn_norm1.weight", ".feed_forward.w1.weight", ".feed_forward.w3.weight",
                 ".feed_forward.w2.weight", ".ffn_norm2.weight"})
                if (!w.has(prefix + suffix) || w.at(prefix + suffix).dtype() != mx::bfloat16)
                    return false;
            return true;
        }();
    if (profile.split_gpu() || profile.detail_gpu())
        require(fully_dense && !w.has_runtime_loras(),
                "Z-Image detailed GPU profiling requires dense BF16 weights without runtime LoRA");
    // A8 normally uses the eager compatibility path. Opt in only for the
    // routed, resident BF16-GPU experiment so its compiled pre/post graphs
    // can be qualified separately without changing the W8 GPU route.
    const bool experimental_compiled_a8 = hybrid && hybrid->activation_precision == "int8" &&
        (hybrid->image_only_token_rows == 1024 ||
         (hybrid->has_channel_route &&
          std::getenv("TURBOCIDER_Z_HYBRID_GPU_W8_DISABLE") &&
          std::getenv("TURBOCIDER_Z_W8A8_COMPILED_HYBRID")));
    if (compile_hybrid_segments && hybrid &&
        (hybrid->activation_precision != "int8" || experimental_compiled_a8) &&
        gpu_graph && bf16_graph &&
        !w.has_runtime_loras() && w.has(prefix + ".adaLN_modulation.0.bias") &&
        !std::getenv("TURBOCIDER_Z_FFN_CAPTURE_DIR") &&
        !std::getenv("TURBOCIDER_Z_HYBRID_EAGER_SEGMENTS") &&
        !std::getenv("TURBOCIDER_Z_EAGER_BLOCKS") &&
        !std::getenv("TURBOCIDER_DISABLE_FUSED_RMSNORM") &&
        !std::getenv("TURBOCIDER_Z_CONVROT_DEBUG") &&
        !std::getenv("TURBOCIDER_Z_HYBRID_VALIDATE"))
        return z_compiled_hybrid_block(x, w, prefix, freqs, temb, hybrid, hybrid_block,
                                      *gpu_graph, profile);
    if (allow_dense_compile && !hybrid && !std::getenv("TURBOCIDER_Z_EAGER_BLOCKS") && bf16_graph &&
        !std::getenv("TURBOCIDER_Z_FFN_CAPTURE_DIR") &&
        !w.has_runtime_loras() && !profile.split_gpu() && !profile.detail_gpu()) {
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
    if (profile.detail_gpu()) {
        profile.detail("modulation", {scale_msa, gate_msa, scale_mlp, gate_mlp});
        auto attention_input = z_modulate_norm(
            x, w.at(prefix + ".attention_norm1.weight"), parts[0], x, false);
        profile.detail("attention_norm_mod", {attention_input});
        auto qkv = mx::matmul(attention_input,
                              mx::transpose(w.at(prefix + ".attention.qkv.weight")));
        profile.detail("qkv_projection", {qkv});
        auto rotated = z_prepare_qkv(qkv,
            w.at(prefix + ".attention.q_norm.weight"),
            w.at(prefix + ".attention.k_norm.weight"), freqs);
        profile.detail("qkv_norm_rope_layout", rotated);
        auto attended = attend(rotated[0], rotated[1], rotated[2], false, {},
                               !std::getenv("TURBOCIDER_Z_DISABLE_FUSED_SDPA"));
        profile.detail("sdpa", {attended});
        auto attention = mx::matmul(
            attended, mx::transpose(w.at(prefix + ".attention.out.weight")));
        profile.detail("attention_out_projection", {attention});
        auto value = z_modulate_norm(attention,
            w.at(prefix + ".attention_norm2.weight"), parts[1], x, true);
        auto feed_input = z_modulate_norm(value,
            w.at(prefix + ".ffn_norm1.weight"), parts[2], value, false);
        profile.detail("residual_feed_norm", {value, feed_input});
        auto up = mx::matmul(feed_input,
            mx::transpose(w.at(prefix + ".feed_forward.w3.weight")));
        profile.detail("ffn_up_projection", {up});
        auto activation = z_metal::swiglu_gemm(
            feed_input, w.at(prefix + ".feed_forward.w1.weight"), up);
        profile.detail("ffn_gate_swiglu", {activation});
        auto feed = mx::matmul(
            activation, mx::transpose(w.at(prefix + ".feed_forward.w2.weight")));
        profile.detail("ffn_down_projection", {feed});
        auto result = z_modulate_norm(feed,
            w.at(prefix + ".ffn_norm2.weight"), parts[3], value, true);
        profile.detail("final_gate_norm", {result});
        profile.finish(result);
        return result;
    }
    auto attention = z_attention(rms(x, w.at(prefix + ".attention_norm1.weight"), 1e-5f) *
                                     scale_msa,
                                 w, prefix, freqs);
    auto value = x + gate_msa * rms(attention, w.at(prefix + ".attention_norm2.weight"), 1e-5f);
    auto feed_input = rms(value, w.at(prefix + ".ffn_norm1.weight"), 1e-5f) * scale_mlp;
    z_capture_ffn_input(feed_input, hybrid_block);
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
        const int actual_rows = feed_input.shape(1);
        const int ane_rows = hybrid->image_only_token_rows
            ? hybrid->image_only_token_rows : actual_rows;
        require(ane_rows <= actual_rows,
                "image-only Z-Image FFN needs at least 1024 image rows");
        auto packed = mx::astype(
            hybrid->image_only_token_rows ? slice_axis(feed_input, 1, 0, ane_rows) : feed_input,
            mx::float16);
        if (!hybrid->image_only_token_rows && actual_rows < hybrid->rows)
            packed = mx::concatenate(
                {packed, mx::zeros({1, hybrid->rows - actual_rows, 3840}, mx::float16)}, 1);
        if (hybrid->mlp_output_kind == "fused_lora") packed = mx::contiguous(packed);
        if (keepalive) keepalive->insert(keepalive->end(), {value, gate_mlp, feed_input, packed});
        mx::eval({feed_input, packed});
        profile.packed();
        const auto ffn = prefix + ".feed_forward";
        const bool dense = !w.quantized(ffn + ".w1") && !w.convrot(ffn + ".w1") &&
                           !w.quantized(ffn + ".w2") && !w.convrot(ffn + ".w2") &&
                           !w.quantized(ffn + ".w3") && !w.convrot(ffn + ".w3");
        auto image_feed = hybrid->image_only_token_rows
            ? slice_axis(feed_input, 1, 0, ane_rows) : feed_input;
        // The compiled dense suffix contains only the frozen base matrices.
        // Runtime adapters must take the slice projection path on every FFN
        // channel, including the suffix of the deliberately lossy diagnostic.
        auto gpu = dense && !w.has_runtime_loras()
            ? (*gpu_graph)({image_feed, w.at(ffn + ".w1.weight"),
                            w.at(ffn + ".w3.weight"), w.at(ffn + ".w2.weight")})[0]
            : z_hybrid_gpu_suffix(image_feed, w, ffn, hybrid->ane_mlp_end,
                                  hybrid->mlp_width);
        if (hybrid->image_only_token_rows && actual_rows > ane_rows) {
            require(dense, "image-only GPU caption requires dense full FFN weights");
            gpu = mx::concatenate({gpu,
                z_ffn(slice_axis(feed_input, 1, ane_rows, actual_rows), w, ffn)}, 1);
        }
        if (keepalive) keepalive->push_back(gpu);
        if (hybrid->mlp_output_kind != "fused_lora") {
            mx::async_eval({gpu});
            profile.submitted(gpu);
        }
        Tensor ane = packed;
        if (hybrid->mlp_output_kind == "fused_lora") {
            const int width = hybrid->ane_mlp_end;
            const char *direct_fp16_flag = std::getenv("TURBOCIDER_Z_RUNTIME_LORA_DIRECT_FP16");
            const auto delta_dtype = direct_fp16_flag && std::string_view(direct_fp16_flag) == "1"
                ? std::optional<mx::Dtype>(mx::float16) : std::nullopt;
            auto deltas = w.has_runtime_loras()
                ? mx::astype(mx::concatenate({
                    w.lora_delta_slice(image_feed, ffn + ".w1", 0, width, 0, hybrid->hidden,
                                       delta_dtype),
                    w.lora_delta_slice(image_feed, ffn + ".w3", 0, width, 0, hybrid->hidden,
                                       delta_dtype)
                  }, -1), mx::float16)
                : mx::zeros({1, ane_rows, 2 * width}, mx::float16);
            if (ane_rows < hybrid->rows)
                deltas = mx::concatenate({deltas,
                    mx::zeros({1, hybrid->rows - ane_rows, 2 * width}, mx::float16)}, 1);
            deltas = mx::contiguous(deltas);
            // The Core ML nonlinearity depends on the GPU's runtime LoRA delta.
            // Start the independent suffix only after this GPU-stream barrier.
            mx::eval(deltas);
            mx::async_eval({gpu});
            profile.submitted(gpu);
            ane = hybrid->predict_with_lora(hybrid_block, packed, deltas);
        } else {
            ane = hybrid->predict(hybrid_block, packed);
        }
        ane = slice_axis(ane, 1, 0, ane_rows);
        if (ane_rows < actual_rows)
            ane = mx::concatenate({ane,
                mx::zeros({1, actual_rows - ane_rows, hybrid->hidden}, ane.dtype())}, 1);
        if (keepalive) keepalive->push_back(ane);
        profile.predicted(gpu);
        if (hybrid->mlp_output_kind == "fused_lora") {
            auto base_down = mx::astype(slice_axis(ane, -1, 0, hybrid->hidden), gpu.dtype()) *
                             Tensor(z_hybrid_output_scale(hybrid), gpu.dtype());
            auto hidden = mx::astype(slice_axis(ane, -1, hybrid->hidden,
                                                hybrid->output_channels), gpu.dtype());
            feed = gpu + base_down;
            if (w.has_runtime_loras())
                feed = feed + w.lora_delta_slice(hidden, ffn + ".w2", 0,
                                                  hybrid->hidden, 0, hybrid->ane_mlp_end);
        } else {
            feed = z_image::join_hybrid_ffn(gpu, ane,
                Tensor(z_hybrid_output_scale(hybrid), gpu.dtype()));
        }
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
                     bool compile_hybrid_segments, ZImageHybridStream *hybrid_stream = nullptr,
                     std::vector<Tensor> *context_cache = nullptr,
                     ane::HybridFfn *runtime = nullptr, bool runtime_gguf_compatibility = false,
                     ZImageGgufStream *gguf_stream = nullptr, bool serial_refiner_eval = false) {
    require(!gguf_stream || (!weight_stream && !exact_stream && !hybrid_stream && !runtime && !hybrid),
            "GGUF bounded execution conflicts with another transformer backend");
    require(!hybrid_stream || (!weight_stream && !exact_stream), "hybrid/exact stream conflict");
    require(!(weight_stream && exact_stream),
            "Z-Image legacy and exact streaming cannot run together");
    if (weight_stream) weight_stream->begin_pass();
    auto patch = z_patchify(latent, caption);
    auto image = linear_compat(patch.image, w, "x_embedder");
    const bool reuse_context = context_cache && !context_cache->empty();
    auto caption_emb = reuse_context ? context_cache->at(0) : linear_compat(
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
             mx::repeat(z_image::padding_token_row(w.at("x_pad_token"), image.shape(1)),
                        image.shape(0) - patch.image_length, 0)}, 0);
    if (!reuse_context && caption_emb.shape(0) > patch.caption_length)
        caption_emb = mx::concatenate(
            {slice_axis(caption_emb, 0, 0, patch.caption_length),
             mx::repeat(z_image::padding_token_row(w.at("cap_pad_token"), caption_emb.shape(1)),
                        caption_emb.shape(0) - patch.caption_length, 0)}, 0);
    // The ConvRot checkpoint stores unquantized tensors as FP32, while Comfy
    // executes the transformer with BF16 manual-cast semantics.  Pin this
    // boundary after embedding and padding so those storage dtypes cannot
    // promote every residual and modulation tensor to FP32.
    image = mx::astype(image, mx::bfloat16);
    if (!reuse_context) caption_emb = mx::astype(caption_emb, mx::bfloat16);
    auto temb = z_timestep((1.f - sigma) * 1000.f, w);
    auto image_freqs = z_rope(patch.image_ids);
    auto caption_freqs = z_rope(patch.caption_ids);
    image = mx::expand_dims(image, 0);
    if (!reuse_context) caption_emb = mx::expand_dims(caption_emb, 0);
    if (gguf_stream && gguf_stream->streams_refiners()) {
        require(!reuse_context, "GGUF refiner streaming does not reuse a partial context stage");
        gguf_stream->run_refiners(pass, image, caption_emb, image_freqs, caption_freqs, temb);
    } else for (int i = 0; i < 2; ++i) {
        checkpoint(cancelled);
        const bool bf16_fallback = hybrid && z_hybrid_bf16_block(i);
        auto block_input = bf16_fallback ? mx::astype(image, mx::bfloat16) : image;
        image = runtime ? z_runtime_block(block_input, w, "noise_refiner." + std::to_string(i),
                                          image_freqs, temb, *runtime, i, cancelled, runtime_gguf_compatibility)
            : hybrid_stream ? hybrid_stream->encode_noise(uint32_t(i), image, image_freqs, temb)
            : z_block(block_input, w, "noise_refiner." + std::to_string(i), image_freqs, temb,
                        bf16_fallback ? nullptr : hybrid,
                        i, gpu_graph, compile_hybrid_segments);
        if (serial_refiner_eval) { mx::eval(image);checkpoint(cancelled); }
        if (!reuse_context)
            caption_emb = z_context_block(caption_emb, w,
                                          "context_refiner." + std::to_string(i), caption_freqs);
        if (serial_refiner_eval && !reuse_context) { mx::eval(caption_emb);checkpoint(cancelled); }
    }
    if (context_cache && !reuse_context) context_cache->push_back(caption_emb);
    auto unified = mx::concatenate({image, caption_emb}, 1);
    auto unified_freqs = mx::concatenate({image_freqs, caption_freqs}, 0);
    if (hybrid_stream) {
        hybrid_stream->run_main(pass, unified, unified_freqs, temb);
    } else if (gguf_stream) {
        gguf_stream->run_pass(pass, unified, unified_freqs, temb);
    } else if (exact_stream) {
        exact_stream->run_pass(pass, pass, unified, unified_freqs, temb);
    } else {
        for (int i = 0; i < 30; ++i) {
            checkpoint(cancelled);
            event("z_image_denoise_block", i, 30);
            auto streamed = weight_stream ? weight_stream->acquire(i) : Weights{};
            const bool bf16_fallback = hybrid && z_hybrid_bf16_block(2 + i);
            auto block_input = bf16_fallback ? mx::astype(unified, mx::bfloat16) : unified;
            unified = runtime ? z_runtime_block(block_input, w, "layers." + std::to_string(i),
                                                 unified_freqs, temb, *runtime, 2 + i, cancelled, runtime_gguf_compatibility)
                : z_block(block_input, weight_stream ? streamed : w,
                              "layers." + std::to_string(i), unified_freqs, temb,
                              bf16_fallback ? nullptr : hybrid,
                              2 + i, gpu_graph, compile_hybrid_segments);
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
    std::vector<Job> jobs_;
    HybridSession *legacy_hybrid_ = nullptr;
    ZImageGpuGraph *gpu_graph_ = nullptr;
    bool compile_segments_ = false;
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
          cancelled_(cancelled), jobs_(slot_count) {
        require(slot_count_ >= 1 && slot_count_ <= 3,
                "Z-Image exact adapter requires one to three slots");
        for (uint32_t slot = 0; slot < jobs_.size(); ++slot) {
            jobs_[slot].owner = this;
            jobs_[slot].slot = slot;
        }
    }

    void bind_hybrid(HybridSession *hybrid, ZImageGpuGraph *graph, bool compile_segments) {
        legacy_hybrid_ = hybrid;
        gpu_graph_ = graph;
        compile_segments_ = compile_segments;
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
                "Z-Image exact adapter requires one compiled slot pool");
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
                hybrid_ ? &hybrid_->session : legacy_hybrid_, int(2 + block),
                hybrid_ ? &hybrid_->graph : gpu_graph_, compile_segments_, hybrid_ ? &hybrid_->pending : nullptr);
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
            *freqs_, *temb_, hybrid_ ? &hybrid_->session : legacy_hybrid_, int(2 + block),
            hybrid_ ? &hybrid_->graph : gpu_graph_, compile_segments_, hybrid_ ? &hybrid_->pending : nullptr);
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

namespace {
class GgufStageAdapter final : public streaming::ModelSlotAdapter {
    struct Job { GgufStageAdapter *owner = nullptr; const streaming::Group *group = nullptr; std::array<char,512> error{}; };
    streaming::GgufWeightPager &source_;
    std::atomic<bool> &cancel_;
    Event event_;
    std::vector<Job> jobs_;
    std::optional<Tensor> value_, freqs_, temb_;
    std::optional<Tensor> image_, caption_, caption_freqs_;
    bool refiners_ = false;
    Weights current_;
    uint32_t pass_ = 0, pool_ = 0;
    uint64_t sequence_ = 0;
  public:
    GgufStageAdapter(streaming::GgufWeightPager &source, uint32_t slots, Event event, std::atomic<bool> &cancel, bool refiners = false)
        : source_(source), cancel_(cancel), event_(std::move(event)), jobs_(slots), refiners_(refiners) {
        for (auto &job : jobs_) job.owner = this;
    }
    void bind_pass(uint32_t pass, const Tensor &value, const Tensor &freqs, const Tensor &temb) {
        require(!value_, "GGUF pass already bound");
        pass_ = pass; value_ = value; freqs_ = freqs; temb_ = temb;
    }
    Tensor result() const { require(value_.has_value(), "GGUF pass not bound"); return *value_; }
    void bind_refiners(uint32_t pass, const Tensor &image, const Tensor &caption, const Tensor &image_freqs,
                       const Tensor &caption_freqs, const Tensor &temb) {
        require(refiners_ && !image_ && !caption_, "GGUF refiners already bound");
        pass_ = pass; image_ = image; caption_ = caption; freqs_ = image_freqs; caption_freqs_ = caption_freqs; temb_ = temb;
    }
    std::pair<Tensor, Tensor> refined() const {
        require(image_ && caption_, "GGUF refiners not bound"); return {*image_, *caption_};
    }
    void unbind(bool safe = true) {
        if (safe) { current_.clear(); value_.reset(); freqs_.reset(); temb_.reset(); image_.reset(); caption_.reset(); caption_freqs_.reset(); }
        else event_ = {};
    }
    void create_pool(const streaming::PoolLayout &pool) override { pool_ = pool.id; source_.create_pool(pool); }
    streaming::FillJob make_fill_job(const streaming::Group &group, const tc_stream_slot_ticket_v1 &ticket) override {
        auto &job = jobs_.at(ticket.slot); job.group = &group; job.error[0] = 0;
        return {ticket, &job, [](void *raw, const tc_stream_slot_ticket_v1 *ticket,
                    const std::atomic<bool> *cancel, uint64_t *bytes) -> int {
            auto &job = *static_cast<Job *>(raw);
            try { *bytes = job.owner->source_.fill(*job.group, *ticket, cancel); return 0; }
            catch (const std::exception &error) { std::snprintf(job.error.data(), job.error.size(), "%s", error.what()); return -1; }
            catch (...) { std::snprintf(job.error.data(), job.error.size(), "%s", "unknown GGUF decode error"); return -1; }
        }};
    }
    std::string error() const { for (const auto &job : jobs_) if (job.error[0]) return job.error.data(); return {}; }
    void encode_prefix(uint32_t pass) override { require(pass == pass_, "GGUF prefix pass mismatch"); source_.check_unchanged(); }
    void prepare_group(const streaming::Group &group, const tc_stream_slot_ticket_v1 &ticket) override {
        require((value_ || image_) && ticket.item.pass == pass_ && ticket.item.step == pass_, "GGUF pass identity mismatch");
        current_ = source_.bind(group, ticket);
    }
    bool overlap_next_fill_after_claim() const noexcept override { return jobs_.size() > 1; }
    streaming::ReaderSet encode_group(const streaming::Group &group, const tc_stream_slot_ticket_v1 &ticket,
                                      streaming::CompletionMailbox &) override {
        require((value_ || image_) && ticket.item.pass == pass_, "GGUF encode pass mismatch");
        checkpoint(cancel_);
        const uint32_t block = group.blocks.front();
        if (refiners_) {
            const auto prefix = std::string(block % 2 ? "context_refiner." : "noise_refiner.") + std::to_string(block / 2);
            event_("z_image_gguf_refiner", int(block), 4);
            if (block % 2) {
                *caption_ = z_context_block(*caption_, current_, prefix, *caption_freqs_); mx::eval(*caption_);
            } else {
                *image_ = z_block(*image_, current_, prefix, *freqs_, *temb_, nullptr, int(block / 2), nullptr, false);
                mx::eval(*image_);
            }
        } else {
            event_("z_image_denoise_block", int(block), 30);
            *value_ = z_block(*value_, current_, "layers." + std::to_string(block), *freqs_, *temb_,
                              nullptr, int(2 + block), nullptr, false, nullptr, false);
            mx::eval(*value_); // Actual last reader completion, not a submission timestamp.
        }
        checkpoint(cancel_);
        require(sequence_ != UINT64_MAX, "GGUF reader sequence overflow");
        streaming::ReaderSet readers;
        readers.count = 1; readers.fences[0] = {1,++sequence_}; readers.already_complete = true;
        current_.clear(); return readers;
    }
    bool drain() noexcept override { try { mx::synchronize(); return true; } catch (...) { return false; } }
    void destroy_pool() noexcept override { current_.clear(); source_.destroy_pool(pool_); }
};
}

struct ZImageGgufStream::Impl {
    std::shared_ptr<const streaming::SourceLease> lease;
    z_image::GgufExecutionPlan plan;
    MemoryLedger ledger;
    std::unique_ptr<streaming::GgufWeightPager> source;
    std::shared_ptr<GgufStageAdapter> adapter;
    std::unique_ptr<streaming::StageExecutor> executor;
    std::unique_ptr<streaming::GgufWeightPager> refiner_source;
    std::shared_ptr<GgufStageAdapter> refiner_adapter;
    std::unique_ptr<streaming::StageExecutor> refiner_executor;
    std::atomic<bool> &cancel;
    uint32_t next = 0;
    bool finished = false;
    Impl(const std::filesystem::path &path, uint32_t p, uint32_t width, uint32_t height,
         uint32_t caption, uint32_t steps, uint64_t budget, Weights &fixed, Event event, std::atomic<bool> &cancelled,
         const std::string &profile, const std::string &residency)
        : ledger(budget), cancel(cancelled) {
        streaming::SourceFileIdentity file; file.logical_id = "transformer"; file.path = path;
        lease = streaming::SourceLease::capture_verified({std::move(file)}, &cancel);
        plan = z_image::describe_gguf_execution(lease, p, width, height, caption, steps, profile, residency);
        require(gguf::checked_add(gguf::checked_add(plan.packed_capacity_upper, plan.read_capacity_upper), plan.dense_capacity_upper) <= budget,
                "qe_budget_floor: packed source and dense slots exceed managed weight ceiling");
        const size_t main = plan.descriptor.stages.size() - 1;
        source = std::make_unique<streaming::GgufWeightPager>(lease, plan.descriptor, plan.descriptor.stages[main],
                                                           plan.layout.stages[main], ledger);
        event("load_gguf_packed_source", 0, 1);
        source->load_packed(&cancel); source->load_resident_aliases(fixed);
        event("load_gguf_packed_source", 1, 1);
        adapter = std::make_shared<GgufStageAdapter>(*source, p + 1, event, cancel);
        executor = std::make_unique<streaming::StageExecutor>(uint32_t(main), lease->generation(), adapter);
        executor->begin(plan.layout.stages[main]);
        if (main) {
            executor->release_drained_backing();
            refiner_source = std::make_unique<streaming::GgufWeightPager>(lease, plan.descriptor,
                plan.descriptor.stages.front(), plan.layout.stages.front(), ledger);
            refiner_source->load_packed(&cancel);
            refiner_adapter = std::make_shared<GgufStageAdapter>(*refiner_source, 1, event, cancel, true);
            refiner_executor = std::make_unique<streaming::StageExecutor>(0, lease->generation(), refiner_adapter);
            refiner_executor->begin(plan.layout.stages.front()); refiner_executor->release_drained_backing();
        }
    }
};
ZImageGgufStream::ZImageGgufStream(const std::filesystem::path &path, uint32_t p, uint32_t width,
        uint32_t height, uint32_t caption, uint32_t steps, uint64_t budget, Weights &fixed,
        const Event &event, std::atomic<bool> &cancel, const std::string &profile, const std::string &residency)
    : impl_(std::make_unique<Impl>(path,p,width,height,caption,steps,budget,fixed,event,cancel,profile,residency)) {}
ZImageGgufStream::~ZImageGgufStream() { if (impl_ && !drain_safely()) (void)impl_.release(); }
bool ZImageGgufStream::drain_safely() noexcept {
    if (!impl_) return true;
    const bool safe = !impl_->executor || impl_->executor->retry_drain();
    const bool refiner_safe = !impl_->refiner_executor || impl_->refiner_executor->retry_drain();
    if (impl_->adapter) impl_->adapter->unbind(safe);
    if (impl_->refiner_adapter) impl_->refiner_adapter->unbind(refiner_safe);
    return safe && refiner_safe;
}
bool ZImageGgufStream::streams_refiners() const noexcept { return impl_ && bool(impl_->refiner_executor); }
void ZImageGgufStream::run_refiners(uint32_t pass, Tensor &image, Tensor &caption,
        const Tensor &image_freqs, const Tensor &caption_freqs, const Tensor &temb) {
    require(streams_refiners() && pass == impl_->next, "GGUF refiner pass out of order");
    impl_->refiner_source->check_unchanged();
    impl_->refiner_adapter->bind_refiners(pass, image, caption, image_freqs, caption_freqs, temb);
    try {
        impl_->refiner_executor->run_pass(pass, pass, impl_->cancel);
        auto results = impl_->refiner_adapter->refined(); image = results.first; caption = results.second;
        impl_->refiner_adapter->unbind(); impl_->refiner_executor->release_drained_backing();
    } catch (...) { drain_safely(); throw; }
}
void ZImageGgufStream::run_pass(uint32_t pass, Tensor &value, const Tensor &freqs, const Tensor &temb) {
    require(impl_ && !impl_->finished && pass == impl_->next, "GGUF pass out of order");
    impl_->source->check_unchanged(); impl_->adapter->bind_pass(pass,value,freqs,temb);
    try {
        impl_->executor->run_pass(pass,pass,impl_->cancel);
        value = impl_->adapter->result(); impl_->adapter->unbind();
        impl_->source->check_unchanged();
        if (streams_refiners()) impl_->executor->release_drained_backing();
        ++impl_->next;
    } catch (const std::exception &error) {
        drain_safely(); checkpoint(impl_->cancel);
        const auto detail = impl_->adapter->error();
        if (!detail.empty()) throw std::runtime_error(std::string(error.what()) + "; " + detail);
        throw;
    }
}
void ZImageGgufStream::finish() {
    require(impl_ && !impl_->finished && impl_->next == impl_->plan.layout.stages.back().pass_count,
            "GGUF execution incomplete");
    impl_->executor->finish();
    if (streams_refiners()) impl_->refiner_executor->finish();
    mx::synchronize(); impl_->source->check_unchanged(); impl_->finished = true;
}
QuantizedExecutionMetrics ZImageGgufStream::metrics() const {
    const auto source = impl_->source->metrics(); const auto execution = impl_->executor->counters();
    QuantizedExecutionMetrics result;
    result.source_sha256 = impl_->lease->file("transformer").content_digest;
    result.layout_digest = impl_->plan.layout.digest;
    result.packed_bytes = source.packed_source_bytes; result.packed_capacity_bytes = source.packed_capacity_bytes;
    result.source_float_bytes = source.source_float_bytes; result.dense_capacity_bytes = source.maximum_dense_pool_capacity_bytes;
    result.managed_peak_bytes = impl_->ledger.snapshot().peak_committed_bytes;
    result.fills = source.fill_count; result.decoded_bytes = source.decoded_bytes;
    result.source_load_seconds = source.packed_read_seconds; result.decode_seconds = source.decode_seconds;
    result.exposed_wait_seconds = execution.wait_seconds;
    result.slots = impl_->plan.layout.stages.back().slot_count; result.prefetch = result.slots - 1;
    result.source_residency = impl_->plan.descriptor.workload.at("source_residency");
    result.source_logical_bytes = source.source_logical_bytes; result.read_buffer_bytes = source.read_buffer_capacity_bytes;
    result.source_read_bytes = source.source_read_bytes; result.streamed_read_seconds = source.streamed_read_seconds;
    if (streams_refiners()) {
        const auto refiners = impl_->refiner_source->metrics();
        const auto ref_execution = impl_->refiner_executor->counters();
        result.packed_bytes += refiners.packed_source_bytes; result.packed_capacity_bytes += refiners.packed_capacity_bytes;
        result.source_float_bytes += refiners.source_float_bytes;
        result.source_logical_bytes += refiners.source_logical_bytes; result.read_buffer_bytes += refiners.read_buffer_capacity_bytes;
        result.source_read_bytes += refiners.source_read_bytes; result.streamed_read_seconds += refiners.streamed_read_seconds;
        result.decode_seconds += refiners.decode_seconds; result.exposed_wait_seconds += ref_execution.wait_seconds;
        result.refiner_fills = refiners.fill_count; result.refiner_slots = 1;
        result.refiner_decoded_bytes = refiners.decoded_bytes;
        result.refiner_capacity_bytes = refiners.maximum_dense_pool_capacity_bytes;
        result.dense_capacity_bytes = std::max(result.dense_capacity_bytes, result.refiner_capacity_bytes);
        require(result.refiner_fills == uint64_t(impl_->next) * 4 && ref_execution.groups_submitted == result.refiner_fills,
                "GGUF refiner fills/readers do not match actual passes");
    }
    require(result.fills == uint64_t(impl_->next) * 30 && execution.groups_submitted == result.fills,
            "GGUF fill/compute counts do not match actual passes");
    return result;
}

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
          source(plan.metadata(), plan.layout().stages.front().prefix,
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
          source(plan.metadata(), plan.layout().stages.front().prefix,
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

void ZImageExactStream::bind_hybrid(HybridSession *hybrid, ZImageGpuGraph *graph,
                                   bool compile_segments) {
    require(impl_ && !impl_->finished, "Z-Image exact executor is unavailable");
    impl_->adapter->bind_hybrid(hybrid, graph, compile_segments);
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
    if (const char *import = std::getenv("TURBOCIDER_Z_GGUF_IMPORT")) {
        require(std::string_view(import)=="mlx" || std::string_view(import)=="cpu_direct",
                "qe_config_conflict: unknown TURBOCIDER_Z_GGUF_IMPORT");
        gguf_direct_import_ = std::string_view(import)=="cpu_direct";
        if (gguf_direct_import_) {
#ifndef TURBOCIDER_ENABLE_QUANTIZED_EXECUTION_EXPERIMENTS
            throw std::invalid_argument("qe_capability_unqualified: direct packed import requires experimental build");
#endif
            require(!transformer_checkpoint.empty(),"qe_config_conflict: direct packed import requires a GGUF transformer");
        }
    }
    const bool gguf_encoder = z_qwen3_gguf_path() != nullptr;
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
        transformer_path_ = std::filesystem::absolute(transformer_checkpoint).lexically_normal();
        transformer_checkpoint_ = transformer_path_;
        gguf_transformer_ = true;
        nvfp4_transformer_ = false;
        if ((gguf_encoder || std::filesystem::is_regular_file(comfy_text)) &&
            std::filesystem::is_regular_file(comfy_vae)) {
            text_path_ = std::move(comfy_text);
            vae_path_ = std::move(comfy_vae);
            return;
        }
        text_path_ = root / "text_encoder";
        vae_path_ = root / "vae";
        require(gguf_encoder || has_safetensors(text_path_),
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
        require(gguf_encoder || std::filesystem::is_regular_file(text_path_) || has_safetensors(text_path_),
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
    require(gguf_encoder || has_safetensors(text_path_),
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
    streaming::SourceFileIdentity vae_file;
    vae_file.logical_id = "vae";
    vae_file.path = vae_path_;
    streaming::SourceFileIdentity tokenizer_file;
    tokenizer_file.logical_id = "tokenizer";
    tokenizer_file.path = root_ / "tokenizer/tokenizer.json";
    std::vector<streaming::SourceFileIdentity> files{
        std::move(transformer_file), std::move(vae_file), std::move(tokenizer_file)};
    auto add_text = [&](const std::filesystem::path &path, std::string logical_id) {
        streaming::SourceFileIdentity file;
        file.logical_id = std::move(logical_id);
        file.path = path;
        files.push_back(std::move(file));
    };
    if (!std::filesystem::is_directory(text_path_)) {
        add_text(text_path_, "text_encoder");
        return files;
    }
    // Match the shared component loader: real shards take precedence over
    // convenience links. Retain the named binding paths so retargeting a
    // directory symlink invalidates the lease instead of silently reopening it.
    std::vector<std::filesystem::path> shards, links;
    for (const auto &entry : std::filesystem::directory_iterator(text_path_)) {
        if (entry.path().extension() != ".safetensors") continue;
        if (entry.is_symlink()) links.push_back(entry.path());
        else if (entry.is_regular_file()) shards.push_back(entry.path());
    }
    if (shards.empty()) {
        require(links.size() == 1,
                "streaming_source_identity: shared text encoder requires regular shards or one checkpoint link");
        shards = std::move(links);
    }
    std::sort(shards.begin(), shards.end());
    for (const auto &path : shards)
        add_text(path, "text_encoder/" + path.filename().string());
    const auto index = text_path_ / "model.safetensors.index.json";
    if (std::filesystem::exists(index))
        add_text(index, "text_encoder/model.safetensors.index.json");
    return files;
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
                ((std::filesystem::is_regular_file(text_path_) &&
                  text_path_.extension() == ".safetensors") || has_safetensors(text_path_)) &&
                std::filesystem::is_regular_file(vae_path_) &&
                vae_path_.extension() == ".safetensors",
            "streaming_route_unsupported: Z-Image public card requires single-file transformer/VAE and valid text weights");

    auto files = streaming_source_files();
    auto lease = streaming::SourceLease::capture_for_query(
        std::move(files), streaming_content_identity_);
    auto tokenizer_fd = lease->duplicate_fd("tokenizer");
    Tokenizer tokenizer(tokenizer_fd.get(), lease->file("tokenizer").bytes);
    const auto tokens = tokenizer.z_image_prompt(request.prompt, request.dynamic_text);
    lease->revalidate_open_files();
    lease->revalidate_paths();
    // The encoder processes the padded token IDs, but encode_text removes
    // those padding rows before z_patchify aligns the valid caption to 32.
    const uint32_t encoder_rows = static_cast<uint32_t>(tokens.ids.size());

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
    workload.feature_digest = z_image_public_feature_digest(request);
    workload.token_shapes.push_back({
        "qwen3", "qwen3-z-image-v1", "z-image-template-v1",
        static_cast<uint32_t>(tokens.valid), encoder_rows, encoder_rows});

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
                streaming::streaming_preset_workload_matches(record, value_probe->workload_identity()) &&
                record.runtime == value_probe->runtime_identity(),
            "streaming_record_identity_mismatch");
    const auto &workload = value_probe->workload_identity();
    require(workload.token_shapes.size() == 1,
            "streaming_workload_invalid: Z-Image token shape count");
    z_image::StreamingWorkload descriptor_workload{
        workload.width, workload.height,
        padded_z_image_rows(workload.token_shapes.front().valid_rows),
        workload.steps};
    auto plan = std::make_shared<z_image::StreamingPlanView>(
        value_probe->lease_ptr(), record.plan.canonical_config,
        descriptor_workload);
    // The public BF16 identity must not accept a renamed quantized checkpoint.
    // INT8 remains on the private/manual candidate route until separately qualified.
    require(!plan->metadata().convrot(),
            "streaming_route_unsupported: Z-Image public card requires BF16 tensor metadata");
    std::string capacity_digest;
    if (record.text_capacity) {
        auto capacity_workload = descriptor_workload;
        capacity_workload.caption_rows = padded_z_image_rows(record.text_capacity->maximum_rows);
        const z_image::StreamingPlanView capacity_plan(
            value_probe->lease_ptr(), record.plan.canonical_config, capacity_workload);
        capacity_digest = capacity_plan.layout().digest;
        require(capacity_digest == record.plan.layout_digest,
                "streaming_capacity_layout_digest_mismatch");
    } else {
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
    if (!record.plan.layout_digest.empty())
#endif
        require(plan->layout().digest == record.plan.layout_digest,
                "streaming_layout_digest_mismatch");
    }
    return std::make_shared<streaming::ValueModelStreamingSnapshot>(
        streaming::ValueModelStreamingSnapshot::Values{
            model_id_, value_probe->source_identity(),
            value_probe->runtime_identity(), plan->descriptor(),
            plan->layout(), std::string(value_probe->component_policy_revision()),
            value_probe->lease_ptr(), workload, std::move(capacity_digest)});
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
                streaming::streaming_snapshot_matches_record(
                    execution->selection.record, *execution->model_snapshot) &&
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
    if (runtime_ffn_) runtime_ffn_->drain();
    cached_lora_identity_ = std::move(identity);
    active_loras_ = std::move(normalized);
    active_lora_strategy_ = strategy;
    // This graph contains only checkpoint weights. Keep the loaded Core ML
    // session across base/adapter switches when the explicit manifest agrees;
    // only MLX transformer weights and the adapter-dependent suffix are reset.
    if (!hybrid_ || hybrid_->mlp_output_kind != "fused_lora" ||
        hybrid_->manifest != request.ane_manifest ||
        request.hybrid_mlp_mode != "lora_fused")
        hybrid_.reset();
    encoder_hybrid_.reset();
    hybrid_gpu_graph_ = {};
    hybrid_gpu_mlp_start_ = -1;
    gpu_w8_suffix_start_ = -1;
    gpu_w8_group_size_ = 0;
    gpu_w8_manifest_.clear();
    gpu_bf16_route_manifest_.clear();
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
        if (gguf_transformer_ && gguf_direct_import_) {
            streaming::SourceFileIdentity file; file.logical_id="transformer";file.path=transformer_path_;
            auto lease=streaming::SourceLease::capture_verified({std::move(file)},&cancelled);
            gguf_packed_ledger_=std::make_unique<MemoryLedger>(z_qwen3_gguf_integer(
                "TURBOCIDER_Z_GGUF_PACKED_WEIGHT_LIMIT_BYTES",std::min<uint64_t>(10ull<<30,device_info().physical_memory/2),
                device_info().physical_memory));
            gguf_packed_bank_=std::make_unique<streaming::GgufPackedBank>(std::move(lease),"transformer",*gguf_packed_ledger_);
            z_image::validate_gguf_model_directory(gguf_packed_bank_->directory());
            gguf_packed_bank_->load(transformer_,&cancelled,event);
        } else if (gguf_transformer_)
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
    if (encoder_gguf_ && !encoder_gguf_->drain_safely()) {
        streaming_quarantined_ = true;
        throw std::runtime_error("Qwen3 GGUF drain unproven; restart the process");
    }
    encoder_gguf_.reset(); cached_encoder_gguf_identity_.clear(); cached_encoder_gguf_metrics_.reset();
    if (gguf_stream_ && !gguf_stream_->drain_safely()) {
        streaming_quarantined_ = true;
        throw std::runtime_error("GGUF drain unproven; restart the process");
    }
    gguf_stream_.reset();
    runtime_ffn_.reset(); runtime_manifest_.clear();
    exact_stream_.reset();
    weight_stream_.reset();
    stream_configuration_.clear();
    hybrid_.reset();
    encoder_hybrid_.reset();
    hybrid_gpu_graph_ = {};
    hybrid_gpu_mlp_start_ = -1;
    gpu_w8_suffix_start_ = -1;
    gpu_w8_group_size_ = 0;
    gpu_w8_manifest_.clear();
    gpu_bf16_route_manifest_.clear();
    cached_conditioning_.reset();
    cached_encoder_hybrid_metrics_.reset();
    cached_prompt_.clear();
    cached_encoder_manifest_.clear();
    text_encoder_.clear();
    transformer_.clear();
    gguf_packed_bank_.reset();gguf_packed_ledger_.reset();
    vae_.clear();
    public_component_cache_ = false;
    mx::clear_cache();
}

Tensor ZImage::encode_text(const Tokens &tokens, const Event &event, std::atomic<bool> &cancelled) try {
    if (encoder_gguf_) {
        auto result = encoder_gguf_->encode(tokens);
        cached_encoder_gguf_metrics_ = encoder_gguf_->metrics();
        require(encoder_gguf_->drain_safely(), "Qwen3 GGUF reader drain unproven");
        encoder_gguf_.reset();
        result = slice_axis(mx::squeeze(result, 0), 0, 0, tokens.valid);
        mx::eval(result); mx::clear_cache(); return result;
    }
    if (text_encoder_.bytes() == 0) {
        if (public_stream_lease_) {
            std::vector<std::string> artifacts;
            for (const auto &file : public_stream_lease_->descriptor().files)
                if (file.logical_id == "text_encoder" ||
                    (file.logical_id.starts_with("text_encoder/") &&
                     file.logical_id.ends_with(".safetensors")))
                    artifacts.push_back(file.logical_id);
            text_encoder_.load_lease(public_stream_lease_, artifacts, event, cancelled);
        } else
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
    if (encoder_gguf_) {
        if (!encoder_gguf_->drain_safely()) streaming_quarantined_ = true;
        else encoder_gguf_.reset();
    }
    // An interrupted prompt must not retain Qwen3 alongside the next denoiser.
    if (optimizations_.z_image_memory_lifecycle) {
        text_encoder_.clear();
        mx::clear_cache();
    }
    throw;
}

bool ZImage::conditioning(const Request &r, const Event &event, std::atomic<bool> &cancelled) {
    std::string encoder_identity;
    if (const char *path = z_qwen3_gguf_path()) {
        require(!public_stream_lease_ && !r.memory_constrained.enabled,
                "qe_envelope_unknown: Qwen3 GGUF whole-request/public qualification is not available");
        require(r.encoder_ane_manifest.empty(), "qe_config_conflict: Qwen3 GGUF encoder ANE is not supported");
        for (const auto &lora : active_loras_) require(lora.role != "text_encoder", "qe_config_conflict: Qwen3 GGUF encoder LoRA is not supported");
        const char *config = std::getenv("TURBOCIDER_Z_QWEN3_GGUF_CONFIG");
        require(config && *config, "qe_config_conflict: Qwen3 GGUF requires bound original config path");
        encoder_gguf_ = std::make_unique<components::Qwen3GgufEncoder>(path, config, root_ / "tokenizer/tokenizer.json",
            uint32_t(z_qwen3_gguf_integer("TURBOCIDER_QWEN3_GGUF_PREFETCH", 1, 2)),
            z_qwen3_gguf_integer("TURBOCIDER_QWEN3_GGUF_WEIGHT_LIMIT_BYTES", 8ull << 30, device_info().physical_memory), event, cancelled,
            gguf::DecodeOptions{z_qwen3_gguf_integer("TURBOCIDER_QWEN3_GGUF_SCALAR_DECODE", 0, 1) == 0},
            std::getenv("TURBOCIDER_QWEN3_GGUF_SOURCE_RESIDENCY") ? std::getenv("TURBOCIDER_QWEN3_GGUF_SOURCE_RESIDENCY") : "packed_resident");
        encoder_identity = encoder_gguf_->identity();
    }
    if (cached_conditioning_ && cached_prompt_ == r.prompt && cached_dynamic_ == r.dynamic_text &&
        cached_encoder_manifest_ == r.encoder_ane_manifest && cached_encoder_gguf_identity_ == encoder_identity) {
        encoder_gguf_.reset();
        event("z_image_text_cache_hit", 1, 1);
        return true;
    }
    auto tokens = encoder_gguf_ ? encoder_gguf_->tokenize(r.prompt, r.dynamic_text) :
        (public_stream_tokenizer_ ? *public_stream_tokenizer_ : tokenizer_).z_image_prompt(r.prompt, r.dynamic_text);
    cached_encoder_gguf_metrics_.reset();
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
    cached_encoder_gguf_identity_ = std::move(encoder_identity);
    return false;
}

std::string ZImage::select_acceleration(Request &r, int rows, const Event &event,
                                        std::atomic<bool> &cancelled) {
    if (r.hybrid_mlp_mode == "runtime") {
        require(!nvfp4_transformer_ && !convrot_transformer_ && !diffusers_layout_ &&
                    (active_loras_.empty() || (!gguf_transformer_ && active_lora_strategy_ == "inference_time")),
                "runtime-weight Z-Image requires BF16 Comfy or base native GGUF; LoRA requires unmerged BF16 base weights");
        hybrid_.reset(); hybrid_gpu_graph_ = {}; hybrid_gpu_mlp_start_ = -1;
        return "gpu_ane explicit runtime-weight token-row FFN; base-only weight slots with optional GPU LoRA activation corrections; physical placement unverified";
    }
    runtime_ffn_.reset(); runtime_manifest_.clear();
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
        const bool marked_image_only = !automatic &&
            z_image_image_only_manifest_rows(r.ane_manifest) == 1024;
        if (marked_image_only)
            require(r.allow_approximation,
                    "image-only Z-Image W8A8 requires allow_approximation=true");
        const bool image_only = !automatic &&
            (marked_image_only || std::getenv("TURBOCIDER_Z_W8A8_IMAGE_ONLY"));
        if (image_only)
            require(r.width == 512 && r.height == 512 &&
                        rows >= 1056 && rows <= 2048 && (rows - 1024) % 32 == 0,
                    "image-only W8A8 requires 512-square image rows and 32...1024 caption rows");
        require(std::filesystem::is_regular_file(transformer_checkpoint_),
                    "Z-Image hybrid requires a safetensors file or index");
        if (automatic && hybrid_ && hybrid_->rows != matched->bucket)
            hybrid_.reset();
        if (!hybrid_ || hybrid_->manifest != r.ane_manifest)
            hybrid_ = std::make_unique<HybridSession>(
                r.ane_manifest, root_, image_only ? 1024 : rows, event, cancelled,
                r.warmup_iterations,
                transformer_checkpoint_, r.hybrid_mlp_mode == "lora_suffix" ||
                                             r.hybrid_mlp_mode == "lora_fused"
                    ? std::vector<LoRAAsset>{} : active_loras_, matched ? matched->bucket : 0);
        hybrid_->set_tokens(image_only ? 1024 : rows);
        require(hybrid_->hidden == 3840 &&
                    hybrid_->block_count == 32 && hybrid_->mlp_width == 10240 &&
                    hybrid_->ane_mlp_start == 0 && hybrid_->ane_mlp_end < 10240,
                "Z-Image Core ML FFN partition geometry mismatch");
        if (r.hybrid_mlp_mode == "lora_suffix")
            require(hybrid_->checkpoint_sha_verified &&
                        hybrid_->activation_precision == "fp16" &&
                        hybrid_->ane_mlp_end == 4096 &&
                        hybrid_->image_only_token_rows == 0 &&
                        !hybrid_->lora_identity_verified,
                    "Z-Image runtime LoRA suffix requires a verified 4096-channel FP16 base graph");
        if (r.hybrid_mlp_mode == "lora_fused")
            require(hybrid_->checkpoint_sha_verified &&
                        hybrid_->mlp_output_kind == "fused_lora" &&
                        hybrid_->tensor_layout == "z_image" &&
                        hybrid_->activation_precision == "fp16" &&
                        (hybrid_->ane_mlp_end == 4096 || hybrid_->ane_mlp_end == 6144 ||
                         hybrid_->ane_mlp_end == 8192) && hybrid_->rows == 1056 &&
                        hybrid_->image_only_token_rows == 0 &&
                        !hybrid_->has_channel_route && !hybrid_->lora_identity_verified &&
                        active_loras_.size() == r.loras.size(),
                    "Z-Image runtime LoRA fused graph requires a base-only 1056-row artifact");
        if (hybrid_->mlp_output_kind == "fused_lora")
            require(r.hybrid_mlp_mode == "lora_fused",
                    "Z-Image runtime LoRA artifact requires explicit lora_fused selection");
        if (image_only)
            require(hybrid_->image_only_token_rows == 1024 &&
                        !hybrid_->has_channel_route &&
                        (marked_image_only ||
                         std::getenv("TURBOCIDER_Z_HYBRID_GPU_W8_DISABLE")),
                    "image-only W8A8 needs a marked 1024-row artifact and BF16 GPU complement");
        else
            require(hybrid_->image_only_token_rows == 0,
                    "image-only W8A8 artifact requires explicit opt-in");
        if (automatic)
            require(hybrid_->ane_mlp_end == 4096,
                    "M4 Max automatic Z-Image profile requires the validated 4096-channel ANE prefix");
        if (!hybrid_gpu_graph_ || hybrid_gpu_mlp_start_ != hybrid_->ane_mlp_end) {
            hybrid_gpu_graph_ = z_image::make_hybrid_gpu_graph(
                hybrid_->hidden, hybrid_->mlp_width, hybrid_->ane_mlp_end);
            hybrid_gpu_mlp_start_ = hybrid_->ane_mlp_end;
        }
        return automatic ? std::string("gpu_ane: measured case ") + matched->id
                         : r.hybrid_mlp_mode == "lora_suffix"
                            ? "gpu_ane: diagnostic base ANE FFN prefix with runtime LoRA GPU suffix only"
                            : r.hybrid_mlp_mode == "lora_fused"
                            ? "gpu_ane: experimental reusable base fused FFN with optional runtime pre-SiLU and down LoRA"
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
    const bool quantized = r.quantized_execution.active();
    if (gguf_direct_import_) {
        require(!quantized && !public_stream_lease_ && !r.memory_constrained.enabled &&
                r.residency=="resident" && r.execution=="gpu" && r.loras.empty() &&
                r.ane_manifest.empty() && r.encoder_ane_manifest.empty(),
                "qe_config_conflict: experimental direct packed import supports only private resident GPU without LoRA/ANE/guard");
        if (gguf_packed_bank_ && transformer_.bytes()) gguf_packed_bank_->check_unchanged();
    }
    if (quantized) {
#ifndef TURBOCIDER_ENABLE_QUANTIZED_EXECUTION_EXPERIMENTS
        throw std::invalid_argument("qe_capability_unqualified: explicit experimental build required");
#endif
        require(gguf_transformer_ && !load_only && !public_stream_lease_, "GGUF bounded mode requires a private generate request");
    }
    const bool exact_streaming = !quantized && z_image_exact_streaming_requested(r);
    const uint32_t exact_slot_count = exact_streaming
        ? z_image_exact_slot_count(r) : 0;
    const bool tight_exact = exact_streaming && exact_slot_count == 1;
    const bool legacy_streamed = r.residency == "streamed";
    const bool streamed = exact_streaming || legacy_streamed || quantized;
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
    if (quantized) {
        mx::synchronize();
        gguf_stream_.reset(); exact_stream_.reset(); weight_stream_.reset();
        transformer_.clear(); vae_.clear(); hybrid_.reset(); runtime_ffn_.reset(); encoder_hybrid_.reset();
        hybrid_gpu_graph_ = {}; stream_configuration_.clear(); mx::clear_cache();
    } else if (exact_streaming) {
        // Exact retention is request-scoped. Never inherit resident weights,
        // a legacy prefetcher, or a previous exact executor into this request.
        mx::synchronize();
        exact_stream_.reset();
        weight_stream_.reset();
        transformer_.clear();
        if (prompt_changed && optimizations_.z_image_memory_lifecycle) {
            hybrid_.reset();
            hybrid_gpu_graph_ = {};
            hybrid_gpu_mlp_start_ = -1;
        }
        if (tight_exact || (prompt_changed && constrained_memory))
            vae_.clear();
        mx::clear_cache();
        stream_configuration_.clear();
    } else if (z_qwen3_gguf_path() || configuration != stream_configuration_ ||
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
        gpu_w8_suffix_start_ = -1;
        gpu_w8_group_size_ = 0;
        gpu_w8_manifest_.clear();
        gpu_bf16_route_manifest_.clear();
        transformer_.clear();
        vae_.clear();
        mx::clear_cache();
        stream_configuration_ = configuration;
    }
    RequestCacheLimit cache_limit(
        streamed || constrained_memory || gguf_direct_import_,
        (tight_exact || quantized || gguf_direct_import_) ? 0 : r.allocator_cache_bytes);
    mx::reset_peak_memory();
    select_loras(r);
    auto text_start = Clock::now();
    bool prompt_hit = conditioning(r, event, cancelled);
    if (quantized) {
        mx::eval(*cached_conditioning_); mx::synchronize();
        text_encoder_.clear(); encoder_hybrid_.reset(); mx::clear_cache();
        event("qwen3_weights_released_before_gguf", 1, 1);
    }
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
    // An opted-in image-only W8A8 manifest needs the full BF16 GPU weights
    // for caption rows. Preserve the original explicit W8 ablation for all
    // other hybrid routes.
    const bool gpu_w8_ablation = (hybrid_ && hybrid_->image_only_token_rows == 1024) ||
        std::getenv("TURBOCIDER_Z_HYBRID_GPU_W8_DISABLE") != nullptr;
    int gpu_w8_group = 32;
    if (const char *configured = std::getenv("TURBOCIDER_Z_GPU_W8_GROUP")) {
        const std::string group(configured);
        require(group == "32" || group == "64" || group == "128",
                "Z-Image GPU W8 group must be 32, 64 or 128");
        gpu_w8_group = std::stoi(group);
    }
    const int requested_w8_start = hybrid_ && hybrid_->activation_precision == "int8" &&
        !gpu_w8_ablation
        ? hybrid_->ane_mlp_end : -1;
    const bool routed_bf16 = hybrid_ && hybrid_->activation_precision == "int8" &&
        hybrid_->has_channel_route && gpu_w8_ablation;
    const bool gpu_shard_changed =
        (gpu_w8_suffix_start_ >= 0 &&
         (requested_w8_start != gpu_w8_suffix_start_ || gpu_w8_group != gpu_w8_group_size_ ||
          (hybrid_ && hybrid_->manifest != gpu_w8_manifest_))) ||
        (!gpu_bf16_route_manifest_.empty() &&
         (!routed_bf16 || gpu_bf16_route_manifest_ != hybrid_->manifest));
    if (gpu_shard_changed) {
        transformer_.clear();
        gpu_w8_suffix_start_ = -1;
        gpu_w8_group_size_ = 0;
        gpu_w8_manifest_.clear();
        gpu_bf16_route_manifest_.clear();
        mx::clear_cache();
    }
    if (requested_w8_start >= 0)
        require(!streamed && !gguf_transformer_ && !convrot_transformer_ &&
                    !nvfp4_transformer_ && active_lora_strategy_ != "inference_time" &&
                    requested_w8_start % 32 == 0 && (10240 - requested_w8_start) % 32 == 0,
                "Z-Image BF16 W8A16 GPU + W8A8 ANE requires aligned resident BF16 weights");
    if (routed_bf16)
        require(!streamed && !gguf_transformer_ && !convrot_transformer_ &&
                    !nvfp4_transformer_ && active_lora_strategy_ != "inference_time",
                "routed Z-Image BF16 GPU + W8A8 ANE requires resident BF16 weights");
    event(r.execution == "gpu_ane" ? "route_gpu_ane" : "route_gpu", 1, 1);
    r.compile_gpu = !exact_streaming && r.execution == "gpu" &&
                    !gguf_transformer_ && !convrot_transformer_ && !nvfp4_transformer_ &&
                    active_lora_strategy_ != "inference_time" &&
                    !std::getenv("TURBOCIDER_Z_EAGER_BLOCKS");
    if (plan.request.execution != r.execution || plan.request.compile_gpu != r.compile_gpu)
        plan = make_plan(r);
    if (quantized) {
        gguf_stream_ = std::make_unique<ZImageGgufStream>(transformer_path_,
            r.quantized_execution.prefetch_layers.value_or(1), uint32_t(r.width), uint32_t(r.height),
            uint32_t(caption_rows), uint32_t(r.steps), std::min<uint64_t>(10ull << 30, device_info().physical_memory / 2),
            transformer_, event, cancelled, r.quantized_execution.precision_profile.value_or("z-source-mixed-v1"),
            r.quantized_execution.source_residency.value_or("packed_resident"));
        selection += "; experimental " + r.quantized_execution.source_residency.value_or("packed_resident") + " GGUF, bounded dequant, " + r.quantized_execution.precision_profile.value_or("z-source-mixed-v1");
    } else if (exact_streaming) {
        require(!hybrid_ || hybrid_->activation_precision != "int8",
                "streaming_route_unsupported: W8A8 ANE requires resident loading");
        require(!load_only && (r.execution == "gpu" || r.execution == "gpu_ane") &&
                    active_loras_.empty() && !gguf_transformer_ && !nvfp4_transformer_ &&
                    !diffusers_layout_ && !std::getenv("TURBOCIDER_Z_HYBRID_VALIDATE"),
                "streaming_route_unsupported: the Z-Image exact candidate "
                "supports generated Comfy BF16/INT8 ConvRot GPU or GPU+ANE requests "
                "without LoRA, full-MLP validation, Diffusers shards, or prepare-only mode");
        require(exact_stream_generation_ != UINT64_MAX,
                "Z-Image exact request generation overflow");
        ++exact_stream_generation_;
        const z_image::StreamingWorkload workload{
            uint32_t(r.width), uint32_t(r.height), uint32_t(caption_rows),
            uint32_t(r.steps),
            hybrid_ && optimizations_.z_image_suffix_streaming ? uint32_t(hybrid_->ane_mlp_end) : 0u,
            std::getenv("TURBOCIDER_Z_CONVROT_FP32_SCALES") != nullptr};
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
        exact_stream_->bind_hybrid(hybrid_.get(),
            hybrid_gpu_graph_ ? &hybrid_gpu_graph_ : nullptr, optimizations_.z_image_hybrid_segments);
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
    if (tight_exact || quantized) {
        // K1 is the minimum-memory exact layout.  Its fixed/prefix tensors are
        // already materialized by ZImageWeightStream; loading the VAE here
        // would keep another 320+ MiB resident throughout all denoise passes.
        transformer_.materialize();
    } else {
        load(event, cancelled);
    }
    if (requested_w8_start >= 0 && gpu_w8_suffix_start_ < 0) {
        for (int ordinal = 0; ordinal < 32; ++ordinal) {
            checkpoint(cancelled);
            const auto prefix = ordinal < 2
                ? "noise_refiner." + std::to_string(ordinal) + ".feed_forward"
                : "layers." + std::to_string(ordinal - 2) + ".feed_forward";
            event("pack_z_image_gpu_w8", ordinal, 32);
            if (z_hybrid_bf16_block(ordinal)) continue;
            if (hybrid_->has_channel_route) {
                const auto &gpu = hybrid_->gpu_channel_indices[size_t(ordinal)];
                transformer_.quantize_dense_indices(prefix + ".w1", gpu, 0, gpu_w8_group);
                transformer_.quantize_dense_indices(prefix + ".w3", gpu, 0, gpu_w8_group);
                transformer_.quantize_dense_indices(prefix + ".w2", gpu, 1, gpu_w8_group);
            } else {
                transformer_.quantize_dense_range(prefix + ".w1", requested_w8_start,
                                                   10240, 0, 3840, gpu_w8_group);
                transformer_.quantize_dense_range(prefix + ".w3", requested_w8_start,
                                                   10240, 0, 3840, gpu_w8_group);
                transformer_.quantize_dense_range(prefix + ".w2", 0, 3840,
                                                   requested_w8_start, 10240, gpu_w8_group);
            }
        }
        gpu_w8_suffix_start_ = requested_w8_start;
        gpu_w8_group_size_ = gpu_w8_group;
        gpu_w8_manifest_ = hybrid_->manifest;
        mx::clear_cache();
    }
    if (routed_bf16 && gpu_bf16_route_manifest_.empty()) {
        for (int ordinal = 0; ordinal < 32; ++ordinal) {
            checkpoint(cancelled);
            const auto prefix = ordinal < 2
                ? "noise_refiner." + std::to_string(ordinal) + ".feed_forward"
                : "layers." + std::to_string(ordinal - 2) + ".feed_forward";
            event("select_z_image_gpu_bf16", ordinal, 32);
            if (z_hybrid_bf16_block(ordinal)) continue;
            const auto &gpu = hybrid_->gpu_channel_indices[size_t(ordinal)];
            transformer_.select_dense_indices(prefix + ".w1", gpu, 0);
            transformer_.select_dense_indices(prefix + ".w3", gpu, 0);
            transformer_.select_dense_indices(prefix + ".w2", gpu, 1);
        }
        gpu_bf16_route_manifest_ = hybrid_->manifest;
        mx::clear_cache();
    }
    if (r.hybrid_mlp_mode == "runtime") {
        const auto manifest = std::filesystem::canonical(r.ane_manifest);
        const std::string identity = manifest.string() + ":" + sha256_file(manifest) + ":" +
            (std::getenv("TURBOCIDER_RUNTIME_ANE_CHUNKS") ? std::getenv("TURBOCIDER_RUNTIME_ANE_CHUNKS") : "auto");
        if (!runtime_ffn_ || !runtime_ffn_->available() || runtime_manifest_ != identity ||
            (!active_loras_.empty() && !runtime_ffn_->supports_lora_inputs())) {
            runtime_ffn_.reset();
            const auto physical = device_info().physical_memory;
            const size_t budget = std::min(uint64_t(2) << 30,
                physical - std::min(physical, uint64_t(mx::get_active_memory()) + (uint64_t(4) << 30)));
            runtime_ffn_ = std::make_unique<ane::HybridFfn>(manifest, 3840, 10240, budget, cancelled,
                                                         !active_loras_.empty());
            runtime_manifest_ = identity;
        }
        runtime_ffn_->begin_request(cached_lora_identity_);
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
            result.precision = requested_w8_start >= 0 ? "gpu_w8a16+coreml_w8a8"
                               : hybrid_ && hybrid_->activation_precision == "int8"
                                   ? (hybrid_->image_only_token_rows
                                      ? "gpu_bf16+coreml_w8a8"
                                      : "gpu_bf16+coreml_w8a8_ablation")
                               : hybrid_ ? hybrid_precision_label(hybrid_->metrics()) : "bf16";
            if (requested_w8_start >= 0)
                result.precision += z_hybrid_dequant_precision();
            if (hybrid_ && hybrid_->has_channel_route)
                result.precision += "+channel_routed";
            if (hybrid_ && hybrid_->image_only_token_rows)
                result.precision += "+image_only_ane_gpu_bf16_caption";
            if (hybrid_ && std::getenv("TURBOCIDER_Z_HYBRID_BF16_BLOCKS"))
                result.precision += "+selective_bf16_gpu_blocks";
        }
        if (hybrid_)
            result.hybrid = hybrid_->metrics();
        if (runtime_ffn_) {
            result.backend = gguf_transformer_ ? "mlx_cpp_metal_gguf+coreml_runtime_weight"
                                               : "mlx_cpp_metal+coreml_runtime_weight";
            result.precision = gguf_transformer_ ? "gguf_native_gpu+runtime_fp16_ffn"
                                                 : "bf16_gpu+runtime_fp16_ffn_bf16_io";
            result.hybrid = runtime_ffn_->metrics();
        }
        result.encoder_hybrid = cached_encoder_hybrid_metrics_;
        result.encoder_quantized_execution = cached_encoder_gguf_metrics_;
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
    dump("z_conditioning", caption);
    // Request-local: never reuse across prompts, LoRA changes or resolutions.
    // First-step refinement remains in denoise timing and is evaluated through
    // the unified graph before subsequent steps can consume the cached tensor.
    std::vector<Tensor> context_cache;
    const bool cache_context = !std::getenv("TURBOCIDER_Z_DISABLE_CACHE_CONTEXT") &&
        (std::getenv("TURBOCIDER_Z_CACHE_CONTEXT") ||
         (z_image_small_shape_metal_default() && r.width <= 512 && r.height <= 512)) &&
        !hybrid_ && !weight_stream_ && !exact_stream_ && !gguf_transformer_ &&
        !nvfp4_transformer_ && !convrot_transformer_;
    auto dit_start = Clock::now();
    profile.phase("denoise_begin");
    for (int i = 0; i < r.steps; ++i) {
        checkpoint(cancelled);
        if (gguf_packed_bank_ && gguf_direct_import_) gguf_packed_bank_->check_unchanged();
        event("denoise", i, r.steps);
        auto noise = denoise(z, caption, sigmas[i], float(r.width), r.height, i, event, cancelled,
                             cache_context ? &context_cache : nullptr);
        z = euler_step(z, noise, sigmas[i + 1] - sigmas[i]);
        mx::eval(z);
        dump("z_latent_step_" + std::to_string(i + 1), z);
        require(mx::all(mx::isfinite(z)).item<bool>(), "nonfinite Z-Image latent");
        event("denoise", i + 1, r.steps);
    }
    const double denoise_seconds = std::chrono::duration<double>(Clock::now() - dit_start).count();
    std::optional<streaming::GgufPackedBankMetrics> packed_import_metrics;
    if (gguf_direct_import_ && gguf_packed_bank_) {
        mx::synchronize();gguf_packed_bank_->check_unchanged();
        packed_import_metrics=gguf_packed_bank_->metrics();
        transformer_.clear();mx::clear_cache();
        require(gguf_packed_ledger_->snapshot().storage_bytes==0,
                "qe_drain_unproven: direct packed bank backing remains live before VAE");
        packed_import_metrics->released_before_vae=true;
        packed_import_metrics->serial_refiner_eval=true;
        event("gguf_packed_bank_released_before_vae",1,1);
    }
    std::optional<QuantizedExecutionMetrics> quantized_metrics;
    if (quantized) {
        gguf_stream_->finish(); quantized_metrics = gguf_stream_->metrics();
        transformer_.clear(); gguf_stream_.reset(); mx::clear_cache();
    }
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
        runtime.weight_format = convrot_transformer_
            ? "comfy-int8-convrot-single-file" : "comfy-bf16-single-file";
        runtime.kernel_revision = convrot_transformer_
            ? "z-image-convrot-packed-q8-v1" : kZImageKernelRevision;
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
    if (tight_exact || quantized) {
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
    if (gguf_packed_bank_ && gguf_direct_import_) { mx::synchronize();gguf_packed_bank_->check_unchanged(); }
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
    result.quantized_execution = quantized_metrics;
    if (packed_import_metrics) {
        result.gguf_import=packed_import_metrics;
        result.selection += "; experimental CPU direct affine packed import, allocator cache=0";
    }
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
        result.precision = gpu_w8_suffix_start_ >= 0 ? "gpu_w8a16+coreml_w8a8"
                           : hybrid_ && hybrid_->activation_precision == "int8"
                               ? (hybrid_->image_only_token_rows
                                  ? "gpu_bf16+coreml_w8a8"
                                  : "gpu_bf16+coreml_w8a8_ablation")
                           : hybrid_ ? hybrid_precision_label(hybrid_->metrics()) : "bf16";
        if (gpu_w8_suffix_start_ >= 0)
            result.precision += z_hybrid_dequant_precision();
        if (hybrid_ && hybrid_->has_channel_route)
            result.precision += "+channel_routed";
        if (hybrid_ && hybrid_->image_only_token_rows)
            result.precision += "+image_only_ane_gpu_bf16_caption";
        if (hybrid_ && std::getenv("TURBOCIDER_Z_HYBRID_BF16_BLOCKS"))
            result.precision += "+selective_bf16_gpu_blocks";
    }
    if (!gguf_transformer_ && !nvfp4_transformer_ &&
        !convrot_transformer_ && !hybrid_) {
        result.backend = "mlx_cpp_metal";
        result.precision = "bf16";
    }
    if (hybrid_) {
        result.hybrid = hybrid_->metrics();
        if (!gguf_transformer_ && !nvfp4_transformer_ && !convrot_transformer_) {
            result.backend = "mlx_cpp_metal+coreml";
            if (hybrid_->activation_precision != "int8")
                result.precision = hybrid_precision_label(*result.hybrid);
        }
    }
    if (runtime_ffn_) {
        runtime_ffn_->drain();
        result.backend = gguf_transformer_ ? "mlx_cpp_metal_gguf+coreml_runtime_weight"
                                           : "mlx_cpp_metal+coreml_runtime_weight";
        result.precision = gguf_transformer_ ? "gguf_native_gpu+runtime_fp16_ffn"
                                             : "bf16_gpu+runtime_fp16_ffn_bf16_io";
        result.hybrid = runtime_ffn_->metrics();
        if (!runtime_ffn_->available()) result.selection += "; GPU fallback: " + runtime_ffn_->reason();
    }
    if (quantized) {
        result.backend = r.quantized_execution.precision_profile == "z-source-native-affine-v1" || r.quantized_execution.precision_profile == "z-mlx-compat-affine-v1"
            ? "mlx_cpp_metal_gguf_bounded_native_affine" : "mlx_cpp_metal_gguf_bounded_cpu_dequant";
        result.precision = r.quantized_execution.precision_profile.value_or("z-source-mixed-v1");
    }
    if (gguf_direct_import_) {
        result.backend="mlx_cpp_metal_gguf_cpu_direct_packed";
        result.precision="z-mlx-compat-affine-v1";
    }
    result.encoder_hybrid = cached_encoder_hybrid_metrics_;
    result.encoder_quantized_execution = cached_encoder_gguf_metrics_;
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
    if (gguf_direct_import_) {
        try { mx::synchronize(); }
        catch (...) { streaming_quarantined_=true;throw; }
        transformer_.clear();gguf_packed_bank_.reset();gguf_packed_ledger_.reset();vae_.clear();mx::clear_cache();
    }
    if (gguf_stream_ && !gguf_stream_->drain_safely()) {
        streaming_quarantined_ = true;
        throw;
    }
    if (requested.quantized_execution.active()) {
        mx::synchronize(); transformer_.clear(); gguf_stream_.reset(); vae_.clear(); mx::clear_cache();
    }
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
                       int height, int step, const Event &event, std::atomic<bool> &cancelled,
                       std::vector<Tensor> *context_cache) {
    auto model_input = mx::astype(latent, mx::bfloat16);
    const bool experimental_compiled_a8 = hybrid_ && hybrid_->activation_precision == "int8" &&
        (hybrid_->image_only_token_rows == 1024 ||
         (hybrid_->has_channel_route &&
          std::getenv("TURBOCIDER_Z_HYBRID_GPU_W8_DISABLE") &&
          std::getenv("TURBOCIDER_Z_W8A8_COMPILED_HYBRID")));
    return mx::astype(
        z_transformer(model_input, caption, sigma, int(width), height, transformer_, event,
                      cancelled, hybrid_.get(), hybrid_ ? &hybrid_gpu_graph_ : nullptr,
                      weight_stream_.get(), exact_stream_.get(), uint32_t(step),
                      optimizations_.z_image_hybrid_segments || experimental_compiled_a8,
                      nullptr, context_cache, runtime_ffn_.get(), gguf_transformer_, gguf_stream_.get(),gguf_direct_import_),
        mx::float32);
}

Tensor ZImage::decode(const Tensor &latent, int, int, const Event &, std::atomic<bool> &) {
    return z_image::decode_vae(latent, vae_);
}

} // namespace tc

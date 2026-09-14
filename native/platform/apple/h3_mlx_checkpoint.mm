#import <Foundation/Foundation.h>

#include "../../models/h3_mlx/checkpoint.hpp"
#include "../../models/h3_mlx/geometry.hpp"

#include <cmath>
#include <set>

namespace tc::h3_mlx {
namespace {
NSDictionary *object(id value, const std::string &name) {
    require([value isKindOfClass:NSDictionary.class],
            "H3 MLX checkpoint expects object: " + name);
    return (NSDictionary *)value;
}

NSArray *array(id value, const std::string &name) {
    require([value isKindOfClass:NSArray.class],
            "H3 MLX checkpoint expects array: " + name);
    return (NSArray *)value;
}

int integer(NSDictionary *value, NSString *key) {
    id number = value[key];
    require([number isKindOfClass:NSNumber.class] &&
                CFGetTypeID((__bridge CFTypeRef)number) != CFBooleanGetTypeID() &&
                [number doubleValue] == [number intValue],
            "H3 MLX checkpoint expects integer: " + std::string(key.UTF8String));
    return [number intValue];
}

float number(NSDictionary *value, NSString *key, float fallback) {
    id item = value[key];
    if (!item) return fallback;
    require([item isKindOfClass:NSNumber.class] &&
                CFGetTypeID((__bridge CFTypeRef)item) != CFBooleanGetTypeID(),
            "H3 MLX checkpoint expects number: " + std::string(key.UTF8String));
    return [item floatValue];
}

std::string text(NSDictionary *value, NSString *key, const std::string &fallback = {}) {
    id item = value[key];
    if (!item) return fallback;
    require([item isKindOfClass:NSString.class],
            "H3 MLX checkpoint expects string: " + std::string(key.UTF8String));
    return [(NSString *)item UTF8String];
}

bool boolean(NSDictionary *value, NSString *key, bool fallback = false) {
    id item = value[key];
    if (!item) return fallback;
    require(CFGetTypeID((__bridge CFTypeRef)item) == CFBooleanGetTypeID(),
            "H3 MLX checkpoint expects bool: " + std::string(key.UTF8String));
    return [item boolValue];
}

void expected_integer(NSDictionary *value, NSString *key, int expected) {
    require(integer(value, key) == expected,
            "unsupported H3 MLX checkpoint field: " + std::string(key.UTF8String));
}

std::optional<int> block_index(const std::string &name, const std::string &prefix) {
    if (!name.starts_with(prefix)) return std::nullopt;
    auto end = name.find('.', prefix.size());
    require(end != std::string::npos, "invalid H3 MLX block key: " + name);
    auto index = name.substr(prefix.size(), end - prefix.size());
    require(!index.empty() && index.size() <= 2 &&
                index.find_first_not_of("0123456789") == std::string::npos,
            "invalid H3 MLX block index: " + name);
    int result = std::stoi(index);
    require(index == std::to_string(result), "noncanonical H3 MLX block index: " + name);
    return result;
}
} // namespace

void Checkpoint::load(const std::filesystem::path &root, const Event &event,
                      std::atomic<bool> &cancelled) {
    require(root.is_absolute() && std::filesystem::is_directory(root),
            "H3 MLX checkpoint root must be an existing absolute directory");
    require(arrays_.empty() && quantized_.empty(), "H3 MLX checkpoint already loaded");
    checkpoint(cancelled);
    auto manifest_path = root / "mlx_h3_dit.json";
    auto weights_path = root / "mlx_h3_dit.safetensors";
    require(std::filesystem::is_regular_file(manifest_path) &&
                std::filesystem::is_regular_file(weights_path),
            "H3 MLX checkpoint files missing");

    NSData *data = [NSData dataWithContentsOfFile:@(manifest_path.c_str())];
    require(data != nil, "cannot read H3 MLX checkpoint manifest");
    NSError *error = nil;
    NSDictionary *manifest = object(
        [NSJSONSerialization JSONObjectWithData:data options:0 error:&error], "manifest");
    expected_integer(manifest, @"format_version", 1);
    expected_integer(manifest, @"num_blocks", config_.num_layers);
    expected_integer(manifest, @"num_refiner_blocks", config_.refiner_layers);
    const std::string profile = text(manifest, @"profile", "fasth3");
    const bool vdn_profile = profile == "minimax-h3-vdn";
    require(profile == "fasth3" || vdn_profile,
            "unsupported H3 MLX checkpoint profile: " + profile);
    const int expected_steps = vdn_profile ? 6 : 4;
    if (manifest[@"steps"])
        expected_integer(manifest, @"steps", expected_steps);

    NSDictionary *config = object(manifest[@"config"], "config");
    expected_integer(config, @"hidden_size", config_.hidden_size);
    expected_integer(config, @"num_layers", config_.num_layers);
    expected_integer(config, @"num_refiner_layers", config_.refiner_layers);
    expected_integer(config, @"num_attention_heads", config_.num_heads);
    expected_integer(config, @"attention_head_dim", config_.head_dim);
    expected_integer(config, @"ffn_dim", config_.ffn_dim);
    expected_integer(config, @"in_channels", config_.latent_channels);
    expected_integer(config, @"audio_in_channels", config_.audio_latent_channels);
    expected_integer(config, @"text_dim", config_.text_dim);
    expected_integer(config, @"freq_dim", config_.frequency_dim);
    expected_integer(config, @"time_embed_dim", config_.time_embed_dim);
    expected_integer(config, @"rope_freq_dim", config_.rope_frequency_dim);
    auto patch = array(config[@"patch_size"], "config.patch_size");
    require([patch isEqual:@[@1, @2, @2]], "unsupported H3 MLX patch size");
    require(std::abs(number(config, @"norm_eps", 0.f) - config_.norm_epsilon) < 1e-9f &&
                std::abs(number(config, @"qk_norm_eps", 0.f) - config_.qk_norm_epsilon) < 1e-9f &&
                std::abs(number(config, @"final_norm_eps", 0.f) - config_.final_norm_epsilon) < 1e-9f &&
                std::abs(number(config, @"rope_theta", 0.f) - config_.rope_theta) < 1e-6f,
            "unsupported H3 MLX normalization epsilon");

    NSDictionary *spec = object(manifest[@"quantization"], "quantization");
    require(text(spec, @"mode") == "affine", "H3 MLX checkpoint requires affine quantization");
    expected_integer(spec, @"bits", 6);
    expected_integer(spec, @"group_size", 64);
    quantization_ = {"affine", 6, 64};

    NSDictionary *adaln = object(manifest[@"adaln_cache"], "adaln_cache");
    expected_integer(adaln, @"num_blocks", config_.num_layers);
    expected_integer(adaln, @"tables_per_block", 6);
    NSArray *timesteps = array(adaln[@"timesteps"], "adaln_cache.timesteps");
    // video shift 12 contributes {0, 1/37, 1/13, 1/5}; audio shift 3
    // contributes {0, 1/10, 1/4, 1/2}; conversion also stores clean time 1.
    const auto expected_ladder = adaln_timestep_union(expected_steps);
    require(timesteps.count == expected_ladder.size(),
            vdn_profile
                ? "VDN H3 MLX checkpoint requires the fixed six-step AdaLN ladder"
                : "H3 MLX INT6 checkpoint requires the fixed four-step AdaLN ladder");
    identity_.adaln_timesteps.reserve(timesteps.count);
    for (id raw in timesteps) {
        require([raw isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)raw) != CFBooleanGetTypeID(),
                "H3 MLX AdaLN timestep must be numeric");
        identity_.adaln_timesteps.push_back([raw floatValue]);
    }
    require(identity_.adaln_timesteps.size() == expected_ladder.size(),
            "H3 MLX AdaLN ladder size mismatch");
    for (size_t index = 0; index < expected_ladder.size(); ++index)
        require(std::abs(identity_.adaln_timesteps[index] - expected_ladder[index]) <= 1e-6f,
                vdn_profile
                    ? "VDN H3 MLX AdaLN ladder does not match the six-step stage-DMD schedule"
                    : "H3 MLX AdaLN ladder does not match the four-step FastH3 schedule");

    NSDictionary *vsa = object(manifest[@"vsa"], "vsa");
    identity_.format_version = 1;
    identity_.steps = expected_steps;
    // The checkpoint schedule and the number of model evaluations are
    // profile-bound.  FastH3 uses four evaluations; the VDN stage-DMD
    // artifact uses six.  Keep this explicit instead of relying on the
    // struct default, so a VDN load cannot silently fall back to the
    // FastH3/default schedule.
    identity_.denoise_steps = expected_steps;
    identity_.video_shift = 12.f;
    identity_.audio_shift = 3.f;
    identity_.vsa_capable = boolean(vsa, @"capable");
    identity_.vsa_gate_matrices = integer(vsa, @"num_gate_matrices");
    require(text(vsa, @"gate_key_suffix") == "attn.to_gate_compress.weight" &&
                text(vsa, @"attention_activations") == "bf16",
            "unsupported H3 VSA checkpoint contract");
    require((identity_.vsa_capable &&
             identity_.vsa_gate_matrices == config_.num_layers) ||
                (!identity_.vsa_capable && identity_.vsa_gate_matrices == 0),
            "H3 VSA checkpoint capability and gate count disagree");
    NSDictionary *source = manifest[@"source"] ? object(manifest[@"source"], "source") : nil;
    if (source) {
        identity_.source_sha256 = text(source, @"checkpoint_sha256",
                                       text(source, @"transformer_index_sha256"));
        identity_.source_repository = text(source, @"repository");
    }

    std::unordered_map<std::string, Tensor> vdn_arrays;
    if (vdn_profile) {
        NSDictionary *vdn = object(manifest[@"vdn"], "vdn");
        require(text(vdn, @"stage") == "stage-dmd-step-250" &&
                    text(vdn, @"repository") == "OpenVDN/vdn-minimax-h3" &&
                    text(vdn, @"turbo_adapter_family") == "larryvrh_v4_step600_ema",
                "unsupported VDN H3 stage or adapter identity");
        const auto branch_sha256 = text(vdn, @"branch_sha256");
        require(branch_sha256.size() == 64,
                "VDN H3 branch SHA-256 must be present in the manifest");
        const int64_t branch_bytes = [vdn[@"branch_bytes"] longLongValue];
        require(branch_bytes == 4'279'428'112ll,
                "unsupported VDN H3 linear branch byte size");

        NSDictionary *attention = object(vdn[@"attention"], "vdn.attention");
        expected_integer(attention, @"version", 2);
        require(text(attention, @"anchor_frames") == "both" &&
                    text(attention, @"delta_rule") == "vdn_solve" &&
                    text(attention, @"bridge") == "alpha",
                "unsupported VDN H3 hybrid attention algorithm");
        expected_integer(attention, @"softmax_chunk", 5);
        expected_integer(attention, @"softmax_radius", 1);
        expected_integer(attention, @"linear_head_dim", config_.head_dim);
        require(boolean(attention, @"a_fp32") &&
                    boolean(attention, @"enable_text_state") &&
                    boolean(attention, @"enable_softmax_gate"),
                "VDN H3 requires FP32 A, text state, and the softmax gate");
        NSArray *conv_targets = array(attention[@"short_conv_targets"],
                                      "vdn.attention.short_conv_targets");
        require([conv_targets isEqual:@[@"k", @"v"]],
                "VDN H3 short convolution must target K and V only");

        identity_.vdn.enabled = true;
        identity_.vdn.version = 2;
        identity_.vdn.softmax_chunk = 5;
        identity_.vdn.softmax_radius = 1;
        identity_.vdn.linear_head_dim = config_.head_dim;
        identity_.vdn.anchor_rows = true;
        identity_.vdn.anchor_columns = true;
        identity_.vdn.a_fp32 = true;
        identity_.vdn.bridge_alpha = true;
        identity_.vdn.enable_text_state = true;
        identity_.vdn.enable_softmax_gate = true;
        identity_.vdn.conv_k = true;
        identity_.vdn.conv_v = true;
        identity_.vdn.delta_rule = "vdn_solve";
        identity_.vdn.branch_sha256 = branch_sha256;
        identity_.vdn.branch_bytes = uint64_t(branch_bytes);

        NSDictionary *assets = object(vdn[@"assets"], "vdn.assets");
        const auto relative = text(assets, @"linear_branch");
        require(!relative.empty() && std::filesystem::path(relative).is_relative(),
                "VDN H3 linear branch path must be relative to the checkpoint");
        std::error_code path_error;
        auto branch_path = std::filesystem::weakly_canonical(root / relative,
                                                              path_error);
        require(!path_error && std::filesystem::is_regular_file(branch_path) &&
                    std::filesystem::file_size(branch_path) == uint64_t(branch_bytes),
                "VDN H3 linear branch asset is missing or has the wrong size");
        event("h3_mlx_load_vdn", 0, 1);
        vdn_arrays = mx::load_safetensors(branch_path.string()).first;
        std::set<std::string> consumed_vdn;
        auto expect_vdn = [&](const std::string &name, const mx::Shape &shape) {
            auto found = vdn_arrays.find(name);
            require(found != vdn_arrays.end(),
                    "missing VDN H3 linear branch tensor: " + name);
            require(found->second.dtype() == mx::bfloat16 &&
                        found->second.shape() == shape,
                    "invalid VDN H3 linear branch tensor: " + name);
            consumed_vdn.insert(name);
        };
        for (int index = 0; index < config_.num_layers; ++index) {
            checkpoint(cancelled);
            const auto prefix = "transformer_blocks." + std::to_string(index) +
                                ".attn.";
            const auto linear = prefix + "linear_attention.";
            expect_vdn(linear + "short_conv.k_sp.weight", {7168, 1, 5, 5});
            expect_vdn(linear + "short_conv.k_tm.weight", {7168, 1, 5});
            expect_vdn(linear + "short_conv.v_sp.weight", {7168, 1, 5, 5});
            expect_vdn(linear + "short_conv.v_tm.weight", {7168, 1, 5});
            expect_vdn(linear + "beta_proj.weight", {56, 5376});
            expect_vdn(linear + "alpha.down.weight", {128, 5376});
            expect_vdn(linear + "alpha.up.weight", {7168, 128});
            expect_vdn(linear + "alpha.dt_bias", {7168});
            expect_vdn(linear + "alpha.A_log", {56});
            expect_vdn(linear + "output_gate.down.weight", {128, 5376});
            expect_vdn(linear + "output_gate.up.weight", {7168, 128});
            expect_vdn(linear + "output_gate.up.bias", {7168});
            expect_vdn(linear + "norm.weight", {128});
            expect_vdn(prefix + "softmax_gate.up.weight", {56, 5376});
            expect_vdn(prefix + "softmax_gate.up.bias", {56});
            expect_vdn(prefix + "to_out_linear.weight", {5376, 7168});
        }
        require(consumed_vdn.size() == vdn_arrays.size() &&
                    consumed_vdn.size() == 800,
                "VDN H3 linear branch contains undeclared tensors");
        event("h3_mlx_load_vdn", 1, 1);
    }

    event("h3_mlx_load_dit", 0, 1);
    auto arrays = mx::load_safetensors(weights_path.string()).first;
    NSDictionary *declared = object(manifest[@"quantized_keys"], "quantized_keys");
    require(declared.count > 0, "H3 MLX quantized key declaration is empty");
    std::unordered_map<std::string, components::AffineMatrix> matrices;
    std::set<std::string> consumed;
    std::set<int> vsa_gates;
    for (NSString *raw_key in declared) {
        checkpoint(cancelled);
        std::string key(raw_key.UTF8String);
        require(key.ends_with(".weight"), "H3 MLX quantized key must name a weight: " + key);
        if (key.ends_with(".attn.to_gate_compress.weight")) {
            auto index = block_index(key, "blocks.");
            require(index.has_value(), "H3 VSA gate must belong to a transformer block");
            vsa_gates.insert(*index);
        }
        NSDictionary *record = object(declared[raw_key], key);
        require(text(record, @"dequantized_dtype") == "bf16",
                "H3 MLX INT6 weights must dequantize to BF16");
        auto packed = arrays.find(key);
        auto scales = arrays.find(key + ".scales");
        auto biases = arrays.find(key + ".biases");
        bool has_biases = boolean(record, @"has_biases");
        require(packed != arrays.end() && scales != arrays.end() &&
                    (!has_biases || biases != arrays.end()),
                "missing declared H3 MLX quantized tensor: " + key);
        // FastVideo's current MLX exporter writes affine scale/bias arrays in
        // FP32 (older internal exports used BF16). Both are valid inputs to
        // MLX quantized_matmul; activations and the dense dequantized path
        // still follow the manifest's BF16 contract.
        require(scales->second.dtype() == mx::bfloat16 ||
                    scales->second.dtype() == mx::float32,
                "H3 MLX affine scales must be BF16 or FP32");
        std::optional<Tensor> offsets;
        if (has_biases) offsets = biases->second;
        int logical_input = scales->second.shape(1) * quantization_.group_size;
        matrices.emplace(key, components::AffineMatrix(
            packed->second, scales->second, std::move(offsets),
            quantization_.group_size, quantization_.bits, logical_input));
        consumed.insert(key);
        consumed.insert(key + ".scales");
        if (has_biases) consumed.insert(key + ".biases");
    }

    std::set<int> blocks, refiners;
    for (const auto &[name, value] : arrays) {
        if (auto index = block_index(name, "blocks.")) {
            require(*index < config_.num_layers, "H3 MLX block index out of range: " + name);
            blocks.insert(*index);
        }
        if (auto index = block_index(name, "refiner.")) {
            require(*index < config_.refiner_layers, "H3 MLX refiner index out of range: " + name);
            refiners.insert(*index);
        }
        if (!consumed.count(name) && !name.starts_with("__adaln_cache.")) {
            require(!name.ends_with(".scales") && !name.ends_with(".biases"),
                    "undeclared H3 MLX quantized tensor: " + name);
            require(value.dtype() == mx::bfloat16 || value.dtype() == mx::float32,
                    "unsupported H3 MLX dense tensor dtype: " + name);
        }
    }
    require(blocks.size() == size_t(config_.num_layers) &&
                refiners.size() == size_t(config_.refiner_layers),
            "H3 MLX checkpoint has missing transformer or refiner blocks");
    require(vsa_gates.size() == size_t(identity_.vsa_gate_matrices),
            "H3 VSA manifest gate count does not match quantized tensors");
    checkpoint(cancelled);
    arrays_ = std::move(arrays);
    quantized_ = std::move(matrices);
    vdn_arrays_ = std::move(vdn_arrays);
    event("h3_mlx_load_dit", 1, 1);
}

void Checkpoint::clear() {
    quantized_.clear();
    arrays_.clear();
    vdn_arrays_.clear();
    quantization_ = {};
    identity_ = {};
    affine_dq_gemm_min_rows_ = 768;
    experimental_fused_qkv_ = false;
    reset_dispatch_metrics();
}

const Tensor &Checkpoint::at(const std::string &name) const {
    auto found = arrays_.find(name);
    require(found != arrays_.end(), "missing H3 MLX weight: " + name);
    return found->second;
}

bool Checkpoint::has(const std::string &name) const { return arrays_.count(name) != 0; }

const Tensor &Checkpoint::vdn_at(const std::string &name) const {
    auto found = vdn_arrays_.find(name);
    require(found != vdn_arrays_.end(), "missing VDN H3 weight: " + name);
    return found->second;
}

bool Checkpoint::vdn_has(const std::string &name) const {
    return vdn_arrays_.count(name) != 0;
}

bool Checkpoint::is_quantized(const std::string &prefix) const {
    return quantized_.count(prefix + ".weight") != 0;
}

mx::Dtype Checkpoint::projection_dtype(const std::string &prefix) const {
    if (is_quantized(prefix)) return mx::bfloat16;
    return at(prefix + ".weight").dtype();
}

Tensor Checkpoint::linear(const Tensor &x, const std::string &prefix,
                          bool prefer_wide_dense) const {
    Tensor output = x;
    auto packed = quantized_.find(prefix + ".weight");
    if (packed != quantized_.end()) {
        int rows = x.size() / x.shape(-1);
        bool dequantize = prefer_wide_dense && affine_dq_gemm_min_rows_ > 0 &&
                          rows >= affine_dq_gemm_min_rows_;
        if (dequantize)
            ++dequantized_gemm_calls_;
        else
            ++quantized_matmul_calls_;
        output = packed->second.project(x, dequantize);
    } else {
        const auto &weight = at(prefix + ".weight");
        require(weight.ndim() == 2 && x.shape(-1) == weight.shape(1),
                "invalid H3 MLX dense projection: " + prefix);
        output = mx::matmul(x, mx::transpose(weight));
    }
    if (has(prefix + ".bias"))
        output = output + mx::astype(at(prefix + ".bias"), output.dtype());
    return output;
}

Tensor Checkpoint::linear_fused(const Tensor &x,
                                const std::vector<std::string> &prefixes,
                                bool prefer_wide_dense) const {
    require(!prefixes.empty(), "H3 fused linear requires at least one prefix");
    for (const auto &prefix : prefixes)
        require(is_quantized(prefix),
                "H3 fused linear currently requires quantized projections: " + prefix);
    const int rows = x.size() / x.shape(-1);
    const bool wide = prefer_wide_dense && affine_dq_gemm_min_rows_ > 0 &&
                      rows >= affine_dq_gemm_min_rows_;
    if (!wide) {
        std::vector<Tensor> outputs;
        outputs.reserve(prefixes.size());
        for (const auto &prefix : prefixes)
            outputs.push_back(linear(x, prefix, prefer_wide_dense));
        return mx::concatenate(outputs, -1);
    }

    std::vector<Tensor> dense_weights;
    dense_weights.reserve(prefixes.size());
    std::vector<Tensor> biases;
    const bool has_bias = has(prefixes.front() + ".bias");
    if (has_bias) biases.reserve(prefixes.size());
    for (const auto &prefix : prefixes) {
        const auto found = quantized_.find(prefix + ".weight");
        require(found != quantized_.end(),
                "missing H3 fused quantized projection: " + prefix);
        dense_weights.push_back(found->second.dequantized(x.dtype()));
        require(has(prefix + ".bias") == has_bias,
                "H3 fused linear bias presence must be consistent");
        if (has_bias)
            biases.push_back(mx::astype(at(prefix + ".bias"), x.dtype()));
    }
    ++dequantized_gemm_calls_;
    auto weight = mx::concatenate(dense_weights, 0);
    auto output = mx::astype(mx::matmul(x, mx::transpose(weight)), x.dtype());
    if (has_bias) {
        output = output + mx::concatenate(biases, 0);
    }
    return output;
}

void Checkpoint::set_affine_dq_gemm_min_rows(int rows) {
    require(rows >= 0, "H3 MLX affine DQ-GEMM row threshold must be nonnegative");
    affine_dq_gemm_min_rows_ = rows;
}

void Checkpoint::reset_dispatch_metrics() const {
    quantized_matmul_calls_ = 0;
    dequantized_gemm_calls_ = 0;
}

size_t Checkpoint::bytes() const {
    size_t result = 0;
    for (const auto &[_, value] : arrays_) result += value.nbytes();
    for (const auto &[_, value] : vdn_arrays_) result += value.nbytes();
    return result;
}

} // namespace tc::h3_mlx

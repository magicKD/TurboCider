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
    require(timesteps.count == 8,
            "H3 MLX INT6 checkpoint requires the fixed four-step AdaLN ladder");
    auto expected_ladder = four_step_adaln_union();
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
                "H3 MLX AdaLN ladder does not match the four-step FastH3 schedule");

    NSDictionary *vsa = object(manifest[@"vsa"], "vsa");
    identity_.format_version = 1;
    identity_.steps = 4;
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
        identity_.source_sha256 = text(source, @"checkpoint_sha256");
        identity_.source_repository = text(source, @"repository");
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
    event("h3_mlx_load_dit", 1, 1);
}

void Checkpoint::clear() {
    quantized_.clear();
    arrays_.clear();
    quantization_ = {};
    identity_ = {};
    affine_dq_gemm_min_rows_ = 768;
    reset_dispatch_metrics();
}

const Tensor &Checkpoint::at(const std::string &name) const {
    auto found = arrays_.find(name);
    require(found != arrays_.end(), "missing H3 MLX weight: " + name);
    return found->second;
}

bool Checkpoint::has(const std::string &name) const { return arrays_.count(name) != 0; }

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
    return result;
}

} // namespace tc::h3_mlx

#import <Foundation/Foundation.h>

#include "../../models/wan/checkpoint.hpp"

#include <set>

namespace tc::wan {
namespace {
NSDictionary *object(id value, const std::string &name) {
    require([value isKindOfClass:NSDictionary.class], "Wan checkpoint expects object: " + name);
    return (NSDictionary *)value;
}

void integer(NSDictionary *config, NSString *key, int expected) {
    id value = config[key];
    require([value isKindOfClass:NSNumber.class] &&
                CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
                [value doubleValue] == expected,
            "unsupported Wan checkpoint config: " + std::string(key.UTF8String));
}
} // namespace

void Checkpoint::load(const std::filesystem::path &root, const Event &event,
                       std::atomic<bool> &cancelled) {
    require(root.is_absolute() && std::filesystem::is_directory(root),
            "Wan checkpoint root must be an existing absolute directory");
    require(arrays_.empty(), "Wan checkpoint already loaded");
    checkpoint(cancelled);
    auto manifest_path = root / "mlx_dit.json";
    auto weights_path = root / "mlx_dit.safetensors";
    require(std::filesystem::is_regular_file(manifest_path) &&
                std::filesystem::is_regular_file(weights_path), "Wan checkpoint files missing");
    NSData *data = [NSData dataWithContentsOfFile:@(manifest_path.c_str())];
    require(data != nil, "cannot read Wan checkpoint manifest");
    NSError *error = nil;
    NSDictionary *manifest = object([NSJSONSerialization JSONObjectWithData:data options:0 error:&error],
                                    "manifest");
    integer(manifest, @"format_version", 1);
    integer(manifest, @"num_blocks", config_.layers);
    NSDictionary *config = object(manifest[@"config"], "config");
    require([config[@"_class_name"] isEqual:@"WanTransformer3DModel"], "not a Wan DiT checkpoint");
    integer(config, @"num_layers", config_.layers);
    integer(config, @"num_attention_heads", config_.heads);
    integer(config, @"attention_head_dim", config_.head_dim);
    integer(config, @"ffn_dim", config_.ffn_dim);
    integer(config, @"freq_dim", config_.frequency_dim);
    integer(config, @"text_dim", config_.text_dim);
    integer(config, @"in_channels", config_.channels);
    integer(config, @"out_channels", config_.channels);
    require([config[@"patch_size"] isEqual:@[@1, @2, @2]] &&
                [config[@"qk_norm"] isEqual:@"rms_norm_across_heads"] &&
                [config[@"cross_attn_norm"] isEqual:@YES] &&
                [config[@"eps"] isKindOfClass:NSNumber.class] &&
                [config[@"eps"] doubleValue] == 1e-6,
            "unsupported Wan patch, normalization or epsilon");
    for (NSString *key in @[@"image_dim", @"added_kv_proj_dim"])
        require(config[key] == nil || config[key] == NSNull.null,
                "Wan image conditioning is not supported by this checkpoint loader");
    NSDictionary *spec = object(manifest[@"quantization"], "quantization");
    require([spec[@"mode"] isEqual:@"affine"], "Wan checkpoint requires affine quantization");
    integer(spec, @"bits", 8);
    integer(spec, @"group_size", 64);
    NSDictionary *quantized = object(manifest[@"quantized_keys"], "quantized_keys");
    require(quantized.count > 0, "Wan checkpoint quantized key declaration is empty");
    event("wan_load_dit", 0, 1);
    // Build temporary state so cancellation or a malformed manifest cannot
    // leave a partially loaded component that a subsequent request can reuse.
    auto arrays = mx::load_safetensors(weights_path.string()).first;
    std::unordered_map<std::string, components::AffineMatrix> matrices;
    std::set<std::string> consumed;
    for (NSString *key in quantized) {
        checkpoint(cancelled);
        std::string name(key.UTF8String);
        require(name.ends_with(".weight"), "Wan quantized key must name a weight");
        NSDictionary *record = object(quantized[key], name);
        require([record[@"dequantized_dtype"] isEqual:@"fp16"] &&
                    [record[@"has_biases"] isEqual:@YES],
                "Wan INT8 checkpoint requires fp16 affine weights with offsets");
        auto packed = arrays.find(name), scales = arrays.find(name + ".scales"),
             offsets = arrays.find(name + ".biases");
        require(packed != arrays.end() && scales != arrays.end() && offsets != arrays.end(),
                "missing declared Wan quantized tensor: " + name);
        require(scales->second.dtype() == mx::float16, "Wan affine scales must be fp16");
        matrices.emplace(name, components::AffineMatrix(packed->second, scales->second,
                                                        offsets->second, 64, 8));
        consumed.insert(name);
        consumed.insert(name + ".scales");
        consumed.insert(name + ".biases");
    }
    std::set<int> blocks;
    for (const auto &[name, value] : arrays) {
        if (!consumed.count(name)) {
            require(!name.ends_with(".scales") && !name.ends_with(".biases") &&
                        value.dtype() == mx::float16,
                    "undeclared or unsupported Wan tensor: " + name);
        }
        if (name.starts_with("blocks.")) {
            const auto end = name.find('.', 7);
            auto index = name.substr(7, end == std::string::npos ? end : end - 7);
            require(end != std::string::npos && !index.empty() && index.size() <= 2 &&
                        index.find_first_not_of("0123456789") == std::string::npos,
                    "invalid Wan block key: " + name);
            const int block = std::stoi(index);
            require(block < config_.layers && index == std::to_string(block),
                    "Wan block index out of range: " + name);
            blocks.insert(block);
        }
    }
    require(blocks.size() == size_t(config_.layers), "Wan checkpoint has missing blocks");
    checkpoint(cancelled);
    arrays_ = std::move(arrays);
    quantized_ = std::move(matrices);
    event("wan_load_dit", 1, 1);
}

const Tensor &Checkpoint::at(const std::string &name) const {
    auto found = arrays_.find(name);
    require(found != arrays_.end(), "missing Wan weight: " + name);
    return found->second;
}

bool Checkpoint::has(const std::string &name) const { return arrays_.count(name) != 0; }

Tensor Checkpoint::linear(const Tensor &x, const std::string &prefix) const {
    auto found = quantized_.find(prefix + ".weight");
    Tensor y = x;
    if (found != quantized_.end()) {
        y = found->second.project(x);
    } else {
        const auto &weight = at(prefix + ".weight");
        require(weight.ndim() == 2 && x.ndim() > 0 && weight.shape(1) == x.shape(-1),
                "invalid Wan dense projection: " + prefix);
        y = mx::matmul(x, mx::transpose(weight));
    }
    if (has(prefix + ".bias")) {
        const auto &bias = at(prefix + ".bias");
        require(bias.ndim() == 1 && bias.shape(0) == y.shape(-1),
                "invalid Wan linear bias: " + prefix);
        y = y + bias;
    }
    return y;
}

const components::AffineMatrix &Checkpoint::affine(const std::string &key) const {
    auto found = quantized_.find(key);
    require(found != quantized_.end(), "Wan projection is not declared affine: " + key);
    return found->second;
}

size_t Checkpoint::bytes() const {
    size_t result = 0;
    for (const auto &[_, value] : arrays_) result += value.nbytes();
    return result;
}

void Checkpoint::clear() { quantized_.clear(); arrays_.clear(); }

} // namespace tc::wan

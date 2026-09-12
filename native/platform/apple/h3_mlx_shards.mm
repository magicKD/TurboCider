#import <Foundation/Foundation.h>

#include "../../models/h3_mlx/conditioner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>

namespace tc::h3_mlx {
namespace {
NSDictionary *json_object_at_path(const std::filesystem::path &path) {
    NSData *data = [NSData dataWithContentsOfFile:@(path.c_str())];
    require(data != nil, "H3 conditioner JSON is missing: " + path.string());
    NSError *error = nil;
    id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    require([value isKindOfClass:NSDictionary.class],
            "H3 conditioner JSON must be an object: " + path.string());
    return (NSDictionary *)value;
}

double json_number(NSDictionary *object, NSString *key, double fallback = NAN) {
    id value = object[key];
    if (!value) {
        require(std::isfinite(fallback),
                "missing H3 conditioner config field: " + std::string(key.UTF8String));
        return fallback;
    }
    require([value isKindOfClass:NSNumber.class] &&
                CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            "H3 conditioner config field must be numeric: " + std::string(key.UTF8String));
    return [value doubleValue];
}

int json_integer(NSDictionary *object, NSString *key, int fallback = -1) {
    double value = json_number(object, key, fallback);
    require(std::isfinite(value) && value >= 0 && value <= INT32_MAX &&
                std::floor(value) == value,
            "invalid H3 conditioner integer: " + std::string(key.UTF8String));
    return static_cast<int>(value);
}

std::vector<uint8_t> read_bytes(NSFileHandle *handle, uint64_t offset, size_t bytes) {
    [handle seekToFileOffset:offset];
    NSData *data = [handle readDataOfLength:bytes];
    require(data.length == bytes, "truncated H3 conditioner safetensors tensor");
    const auto *begin = static_cast<const uint8_t *>(data.bytes);
    return std::vector<uint8_t>(begin, begin + data.length);
}

uint64_t little_u64(const uint8_t *bytes) {
    uint64_t value = 0;
    for (int index = 0; index < 8; ++index)
        value |= uint64_t(bytes[index]) << (8 * index);
    return value;
}

struct TensorInfo {
    std::string path;
    std::vector<int> shape;
    std::string dtype;
    uint64_t begin = 0;
    uint64_t end = 0;
};

struct Header {
    uint64_t data_offset = 0;
    std::map<std::string, TensorInfo> tensors;
};

Header read_header(const std::string &path) {
    NSFileHandle *handle = [NSFileHandle fileHandleForReadingAtPath:@(path.c_str())];
    require(handle != nil, "cannot open H3 conditioner shard: " + path);
    @try {
        auto size_bytes = read_bytes(handle, 0, 8);
        uint64_t header_size = little_u64(size_bytes.data());
        require(header_size > 0 && header_size < (1ull << 30),
                "invalid H3 conditioner safetensors header size");
        auto header_bytes = read_bytes(handle, 8, static_cast<size_t>(header_size));
        NSData *json_data = [NSData dataWithBytes:header_bytes.data()
                                            length:header_bytes.size()];
        NSError *error = nil;
        id parsed = [NSJSONSerialization JSONObjectWithData:json_data options:0 error:&error];
        require([parsed isKindOfClass:NSDictionary.class],
                "invalid H3 conditioner safetensors header JSON");
        Header result;
        result.data_offset = 8 + header_size;
        for (NSString *raw_key in (NSDictionary *)parsed) {
            if ([raw_key isEqualToString:@"__metadata__"])
                continue;
            NSDictionary *record = parsed[raw_key];
            require([record isKindOfClass:NSDictionary.class],
                    "invalid H3 conditioner tensor record");
            NSArray *shape = record[@"shape"];
            NSArray *offsets = record[@"data_offsets"];
            NSString *dtype = record[@"dtype"];
            require([shape isKindOfClass:NSArray.class] &&
                        [offsets isKindOfClass:NSArray.class] && offsets.count == 2 &&
                        [dtype isKindOfClass:NSString.class],
                    "invalid H3 conditioner tensor metadata");
            TensorInfo info;
            info.path = path;
            info.dtype = dtype.UTF8String;
            for (NSNumber *dimension in shape) {
                require([dimension isKindOfClass:NSNumber.class] &&
                            dimension.unsignedLongLongValue <= INT32_MAX,
                        "invalid H3 conditioner tensor shape");
                info.shape.push_back(static_cast<int>(dimension.intValue));
            }
            info.begin = [offsets[0] unsignedLongLongValue];
            info.end = [offsets[1] unsignedLongLongValue];
            require(info.end >= info.begin,
                    "invalid H3 conditioner tensor offsets");
            result.tensors.emplace(raw_key.UTF8String, std::move(info));
        }
        [handle closeFile];
        return result;
    } @catch (NSException *exception) {
        [handle closeFile];
        throw std::runtime_error(exception.reason.UTF8String ?: "H3 conditioner shard read failed");
    }
}

mx::Dtype dtype_for(const std::string &dtype) {
    if (dtype == "BF16") return mx::bfloat16;
    if (dtype == "F16") return mx::float16;
    if (dtype == "F32") return mx::float32;
    throw std::invalid_argument("unsupported H3 conditioner tensor dtype: " + dtype);
}

size_t dtype_size(const std::string &dtype) {
    if (dtype == "BF16" || dtype == "F16") return 2;
    if (dtype == "F32") return 4;
    throw std::invalid_argument("unsupported H3 conditioner tensor dtype: " + dtype);
}

size_t element_count(const std::vector<int> &shape) {
    size_t result = 1;
    for (int dimension : shape) {
        require(dimension > 0 && result <= SIZE_MAX / size_t(dimension),
                "H3 conditioner tensor shape overflows");
        result *= size_t(dimension);
    }
    return result;
}

Tensor make_tensor(std::vector<uint8_t> bytes, const TensorInfo &info) {
    require(bytes.size() == element_count(info.shape) * dtype_size(info.dtype),
            "H3 conditioner tensor byte size mismatch");
    auto owner = std::make_shared<std::vector<uint8_t>>(std::move(bytes));
    mx::Shape shape;
    for (int dimension : info.shape) shape.push_back(dimension);
    return Tensor(owner->data(), std::move(shape), dtype_for(info.dtype),
                  [owner](void *) {});
}

Tensor conditioner_fp32(Tensor value) {
    if (value.dtype() == mx::float32)
        return value;
    // FastVideo evaluates streamed BF16/F16 weights into FP32 before building
    // each layer graph. Keep the same accumulation contract and release the
    // source-backed temporary before loading the next projection.
    value = mx::astype(value, mx::float32);
    mx::eval(value);
    return value;
}
} // namespace

ConditionerConfig load_conditioner_config(const std::filesystem::path &path) {
    auto root = json_object_at_path(path);
    NSDictionary *text = root[@"text_config"];
    if (![text isKindOfClass:NSDictionary.class]) text = root;
    NSDictionary *rope = text[@"rope_scaling"];
    if (![rope isKindOfClass:NSDictionary.class]) rope = @{};
    NSArray *sections = rope[@"mrope_section"];
    if (![sections isKindOfClass:NSArray.class]) sections = @[@24, @20, @20];
    require(sections.count == 3, "H3 conditioner mRoPE requires three sections");
    ConditionerConfig config;
    config.hidden_size = json_integer(text, @"hidden_size");
    config.num_layers = json_integer(text, @"num_hidden_layers");
    config.num_heads = json_integer(text, @"num_attention_heads");
    config.kv_heads = json_integer(text, @"num_key_value_heads");
    config.head_dim = json_integer(text, @"head_dim", config.hidden_size / config.num_heads);
    config.intermediate_size = json_integer(text, @"intermediate_size");
    config.vocabulary_size = json_integer(text, @"vocab_size", 151936);
    config.norm_epsilon = static_cast<float>(json_number(text, @"rms_norm_eps"));
    config.rope_theta = static_cast<float>(json_number(text, @"rope_theta"));
    for (NSUInteger index = 0; index < 3; ++index)
        config.mrope_sections[index] = [sections[index] intValue];
    return config;
}

struct ShardIndex::Impl {
    std::map<std::string, TensorInfo> infos;
    mutable std::map<std::string, Header> headers;

    explicit Impl(const std::filesystem::path &root) {
        require(std::filesystem::is_directory(root),
                "H3 conditioner directory is missing: " + root.string());
        auto index_path = root / "model.safetensors.index.json";
        if (std::filesystem::is_regular_file(index_path)) {
            auto index = json_object_at_path(index_path);
            NSDictionary *map = index[@"weight_map"];
            require([map isKindOfClass:NSDictionary.class],
                    "H3 conditioner safetensors index has no weight_map");
            for (NSString *key in map) {
                NSString *shard = map[key];
                require([shard isKindOfClass:NSString.class],
                        "invalid H3 conditioner shard name");
                TensorInfo info;
                info.path = (root / shard.UTF8String).string();
                infos.emplace(key.UTF8String, std::move(info));
            }
        } else {
            auto single = root / "model.safetensors";
            require(std::filesystem::is_regular_file(single),
                    "H3 conditioner safetensors index and single file are missing");
            auto header = read_header(single.string());
            for (const auto &[key, info] : header.tensors)
                infos.emplace(key, info);
            headers.emplace(single.string(), std::move(header));
        }
        require(!infos.empty(), "H3 conditioner has no indexed tensors");
    }

    const TensorInfo &info(const std::string &key) const {
        auto found = infos.find(key);
        require(found != infos.end(), "missing H3 conditioner tensor: " + key);
        if (found->second.begin == found->second.end &&
            found->second.shape.empty()) {
            auto header = headers.find(found->second.path);
            if (header == headers.end())
                header = headers.emplace(found->second.path,
                                         read_header(found->second.path)).first;
            auto tensor = header->second.tensors.find(key);
            require(tensor != header->second.tensors.end(),
                    "indexed H3 conditioner tensor is absent from shard: " + key);
            found = infos.find(key);
            const_cast<TensorInfo &>(found->second) = tensor->second;
        }
        return found->second;
    }

    Header &header_for(const std::string &path) const {
        auto found = headers.find(path);
        if (found == headers.end())
            found = headers.emplace(path, read_header(path)).first;
        return found->second;
    }

    Tensor load(const std::string &key) const {
        const auto &metadata = info(key);
        NSFileHandle *handle = [NSFileHandle fileHandleForReadingAtPath:@(metadata.path.c_str())];
        require(handle != nil, "cannot open H3 conditioner shard: " + metadata.path);
        auto bytes = read_bytes(handle, header_for(metadata.path).data_offset + metadata.begin,
                                static_cast<size_t>(metadata.end - metadata.begin));
        [handle closeFile];
        return conditioner_fp32(make_tensor(std::move(bytes), metadata));
    }

    Tensor load_rows(const std::string &key, const std::vector<int> &rows) const {
        const auto &metadata = info(key);
        require(metadata.shape.size() == 2, "H3 conditioner row gather requires rank-2 tensor");
        const int row_count = metadata.shape[0];
        const size_t row_bytes = size_t(metadata.shape[1]) * dtype_size(metadata.dtype);
        for (int row : rows)
            require(row >= 0 && row < row_count, "H3 conditioner token is out of range");
        NSFileHandle *handle = [NSFileHandle fileHandleForReadingAtPath:@(metadata.path.c_str())];
        require(handle != nil, "cannot open H3 conditioner shard: " + metadata.path);
        std::vector<uint8_t> bytes;
        bytes.reserve(row_bytes * rows.size());
        for (int row : rows) {
            auto part = read_bytes(handle,
                header_for(metadata.path).data_offset + metadata.begin + row_bytes * size_t(row),
                row_bytes);
            bytes.insert(bytes.end(), part.begin(), part.end());
        }
        [handle closeFile];
        TensorInfo gathered = metadata;
        gathered.shape[0] = static_cast<int>(rows.size());
        return conditioner_fp32(make_tensor(std::move(bytes), gathered));
    }
};

ShardIndex::ShardIndex(const std::filesystem::path &root)
    : impl_(std::make_unique<Impl>(root)) {}
ShardIndex::~ShardIndex() = default;
Tensor ShardIndex::tensor(const std::string &key) const { return impl_->load(key); }
Tensor ShardIndex::rows(const std::string &key, const std::vector<int> &rows) const {
    return impl_->load_rows(key, rows);
}
bool ShardIndex::has(const std::string &key) const { return impl_->infos.count(key) != 0; }

} // namespace tc::h3_mlx

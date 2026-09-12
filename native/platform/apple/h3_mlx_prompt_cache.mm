#import <Foundation/Foundation.h>

#include "../../models/h3_mlx/prompt_cache.hpp"
#include "platform.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace tc::h3_mlx {
namespace {
std::string hex_digest(const unsigned char *bytes, size_t count) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(count * 2, '0');
    for (size_t index = 0; index < count; ++index) {
        result[index * 2] = digits[bytes[index] >> 4];
        result[index * 2 + 1] = digits[bytes[index] & 15];
    }
    return result;
}

std::string digest_string(const std::string &value) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    require(CC_SHA256(value.data(), static_cast<CC_LONG>(value.size()), digest) != nullptr,
            "cannot hash H3 prompt cache identity");
    return hex_digest(digest, sizeof(digest));
}

std::string tree_metadata(const std::filesystem::path &root) {
    require(std::filesystem::is_directory(root),
            "H3 prompt cache identity root is missing: " + root.string());
    std::vector<std::string> entries;
    for (const auto &item : std::filesystem::recursive_directory_iterator(root)) {
        if (!item.is_regular_file() || item.is_symlink()) continue;
        std::error_code error;
        const auto size = item.file_size(error);
        require(!error, "cannot inspect H3 prompt cache file: " + item.path().string());
        const auto mtime = item.last_write_time(error);
        require(!error, "cannot inspect H3 prompt cache timestamp: " + item.path().string());
        entries.push_back(item.path().lexically_relative(root).generic_string() + "\n" +
                          std::to_string(size) + "\n" +
                          std::to_string(static_cast<long long>(
                              mtime.time_since_epoch().count())));
    }
    std::sort(entries.begin(), entries.end());
    std::string result;
    for (const auto &entry : entries) result += entry + "\n";
    return result;
}

NSString *string_value(NSDictionary *object, NSString *key) {
    id value = object[key];
    return [value isKindOfClass:NSString.class] ? value : nil;
}

std::filesystem::path tensor_path(const std::filesystem::path &root,
                                  const std::string &identity) {
    return root / (identity + ".safetensors");
}

std::filesystem::path meta_path(const std::filesystem::path &root,
                                const std::string &identity) {
    return root / (identity + ".json");
}

void atomic_rename(const std::filesystem::path &from,
                   const std::filesystem::path &to) {
    std::error_code error;
    std::filesystem::rename(from, to, error);
    require(!error, "cannot atomically publish H3 prompt cache: " + error.message());
}
} // namespace

std::string prompt_cache_identity(const std::filesystem::path &component_root,
                                  const std::filesystem::path &tokenizer_root,
                                  const std::string &prompt) {
    require(!prompt.empty(), "H3 prompt cache cannot key an empty prompt");
    std::ostringstream canonical;
    canonical << "schema=h3-prompt-cache-v2\n"
              << "encoder=Qwen3-VL:first-50-language-layers\n"
              << "mrope=numpy-float32-cody-waite-fma-v1\n"
              << "component=" << component_root.lexically_normal().string() << "\n"
              << tree_metadata(component_root)
              << "tokenizer=" << tokenizer_root.lexically_normal().string() << "\n"
              << tree_metadata(tokenizer_root)
              << "prompt_bytes=" << prompt.size() << "\n" << prompt;
    return digest_string(canonical.str());
}

std::optional<ConditioningResult> load_prompt_cache(
    const std::filesystem::path &cache_root, const std::string &identity,
    const Event &event, std::atomic<bool> &cancelled) {
    checkpoint(cancelled);
    const auto tensors = tensor_path(cache_root, identity);
    const auto metadata = meta_path(cache_root, identity);
    if (!std::filesystem::is_regular_file(tensors) ||
        !std::filesystem::is_regular_file(metadata))
        return std::nullopt;
    NSData *data = [NSData dataWithContentsOfFile:@(metadata.c_str())];
    if (!data) return std::nullopt;
    NSError *error = nil;
    id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (![parsed isKindOfClass:NSDictionary.class] ||
        ![string_value((NSDictionary *)parsed, @"identity") isEqual:@(identity.c_str())] ||
        [((NSDictionary *)parsed)[@"schema"] integerValue] != 1)
        return std::nullopt;
    try {
        auto arrays = mx::load_safetensors(tensors.string()).first;
        auto hidden = arrays.find("hidden_states");
        auto tags = arrays.find("token_tags");
        require(hidden != arrays.end() && tags != arrays.end(),
                "H3 prompt cache tensors are incomplete");
        require(hidden->second.ndim() == 2 && hidden->second.dtype() == mx::float32 &&
                    tags->second.ndim() == 1 && tags->second.dtype() == mx::int32 &&
                    hidden->second.shape(0) == tags->second.shape(0),
                "H3 prompt cache tensor geometry is invalid");
        std::vector<int32_t> token_tags(static_cast<size_t>(tags->second.shape(0)));
        mx::eval(hidden->second, tags->second);
        std::memcpy(token_tags.data(), tags->second.data<int32_t>(),
                    token_tags.size() * sizeof(int32_t));
        event("h3_mlx_prompt_cache_hit", 1, 1);
        return ConditioningResult{std::move(hidden->second), std::move(token_tags),
                                  static_cast<int>(tags->second.shape(0))};
    } catch (...) {
        // Corrupt/incomplete cache entries are not fatal to generation.  Do
        // not leave a repeatedly failing entry in the cache.
        std::error_code ignored;
        std::filesystem::remove(tensors, ignored);
        std::filesystem::remove(metadata, ignored);
        return std::nullopt;
    }
}

void save_prompt_cache(const std::filesystem::path &cache_root,
                       const std::string &identity,
                       const ConditioningResult &result, const Event &event,
                       std::atomic<bool> &cancelled) {
    checkpoint(cancelled);
    require(result.hidden_states.ndim() == 2 && result.hidden_states.dtype() == mx::float32 &&
                result.hidden_states.shape(0) == static_cast<int>(result.token_tags.size()) &&
                result.tokens == static_cast<int>(result.token_tags.size()),
            "invalid H3 conditioning result for cache");
    std::filesystem::create_directories(cache_root);
    const auto suffix = ".tmp." + std::to_string(getpid()) + "." +
                        NSUUID.UUID.UUIDString.UTF8String;
    const auto final_tensors = tensor_path(cache_root, identity);
    const auto final_metadata = meta_path(cache_root, identity);
    // MLX appends `.safetensors` when the supplied path lacks that suffix, so
    // keep the format suffix last on staging files as well.
    const auto temp_tensors = cache_root / (identity + suffix + ".safetensors");
    const auto temp_metadata = cache_root / (identity + suffix + ".json");
    try {
        NSError *error = nil;
        auto tag_tensor = Tensor(result.token_tags.data(),
                                 {static_cast<int>(result.token_tags.size())}, mx::int32);
        mx::eval(result.hidden_states, tag_tensor);
        mx::save_safetensors(temp_tensors,
                             {{"hidden_states", result.hidden_states},
                              {"token_tags", tag_tensor}});
        NSDictionary *metadata = @{
            @"schema": @1,
            @"identity": @(identity.c_str()),
            @"tokens": @(result.tokens),
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:metadata options:0 error:&error];
        require(json != nil, "cannot serialize H3 prompt cache metadata");
        require([json writeToFile:@(temp_metadata.c_str()) options:NSDataWritingAtomic error:&error],
                "cannot write H3 prompt cache metadata");
        atomic_rename(temp_tensors, final_tensors);
        atomic_rename(temp_metadata, final_metadata);
        event("h3_mlx_prompt_cache_store", 1, 1);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temp_tensors, ignored);
        std::filesystem::remove(temp_metadata, ignored);
        throw;
    }
}

} // namespace tc::h3_mlx

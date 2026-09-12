#pragma once

#include "conditioner.hpp"

namespace tc::h3_mlx {

// The cache key is content/identity bound.  The implementation deliberately
// fingerprints metadata for large safetensors shards instead of reading tens
// of gigabytes just to calculate a cache key.
std::string prompt_cache_identity(const std::filesystem::path &component_root,
                                  const std::filesystem::path &tokenizer_root,
                                  const std::string &prompt);

std::optional<ConditioningResult> load_prompt_cache(
    const std::filesystem::path &cache_root, const std::string &identity,
    const Event &, std::atomic<bool> &);

void save_prompt_cache(const std::filesystem::path &cache_root,
                       const std::string &identity,
                       const ConditioningResult &, const Event &,
                       std::atomic<bool> &);

} // namespace tc::h3_mlx

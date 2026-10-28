#pragma once
#include "model.hpp"
#include <charconv>
#include <cstdlib>
#include <cstring>

namespace tc::seedvr2 {
inline size_t allocator_cache_limit() {
    size_t bytes = 4ull << 30;
    if (const char *value = std::getenv("TURBOCIDER_SEEDVR2_ALLOCATOR_CACHE_BYTES")) {
        const char *end = value + std::strlen(value);
        const auto parsed = std::from_chars(value, end, bytes);
        require(parsed.ec == std::errc{} && parsed.ptr == end && bytes <= (16ull << 30),
                "SeedVR2 allocator cache bytes must be an integer in 0...17179869184");
    }
    return bytes;
}
// Native requests hold the global GPU execution lock. Do not leave temporary
// VAE/DiT buffers in MLX's default, device-sized cache after a request ends.
class AllocatorCacheScope {
    size_t previous_;
  public:
    explicit AllocatorCacheScope(size_t bytes) : previous_(mx::set_cache_limit(bytes)) {
        mx::clear_cache();
    }
    AllocatorCacheScope(const AllocatorCacheScope &) = delete;
    AllocatorCacheScope &operator=(const AllocatorCacheScope &) = delete;
    ~AllocatorCacheScope() {
        mx::clear_cache();
        mx::set_cache_limit(previous_);
    }
};
} // namespace tc::seedvr2

#pragma once

#include "../core/contracts.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace tc {

/* Content identity for a separately supplied adapter after the platform
 * boundary has verified the file.  Model sessions use the same material for
 * resident-cache keys so path aliases, role changes, strength changes, and
 * same-sized file replacements cannot reuse incompatible prepared weights. */
struct VerifiedLoRAIdentity {
    std::filesystem::path path;
    uintmax_t bytes = 0;
    std::string sha256;
    std::string role;
    float strength = 1.0f;

    std::string cache_key(std::string_view algorithm) const;
};

VerifiedLoRAIdentity make_verified_lora_identity(
    const LoRAAsset& request,
    const std::filesystem::path& resolved_path,
    uintmax_t bytes,
    std::string sha256);

}  // namespace tc

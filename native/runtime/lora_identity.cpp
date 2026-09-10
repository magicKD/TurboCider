#include "lora_identity.hpp"
#include "../core/common.hpp"

#include <bit>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace tc {
namespace {

bool valid_sha256(std::string_view value) {
    if (value.size() != 64) return false;
    for (char character : value)
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f')))
            return false;
    return true;
}

void append_field(std::ostringstream& output, std::string_view name,
                  std::string_view value) {
    output << '|' << name << '=' << value.size() << ':' << value;
}

}  // namespace

std::string VerifiedLoRAIdentity::cache_key(
        std::string_view algorithm) const {
    require(!algorithm.empty(), "LoRA cache algorithm is empty");
    require(!path.empty(), "LoRA cache identity path is empty");
    require(bytes > 0, "LoRA cache identity size is empty");
    require(valid_sha256(sha256), "LoRA cache identity SHA-256 is invalid");
    require(!role.empty(), "LoRA cache identity role is empty");
    require(std::isfinite(strength) && strength > 0.0f,
            "LoRA cache identity strength is invalid");
    std::ostringstream output;
    output << "verified-lora-v1";
    append_field(output, "algorithm", algorithm);
    append_field(output, "path", path.string());
    output << "|bytes=" << bytes;
    append_field(output, "sha256", sha256);
    append_field(output, "role", role);
    output << "|strength_bits=" << std::hex << std::setw(8)
           << std::setfill('0') << std::bit_cast<uint32_t>(strength);
    return output.str();
}

VerifiedLoRAIdentity make_verified_lora_identity(
        const LoRAAsset& request,
        const std::filesystem::path& resolved_path,
        uintmax_t bytes,
        std::string sha256) {
    require(!resolved_path.empty(), "verified LoRA path is empty");
    require(bytes > 0, "verified LoRA file is empty");
    require(valid_sha256(sha256), "verified LoRA SHA-256 is invalid");
    require(!request.role.empty(), "verified LoRA role is empty");
    require(std::isfinite(request.strength) && request.strength > 0.0f,
            "verified LoRA strength is invalid");
    auto absolute_path = std::filesystem::absolute(resolved_path).lexically_normal();
    return {std::move(absolute_path), bytes, std::move(sha256),
            request.role, request.strength};
}

}  // namespace tc

#pragma once
#include <string_view>

namespace tc::qwen21 {
struct ViggleAdapter {
    std::string_view filename;
    std::string_view sha256;
    std::string_view rank;
};

// The two released v0.2.1 students default to six nodes and alpha/r=1.
// Pin both identities; a filename alone never establishes adapter provenance.
inline constexpr ViggleAdapter viggle_v021_adapters[] = {
    {"Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
     "bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b", "r128"},
    {"Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",
     "2a0148f5c73abbed5f97da5ea356e439318aadb281d01fce4af39cdf43728803", "r256"},
};

inline const ViggleAdapter *viggle_v021_adapter(std::string_view filename) {
    for (const auto &adapter : viggle_v021_adapters)
        if (adapter.filename == filename) return &adapter;
    return nullptr;
}

inline std::string_view viggle_v021_sampler_identity(int steps) {
    return steps == 6 ? "viggle-v021-six-node-qwen-shift-no-terminal-v1" :
           steps == 5 ? "viggle-v021-five-node-qwen-shift-no-terminal-v1" : "";
}
} // namespace tc::qwen21

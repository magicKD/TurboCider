#include "../../native/models/qwen21/viggle_adapter.hpp"
#include <cassert>
#include <string>

int main() {
    using tc::qwen21::viggle_v021_adapter;
    const auto *small = viggle_v021_adapter(
        "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors");
    const auto *large = viggle_v021_adapter(
        "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors");
    assert(small && large && small != large);
    assert(small->rank == "r128" && large->rank == "r256");
    // These are release identities, independent of the lookup implementation.
    // Accidentally assigning the other rank's hash must fail provenance checks.
    assert(small->sha256 ==
           "bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b");
    assert(large->sha256 ==
           "2a0148f5c73abbed5f97da5ea356e439318aadb281d01fce4af39cdf43728803");
    assert(small->sha256 != large->sha256);
    assert(!viggle_v021_adapter(""));
    assert(!viggle_v021_adapter("unrelated-r128.safetensors"));
    assert(!viggle_v021_adapter(
        "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r64.safetensors"));
    assert(!viggle_v021_adapter(std::string(small->filename) + ".bak"));
    assert(!viggle_v021_adapter(std::string(small->filename) + "\n"));
}

#pragma once
#include "model.hpp"
#include "prefix_buffers.hpp"

namespace tc::seedvr2 {
struct HybridConfig {
    int rows = 0, channels = 0;
    std::vector<int> blocks;
    std::vector<std::filesystem::path> artifacts;
    std::string manifest_sha256, source_operand_recipe;
};

// Original-FP16, independent-output-backing FFN prefixes. This low-level
// implementation does not grant public qualification to an arbitrary bank.
class HybridFFN {
    const DiT &dit_;
    HybridConfig config_;
    std::atomic<bool> &cancelled_;
    PrefixBuffers buffered_;
    std::array<int,32> slots_{};
    uint64_t hybrid_blocks_ = 0, gpu_blocks_ = 0, batches_ = 0;
    double prediction_seconds_ = 0;
    bool cpu_only_control_ = false;
  public:
    HybridFFN(const DiT &, HybridConfig, const Event &, std::atomic<bool> &, bool cpu_only_control = false);
    Tensor predict(int block, const Tensor &input);
    uint64_t calls() const { return buffered_.calls(); }
    uint64_t copied_bytes() const { return buffered_.copied_bytes(); }
    uint64_t hybrid_blocks() const { return hybrid_blocks_; }
    uint64_t gpu_blocks() const { return gpu_blocks_; }
    uint64_t batches() const { return batches_; }
    double prediction_seconds() const { return prediction_seconds_; }
    const char *schedule() const { return "buffered_suffix"; }
    const char *prefix_compute_units() const { return cpu_only_control_ ? "cpuOnly" : "cpuAndNeuralEngine"; }
    uint64_t snapshot_bytes() const { return 0; }
    size_t prefix_backing_slots() const { return buffered_.backing_slots(); }
    size_t prefix_backing_bytes() const { return buffered_.backing_bytes(); }
    const HybridConfig &config() const { return config_; }
};
HybridConfig read_hybrid_config(const std::filesystem::path &);
// Planning checks only the bound small manifest. Full tree/feature ABI
// verification is still mandatory before a runtime partition is created.
void validate_public_hybrid_manifest(const std::filesystem::path &);
} // namespace tc::seedvr2

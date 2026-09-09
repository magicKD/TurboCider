#pragma once
#include "mlx.hpp"

namespace tc {

// Low-level fixed-shape Core ML partitions, independent of any model manifest.
// Each partition uses the x/y FP16 [1,H,1,R] feature ABI and shares one output
// backing. The caller must materialize every consumer before the next predict.
// This set is session-owned and must not be used concurrently.
class CoreMLPartitions {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int rows_, hidden_;

  public:
    CoreMLPartitions(const std::vector<std::filesystem::path> &, int rows, int hidden,
                     const Event &, std::atomic<bool> &);
    ~CoreMLPartitions();
    Tensor predict(int block, const Tensor &input);
    uint64_t calls() const;
    uint64_t copied_bytes() const;
};

} // namespace tc

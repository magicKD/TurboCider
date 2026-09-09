#pragma once

#include "checkpoint.hpp"
#include "../../backends/coreml_partitions.hpp"

namespace tc::wan {

class HybridFFN {
    CoreMLPartitions partitions_;
    std::vector<std::function<std::vector<Tensor>(const std::vector<Tensor> &)>> gpu_;
    std::function<std::vector<Tensor>(const std::vector<Tensor> &)> join_;
    int rows_;

  public:
    HybridFFN(const Checkpoint &, const std::vector<std::filesystem::path> &artifacts,
              int rows, int ane_width, const Event &, std::atomic<bool> &);
    Tensor predict(int block, const Tensor &input, const Tensor &residual, const Tensor &gate);
    uint64_t calls() const { return partitions_.calls(); }
    uint64_t copied_bytes() const { return partitions_.copied_bytes(); }
};

std::unique_ptr<HybridFFN> load_hybrid(const std::filesystem::path &manifest,
                                     const std::filesystem::path &model_root,
                                     const Checkpoint &, const Event &, std::atomic<bool> &);

} // namespace tc::wan

#pragma once

#include "ane_runtime.hpp"
#include "ane_memory.hpp"
#include "ane_qkv_scheduler.hpp"
#include "mlx.hpp"
#include "../runtime/session.hpp"

namespace tc::ane {

// One optional, request-owned Qwen Q/K/V MatMul graph. This deliberately does
// not share a Core ML worker or memory budget with the FFN runtime.
class HybridQkv {
  public:
    using Gpu = std::function<Tensor(int, const Tensor &)>;
    HybridQkv(const std::filesystem::path &manifest, size_t budget,
              std::atomic<bool> &cancelled);
    ~HybridQkv();
    void begin_request();
    QkvScheduler::Plan plan_block(int layer, int rows);
    void observe_block(int layer, int rows, double wall);
    void stage(int layer, int rows, std::vector<Tensor> parts);
    Tensor run(int layer, const Tensor &input, const Gpu &gpu,
               std::atomic<bool> &cancelled);
    void drain();
    bool available() const { return graph_ && !metrics_.failed; }
    bool retains_resources() const {
        return graph_ || !weights_.empty() || output_.capacity();
    }
    const tc::QkvMetrics &metrics() const { return metrics_; }

  private:
    std::unique_ptr<RuntimeGraph> graph_;
    std::unique_ptr<QkvScheduler> scheduler_;
    std::vector<Tensor> weights_;
    std::vector<uint16_t> output_;
    tc::QkvMetrics metrics_;
    size_t budget_;
    int layer_ = -1, rows_ = 0, chunks_ = 0;
    std::optional<QkvScheduler::Plan> block_plan_;
    bool pending_ = false;
    void fail(std::string reason, int layer);
    bool admit_output(int ane_rows);
};

} // namespace tc::ane

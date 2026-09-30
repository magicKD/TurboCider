#pragma once

#include "ane_runtime.hpp"
#include "ane_memory.hpp"
#include "ane_scheduler.hpp"
#include "mlx.hpp"
#include "../runtime/session.hpp"

namespace tc::ane {

// Explicit representation, never inferred from checkpoint filenames. Packed
// matrices use MLX affine uint32 codes; scales/offsets stay owned until drain.
struct FfnWeight {
    Tensor values;
    std::optional<Tensor> scales, offsets;
    int group_size = 32, bits = 4;
};

// Explicit runtime-weight FFN route. The caller supplies its optimized GPU
// implementation; the backend never substitutes a different GPU baseline.
// No adapter or model weights are embedded in the Core ML graph.
class HybridFfn {
  public:
    using Gpu = std::function<Tensor(const Tensor &)>;
    struct Adapter {
        std::function<std::pair<Tensor, Tensor>(const Tensor &)> gate_up;
        // Complete the ANE tail from corrected/restored hidden and base-down
        // output. Keeping the add inside the callback lets a family's GPU
        // graph fuse low-rank output handling without merging any weights.
        // Preserve the delta's existing rounding boundary, return base dtype.
        std::function<Tensor(const Tensor &, const Tensor &)> down_and_add;
    };
    HybridFfn(const std::filesystem::path &manifest, int hidden, int width,
              size_t memory_budget, std::atomic<bool> &cancelled, bool require_lora_inputs = false);
    ~HybridFfn();
    // Observation override is for deterministic host tests; production callers
    // use an owner-thread Mach observation on every resident request.
    void begin_request(const std::string &adapter_identity = {},
                       std::optional<MemoryObservation> observation = std::nullopt);
    // Optional early decision, before the family chooses its compiled block.
    // Hybrid/HybridUntimed/SplitProbe: stage/run once. HybridUntimed owns its
    // completed output and ends the plan in run(), with no observe callback.
    // GpuProbe: run the complete GPU block
    // without the bridge. Gpu: ordinary unsplit GPU, no timing fence/sample.
    // For measured plans, exclude earlier GPU work BEFORE starting the block
    // clock, then observe_block after its residual output has been evaluated.
    RowScheduler::Plan plan_block(int layer, int rows);
    void observe_block(int layer, int rows, double seconds);
    // Called BEFORE attention submission. Own references until staging joins.
    void stage(int layer, int rows, std::vector<Tensor> weights);
    void stage_weights(int layer, int rows, std::vector<FfnWeight> weights);
    Tensor run(int layer, const Tensor &input, const Gpu &gpu,
               std::atomic<bool> &cancelled, const Adapter *adapter = nullptr);
    void drain();
    HybridMetrics metrics() const;
    const std::string &reason() const { return reason_; }
    bool available() const { return graph_ && !failed_; }
    // Current ownership, unlike the historical slot-byte metrics. A failed
    // optional route must release all of these before returning GPU output.
    bool retains_resources() const {
        return graph_ || !weights_.empty() || output_.capacity() || hidden_.capacity();
    }
    bool supports_lora_inputs() const { return graph_ && graph_->shape().lora_inputs; }
  private:
    std::unique_ptr<RuntimeGraph> graph_;
    std::unique_ptr<RowScheduler> scheduler_;
    std::vector<Tensor> weights_;
    // Worker scratch only. Copy completed results into independently owned
    // tensors before publishing them to GPU consumers or adapter callbacks.
    std::vector<uint16_t> output_, hidden_;
    HybridMetrics metrics_;
    bool failed_ = false, pending_ = false;
    bool planned_ = false;
    std::optional<RowScheduler::Plan> block_plan_;
    bool block_sample_valid_ = false;
    double block_gpu_seconds_ = 0, block_ane_seconds_ = 0;
    int layer_ = -1, rows_ = 0, chunks_ = 0;
    bool profile_ = false;
    size_t memory_budget_ = 0;
    std::string reason_;
    std::string adapter_identity_;
    std::chrono::steady_clock::time_point pre_start_;
    void degrade(const std::string &, int layer);
    void release_for_memory(const std::string &reason);
    bool admit_scratch(int ane_rows, bool adapter);
};

} // namespace tc::ane

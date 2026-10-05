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
    enum class Transform { None, ComfyH256Inverse };
    Tensor values;
    std::optional<Tensor> scales, offsets;
    int group_size = 32, bits = 4;
    Transform transform = Transform::None;
};

// Explicit runtime-weight FFN route. The caller supplies its optimized GPU
// implementation; the backend never substitutes a different GPU baseline.
// No adapter or model weights are embedded in the Core ML graph.
class HybridFfn {
  public:
    using Gpu = std::function<Tensor(const Tensor &)>;
    // Base-down partial and corrected hidden for a logical channel range.
    // Do NOT apply down-LoRA here: it must consume joined hidden exactly once.
    using ChannelGpu = std::function<std::pair<Tensor, Tensor>(const Tensor &, int, int)>;
    using NextWeights = std::function<std::vector<FfnWeight>(int)>;
    struct Adapter {
        std::function<std::pair<Tensor, Tensor>(const Tensor &)> gate_up;
        // Complete the ANE tail from corrected/restored hidden and base-down
        // output. Keeping the add inside the callback lets a family's GPU
        // graph fuse low-rank output handling without merging any weights.
        // Preserve the delta's existing rounding boundary, return base dtype.
        std::function<Tensor(const Tensor &, const Tensor &)> down_and_add;
        // Optional channel-only correction: exactly [first, first+count) of
        // each logical gate/up half. Row executors keep the full callback;
        // channel callers without this extension remain compatible.
        std::function<std::pair<Tensor, Tensor>(const Tensor &, int, int)> gate_up_channels = {};
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
               std::atomic<bool> &cancelled, const Adapter *adapter = nullptr,
               const ChannelGpu &channel_gpu = {}, const NextWeights &next_weights = {});
    void drain(bool discard_future = true);
    HybridMetrics metrics() const;
    const std::string &reason() const { return reason_; }
    bool available() const { return graph_ && !failed_; }
    bool supports_lora_inputs() const { return graph_ && graph_->shape().lora_inputs; }
    bool channel_split() const { return axis_ == PartitionAxis::IntermediateChannels; }
    int gpu_channels() const { return metrics_.runtime_weight_gpu_channels; }
    int ane_channels() const { return metrics_.runtime_weight_ane_channels; }
    static std::string executor_configuration_identity();
    std::string backend_label(bool gguf = false) const {
        if (metrics_.runtime_weight_backend.empty()) return gguf ? "mlx_cpp_metal_gguf" : "mlx_cpp_metal";
        return std::string(gguf ? "mlx_cpp_metal_gguf+" : "mlx_cpp_metal+") +
            (metrics_.runtime_weight_backend == "private_ane" ? "private_ane_runtime_weight_experimental" : "coreml_runtime_weight");
    }
    std::string precision_label(bool gguf = false) const {
        if (metrics_.runtime_weight_backend.empty()) return gguf ? "gguf_native_gpu" : "bf16";
        const bool w8 = metrics_.runtime_weight_data_path == "w8a8_hadamard";
        return std::string(gguf ? "gguf_native_gpu+" : "bf16_gpu+") +
            (w8 ? "runtime_w8a8_ffn" : "runtime_fp16_ffn") + (gguf ? "" : "_bf16_io");
    }
    std::string selection_label() const {
        if (metrics_.runtime_weight_backend.empty())
            return "gpu: runtime-weight executor unavailable; full GPU FFN fallback";
        return std::string("gpu_ane runtime-weight ") +
            (channel_split() ? "intermediate-channel" : "token-row") + " FFN contract (" +
            metrics_.runtime_weight_data_path + "); base-only weight slots with optional GPU LoRA activation corrections; physical placement unverified";
    }
    std::string resolve_selection(const std::string &requested) const {
        constexpr std::string_view end = "physical placement unverified";
        const auto at = requested.find(end);
        return selection_label() + (at == std::string::npos ? std::string{} : requested.substr(at + end.size()));
    }
  private:
    std::unique_ptr<Executor> graph_;
    std::unique_ptr<RowScheduler> scheduler_;
    PartitionAxis axis_ = PartitionAxis::Rows;
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
    bool prefetch_ = false;
    bool lora_channel_range_ = true;
    bool fixed_async_ = false;
    int prefetched_layer_ = -1;
    size_t memory_budget_ = 0;
    std::string reason_;
    std::string adapter_identity_;
    std::chrono::steady_clock::time_point pre_start_;
    void degrade(const std::string &, int layer);
    void release_for_memory(const std::string &reason);
    bool admit_scratch(int ane_rows, bool adapter);
    Tensor run_channels(int, const Tensor &, const Gpu &, const ChannelGpu &,
                        std::atomic<bool> &, const Adapter *, const NextWeights &);
    std::vector<DeviceWeightRegion> device_regions(const std::vector<FfnWeight> &) const;
    void maybe_prefetch(int, int, const NextWeights &);
};

} // namespace tc::ane

#pragma once

#include "ane_runtime.hpp"
#include "ane_memory.hpp"
#include "ane_scheduler.hpp"
#include "ane_channel_selection.hpp"
#include "ane_row_window.hpp"
#include "mlx.hpp"
#include "../runtime/session.hpp"
#include <cstdlib>

namespace tc::ane {

inline bool configured_fp32_channel_join() {
    const char *raw=std::getenv("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN");
    require(!raw || std::string(raw)=="0" || std::string(raw)=="1",
        "runtime ANE F32 channel join requires 0 or 1");
    return raw && std::string(raw)=="1";
}

// Explicit representation, never inferred from checkpoint filenames. Raw GGML
// bytes and MLX affine planes are distinct; all owners stay alive until drain.
struct FfnWeight {
    enum class Transform { None, ComfyH256Inverse };
    struct RawGguf {
        uint32_t type; int columns;
        std::shared_ptr<void> logical_content_identity{};
    };
    Tensor values;
    std::optional<Tensor> scales, offsets;
    int group_size = 32, bits = 4;
    Transform transform = Transform::None;
    std::optional<RawGguf> raw_gguf = std::nullopt;
};

// Explicit runtime-weight FFN route. The caller supplies its optimized GPU
// implementation; the backend never substitutes a different GPU baseline.
// No adapter or model weights are embedded in the Core ML graph.
class HybridFfn {
  public:
    // Declare AFTER the borrowed source owners. A retained executor can
    // outlive those owners, including when attention throws after staging.
    // finish() drains and disarms on success; unwinding preserves the primary
    // exception while still waiting for all submitted source readers.
    class SourceScope {
      public:
        explicit SourceScope(HybridFfn *runtime) : runtime_(runtime) {}
        SourceScope(const SourceScope &) = delete;
        SourceScope &operator=(const SourceScope &) = delete;
        ~SourceScope() noexcept { if(runtime_)try { runtime_->drain(); } catch(...) {} }
        void finish() { if(runtime_)runtime_->drain();runtime_=nullptr; }
      private:
        HybridFfn *runtime_;
    };
    using Gpu = std::function<Tensor(const Tensor &)>;
    // Base-down partial and corrected hidden for a logical channel range.
    // Explicit fp32_channel_join() uses an F32 base partial; hidden retains
    // the input dtype. Default callers preserve both original dtypes.
    // Do NOT apply down-LoRA here: it must consume joined hidden exactly once.
    using ChannelGpu = std::function<std::pair<Tensor, Tensor>(const Tensor &, int, int)>;
    using NextWeights = std::function<std::vector<FfnWeight>(int)>;
    struct Adapter {
        struct ChannelDownRanks {
            uint64_t scratch_bytes_per_row = 0;
            // Return lazy GPU-head rank partials only; the wrapper owns their
            // submission/drain. No adapter mutation or hidden-dtype change.
            std::function<std::vector<Tensor>(const Tensor &)> prepare_gpu;
            // Restore the ANE hidden shard, sum rank partials in FP32, then
            // ONE B projection/delta rounding/base add. No full concatenation.
            std::function<Tensor(const Tensor &,const std::vector<Tensor> &,const Tensor &)> finish;
        };
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
        std::optional<ChannelDownRanks> channel_down_ranks = std::nullopt;
    };
    struct CalibrationWorkload {
        std::string model_sha256, adapter_identity, source_generation, encoding, gpu_configuration;
        std::vector<std::weak_ptr<void>> source_owners;
        int rows = 0, layers = 32;
        mx::Dtype dtype = mx::bfloat16;
        NextWeights weights;
        std::function<Tensor(int, const Tensor &)> gpu;
        std::function<std::pair<Tensor, Tensor>(int, const Tensor &, int, int)> channel_gpu;
        // Reconstruct per candidate width before using any range captures.
        std::function<Adapter(int)> adapter;
    };
    HybridFfn(const std::filesystem::path &manifest, int hidden, int width,
              size_t memory_budget, std::atomic<bool> &cancelled, bool require_lora_inputs = false,
              const CalibrationWorkload *calibration = nullptr,
              std::optional<int> calibrated_channels = std::nullopt,
              std::optional<int> channel_override = std::nullopt);
    ~HybridFfn();
    // Observation override is for deterministic host tests; production callers
    // use an owner-thread Mach observation on every resident request.
    void begin_request(const std::string &adapter_identity = {},
                       std::optional<MemoryObservation> observation = std::nullopt);
    // Explicit family policy, not an error fallback or a precision change.
    // Install only while idle; these blocks use the family's complete GPU
    // path without staging or invoking any FFN/LoRA bridge callback.
    void set_gpu_layers(std::vector<int>);
    // Optional early decision, before the family chooses its compiled block.
    // Hybrid/HybridUntimed/SplitProbe: stage/run once. HybridUntimed owns its
    // completed output and ends the plan in run(), with no observe callback.
    // GpuProbe: run the complete GPU block
    // without the bridge. Gpu: ordinary unsplit GPU, no timing fence/sample.
    // For measured plans, exclude earlier GPU work BEFORE starting the block
    // clock, then observe_block after its residual output has been evaluated.
    RowScheduler::Plan plan_block(int layer, int rows,RowPolicy row_policy={});
    void observe_block(int layer, int rows, double seconds);
    // Called BEFORE attention submission. Own references until staging joins.
    void stage(int layer, int rows, std::vector<Tensor> weights);
    void stage_weights(int layer, int rows, std::vector<FfnWeight> weights);
    // Model-supplied source acquisition failed before an executor could stage.
    // Match stage_weights' sticky fallback/metrics and end the planned block.
    void fail_staging(int layer,int rows,const std::string &reason);
    Tensor run(int layer, const Tensor &input, const Gpu &gpu,
               std::atomic<bool> &cancelled, const Adapter *adapter = nullptr,
               const ChannelGpu &channel_gpu = {}, const NextWeights &next_weights = {});
    void drain(bool discard_future = true);
    HybridMetrics metrics() const;
    const std::string &reason() const { return reason_; }
    bool available() const { return graph_ && !failed_; }
    bool usable_configuration() const { return available() || calibration_declined_; }
    bool supports_lora_inputs() const { return graph_ && graph_->shape().lora_inputs; }
    bool channel_split() const { return axis_ == PartitionAxis::IntermediateChannels; }
    int gpu_channels() const { return metrics_.runtime_weight_gpu_channels; }
    int ane_channels() const { return metrics_.runtime_weight_ane_channels; }
    bool fp32_channel_join() const { return fp32_channel_join_; }
    static std::string executor_configuration_identity();
    std::string backend_label(bool gguf = false) const {
        if (metrics_.runtime_weight_backend.empty()) return gguf ? "mlx_cpp_metal_gguf" : "mlx_cpp_metal";
        return std::string(gguf ? "mlx_cpp_metal_gguf+" : "mlx_cpp_metal+") +
            (metrics_.runtime_weight_backend == "private_ane" ? "private_ane_runtime_weight_experimental" : "coreml_runtime_weight");
    }
    std::string precision_label(bool gguf = false) const {
        if (metrics_.runtime_weight_backend.empty()) return gguf ? "gguf_native_gpu" : "bf16";
        const bool comfy = metrics_.runtime_weight_data_path == "w8a8_convrot";
        const bool w8 = metrics_.runtime_weight_data_path == "w8a8_hadamard" || comfy;
        return std::string(gguf ? "gguf_native_gpu+" : "bf16_gpu+") +
            (comfy ? "runtime_convrot_w8a8_ffn" : w8 ? "runtime_w8a8_ffn" : "runtime_fp16_ffn") + (gguf ? "" : "_bf16_io");
    }
    std::string selection_label() const {
        const auto calibration = calibration_reason_.empty() ? std::string{} :
            "; native channel auto: " + calibration_reason_;
        if (metrics_.runtime_weight_backend.empty())
            return (calibration_declined_ ? "gpu: native channel calibration declined hybrid" :
                "gpu: runtime-weight executor unavailable; full GPU FFN fallback") + calibration;
        return std::string("gpu_ane runtime-weight ") +
            (channel_split() ? "intermediate-channel" : "token-row") + " FFN contract (" +
            metrics_.runtime_weight_data_path + "); base-only weight slots with optional GPU LoRA activation corrections; physical placement unverified" + calibration;
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
    RowPolicy row_policy_;
    std::vector<Tensor> weights_;
    // Worker scratch only. Copy completed results into independently owned
    // tensors before publishing them to GPU consumers or adapter callbacks.
    std::vector<uint16_t> output_, hidden_;
    HybridMetrics metrics_;
    bool failed_ = false, pending_ = false;
    bool calibration_declined_ = false;
    std::string calibration_reason_;
    bool planned_ = false;
    std::optional<RowScheduler::Plan> block_plan_;
    bool block_sample_valid_ = false;
    double block_gpu_seconds_ = 0, block_ane_seconds_ = 0;
    int layer_ = -1, rows_ = 0, chunks_ = 0;
    bool profile_ = false;
    bool prefetch_ = false;
    bool prefetch_after_gpu_ = true;
    bool fp32_channel_join_ = false;
    bool lora_channel_range_ = true;
    bool fixed_async_ = false;
    bool defer_channel_join_ = false;
    bool channel_gpu_first_ = false;
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
    static DeviceWeightView calibration_source(const FfnWeight &);
    static ChannelSelection calibrate_channels(const std::filesystem::path &, int, int, size_t,
        std::atomic<bool> &, bool, const CalibrationWorkload &);
};

} // namespace tc::ane

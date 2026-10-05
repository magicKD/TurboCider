#pragma once
#include "../ane_runtime.hpp"

namespace tc::ane {
// W8A8 representation, GPU source staging/A8/FP32 epilogue, two fixed banks.
// Actual ANE INT8 arithmetic is not asserted. Caller owns one host thread.
class PrivateW8Graph final : public Executor {
  public:
    PrivateW8Graph(GraphShape, size_t budget, const std::filesystem::path &cache = {},
                   W8Basis basis = W8Basis::SylvesterDH);
    ~PrivateW8Graph() override;
    BackendKind backend() const override { return BackendKind::PrivateANE; }
    const GraphShape &shape() const override;
    size_t slot_bytes() const override;
    size_t estimated_bytes() const override;
    double load_seconds() const override;
    bool self_test(std::string &) override;
    bool supports_device_io() const override { return true; }
    bool supports_device_weights() const override { return true; }
    bool supports_device_weight_regions() const override { return true; }
    bool supports_weight_prefetch() const override { return true; }
    std::string data_path() const override;
    std::string weight_recipe() const override;
    WeightCacheStats weight_cache_stats() const override;
    StagePipelineStats stage_pipeline_stats() const override;
    bool device_submission_fence_enabled() const override;
    bool activation_lookahead_enabled() const override;
    void stage_weights(std::vector<WeightView>) override;
    void stage_device_weights(std::vector<DeviceWeightView>) override;
    void stage_device_weight_regions(std::vector<DeviceWeightRegion>) override;
    void prefetch_device_weight_regions(std::vector<DeviceWeightRegion>) override;
    std::optional<RunResult> activate_prefetched_weights(std::span<const DeviceWeightRegion>) override;
    void discard_prefetched_weights() override;
    RunResult wait_stage() override;
    void launch(MatrixView, uint16_t *, size_t, DType, std::optional<AdapterInput>) override;
    void launch_device(DeviceMatrixView, DeviceMatrixView, std::optional<DeviceAdapterInput> = std::nullopt) override;
    RunResult finish() override;
    bool supports_fp32_device_output() const override;
    int activation_group_size() const override;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

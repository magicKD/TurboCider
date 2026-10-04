#pragma once
#include "../ane_runtime.hpp"

namespace tc::ane {
// Explicit research backend, native-emitted FP16 micrograph. The same graph,
// source and scheduler contract as Public Core ML; GPU input handoff is still
// host-bridged. W8A8/double-buffer/GPU staging are separate pending profiles.
class PrivateGraph final : public Executor {
  public:
    PrivateGraph(GraphShape shape, size_t memory_budget_bytes,
                 const std::filesystem::path &cache = {});
    ~PrivateGraph() override;
    BackendKind backend() const override { return BackendKind::PrivateANE; }
    const GraphShape &shape() const override;
    size_t slot_bytes() const override;
    size_t estimated_bytes() const override;
    double load_seconds() const override;
    bool self_test(std::string &error) override;
    void stage_weights(std::vector<WeightView> weights) override;
    RunResult wait_stage() override;
    void launch(MatrixView input, uint16_t *output, size_t output_elements,
                DType output_dtype = DType::FP16,
                std::optional<AdapterInput> adapter = std::nullopt) override;
    RunResult finish() override;
    bool supports_device_io() const override;
    void launch_device(DeviceMatrixView input, DeviceMatrixView output,
                       std::optional<DeviceAdapterInput> adapter = std::nullopt) override;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

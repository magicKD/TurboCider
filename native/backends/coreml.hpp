#pragma once
#include "mlx.hpp"
#include "../runtime/session.hpp"
namespace tc {
struct HybridBucketPlan {
    int requested_rows = 0;
    int selected_rows = 0;
    int minimum_profitable_rows = 0;
    int maximum_rows = 0;
    bool supported = false;
    bool flexible = false;
};

// Inspect only the manifest's fixed/enumerated row ABI. Malformed manifests
// throw; a well-formed manifest that is too small returns supported=false so
// callers can take an intentional GPU fallback before loading Core ML models.
HybridBucketPlan hybrid_bucket_plan(const std::filesystem::path &, int tokens);

// Public Core ML only. The framework may execute portions on CPU.
class HybridSession {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    std::string manifest;
    int rows = 0, hidden = 0, block_count = 0;
    int minimum_profitable_rows = 0;
    int mlp_width = 0, ane_mlp_start = 0, ane_mlp_end = 0;
    float output_scale = 1.f;
    bool checkpoint_sha_verified = false;
    bool lora_identity_verified = false;
    double load_seconds = 0;
    double manifest_validation_seconds = 0;
    double output_backing_setup_seconds = 0;
    double zero_input_warmup_seconds = 0;
    uint64_t runtime_failures = 0;
    bool runtime_failed = false;
    int runtime_failure_block = -1;
    uint64_t quality_validation_calls = 0;
    double quality_max_relative_l2 = 0;
    double quality_min_cosine = 1;
    double quality_max_abs = 0;
    double quality_max_relative_abs = 0;
    bool quality_validation_passed = true;
    int prefill_actual_tokens = 0, prefill_selected_bucket = 0,
        prefill_compute_tokens = 0, prefill_padding_tokens = 0;
    bool prefill_fixed_shape = false;
    std::string prefill_plan_reason;
    HybridSession(const std::filesystem::path &, const std::filesystem::path &model, int tokens,
                  const Event &, std::atomic<bool> &, int warmups = 0,
                  const std::filesystem::path &checkpoint = {},
                  const std::vector<LoRAAsset> &loras = {}, int policy_rows = 0,
                  int required_blocks = 0,
                  bool qualified_flexible_backing = false);
    ~HybridSession();
    void set_tokens(int tokens);
    Tensor predict(int block, const Tensor &input);
    bool runtime_available() const { return !runtime_failed; }
    void record_runtime_failure(int block);
    void record_quality(double relative_l2, double cosine, double max_abs,
                        double relative_max_abs, bool passed);
    void record_prefill_plan(int actual_tokens, int selected_bucket,
                             int compute_tokens, int padding_tokens,
                             bool fixed_shape,
                             const std::string &reason);
    HybridMetrics metrics() const;
};
} // namespace tc

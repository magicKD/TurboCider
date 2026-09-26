#pragma once
#include "coreml_generation.hpp"

namespace tc::z_image {
struct HybridPartitionSpec final {
    uint32_t hidden = 3840, mlp_width = 10240, ane_begin = 0, ane_end = 0, bucket_rows = 0;
    float activation_scale = 0, output_scale = 0;
    std::string precision_revision, parent_checkpoint_digest, artifact_content_digest, identity;
};
// Internal source/semantic binding, not release qualification or public route
// authority. Export provenance is a verified metadata claim, not proof that
// arbitrary supplied compiled code computes the claimed parent weights.
class VerifiedCoreMLBundleLease final {
  public:
    static std::shared_ptr<const VerifiedCoreMLBundleLease> bind(
        std::shared_ptr<const CoreMLGeneration>, const std::filesystem::path &relative_manifest,
        std::shared_ptr<const streaming::SourceLease> parent, std::string_view parent_logical_id);
    ~VerifiedCoreMLBundleLease();
    VerifiedCoreMLBundleLease(const VerifiedCoreMLBundleLease &) = delete;
    VerifiedCoreMLBundleLease &operator=(const VerifiedCoreMLBundleLease &) = delete;
    const HybridPartitionSpec &partition() const noexcept;
    const std::vector<std::filesystem::path> &models() const noexcept;
    void revalidate() const;
  private:
    struct State;
    explicit VerifiedCoreMLBundleLease(std::unique_ptr<State>);
    std::unique_ptr<State> state_;
};
} // namespace tc::z_image

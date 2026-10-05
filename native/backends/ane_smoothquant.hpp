#pragma once

#include "ane_calibration.hpp"
#include <optional>

namespace tc::ane::smoothquant {
inline constexpr size_t profile_limit_bytes = 4 * 1024 * 1024;
inline constexpr const char *s1_recipe = "smoothquant-s1-fp32-v1";

// Same canonical/stat encoding and domain as ScopedCapture, with no checkpoint
// contents read. This is a change detector, never content/quality attestation.
std::string checkpoint_fingerprint(const std::filesystem::path &checkpoint);
std::string fingerprint_adapter(const std::string &adapter_identity, size_t index);

struct Binding {
    bool experimental = false; // the caller must explicitly opt in
    std::filesystem::path checkpoint;
    std::string model_id;
    int hidden = 0, width = 512, height = 512, total_steps = 0;
    int reference_size = 0, reference_count = 0;
    std::vector<calibration::LoRAIdentity> loras;
    std::string runtime_recipe, configured_backend;
    std::vector<int> layers = [] {
        std::vector<int> value; for (int i = 0; i < 32; ++i) value.push_back(i); return value;
    }();
    // Empty steps derives {0,total_steps/2,total_steps-1}, like ScopedCapture.
    std::vector<int> steps;
    size_t rows_per_point = 8;
    std::optional<double> alpha;
};
struct LayerScale { int layer = 0; std::vector<float> s1; };

class ExperimentalProfile {
public:
    static ExperimentalProfile load(const std::filesystem::path &path, const Binding &binding);
    const LayerScale *find(int layer) const noexcept;
    std::span<const float> scales_for(int layer) const;
    const std::vector<LayerScale> &layers() const { return layers_; }
    const std::string &content_digest() const { return content_digest_; }
    const std::string &calibration_digest() const { return calibration_digest_; }
    const calibration::Metadata &provenance() const { return provenance_; }
    double alpha() const { return alpha_; }
private:
    std::vector<LayerScale> layers_;
    std::string content_digest_, calibration_digest_;
    calibration::Metadata provenance_;
    double alpha_ = 0;
};
} // namespace tc::ane::smoothquant

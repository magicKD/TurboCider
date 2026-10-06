#pragma once

// SDK/MLX/Private-independent calibration data shared by both backends,
// the runtime result and deterministic host verifiers. Times are seconds.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace tc::ane {
struct CalibrationPoint { double share = 0, gpu = 0, ane = 0, both = 0; };
struct GpuCalibrationSamples {
    std::array<std::vector<double>, 2> seconds; // full GPU one/four
    double layer_seconds = 0;
};
struct ChannelCalibrationSamples {
    std::array<std::array<std::vector<double>, 2>, 3> seconds; // [GPU,ANE,Both][one,four]
    CalibrationPoint point;
    bool prefetch = false;
    uint64_t ane_calls = 0;
    uint64_t correction_computations = 0, correction_uploads = 0;
};

struct ChannelCalibrationIdentity {
    std::string model_sha256, adapter, encoding, precision, backend, recipe;
    std::string soc, os_build, runtime_build, metal_abi, graph_abi, source_generation;
    int rows = 0, hidden = 0, width = 0, tile_k = 0, tile_n = 0;
    bool prefetch = false;
    auto fields() const {
        return std::tie(model_sha256, adapter, encoding, precision, backend, recipe,
            soc, os_build, runtime_build, metal_abi, graph_abi, source_generation,
            rows, hidden, width, tile_k, tile_n, prefetch);
    }
    bool operator<(const ChannelCalibrationIdentity &other) const { return fields() < other.fields(); }
    bool valid() const {
        return model_sha256.size() == 64 && model_sha256.find_first_not_of("0123456789abcdef") == std::string::npos &&
            !encoding.empty() && !precision.empty() && !backend.empty() && !recipe.empty() && !soc.empty() &&
            !os_build.empty() && !runtime_build.empty() && !metal_abi.empty() && !graph_abi.empty() &&
            !source_generation.empty() && rows > 0 && rows <= 4224 && hidden > 0 && hidden <= 4096 &&
            hidden % 128 == 0 && width > 512 && width <= 16384 && width % 512 == 0 && tile_k > 0 && tile_n > 0;
    }
};

struct ChannelTrialEvidence {
    std::vector<double> gpu_seconds, candidate_seconds; // matched complete four-FFN windows; cold excluded
    uint64_t calls = 0, fallbacks = 0, retries = 0;
    double relative_l2 = INFINITY, cosine = -INFINITY;
    bool completed = false;
    bool observe_quality(double delta, double reference_norm, double candidate_norm, double dot, bool finite_values) {
        if (!finite_values || !std::isfinite(delta) || delta < 0 ||
            !std::isfinite(reference_norm) || reference_norm <= 0 ||
            !std::isfinite(candidate_norm) || candidate_norm <= 0 || !std::isfinite(dot)) {
            relative_l2 = INFINITY; cosine = -INFINITY; return false;
        }
        const double error = std::sqrt(delta / reference_norm);
        const double similarity = dot / (std::sqrt(candidate_norm) * std::sqrt(reference_norm));
        if (!std::isfinite(error) || !std::isfinite(similarity)) {
            relative_l2 = INFINITY; cosine = -INFINITY; return false;
        }
        relative_l2 = std::max(relative_l2, error);
        cosine = std::min(cosine, similarity);
        return true;
    }
    bool accepts(double minimum_gain = .05) const {
        if (!completed || !calls || fallbacks || retries || gpu_seconds.size() != candidate_seconds.size() ||
            gpu_seconds.size() < 3 || gpu_seconds.size() > 31 || !(gpu_seconds.size() % 2) ||
            !std::isfinite(relative_l2) || relative_l2 < 0 || relative_l2 > .03 ||
            !std::isfinite(cosine) || cosine < .999 || cosine > 1.000001 ||
            !std::isfinite(minimum_gain) || minimum_gain < 0 || minimum_gain >= 1) return false;
        auto valid = [](const auto &v) {
            return std::all_of(v.begin(), v.end(), [](double s) { return std::isfinite(s) && s > 0; });
        };
        if (!valid(gpu_seconds) || !valid(candidate_seconds)) return false;
        auto gpu = gpu_seconds, candidate = candidate_seconds;
        std::sort(gpu.begin(), gpu.end()); std::sort(candidate.begin(), candidate.end());
        // Require the measured complete window, not only the fitted slope,
        // to earn admission. The E2E/multi-prompt gates remain separate.
        return candidate[candidate.size()/2] <= (1-minimum_gain)*gpu[gpu.size()/2];
    }
};

struct ChannelCalibrationReport {
    int schema_version = 1;
    bool enabled = true, cache_hit = false, trial_passed = false, complete = false;
    bool lora = false;
    int selected_channels = 0, proposed_channels = 0, bucket_rows = 0, layer_count = 0;
    int actual_rows = 0, hidden = 0, width = 0;
    int warmups = 2, repeats = 7;
    std::string status = "pending", reason;
    std::string scope = "complete FFN host spans; not E2E qualification or physical engine trace";
    std::string input_recipe = "synthetic-normal-key17-scale0.25; zero-padded ANE bucket";
    std::vector<int> sampled_depths;
    std::optional<ChannelCalibrationIdentity> identity;
    std::optional<GpuCalibrationSamples> baseline;
    std::vector<ChannelCalibrationSamples> points;
    std::optional<ChannelTrialEvidence> trial;
    double predicted_layer_seconds = 0;
};
} // namespace tc::ane

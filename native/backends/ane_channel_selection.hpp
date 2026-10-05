#pragma once

#include "ane_cost_model.hpp"
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace tc::ane {

// In-process only. Model callers supply the observed file digest; resident
// source allocation IDs/owners additionally bind the actual sampled weights.
// Never import an offline proposal as accepted runtime evidence.
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

struct ChannelSelection {
    int channels = 0;
    bool cache_hit = false, trial_passed = false;
    std::string reason;
};

class ChannelSelectionCache {
    struct Entry { ChannelSelection selection; uint64_t use; std::vector<std::weak_ptr<void>> sources; };
    std::map<ChannelCalibrationIdentity, Entry> entries_;
    uint64_t use_ = 0;
    size_t capacity_;
    std::mutex mutex_;
  public:
    explicit ChannelSelectionCache(size_t capacity = 8) : capacity_(capacity) {
        if (!capacity || capacity > 64) throw std::invalid_argument("channel cache requires bounded capacity");
    }
    std::optional<ChannelSelection> find(const ChannelCalibrationIdentity &key) {
        if (!key.valid()) throw std::invalid_argument("invalid native channel calibration identity");
        std::lock_guard lock(mutex_);
        auto at = entries_.find(key);
        if (at == entries_.end()) return std::nullopt;
        if (std::any_of(at->second.sources.begin(), at->second.sources.end(), [](const auto &source) { return source.expired(); })) {
            entries_.erase(at); return std::nullopt;
        }
        at->second.use = ++use_;
        auto result = at->second.selection; result.cache_hit = true;
        return result;
    }
    void admit(const ChannelCalibrationIdentity &key, ChannelSelection selection,
               const ChannelTrialEvidence &trial, std::vector<std::weak_ptr<void>> sources) {
        if (!key.valid() || selection.channels <= 0 || selection.channels >= key.width || selection.channels % 512 ||
            !selection.trial_passed || !trial.accepts() || sources.empty() ||
            std::any_of(sources.begin(), sources.end(), [](const auto &source) { return source.expired(); }))
            throw std::invalid_argument("channel cache requires an independently accepted complete-runtime candidate");
        std::lock_guard lock(mutex_);
        if (!entries_.contains(key) && entries_.size() == capacity_) {
            auto victim = std::min_element(entries_.begin(), entries_.end(),
                [](const auto &a, const auto &b) { return a.second.use < b.second.use; });
            entries_.erase(victim);
        }
        selection.cache_hit = false;
        entries_.insert_or_assign(key, Entry{std::move(selection), ++use_, std::move(sources)});
    }
};
} // namespace tc::ane

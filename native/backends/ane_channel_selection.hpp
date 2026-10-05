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

struct ChannelSelection {
    int channels = 0;
    bool cache_hit = false, trial_passed = false;
    std::string reason;
    std::shared_ptr<const ChannelCalibrationReport> report = {};
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
        if (result.report) {
            auto snapshot = std::make_shared<ChannelCalibrationReport>(*result.report);
            snapshot->cache_hit = true; result.report = std::move(snapshot);
        }
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

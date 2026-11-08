#pragma once

#include "../../backends/ane_memory.hpp"
#include <chrono>
#include <optional>

namespace tc::qwen21 {

// Pure admission state. The caller owns cancellation, sleeping and releasing
// optional resources, and must reobserve after doing so. This never relaxes
// streaming's pressure policy or full 4 GiB system-reserve requirements.
class StreamingMemoryRecovery {
  public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    enum class Action { Ready, Wait, Denied, TimedOut };
    struct Status {
        ane::MemoryDecision decision;
        Action action;
    };

    static constexpr auto timeout = std::chrono::seconds(120);
    static constexpr auto stability_window = std::chrono::milliseconds(500);

    explicit StreamingMemoryRecovery(uint64_t growth, TimePoint now = Clock::now())
        : start_(now), growth_(growth) {}

    Status evaluate(const ane::StreamingMemoryObservation &observation, TimePoint now) {
        const auto decision = ane::admit_streaming_memory(observation,
            {uint64_t(4) << 30, observation.memory.physical_bytes}, growth_);
        if (timed_out_) return {decision, Action::TimedOut};
        if (decision.allowed()) {
            if (!waited_) return {decision, Action::Ready};
            if (!stable_since_) stable_since_ = now;
            if (now - *stable_since_ >= stability_window) return {decision, Action::Ready};
        } else {
            stable_since_.reset();
            if (!recoverable(decision.denial)) return {decision, Action::Denied};
            waited_ = true;
        }
        if (now - start_ >= timeout) {
            timed_out_ = true;
            return {decision, Action::TimedOut};
        }
        return {decision, Action::Wait};
    }

    bool waited() const { return waited_; }
    TimePoint started() const { return start_; }
    double elapsed_seconds(TimePoint now = Clock::now()) const {
        return now > start_ ? std::chrono::duration<double>(now - start_).count() : 0.0;
    }

  private:
    static bool recoverable(ane::MemoryDenial denial) {
        return denial == ane::MemoryDenial::SystemPressure ||
            denial == ane::MemoryDenial::SystemReserve ||
            denial == ane::MemoryDenial::GrowthLimit;
    }

    TimePoint start_;
    uint64_t growth_;
    bool waited_ = false;
    bool timed_out_ = false;
    std::optional<TimePoint> stable_since_;
};

} // namespace tc::qwen21

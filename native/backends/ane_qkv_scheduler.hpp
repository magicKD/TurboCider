#pragma once

#include <cmath>
#include <map>
#include <utility>

namespace tc::ane {

// Complete-block QKV admission, independent of Core ML and MLX. QKV joins
// before attention, so a projection-only speedup is not sufficient evidence.
class QkvScheduler {
    static constexpr int kReprobeVisits = 32;
    struct State {
        int visits = 0, gpu_samples = 0, hybrid_samples = 0;
        double gpu = 0, hybrid = 0;
        bool disabled = false;
    };
    std::map<std::pair<int, int>, State> states_;
    static double ema(double old, double value) {
        return old ? .75 * old + .25 * value : value;
    }

  public:
    enum class Mode { Hybrid, HybridUntimed, GpuProbe, Gpu };
    struct Plan {
        Mode mode;
        bool hybrid() const { return mode == Mode::Hybrid || mode == Mode::HybridUntimed; }
        bool measured() const { return mode == Mode::Hybrid || mode == Mode::GpuProbe; }
    };
    Plan plan(int layer, int rows) {
        auto &s = states_[{layer, rows}];
        ++s.visits;
        if (s.visits <= 2) return {Mode::Hybrid};
        if (s.visits <= 4) return {Mode::GpuProbe};
        if (s.visits % kReprobeVisits == 3) return {Mode::GpuProbe};
        if (s.visits % kReprobeVisits == 4) return {Mode::Hybrid};
        // Do not impose per-block GPU fences after both paths are known.
        if (s.disabled) return {Mode::Gpu};
        if (s.visits % kReprobeVisits == 2 || !s.hybrid || !s.gpu) return {Mode::Hybrid};
        return {Mode::HybridUntimed};
    }
    void observe(int layer, int rows, Plan plan, double wall) {
        if (!plan.measured() || !std::isfinite(wall) || wall <= 0) return;
        auto &s = states_[{layer, rows}];
        if (plan.hybrid()) {
            if (++s.hybrid_samples > 1) s.hybrid = ema(s.hybrid, wall);
        } else if (++s.gpu_samples > 1) s.gpu = ema(s.gpu, wall);
        if (s.hybrid && s.gpu) {
            // Stricter than FFN: attention cannot start until all Q/K/V join.
            if (!s.disabled && s.hybrid >= .95 * s.gpu) s.disabled = true;
            else if (s.disabled && s.hybrid < .92 * s.gpu) s.disabled = false;
        }
    }
};

} // namespace tc::ane

#pragma once

#include "ane_cost_model.hpp"
#include "ane_runtime.hpp"
#include <chrono>
#include <exception>
#include <functional>

namespace tc::ane {

struct GpuCalibrationSamples {
    std::array<std::vector<double>, 2> seconds; // one/four complete optimized GPU FFNs
    double layer_seconds = 0;
};

inline bool calibration_sampling_valid(int warmups, int repeats) noexcept {
    return warmups >= 1 && warmups <= 8 && repeats >= 3 && repeats <= 31 && repeats % 2;
}

namespace detail {
// Clock injection is only for deterministic host tests. Production uses
// steady_clock below; neither backend's exposed finish wait is a baseline.
template <class Clock>
GpuCalibrationSamples measure_full_gpu_calibration(
    const std::function<void()> &reset, const std::function<void(int)> &submit,
    const std::function<void()> &finish, int warmups, int repeats) {
    if (!reset || !submit || !finish || !calibration_sampling_valid(warmups, repeats))
        throw std::invalid_argument("GPU calibration requires complete callbacks and bounded odd repeats");
    GpuCalibrationSamples result;
    for (int sweep = 0; sweep < warmups + repeats; ++sweep) {
        for (int position = 0; position < 2; ++position) {
            const int index = (sweep + position) % 2, count = index ? 4 : 1;
            reset();
            const auto start = Clock::now();
            std::exception_ptr error;
            try { submit(count); } catch (...) { error = std::current_exception(); }
            // A callback can throw after submitting part of a batch. Drain it
            // exactly once before discarding scratch; preserve its first error.
            try { finish(); } catch (...) { if (!error) error = std::current_exception(); }
            const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
            if (error) std::rethrow_exception(error);
            if (!std::isfinite(seconds) || seconds <= 0)
                throw CapabilityError("invalid full GPU calibration span");
            if (sweep >= warmups) result.seconds[index].push_back(seconds);
        }
    }
    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };
    result.layer_seconds = calibration_layer_seconds(median(result.seconds[0]), median(result.seconds[1]));
    return result;
}
} // namespace detail

// Shared Public/Private baseline adapter: no Core ML, Private API, Metal or
// MLX dependency. Reset is outside the clock and must not submit new work.
// Submit schedules count complete family-optimized FFNs; finish drains every
// attempted submission, including failures. Every arm is warmed, then timed
// serially in cyclic order. Retain all hot samples; use medians, not minima.
inline GpuCalibrationSamples measure_full_gpu_calibration(
    const std::function<void()> &reset, const std::function<void(int)> &submit,
    const std::function<void()> &finish, int warmups = 2, int repeats = 7) {
    return detail::measure_full_gpu_calibration<std::chrono::steady_clock>(
        reset, submit, finish, warmups, repeats);
}

} // namespace tc::ane

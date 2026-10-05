#include "ane_calibration.hpp"

#include <algorithm>
#include <exception>
#include <set>

namespace tc::ane::private_api {
CalibrationBatch::CalibrationBatch(Device &device, Program &program,
                                   std::vector<CalibrationBindings> bindings, uint64_t &timeline)
    : device_(device) {
    if (bindings.empty() || bindings.size() > 4)
        throw std::invalid_argument("calibration batch requires one to four evaluations");
    // Reject accidental scratch reuse BEFORE issuing any driver work. The
    // returned ticket's surface owners, not these borrowed data addresses,
    // provide lifetime retention; addresses are only an alias check here.
    std::set<void *> inputs, outputs;
    for (const auto &row : bindings)
        for (const auto &input : row.inputs) inputs.insert(input.second.data());
    for (const auto &row : bindings) for (const auto &output : row.outputs)
        if (inputs.contains(output.second.data()) || !outputs.insert(output.second.data()).second)
            throw std::invalid_argument("calibration evaluations require independent nonaliased outputs");
    const auto previous = std::max(timeline, device.value());
    if (previous > UINT64_MAX - 2) throw std::overflow_error("calibration event timeline exhausted");
    ready_ = previous + 1;
    const auto done = previous + 2;
    prepared_.reserve(bindings.size());
    for (const auto &row : bindings) prepared_.push_back(program.prepare(row.inputs, row.outputs, ready_, done));
    timeline = done;
}

CalibrationBatchResult CalibrationBatch::measure(bool ane,
                                                 const std::function<void()> &submit_gpu,
                                                 const std::function<void()> &finish_gpu) {
    if (consumed_) throw std::invalid_argument("calibration batch already consumed");
    if (bool(submit_gpu) != bool(finish_gpu) || (!ane && !submit_gpu))
        throw std::invalid_argument("calibration requires ANE or complete GPU submission/drain callbacks");
    consumed_ = true;
    std::vector<Ticket> tickets;
    tickets.reserve(prepared_.size());
    // No driver evaluation has been submitted yet. Meet the dependency on
    // the CPU before timing, not via a signal CB hidden in an ANE-alone arm.
    if (ane) device_.release_prepared(ready_);
    const auto start = std::chrono::steady_clock::now();
    std::exception_ptr error;
    bool gpu_attempted = false;
    try {
        if (ane) for (auto &request : prepared_) tickets.push_back(request.submit());
        if (submit_gpu) { gpu_attempted = true; submit_gpu(); }
    } catch (...) { error = std::current_exception(); }
    // Never let a GPU error/cancellation free either engine's borrowed input
    // or scratch before all successfully submitted producers/consumers join.
    if (gpu_attempted) {
        try { finish_gpu(); }
        catch (...) { if (!error) error = std::current_exception(); }
    }
    CalibrationBatchResult result;
    result.ok = true;
    for (auto &ticket : tickets) {
        try {
            const auto completion = ticket.finish();
            ++result.ane_calls;
            if (!completion.ok) {
                result.ok = false;
                if (result.error.empty()) result.error = completion.error;
            }
        } catch (...) { if (!error) error = std::current_exception(); }
    }
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    // Unsubmitted PreparedRequests can now be discarded outside the clock.
    prepared_.clear();
    if (error) std::rethrow_exception(error);
    return result;
}
} // namespace tc::ane::private_api

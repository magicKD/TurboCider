#pragma once

#include "ane_program.hpp"

namespace tc::ane::private_api {

struct CalibrationBindings {
    std::vector<std::pair<std::string, Surface>> inputs, outputs;
};
struct CalibrationBatchResult {
    bool ok = false;
    double seconds = 0;
    uint64_t ane_calls = 0;
    std::string error;
};

// A standalone calibration batch, NOT an inference result. Producers must
// have joined before construction; inputs stay immutable through measure().
// Each evaluation has independently owned output surfaces. Binding and the
// already-met ANE wait are outside the clock; actual client submissions and
// completion joins remain inside. This is a host batch span, not a physical
// ANE kernel duration, and does not by itself prove device overlap.
//
// GPU callbacks must operate on separate mutable scratch and must not wait
// for this batch's ANE output. finish_gpu must drain every producer even when
// submit_gpu throws after partial submission. It is called on that path too.
class CalibrationBatch {
  public:
    CalibrationBatch(Device &, Program &, std::vector<CalibrationBindings>, uint64_t &timeline);
    CalibrationBatch(const CalibrationBatch &) = delete;
    CalibrationBatch &operator=(const CalibrationBatch &) = delete;
    CalibrationBatchResult measure(bool ane,
                                   const std::function<void()> &submit_gpu = {},
                                   const std::function<void()> &finish_gpu = {});
  private:
    Device device_;
    std::vector<PreparedRequest> prepared_;
    uint64_t ready_ = 0;
    bool consumed_ = false;
};
} // namespace tc::ane::private_api

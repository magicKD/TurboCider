#pragma once
#include <array>

#include "common.h"
namespace rv {
// Native port of RelayVSR e856773 Wan streaming path; one state per video.
class Wan {
 public:
  explicit Wan(Weights& weights);
  // LQ is the normalized, upsampled Wan conditioning frame, BCHW.
  Tensor step(const Tensor& lq, const Tensor& noise);

 private:
  Weights& w_;
  Tensor temb_, time_proj_;
  std::array<Tensor, 30> keys_, values_, cross_keys_, cross_values_;
  int64_t frame_ = 0, height_ = 0, width_ = 0;
};
}  // namespace rv

#pragma once
#include <array>
#include <memory>

#include "common.h"

namespace rv {
struct FlashKV {
  Tensor key, value;
};
struct FlashCondition {
  std::array<FlashKV, 10> caches;
  int64_t batch_size = 0, height = 0, width = 0;
  const void* owner = nullptr;
};
struct FlashState {
  std::array<FlashKV, 10> video, condition;
  std::shared_ptr<const FlashCondition> start, end;
  std::pair<Tensor, Tensor> rope_now, rope_previous;
  int frame_index = 0, total_frames = 0, height = 0, width = 0;
  int64_t batch_size = 0;
  const void* owner = nullptr;
};
class Flash {
 public:
  explicit Flash(Weights& weights, bool prefer_tensorops = true);
  bool tensorops() const { return tensorops_; }
  std::shared_ptr<FlashCondition> prepare(const Tensor& latent_BCHW);
  FlashState begin(std::shared_ptr<FlashCondition> start,
                   std::shared_ptr<FlashCondition> end, int total_frames,
                   int height, int width);
  Tensor decode(const Tensor& lr_bf16_BCHW, FlashState& state);

 private:
  Weights& weights_;
  bool tensorops_;
};
}  // namespace rv

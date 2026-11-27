#pragma once

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace rv {
using Tensor = at::Tensor;
enum class WanBlockUse { text, endpoint };
enum class RelayCheckpoint { wan, flash };

struct WeightSlice {
  at::ScalarType type;
  std::vector<int64_t> shape;
  uint64_t offset, bytes;
};
using WeightIndex = std::unordered_map<std::string, WeightSlice>;

struct ConvRotStats {
  uint64_t quantizations = 0, projections = 0, fused_gelu = 0;
};
struct WanStreamStats {
  uint64_t blocks = 0, read_bytes = 0, prefetched = 0, synchronous = 0, prefetch_denied = 0;
  unsigned peak_device_blocks = 0, peak_block_slots = 0;
  double overlapped_read_ms = 0;
};

class WanWeightStream;
class Weights {
 public:
  Weights(const std::filesystem::path&, RelayCheckpoint, bool prefer_tensorops = true);
  ~Weights();
  Tensor get(const std::string& name) const;
  bool contains(const std::string& name) const;
  void enable_wan_streaming(std::atomic<bool>& cancelled);
  void with_wan_block(int index, WanBlockUse use, const std::function<void()>& compute);
  void release_wan_conditioner();
  void verify_source() const;
  WanStreamStats wan_stream_stats() const;
  ConvRotStats convrot_stats() const { return convrot_stats_; }
  bool convrot() const { return checkpoint_ == RelayCheckpoint::wan; }
  bool tensorops() const { return tensorops_; }
  void record_int8(size_t projections, bool gelu) const;

 private:
  struct File;
  std::unique_ptr<File> file_;
  WeightIndex index_;
  RelayCheckpoint checkpoint_;
  bool tensorops_;
  mutable ConvRotStats convrot_stats_;
  mutable std::unordered_map<std::string, Tensor> cache_;
  std::unique_ptr<WanWeightStream> stream_;
};

Tensor load_cpu_weight(int fd, const WeightSlice&, const std::function<void()>& checkpoint,
                       bool background = false);
Tensor to_gpu_weight(Tensor, const std::string& name, bool tensorops);
void trim_mps_cache();

Tensor linear(Tensor x, const Weights& w, const std::string& prefix);
Tensor linear_gelu(Tensor x, const Weights& w, const std::string& prefix);
std::vector<Tensor> linear_shared(Tensor x, const Weights& w,
                                  std::initializer_list<std::string> prefixes);
Tensor rms(Tensor x, Tensor weight, double eps);
Tensor layernorm(Tensor x, Tensor weight = {}, Tensor bias = {}, double eps = 1e-6);
Tensor attention(Tensor q, Tensor k, Tensor v,
                 std::function<Tensor(int64_t, int64_t)> mask = {}, int64_t chunk = 128);
}  // namespace rv

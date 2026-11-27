#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

namespace rv {

inline constexpr char kModelID[] = "relayvsr-2x";
inline constexpr char kWanFile[] = "diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors";
inline constexpr char kFlashFile[] = "vae/relayvsr_flash_vae_x2_fp32.safetensors";
inline constexpr int kMaxInputSide = 2048;

struct Frame {
  int width = 0, height = 0;
  int64_t pts = 0, duration = 0;
  std::vector<uint8_t> rgb;
};

struct RestoreResult {
  bool wan_tensorops = false, flash_tensorops = false;
  int model_width = 0, model_height = 0, frames = 0, peak_source_frames = 0;
  unsigned peak_weight_slots = 0, peak_device_blocks = 0;
  uint64_t blocks = 0, bytes_read = 0, prefetched_blocks = 0, synchronous_blocks = 0, prefetch_denied = 0;
  uint64_t quantizations = 0, projections = 0, fused_gelu = 0;
  uint64_t peak_mps_bytes = 0;
  double overlapped_read_ms = 0;
};

using ReadFrame = std::function<bool(Frame&)>;
using OutputFrame = std::function<void(const Frame&, const uint8_t*)>;

// Frame timing is carried unchanged to the sink. RGB from the sink is exactly
// 2× each source axis and is valid until the callback returns.
RestoreResult restore(const std::filesystem::path& root, Frame first, double pixel_aspect,
                      uint64_t seed, const ReadFrame& read, const OutputFrame& output,
                      std::atomic<bool>& cancelled);

}  // namespace rv

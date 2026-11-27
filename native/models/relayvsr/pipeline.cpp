#include "pipeline.hpp"
#include "flash.h"
#include "wan.h"
#include "../../core/common.hpp"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <torch/mps.h>

#include <algorithm>
#include <cmath>
#include <future>
#include <mutex>

namespace rv {
namespace {
constexpr size_t kWindowFrames = 16;

struct Geometry {
  int width, height, padded_width, padded_height;
};

Geometry inference_geometry(const Frame& first, double pixel_aspect) {
  tc::require(first.width > 0 && first.height > 0 &&
              first.width <= kMaxInputSide && first.height <= kMaxInputSide,
              "RelayVSR 2× supports source sides up to 2048 pixels");
  const double width = first.width * pixel_aspect;
  const double height = first.height;
  const auto scale = std::min({1.0, 640.0 / std::max(width, height), 512.0 / std::min(width, height)});
  const auto w = std::max(1, int(std::lround(width * scale)));
  const auto h = std::max(1, int(std::lround(height * scale)));
  return {w, h, (w + 15) / 16 * 16, (h + 15) / 16 * 16};
}

Tensor prepare(const Frame& frame, const Geometry& geometry) {
  // .to() owns its GPU storage before the input buffer is released.
  auto rgb = at::from_blob(const_cast<uint8_t*>(frame.rgb.data()),
                           {frame.height, frame.width, 3}, at::kByte)
                 .permute({2, 0, 1}).unsqueeze(0);
  if (frame.width != geometry.width || frame.height != geometry.height)
    rgb = at::_upsample_bilinear2d_aa(rgb.to(at::kFloat), {geometry.height, geometry.width},
                                      false, std::nullopt, std::nullopt);
  rgb = rgb.to(at::kMPS, at::kBFloat16).div_(127.5).sub_(1.0);
  const int right = geometry.padded_width - geometry.width;
  const int bottom = geometry.padded_height - geometry.height;
  return right || bottom ? at::replication_pad2d(rgb, {0, right, 0, bottom}) : rgb;
}

Tensor output_rgb(const Tensor& prediction, const Geometry& geometry, int width, int height) {
  auto finite = prediction.isfinite().all();
  auto cropped = prediction.slice(2, 0, geometry.height * 2).slice(3, 0, geometry.width * 2);
  if (cropped.size(2) != height || cropped.size(3) != width)
    cropped = at::upsample_bilinear2d(cropped.to(at::kFloat), {height, width}, false,
                                     std::nullopt, std::nullopt);
  auto rgb = cropped[0].add(1).mul_(127.5).clamp_(0, 255).to(at::kByte)
                 .permute({1, 2, 0}).contiguous().cpu();
  tc::require(finite.item<bool>(), "RelayVSR produced non-finite pixels");
  return rgb;
}

struct MPSCleanup {
  ~MPSCleanup() {
    try {
      torch::mps::synchronize();
      at::mps::getIMPSAllocator()->emptyCache();
    } catch (...) {}
  }
};

}  // namespace

RestoreResult restore(const std::filesystem::path& root, Frame first, double pixel_aspect,
                      uint64_t seed, const ReadFrame& read, const OutputFrame& output,
                      std::atomic<bool>& cancelled) {
  // LibTorch MPS submission and its allocator are process-wide.
  static std::timed_mutex mps;
  std::unique_lock lock(mps, std::defer_lock);
  while (!lock.try_lock_for(std::chrono::milliseconds(20))) tc::checkpoint(cancelled);
  tc::checkpoint(cancelled);
  tc::require(torch::mps::is_available(), "RelayVSR requires an Apple MPS GPU");
  tc::require(std::isfinite(pixel_aspect) && pixel_aspect > 0, "Invalid source pixel aspect ratio");
  MPSCleanup cleanup;
  at::NoGradGuard no_grad;
  const auto geometry = inference_geometry(first, pixel_aspect);
  const int width = first.width, height = first.height;
  const size_t frame_bytes = size_t(width) * height * 3;

  auto read_window = [&](size_t count) {
    std::vector<Frame> frames;
    frames.reserve(kWindowFrames);
    while (frames.size() < count) {
      tc::checkpoint(cancelled);
      Frame frame;
      if (!read(frame)) break;
      tc::require(frame.width == width && frame.height == height && frame.rgb.size() == frame_bytes,
                  "RelayVSR source geometry changed while streaming");
      frames.push_back(std::move(frame));
    }
    return frames;
  };

  auto window = read_window(kWindowFrames - 1);
  tc::require(first.rgb.size() == frame_bytes, "Invalid RelayVSR input frame");
  window.insert(window.begin(), std::move(first));
  // Preload at most one additional host window, and avoid doubling the
  // buffered RGB working set on large source rasters.
  const bool read_ahead = frame_bytes * kWindowFrames <= (128ull << 20);
  RestoreResult result;
  result.peak_source_frames = static_cast<int>(window.size());
  result.model_width = geometry.width;
  result.model_height = geometry.height;
  Weights wan_weights(root / kWanFile, RelayCheckpoint::wan);
  Weights flash_weights(root / kFlashFile, RelayCheckpoint::flash);
  wan_weights.enable_wan_streaming(cancelled);
  Wan wan(wan_weights);
  Flash flash(flash_weights);
  result.wan_tensorops = wan_weights.tensorops();
  result.flash_tensorops = flash.tensorops();
  auto generator = at::make_generator<at::CPUGeneratorImpl>(seed);
  auto* allocator = at::mps::getIMPSAllocator();

  auto sample_memory = [&] {
    result.peak_mps_bytes = std::max<uint64_t>(result.peak_mps_bytes,
                                               allocator->getDriverAllocatedMemory());
  };
  auto condition = [&](const Tensor& input) {
    tc::checkpoint(cancelled);
    wan_weights.verify_source();
    auto noise = at::randn({1, 48, 1, geometry.padded_height / 8, geometry.padded_width / 8},
                           generator, at::TensorOptions().dtype(at::kFloat)).to(at::kMPS);
    auto endpoint = flash.prepare(wan.step(input, noise));
    sample_memory();
    return endpoint;
  };

  auto first_input = prepare(window.front(), geometry);
  auto first_condition = condition(first_input);
  bool shared_first = false;

  for (;;) {
    const auto count = window.size();
    std::future<std::vector<Frame>> next;
    if (read_ahead && count == kWindowFrames) {
      try { next = std::async(std::launch::async, read_window, kWindowFrames - 1); }
      catch (const std::system_error&) {}
    }

    auto last_input = count == 1 ? first_input : prepare(window.back(), geometry);
    auto last_condition = count == 1 ? first_condition : condition(last_input);
    auto state = flash.begin(first_condition, last_condition, static_cast<int>(count),
                             geometry.padded_height, geometry.padded_width);
    for (size_t i = 0; i < count; ++i) {
      tc::checkpoint(cancelled);
      const auto& frame = window[i];
      auto input = i == 0 ? first_input : i + 1 == count ? last_input : prepare(frame, geometry);
      auto prediction = flash.decode(input, state);
      // Decode the shared boundary to seed the new Flash history. It has
      // already been published by the previous window.
      if (shared_first && i == 0) continue;
      auto rgb = output_rgb(prediction, geometry, width * 2, height * 2);
      tc::checkpoint(cancelled);
      output(frame, rgb.const_data_ptr<uint8_t>());
      ++result.frames;
      sample_memory();
      trim_mps_cache();
    }
    if (count < kWindowFrames) break;

    auto boundary = std::move(window.back());
    window.clear();
    if (next.valid()) {
      window = next.get();
      result.peak_source_frames = std::max(result.peak_source_frames,
                                           static_cast<int>(count + window.size()));
    } else {
      window = read_window(kWindowFrames - 1);
    }
    if (window.empty()) break;
    window.insert(window.begin(), std::move(boundary));
    first_input = std::move(last_input);
    first_condition = std::move(last_condition);
    shared_first = true;
  }

  tc::checkpoint(cancelled);
  wan_weights.verify_source();
  flash_weights.verify_source();
  const auto weights = wan_weights.wan_stream_stats();
  result.blocks = weights.blocks;
  result.bytes_read = weights.read_bytes;
  result.prefetched_blocks = weights.prefetched;
  result.synchronous_blocks = weights.synchronous;
  result.prefetch_denied = weights.prefetch_denied;
  result.peak_device_blocks = weights.peak_device_blocks;
  result.peak_weight_slots = weights.peak_block_slots;
  result.overlapped_read_ms = weights.overlapped_read_ms;
  const auto convrot = wan_weights.convrot_stats();
  result.quantizations = convrot.quantizations;
  result.projections = convrot.projections;
  result.fused_gelu = convrot.fused_gelu;
  return result;
}
}  // namespace rv

#pragma once

#include "common.h"
#include "../../core/common.hpp"

#include <ATen/mps/MPSAllocatorInterface.h>
#include <torch/mps.h>
#include <mach/mach.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <future>

namespace rv {

// LibTorch device tensors are owned by the inference thread. A read-ahead
// worker fills at most one CPU slot while the current GPU block executes.
class WanWeightStream {
 public:
  WanWeightStream(int fd, const WeightIndex& index, std::atomic<bool>& cancelled, bool tensorops)
      : fd_(fd), cancelled_(cancelled), tensorops_(tensorops) {
    size_t length = sizeof(physical_bytes_);
    if (sysctlbyname("hw.memsize", &physical_bytes_, &length, nullptr, 0) != 0)
      physical_bytes_ = 0;
    for (const auto& [name, slice] : index) {
      if (!name.starts_with("blocks.")) continue;
      const char* end = name.data() + name.size();
      int block = -1;
      const auto [dot, error] = std::from_chars(name.data() + 7, end, block);
      tc::require(error == std::errc{} && dot != end && *dot == '.' && block >= 0 && block < 30,
                  "Invalid RelayVSR Wan block name");
      const std::string_view key(dot + 1, end - dot - 1);
      const bool text = key.starts_with("attn2.to_k.") || key.starts_with("attn2.to_v.") ||
                        key.starts_with("attn2.norm_k.");
      auto& group = plan(block, text ? WanBlockUse::text : WanBlockUse::endpoint);
      group.params.emplace_back(&name, &slice);
      group.bytes += slice.bytes;
    }
    for (auto& block : plans_)
      for (auto& group : block) {
        tc::require(!group.params.empty(), "Incomplete RelayVSR Wan block");
        std::sort(group.params.begin(), group.params.end(),
                  [](const auto& a, const auto& b) { return a.second->offset < b.second->offset; });
      }
  }

  ~WanWeightStream() { stop_worker(); }

  void run(int block, WanBlockUse use, const std::function<void()>& compute) {
    try {
      tc::checkpoint(cancelled_);
      active_block_ = block;
      ++stats_.blocks;
      stats_.peak_block_slots = std::max(1u, stats_.peak_block_slots);
      receive_prefetch(block);
      if (block + 1 < static_cast<int>(plans_.size()))
        start_prefetch(block + 1, use);
      previous_compute_begin_ = Clock::now();
      compute();
      // Synchronize before releasing the only device weight block. The
      // next block may reuse MPS allocations from this one.
      torch::mps::synchronize();
      previous_compute_end_ = Clock::now();
      evict();
    } catch (...) {
      stop_worker();
      try { torch::mps::synchronize(); } catch (...) {}
      evict();
      throw;
    }
  }

  Tensor get(const std::string& name, const WeightSlice& slice) {
    if (auto it = device_weights_.find(name); it != device_weights_.end()) return it->second;
    const auto block_end = name.find('.', 7);
    if (active_block_ < 0 || name.substr(7, block_end - 7) != std::to_string(active_block_))
      throw std::logic_error("RelayVSR Wan weight accessed outside its block");
    tc::checkpoint(cancelled_);
    auto host = host_.extract(name);
    auto cpu = host.empty() ? read(slice) : std::move(host.mapped());
    auto gpu = to_gpu_weight(std::move(cpu), name, tensorops_);
    device_weights_.emplace(name, gpu);
    stats_.peak_device_blocks = 1;
    return gpu;
  }

  WanStreamStats stats() const {
    auto result = stats_;
    result.read_bytes = read_bytes_.load(std::memory_order_relaxed);
    return result;
  }

 private:
  using Clock = std::chrono::steady_clock;
  using HostWeights = std::unordered_map<std::string, Tensor>;
  struct Plan {
    uint64_t bytes = 0;
    std::vector<std::pair<const std::string*, const WeightSlice*>> params;
  };
  struct Prepared {
    HostWeights weights;
    Clock::time_point begin, end;
  };

  Plan& plan(int block, WanBlockUse use) { return plans_.at(block).at(static_cast<size_t>(use)); }

  bool can_prefetch(uint64_t bytes) const {
    // One 128–160 MiB lookahead slot is cheap on larger Macs. Base admission
    // on available memory (including reusable file pages) and on the current
    // MPS working set. A tight system falls back to single-slot reads.
    constexpr uint64_t reserve = 2ull << 30;
    if (physical_bytes_ < reserve + bytes) return false;
    const auto* allocator = at::mps::getIMPSAllocator();
    if (allocator->getDriverAllocatedMemory() > physical_bytes_ - reserve - bytes)
      return false;
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const auto host = mach_host_self();
    const auto status = host_statistics64(host, HOST_VM_INFO64,
                                         reinterpret_cast<host_info64_t>(&vm), &count);
    mach_port_deallocate(mach_task_self(), host);
    const auto available = (uint64_t(vm.free_count) + std::min(vm.inactive_count, vm.external_page_count)) *
                           vm_kernel_page_size;
    return status == KERN_SUCCESS && available > bytes + (512ull << 20);
  }

  void start_prefetch(int block, WanBlockUse use) {
    const auto& next_plan = plan(block, use);
    if (prefetch_disabled_ || !can_prefetch(next_plan.bytes)) {
      ++stats_.prefetch_denied;
      trim_mps_cache();
      return;
    }
    next_block_ = block;
    try {
      next_ = std::async(std::launch::async, [this, group = &next_plan] {
        Prepared result;
        result.begin = Clock::now();
        for (const auto& [name, slice] : group->params)
          result.weights.emplace(*name, read(*slice, true));
        result.end = Clock::now();
        return result;
      });
      stats_.peak_block_slots = 2;
    } catch (const std::system_error&) {
      prefetch_disabled_ = true;
      ++stats_.prefetch_denied;
    } catch (const std::bad_alloc&) {
      prefetch_disabled_ = true;
      ++stats_.prefetch_denied;
    }
  }

  void receive_prefetch(int block) {
    if (!next_.valid()) {
      ++stats_.synchronous;
      return;
    }
    if (next_block_ != block) throw std::logic_error("RelayVSR Wan prefetch block order");
    while (next_.wait_for(std::chrono::milliseconds(20)) != std::future_status::ready)
      tc::checkpoint(cancelled_);
    try {
      auto prepared = next_.get();
      host_ = std::move(prepared.weights);
      const auto overlap = std::min(prepared.end, previous_compute_end_) -
                           std::max(prepared.begin, previous_compute_begin_);
      stats_.overlapped_read_ms += std::chrono::duration<double, std::milli>(
          std::max(overlap, Clock::duration::zero())).count();
      ++stats_.prefetched;
    } catch (const std::bad_alloc&) {
      prefetch_disabled_ = true;
      ++stats_.synchronous;
    }
  }

  Tensor read(const WeightSlice& slice, bool background = false) {
    auto value = load_cpu_weight(fd_, slice, [&] {
      tc::checkpoint(cancelled_);
      if (stop_.load(std::memory_order_relaxed)) throw tc::Cancelled();
    }, background);
    read_bytes_.fetch_add(slice.bytes, std::memory_order_relaxed);
    return value;
  }

  void stop_worker() {
    stop_.store(true, std::memory_order_relaxed);
    if (next_.valid()) { try { (void)next_.get(); } catch (...) {} }
  }
  void evict() {
    host_.clear();
    device_weights_.clear();
    active_block_ = -1;
  }

  int fd_;
  std::atomic<bool>& cancelled_;
  bool tensorops_;
  uint64_t physical_bytes_ = 0;
  std::array<std::array<Plan, 2>, 30> plans_;
  HostWeights host_, device_weights_;
  WanStreamStats stats_;
  std::atomic<uint64_t> read_bytes_{0};
  std::atomic<bool> stop_{false};
  std::future<Prepared> next_;
  Clock::time_point previous_compute_begin_, previous_compute_end_;
  int active_block_ = -1, next_block_ = -1;
  bool prefetch_disabled_ = false;
};
}  // namespace rv

#pragma once
#include "../../runtime/memory_accounting.hpp"
#import <Metal/Metal.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace tc {
// Read-only process observations. Sampled peaks are lower bounds, not hard
// budget proofs. Metal allocations and physical footprint overlap: never add
// the two figures. Core ML/other driver allocations may be outside Metal.
class SeedVR2MemorySampler {
    id<MTLDevice> device_ = MTLCreateSystemDefaultDevice();
    std::atomic<bool> stopped_{false}, footprint_available_{false};
    std::atomic<uint64_t> footprint_peak_{0}, metal_peak_{0}, samples_{0}, failures_{0};
    std::mutex mutex_;
    std::condition_variable ready_;
    std::thread worker_;
    static void maximum(std::atomic<uint64_t> &target, uint64_t value) {
        auto previous = target.load(std::memory_order_relaxed);
        while (previous < value && !target.compare_exchange_weak(previous, value, std::memory_order_relaxed)) {}
    }
    void sample() noexcept {
        try {
            @autoreleasepool {
                const auto process = observe_process_memory();
                if (process.available) {
                    footprint_available_.store(true, std::memory_order_relaxed);
                    maximum(footprint_peak_, process.process_footprint_bytes);
                }
                if (device_) maximum(metal_peak_, device_.currentAllocatedSize);
                samples_.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (...) {
            // An optional observer failure must not masquerade as a codec
            // failure or turn a missing reading into a valid zero peak.
            failures_.fetch_add(1, std::memory_order_relaxed);
        }
    }
  public:
    SeedVR2MemorySampler() {
        sample();
        worker_ = std::thread([this] {
            std::unique_lock lock(mutex_);
            while (!stopped_.load(std::memory_order_relaxed)) {
                lock.unlock(); sample(); lock.lock();
                ready_.wait_for(lock, std::chrono::milliseconds(20), [this] { return stopped_.load(); });
            }
        });
    }
    SeedVR2MemorySampler(const SeedVR2MemorySampler &) = delete;
    SeedVR2MemorySampler &operator=(const SeedVR2MemorySampler &) = delete;
    ~SeedVR2MemorySampler() { stop(); }
    void stop() {
        if (!stopped_.exchange(true)) {
            ready_.notify_all();
            if (worker_.joinable()) worker_.join();
            sample();
        }
    }
    bool footprint_available() const { return footprint_available_.load(); }
    bool metal_available() const { return device_ && metal_peak_.load() > 0; }
    uint64_t footprint_peak() const { return footprint_peak_.load(); }
    uint64_t metal_peak() const { return metal_peak_.load(); }
    uint64_t samples() const { return samples_.load(); }
    uint64_t failures() const { return failures_.load(); }
};
} // namespace tc

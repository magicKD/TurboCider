#pragma once

#include "ane_gpu.hpp"
#include "ane_memory.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace tc::ane::executor_support {

class Worker {
public:
    Worker() : thread_([this] { run(); }) {}

    ~Worker() {
        try { join(); } catch (...) {}
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }

    void join() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return !busy_; });
        auto error = std::exchange(error_, {});
        if (error) std::rethrow_exception(error);
    }

    void submit(std::function<void()> job) {
        join();
        {
            std::lock_guard lock(mutex_);
            job_ = std::move(job);
            busy_ = true;
        }
        cv_.notify_all();
    }

private:
    void run() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || job_; });
                if (stop_) return;
                job = std::exchange(job_, {});
            }
            std::exception_ptr error;
            @autoreleasepool {
                try { job(); } catch (...) { error = std::current_exception(); }
            }
            {
                std::lock_guard lock(mutex_);
                error_ = error;
                busy_ = false;
            }
            cv_.notify_all();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::function<void()> job_;
    std::exception_ptr error_;
    bool busy_ = false;
    bool stop_ = false;
    std::thread thread_;
};

// A submitted GPU producer retains the slot until it completes. Disable the
// executor after a failed drain so recovery cannot reuse that slot.
template <size_t Count>
struct QuantizationDrain {
    std::array<std::optional<gpu::QuantStage>, Count> &tickets;
    std::atomic<bool> &disabled;

    ~QuantizationDrain() {
        for (auto &ticket : tickets) {
            if (ticket) {
                try { if (ticket->finish().timed_out) disabled = true; }
                catch (...) { disabled = true; }
            }
        }
    }
};

// Shared Metal storage used by the W8 executor self-tests.
struct Buffer {
    id<MTLBuffer> value;
    std::shared_ptr<void> owner;
    Buffer(id<MTLDevice> d, size_t bytes) {
        value = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!value) throw MemoryBudgetError("W8 self-test buffer allocation failed");
        owner = {(__bridge_retained void *)value, [](void *p) { CFRelease(p); }}; std::memset(value.contents, 0, bytes);
    }
    DeviceWeightView weight(int rows, int cols) { return {(__bridge void *)value, value.length, 0, size_t(cols) * 4, rows, cols,
        DeviceWeightEncoding::Dense, DType::FP32, 32, {}, {}, owner}; }
    DeviceMatrixView matrix(int rows, int cols, DType dtype = DType::FP32) { return {(__bridge void *)value, value.length, 0, rows, cols,
        size_t(cols) * (dtype == DType::FP32 ? 4 : 2), dtype, owner}; }
};

} // namespace tc::ane::executor_support

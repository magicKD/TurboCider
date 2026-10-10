#pragma once

#include "../../core/common.hpp"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <unistd.h>

namespace tc::streaming {

// One bounded pread job in flight. The owner supplies already-admitted storage
// and must keep its fd/buffers alive until stop_and_drain() (also done by RAII).
// No model parser, MLX allocation, digest or output publication runs here.
class ReadAheadWorker final {
  public:
    struct Result { uint64_t bytes = 0; double seconds = 0; };
  private:
    struct Job { std::span<std::byte> target; uint64_t offset; };
    struct Completion { Result result; std::exception_ptr error; };
    const int fd_;
    const uint64_t file_bytes_, capacity_;
    const std::atomic<bool> *cancel_;
    const std::thread::id owner_ = std::this_thread::get_id();
    std::atomic<bool> stopped_{false};
    std::mutex mutex_;
    std::condition_variable work_, done_;
    std::optional<Job> pending_;
    std::optional<Completion> ready_;
    bool busy_ = false, failed_ = false;
    std::thread worker_;

    void check_owner() const {
        require(std::this_thread::get_id() == owner_, "gguf_read_ahead: owner violation");
    }
    void check_cancelled() const {
        if (stopped_.load(std::memory_order_acquire) ||
            (cancel_ && cancel_->load(std::memory_order_acquire))) throw Cancelled();
    }
    void run() noexcept {
        std::unique_lock lock(mutex_);
        for (;;) {
            work_.wait(lock, [&] { return stopped_.load(std::memory_order_acquire) || pending_.has_value(); });
            if (stopped_.load(std::memory_order_acquire)) return;
            const auto job = *pending_; pending_.reset();
            lock.unlock();
            Completion completion;
            const auto start = std::chrono::steady_clock::now();
            try {
                while (completion.result.bytes < job.target.size()) {
                    check_cancelled();
                    const auto count = ::pread(fd_, job.target.data() + completion.result.bytes,
                        job.target.size() - size_t(completion.result.bytes),
                        off_t(job.offset + completion.result.bytes));
                    if (count < 0 && errno == EINTR) continue;
                    require(count > 0, "gguf_read_ahead: payload read failed or source changed");
                    completion.result.bytes += uint64_t(count);
                }
                check_cancelled();
            } catch (...) { completion.error = std::current_exception(); }
            completion.result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            lock.lock(); ready_ = std::move(completion); done_.notify_all();
        }
    }
  public:
    ReadAheadWorker(int fd, uint64_t file_bytes, uint64_t capacity,
                    const std::atomic<bool> *cancel = nullptr)
        : fd_(fd), file_bytes_(file_bytes), capacity_(capacity), cancel_(cancel) {
        require(fd >= 0 && file_bytes && file_bytes <= uint64_t(std::numeric_limits<off_t>::max()) &&
            capacity && capacity <= SIZE_MAX, "gguf_read_ahead: invalid source/capacity");
        check_cancelled(); worker_ = std::thread([this] { run(); });
    }
    ReadAheadWorker(const ReadAheadWorker &) = delete;
    ReadAheadWorker &operator=(const ReadAheadWorker &) = delete;
    ~ReadAheadWorker() { stop_and_drain(); }

    void submit(std::span<std::byte> target, uint64_t offset) {
        check_owner(); check_cancelled();
        require(!target.empty() && target.data() && target.size() <= capacity_ &&
            offset <= file_bytes_ && target.size() <= file_bytes_ - offset,
            "gguf_read_ahead: job exceeds source/buffer range");
        std::lock_guard lock(mutex_);
        require(!busy_ && !failed_ && !pending_ && !ready_, "gguf_read_ahead: queue busy or failed");
        pending_ = Job{target, offset}; busy_ = true; work_.notify_one();
    }
    Result take() {
        check_owner(); check_cancelled();
        std::unique_lock lock(mutex_);
        require(busy_, "gguf_read_ahead: no pending job");
        while (!ready_) {
            check_cancelled(); done_.wait_for(lock, std::chrono::milliseconds(10));
        }
        check_cancelled();
        const auto completed = *ready_; ready_.reset(); busy_ = false;
        if (completed.error) { failed_ = true; std::rethrow_exception(completed.error); }
        return completed.result;
    }
    bool idle() {
        check_owner(); std::lock_guard lock(mutex_);
        return !busy_ && !pending_ && !ready_;
    }
    void stop_and_drain() noexcept {
        stopped_.store(true, std::memory_order_release); work_.notify_all(); done_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
};
} // namespace tc::streaming

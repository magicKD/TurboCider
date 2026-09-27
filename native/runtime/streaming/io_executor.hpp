#pragma once
#include "../../core/stream_slot_c.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace tc::streaming {

// Fixed-capacity multi-producer mailbox. Overflow is a separate sticky latch.
// Lifetime must cover every producer (join/drain before destruction).
class CompletionMailbox {
public:
    explicit CompletionMailbox(size_t capacity);
    bool post(const tc_stream_completion_v1 &) noexcept;
    bool pop(tc_stream_completion_v1 &);
    bool overflowed() const noexcept { return overflow_.load(std::memory_order_acquire); }
    void wait_for(std::chrono::milliseconds timeout);
private:
    std::vector<tc_stream_completion_v1> records_;
    std::mutex mutex_;
    std::condition_variable changed_;
    size_t head_ = 0, count_ = 0;
    std::atomic<bool> overflow_{false};
};

struct FillJob {
    tc_stream_slot_ticket_v1 ticket{};
    void *user = nullptr;
    // Must catch its errors and check cancel at chunk boundaries. Owns no
    // model context allocation rights. The job's referenced spans stay alive
    // until completion or shutdown_and_join().
    // The output count is MATERIALIZED CONTENT bytes (excluding allocation
    // padding), not source bytes read. Converting readers account I/O separately.
    int (*fill)(void *, const tc_stream_slot_ticket_v1 *, const std::atomic<bool> *, uint64_t *) = nullptr;
};

class IoExecutor {
public:
    IoExecutor(uint32_t workers, uint32_t capacity, CompletionMailbox &);
    ~IoExecutor();
    IoExecutor(const IoExecutor &) = delete;
    IoExecutor &operator=(const IoExecutor &) = delete;
    bool enqueue(const FillJob &); // false means backpressure, not lost work
    void shutdown_and_join() noexcept;
    uint32_t worker_count() const noexcept { return uint32_t(threads_.size()); }
private:
    CompletionMailbox &mailbox_;
    std::vector<FillJob> jobs_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable changed_;
    size_t head_ = 0, count_ = 0, running_ = 0;
    std::atomic<bool> cancelled_{false};
    void run();
};
} // namespace tc::streaming

#include "io_executor.hpp"
#include "audit.hpp"
#include "layout.hpp"
#include <stdexcept>

namespace tc::streaming {
CompletionMailbox::CompletionMailbox(size_t capacity) {
    if (!capacity || capacity > max_slots * (1 + TC_STREAM_MAX_READER_QUEUES))
        throw std::invalid_argument("invalid streaming mailbox capacity");
    records_.resize(capacity);
}
bool CompletionMailbox::post(const tc_stream_completion_v1 &record) noexcept {
    std::lock_guard lock(mutex_);
    if (count_ == records_.size()) {
        overflow_.store(true, std::memory_order_release);
        changed_.notify_one();
        return false;
    }
    records_[(head_+count_)%records_.size()] = record;
    ++count_; changed_.notify_one();
    return true;
}
bool CompletionMailbox::pop(tc_stream_completion_v1 &record) {
    std::lock_guard lock(mutex_);
    if (!count_) return false;
    record = records_[head_]; head_ = (head_+1)%records_.size(); --count_;
    return true;
}
void CompletionMailbox::wait_for(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    changed_.wait_for(lock, timeout, [&]{return count_ || overflowed();});
}
IoExecutor::IoExecutor(uint32_t workers, uint32_t capacity, CompletionMailbox &mailbox)
    : mailbox_(mailbox) {
    if (!workers || workers>capacity || capacity>max_slots)
        throw std::invalid_argument("invalid streaming I/O worker/queue count");
    jobs_.resize(capacity); threads_.reserve(workers);
    try {
        for (uint32_t i=0; i<workers; ++i) {
            threads_.emplace_back([this]{run();});
            audit_increment(AuditCounter::WorkerThreads);
        }
    } catch (...) {
        shutdown_and_join();
        throw;
    }
}
IoExecutor::~IoExecutor() { shutdown_and_join(); }
bool IoExecutor::enqueue(const FillJob &job) {
    if (!job.fill) throw std::invalid_argument("missing streaming fill function");
    std::lock_guard lock(mutex_);
    if (cancelled_.load(std::memory_order_acquire))
        throw std::logic_error("streaming I/O executor stopped");
    if (count_+running_ == jobs_.size()) return false;
    jobs_[(head_+count_)%jobs_.size()] = job; ++count_;
    changed_.notify_one(); return true;
}
void IoExecutor::run() {
    while (true) {
        FillJob job;
        {
            std::unique_lock lock(mutex_);
            changed_.wait(lock,[&]{return count_ || cancelled_.load(std::memory_order_acquire);});
            if (cancelled_.load(std::memory_order_acquire)) return;
            job=jobs_[head_]; head_=(head_+1)%jobs_.size(); --count_; ++running_;
        }
        tc_stream_completion_v1 done{};
        done.struct_size=sizeof(done); done.version=TC_STREAM_SLOT_ABI_V1;
        done.kind=TC_STREAM_FILL_COMPLETE; done.ticket=job.ticket;
        try { done.status = job.fill(job.user,&job.ticket,&cancelled_,&done.bytes); }
        catch (...) { done.status=-1; }
        {
            // Release job credit BEFORE publishing completion, so an owner
            // observing it cannot race an unavailable worker credit.
            std::lock_guard lock(mutex_); --running_;
        }
        if (!mailbox_.post(done)) cancelled_.store(true,std::memory_order_release);
        changed_.notify_all();
    }
}
void IoExecutor::shutdown_and_join() noexcept {
    cancelled_.store(true,std::memory_order_release); changed_.notify_all();
    for (auto &thread : threads_) if (thread.joinable()) thread.join();
    // Queued jobs are not started after cancellation; caller retains their
    // target storage until this join. These slots must not be reused as Ready.
}
} // namespace tc::streaming

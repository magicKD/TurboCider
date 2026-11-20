#pragma once

#include "turbocider/turbocider.h"
#include <atomic>
#include <chrono>
#include <csignal>
#include <thread>

namespace tc_worker {

inline std::atomic<bool> cancelled{false};
static_assert(std::atomic<bool>::is_always_lock_free);

inline void signal_stop(int) {
    cancelled.store(true, std::memory_order_relaxed);
}

class SignalHandlers {
public:
    SignalHandlers()
        : interrupt_(std::signal(SIGINT, signal_stop)),
          terminate_(std::signal(SIGTERM, signal_stop)) {}

    SignalHandlers(const SignalHandlers &) = delete;
    SignalHandlers &operator=(const SignalHandlers &) = delete;

    ~SignalHandlers() {
        std::signal(SIGINT, interrupt_);
        std::signal(SIGTERM, terminate_);
    }

private:
    using Handler = void (*)(int);
    Handler interrupt_;
    Handler terminate_;
};

// Keep engine cancellation on a regular thread, including while source
// verification runs without progress callbacks. The engine outlives the relay.
class CancellationRelay {
public:
    explicit CancellationRelay(tc_engine *engine) : worker_([this, engine] {
        while (!stopping_.load(std::memory_order_relaxed)) {
            if (cancelled.load(std::memory_order_relaxed)) tc_engine_cancel(engine);
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    }) {}

    ~CancellationRelay() {
        stopping_.store(true, std::memory_order_relaxed);
        worker_.join();
    }

private:
    std::atomic<bool> stopping_{false};
    std::thread worker_;
};

} // namespace tc_worker

#pragma once

#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tc {

template<class T> struct PreparationResult {
    std::unique_ptr<T> value;
    std::exception_ptr error;
    double seconds = 0, wait_seconds = 0, before_join_seconds = 0;
};

// Request-scoped host preparation. The factory must not touch MLX, request
// callbacks or borrowed request state. Destruction joins even during exception
// unwinding, then releases the completed value on the owning request thread.
// A synchronous platform load cannot be interrupted midway by cancellation.
template<class T> class AsyncPreparation {
    using Clock = std::chrono::steady_clock;
    struct Completion {
        PreparationResult<T> result;
        Clock::time_point start, finish;
    };
    std::future<Completion> future_;
  public:
    template<class Factory> explicit AsyncPreparation(Factory factory)
        : future_(std::async(std::launch::async, [factory = std::move(factory)]() mutable {
            Completion completion;
            completion.start = Clock::now();
            try { completion.result.value = factory(); }
            catch (...) { completion.result.error = std::current_exception(); }
            completion.finish = Clock::now();
            return completion;
        })) {}
    AsyncPreparation(const AsyncPreparation &) = delete;
    AsyncPreparation &operator=(const AsyncPreparation &) = delete;
    ~AsyncPreparation() {
        if (future_.valid()) {
            try { (void)future_.get(); } catch (...) {}
        }
    }
    PreparationResult<T> take() {
        if (!future_.valid()) throw std::logic_error("preparation result was already consumed");
        const auto join = Clock::now();
        auto completion = future_.get();
        auto &result = completion.result;
        result.seconds = std::chrono::duration<double>(completion.finish - completion.start).count();
        result.wait_seconds = std::chrono::duration<double>(Clock::now() - join).count();
        result.before_join_seconds = std::max(0., std::chrono::duration<double>(
            std::min(join, completion.finish) - completion.start).count());
        return std::move(result);
    }
};

} // namespace tc

#include "../../native/runtime/async_preparation.hpp"
#include <atomic>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <thread>

struct Resource {
    std::atomic<int> &alive;
    std::thread::id &destroyed_on;
    Resource(std::atomic<int> &count, std::thread::id &thread) : alive(count), destroyed_on(thread) { ++alive; }
    ~Resource() { destroyed_on = std::this_thread::get_id(); --alive; }
};

int main() {
    const auto owner = std::this_thread::get_id();
    std::atomic<int> alive{0};
    std::thread::id destroyed_on;
    {
        std::promise<void> entered, proceed;
        auto gate = proceed.get_future();
        tc::AsyncPreparation<Resource> task([&] {
            assert(std::this_thread::get_id() != owner);
            auto value = std::make_unique<Resource>(alive, destroyed_on);
            entered.set_value();
            gate.get();
            return value;
        });
        entered.get_future().get();
        assert(alive == 1); // constructor returned while work is still blocked
        proceed.set_value();
        auto result = task.take();
        assert(result.value && !result.error && alive == 1);
        assert(result.seconds >= 0 && result.wait_seconds >= 0 && result.before_join_seconds >= 0);
        assert(result.before_join_seconds <= result.seconds);
        try { (void)task.take(); assert(false); } catch (const std::logic_error &) {}
    }
    assert(alive == 0 && destroyed_on == owner);
    {
        tc::AsyncPreparation<Resource> task([]() -> std::unique_ptr<Resource> {
            throw std::invalid_argument("invalid graph");
        });
        auto result = task.take();
        assert(!result.value && result.error);
        try { std::rethrow_exception(result.error); assert(false); }
        catch (const std::invalid_argument &e) { assert(std::string(e.what()) == "invalid graph"); }
    }
    // Simulate encoder failure/cancellation while the platform load is active.
    // Unwinding must join and destroy the eventual resource on the owner thread.
    std::promise<void> entered, unwind;
    auto gate = unwind.get_future();
    try {
        tc::AsyncPreparation<Resource> task([&] {
            entered.set_value();
            gate.get();
            return std::make_unique<Resource>(alive, destroyed_on);
        });
        entered.get_future().get();
        unwind.set_value();
        throw std::runtime_error("encoder cancelled");
    } catch (const std::runtime_error &) {}
    assert(alive == 0 && destroyed_on == owner);
    // A failed background load must not replace the primary encoder exception.
    try {
        tc::AsyncPreparation<Resource> task([]() -> std::unique_ptr<Resource> {
            throw std::logic_error("background failure");
        });
        throw std::runtime_error("primary failure");
    } catch (const std::runtime_error &e) { assert(std::string(e.what()) == "primary failure"); }
    std::cout << "PASS async preparation ownership, join, errors and timing\n";
}

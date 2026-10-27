#pragma once
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "contracts.hpp"
namespace tc {
using Clock = std::chrono::steady_clock;
struct Event : std::function<void(const std::string &, int, int)> {
    using std::function<void(const std::string &, int, int)>::function;
    // Optional sidecar telemetry, never authority to publish a final artifact.
    std::function<void(const std::string &, int, int)> preview;
};
void require(bool, const std::string &);
struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("generation cancelled") {}
};
void checkpoint(std::atomic<bool> &);
} // namespace tc

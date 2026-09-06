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
using Event = std::function<void(const std::string &, int, int)>;
void require(bool, const std::string &);
struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("generation cancelled") {}
};
void checkpoint(std::atomic<bool> &);
} // namespace tc

#pragma once

// Private API stays in ane_program.mm, which public-distribution builds omit.
// Adapted from Splash #260, runtime/ane/Program.{hpp,mm}, 0ac3d5b.
// See THIRD_PARTY_NOTICES/splash-ane.md and Apache-2.0.txt.
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include "../ane_gpu.hpp"
#include "native_handoff_types.hpp"
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace tc::ane::private_api {

using gpu::Element;
using gpu::Device;
using gpu::Surface;
using gpu::Upload;
using gpu::Download;
using gpu::RowScalePolicy;
using gpu::Completion;
using gpu::QuantStage;
using gpu::Transfer;
inline constexpr size_t scale_cache_budget_bytes=gpu::scale_cache_budget_bytes;
class Program;
class Ticket;
class PreparedRequest;
std::filesystem::path default_cache_directory();

class Ticket {
  public:
    // Nonblocking observation; unlike finish(0), this cannot retire a request.
    // Only a driver callback proves physical_complete after submit. Event
    // safety-release, !timed_out and logical retirement do not grant permission
    // to read/reuse a surface; publication requires ok && physical_complete &&
    // !retired, plus the enclosing handoff's producer/epoch checks.
    Completion status() const;
    // Present only for explicit external-event requests. Immutable job identity;
    // it must match the enclosing completion domain before publishing output.
    std::optional<AneRequestProvenance> provenance() const;
    Completion finish(std::chrono::milliseconds timeout = std::chrono::seconds(30));
  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Program;
    friend class PreparedRequest;
};

// Bind/retain the immutable inputs and independent outputs without issuing
// driver work. Move-only; destroying an unsubmitted request breaks its
// completion-handler ownership cycle. submit() consumes it exactly once.
class PreparedRequest {
  public:
    ~PreparedRequest();
    PreparedRequest(PreparedRequest &&) noexcept;
    PreparedRequest &operator=(PreparedRequest &&) noexcept;
    PreparedRequest(const PreparedRequest &) = delete;
    PreparedRequest &operator=(const PreparedRequest &) = delete;
    Ticket submit();
  private:
    PreparedRequest() = default;
    std::shared_ptr<Ticket::Impl> state_;
    void discard() noexcept;
    friend class Program;
};

class Program {
  public:
    // A missing/failed completion disables private submissions process-wide;
    // do not accumulate an unbounded number of quarantined driver requests.
    static bool healthy();
    // Cache is keyed by SHA256(source, constants, OS build, device, ABI), with
    // framed inputs. Cache source files are checked byte-for-byte, not by size.
    Program(Device &, std::string_view mil, std::span<const uint8_t> constants,
            const std::filesystem::path &cache);
    const std::vector<std::string> &inputs() const;
    const std::vector<std::string> &outputs() const;
    const std::string &cache_key() const;
    uint64_t binding_generation() const;
    bool compiled_now() const;
    Ticket enqueue(std::span<const std::pair<std::string, Surface>> inputs,
                   std::span<const std::pair<std::string, Surface>> outputs,
                   uint64_t wait_value, uint64_t signal_value,
                   std::function<void()> failure = {});
    PreparedRequest prepare(std::span<const std::pair<std::string, Surface>> inputs,
                            std::span<const std::pair<std::string, Surface>> outputs,
                            uint64_t wait_value, uint64_t signal_value,
                            std::function<void()> failure = {});
    // W0 native bridge. Caller proves successful physical input completion and
    // exclusive epoch/slot ownership; submit rechecks ready/done Metal values.
    // Failure never automatically signals either external event. No output
    // read permission is implied by preparation, event advancement or timeout.
    PreparedRequest prepare(std::span<const std::pair<std::string, Surface>> inputs,
                            std::span<const std::pair<std::string, Surface>> outputs,
                            AneEventBinding events, std::function<void()> failure = {});
  private:
    PreparedRequest prepare_impl(std::span<const std::pair<std::string, Surface>> inputs,
                                 std::span<const std::pair<std::string, Surface>> outputs,
                                 uint64_t wait_value, uint64_t signal_value,
                                 std::optional<AneEventBinding> external_events,
                                 std::function<void()> failure);
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class PreparedRequest;
};
} // namespace tc::ane::private_api

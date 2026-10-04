#pragma once

// Private API stays in ane_program.mm, which public-distribution builds omit.
// Adapted from Splash #260, runtime/ane/Program.{hpp,mm}, 0ac3d5b.
// See THIRD_PARTY_NOTICES/splash-ane.md and Apache-2.0.txt.
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include "../ane_runtime.hpp"
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace tc::ane::private_api {

enum class Element { FP16, I8 };
class Device;
class Program;
class Ticket;
class Transfer;
class QuantStage;
inline constexpr size_t scale_cache_budget_bytes = 4u << 20;

class Surface {
  public:
    Surface(Device &, uint32_t rows, uint32_t columns, Element);
    size_t bytes() const;
    size_t pitch() const;
    uint32_t rows() const;
    uint32_t columns() const;
    Element element() const;
    void *data() const; // caller must not access while the ticket is live
    Surface slice_rows(uint32_t begin, uint32_t count) const; // consumer view only, not an ANE binding
  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    uint32_t row_begin_ = 0, row_count_ = 0;
    friend class Program;
    friend class Transfer;
    friend class Device;
};

struct Upload { DeviceMatrixView source; Surface destination; int begin_row = 0; float scale = 1.f; };
struct Download {
    Surface source;
    std::optional<DeviceMatrixView> destination; // null: finite/range validation only
    int begin_row = 0;
    DType dtype = DType::FP16;
    float scale = 1.f;
    std::optional<Surface> row_scales = std::nullopt, token_scales = std::nullopt;
    std::optional<Surface> second_token_scales = std::nullopt;
};

class Device {
  public:
    Device();
    std::string name() const;
    // Commit a separate leading command buffer. Do not defer the signal
    // until the command buffer containing the matching wait is committed.
    void signal(uint64_t value);
    bool wait(uint64_t value, std::chrono::milliseconds timeout);
    uint64_t value() const;
    Transfer prepare_transfer(std::vector<Upload>, std::vector<Download>,
                              uint64_t ready, uint64_t done);
    // A separate staging queue/event: next-layer readiness MUST NOT advance
    // the current layer's activation/ANE-done timeline.
    QuantStage stage_w8(DeviceWeightView, W8StageSpec, Surface codes, Surface scales);
    WeightCacheStats scale_cache_stats() const;
    StagePipelineStats stage_pipeline_stats() const;
  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Surface;
    friend class Program;
    friend class Transfer;
    friend class QuantStage;
};

struct Completion {
    bool ok = false, timed_out = false;
    std::string error;
};
class QuantStage {
  public:
    Completion finish(std::chrono::milliseconds timeout = std::chrono::seconds(30));
    uint32_t validation_flags() const;
    void *ready_event() const; // opaque MTLSharedEvent retained by this ticket
    uint64_t ready_value() const { return 1; }
  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Device;
};
// One input CB commits its ready signal before a separate output CB waits for
// ANE done. A sticky CPU failure flag suppresses output reads after failure/
// timeout; GPU validation has its own flag, never clearing the CPU flag.
class Transfer {
  public:
    std::function<void()> failure_callback() const;
    void submit();
    Completion finish(std::chrono::milliseconds timeout = std::chrono::seconds(30));
    uint32_t validation_flags() const;
  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Device;
};
class Ticket {
  public:
    Completion finish(std::chrono::milliseconds timeout = std::chrono::seconds(30));
  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
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
    bool compiled_now() const;
    Ticket enqueue(std::span<const std::pair<std::string, Surface>> inputs,
                   std::span<const std::pair<std::string, Surface>> outputs,
                   uint64_t wait_value, uint64_t signal_value,
                   std::function<void()> failure = {});
  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
} // namespace tc::ane::private_api

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
class PreparedRequest;
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
enum class RowScalePolicy : uint32_t { Positive, SignedFinite };
struct Download {
    Surface source;
    std::optional<DeviceMatrixView> destination; // null: finite/range validation only
    int begin_row = 0;
    DType dtype = DType::FP16;
    float scale = 1.f;
    std::optional<Surface> row_scales = std::nullopt, token_scales = std::nullopt;
    std::optional<Surface> second_token_scales = std::nullopt;
    // Direct ConvRot retains finite signed/zero checkpoint row scales. Token
    // quantizer scales stay strictly positive; Sylvester's default unchanged.
    RowScalePolicy row_scale_policy = RowScalePolicy::Positive;
};

class Device {
  public:
    Device();
    std::string name() const;
    // Commit a separate leading command buffer. Do not defer the signal
    // until the command buffer containing the matching wait is committed.
    void signal(uint64_t value);
    // Calibration only: all producers are already joined and all requests
    // prepared but not submitted. Meet their dependency on the CPU BEFORE
    // starting the timer; no Metal/ANE handoff is part of an alone sample.
    // Normal inference continues to signal from its GPU producer.
    void release_prepared(uint64_t value);
    bool wait(uint64_t value, std::chrono::milliseconds timeout);
    uint64_t value() const;
    Transfer prepare_transfer(std::vector<Upload>, std::vector<Download>,
                              uint64_t ready, uint64_t done);
    // Calibration GPU arm: one command buffer, no shared-event signal/wait
    // and no live ANE output dependency. Producers/source snapshots must be
    // ready and remain immutable until completion; normal inference uses
    // prepare_transfer(), never this independent restore path.
    Transfer prepare_gpu_transfer(std::vector<Upload>, std::vector<Download>);
    // A separate staging queue/event: next-layer readiness MUST NOT advance
    // the current layer's activation/ANE-done timeline.
    QuantStage stage_w8(DeviceWeightView, W8StageSpec, Surface codes, Surface scales);
    WeightCacheStats scale_cache_stats() const;
    StagePipelineStats stage_pipeline_stats() const;
  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    Transfer prepare_transfer_impl(std::vector<Upload>, std::vector<Download>,
                                  std::optional<std::pair<uint64_t, uint64_t>>);
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
    bool independent_gpu() const;
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
    bool compiled_now() const;
    Ticket enqueue(std::span<const std::pair<std::string, Surface>> inputs,
                   std::span<const std::pair<std::string, Surface>> outputs,
                   uint64_t wait_value, uint64_t signal_value,
                   std::function<void()> failure = {});
    PreparedRequest prepare(std::span<const std::pair<std::string, Surface>> inputs,
                            std::span<const std::pair<std::string, Surface>> outputs,
                            uint64_t wait_value, uint64_t signal_value,
                            std::function<void()> failure = {});
  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class PreparedRequest;
};
} // namespace tc::ane::private_api

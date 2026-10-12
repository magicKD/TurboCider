#pragma once

#include "backend.hpp"
#include "../private/ane_program.hpp"
#include <optional>

namespace tc::metal_graph {

// Private-distribution W0/W1 submission bridge. No MLX or long-lived worker. All
// controlling calls on one handoff must be serialized on one host thread.
using NamedAneSurface = std::pair<std::string, tc::ane::gpu::Surface>;

struct ImportedAneSurface {
    tc::ane::gpu::SurfaceBufferLease lease;
    Buffer buffer;
    BufferView view;
};
// Imports the existing IOSurface backing; no allocation or second ledger charge.
// This returns storage, not a producer-completion or output-read certificate.
ImportedAneSurface import_ane_surface(Backend&, const tc::ane::gpu::Surface&);

struct NativeAneJobIdentity {
    uint64_t domain = 0, epoch = 0, request = 0;
    uint64_t slot_generation = 0, producer_generation = 0;
};

struct NativeAneHandoffStatus {
    bool ok = false, retired = false, timed_out = false;
    bool producer_submitted = false, producer_physical_complete = false;
    bool ane_submission_attempted = false, ane_physical_complete = false;
    bool all_work_physically_complete = false;
    uint64_t ready_value = 0, done_value = 0;
    std::string error;
    // Diagnostic from the captured producer ticket only. Zero is unavailable.
    double producer_gpu_span_seconds = 0;
};

namespace detail {
struct NativeAneHandoffState;
inline bool same_ane_request(const tc::ane::private_api::AneRequestProvenance &a,
                             const tc::ane::private_api::AneRequestProvenance &b) {
    return a.program_key == b.program_key && a.program_generation == b.program_generation &&
           a.domain == b.domain && a.epoch == b.epoch && a.request == b.request &&
           a.slot_generation == b.slot_generation && a.producer_generation == b.producer_generation &&
           a.function_index == b.function_index;
}
} // namespace detail

// Proof of this handoff's captured producer ticket and fresh ready reservation.
// It grants CPU observation of caller-bound producer status, NOT quantization
// validity, ANE output access, or a generic certificate for another graph.
// The controller must check its own W8/A8 flags and matching generation before
// consuming this permit with submit_ane(). External callers cannot mint it.
class NativeAneProducerPermit {
  public:
    NativeAneProducerPermit() = default;
    NativeAneProducerPermit(const NativeAneProducerPermit&) = delete;
    NativeAneProducerPermit &operator=(const NativeAneProducerPermit&) = delete;
    NativeAneProducerPermit(NativeAneProducerPermit&&) noexcept = default;
    NativeAneProducerPermit &operator=(NativeAneProducerPermit&&) noexcept = default;
    bool is_valid() const noexcept;
    explicit operator bool() const noexcept { return is_valid(); }
    const tc::ane::private_api::AneRequestProvenance &provenance() const;
  private:
    explicit NativeAneProducerPermit(std::shared_ptr<detail::NativeAneHandoffState> state)
        : state_(std::move(state)) {}
    std::shared_ptr<detail::NativeAneHandoffState> state_;
    friend class NativeAneHandoff;
};

// Only NativeAneHandoff can grant this permit after checking the actual captured
// producer ticket and actual private ticket. It retains output/event/program
// owners. When encoding restore, copy each output BufferView into the graph so
// the graph ticket retains the backing until that restore physically completes.
// Exposed views are for readers: never overwrite an ANE output through a permit.
class NativeAneReadPermit {
  public:
    NativeAneReadPermit() = default;
    NativeAneReadPermit(const NativeAneReadPermit&) = delete;
    NativeAneReadPermit &operator=(const NativeAneReadPermit&) = delete;
    NativeAneReadPermit(NativeAneReadPermit&&) noexcept = default;
    NativeAneReadPermit &operator=(NativeAneReadPermit&&) noexcept = default;
    explicit operator bool() const noexcept { return bool(state_); }
    const ImportedAneSurface &output(std::string_view name) const;
    const tc::ane::private_api::AneRequestProvenance &provenance() const;
  private:
    explicit NativeAneReadPermit(std::shared_ptr<detail::NativeAneHandoffState> state)
        : state_(std::move(state)) {}
    std::shared_ptr<detail::NativeAneHandoffState> state_;
    friend class NativeAneHandoff;
};

class NativeAneHandoff {
  public:
    // Program is a retained copy of the existing prepared model; no compilation
    // happens here. Backend must outlive submit_producer(). Inputs and outputs
    // are separate full surfaces; external code must not mutate or reuse them
    // until all physical work finishes, including after logical retirement.
    NativeAneHandoff(Backend&, tc::ane::private_api::Program program,
                     std::vector<NamedAneSurface> inputs, std::vector<NamedAneSurface> outputs,
                     NativeAneJobIdentity);
    NativeAneHandoff(const NativeAneHandoff&) = delete;
    NativeAneHandoff &operator=(const NativeAneHandoff&) = delete;
    NativeAneHandoff(NativeAneHandoff&&) noexcept = default;
    NativeAneHandoff &operator=(NativeAneHandoff&&) noexcept = default;
    ~NativeAneHandoff() = default; // never blocks; underlying tickets retain live owners

    const ImportedAneSurface &input(std::string_view name) const;
    const tc::ane::private_api::AneRequestProvenance &provenance() const;
    // This method appends its own fresh ready signal, then captures precisely
    // that producer Ticket. It cannot accept an unrelated completed ticket.
    // Caller is responsible for writing ALL inputs/weights (or retaining their
    // already-produced immutable snapshot): GraphStep has no write-set metadata.
    void submit_producer(CommandGraph graph);
    using Deadline = std::chrono::steady_clock::time_point;
    // One-shot producer admission, using only the ticket captured above. Never
    // accepts a caller-provided completion or verification callback.
    NativeAneProducerPermit finish_producer(std::chrono::milliseconds timeout = std::chrono::seconds(30));
    NativeAneProducerPermit finish_producer_until(Deadline);
    // Consumes a permit from THIS state and prepares/submits exactly one ANE
    // request. Does not wait for ANE. The generic bridge cannot certify W8/A8
    // flags; its caller must validate flags before transferring the permit.
    // The submit deadline is retained; a later finish cannot extend it.
    void submit_ane(NativeAneProducerPermit, std::chrono::milliseconds timeout = std::chrono::seconds(30));
    void submit_ane_until(NativeAneProducerPermit, Deadline);
    NativeAneReadPermit finish_ane(std::chrono::milliseconds timeout = std::chrono::seconds(30));
    NativeAneReadPermit finish_ane_until(Deadline);
    // *_until methods share the controller's absolute step deadline. A blocking
    // driver prepare/submit cannot be preempted; an overrun is rejected after
    // return, retaining any submitted owners. No timeout force-signals events.
    // W0 host wait: join GPU producer -> prepare/submit ANE -> join callback ->
    // grant one result permit. Never prequeues restore and never force-signals.
    // ANE budget starts after successful producer readiness, includes request
    // preparation/submission, and cannot resurrect a retired result.
    NativeAneReadPermit finish(std::chrono::milliseconds producer_timeout = std::chrono::seconds(30),
                              std::chrono::milliseconds ane_timeout = std::chrono::seconds(30));
    // Only producer + ANE are covered. GPU complement and restore/join tickets
    // remain the enclosing controller's responsibility. Unknown submission
    // without a returned private ticket remains physically incomplete forever.
    NativeAneHandoffStatus status() const;
  private:
    std::shared_ptr<detail::NativeAneHandoffState> state_;
};

} // namespace tc::metal_graph

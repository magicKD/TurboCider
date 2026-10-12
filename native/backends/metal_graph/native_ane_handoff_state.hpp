#pragma once

#include "backend.hpp"
#include "../ane_gpu.hpp"

namespace tc::metal_graph::detail {

using AneDeadline = std::chrono::steady_clock::time_point;
// Round up only the wait argument; callers check the original absolute end
// after every blocking call. A positive sub-millisecond interval is not zero.
inline std::chrono::milliseconds ane_remaining(AneDeadline end, AneDeadline now) {
    return end <= now ? std::chrono::milliseconds(0)
                     : std::chrono::ceil<std::chrono::milliseconds>(end - now);
}
inline AneDeadline ane_deadline(std::chrono::milliseconds timeout, AneDeadline now) {
    if (timeout.count() < 0)
        throw tc::ane::CapabilityError("native ANE negative deadline");
    if (!timeout.count()) return now;
    const auto maximum = std::chrono::duration_cast<std::chrono::milliseconds>(AneDeadline::max() - now);
    return timeout >= maximum ? AneDeadline::max() : now + timeout;
}

// CPU-only state machine used by the real bridge, not a public certificate.
// Only the bridge's private constructors create permits from captured tickets.
// No receipt or event alone can advance a retired logical result.
class NativeAnePermitProtocol {
  public:
    enum class Phase { ProducerPending, ProducerReady, Preparing, Submitted, Readable, Retired };
    Phase phase() const noexcept { return phase_; }
    bool retired() const noexcept { return phase_ == Phase::Retired; }
    bool readable() const noexcept { return phase_ == Phase::Readable; }
    bool timed_out() const noexcept { return timed_out_; }
    const std::string &error() const noexcept { return error_; }
    void retire(std::string diagnostic, bool timeout = false) {
        if (retired()) return;
        phase_ = Phase::Retired; timed_out_ = timeout; error_ = std::move(diagnostic);
    }
    void producer_complete(const Completion &receipt, uint64_t ready, uint64_t done) {
        expect(Phase::ProducerPending, "native ANE producer permit already issued/retired");
        if (!receipt.ok || !receipt.physically_complete || receipt.timed_out ||
            !receipt.submitted_buffers || receipt.completed_buffers != receipt.submitted_buffers ||
            ready != 1 || done != 0) {
            retire(receipt.error.empty() ? "native ANE producer receipt/event mismatch" : receipt.error,
                   receipt.timed_out);
            throw tc::ane::CapabilityError(error_);
        }
        phase_ = Phase::ProducerReady;
    }
    void consume_producer(bool same_owner) {
        expect(Phase::ProducerReady, "native ANE producer permit unavailable/consumed");
        if (!same_owner) throw tc::ane::CapabilityError("native ANE producer permit belongs to another handoff");
        phase_ = Phase::Preparing;
    }
    void submitted() {
        expect(Phase::Preparing, "native ANE request is not prepared for submission");
        phase_ = Phase::Submitted;
    }
    void ane_complete(const tc::ane::gpu::Completion &receipt, bool provenance_matches,
                      uint64_t ready, uint64_t done) {
        expect(Phase::Submitted, "native ANE result permit unavailable/already issued");
        if (!receipt.ok || !receipt.physical_complete || receipt.retired || receipt.timed_out ||
            !provenance_matches || ready != 1 || done != 1) {
            retire(receipt.error.empty() ? "native ANE callback/provenance/event mismatch" : receipt.error,
                   receipt.timed_out);
            throw tc::ane::CapabilityError(error_);
        }
        phase_ = Phase::Readable;
    }
  private:
    void expect(Phase wanted, const char *message) const {
        if (phase_ != wanted) throw tc::ane::CapabilityError(retired() ? error_ : message);
    }
    Phase phase_ = Phase::ProducerPending;
    bool timed_out_ = false;
    std::string error_;
};

} // namespace tc::metal_graph::detail

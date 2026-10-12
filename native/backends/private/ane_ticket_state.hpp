#pragma once

#include "../ane_gpu.hpp"
#include <utility>

namespace tc::ane::private_api::detail {

// CPU-only state transitions shared by the real driver ticket and contract
// tests. Its caller serializes access. Logical retirement and physical driver
// completion are deliberately independent; a safety-released event is neither.
class TicketCompletionState {
  public:
    const gpu::Completion &status() const { return result_; }
    bool terminal() const { return result_.retired || result_.physical_complete; }

    // First failure owns the diagnostic and failure-release action. Rejection
    // must also retire a success callback that raced evaluateWithModel's return.
    bool retire(std::string error, bool timed_out = false) {
        if (result_.retired) return false;
        result_.ok = false;
        result_.timed_out = timed_out;
        result_.error = std::move(error);
        result_.retired = true;
        return true;
    }

    // Returns true only when this callback causes the first retirement. Even
    // after timeout/rejection, a late callback supplies physical completion
    // evidence without resurrecting success or replacing the first diagnostic.
    bool completed(bool ok, std::string error = {}) {
        if (result_.physical_complete) return false;
        result_.physical_complete = true;
        if (result_.retired) return false;
        if (!ok) return retire(std::move(error));
        result_.ok = true;
        return false;
    }

    // Valid only before submit: no driver work exists, so resources are idle.
    // This is not an output-success path and cannot issue a ResultReadPermit.
    void discard_unsubmitted() noexcept {
        // Destruction must not allocate a diagnostic or throw.
        result_.ok = false;
        result_.retired = true;
        result_.physical_complete = true;
    }

  private:
    gpu::Completion result_;
};

} // namespace tc::ane::private_api::detail

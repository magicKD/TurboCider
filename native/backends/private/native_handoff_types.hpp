#pragma once

#include "../ane_runtime.hpp"
#include <string_view>
#include <utility>

namespace tc::ane::private_api {

enum class AneEventRole : uint8_t { GpuReady, AneDone };

// Opaque native MTLSharedEvent. owner must retain the actual event through
// driver completion; the Program additionally retains the native ObjC object.
// Handles are trusted native imports, not arbitrary unvalidated user pointers.
struct AneEventEndpoint {
    void *native_event = nullptr, *native_device = nullptr;
    std::shared_ptr<void> owner;
    uint64_t value = 0;
    AneEventRole role = AneEventRole::GpuReady;
};

namespace detail {
// MTLEvent.device is explicitly nullable for MTLSharedEvent (Metal SDK header).
// Preserve the retained endpoint's creator-device provenance even when Metal
// reports no single device. A non-null reported device must still agree.
inline void validate_shared_event_device(void *creator_device, void *reported_device, void *expected_device) {
    if (!expected_device || creator_device != expected_device ||
        (reported_device && reported_device != expected_device))
        throw CapabilityError("private ANE external event device provenance mismatch");
}
} // namespace detail

struct AneRequestProvenance {
    std::string program_key;
    uint64_t program_generation = 0; // Program::binding_generation(), not a cache identity
    uint64_t domain = 0, epoch = 0, request = 0;
    uint64_t slot_generation = 0, producer_generation = 0;
    uint32_t function_index = 0; // current Program only exposes procedure 0
};

// One request's event reservation. Move-only: prepare consumes the binding,
// then PreparedRequest consumes a fresh driver request exactly once. Creating
// another binding for the same reservation is forbidden by the caller's
// CompletionDomain; these metadata checks do not supply a global epoch ledger.
// Prefer separate ready/done events so an ANE completion cannot advance ready.
class AneEventBinding {
  public:
    AneEventBinding(AneEventEndpoint ready, AneEventEndpoint done, AneRequestProvenance provenance)
        : ready_(std::move(ready)), done_(std::move(done)), provenance_(std::move(provenance)) {}
    AneEventBinding(AneEventBinding &&) noexcept = default;
    AneEventBinding &operator=(AneEventBinding &&) noexcept = default;
    AneEventBinding(const AneEventBinding &) = delete;
    AneEventBinding &operator=(const AneEventBinding &) = delete;
    const AneEventEndpoint &ready() const { return ready_; }
    const AneEventEndpoint &done() const { return done_; }
    const AneRequestProvenance &provenance() const { return provenance_; }

    // CPU-only metadata checks, also run against actual Metal object devices by
    // Program. No event value, role or declared generation proves a producer's
    // semantic success. NativeHandoff must separately bind a completion permit
    // (GPU ok, physically complete, not retired) to this exact provenance.
    void validate_metadata(void *expected_device, std::string_view program_key,
                           uint64_t program_generation) const {
        if (!expected_device || !ready_.native_event || !done_.native_event ||
            !ready_.owner || !done_.owner || ready_.native_device != expected_device ||
            done_.native_device != expected_device || ready_.role != AneEventRole::GpuReady ||
            done_.role != AneEventRole::AneDone || !ready_.value || !done_.value ||
            (ready_.native_event == done_.native_event && done_.value <= ready_.value) ||
            program_key.empty() || provenance_.program_key != program_key ||
            !program_generation || provenance_.program_generation != program_generation ||
            !provenance_.domain || !provenance_.epoch || !provenance_.request ||
            !provenance_.slot_generation || !provenance_.producer_generation || provenance_.function_index != 0)
            throw CapabilityError("private ANE external event binding metadata mismatch");
    }

    // W0 admission only: all input producers must already be successfully
    // joined by the caller. Exact ready value rejects advancement by another
    // epoch; done must remain unsatisfied. This is not W1 admission or a permit.
    void validate_w0_values(uint64_t ready_current, uint64_t done_current) const {
        if (ready_current != ready_.value || done_current >= done_.value ||
            (ready_.native_event == done_.native_event && ready_current != done_current))
            throw CapabilityError("private ANE W0 event reservation is unready, stale or already completed");
    }

  private:
    AneEventEndpoint ready_, done_;
    AneRequestProvenance provenance_;
};

} // namespace tc::ane::private_api

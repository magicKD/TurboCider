#pragma once

#include "native_ane_handoff.hpp"

namespace tc::metal_graph {

// Receipts of the three independently submitted legs, not event-value guesses.
// An attempted submission without a ticket is deliberately unknown forever.
struct HybridPhysicalEvidence {
    NativeAneHandoffStatus handoff;
    bool gpu_attempted = false, consumer_attempted = false;
    std::optional<Completion> gpu, consumer;
};

inline bool hybrid_all_physically_complete(const HybridPhysicalEvidence& e) noexcept {
    const auto complete = [](bool attempted, const std::optional<Completion>& ticket) {
        return !attempted || (ticket && ticket->physically_complete);
    };
    return e.handoff.all_work_physically_complete && complete(e.gpu_attempted, e.gpu) &&
           complete(e.consumer_attempted, e.consumer);
}

inline bool hybrid_publishable(const HybridPhysicalEvidence& e, bool retired,
                               bool validated_arithmetic) noexcept {
    const auto successful = [](const std::optional<Completion>& ticket) {
        return ticket && ticket->ok && ticket->physically_complete && !ticket->timed_out;
    };
    return !retired && validated_arithmetic && e.handoff.ok && !e.handoff.retired &&
           e.gpu_attempted && e.consumer_attempted && successful(e.gpu) &&
           successful(e.consumer) && hybrid_all_physically_complete(e);
}

} // namespace tc::metal_graph

#include "native_ane_handoff.hpp"
#include "native_ane_handoff_state.hpp"
#include <algorithm>
#include <set>
#include <cmath>

namespace tc::metal_graph {
namespace {
using tc::ane::CapabilityError;
using tc::ane::private_api::AneEventBinding;
using tc::ane::private_api::AneEventRole;
using tc::ane::private_api::AneRequestProvenance;
constexpr uint64_t event_target = 1;
using Clock = std::chrono::steady_clock;
void require(bool value, const char *message) { if (!value) throw CapabilityError(message); }

struct NamedImport { std::string name; ImportedAneSurface surface; };
const ImportedAneSurface &find_surface(const std::vector<NamedImport> &surfaces, std::string_view name) {
    const auto it = std::find_if(surfaces.begin(), surfaces.end(), [&](const auto &s) { return s.name == name; });
    if (it == surfaces.end()) throw CapabilityError("native ANE unknown surface: " + std::string(name));
    return it->surface;
}
void validate_names(const std::vector<NamedAneSurface> &surfaces, const std::vector<std::string> &expected) {
    require(surfaces.size() == expected.size(), "native ANE surface count does not match program");
    std::set<std::string> observed;
    for (const auto &[name, surface] : surfaces) {
        require(observed.insert(name).second && std::find(expected.begin(), expected.end(), name) != expected.end(),
                "native ANE duplicate/unknown surface name");
        require(!surface.is_view(), "native ANE driver binding requires a full surface");
    }
}
} // namespace

ImportedAneSurface import_ane_surface(Backend &backend, const tc::ane::gpu::Surface &surface) {
    auto lease = surface.metal_buffer_lease();
    lease.validate(backend.native_device());
    auto buffer = backend.import_buffer(lease.native_buffer, lease.owner);
    require(buffer.size_bytes() == lease.backing_bytes, "native ANE imported buffer extent changed");
    const auto type = lease.element == tc::ane::gpu::Element::FP16 ? ScalarType::FP16 : ScalarType::Int8;
    auto view = buffer.view(lease.offset_bytes, lease.size_bytes, type);
    return {std::move(lease), std::move(buffer), std::move(view)};
}

namespace detail {
struct NativeAneHandoffState {
    Backend *backend;
    void *native_device;
    tc::ane::private_api::Program program;
    std::vector<NamedAneSurface> inputs, outputs;
    std::vector<NamedImport> imported_inputs, imported_outputs;
    std::shared_ptr<SharedEvent> ready, done;
    AneRequestProvenance provenance;
    std::optional<Ticket> producer;
    std::optional<tc::ane::private_api::Ticket> ane;
    bool producer_attempted = false, ane_attempted = false;
    NativeAnePermitProtocol protocol;
    // submit_ane captures its end; finish_ane may shorten but never extend it.
    std::optional<Clock::time_point> ane_deadline;

    NativeAneHandoffState(Backend &b, tc::ane::private_api::Program p,
                         std::vector<NamedAneSurface> in, std::vector<NamedAneSurface> out)
        : backend(&b), native_device(b.native_device()), program(std::move(p)), inputs(std::move(in)), outputs(std::move(out)) {}
    void retire(std::string diagnostic, bool timeout = false) {
        protocol.retire(std::move(diagnostic), timeout);
    }
    AneEventBinding binding() const {
        return {{ready->native_handle(), native_device, ready, event_target, AneEventRole::GpuReady},
                {done->native_handle(), native_device, done, event_target, AneEventRole::AneDone},
                provenance};
    }
};
} // namespace detail

NativeAneHandoff::NativeAneHandoff(Backend &backend, tc::ane::private_api::Program program,
                                 std::vector<NamedAneSurface> inputs, std::vector<NamedAneSurface> outputs,
                                 NativeAneJobIdentity identity) {
    require(identity.domain && identity.epoch && identity.request && identity.slot_generation && identity.producer_generation,
            "native ANE job provenance must be nonzero");
    validate_names(inputs, program.inputs()); validate_names(outputs, program.outputs());
    auto state = std::make_shared<detail::NativeAneHandoffState>(backend, std::move(program), std::move(inputs), std::move(outputs));
    state->provenance = {state->program.cache_key(), state->program.binding_generation(), identity.domain, identity.epoch,
                         identity.request, identity.slot_generation, identity.producer_generation, 0};
    for (const auto &[name, surface] : state->inputs)
        state->imported_inputs.push_back({name, import_ane_surface(backend, surface)});
    for (const auto &[name, surface] : state->outputs) {
        auto imported = import_ane_surface(backend, surface);
        for (const auto &previous : state->imported_inputs)
            require(previous.surface.lease.native_buffer != imported.lease.native_buffer, "native ANE output aliases input");
        for (const auto &previous : state->imported_outputs)
            require(previous.surface.lease.native_buffer != imported.lease.native_buffer, "native ANE outputs alias");
        state->imported_outputs.push_back({name, std::move(imported)});
    }
    // Separate, fresh events avoid timeline reuse and failure-release ambiguity.
    state->ready = std::make_shared<SharedEvent>(backend.make_event(EventSource::ThisQueue));
    state->done = std::make_shared<SharedEvent>(backend.make_event(EventSource::External));
    state->binding().validate_metadata(backend.native_device(), state->program.cache_key(), state->program.binding_generation());
    state_ = std::move(state);
}

const ImportedAneSurface &NativeAneHandoff::input(std::string_view name) const {
    require(bool(state_), "native ANE empty handoff");
    require(!state_->producer_attempted, "native ANE input encoding is closed after producer submission");
    return find_surface(state_->imported_inputs, name);
}
const AneRequestProvenance &NativeAneHandoff::provenance() const {
    require(bool(state_), "native ANE empty handoff");
    return state_->provenance;
}
void NativeAneHandoff::submit_producer(CommandGraph graph) {
    require(bool(state_) && !state_->producer_attempted && !state_->protocol.retired(), "native ANE producer already submitted/retired");
    // No caller-supplied ticket or caller-generated ready event is trusted.
    graph.signal(*state_->ready, event_target);
    state_->producer_attempted = true;
    try {
        state_->producer.emplace(state_->backend->submit(std::move(graph)));
    } catch (const std::exception &error) {
        // Backend throws only before Metal submission; partial submissions have
        // a retained failed Ticket. No ANE request exists on this path.
        state_->producer_attempted = false;
        state_->retire(error.what());
        throw;
    } catch (...) {
        state_->producer_attempted = false;
        state_->retire("native ANE unknown producer submission failure");
        throw;
    }
}

NativeAneProducerPermit NativeAneHandoff::finish_producer(std::chrono::milliseconds timeout) {
    return finish_producer_until(detail::ane_deadline(timeout, Clock::now()));
}
NativeAneProducerPermit NativeAneHandoff::finish_producer_until(Deadline end) {
    require(bool(state_) && state_->producer.has_value(), "native ANE producer has not been submitted");
    require(state_->protocol.phase() == detail::NativeAnePermitProtocol::Phase::ProducerPending,
            "native ANE producer permit already issued/retired");
    try {
        const auto producer = state_->producer->wait(detail::ane_remaining(end, Clock::now()));
        if (Clock::now() >= end) {
            state_->retire("native ANE producer completion deadline", true);
            throw CapabilityError(state_->protocol.error());
        }
        state_->protocol.producer_complete(producer, state_->ready->value(), state_->done->value());
        return NativeAneProducerPermit(state_);
    } catch (const std::exception &error) {
        state_->retire(error.what());
        throw;
    } catch (...) {
        state_->retire("native ANE unknown producer completion failure");
        throw;
    }
}

void NativeAneHandoff::submit_ane(NativeAneProducerPermit permit, std::chrono::milliseconds timeout) {
    submit_ane_until(std::move(permit), detail::ane_deadline(timeout, Clock::now()));
}
void NativeAneHandoff::submit_ane_until(NativeAneProducerPermit permit, Deadline end) {
    require(bool(state_), "native ANE empty handoff");
    // Pointer identity is to this private retained state, not a caller-written
    // generation field. Transfer consumes exactly the issued producer permit.
    state_->protocol.consume_producer(permit.state_ == state_);
    permit.state_.reset();
    state_->ane_deadline = end;
    try {
        if (Clock::now() >= end) {
            state_->retire("native ANE admission deadline", true);
            throw CapabilityError(state_->protocol.error());
        }
        // Recheck actual provenance/event reservations after the caller's flag
        // checks. Inputs remain immutable; no arbitrary validator is invoked.
        auto binding = state_->binding();
        binding.validate_w0_values(state_->ready->value(), state_->done->value());
        auto prepared = state_->program.prepare(state_->inputs, state_->outputs, std::move(binding));
        if (Clock::now() >= end) {
            state_->retire("native ANE request preparation deadline", true);
            throw CapabilityError(state_->protocol.error()); // unsubmitted PreparedRequest breaks its own callback cycle
        }
        // A throwing/NO submission may have reached the driver. Keep that fact
        // distinct from possessing a returned ticket; private request owners
        // remain quarantined by its callback cycle if no callback is observed.
        state_->ane_attempted = true;
        state_->protocol.submitted();
        state_->ane.emplace(prepared.submit());
        if (Clock::now() >= end) {
            state_->retire("native ANE submission deadline", true);
            (void)state_->ane->finish(std::chrono::milliseconds(0)); // actual expired deadline, never polling
            throw CapabilityError(state_->protocol.error());
        }
        // Inspect immediate rejection without waiting. A NO/exception does not
        // prove physical completion; the ticket/callback retain live owners.
        const auto observed = state_->ane->status();
        if (observed.retired || (observed.physical_complete && !observed.ok)) {
            state_->retire(observed.error.empty() ? "native ANE submission rejected" : observed.error,
                           observed.timed_out);
            throw CapabilityError(state_->protocol.error());
        }
    } catch (const std::exception &error) {
        state_->retire(error.what());
        throw;
    } catch (...) {
        state_->retire("native ANE unknown preparation/submission failure");
        throw;
    }
}

NativeAneReadPermit NativeAneHandoff::finish_ane(std::chrono::milliseconds timeout) {
    return finish_ane_until(detail::ane_deadline(timeout, Clock::now()));
}
NativeAneReadPermit NativeAneHandoff::finish_ane_until(Deadline end) {
    require(bool(state_) && state_->ane.has_value() && state_->ane_deadline.has_value(),
            "native ANE request has no returned submission ticket");
    require(state_->protocol.phase() == detail::NativeAnePermitProtocol::Phase::Submitted,
            "native ANE result permit unavailable/already issued/retired");
    end = std::min(end, *state_->ane_deadline);
    try {
        const auto ane = state_->ane->finish(detail::ane_remaining(end, Clock::now()));
        const auto receipt = state_->ane->provenance();
        if (Clock::now() >= end) {
            state_->retire("native ANE completion deadline", true);
            throw CapabilityError(state_->protocol.error());
        }
        state_->protocol.ane_complete(ane, receipt && detail::same_ane_request(*receipt, state_->provenance),
                                     state_->ready->value(), state_->done->value());
        return NativeAneReadPermit(state_);
    } catch (const std::exception &error) {
        state_->retire(error.what());
        throw;
    } catch (...) {
        state_->retire("native ANE unknown completion failure");
        throw;
    }
}

NativeAneReadPermit NativeAneHandoff::finish(std::chrono::milliseconds producer_timeout,
                                           std::chrono::milliseconds ane_timeout) {
    require(producer_timeout.count() >= 0 && ane_timeout.count() >= 0, "native ANE negative deadline");
    auto producer = finish_producer(producer_timeout);
    const auto end = detail::ane_deadline(ane_timeout, Clock::now());
    submit_ane_until(std::move(producer), end);
    return finish_ane_until(end);
}

NativeAneHandoffStatus NativeAneHandoff::status() const {
    if (!state_) return {false, true, false, false, false, false, false, true, 0, 0, "native ANE empty handoff"};
    const auto producer = state_->producer ? state_->producer->status() : Completion{};
    const auto ane = state_->ane ? state_->ane->status() : tc::ane::gpu::Completion{};
    NativeAneHandoffStatus result;
    result.retired = state_->protocol.retired() || producer.timed_out || ane.retired;
    result.ok = state_->protocol.readable() && !result.retired && producer.ok && producer.physically_complete &&
                ane.ok && ane.physical_complete;
    result.timed_out = state_->protocol.timed_out() || producer.timed_out || ane.timed_out;
    result.producer_submitted = state_->producer_attempted;
    result.producer_physical_complete = producer.physically_complete;
    result.ane_submission_attempted = state_->ane_attempted;
    result.ane_physical_complete = ane.physical_complete;
    result.all_work_physically_complete = (!state_->producer_attempted || producer.physically_complete) &&
                                         (!state_->ane_attempted || ane.physical_complete);
    result.ready_value = state_->ready->value(); result.done_value = state_->done->value();
    result.error = !state_->protocol.error().empty() ? state_->protocol.error() : !producer.error.empty() ? producer.error : ane.error;
    if (state_->producer && producer.ok && producer.physically_complete &&
        std::isfinite(producer.gpu_seconds) && producer.gpu_seconds > 0)
        result.producer_gpu_span_seconds = producer.gpu_seconds;
    return result;
}

const ImportedAneSurface &NativeAneReadPermit::output(std::string_view name) const {
    require(bool(state_) && state_->protocol.readable(), "native ANE output has no successful read permit");
    return find_surface(state_->imported_outputs, name);
}
const AneRequestProvenance &NativeAneReadPermit::provenance() const {
    require(bool(state_) && state_->protocol.readable(), "native ANE empty/retired result permit");
    return state_->provenance;
}

const AneRequestProvenance &NativeAneProducerPermit::provenance() const {
    require(is_valid(),
            "native ANE empty/consumed/retired producer permit");
    return state_->provenance;
}
bool NativeAneProducerPermit::is_valid() const noexcept {
    // This phase is reachable only once, from the captured successful producer
    // receipt. Callers serialize controlling methods; callbacks cannot promote
    // or retire protocol state behind an outstanding producer permit.
    return state_ && state_->protocol.phase() == detail::NativeAnePermitProtocol::Phase::ProducerReady;
}

} // namespace tc::metal_graph

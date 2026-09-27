#include "run_context.hpp"

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace tc::streaming {
namespace {

std::atomic<uint64_t> g_next_request_generation{1};

struct QuarantineRegistry final {
    std::mutex mutex;
    std::vector<std::unique_ptr<StageExecutor>> executors;
    std::vector<std::string> reasons;
};

QuarantineRegistry &quarantine_registry() {
    // Unsafe GPU/worker-visible backing must not be destroyed during static
    // teardown. A later explicit retry API can reclaim entries; until then the
    // registry intentionally has process lifetime.
    static auto *value = new QuarantineRegistry;
    return *value;
}

uint64_t next_request_generation() {
    uint64_t value = g_next_request_generation.fetch_add(
        1, std::memory_order_relaxed);
    if (value != 0) return value;
    value = g_next_request_generation.fetch_add(1, std::memory_order_relaxed);
    if (value == 0) throw std::overflow_error("streaming_request_generation_overflow");
    return value;
}

void require_context(bool condition, const char *detail) {
    if (!condition) throw std::logic_error(detail);
}

} // namespace

struct PublicStreamingRunContext::State {
    std::shared_ptr<const ResolvedRequestExecution> execution;
    std::atomic<bool> *cancel = nullptr;
    uint64_t request_generation = 0;
    PublicRunPhase phase = PublicRunPhase::created;
    std::vector<std::unique_ptr<StageExecutor>> executors;
    std::vector<std::shared_ptr<const ActualStageReceipt>> stage_receipts;
    std::vector<ActualBoundaryReceipt> boundaries;
    std::string quarantine_reason;
};

PublicStreamingRunContext::PublicStreamingRunContext(
        std::shared_ptr<const ResolvedRequestExecution> execution,
        std::atomic<bool> &cancel)
    : state_(std::make_unique<State>()) {
    require_context(execution != nullptr, "streaming_context_missing_execution");
    require_context(execution->model_snapshot != nullptr,
                    "streaming_context_missing_snapshot");
    require_context(execution->probe != nullptr,
                    "streaming_context_missing_probe");
    require_context(execution->selection.authority != nullptr,
                    "streaming_context_missing_authority");
    require_context(execution->model_snapshot->source_lease() != nullptr,
                    "streaming_context_missing_lease");
    require_context(execution->probe->source_lease() ==
                        execution->model_snapshot->source_lease(),
                    "streaming_context_lease_mismatch");
    state_->execution = std::move(execution);
    state_->cancel = &cancel;
    state_->request_generation = next_request_generation();
    state_->executors.resize(state_->execution->model_snapshot->layout().stages.size());
    state_->stage_receipts.resize(state_->executors.size());
}

PublicStreamingRunContext::~PublicStreamingRunContext() {
    if (!state_ || state_->phase == PublicRunPhase::completed ||
        state_->phase == PublicRunPhase::created ||
        state_->phase == PublicRunPhase::gpu_revalidated ||
        state_->phase == PublicRunPhase::receipt_sealed ||
        state_->phase == PublicRunPhase::source_revalidated ||
        state_->phase == PublicRunPhase::quarantined)
        return;
    try {
        drain_all();
    } catch (const std::exception &error) {
        quarantine(error.what());
    } catch (...) {
        quarantine("streaming_context_destructor_failure");
    }
}

const ResolvedRequestExecution &PublicStreamingRunContext::execution() const noexcept {
    return *state_->execution;
}

const SourceLease &PublicStreamingRunContext::lease() const {
    require_context(state_ && state_->execution &&
                        state_->execution->model_snapshot &&
                        state_->execution->model_snapshot->source_lease(),
                    "streaming_context_missing_lease");
    return *state_->execution->model_snapshot->source_lease();
}

uint64_t PublicStreamingRunContext::source_generation() const {
    return lease().generation();
}

uint64_t PublicStreamingRunContext::request_generation() const noexcept {
    return state_ ? state_->request_generation : 0;
}

StageExecutor &PublicStreamingRunContext::attach_stage(
        uint32_t stage_index, std::shared_ptr<ModelSlotAdapter> adapter,
        std::string implementation) {
    require_context(state_ &&
                        (state_->phase == PublicRunPhase::gpu_revalidated ||
                         state_->phase == PublicRunPhase::running),
                    "streaming_context_not_attachable");
    require_context(stage_index < state_->executors.size(),
                    "streaming_context_stage_out_of_range");
    require_context(stage_index == 0 ||
                        (state_->executors[stage_index - 1] != nullptr &&
                         state_->stage_receipts[stage_index - 1] != nullptr &&
                         state_->boundaries.size() >= stage_index),
                    "streaming_context_stage_order");
    require_context(!state_->executors[stage_index],
                    "streaming_context_duplicate_stage");
    require_context(adapter != nullptr, "streaming_context_missing_adapter");
    const auto &layout = state_->execution->model_snapshot->layout().stages[
        stage_index];
    auto executor = std::make_unique<StageExecutor>(
        stage_index, state_->request_generation, std::move(adapter));
    state_->executors[stage_index] = std::move(executor);
    try {
        state_->executors[stage_index]->begin(layout);
        state_->executors[stage_index]->enable_receipt({
            state_->execution->model_snapshot->layout().digest,
            std::move(implementation), source_generation()});
    } catch (...) {
        quarantine("streaming_context_stage_attach_failed");
        throw;
    }
    state_->phase = PublicRunPhase::running;
    return *state_->executors[stage_index];
}

void PublicStreamingRunContext::finish_stage(uint32_t stage_index) {
    require_context(state_ && state_->phase == PublicRunPhase::running &&
                        stage_index < state_->executors.size(),
                    "streaming_context_stage_out_of_range");
    auto &executor = state_->executors[stage_index];
    require_context(executor != nullptr, "streaming_context_stage_missing");
    require_context(!state_->stage_receipts[stage_index],
                    "streaming_context_stage_already_finished");
    try {
        executor->finish();
        state_->stage_receipts[stage_index] = executor->receipt();
    } catch (...) {
        quarantine("streaming_context_stage_finish_failed");
        throw;
    }
    state_->phase = PublicRunPhase::draining;
}

void PublicStreamingRunContext::record_boundary(ActualBoundaryReceipt boundary) {
    require_context(state_ && state_->phase == PublicRunPhase::draining,
                    "streaming_context_boundary_phase");
    const uint32_t expected = static_cast<uint32_t>(state_->boundaries.size());
    require_context(boundary.boundary_index == expected,
                    "streaming_context_boundary_order");
    require_context(boundary.from_stage_index == expected &&
                        boundary.to_stage_index == expected + 1,
                        "streaming_context_boundary_stage_order");
    require_context(boundary.from_stage_index < state_->stage_receipts.size() &&
                        state_->stage_receipts[boundary.from_stage_index] != nullptr &&
                        boundary.to_stage_index < state_->executors.size() &&
                        state_->executors[boundary.to_stage_index] == nullptr,
                    "streaming_context_boundary_lifecycle");
    require_context(boundary.source_generation == source_generation(),
                    "streaming_context_boundary_generation");
    require_context(!boundary.next_stage_started,
                    "streaming_context_boundary_started");
    verify_actual_boundary_receipt(
        boundary, expected, expected, expected + 1, source_generation());
    state_->boundaries.push_back(std::move(boundary));
    state_->phase = PublicRunPhase::running;
}

void PublicStreamingRunContext::mark_gpu_revalidated() {
    require_context(state_ && state_->phase == PublicRunPhase::created,
                    "streaming_context_invalid_gpu_phase");
    state_->phase = PublicRunPhase::gpu_revalidated;
}

void PublicStreamingRunContext::mark_running() {
    require_context(state_ &&
                        state_->phase == PublicRunPhase::gpu_revalidated,
                    "streaming_context_invalid_running_phase");
    state_->phase = PublicRunPhase::running;
}

void PublicStreamingRunContext::drain_all() {
    require_context(state_ &&
                        (state_->phase == PublicRunPhase::running ||
                         state_->phase == PublicRunPhase::draining),
                    "streaming_context_not_draining");
    state_->phase = PublicRunPhase::draining;
    for (size_t index = state_->executors.size(); index-- > 0;) {
        auto &executor = state_->executors[index];
        if (!executor) continue;
        if (state_->stage_receipts[index]) continue;
        if (!executor->retry_drain()) {
            quarantine("streaming_context_drain_unknown");
            throw std::runtime_error("streaming_context_drain_unknown");
        }
    }
}

std::shared_ptr<const ActualExecutionReceipt>
PublicStreamingRunContext::seal_receipt(
        std::string implementation, std::string component_policy_revision) {
    require_context(state_ && state_->phase == PublicRunPhase::draining,
                    "streaming_context_receipt_phase");
    for (const auto &receipt : state_->stage_receipts)
        require_context(receipt != nullptr, "streaming_context_stage_receipt_missing");
    require_context(state_->boundaries.size() + 1 == state_->stage_receipts.size(),
                    "streaming_context_boundary_count");
    std::vector<ActualStageReceipt> stages;
    stages.reserve(state_->stage_receipts.size());
    for (const auto &receipt : state_->stage_receipts) stages.push_back(*receipt);
    ActualExecutionReceipt value;
    if (stages.size() == 1 && state_->boundaries.empty()) {
        value = make_actual_execution_receipt(
            std::move(implementation),
            state_->execution->model_snapshot->layout().digest,
            std::move(component_policy_revision), std::move(stages));
    } else {
        value = make_actual_execution_receipt_v3(
            std::move(implementation),
            state_->execution->model_snapshot->layout().digest,
            std::move(component_policy_revision), std::move(stages),
            state_->boundaries);
    }
    verify_actual_execution_receipt(
        state_->execution->model_snapshot->layout(), value.implementation,
        value.component_policy_revision, source_generation(), value);
    state_->phase = PublicRunPhase::receipt_sealed;
    return std::make_shared<const ActualExecutionReceipt>(std::move(value));
}

void PublicStreamingRunContext::revalidate_source_after_drain() {
    require_context(state_ && state_->phase == PublicRunPhase::receipt_sealed,
                    "streaming_context_source_phase");
    lease().revalidate_after_drain();
    state_->phase = PublicRunPhase::source_revalidated;
}

void PublicStreamingRunContext::complete() {
    require_context(state_ && state_->phase == PublicRunPhase::source_revalidated,
                    "streaming_context_complete_phase");
    state_->phase = PublicRunPhase::completed;
    for (auto &executor : state_->executors) executor.reset();
}

void PublicStreamingRunContext::quarantine(std::string reason) noexcept {
    if (!state_ || state_->phase == PublicRunPhase::completed) return;
    state_->phase = PublicRunPhase::quarantined;
    state_->quarantine_reason = std::move(reason);
    auto &registry = quarantine_registry();
    std::lock_guard lock(registry.mutex);
    for (auto &executor : state_->executors) {
        if (!executor) continue;
        if (executor->retry_drain()) {
            executor.reset();
            continue;
        }
        registry.reasons.push_back(state_->quarantine_reason);
        registry.executors.push_back(std::move(executor));
    }
}

bool PublicStreamingRunContext::quarantined() const noexcept {
    return state_ && state_->phase == PublicRunPhase::quarantined;
}

PublicRunPhase PublicStreamingRunContext::phase() const noexcept {
    return state_ ? state_->phase : PublicRunPhase::quarantined;
}

const std::string &PublicStreamingRunContext::quarantine_reason() const noexcept {
    static const std::string empty;
    return state_ ? state_->quarantine_reason : empty;
}

} // namespace tc::streaming

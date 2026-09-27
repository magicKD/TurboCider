#include "streaming/actual_receipt.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace tc::streaming;

namespace {

template <class Function>
void rejects(Function &&function, const char *detail) {
    bool failed = false;
    try {
        function();
    } catch (const std::exception &error) {
        failed = std::string(error.what()).find(detail) != std::string::npos;
    }
    assert(failed);
}

StageLayout stage_layout(bool carry = false, bool multi_pool = false) {
    StageLayout stage;
    stage.id = "denoiser";
    stage.slot_count = 2;
    stage.distance = 1;
    stage.workers = 2;
    stage.pass_count = carry ? 3 : 2;
    stage.pass_transition = carry ? PassTransition::carry_first_group
                                  : PassTransition::reload;
    const uint32_t pool_count = multi_pool ? 2 : 1;
    for (uint32_t pool = 0; pool < pool_count; ++pool) {
        PoolLayout layout;
        layout.id = pool;
        layout.layout_class = pool ? "single" : "dual";
        layout.slots.resize(2);
        for (auto &slot : layout.slots) slot.capacity_bytes = 16;
        stage.pools.push_back(layout);
        const uint32_t groups = carry ? 5 : 2;
        for (uint32_t ordinal = 0; ordinal < groups; ++ordinal) {
            const uint32_t id = static_cast<uint32_t>(stage.groups.size());
            stage.groups.push_back(
                {id, pool, ordinal % 2, {id}, {8}, 8});
        }
    }
    return stage;
}

tc_stream_slot_ticket_v1 ticket(
        const StageLayout &layout, uint32_t stage, uint32_t pass,
        uint32_t step, uint32_t group, uint64_t request_generation,
        uint64_t content_generation) {
    uint32_t slot = layout.groups[group].slot;
    if (layout.pass_transition == PassTransition::carry_first_group) {
        const uint64_t offset =
            (static_cast<uint64_t>(pass) * layout.groups.size()) %
            layout.slot_count;
        slot = static_cast<uint32_t>(
            (group % layout.slot_count + offset) % layout.slot_count);
    }
    return {
        sizeof(tc_stream_slot_ticket_v1), TC_STREAM_SLOT_ABI_V1,
        layout.groups[group].pool, slot, request_generation,
        content_generation, {stage, pass, step, group}};
}

void complete_fill(
        ActualReceiptRecorder &recorder,
        const tc_stream_slot_ticket_v1 &value, uint64_t bytes = 8) {
    recorder.fill_submitted(value);
    tc_stream_completion_v1 completion{};
    completion.struct_size = sizeof(completion);
    completion.version = TC_STREAM_SLOT_ABI_V1;
    completion.kind = TC_STREAM_FILL_COMPLETE;
    completion.ticket = value;
    completion.bytes = bytes;
    recorder.fill_completed(completion);
}

void complete_group(
        ActualReceiptRecorder &recorder,
        const tc_stream_slot_ticket_v1 &value, uint64_t sequence) {
    const tc_stream_reader_fence_v1 fence{1, sequence};
    recorder.readers_issued(value, {&fence, 1});
    tc_stream_completion_v1 completion{};
    completion.struct_size = sizeof(completion);
    completion.version = TC_STREAM_SLOT_ABI_V1;
    completion.kind = TC_STREAM_READER_COMPLETE;
    completion.ticket = value;
    completion.fence = fence;
    recorder.reader_completed(completion);
}

std::shared_ptr<const ActualStageReceipt> make_receipt(
        const StageLayout &layout, uint64_t source_generation = 17,
        uint32_t stage_index = 0) {
    constexpr uint64_t request_generation = 9;
    ActualReceiptRecorder recorder(
        layout, stage_index, request_generation,
        {std::string(64, 'a'), "generic_stage_executor_v2",
         source_generation});
    uint64_t generation = 0;
    uint64_t sequence = 0;
    std::optional<tc_stream_slot_ticket_v1> incoming;
    for (uint32_t pass = 0; pass < layout.pass_count; ++pass) {
        uint32_t last_pool = UINT32_MAX;
        std::optional<tc_stream_slot_ticket_v1> outgoing;
        for (uint32_t group = 0; group < layout.groups.size(); ++group) {
            if (layout.groups[group].pool != last_pool) {
                last_pool = layout.groups[group].pool;
                recorder.pool_selected(pass, last_pool);
            }
            tc_stream_slot_ticket_v1 current{};
            if (group == 0 && incoming) {
                current = *incoming;
                incoming.reset();
            } else {
                current = ticket(
                    layout, stage_index, pass, pass + 10, group,
                    request_generation, ++generation);
                complete_fill(recorder, current);
            }
            complete_group(recorder, current, ++sequence);
            if (layout.pass_transition ==
                    PassTransition::carry_first_group &&
                pass + 1 < layout.pass_count &&
                group + 1 == layout.groups.size()) {
                outgoing = ticket(
                    layout, stage_index, pass + 1, pass + 11, 0,
                    request_generation, ++generation);
                complete_fill(recorder, *outgoing);
            }
        }
        recorder.pass_completed(pass, outgoing);
        incoming = outgoing;
    }
    recorder.drained();
    return recorder.finish();
}

} // namespace

int main() {
    const auto layout = stage_layout();
    const auto receipt = make_receipt(layout);
    ExecutionReceiptOptions options{
        std::string(64, 'a'), "generic_stage_executor_v2", 17};
    verify_actual_stage_receipt(layout, 0, 9, options, *receipt);
    assert(receipt->completed_passes == 2);
    assert(receipt->completed_groups == 4);
    assert(receipt->fills == 4 && receipt->groups_submitted == 4);
    assert(receipt->logical_read_bytes == 32);
    assert(receipt->reader_fences_issued == 4 &&
           receipt->reader_fences_completed == 4);
    assert(receipt->event_digest.size() == 64 &&
           receipt->canonical_digest.size() == 64);

    std::vector<tc_stream_group_receipt_v2> c_groups(
        receipt->groups.size());
    for (size_t index = 0; index < receipt->groups.size(); ++index) {
        const auto &source = receipt->groups[index];
        auto &target = c_groups[index];
        target.pass = source.pass;
        target.group = source.group;
        target.pool = source.pool;
        target.slot = source.slot;
        target.step = source.step;
        target.fill_count = source.fill_count;
        target.reader_count = source.reader_count;
        target.flags = (source.fill_completed ?
                            TC_STREAM_GROUP_FILL_COMPLETED_V2 : 0u) |
                       (source.group_submitted ?
                            TC_STREAM_GROUP_SUBMITTED_V2 : 0u);
        target.request_generation = source.request_generation;
        target.content_generation = source.content_generation;
        target.expected_bytes = source.expected_bytes;
        target.actual_bytes = source.actual_bytes;
        target.source_generation = source.source_generation;
        for (uint32_t reader = 0; reader < source.reader_count; ++reader) {
            target.readers[reader].fence = source.readers[reader].fence;
            target.readers[reader].completed =
                source.readers[reader].completed ? 1 : 0;
        }
    }
    std::vector<tc_stream_pool_selection_receipt_v2> c_pools;
    for (const auto &source : receipt->pool_selections)
        c_pools.push_back({source.pass, source.ordinal, source.pool});
    std::vector<tc_stream_carry_receipt_v2> c_carries;
    for (const auto &source : receipt->carries)
        c_carries.push_back({source.from_pass, source.to_pass, source.group,
                             source.pool, source.slot,
                             source.content_generation});
    tc_stream_receipt_v2 c_receipt{};
    c_receipt.struct_size = sizeof(c_receipt);
    c_receipt.version = TC_STREAM_RECEIPT_ABI_V2;
    c_receipt.stage_index = receipt->stage_index;
    c_receipt.completed_passes = receipt->completed_passes;
    c_receipt.completed_groups = receipt->completed_groups;
    c_receipt.fills = receipt->fills;
    c_receipt.groups_submitted = receipt->groups_submitted;
    c_receipt.logical_read_bytes = receipt->logical_read_bytes;
    c_receipt.reader_fences_issued = receipt->reader_fences_issued;
    c_receipt.reader_fences_completed = receipt->reader_fences_completed;
    c_receipt.source_generation = receipt->source_generation;
    c_receipt.drained = 1;
    c_receipt.verified = 1;
    std::snprintf(c_receipt.stage_id, sizeof(c_receipt.stage_id), "%s",
                  receipt->stage_id.c_str());
    std::snprintf(c_receipt.layout_digest,
                  sizeof(c_receipt.layout_digest), "%s",
                  receipt->layout_digest.c_str());
    std::snprintf(c_receipt.implementation,
                  sizeof(c_receipt.implementation), "%s",
                  receipt->implementation.c_str());
    std::snprintf(c_receipt.event_digest,
                  sizeof(c_receipt.event_digest), "%s",
                  receipt->event_digest.c_str());
    std::snprintf(c_receipt.canonical_digest,
                  sizeof(c_receipt.canonical_digest), "%s",
                  receipt->canonical_digest.c_str());
    c_receipt.group_capacity = c_receipt.group_count =
        static_cast<uint32_t>(c_groups.size());
    c_receipt.groups = c_groups.data();
    c_receipt.pool_selection_capacity = c_receipt.pool_selection_count =
        static_cast<uint32_t>(c_pools.size());
    c_receipt.pool_selections = c_pools.data();
    c_receipt.carry_capacity = c_receipt.carry_count =
        static_cast<uint32_t>(c_carries.size());
    c_receipt.carries = c_carries.data();
    const auto copied = actual_stage_receipt_from_c_v2(c_receipt);
    verify_actual_stage_receipt(layout, 0, 9, options, copied);
    assert(copied.canonical_digest == receipt->canonical_digest);

    Layout execution_layout;
    execution_layout.digest = std::string(64, 'a');
    execution_layout.stages.push_back(layout);
    auto execution = make_actual_execution_receipt(
        "generic_stage_executor_v2", execution_layout.digest,
        "components-v1", {*receipt});
    verify_actual_execution_receipt(
        execution_layout, "generic_stage_executor_v2", "components-v1",
        17, execution);

    auto wrong = *receipt;
    wrong.groups[0].fill_count = 0;
    rejects([&] {
        verify_actual_stage_receipt(layout, 0, 9, options, wrong);
    }, "group_incomplete");

    wrong = *receipt;
    wrong.groups[0].slot ^= 1;
    rejects([&] {
        verify_actual_stage_receipt(layout, 0, 9, options, wrong);
    }, "pool_slot_mismatch");

    wrong = *receipt;
    wrong.groups[0].actual_bytes = 4;
    rejects([&] {
        verify_actual_stage_receipt(layout, 0, 9, options, wrong);
    }, "logical_bytes_mismatch");

    wrong = *receipt;
    wrong.groups[0].readers[0].completed = false;
    rejects([&] {
        verify_actual_stage_receipt(layout, 0, 9, options, wrong);
    }, "fence_incomplete");

    wrong = *receipt;
    wrong.source_generation = 18;
    rejects([&] {
        verify_actual_stage_receipt(layout, 0, 9, options, wrong);
    }, "source_generation_mismatch");

    wrong = *receipt;
    wrong.drain_completed = false;
    rejects([&] {
        verify_actual_stage_receipt(layout, 0, 9, options, wrong);
    }, "drain_incomplete");

    wrong = *receipt;
    wrong.event_digest = std::string(64, 'f');
    rejects([&] {
        verify_actual_stage_receipt(layout, 0, 9, options, wrong);
    }, "event_digest_mismatch");

    const auto multi = stage_layout(false, true);
    const auto multi_receipt = make_receipt(multi);
    verify_actual_stage_receipt(multi, 0, 9, options, *multi_receipt);
    assert(multi_receipt->pool_selections.size() == 4);

    const auto carry = stage_layout(true, false);
    const auto carry_receipt = make_receipt(carry);
    verify_actual_stage_receipt(carry, 0, 9, options, *carry_receipt);
    assert(carry_receipt->carries.size() == 2);

    ActualReceiptRecorder duplicate(
        layout, 0, 9,
        {std::string(64, 'a'), "generic_stage_executor_v2", 17});
    duplicate.pool_selected(0, 0);
    const auto first = ticket(layout, 0, 0, 10, 0, 9, 1);
    duplicate.fill_submitted(first);
    rejects([&] { duplicate.fill_submitted(first); }, "group_duplicate");

    auto second_stage = stage_layout();
    second_stage.id = "upsampler";
    const auto first_stage_receipt = make_receipt(layout, 17, 0);
    const auto second_stage_receipt = make_receipt(second_stage, 17, 1);
    Layout multi_stage_layout;
    multi_stage_layout.digest = std::string(64, 'a');
    multi_stage_layout.stages.push_back(layout);
    multi_stage_layout.stages.push_back(second_stage);
    ActualBoundaryReceipt boundary;
    boundary.boundary_index = 0;
    boundary.id = "denoiser-to-upsampler";
    boundary.from_stage_index = 0;
    boundary.to_stage_index = 1;
    boundary.source_generation = 17;
    boundary.last_reader_sequence = 4;
    boundary.completed_reader_sequence = 4;
    boundary.live_slot_bytes_before = 32;
    boundary.live_slot_bytes_after = 0;
    boundary.released_slot_bytes = 32;
    boundary.pending_readers_before = 1;
    boundary.pending_readers_after = 0;
    boundary.source_stage_drained = true;
    boundary.source_stage_backing_released = true;
    boundary.next_stage_started = false;
    boundary.event_digest = actual_boundary_event_digest(boundary);
    boundary.canonical_digest = actual_boundary_canonical_digest(boundary);
    auto multi_stage = make_actual_execution_receipt_v3(
        "generic_stage_executor_v2", multi_stage_layout.digest,
        "components-v1", {*first_stage_receipt, *second_stage_receipt},
        {boundary});
    verify_actual_execution_receipt(
        multi_stage_layout, "generic_stage_executor_v2", "components-v1",
        17, multi_stage);
    assert(multi_stage.schema_version == actual_receipt_schema_v3);
    assert(multi_stage.boundaries.size() == 1);
    rejects([&] {
        (void)make_actual_execution_receipt_v3(
            "generic_stage_executor_v2", multi_stage_layout.digest,
            "components-v1",
            {*first_stage_receipt, *second_stage_receipt}, {});
    }, "boundary_count_mismatch");
    auto broken_boundary = boundary;
    broken_boundary.to_stage_index = 2;
    broken_boundary.event_digest =
        actual_boundary_event_digest(broken_boundary);
    broken_boundary.canonical_digest =
        actual_boundary_canonical_digest(broken_boundary);
    rejects([&] {
        (void)make_actual_execution_receipt_v3(
            "generic_stage_executor_v2", multi_stage_layout.digest,
            "components-v1",
            {*first_stage_receipt, *second_stage_receipt},
            {broken_boundary});
    }, "boundary_order_mismatch");
    broken_boundary = boundary;
    broken_boundary.pending_readers_after = 1;
    broken_boundary.event_digest =
        actual_boundary_event_digest(broken_boundary);
    broken_boundary.canonical_digest =
        actual_boundary_canonical_digest(broken_boundary);
    rejects([&] {
        verify_actual_boundary_receipt(broken_boundary, 0, 0, 1, 17);
    }, "boundary_pending_readers");
    broken_boundary = boundary;
    broken_boundary.source_stage_backing_released = false;
    broken_boundary.event_digest =
        actual_boundary_event_digest(broken_boundary);
    broken_boundary.canonical_digest =
        actual_boundary_canonical_digest(broken_boundary);
    rejects([&] {
        verify_actual_boundary_receipt(broken_boundary, 0, 0, 1, 17);
    }, "boundary_release_incomplete");
    broken_boundary = boundary;
    broken_boundary.next_stage_started = true;
    broken_boundary.event_digest =
        actual_boundary_event_digest(broken_boundary);
    broken_boundary.canonical_digest =
        actual_boundary_canonical_digest(broken_boundary);
    rejects([&] {
        verify_actual_boundary_receipt(broken_boundary, 0, 0, 1, 17);
    }, "boundary_next_stage_already_started");
    broken_boundary = boundary;
    broken_boundary.live_slot_bytes_after = 1;
    broken_boundary.released_slot_bytes = 31;
    broken_boundary.event_digest =
        actual_boundary_event_digest(broken_boundary);
    broken_boundary.canonical_digest =
        actual_boundary_canonical_digest(broken_boundary);
    rejects([&] {
        verify_actual_boundary_receipt(broken_boundary, 0, 0, 1, 17);
    }, "boundary_live_bytes_remaining");
    broken_boundary = boundary;
    broken_boundary.event_digest = std::string(64, 'f');
    rejects([&] {
        verify_actual_boundary_receipt(broken_boundary, 0, 0, 1, 17);
    }, "boundary_event_digest_mismatch");
    auto legacy_multi = make_actual_execution_receipt(
        "generic_stage_executor_v2", multi_stage_layout.digest,
        "components-v1", {*first_stage_receipt, *second_stage_receipt});
    rejects([&] {
        verify_actual_execution_receipt(
            multi_stage_layout, "generic_stage_executor_v2",
            "components-v1", 17, legacy_multi);
    }, "legacy_receipt_requires_single_stage");

    std::cout << "PASS actual receipt v2: single/multi/carry, canonical "
                 "digests and mismatch fail-closed; v3 multi-stage boundary\n";
}

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
        const StageLayout &layout, uint64_t source_generation = 17) {
    constexpr uint32_t stage_index = 0;
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

    std::cout << "PASS actual receipt v2: single/multi/carry, canonical "
                 "digests and mismatch fail-closed\n";
}

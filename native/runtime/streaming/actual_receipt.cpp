#include "actual_receipt.hpp"

#include "canonical_encoding.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tc::streaming {
namespace {

[[noreturn]] void receipt_fail(std::string_view detail) {
    throw std::runtime_error(
        "streaming_actual_receipt_mismatch: " + std::string(detail));
}

void receipt_check(bool value, std::string_view detail) {
    if (!value) receipt_fail(detail);
}

void checked_add(uint64_t &target, uint64_t value) {
    receipt_check(value <= std::numeric_limits<uint64_t>::max() - target,
                  "counter_overflow");
    target += value;
}

std::string stage_identity(const StageLayout &layout, uint32_t stage) {
    return layout.id.empty() ? "stage-" + std::to_string(stage) : layout.id;
}

template <size_t N>
std::string receipt_c_string(const char (&value)[N],
                             std::string_view detail) {
    const size_t length = ::strnlen(value, N);
    receipt_check(length != 0 && length < N, detail);
    return std::string(value, length);
}

std::vector<uint32_t> pool_sequence(const StageLayout &layout) {
    std::vector<uint32_t> result;
    result.reserve(layout.pools.size());
    for (const auto &group : layout.groups) {
        if (result.empty() || result.back() != group.pool)
            result.push_back(group.pool);
    }
    return result;
}

std::string execution_canonical_digest(
        const ActualExecutionReceipt &receipt) {
    CanonicalEncoder out(receipt.schema_version >= actual_receipt_schema_v3
                             ? "tc-streaming-actual-execution-receipt-v3"
                             : "tc-streaming-actual-execution-receipt-v2");
    out.unsigned_field("schema_version", receipt.schema_version);
    out.string_field("implementation", receipt.implementation);
    out.string_field("layout_digest", receipt.layout_digest);
    out.string_field("component_policy_revision",
                     receipt.component_policy_revision);
    out.begin_list("stages", receipt.stages.size());
    for (const auto &stage : receipt.stages) {
        out.unsigned_field("stage.index", stage.stage_index);
        out.string_field("stage.id", stage.stage_id);
        out.string_field("stage.digest", stage.canonical_digest);
    }
    if (receipt.schema_version >= actual_receipt_schema_v3) {
        out.begin_list("boundaries", receipt.boundaries.size());
        for (const auto &boundary : receipt.boundaries) {
            out.unsigned_field("boundary.index", boundary.boundary_index);
            out.string_field("boundary.id", boundary.id);
            out.unsigned_field("boundary.from", boundary.from_stage_index);
            out.unsigned_field("boundary.to", boundary.to_stage_index);
            out.unsigned_field("boundary.source_generation",
                               boundary.source_generation);
            out.unsigned_field("boundary.last_reader_sequence",
                               boundary.last_reader_sequence);
            out.unsigned_field("boundary.completed_reader_sequence",
                               boundary.completed_reader_sequence);
            out.unsigned_field("boundary.live_before",
                               boundary.live_slot_bytes_before);
            out.unsigned_field("boundary.live_after",
                               boundary.live_slot_bytes_after);
            out.unsigned_field("boundary.released",
                               boundary.released_slot_bytes);
            out.unsigned_field("boundary.pending_before",
                               boundary.pending_readers_before);
            out.unsigned_field("boundary.pending_after",
                               boundary.pending_readers_after);
            out.boolean_field("boundary.drained",
                              boundary.source_stage_drained);
            out.boolean_field("boundary.released_ok",
                              boundary.source_stage_backing_released);
            out.boolean_field("boundary.next_started",
                              boundary.next_stage_started);
            out.string_field("boundary.event_digest",
                             boundary.event_digest);
        }
    }
    return out.sha256();
}

} // namespace

ActualReceiptRecorder::ActualReceiptRecorder(
        const StageLayout &layout, uint32_t stage_index,
        uint64_t request_generation, ExecutionReceiptOptions options)
    : layout_(layout), options_(std::move(options)),
      request_generation_(request_generation),
      pool_sequence_(pool_sequence(layout)),
      pass_completed_(layout.pass_count, false),
      owner_(std::this_thread::get_id()) {
    receipt_check(request_generation_ != 0, "request_generation_missing");
    receipt_check(options_.source_generation != 0,
                  "source_generation_missing");
    receipt_check(!options_.layout_digest.empty(), "layout_digest_missing");
    receipt_check(!options_.implementation.empty(), "implementation_missing");
    receipt_check(!layout_.resident && !layout_.groups.empty() &&
                      !layout_.pools.empty() && layout_.pass_count != 0,
                  "invalid_stage_layout");
    receipt_check(!pool_sequence_.empty(), "pool_sequence_missing");

    receipt_.stage_index = stage_index;
    receipt_.stage_id = stage_identity(layout_, stage_index);
    receipt_.layout_digest = options_.layout_digest;
    receipt_.implementation = options_.implementation;
    receipt_.source_generation = options_.source_generation;
    const uint64_t entry_count =
        static_cast<uint64_t>(layout_.pass_count) * layout_.groups.size();
    receipt_check(entry_count <= std::numeric_limits<size_t>::max(),
                  "group_matrix_too_large");
    receipt_.groups.resize(static_cast<size_t>(entry_count));
    receipt_.pool_selections.reserve(
        static_cast<size_t>(layout_.pass_count) * pool_sequence_.size());
    if (layout_.pass_transition == PassTransition::carry_first_group)
        receipt_.carries.reserve(layout_.pass_count - 1);

    for (uint32_t pass = 0; pass < layout_.pass_count; ++pass) {
        for (uint32_t group = 0; group < layout_.groups.size(); ++group) {
            const auto &expected = layout_.groups[group];
            auto &actual = receipt_.groups[
                static_cast<size_t>(pass) * layout_.groups.size() + group];
            actual.pass = pass;
            actual.group = group;
            actual.pool = expected.pool;
            actual.slot = expected_slot(pass, group);
            actual.expected_bytes = expected.bytes;
            actual.source_generation = options_.source_generation;
        }
    }
}

void ActualReceiptRecorder::owner() const {
    if (owner_ != std::this_thread::get_id())
        throw std::logic_error("streaming_owner_violation");
}

uint32_t ActualReceiptRecorder::expected_slot(
        uint32_t pass, uint32_t group) const {
    receipt_check(group < layout_.groups.size(), "group_out_of_range");
    if (layout_.pass_transition == PassTransition::carry_first_group) {
        receipt_check(layout_.slot_count != 0, "slot_count_missing");
        const uint64_t offset =
            (static_cast<uint64_t>(pass) * layout_.groups.size()) %
            layout_.slot_count;
        return static_cast<uint32_t>(
            (group % layout_.slot_count + offset) % layout_.slot_count);
    }
    return layout_.groups[group].slot;
}

ActualGroupReceipt &ActualReceiptRecorder::group_for(
        const tc_stream_slot_ticket_v1 &ticket) {
    return const_cast<ActualGroupReceipt &>(
        std::as_const(*this).group_for(ticket));
}

const ActualGroupReceipt &ActualReceiptRecorder::group_for(
        const tc_stream_slot_ticket_v1 &ticket) const {
    owner();
    receipt_check(ticket.struct_size == sizeof(ticket) &&
                      ticket.version == TC_STREAM_SLOT_ABI_V1,
                  "ticket_abi_mismatch");
    receipt_check(ticket.request_generation == request_generation_,
                  "request_generation_mismatch");
    receipt_check(ticket.item.stage == receipt_.stage_index,
                  "stage_mismatch");
    receipt_check(ticket.item.pass < layout_.pass_count &&
                      ticket.item.group < layout_.groups.size(),
                  "group_out_of_range");
    const auto &result = receipt_.groups[
        static_cast<size_t>(ticket.item.pass) * layout_.groups.size() +
        ticket.item.group];
    receipt_check(ticket.pool == result.pool && ticket.slot == result.slot,
                  "pool_slot_mismatch");
    return result;
}

void ActualReceiptRecorder::pool_selected(uint32_t pass, uint32_t pool) {
    owner();
    receipt_check(!finished_ && pass < layout_.pass_count,
                  "pool_selection_after_finish");
    const uint64_t expected_index =
        static_cast<uint64_t>(pass) * pool_sequence_.size() +
        (pool_selection_cursor_ % pool_sequence_.size());
    receipt_check(expected_index == pool_selection_cursor_,
                  "pool_selection_order_mismatch");
    const uint32_t ordinal =
        pool_selection_cursor_ % static_cast<uint32_t>(pool_sequence_.size());
    receipt_check(pool_sequence_[ordinal] == pool,
                  "pool_selection_mismatch");
    receipt_.pool_selections.push_back({pass, ordinal, pool});
    ++pool_selection_cursor_;
}

void ActualReceiptRecorder::fill_submitted(
        const tc_stream_slot_ticket_v1 &ticket) {
    auto &group = group_for(ticket);
    receipt_check(!finished_ && group.fill_count == 0,
                  "group_duplicate");
    receipt_check(ticket.content_generation != 0,
                  "content_generation_missing");
    group.fill_count = 1;
    group.step = ticket.item.step;
    group.request_generation = ticket.request_generation;
    group.content_generation = ticket.content_generation;
}

void ActualReceiptRecorder::fill_completed(
        const tc_stream_completion_v1 &completion) {
    owner();
    receipt_check(completion.struct_size == sizeof(completion) &&
                      completion.version == TC_STREAM_SLOT_ABI_V1 &&
                      completion.kind == TC_STREAM_FILL_COMPLETE &&
                      completion.status == 0,
                  "fill_completion_invalid");
    auto &group = group_for(completion.ticket);
    receipt_check(group.fill_count == 1 && !group.fill_completed,
                  "group_duplicate");
    receipt_check(group.step == completion.ticket.item.step &&
                      group.content_generation ==
                          completion.ticket.content_generation,
                  "ticket_generation_mismatch");
    receipt_check(completion.bytes == group.expected_bytes,
                  "logical_bytes_mismatch");
    group.actual_bytes = completion.bytes;
    group.fill_completed = true;
    ++receipt_.fills;
    checked_add(receipt_.logical_read_bytes, completion.bytes);
}

void ActualReceiptRecorder::readers_issued(
        const tc_stream_slot_ticket_v1 &ticket,
        std::span<const tc_stream_reader_fence_v1> readers) {
    auto &group = group_for(ticket);
    receipt_check(!finished_ && group.fill_completed &&
                      !group.group_submitted && !readers.empty() &&
                      readers.size() <= TC_STREAM_MAX_READER_QUEUES,
                  "reader_set_invalid");
    receipt_check(group.step == ticket.item.step &&
                      group.content_generation == ticket.content_generation,
                  "ticket_generation_mismatch");
    for (size_t index = 0; index < readers.size(); ++index) {
        receipt_check(readers[index].queue != 0 &&
                          readers[index].sequence != 0,
                      "reader_identity_invalid");
        for (size_t previous = 0; previous < index; ++previous)
            receipt_check(readers[index].queue != readers[previous].queue,
                          "reader_queue_duplicate");
        group.readers[index].fence = readers[index];
    }
    group.reader_count = static_cast<uint32_t>(readers.size());
    group.group_submitted = true;
    ++receipt_.groups_submitted;
    checked_add(receipt_.reader_fences_issued, readers.size());
}

void ActualReceiptRecorder::reader_completed(
        const tc_stream_completion_v1 &completion) {
    owner();
    receipt_check(completion.struct_size == sizeof(completion) &&
                      completion.version == TC_STREAM_SLOT_ABI_V1 &&
                      completion.kind == TC_STREAM_READER_COMPLETE &&
                      completion.status == 0,
                  "reader_completion_invalid");
    auto &group = group_for(completion.ticket);
    receipt_check(group.group_submitted, "reader_before_submit");
    uint32_t found = group.reader_count;
    for (uint32_t index = 0; index < group.reader_count; ++index) {
        const auto &fence = group.readers[index].fence;
        if (fence.queue == completion.fence.queue &&
            fence.sequence == completion.fence.sequence) {
            found = index;
            break;
        }
    }
    receipt_check(found < group.reader_count, "reader_unknown");
    receipt_check(!group.readers[found].completed,
                  "reader_duplicate");
    group.readers[found].completed = true;
    ++receipt_.reader_fences_completed;
}

void ActualReceiptRecorder::pass_completed(
        uint32_t pass,
        const std::optional<tc_stream_slot_ticket_v1> &carry) {
    owner();
    receipt_check(!finished_ && pass < layout_.pass_count &&
                      !pass_completed_[pass] &&
                      receipt_.completed_passes == pass,
                  "pass_order_mismatch");
    const uint64_t expected_pool_selections =
        static_cast<uint64_t>(pass + 1) * pool_sequence_.size();
    receipt_check(pool_selection_cursor_ == expected_pool_selections,
                  "pool_selection_incomplete");
    for (uint32_t group = 0; group < layout_.groups.size(); ++group) {
        const auto &actual = receipt_.groups[
            static_cast<size_t>(pass) * layout_.groups.size() + group];
        receipt_check(actual.fill_count == 1 && actual.fill_completed &&
                          actual.group_submitted,
                      "group_incomplete");
        for (uint32_t reader = 0; reader < actual.reader_count; ++reader)
            receipt_check(actual.readers[reader].completed,
                          "fence_incomplete");
    }

    const bool expects_carry =
        layout_.pass_transition == PassTransition::carry_first_group &&
        pass + 1 < layout_.pass_count;
    receipt_check(carry.has_value() == expects_carry,
                  "carry_mismatch");
    if (carry) {
        const auto &next = group_for(*carry);
        receipt_check(carry->item.pass == pass + 1 &&
                          carry->item.group == 0 &&
                          next.fill_count == 1 && next.fill_completed &&
                          !next.group_submitted,
                      "carry_mismatch");
        receipt_.carries.push_back({
            pass, pass + 1, 0, carry->pool, carry->slot,
            carry->content_generation});
    }
    pass_completed_[pass] = true;
    ++receipt_.completed_passes;
    receipt_.completed_groups +=
        static_cast<uint32_t>(layout_.groups.size());
}

void ActualReceiptRecorder::drained() {
    owner();
    receipt_check(!finished_ && !drained_ &&
                      receipt_.completed_passes == layout_.pass_count,
                  "drain_order_mismatch");
    drained_ = true;
    receipt_.drain_completed = true;
}

std::shared_ptr<const ActualStageReceipt>
ActualReceiptRecorder::finish() {
    owner();
    receipt_check(!finished_ && drained_, "drain_incomplete");
    receipt_.event_digest = actual_stage_event_digest(receipt_);
    receipt_.canonical_digest = actual_stage_canonical_digest(receipt_);
    verify_actual_stage_receipt(
        layout_, receipt_.stage_index, request_generation_, options_,
        receipt_);
    finished_ = true;
    return std::make_shared<const ActualStageReceipt>(receipt_);
}

std::string actual_stage_event_digest(const ActualStageReceipt &receipt) {
    CanonicalEncoder out("tc-streaming-actual-stage-events-v2");
    out.unsigned_field("schema_version", receipt.schema_version);
    out.unsigned_field("stage_index", receipt.stage_index);
    out.string_field("stage_id", receipt.stage_id);
    out.string_field("layout_digest", receipt.layout_digest);
    out.string_field("implementation", receipt.implementation);
    out.unsigned_field("source_generation", receipt.source_generation);
    out.begin_list("groups", receipt.groups.size());
    for (const auto &group : receipt.groups) {
        out.unsigned_field("group.pass", group.pass);
        out.unsigned_field("group.id", group.group);
        out.unsigned_field("group.pool", group.pool);
        out.unsigned_field("group.slot", group.slot);
        out.unsigned_field("group.step", group.step);
        out.unsigned_field("group.fill_count", group.fill_count);
        out.unsigned_field("group.request_generation",
                           group.request_generation);
        out.unsigned_field("group.content_generation",
                           group.content_generation);
        out.unsigned_field("group.expected_bytes", group.expected_bytes);
        out.unsigned_field("group.actual_bytes", group.actual_bytes);
        out.unsigned_field("group.source_generation",
                           group.source_generation);
        out.boolean_field("group.fill_completed", group.fill_completed);
        out.boolean_field("group.submitted", group.group_submitted);
        out.begin_list("group.readers", group.reader_count);
        for (uint32_t index = 0; index < group.reader_count; ++index) {
            out.unsigned_field("reader.queue",
                               group.readers[index].fence.queue);
            out.unsigned_field("reader.sequence",
                               group.readers[index].fence.sequence);
            out.boolean_field("reader.completed",
                              group.readers[index].completed);
        }
    }
    out.begin_list("pool_selections", receipt.pool_selections.size());
    for (const auto &selection : receipt.pool_selections) {
        out.unsigned_field("pool.pass", selection.pass);
        out.unsigned_field("pool.ordinal", selection.ordinal);
        out.unsigned_field("pool.id", selection.pool);
    }
    out.begin_list("carries", receipt.carries.size());
    for (const auto &carry : receipt.carries) {
        out.unsigned_field("carry.from_pass", carry.from_pass);
        out.unsigned_field("carry.to_pass", carry.to_pass);
        out.unsigned_field("carry.group", carry.group);
        out.unsigned_field("carry.pool", carry.pool);
        out.unsigned_field("carry.slot", carry.slot);
        out.unsigned_field("carry.content_generation",
                           carry.content_generation);
    }
    return out.sha256();
}

std::string actual_stage_canonical_digest(
        const ActualStageReceipt &receipt) {
    CanonicalEncoder out("tc-streaming-actual-stage-receipt-v2");
    out.unsigned_field("schema_version", receipt.schema_version);
    out.unsigned_field("stage_index", receipt.stage_index);
    out.string_field("stage_id", receipt.stage_id);
    out.string_field("layout_digest", receipt.layout_digest);
    out.string_field("implementation", receipt.implementation);
    out.unsigned_field("completed_passes", receipt.completed_passes);
    out.unsigned_field("completed_groups", receipt.completed_groups);
    out.unsigned_field("fills", receipt.fills);
    out.unsigned_field("groups_submitted", receipt.groups_submitted);
    out.unsigned_field("logical_read_bytes", receipt.logical_read_bytes);
    out.unsigned_field("reader_fences_issued",
                       receipt.reader_fences_issued);
    out.unsigned_field("reader_fences_completed",
                       receipt.reader_fences_completed);
    out.unsigned_field("source_generation", receipt.source_generation);
    out.boolean_field("drain_completed", receipt.drain_completed);
    out.string_field("event_digest", receipt.event_digest);
    return out.sha256();
}

void verify_actual_stage_receipt(
        const StageLayout &layout, uint32_t stage_index,
        uint64_t request_generation, const ExecutionReceiptOptions &options,
        const ActualStageReceipt &receipt) {
    receipt_check(receipt.schema_version == actual_receipt_schema_v2,
                  "schema_mismatch");
    receipt_check(receipt.stage_index == stage_index &&
                      receipt.stage_id == stage_identity(layout, stage_index),
                  "stage_mismatch");
    receipt_check(receipt.layout_digest == options.layout_digest,
                  "layout_mismatch");
    receipt_check(receipt.implementation == options.implementation,
                  "implementation_mismatch");
    receipt_check(receipt.source_generation == options.source_generation &&
                      options.source_generation != 0,
                  "source_generation_mismatch");
    const uint64_t expected_group_count =
        static_cast<uint64_t>(layout.pass_count) * layout.groups.size();
    receipt_check(receipt.groups.size() == expected_group_count,
                  "group_matrix_mismatch");
    receipt_check(receipt.completed_passes == layout.pass_count &&
                      receipt.completed_groups == expected_group_count &&
                      receipt.fills == expected_group_count &&
                      receipt.groups_submitted == expected_group_count,
                  "group_count_mismatch");

    uint64_t logical_bytes = 0;
    uint64_t issued = 0;
    uint64_t completed = 0;
    for (uint32_t pass = 0; pass < layout.pass_count; ++pass) {
        std::optional<uint32_t> pass_step;
        for (uint32_t group = 0; group < layout.groups.size(); ++group) {
            const auto &expected = layout.groups[group];
            const auto &actual = receipt.groups[
                static_cast<size_t>(pass) * layout.groups.size() + group];
            const uint32_t slot =
                layout.pass_transition == PassTransition::carry_first_group
                    ? static_cast<uint32_t>(
                          (group % layout.slot_count +
                           (static_cast<uint64_t>(pass) *
                                layout.groups.size()) % layout.slot_count) %
                          layout.slot_count)
                    : expected.slot;
            receipt_check(actual.pass == pass && actual.group == group &&
                              actual.pool == expected.pool &&
                              actual.slot == slot,
                          "pool_slot_mismatch");
            receipt_check(actual.request_generation == request_generation &&
                              actual.content_generation != 0,
                          "ticket_generation_mismatch");
            if (!pass_step)
                pass_step = actual.step;
            else
                receipt_check(*pass_step == actual.step,
                              "step_mismatch");
            receipt_check(actual.fill_count == 1 &&
                              actual.fill_completed &&
                              actual.group_submitted,
                          "group_incomplete");
            receipt_check(actual.expected_bytes == expected.bytes &&
                              actual.actual_bytes == expected.bytes,
                          "logical_bytes_mismatch");
            receipt_check(actual.source_generation ==
                              options.source_generation,
                          "source_generation_mismatch");
            receipt_check(actual.reader_count != 0 &&
                              actual.reader_count <=
                                  TC_STREAM_MAX_READER_QUEUES,
                          "reader_set_invalid");
            for (uint32_t reader = 0; reader < actual.reader_count;
                 ++reader) {
                const auto &entry = actual.readers[reader];
                receipt_check(entry.fence.queue != 0 &&
                                  entry.fence.sequence != 0 &&
                                  entry.completed,
                              "fence_incomplete");
                for (uint32_t previous = 0; previous < reader; ++previous)
                    receipt_check(
                        actual.readers[previous].fence.queue !=
                            entry.fence.queue,
                        "reader_queue_duplicate");
            }
            checked_add(logical_bytes, actual.actual_bytes);
            checked_add(issued, actual.reader_count);
            checked_add(completed, actual.reader_count);
        }
    }
    receipt_check(receipt.logical_read_bytes == logical_bytes,
                  "logical_bytes_mismatch");
    receipt_check(receipt.reader_fences_issued == issued &&
                      receipt.reader_fences_completed == completed,
                  "fence_incomplete");

    const auto pools = pool_sequence(layout);
    receipt_check(receipt.pool_selections.size() ==
                      static_cast<uint64_t>(layout.pass_count) * pools.size(),
                  "pool_selection_incomplete");
    for (uint32_t pass = 0; pass < layout.pass_count; ++pass) {
        for (uint32_t ordinal = 0; ordinal < pools.size(); ++ordinal) {
            const auto &actual = receipt.pool_selections[
                static_cast<size_t>(pass) * pools.size() + ordinal];
            receipt_check(actual.pass == pass &&
                              actual.ordinal == ordinal &&
                              actual.pool == pools[ordinal],
                          "pool_selection_mismatch");
        }
    }

    const size_t expected_carries =
        layout.pass_transition == PassTransition::carry_first_group
            ? layout.pass_count - 1
            : 0;
    receipt_check(receipt.carries.size() == expected_carries,
                  "carry_mismatch");
    for (uint32_t pass = 0; pass < receipt.carries.size(); ++pass) {
        const auto &carry = receipt.carries[pass];
        const auto &next = receipt.groups[
            static_cast<size_t>(pass + 1) * layout.groups.size()];
        receipt_check(carry.from_pass == pass &&
                          carry.to_pass == pass + 1 && carry.group == 0 &&
                          carry.pool == next.pool && carry.slot == next.slot &&
                          carry.content_generation == next.content_generation,
                      "carry_mismatch");
    }
    receipt_check(receipt.drain_completed, "drain_incomplete");
    receipt_check(receipt.event_digest ==
                      actual_stage_event_digest(receipt),
                  "event_digest_mismatch");
    receipt_check(receipt.canonical_digest ==
                      actual_stage_canonical_digest(receipt),
                  "canonical_digest_mismatch");
}

ActualStageReceipt actual_stage_receipt_from_c_v2(
        const tc_stream_receipt_v2 &source) {
    receipt_check(source.struct_size == sizeof(source) &&
                      source.version == TC_STREAM_RECEIPT_ABI_V2,
                  "c_receipt_abi_mismatch");
    receipt_check(source.verified && source.drained,
                  "c_receipt_not_sealed");
    receipt_check(source.group_count != 0 && source.groups != nullptr,
                  "c_receipt_group_matrix_missing");
    receipt_check(source.group_count <= source.group_capacity,
                  "c_receipt_group_capacity_mismatch");
    receipt_check(source.pool_selection_count != 0 &&
                      source.pool_selections != nullptr &&
                      source.pool_selection_count <=
                          source.pool_selection_capacity,
                  "c_receipt_pool_matrix_missing");
    receipt_check(source.carry_count <= source.carry_capacity &&
                      (!source.carry_count || source.carries != nullptr),
                  "c_receipt_carry_matrix_missing");

    ActualStageReceipt result;
    result.stage_index = source.stage_index;
    result.stage_id = receipt_c_string(
        source.stage_id, "c_receipt_stage_id_invalid");
    result.layout_digest = receipt_c_string(
        source.layout_digest, "c_receipt_layout_digest_invalid");
    result.implementation = receipt_c_string(
        source.implementation, "c_receipt_implementation_invalid");
    result.completed_passes = source.completed_passes;
    result.completed_groups = source.completed_groups;
    result.fills = source.fills;
    result.groups_submitted = source.groups_submitted;
    result.logical_read_bytes = source.logical_read_bytes;
    result.reader_fences_issued = source.reader_fences_issued;
    result.reader_fences_completed = source.reader_fences_completed;
    result.source_generation = source.source_generation;
    result.drain_completed = source.drained != 0;
    result.groups.reserve(source.group_count);
    for (uint32_t index = 0; index < source.group_count; ++index) {
        const auto &input = source.groups[index];
        receipt_check(input.reader_count <= TC_STREAM_MAX_READER_QUEUES,
                      "c_receipt_reader_count_invalid");
        receipt_check((input.flags &
                      ~(TC_STREAM_GROUP_FILL_COMPLETED_V2 |
                        TC_STREAM_GROUP_SUBMITTED_V2)) == 0,
                      "c_receipt_group_flags_invalid");
        ActualGroupReceipt group;
        group.pass = input.pass;
        group.group = input.group;
        group.pool = input.pool;
        group.slot = input.slot;
        group.step = input.step;
        group.fill_count = input.fill_count;
        group.request_generation = input.request_generation;
        group.content_generation = input.content_generation;
        group.expected_bytes = input.expected_bytes;
        group.actual_bytes = input.actual_bytes;
        group.source_generation = input.source_generation;
        group.fill_completed =
            (input.flags & TC_STREAM_GROUP_FILL_COMPLETED_V2) != 0;
        group.group_submitted =
            (input.flags & TC_STREAM_GROUP_SUBMITTED_V2) != 0;
        group.reader_count = input.reader_count;
        for (uint32_t reader = 0; reader < input.reader_count; ++reader) {
            group.readers[reader].fence = input.readers[reader].fence;
            group.readers[reader].completed =
                input.readers[reader].completed != 0;
        }
        result.groups.push_back(std::move(group));
    }
    result.pool_selections.reserve(source.pool_selection_count);
    for (uint32_t index = 0; index < source.pool_selection_count; ++index) {
        const auto &input = source.pool_selections[index];
        result.pool_selections.push_back(
            {input.pass, input.ordinal, input.pool});
    }
    result.carries.reserve(source.carry_count);
    for (uint32_t index = 0; index < source.carry_count; ++index) {
        const auto &input = source.carries[index];
        result.carries.push_back({
            input.from_pass, input.to_pass, input.group, input.pool,
            input.slot, input.content_generation});
    }
    result.event_digest = receipt_c_string(
        source.event_digest, "c_receipt_event_digest_invalid");
    result.canonical_digest = receipt_c_string(
        source.canonical_digest, "c_receipt_canonical_digest_invalid");
    receipt_check(result.event_digest == actual_stage_event_digest(result),
                  "c_receipt_event_digest_mismatch");
    receipt_check(result.canonical_digest ==
                      actual_stage_canonical_digest(result),
                  "c_receipt_canonical_digest_mismatch");
    return result;
}

ActualExecutionReceipt make_actual_execution_receipt(
        std::string implementation, std::string layout_digest,
        std::string component_policy_revision,
        std::vector<ActualStageReceipt> stages) {
    receipt_check(!implementation.empty(), "implementation_missing");
    receipt_check(!layout_digest.empty(), "layout_digest_missing");
    receipt_check(!component_policy_revision.empty(),
                  "component_policy_missing");
    receipt_check(!stages.empty(), "stage_missing");
    ActualExecutionReceipt result;
    result.implementation = std::move(implementation);
    result.layout_digest = std::move(layout_digest);
    result.component_policy_revision =
        std::move(component_policy_revision);
    result.stages = std::move(stages);
    for (const auto &stage : result.stages)
        receipt_check(stage.implementation == result.implementation &&
                          stage.layout_digest == result.layout_digest,
                      "stage_identity_mismatch");
    result.canonical_digest = execution_canonical_digest(result);
    return result;
}

ActualExecutionReceipt make_actual_execution_receipt_v3(
        std::string implementation, std::string layout_digest,
        std::string component_policy_revision,
        std::vector<ActualStageReceipt> stages,
        std::vector<ActualBoundaryReceipt> boundaries) {
    receipt_check(!implementation.empty(), "implementation_missing");
    receipt_check(!layout_digest.empty(), "layout_digest_missing");
    receipt_check(!component_policy_revision.empty(),
                  "component_policy_missing");
    receipt_check(!stages.empty(), "stage_missing");
    receipt_check(boundaries.size() + 1 == stages.size(),
                  "boundary_count_mismatch");
    ActualExecutionReceipt result;
    result.schema_version = actual_receipt_schema_v3;
    result.implementation = std::move(implementation);
    result.layout_digest = std::move(layout_digest);
    result.component_policy_revision =
        std::move(component_policy_revision);
    result.stages = std::move(stages);
    result.boundaries = std::move(boundaries);
    for (const auto &stage : result.stages)
        receipt_check(stage.implementation == result.implementation &&
                          stage.layout_digest == result.layout_digest,
                      "stage_identity_mismatch");
    for (uint32_t index = 0; index < result.boundaries.size(); ++index) {
        const auto &boundary = result.boundaries[index];
        receipt_check(boundary.boundary_index == index &&
                          boundary.from_stage_index == index &&
                          boundary.to_stage_index == index + 1,
                      "boundary_order_mismatch");
        receipt_check(!boundary.id.empty(), "boundary_id_missing");
        receipt_check(boundary.event_digest ==
                          actual_boundary_event_digest(boundary),
                      "boundary_event_digest_mismatch");
        receipt_check(boundary.canonical_digest ==
                          actual_boundary_canonical_digest(boundary),
                      "boundary_canonical_digest_mismatch");
    }
    result.canonical_digest = execution_canonical_digest(result);
    return result;
}

std::string actual_boundary_event_digest(
        const ActualBoundaryReceipt &boundary) {
    CanonicalEncoder out("tc-streaming-actual-boundary-events-v1");
    out.unsigned_field("index", boundary.boundary_index);
    out.string_field("id", boundary.id);
    out.unsigned_field("from", boundary.from_stage_index);
    out.unsigned_field("to", boundary.to_stage_index);
    out.unsigned_field("source_generation", boundary.source_generation);
    out.unsigned_field("last_reader_sequence",
                       boundary.last_reader_sequence);
    out.unsigned_field("completed_reader_sequence",
                       boundary.completed_reader_sequence);
    out.unsigned_field("live_before", boundary.live_slot_bytes_before);
    out.unsigned_field("live_after", boundary.live_slot_bytes_after);
    out.unsigned_field("released", boundary.released_slot_bytes);
    out.unsigned_field("pending_before", boundary.pending_readers_before);
    out.unsigned_field("pending_after", boundary.pending_readers_after);
    out.boolean_field("drained", boundary.source_stage_drained);
    out.boolean_field("released_ok", boundary.source_stage_backing_released);
    out.boolean_field("next_started", boundary.next_stage_started);
    return out.sha256();
}

std::string actual_boundary_canonical_digest(
        const ActualBoundaryReceipt &boundary) {
    CanonicalEncoder out("tc-streaming-actual-boundary-receipt-v1");
    out.string_field("event_digest", boundary.event_digest);
    out.unsigned_field("index", boundary.boundary_index);
    out.unsigned_field("from", boundary.from_stage_index);
    out.unsigned_field("to", boundary.to_stage_index);
    out.unsigned_field("source_generation", boundary.source_generation);
    return out.sha256();
}

void verify_actual_boundary_receipt(
        const ActualBoundaryReceipt &boundary, uint32_t expected_index,
        uint32_t expected_from_stage, uint32_t expected_to_stage,
        uint64_t source_generation) {
    receipt_check(boundary.boundary_index == expected_index &&
                      boundary.from_stage_index == expected_from_stage &&
                      boundary.to_stage_index == expected_to_stage,
                  "boundary_order_mismatch");
    receipt_check(!boundary.id.empty(), "boundary_id_missing");
    receipt_check(boundary.source_generation == source_generation &&
                      source_generation != 0,
                  "boundary_source_generation_mismatch");
    receipt_check(boundary.source_stage_drained,
                  "boundary_drain_incomplete");
    receipt_check(boundary.source_stage_backing_released,
                  "boundary_release_incomplete");
    receipt_check(!boundary.next_stage_started,
                  "boundary_next_stage_already_started");
    receipt_check(boundary.pending_readers_after == 0,
                  "boundary_pending_readers");
    receipt_check(boundary.completed_reader_sequence >=
                      boundary.last_reader_sequence,
                  "boundary_reader_sequence_mismatch");
    receipt_check(boundary.live_slot_bytes_after == 0,
                  "boundary_live_bytes_remaining");
    receipt_check(boundary.released_slot_bytes ==
                      boundary.live_slot_bytes_before,
                  "boundary_released_bytes_mismatch");
    receipt_check(boundary.event_digest ==
                      actual_boundary_event_digest(boundary),
                  "boundary_event_digest_mismatch");
    receipt_check(boundary.canonical_digest ==
                      actual_boundary_canonical_digest(boundary),
                  "boundary_canonical_digest_mismatch");
}

void verify_actual_execution_receipt(
        const Layout &layout, std::string_view implementation,
        std::string_view component_policy_revision,
        uint64_t source_generation,
        const ActualExecutionReceipt &receipt) {
    receipt_check(receipt.schema_version == actual_receipt_schema_v2 ||
                      receipt.schema_version == actual_receipt_schema_v3,
                  "schema_mismatch");
    receipt_check(receipt.implementation == implementation,
                  "implementation_mismatch");
    receipt_check(receipt.layout_digest == layout.digest,
                  "layout_mismatch");
    receipt_check(receipt.component_policy_revision ==
                      component_policy_revision,
                  "component_policy_mismatch");
    receipt_check(receipt.stages.size() == layout.stages.size(),
                  "stage_count_mismatch");
    for (uint32_t stage = 0; stage < layout.stages.size(); ++stage) {
        const auto &actual = receipt.stages[stage];
        receipt_check(!actual.groups.empty(), "group_matrix_mismatch");
        const uint64_t request_generation =
            actual.groups.front().request_generation;
        ExecutionReceiptOptions options{
            layout.digest, std::string(implementation), source_generation};
        verify_actual_stage_receipt(
            layout.stages[stage], stage, request_generation, options,
            actual);
    }
    if (receipt.schema_version == actual_receipt_schema_v2) {
        receipt_check(layout.stages.size() == 1 &&
                          receipt.boundaries.empty(),
                      "legacy_receipt_requires_single_stage");
    } else {
        receipt_check(receipt.boundaries.size() + 1 ==
                          layout.stages.size(),
                      "boundary_count_mismatch");
        for (uint32_t index = 0; index < receipt.boundaries.size(); ++index)
            verify_actual_boundary_receipt(
                receipt.boundaries[index], index, index, index + 1,
                source_generation);
    }
    receipt_check(receipt.canonical_digest ==
                      execution_canonical_digest(receipt),
                  "execution_digest_mismatch");
}

} // namespace tc::streaming

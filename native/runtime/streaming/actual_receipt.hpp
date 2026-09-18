#pragma once

#include "../../core/stream_slot_c.h"
#include "layout.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace tc::streaming {

inline constexpr uint32_t actual_receipt_schema_v2 = 2;
// Schema v3 is additive: v2 stage receipts remain valid, while a v3
// execution receipt can describe ordered stage-boundary drain/release events.
inline constexpr uint32_t actual_receipt_schema_v3 = 3;
inline constexpr std::string_view actual_receipt_verifier_revision =
    "tc-streaming-receipt-verifier-v2";
inline constexpr std::string_view actual_receipt_boundary_verifier_revision =
    "tc-streaming-boundary-verifier-v1";

struct ExecutionReceiptOptions final {
    std::string layout_digest;
    std::string implementation;
    uint64_t source_generation = 0;
};

struct ActualReaderReceipt final {
    tc_stream_reader_fence_v1 fence{};
    bool completed = false;
};

struct ActualGroupReceipt final {
    uint32_t pass = 0;
    uint32_t group = 0;
    uint32_t pool = 0;
    uint32_t slot = 0;
    uint32_t step = 0;
    uint32_t fill_count = 0;
    uint64_t request_generation = 0;
    uint64_t content_generation = 0;
    uint64_t expected_bytes = 0;
    uint64_t actual_bytes = 0;
    uint64_t source_generation = 0;
    bool fill_completed = false;
    bool group_submitted = false;
    uint32_t reader_count = 0;
    std::array<ActualReaderReceipt, TC_STREAM_MAX_READER_QUEUES> readers{};
};

struct ActualPoolSelectionReceipt final {
    uint32_t pass = 0;
    uint32_t ordinal = 0;
    uint32_t pool = 0;
};

struct ActualCarryReceipt final {
    uint32_t from_pass = 0;
    uint32_t to_pass = 0;
    uint32_t group = 0;
    uint32_t pool = 0;
    uint32_t slot = 0;
    uint64_t content_generation = 0;
};

struct ActualStageReceipt final {
    uint32_t schema_version = actual_receipt_schema_v2;
    uint32_t stage_index = 0;
    std::string stage_id;
    std::string layout_digest;
    std::string implementation;
    uint32_t completed_passes = 0;
    uint32_t completed_groups = 0;
    uint64_t fills = 0;
    uint64_t groups_submitted = 0;
    uint64_t logical_read_bytes = 0;
    uint64_t reader_fences_issued = 0;
    uint64_t reader_fences_completed = 0;
    uint64_t source_generation = 0;
    bool drain_completed = false;
    std::vector<ActualGroupReceipt> groups;
    std::vector<ActualPoolSelectionReceipt> pool_selections;
    std::vector<ActualCarryReceipt> carries;
    std::string event_digest;
    std::string canonical_digest;
};

// A boundary is recorded by the adapter at the point where the previous
// stage has drained and its slot/prefix backing has been released.  It is not
// inferred from two adjacent stage summaries: the ordering is part of the
// safety contract for multi-stage models such as LTX.
struct ActualBoundaryReceipt final {
    uint32_t boundary_index = 0;
    std::string id;
    uint32_t from_stage_index = 0;
    uint32_t to_stage_index = 0;
    uint64_t source_generation = 0;
    uint64_t last_reader_sequence = 0;
    uint64_t completed_reader_sequence = 0;
    uint64_t live_slot_bytes_before = 0;
    uint64_t live_slot_bytes_after = 0;
    uint64_t released_slot_bytes = 0;
    uint64_t pending_readers_before = 0;
    uint64_t pending_readers_after = 0;
    bool source_stage_drained = false;
    bool source_stage_backing_released = false;
    bool next_stage_started = false;
    std::string event_digest;
    std::string canonical_digest;
};

struct ActualExecutionReceipt final {
    uint32_t schema_version = actual_receipt_schema_v2;
    std::string implementation;
    std::string layout_digest;
    std::string component_policy_revision;
    std::vector<ActualStageReceipt> stages;
    std::vector<ActualBoundaryReceipt> boundaries;
    std::string canonical_digest;
};

// Owner-thread-only recorder. All vectors are sized before the first refill;
// fill workers and GPU callbacks continue to publish POD completion records.
class ActualReceiptRecorder final {
  public:
    ActualReceiptRecorder(
        const StageLayout &, uint32_t stage_index,
        uint64_t request_generation, ExecutionReceiptOptions);

    void pool_selected(uint32_t pass, uint32_t pool);
    void fill_submitted(const tc_stream_slot_ticket_v1 &);
    void fill_completed(const tc_stream_completion_v1 &);
    void readers_issued(
        const tc_stream_slot_ticket_v1 &,
        std::span<const tc_stream_reader_fence_v1>);
    void reader_completed(const tc_stream_completion_v1 &);
    void pass_completed(
        uint32_t pass,
        const std::optional<tc_stream_slot_ticket_v1> &carry);
    void drained();
    std::shared_ptr<const ActualStageReceipt> finish();

  private:
    StageLayout layout_;
    ExecutionReceiptOptions options_;
    ActualStageReceipt receipt_;
    uint64_t request_generation_ = 0;
    uint32_t pool_selection_cursor_ = 0;
    std::vector<uint32_t> pool_sequence_;
    std::vector<bool> pass_completed_;
    bool drained_ = false;
    bool finished_ = false;
    std::thread::id owner_;

    ActualGroupReceipt &group_for(
        const tc_stream_slot_ticket_v1 &);
    const ActualGroupReceipt &group_for(
        const tc_stream_slot_ticket_v1 &) const;
    uint32_t expected_slot(uint32_t pass, uint32_t group) const;
    void owner() const;
};

std::string actual_stage_event_digest(const ActualStageReceipt &);
std::string actual_stage_canonical_digest(const ActualStageReceipt &);

void verify_actual_stage_receipt(
    const StageLayout &, uint32_t stage_index,
    uint64_t request_generation, const ExecutionReceiptOptions &,
    const ActualStageReceipt &);

// Deep-copy a sealed C ABI receipt produced by the same common executor. This
// preserves the real event matrix for C model runtimes such as H3; callers
// must still verify the result against the authorized Layout.
ActualStageReceipt actual_stage_receipt_from_c_v2(
    const tc_stream_receipt_v2 &);

ActualExecutionReceipt make_actual_execution_receipt(
    std::string implementation, std::string layout_digest,
    std::string component_policy_revision,
    std::vector<ActualStageReceipt> stages);

ActualExecutionReceipt make_actual_execution_receipt_v3(
    std::string implementation, std::string layout_digest,
    std::string component_policy_revision,
    std::vector<ActualStageReceipt> stages,
    std::vector<ActualBoundaryReceipt> boundaries);

std::string actual_boundary_event_digest(const ActualBoundaryReceipt &);
std::string actual_boundary_canonical_digest(const ActualBoundaryReceipt &);

void verify_actual_boundary_receipt(
    const ActualBoundaryReceipt &, uint32_t expected_index,
    uint32_t expected_from_stage, uint32_t expected_to_stage,
    uint64_t source_generation);

void verify_actual_execution_receipt(
    const Layout &, std::string_view implementation,
    std::string_view component_policy_revision,
    uint64_t source_generation,
    const ActualExecutionReceipt &);

} // namespace tc::streaming

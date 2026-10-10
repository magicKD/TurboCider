#pragma once

#include "native_denoiser.hpp"
#include "../../backends/metal_graph/memory_bridge.hpp"
#include "../../runtime/memory_execution.hpp"

namespace tc::qwen21::native_graph {

struct NativePhaseShape {
    DenoiserGeometry geometry;
    uint32_t target_rows = 0, text_rows = 0;
    std::vector<uint32_t> reference_rows;
    // W1 uses one bank and one live producer/restore status pair. Prefix build
    // stays GPU-only; these allocations are reused by every steady block.
    bool hybrid_ffn = false;
    // Zero disables reuse. Otherwise one compact gate/up/down entry per layer.
    uint32_t weight_scale_cache_entries = 0;
    // Actual resolved runtime factors, not a user-declared adapter capability.
    // One low workspace is shared by all sequential global/block projections.
    uint32_t lora_max_rank = 0;
    // Any FFN term currently needs gate/up BF16 scratch; BaseOnly and
    // attention/global-only adapters do not reserve those two backings.
    bool lora_ffn = false;
};
struct NativePhaseLimits {
    // Optional full process cap; never pass the legacy denoiser-only budget hint.
    uint64_t request_budget_bytes = 0; // Zero: physical memory minus system reserve.
    // Explicit allowance, NOT a certified compiler/driver envelope. Includes
    // PSOs, host graph/request/event objects, ANE program/constants/compiler
    // workspace and the small final MLX publication backing. Owned Metal and
    // IOSurface storage below is charged separately through the same bridge.
    uint64_t framework_upper_bytes = uint64_t(1) << 30;
    uint64_t system_reserve_bytes = uint64_t(4) << 30;
};
struct NativePhaseAllocation {
    std::string label, site_id;
    MemoryClass memory_class = MemoryClass::Activation;
    uint64_t payload_bytes = 0, capacity_upper_bytes = 0;
    uint32_t instances = 0;
};
struct NativePhaseMemoryPlan {
    MemoryManifest manifest;
    CompiledMemoryPlan compiled;
    EffectiveMemoryPolicy policy;
    std::vector<metal_graph::AllocationSiteBinding> bindings;
    std::vector<NativePhaseAllocation> allocations;
    uint64_t payload_bytes = 0, allocation_upper_bytes = 0;
    uint32_t backing_count = 0;
};

// Metadata only: no checkpoint/device/MLX access and no capability grant.
// Baseline must be captured AFTER source DiT/VAE/input production, external
// queue synchronization and allocator-cache clearing. Their immutable retained
// owners remain covered by that baseline through physical drain. They are not
// fabricated per-array ledger claims. Only native begin->drain allocations are
// described here; encoder/VAE/full-request constrained execution is uncertified.
NativePhaseMemoryPlan plan_native_phase(const NativePhaseShape&, std::string source_revision,
    const ProcessMemoryObservation& baseline, NativePhaseLimits = {});

// One existing MemoryExecutionContext + existing MemoryBridge for this phase.
// A parent context is rejected: no nested ledger or silent full-request claim.
// Caller retains this owner through physical drain, destroys native run/buffer/
// source wrappers, then finishes; source baseline owners themselves must not be
// mutated/reallocated while GPU work may still read them.
class NativePhaseMemory {
public:
    NativePhaseMemory(const NativePhaseShape&, std::string source_revision,
        const ProcessMemoryObservation& baseline, NativePhaseLimits,
        uint64_t allocator_domain, uint64_t generation,
        MemoryExecutionContext* parent = nullptr, ProcessMemoryObserver observer = {});
    ~NativePhaseMemory() noexcept;
    NativePhaseMemory(const NativePhaseMemory&) = delete;
    NativePhaseMemory& operator=(const NativePhaseMemory&) = delete;
    metal_graph::AllocationHooks hooks() const;
    MemoryExecutionContext& context() noexcept { return *context_; }
    const NativePhaseMemoryPlan& plan() const noexcept { return plan_; }
    void finish_after_release(bool physically_drained);
    MemoryFailureDisposition fail_after_drain(bool physically_drained, std::string reason) noexcept;
    void close() noexcept;
private:
    NativePhaseMemoryPlan plan_;
    std::unique_ptr<MemoryExecutionContext> context_;
    std::unique_ptr<metal_graph::MemoryBridge> bridge_;
};

} // namespace tc::qwen21::native_graph

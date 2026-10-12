#pragma once

#include "native_ane_stages.hpp"
#include "native_hybrid_state.hpp"

#include <filesystem>

namespace tc::metal_graph {

struct NativeHybridFfnOptions {
    uint32_t rows = 1024, hidden = 4096, width = 12288, ane_channels = 6144;
    // Explicit cache ownership: no implicit user/global cache for this route.
    std::filesystem::path program_cache;
    // Adapt the SAME MemoryBridge hooks used by Backend, never a new ledger.
    // Empty only for standalone components. Compiler/driver workspace remains
    // an observed/guarded external cost, not certified by these storage hooks.
    tc::ane::gpu::SurfaceAllocationHooks surface_allocation;
    // Explicit candidate; generic is retained as the same-source oracle.
    bool register_weight_staging = false;
    // Request-local immutable-source scale reuse. Empty/zero keeps the old
    // route. Trusted DenoiserRun supplies its held source revision/generation;
    // a public caller string alone is not an immutability certificate.
    uint32_t weight_scale_cache_entries = 0;
    std::string weight_scale_source_revision;
    uint64_t weight_scale_source_generation = 0;
};

struct NativeHybridFfnPlan {
    // Internal trusted builder input: NativeBlock::plan_parallel or the typed
    // generated FFN fixture must record graphs for THESE exact bindings using
    // gpu_complement(). CommandGraph has no semantic write-set certificate;
    // this controller cannot authenticate arbitrary caller-provided shaders.
    // It is not an artifact/plugin or public request execution API.
    CommandGraph before_ffn, gpu_ffn, after_join;
    MatrixView input, gate, up, down;
    MatrixView gpu_partial, ane_partial; // FP32, distinct and disjoint sources.
    MatrixView published_output; // BF16 result scanned after after_join.
    uint32_t weight_scale_cache_slot = UINT32_MAX;
};

struct NativeHybridFfnStatus {
    bool ok = false, retired = false, running = false;
    bool all_work_physically_complete = true;
    uint64_t generation = 0;
    HybridPhysicalEvidence evidence;
    std::string error;
};

struct NativeHybridFfnResult {
    NativeHybridFfnStatus completion;
    double producer_seconds = 0, submit_ane_seconds = 0;
    double branch_join_seconds = 0, consumer_seconds = 0, total_seconds = 0;
    // Actual Metal ticket first-start/last-end spans, including inter-CB gaps.
    // Zero means unavailable; these do not grant permits or prove ANE overlap.
    double producer_gpu_span_seconds = 0, gpu_complement_span_seconds = 0;
    bool weight_scale_cache_hit = false, weight_scale_cache_miss = false;
};

// W1: one channel bank, one request slot, explicit producer admission, ANE
// submission BEFORE the complementary GPU FFN, then restore/join. Host control
// is synchronous; the two device branches may overlap. No W2 prefetch, LoRA,
// full-denoiser or hardware-INT8/performance qualification is implied.
class NativeHybridFfn {
public:
    static NativeHybridFfn prepare(Backend&, NativeHybridFfnOptions);
    NativeHybridFfn(const NativeHybridFfn&) = delete;
    NativeHybridFfn& operator=(const NativeHybridFfn&) = delete;
    NativeHybridFfn(NativeHybridFfn&&) noexcept;
    NativeHybridFfn& operator=(NativeHybridFfn&&) noexcept;
    ~NativeHybridFfn() noexcept;

    // Prepare a NativeBlock::plan_parallel using this immutable operator bundle.
    const NativeFfn& gpu_complement() const;
    NativeHybridFfnResult execute(NativeHybridFfnPlan,
        std::chrono::milliseconds timeout = std::chrono::seconds(30));
    using Deadline = std::chrono::steady_clock::time_point;
    // Enclosing DiT steps share ONE deadline across all layers and postgraph.
    // Converting a relative ceil wait back to a new deadline would extend it.
    NativeHybridFfnResult execute_until(NativeHybridFfnPlan, Deadline,
        const std::function<void()>& concurrent_prepare = {});
    // Nonblocking physical collection never revives a retired result. On
    // failure the slot is permanently retired even when late callbacks arrive.
    NativeHybridFfnStatus status() const;
    NativeHybridFfnStatus drain(std::chrono::milliseconds timeout);
    size_t surface_bytes() const;
    size_t weight_scale_cache_payload_bytes() const;
    uint32_t weight_scale_cache_capacity() const;
    const std::string& program_key() const;
    static void validate_options(const NativeHybridFfnOptions&);
private:
    struct Impl;
    explicit NativeHybridFfn(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
    void preserve_pending() noexcept;
};

} // namespace tc::metal_graph

#pragma once

#include "native_block.hpp"
#include "../../backends/metal_graph/native_step_ops.hpp"

#include <memory>
#include <filesystem>
#include <string>

namespace tc::qwen21::native_graph {

struct DenoiserGeometry {
    BlockGeometry block;
    uint32_t layers = 32;
    uint32_t maximum_rows = 8192;
};
void validate_geometry(DenoiserGeometry);

// Constants only, not host tensor arithmetic. Same precision/ordering as the
// current Qwen Transformer. The second time row is the zero-time prefix.
std::array<uint16_t,512> timestep_input(float sigma);
struct RopeTables { std::vector<float> cosine, sine; };
RopeTables make_rope_tables(const SequenceGeometry&);

struct DenoiserWeights {
    Projection image_input, time_input, time_output, text_input, text_output;
    Projection modulation, output_scale, output;
    MatrixView text_norm;
    std::vector<BlockWeights> blocks;
};
// Revisions come from the enclosing source/conditioning lease. These are
// provenance labels, not a substitute for validating those leases at import.
struct DenoiserIdentity {
    std::string source_revision, conditioning_revision, adapter_revision;
    uint64_t generation = 0;
};
struct NativeReferenceInput { MatrixView latents; ReferenceGeometry geometry; };
struct DenoiserInput {
    MatrixView latents; // Produced BF16 [height*width,64].
    MatrixView text;    // Produced BF16 [text_tokens,H].
    std::vector<NativeReferenceInput> references;
    int height = 0, width = 0; // Packed latent token geometry, not image pixels.
    DenoiserIdentity identity;
    // Exact schedule from the existing pipeline; includes the final zero.
    std::vector<float> sigmas;
};
struct DenoiserMemory {
    // Payloads allocated directly by this denoiser (including partials), not a
    // process peak. Controller surfaces are reported separately; its status/
    // signs allocations and external compiler costs belong to the phase plan.
    size_t request_bytes = 0, block_arena_bytes = 0, prefix_bytes = 0;
    size_t hybrid_partial_bytes = 0, hybrid_surface_bytes = 0;
    size_t lora_scratch_bytes = 0;
};
enum class DenoiserScheduleVariant : uint8_t {
    // Compatibility for existing base fixtures only; never infers a LoRA variant.
    AutomaticBase, Base20, ViggleLoRA6, GeneratedTiny
};
struct NativeDenoiserExecutionOptions {
    bool hybrid_ffn = false;
    std::filesystem::path ane_program_cache;
    // The SAME hooks used by Backend. Surface imports reuse their existing
    // token; they must not create another reservation or memory ledger.
    metal_graph::AllocationHooks allocation;
    bool register_weight_staging = false;
    bool cache_weight_scales = false;
    DenoiserScheduleVariant schedule_variant = DenoiserScheduleVariant::AutomaticBase;
    std::string adapter_revision;
    metal_graph::native_lora::KernelVariant lora_kernel = metal_graph::native_lora::KernelVariant::Scalar;
};
struct DenoiserHybridMetrics {
    bool enabled = false;
    uint64_t successful_ffns = 0, completed_hybrid_steps = 0;
    size_t surface_bytes = 0, partial_bytes = 0;
    // Successful FFN calls only, including calls before a later step failure.
    // These are host stage intervals; branch_join includes the overlap and
    // must never be presented as GPU time plus ANE time or as a speedup.
    double producer_seconds = 0, submit_ane_seconds = 0, branch_join_seconds = 0;
    double consumer_seconds = 0, total_seconds = 0;
    double producer_gpu_span_seconds = 0, gpu_complement_span_seconds = 0;
    uint64_t device_timed_ffns = 0;
    std::string weight_staging = "generic";
    bool weight_scale_cache_enabled = false;
    uint32_t weight_scale_cache_entries = 0;
    size_t weight_scale_cache_payload_bytes = 0;
    // One lookup covers gate/up/down for one successful layer FFN.
    uint64_t weight_scale_cache_hits = 0, weight_scale_cache_misses = 0;
};
enum class DenoiserState : uint8_t { Prepared, Running, Complete, Retired };
class DenoiserRun;

// All DiT dispatches use the owned native Metal queue. No MLX operations,
// source copies, per-layer PSO preparation or host activation arithmetic.
// Every adapted projection preserves ordered terms and the original BF16
// rounding boundaries. Hybrid LoRA remains explicitly rejected.
class NativeDenoiser {
public:
    static NativeDenoiser prepare(metal_graph::Backend&, DenoiserWeights,
                                 std::string source_revision, DenoiserGeometry = {},
                                 NativeDenoiserExecutionOptions = {});
    DenoiserGeometry geometry() const noexcept;
    // Prepared bundle's actual arithmetic recipe; empty for BaseOnly.
    std::string_view lora_precision_recipe() const noexcept;
    // The backend and immutable imported source backings must remain alive.
    // Caller proves external queue readiness and conditioning immutability.
    std::unique_ptr<DenoiserRun> begin(DenoiserInput) const;
private:
    struct Impl;
    explicit NativeDenoiser(std::shared_ptr<Impl>);
    std::shared_ptr<Impl> impl_;
    friend class DenoiserRun;
};

// Request-local cache/workspace transaction. One arena is reused across all
// layers and both phases. Results become publishable only after success,
// physical GPU completion and the finite check. A failed run is retired.
class DenoiserRun {
public:
    ~DenoiserRun();
    DenoiserRun(const DenoiserRun&) = delete;
    DenoiserRun& operator=(const DenoiserRun&) = delete;
    metal_graph::Completion step(std::chrono::milliseconds timeout = std::chrono::seconds(30));
    // Aggregate of the retained GPU ticket and every live hybrid leg, not a
    // result-publication permit. Previous slots are discarded only after their
    // producer, ANE, GPU complement and restore consumer physically complete.
    // For active hybrid work, gpu_seconds is unavailable (zero), and buffer
    // counts cover only the directly retained prefill/postprocessing GPU ticket.
    // Before any submission, returns ok/physically_complete with zero buffers:
    // there is no device work to drain, but the schedule is not yet complete.
    // Serialized with the synchronous step; not an asynchronous status API.
    metal_graph::Completion physical_status() const;
    // Retains every pending domain after failure so a caller can prove late device
    // completion. Zero is a read-only status query; a positive timeout may
    // retire the run. Neither success nor a late callback revives a retired run
    // or publishes results. Negative timeouts are rejected.
    metal_graph::Completion drain(std::chrono::milliseconds timeout = std::chrono::seconds(30));
    // Available only after the final step; no further writes can follow.
    MatrixView final_latents() const;
    DenoiserState state() const;
    size_t completed_steps() const;
    DenoiserMemory memory() const noexcept;
    DenoiserHybridMetrics hybrid_metrics() const;
private:
    struct Impl;
    explicit DenoiserRun(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
    friend class NativeDenoiser;
};

MatrixView matrix_rows(const MatrixView&, uint32_t first, uint32_t count);
MatrixView matrix_columns(const MatrixView&, uint32_t first, uint32_t count);

} // namespace tc::qwen21::native_graph

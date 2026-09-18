#pragma once
#include "common.hpp"
#include "memory_accounting.hpp"
#include "memory_manifest.hpp"
#include "memory_policy.hpp"
#include "memory_trace.hpp"
#include <chrono>
#include <map>
#include <memory>
#include <optional>
namespace tc {
namespace streaming {
struct PublicResolveInput;
struct StreamingPresetRecord;
class ModelStreamingProbe;
class ModelStreamingSnapshot;
struct ResolvedRequestExecution;
struct ActualExecutionReceipt;
} // namespace streaming

class MemoryExecutionContext;
struct ExecutionPlan {
    Request request;
    Recipe recipe;
    std::optional<uint64_t> memory_estimate_bytes;
    std::optional<EffectiveMemoryPolicy> memory_policy;
};
ExecutionPlan make_plan(const Request &);
// Internal public-streaming path: request-only public validation has already
// succeeded and is sealed by PublicStreamingPreflight. Default callers must
// continue to use make_plan().
ExecutionPlan make_plan_after_public_streaming_preflight(const Request &);
std::string effective_lora_strategy(const Request &);
struct HybridMetrics {
    double load_seconds = 0;
    double manifest_validation_seconds = 0;
    double output_backing_setup_seconds = 0;
    double model_load_seconds = 0;
    double model_interface_setup_seconds = 0;
    double zero_input_warmup_seconds = 0;
    double prediction_seconds = 0;
    double first_runtime_prediction_seconds = 0;
    double subsequent_runtime_prediction_seconds = 0;
    uint64_t calls = 0, copied_bytes = 0;
    uint64_t warmup_calls = 0, runtime_calls = 0;
    uint64_t first_runtime_prediction_calls = 0;
    uint64_t subsequent_runtime_prediction_calls = 0;
    uint64_t runtime_failures = 0;
    bool runtime_failed = false;
    int runtime_failure_block = -1;
    uint64_t quality_validation_calls = 0;
    double quality_max_relative_l2 = 0;
    double quality_min_cosine = 1;
    double quality_max_abs = 0;
    double quality_max_relative_abs = 0;
    bool quality_validation_passed = true;
    int prefill_actual_tokens = 0, prefill_selected_bucket = 0,
        prefill_compute_tokens = 0, prefill_padding_tokens = 0;
    bool prefill_fixed_shape = false;
    std::string prefill_plan_reason;
    int bucket = 0, hidden = 0, block_count = 0;
    int minimum_profitable_rows = 0;
    int mlp_width = 0, ane_mlp_start = 0, ane_mlp_end = 0;
    float output_scale = 1.f;
    bool qualified_flexible_backing = false;
    bool checkpoint_sha_verified = false;
    bool lora_identity_verified = false;
};
struct LoadResult {
    uint64_t weight_bytes = 0, active_bytes = 0;
};
struct Timings {
    double wall = 0, text = 0, image = 0, hybrid = 0, denoise = 0, decode = 0;
};
struct BlockResidencyMetrics {
    bool enabled = false, fully_resident = false, quantized = false;
    unsigned active_blocks = 0, pinned_blocks = 0, streamed_blocks = 0;
    unsigned refill_slots = 0;
    uint64_t memory_budget_bytes = 0, activation_reserve_bytes = 0;
    uint64_t block_bytes = 0, estimated_working_set_bytes = 0;
    uint64_t request_bytes_loaded = 0, request_slot_allocations = 0;
    uint64_t request_slot_refills = 0, request_slot_fills = 0;
    double request_load_seconds = 0, request_wait_seconds = 0;
    double request_refill_load_seconds = 0, request_max_refill_seconds = 0;
    int request_max_refill_block = -1;
};
struct StreamingRuntimeMetrics {
    std::string implementation, layout_digest;
    std::string stage = "denoiser";
    uint32_t resident_prefix_blocks = 0, block_group_size = 0;
    uint32_t slot_count = 0, prefetch_distance = 0, io_workers = 0;
    uint32_t group_count = 0, pass_count = 0;
    std::string startup_policy, pass_transition, retention;
    uint32_t reader_revision = 0;
    std::string weight_format, kernel_revision, conditioning_recipe;
    std::string upsample_boundary;
    std::string component_policy_revision, multi_pool_policy;
    uint32_t pool_count = 0, slot_bundle_count = 0;
    uint32_t refill_worker_count = 0;
    bool source_lease_verified = false, drained = false;
    uint32_t receipt_schema_version = 0;
    uint64_t receipt_fills = 0, receipt_groups_submitted = 0;
    uint64_t receipt_logical_read_bytes = 0;
    uint64_t receipt_reader_fences_issued = 0;
    uint64_t receipt_reader_fences_completed = 0;
    uint64_t receipt_source_generation = 0;
    std::string receipt_event_digest, receipt_digest;
    std::string receipt_verifier_revision;
};
struct StreamingStageRuntimeMetrics {
    uint32_t stage_index = 0;
    StreamingRuntimeMetrics runtime;
};
struct StreamingBoundaryRuntimeMetrics {
    uint32_t boundary_index = 0;
    std::string id;
    std::string from_stage;
    std::string to_stage;
    bool source_stage_drained = false;
    bool source_stage_backing_released = false;
    uint64_t live_slot_bytes_before = 0;
    uint64_t live_slot_bytes_after = 0;
    uint64_t pending_readers_before = 0;
    uint64_t pending_readers_after = 0;
    uint64_t released_slot_bytes = 0;
    std::string event_digest;
};
struct PublicStreamingSelectionMetrics {
    uint64_t target_request_memory_bytes = 0;
    uint64_t calibrated_request_bytes = 0;
    uint32_t preset_revision = 0;
    std::string preset_id, catalog_revision, record_digest;
    std::string resolution_digest, source_digest, workload_digest;
    std::string runtime_digest, device_digest;
    std::string authorized_layout_digest, actual_layout_digest;
    std::string component_policy_revision, execution_container;
    std::string memory_scope;
    uint32_t receipt_schema_version = 0;
    uint64_t receipt_source_generation = 0;
    std::string receipt_digest, receipt_verifier_revision;
    bool actual_plan_verified = false;
};
struct RunResult {
    bool prepared = false, warmup = false, prompt_cache_hit = false;
    std::string selection, backend, precision, checkpoint;
    Request request;
    ExecutionPlan plan;
    int text_tokens = 0, valid_text_tokens = 0, total_tokens = 0, reference_tokens = 0,
        actual_steps = 0;
    size_t lora_applied_projections = 0;
    Timings timings;
    uint64_t active_bytes = 0, peak_bytes = 0;
    std::optional<HybridMetrics> hybrid;
    std::optional<HybridMetrics> encoder_hybrid;
    std::optional<BlockResidencyMetrics> block_residency;
    std::optional<StreamingRuntimeMetrics> streaming_runtime;
    std::vector<StreamingStageRuntimeMetrics> streaming_stages;
    std::vector<StreamingBoundaryRuntimeMetrics> streaming_boundaries;
    std::shared_ptr<const streaming::ActualExecutionReceipt>
        streaming_receipt;
    std::optional<PublicStreamingSelectionMetrics> public_streaming;
    std::optional<MemoryAdmissionMetrics> memory_admission;
    std::vector<MemoryTraceEvent> memory_trace;
    std::string native_json;
};
struct MemoryDrainResult {
    bool completed = true;
    uint64_t completions = 0;
    uint64_t pending_after = 0;
    std::string failure;
};
class ModelSession {
  public:
    virtual ~ModelSession() = default;
    virtual bool uses_parent_mlx() const { return true; }
    virtual bool uses_parent_mlx(const Request &) const { return uses_parent_mlx(); }
    /* Non-owning binding valid only for the duration of one admitted API
     * call. Backends that install native allocator callbacks must keep their
     * callback bridge stable and clear the admission at call exit. */
    virtual void set_memory_admission(MemoryAdmission *) {}
    /* New constrained-memory adapters bind the complete request context.
     * The default is deliberately inert so legacy/default execution has no
     * extra work.  During migration the API layer also installs the older
     * admission pointer before calling this hook. */
    virtual void bind_memory_context(MemoryExecutionContext *) {}
    virtual void unbind_memory_context() noexcept {}
    /* Constrained adapters must make all GPU uses of accounted backings
     * complete before finish_success(). Synchronous legacy adapters may use
     * the inert default; adapters with asynchronous manifest sites override
     * this hook and report any remaining completions. */
    virtual MemoryDrainResult drain_memory_completions(
        MemoryExecutionContext &, std::chrono::milliseconds) {
        return {};
    }
    /* Metadata-only probe. Implementations may inspect checkpoint indexes and
     * immutable model manifests, but must not materialize weights, compile
     * graphs, create GPU buffers, or grant capability themselves. */
    virtual std::optional<MemoryCapabilityProbe> probe_memory_capability(
        const ExecutionPlan &, const MemoryDeviceIdentity &) const {
        return std::nullopt;
    }
    /* Public streaming is a separate authority path from private/manual exact
     * layouts.  Metadata probes and snapshot compilation must not allocate GPU
     * buffers or start refill workers.  Models gain no public eligibility until
     * all three methods are explicitly overridden. */
    virtual std::shared_ptr<const streaming::ModelStreamingProbe>
    probe_public_streaming(const streaming::PublicResolveInput &) const {
        throw std::runtime_error("streaming_public_adapter_unsupported");
    }
    virtual std::shared_ptr<const streaming::ModelStreamingSnapshot>
    compile_public_streaming(
        std::shared_ptr<const streaming::ModelStreamingProbe>,
        const streaming::StreamingPresetRecord &) const {
        throw std::runtime_error("streaming_public_adapter_unsupported");
    }
    virtual RunResult generate_resolved(
        std::shared_ptr<const streaming::ResolvedRequestExecution>,
        const Event &, std::atomic<bool> &) {
        throw std::runtime_error("streaming_public_adapter_unsupported");
    }
    virtual RunResult generate(const Request &, const Event &, std::atomic<bool> &) = 0;
    virtual LoadResult load(const Event &, std::atomic<bool> &) {
        throw std::runtime_error("explicit loading unavailable");
    }
    virtual void unload() = 0;
    virtual RunResult prepare(const Request &, bool, const Event &, std::atomic<bool> &) {
        throw std::runtime_error("preparation unavailable");
    }
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
    /* Test-build-only lifecycle control. It is intentionally absent from
     * release binaries and from the public C header/request schema. */
    virtual void test_set_ltx_exact_destroy_failures(uint32_t) {
        throw std::runtime_error("LTX exact lifecycle test hook unavailable");
    }
    virtual void test_cancel_ltx_exact_first_fill() {
        throw std::runtime_error("LTX exact first-fill test hook unavailable");
    }
#endif
};
struct ModelDescriptor {
    std::string id, name;
    bool executable = false;
    std::vector<std::string> operations, inputs, roles;
    int max_images = 0;
    std::string output;
    int steps = 4, frames = 1, width = 512, height = 512;
    int fps = 0;
    bool default_audio = true;
    std::string default_residency = "resident";
    bool weight_validation_pending = false;
    bool supports_lora = false, runtime_lora = false, supports_gpu_ane = false;
    bool supports_encoder_gpu_ane = false;
    bool native_gemma4_candidate = false, native_conditioning_connector = false;
    bool native_i2v_clean_prefix = false, native_gpu_ane_profile = false;
    bool native_audio_output_candidate = false, native_audio_vae_candidate = false;
    bool native_base_vocoder_candidate = false, audio_output = false;
    bool request_lora_identity_validation = false;
    std::string backend, lora_mode, runtime_dependency, parallel_strategy, audio_capability;
    std::vector<std::string> lora_strategies;
    std::string default_lora_strategy;
    std::vector<std::string> executor_operations, candidate_limitations;
};
struct ModelModule {
    std::string id;
    std::function<Recipe()> recipe;
    std::function<void(const Request &)> validate;
    std::function<std::unique_ptr<ModelSession>(const std::filesystem::path &)> create;
    std::function<ModelDescriptor()> describe;
};
const ModelModule &module_for(const std::string &);
std::vector<ModelDescriptor> describe_modules();
Recipe model_recipe(const std::string &);
void validate_recipe(const Recipe &);
std::vector<float> flux_sigmas(int, int);

} // namespace tc

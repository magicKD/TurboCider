#pragma once
#include "common.hpp"
#include "../core/ane_calibration_report.hpp"
#include "../core/ane_overflow_report.hpp"
#include "../core/ane_weight_code_cache_report.hpp"
#include "memory_accounting.hpp"
#include "memory_manifest.hpp"
#include "memory_policy.hpp"
#include "memory_trace.hpp"
#include "streaming/gguf_packed_metrics.hpp"
#include "tensor_metrics.hpp"
#include <chrono>
#include <map>
#include <memory>
#include <optional>
namespace tc {
namespace streaming {
struct PublicResolveInput;
struct StreamingPresetRecord;
class ModelStreamingProbe;
class SourceLease;
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
    std::shared_ptr<const ane::ChannelCalibrationReport> runtime_weight_calibration;
    std::string runtime_weight_backend, runtime_weight_backend_fallback_reason;
    std::string runtime_weight_io_path;
    std::string runtime_weight_data_path;
    std::string runtime_weight_partition_axis = "rows";
    std::string runtime_weight_row_placement = "suffix";
    uint64_t runtime_weight_row_suffix_blocks=0,runtime_weight_row_prefix_blocks=0;
    uint64_t runtime_weight_row_image_tail_blocks=0,runtime_weight_row_protected_rows=0;
    uint64_t runtime_weight_row_pack_peak_bytes=0;
    int runtime_weight_ane_channels = 0, runtime_weight_gpu_channels = 0;
    uint64_t runtime_weight_channel_blocks = 0;
    bool runtime_weight_prefetch_enabled = false;
    bool runtime_weight_prefetch_after_gpu = true;
    uint64_t runtime_weight_prefetch_submissions = 0, runtime_weight_prefetch_hits = 0;
    uint64_t runtime_weight_prefetch_discards = 0, runtime_weight_prefetch_failures = 0;
    double runtime_weight_prefetch_wait_seconds = 0;
    bool runtime_weight_scale_cache_enabled = false;
    ane::WeightCodeCacheReport runtime_weight_code_cache;
    bool runtime_weight_stage_specialized = false;
    uint64_t runtime_weight_stage_pipeline_variants = 0;
    bool runtime_weight_launch_fence_enabled = false;
    bool runtime_weight_a8_lookahead_enabled = false;
    bool runtime_weight_fp32_channel_join_enabled = false;
    int runtime_weight_a8_group_size = 0;
    int runtime_weight_hidden_a8_group_size = 0;
    uint64_t runtime_weight_a8_prefetches = 0;
    double runtime_weight_a8_wait_seconds = 0;
    uint64_t runtime_weight_scale_cache_hits = 0, runtime_weight_scale_cache_misses = 0;
    uint64_t runtime_weight_scale_cache_entries = 0, runtime_weight_scale_cache_bytes = 0, runtime_weight_scale_cache_evictions = 0;
    uint64_t runtime_weight_device_io_calls = 0;
    uint64_t runtime_weight_lora_channel_range_calls = 0, runtime_weight_lora_channel_full_calls = 0;
    // Runtime-weight route only; all times/counts are session cumulative.
    uint64_t runtime_weight_slot_bytes = 0, runtime_weight_estimated_bytes = 0;
    uint64_t runtime_weight_hybrid_blocks = 0, runtime_weight_gpu_blocks = 0;
    uint64_t runtime_weight_untimed_hybrid_blocks = 0; // hybrid subset without whole-block timing fences
    uint64_t runtime_weight_async_hybrid_blocks = 0; // untimed subset without a separate GPU-head wait
    bool runtime_weight_deferred_join_enabled = false;
    uint64_t runtime_weight_deferred_join_blocks = 0; // explicit channel/untimed subset; output remains owned
    uint64_t runtime_weight_unsplit_gpu_blocks = 0; // subset of GPU blocks; no FFN bridge
    uint64_t runtime_weight_full_gpu_probe_blocks = 0; // measured subset of unsplit GPU blocks
    double runtime_weight_full_gpu_probe_seconds = 0;
    uint64_t runtime_weight_fallback_blocks = 0, runtime_weight_ane_rows = 0;
    uint64_t runtime_weight_overflow_retries = 0;
    ane::OverflowReport runtime_weight_overflow_events;
    std::vector<int> runtime_weight_gpu_layers;
    uint64_t runtime_weight_forced_gpu_blocks = 0;
    double runtime_weight_stage_seconds = 0, runtime_weight_stage_wait_seconds = 0;
    double runtime_weight_join_seconds = 0, runtime_weight_gpu_seconds = 0;
    // Async steady blocks are excluded from the two branch timers above.
    // This host wait overlaps GPU work; it is NOT exposed ANE latency.
    double runtime_weight_async_ane_wait_seconds = 0;
    // Serialized host spans around the parallel branches. post_join includes
    // output restoration, optional down-LoRA and final concatenation/eval;
    // async post_join also includes any GPU-head work still outstanding at
    // the final output fence. None is a GPU kernel/physical overlap timer.
    // For a deferred block this is only host graph construction;
    // subsequent GPU consumption is charged to the complete request wall.
    double runtime_weight_lora_gate_up_seconds = 0, runtime_weight_post_join_seconds = 0;
    // Combined attention/input + LoRA correction readiness; subset of pre,
    // not a pure LoRA kernel timer or part of the parallel FFN window.
    double runtime_weight_lora_input_ready_seconds = 0;
    double runtime_weight_wall_seconds = 0;
    double runtime_weight_pre_seconds = 0;
    float runtime_weight_headroom = 1.f;
    std::string runtime_weight_source_recipe;
    uint64_t runtime_weight_convrot_stage_submissions = 0;
    // Exporter-declared weight variant; unknown for legacy manifests without it.
    std::string weight_variant = "unknown";
    double load_seconds = 0;
    double manifest_validation_seconds = 0;
    double output_backing_setup_seconds = 0;
    double model_load_seconds = 0;
    double model_interface_setup_seconds = 0;
    double zero_input_warmup_seconds = 0;
    double prediction_seconds = 0;
    // Cumulative Core ML API phase wall times (including warmup). These are
    // diagnostic host spans; none establishes physical ANE placement.
    double feature_binding_seconds = 0;
    double model_prediction_seconds = 0;
    double output_handling_seconds = 0;
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
    int bucket = 0, hidden = 0, output_channels = 0, block_count = 0;
    std::string mlp_output_kind;
    int minimum_profitable_rows = 0;
    int mlp_width = 0, ane_mlp_start = 0, ane_mlp_end = 0;
    float output_scale = 1.f;
    bool qualified_flexible_backing = false;
    bool checkpoint_sha_verified = false;
    bool lora_identity_verified = false;
    // Native session handles were released at the encoder/denoiser boundary.
    // Does not assert that Core ML's out-of-process caches were evicted.
    bool session_released_after_encoding = false;
};
// Independent from HybridMetrics: QKV projections and FFNs must never be
// reported as the same accelerator or share their cumulative call counts.
struct QkvMetrics {
    uint64_t calls = 0, hybrid_blocks = 0, gpu_blocks = 0, fallback_blocks = 0;
    uint64_t gpu_probe_blocks = 0, measured_hybrid_blocks = 0;
    uint64_t ane_rows = 0, failures = 0, slot_bytes = 0, estimated_bytes = 0;
    int failure_block = -1, chunk_rows = 0, chunks = 0;
    double load_seconds = 0, stage_seconds = 0, stage_wait_seconds = 0;
    double prediction_seconds = 0, output_seconds = 0, wall_seconds = 0;
    double gpu_probe_seconds = 0, measured_hybrid_seconds = 0;
    bool auto_scheduling = false;
    bool failed = false;
    std::string failure_reason;
};
inline std::string hybrid_precision_label(const HybridMetrics &metrics) {
    if (metrics.weight_variant == "fp16") return "bf16_gpu+fp16_mlp_fp16_io";
    if (metrics.weight_variant == "int8_pc") return "bf16_gpu+int8_mlp_fp16_io";
    return "bf16_gpu+coreml_mlp_fp16_io";
}
struct LoadResult {
    uint64_t weight_bytes = 0, active_bytes = 0;
};
// Request-local evidence for an optional encoder executor. HybridMetrics
// counters remain session cumulative when a resident graph is reused.
struct EncoderRuntimeReuseMetrics {
    bool enabled = false, reused = false, retained = false;
    uint64_t calls_this_request = 0, retained_estimated_bytes = 0;
};
struct EncoderWeightResidencyMetrics {
    bool enabled = false, reused = false, retained = false;
    uint64_t source_bytes = 0, retained_bytes = 0, loads_session_total = 0;
    std::string decline_reason;
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
    // Z-Image hybrid streaming: compact GPU suffix, prepared once per stream.
    int mlp_prefix_channels = 0;
    uint64_t suffix_pack_bytes = 0;
    uint64_t request_pack_read_bytes = 0, request_pack_write_bytes = 0;
    double request_pack_seconds = 0;
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
struct QuantizedExecutionMetrics {
    std::string source_sha256, layout_digest, decode_backend;
    uint64_t packed_bytes = 0, packed_capacity_bytes = 0, source_float_bytes = 0;
    uint64_t dense_capacity_bytes = 0, managed_peak_bytes = 0, fills = 0, decoded_bytes = 0;
    double source_load_seconds = 0, decode_seconds = 0, exposed_wait_seconds = 0;
    uint32_t slots = 0, prefetch = 0;
    std::string source_residency;
    uint64_t source_logical_bytes = 0, read_buffer_bytes = 0, source_read_bytes = 0;
    double streamed_read_seconds = 0;
    uint64_t refiner_fills = 0, refiner_capacity_bytes = 0, refiner_decoded_bytes = 0;
    uint32_t refiner_slots = 0;
    std::string precision_profile;
    uint64_t gpu_affine_preparations=0,gpu_affine_output_bytes=0,gpu_prepare_capacity_upper=0;
    uint64_t allocator_cache_limit_bytes=0;
    double gpu_prepare_seconds=0;
    uint64_t gpu_fixed_output_banks=0,gpu_fixed_output_bank_bytes=0;
    std::string source_metadata_policy;
    bool source_metadata_reused=false;
    uint64_t source_metadata_preparations=0,conditioning_producer_generation=0;
};
struct QuantizedSourceComparison {
    std::string name;
    uint32_t step=0;
    Float32Comparison metrics;
    bool final_latent=false;
};
struct SharedLoraRankMetrics {
    bool enabled = false;
    uint64_t prepared_sets = 0;
    uint64_t completed_hybrid_blocks = 0;
    uint64_t completed_adapter_rank_arrays = 0;
};
struct RunResult {
    bool prepared = false, warmup = false, prompt_cache_hit = false;
    std::string selection, backend, precision, checkpoint;
    std::string original_prompt, enhanced_prompt, enhanced_wh_ratio;
    std::string enhanced_ratio_follow;
    double prompt_enhance_seconds = 0;
    int prompt_enhance_tokens = 0;
    bool prompt_enhance_chunked_prefill = false;
    Request request;
    ExecutionPlan plan;
    int text_tokens = 0, valid_text_tokens = 0, total_tokens = 0, reference_tokens = 0,
        actual_steps = 0;
    size_t lora_applied_projections = 0;
    bool db_cache_enabled = false;
    float db_cache_threshold = 0.f;
    int db_cache_steps = 0, db_cache_max_consecutive = 0;
    Timings timings;
    uint64_t active_bytes = 0, peak_bytes = 0;
    std::optional<HybridMetrics> hybrid;
    std::optional<QkvMetrics> qkv;
    std::optional<HybridMetrics> encoder_hybrid;
    std::optional<EncoderRuntimeReuseMetrics> encoder_runtime_reuse;
    std::optional<EncoderWeightResidencyMetrics> encoder_weight_residency;
    std::optional<SharedLoraRankMetrics> shared_lora_ranks;
    std::optional<BlockResidencyMetrics> block_residency;
    std::optional<StreamingRuntimeMetrics> streaming_runtime;
    std::vector<StreamingStageRuntimeMetrics> streaming_stages;
    std::vector<StreamingBoundaryRuntimeMetrics> streaming_boundaries;
    std::shared_ptr<const streaming::ActualExecutionReceipt>
        streaming_receipt;
    std::optional<PublicStreamingSelectionMetrics> public_streaming;
    std::optional<MemoryAdmissionMetrics> memory_admission;
    std::optional<QuantizedExecutionMetrics> quantized_execution;
    std::optional<QuantizedExecutionMetrics> encoder_quantized_execution;
    std::optional<streaming::GgufPackedBankMetrics> gguf_import;
    std::vector<QuantizedSourceComparison> quantized_source_comparisons;
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
    // An unsafe streaming drain requires process lifetime retention. API
    // callers must not unload, reuse or free this session's visible storage.
    virtual bool streaming_quarantined() const noexcept { return false; }
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
    // Explicit CPU/file-I/O verification; never performed implicitly by options.
    virtual std::shared_ptr<const streaming::SourceLease>
    verify_streaming_sources(std::atomic<bool> &) {
        throw std::runtime_error("streaming_artifact_verification_unsupported");
    }
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
    virtual void test_set_streaming_drain_failure(bool) {
        throw std::runtime_error("streaming drain fault unavailable");
    }
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
    // Optional model-owned validation for private/manual streaming routes.
    // Absent means GPU-only. Public selectors retain their separate validator.
    std::function<void(const Request &)> validate_manual_streaming_execution = {};
};
const ModelModule &module_for(const std::string &);
std::vector<ModelDescriptor> describe_modules();
Recipe model_recipe(const std::string &);
void validate_recipe(const Recipe &);
std::vector<float> flux_sigmas(int, int);

} // namespace tc

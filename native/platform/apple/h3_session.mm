#include "bridge.hpp"
#include "platform.hpp"
#include "../../runtime/lora_identity.hpp"
#include "../../runtime/memory_execution.hpp"
#include "memory_probe.hpp"
#include "../../runtime/residency.hpp"
#include "../../models/h3_runtime/h3.h"
#include "../../models/h3_runtime/h3_runtime_config.h"
#include "../../models/h3_runtime/h3_streaming_descriptor.hpp"
#include "../../models/h3_runtime/h3_tokenizer.h"
#include "../../runtime/streaming/actual_receipt.hpp"
#include "../../runtime/streaming/canonical_encoding.hpp"
#include "../../runtime/streaming/resolved_request.hpp"
#include "../../runtime/streaming/source_lease.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace {
struct H3MemoryBridge {
    tc::MemoryAdmission* admission = nullptr;
    tc::MemoryExecutionContext* context = nullptr;
    uint64_t allocator_domain = 0;
    uint64_t generation = 0;
    std::atomic<bool> completion_failure{false};
};

struct H3MemoryToken {
    std::optional<tc::MemoryReservation> reservation;
    std::optional<tc::StorageLease> lease;
    uint64_t allocator_domain = 0;
    uint64_t generation = 0;
    std::optional<tc::MemoryCompletionToken> completion;
};

static tc::MemoryClass h3_memory_class(h3_gpu_memory_class value) {
    switch (value) {
    case H3_GPU_MEMORY_WEIGHTS: return tc::MemoryClass::Weights;
    case H3_GPU_MEMORY_ACTIVATION: return tc::MemoryClass::Activation;
    case H3_GPU_MEMORY_CONDITIONING: return tc::MemoryClass::Conditioning;
    case H3_GPU_MEMORY_REFILL_SLOT: return tc::MemoryClass::RefillSlot;
    case H3_GPU_MEMORY_CONVERSION_SCRATCH:
        return tc::MemoryClass::ConversionScratch;
    case H3_GPU_MEMORY_OUTPUT: return tc::MemoryClass::Output;
    case H3_GPU_MEMORY_UNKNOWN: break;
    }
    return tc::MemoryClass::UnknownExternal;
}

static tc::MemoryClass h3_host_class(h3_host_memory_class value) {
    switch (value) {
    case H3_HOST_MEMORY_CONDITIONING: return tc::MemoryClass::Conditioning;
    case H3_HOST_MEMORY_LATENT: return tc::MemoryClass::Activation;
    case H3_HOST_MEMORY_DECODED_F32: return tc::MemoryClass::Output;
    case H3_HOST_MEMORY_OUTPUT: return tc::MemoryClass::Output;
    case H3_HOST_MEMORY_STAGING: return tc::MemoryClass::ConversionScratch;
    case H3_HOST_MEMORY_CONTROL: return tc::MemoryClass::CompileTemporary;
    case H3_HOST_MEMORY_UNKNOWN: break;
    }
    return tc::MemoryClass::UnknownExternal;
}

static void h3_write_memory_error(char* error, size_t error_size,
                                  const std::string& message) {
    if (error && error_size)
        std::snprintf(error, error_size, "%s", message.c_str());
}

static int h3_memory_reserve_common(void* opaque, tc::MemoryClass memory_class,
                                    uint64_t upper_bytes, const char* tag,
                                    void** token, char* error,
                                    size_t error_size) {
    if (token) *token = nullptr;
    auto* bridge = static_cast<H3MemoryBridge*>(opaque);
    if (!bridge || !bridge->admission || !bridge->context ||
        &bridge->context->admission() != bridge->admission ||
        !bridge->allocator_domain ||
        !bridge->generation || !token || !upper_bytes) {
        h3_write_memory_error(
            error, error_size,
            "memory_policy_invalid: invalid H3 memory reservation");
        return 0;
    }
    try {
        const char* site_id = tag ? tag : "h3_gpu_tensor";
        auto reservation = bridge->context->try_reserve_site(
            site_id, memory_class, upper_bytes);
        if (!reservation) {
            h3_write_memory_error(
                error, error_size,
                "memory_budget_too_small: H3 buffer reservation denied");
            return 0;
        }
        auto* value = new (std::nothrow) H3MemoryToken;
        if (!value) {
            reservation->cancel();
            h3_write_memory_error(
                error, error_size,
                "memory_policy_allocation_failed: H3 reservation token");
            return 0;
        }
        value->reservation.emplace(std::move(*reservation));
        value->allocator_domain = bridge->allocator_domain;
        value->generation = bridge->generation;
        *token = value;
        return 1;
    } catch (const std::exception& exception) {
        h3_write_memory_error(error, error_size, exception.what());
        return 0;
    } catch (...) {
        h3_write_memory_error(
            error, error_size,
            "memory_lifetime_violation: H3 reserve callback");
        return 0;
    }
}

static int h3_memory_reserve(void* opaque, uint32_t memory_class,
                             uint64_t upper_bytes, const char* tag,
                             void** token, char* error, size_t error_size) {
    return h3_memory_reserve_common(
        opaque, h3_memory_class(static_cast<h3_gpu_memory_class>(memory_class)),
        upper_bytes, tag, token, error, error_size);
}

static int h3_memory_commit(void* opaque, void* opaque_token,
                            uint64_t allocator_domain, uint64_t handle,
                            uint64_t actual_bytes, uint64_t generation,
                            char* error, size_t error_size) {
    auto* bridge = static_cast<H3MemoryBridge*>(opaque);
    auto* token = static_cast<H3MemoryToken*>(opaque_token);
    if (!bridge || !token || !token->reservation || !handle ||
        !actual_bytes || allocator_domain != token->allocator_domain ||
        generation != token->generation ||
        allocator_domain != bridge->allocator_domain ||
        generation != bridge->generation || !bridge->admission) {
        h3_write_memory_error(
            error, error_size,
            "memory_lifetime_violation: invalid or stale H3 commit token");
        return 0;
    }
    try {
        token->lease.emplace(token->reservation->commit(tc::StorageId{
            allocator_domain, handle, actual_bytes, generation}));
        token->reservation.reset();
        return 1;
    } catch (const std::exception& exception) {
        h3_write_memory_error(error, error_size, exception.what());
        return 0;
    } catch (...) {
        h3_write_memory_error(
            error, error_size,
            "memory_lifetime_violation: H3 commit callback");
        return 0;
    }
}

static void h3_memory_cancel(void*, void* opaque_token) {
    delete static_cast<H3MemoryToken*>(opaque_token);
}

static void h3_memory_release(void*, void* opaque_token) {
    auto* token = static_cast<H3MemoryToken*>(opaque_token);
    if (!token) return;
    token->lease.reset();
    delete token;
}

static int h3_memory_retire(void* opaque, void* opaque_token,
                            uint32_t stage_id, uint32_t slot_id,
                            char* error, size_t error_size) {
    auto* bridge = static_cast<H3MemoryBridge*>(opaque);
    auto* token = static_cast<H3MemoryToken*>(opaque_token);
    if (!bridge || !bridge->context || !bridge->admission ||
        &bridge->context->admission() != bridge->admission || !token ||
        !token->lease || token->completion ||
        token->allocator_domain != bridge->allocator_domain ||
        token->generation != bridge->generation) {
        h3_write_memory_error(
            error, error_size,
            "memory_lifetime_violation: invalid H3 retirement token");
        return 0;
    }
    try {
        token->completion.emplace(bridge->context->scheduler().retire_async(
            std::move(*token->lease), token->allocator_domain,
            token->generation, stage_id, slot_id));
        token->lease.reset();
        return 1;
    } catch (const std::exception& exception) {
        h3_write_memory_error(error, error_size, exception.what());
        return 0;
    } catch (...) {
        h3_write_memory_error(
            error, error_size,
            "memory_lifetime_violation: H3 retirement callback");
        return 0;
    }
}

static void h3_memory_complete(void* opaque, void* opaque_token, int status) {
    auto* bridge = static_cast<H3MemoryBridge*>(opaque);
    auto* token = static_cast<H3MemoryToken*>(opaque_token);
    if (!bridge || !bridge->context || !token || !token->completion ||
        token->allocator_domain != bridge->allocator_domain ||
        token->generation != bridge->generation ||
        !bridge->context->scheduler().post_completion(
            *token->completion, status)) {
        if (bridge)
            bridge->completion_failure.store(true,
                                             std::memory_order_release);
        return;
    }
    token->completion.reset();
}

static int h3_host_memory_reserve(void* opaque, uint32_t memory_class,
                                  uint64_t upper_bytes, const char* tag,
                                  void** token, char* error,
                                  size_t error_size) {
    return h3_memory_reserve_common(
        opaque, h3_host_class(static_cast<h3_host_memory_class>(memory_class)),
        upper_bytes, tag ? tag : "h3_host_memory", token, error, error_size);
}

static int h3_host_memory_commit(void* opaque, void* token,
                                 uint64_t allocator_domain, uint64_t handle,
                                 uint64_t actual_bytes, uint64_t generation,
                                 char* error, size_t error_size) {
    return h3_memory_commit(opaque, token, allocator_domain, handle,
                             actual_bytes, generation, error, error_size);
}

static void h3_host_memory_cancel(void* opaque, void* token) {
    h3_memory_cancel(opaque, token);
}

static void h3_host_memory_release(void* opaque, void* token) {
    h3_memory_release(opaque, token);
}

static uint64_t h3_allocator_domain(const void* owner) {
    constexpr uint64_t salt = UINT64_C(0x48334449544d4554);
    const uint64_t value = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(owner)) ^ salt;
    return value ? value : salt;
}

static std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.good()) return {};
    CC_SHA256_CTX context;
    if (CC_SHA256_Init(&context) != 1) return {};
    std::array<char, 1 << 20> buffer{};
    while (stream.good()) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        std::streamsize count = stream.gcount();
        if (count > 0 && CC_SHA256_Update(&context, buffer.data(),
                                          static_cast<CC_LONG>(count)) != 1)
            return {};
    }
    if (!stream.eof()) return {};
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    if (CC_SHA256_Final(digest, &context) != 1) return {};
    char output[CC_SHA256_DIGEST_LENGTH * 2 + 1];
    for (size_t index = 0; index < CC_SHA256_DIGEST_LENGTH; index++)
        snprintf(output + index * 2, 3, "%02x", digest[index]);
    return output;
}

constexpr const char *kH3PublicImplementation =
    "generic_stage_executor_v3";
constexpr const char *kH3PublicComponentPolicy =
    "h3-components-v1";
constexpr const char *kH3PublicKernelRevision =
    "h3-metal-dense-block-v1";

static void h3_append_public_tree(
        const std::filesystem::path& model_root,
        const std::filesystem::path& relative_root,
        std::vector<tc::streaming::SourceFileIdentity>& files) {
    const auto directory = model_root / relative_root;
    std::error_code error;
    tc::require(std::filesystem::is_directory(directory, error) && !error,
                "streaming_route_unsupported: H3 source directory is missing: " +
                    relative_root.generic_string());
    std::vector<std::filesystem::path> paths;
    std::filesystem::recursive_directory_iterator iterator(
        directory,
        std::filesystem::directory_options::skip_permission_denied, error);
    const std::filesystem::recursive_directory_iterator end;
    for (; !error && iterator != end; iterator.increment(error)) {
        std::error_code status_error;
        if (iterator->is_regular_file(status_error) && !status_error)
            paths.push_back(iterator->path());
        else
            tc::require(!status_error,
                        "streaming_source_identity: cannot inspect H3 artifact");
    }
    tc::require(!error && !paths.empty(),
                "streaming_source_identity: H3 artifact closure is empty");
    std::sort(paths.begin(), paths.end());
    for (const auto& path : paths) {
        const auto relative = std::filesystem::relative(
            path, model_root, error);
        tc::require(!error && !relative.empty(),
                    "streaming_source_identity: invalid H3 artifact path");
        tc::streaming::SourceFileIdentity file;
        file.logical_id = relative.generic_string();
        file.path = path;
        files.push_back(std::move(file));
    }
}

static std::vector<std::string> h3_public_transformer_logical_ids(
        const tc::streaming::SourceLease& lease) {
    std::vector<std::string> result;
    for (const auto& file : lease.descriptor().files) {
        if (file.logical_id.starts_with("FL2VA/transformer/") &&
            file.logical_id.ends_with(".safetensors"))
            result.push_back(file.logical_id);
    }
    tc::require(result.size() == 13,
                "streaming_route_unsupported: H3 Turbo requires 13 transformer shards");
    return result;
}

static tc::streaming::PresetSourceIdentity h3_public_source_identity(
        const tc::streaming::SourceLease& lease,
        const std::string& manifest_digest) {
    tc::streaming::CanonicalEncoder encoder(
        "h3-public-artifact-manifest-v1");
    encoder.string_field("merge_manifest_sha256", manifest_digest);
    encoder.string_field("lease_digest", lease.digest());
    encoder.unsigned_field("artifact_count", lease.file_count());
    return {"minimax-h3-turbo-original-bf16",
            "modelscope-bf16-sharded", encoder.sha256(),
            std::string(lease.digest())};
}

static tc::streaming::PresetRuntimeIdentity h3_public_runtime_identity() {
    return {"turbocider-streaming-2026-09-18",
            "public-streaming-runtime-v2",
            "h3-turbo-public-adapter-v1",
            "h3-pread-bf16-source-lease-v2",
            kH3PublicKernelRevision,
            "h3-request-cache-disabled-v1"};
}

static std::string h3_public_feature_digest(
        const tc::Request& request, uint32_t text_rows,
        std::string_view manifest_digest) {
    tc::streaming::CanonicalEncoder encoder(
        "h3-public-workload-features-v1");
    encoder.boolean_field("inputs_empty", request.inputs.empty());
    encoder.boolean_field("loras_empty", request.loras.empty());
    encoder.boolean_field("audio_disabled", !request.audio);
    encoder.boolean_field("ane_disabled", request.ane_manifest.empty());
    encoder.boolean_field(
        "encoder_ane_disabled", request.encoder_ane_manifest.empty());
    encoder.boolean_field("compile_gpu", request.compile_gpu);
    encoder.boolean_field("dynamic_text", request.dynamic_text);
    encoder.unsigned_field("text_rows", text_rows);
    encoder.string_field("merge_manifest_sha256", manifest_digest);
    return encoder.sha256();
}

// Published only inside the process-wide inference lease. Worker threads may
// read this immutable table; neither private ANE flags nor shell environment
// can silently alter a request's math or placement.
std::map<std::string,std::string> h3_configuration;
struct ConfigurationLease {
    explicit ConfigurationLease(const tc::Request& r) {
        h3_configuration.clear();
        if(!r.audio)h3_configuration.emplace("H3_OUTPUT_SILENT","1");
        if(r.memory_constrained.enabled)
            h3_configuration.emplace("H3_MEMORY_CONSTRAINED","1");
        if(r.execution=="gpu_ane"&&!r.ane_manifest.empty()) {
            auto path=std::filesystem::path(r.ane_manifest);
            if(std::filesystem::is_directory(path))h3_configuration.emplace("H3_COREML_ANE_DIR",path.string());
            else h3_configuration.emplace("H3_COREML_ANE_MANIFEST",path.string());
            h3_configuration.emplace("H3_COREML_ANE_BLOCKS","0-49");
            h3_configuration.emplace("H3_COREML_ANE_INTERMEDIATE","4096");
        }
    }
    ~ConfigurationLease(){h3_configuration.clear();}
};
struct Progress {
    const tc::Event& event;
    std::atomic<bool>& cancelled;
    std::exception_ptr failure;
    static int receive(const char *phase,int current,int total,void *opaque) noexcept {
        auto& p=*static_cast<Progress*>(opaque);
        try {
            std::string name=phase?phase:"inference";
            if(name=="FFmpeg")name="export";
            p.event(name,current,total);
            return name=="export"&&current==total?0:int(p.cancelled.load());
        }catch(...){p.failure=std::current_exception();return 1;}
    }
};

static int h3_exact_cancel_query(const void* opaque) {
    const auto* cancelled = static_cast<const std::atomic<bool>*>(opaque);
    return cancelled && cancelled->load(std::memory_order_acquire) ? 1 : 0;
}

static const tc::StreamingStageConfig* h3_exact_streaming_stage(
        const tc::Request& request) {
    if (!request.streaming.active()) return nullptr;
    const auto stage = request.streaming.stages.find("denoiser");
    return stage == request.streaming.stages.end() ? nullptr : &stage->second;
}

static tc::StreamingRuntimeMetrics h3_streaming_runtime_metrics(
        const h3_result& result, const tc::Request& request,
        bool exact_streaming) {
    tc::StreamingRuntimeMetrics metrics;
    metrics.implementation = exact_streaming ?
        "generic-stage-executor-v3" : "h3-legacy-pthread-stream-v1";
    metrics.stage = "denoiser";
    metrics.resident_prefix_blocks = static_cast<uint32_t>(
        result.ssd_pinned_blocks);
    metrics.block_group_size = 1;
    metrics.slot_count = 2;
    metrics.prefetch_distance = 1;
    metrics.io_workers = 1;
    metrics.group_count = static_cast<uint32_t>(
        result.ssd_streamed_blocks);
    metrics.pass_count = static_cast<uint32_t>(request.steps);
    metrics.startup_policy = "prefetch_window_before_prefix";
    metrics.pass_transition = "carry_first_group";
    metrics.retention = "request";
    metrics.reader_revision = 1;
    metrics.weight_format = "bf16-safetensors";
    metrics.kernel_revision = "h3-metal-dense-block-v1";
    metrics.conditioning_recipe = "h3-refined-text-adaln-v1";
    metrics.upsample_boundary = "dit-output-before-video-vae";
    return metrics;
}

class H3Session final:public tc::ModelSession {
    std::filesystem::path root_;
    bool allow_experimental_streaming_ = false;
    std::filesystem::path loaded_root_;
    std::string loaded_context_identity_;
    std::unique_ptr<h3_ctx,decltype(&h3_free)> context_{nullptr,h3_free};
    std::filesystem::path verified_lora_path_;
    std::filesystem::file_time_type verified_lora_mtime_{};
    uintmax_t verified_lora_bytes_=0;
    std::string verified_lora_sha256_;
    H3MemoryBridge memory_bridge_;
    h3_gpu_memory_hooks memory_hooks_{};
    h3_host_memory_hooks host_memory_hooks_{};
    tc_memory_schedule_hooks_v1 memory_schedule_hooks_{};
    uint64_t memory_generation_=0;
    uint64_t exact_generation_=0;
    tc::MemoryExecutionContext* memory_context_=nullptr;
    mutable tc::MemoryCheckpointHashCache memory_probe_hash_cache_;
    std::shared_ptr<const tc::streaming::SourceLease> public_stream_lease_;
    std::vector<tc::streaming::OwnedSourceFd> public_stream_descriptors_;
    std::vector<h3_weight_source_v1> public_stream_sources_;
    std::string public_stream_layout_digest_;
    uint64_t public_stream_target_bytes_ = 0;
    bool public_streaming_active_ = false;
public:
    explicit H3Session(const std::filesystem::path& root,
                       bool allow_experimental_streaming = false)
        : root_(root), allow_experimental_streaming_(allow_experimental_streaming) {
        tc::require(std::filesystem::is_directory(root),"H3 model directory missing: "+root.string());
    }
    bool uses_parent_mlx() const override { return false; }
    bool uses_parent_mlx(const tc::Request&) const override { return false; }
    std::optional<tc::MemoryCapabilityProbe> probe_memory_capability(
            const tc::ExecutionPlan& plan,
            const tc::MemoryDeviceIdentity& device) const override {
        if (!plan.memory_policy || !plan.memory_policy->enabled)
            return std::nullopt;
        tc::require(plan.memory_policy->adapter_candidate ==
                        "h3_c_metal_streamed_v1",
                    "memory_policy_unsupported: H3 probe adapter mismatch");
        const auto sidecar = tc::memory_capability_probe_path(
            root_, plan.memory_policy->adapter_candidate, plan.request,
            plan.memory_policy->refill_slots);
        return tc::load_memory_capability_probe(
            root_, sidecar, plan, device,
            tc::MemoryModelRootTrust::ExternalMutable,
            memory_probe_hash_cache_);
    }
    void unload() override { context_.reset(); loaded_root_.clear(); loaded_context_identity_.clear(); }
    std::shared_ptr<const tc::streaming::ModelStreamingProbe>
    probe_public_streaming(
            const tc::streaming::PublicResolveInput& input) const override {
        const auto& request = input.request;
        tc::require(request.model == "minimax-h3-turbo",
                    "streaming_engine_model_mismatch");
        tc::require(request.operation == "video.generate" &&
                        request.execution == "gpu" && request.inputs.empty() &&
                        request.frames >= 22 && request.frames <= 362 &&
                        (request.frames - 5) % 17 == 0 && request.fps == 24 &&
                        !request.audio && request.ane_manifest.empty() &&
                        request.encoder_ane_manifest.empty() &&
                        request.loras.empty() && request.quantized_cache.empty() &&
                        !request.allow_approximation && !request.compile_gpu,
                    "streaming_route_unsupported: H3 public card requires "
                    "original BF16 GPU text-to-video without audio/input/LoRA/ANE");
        tc::require(request.width >= 32 && request.height >= 32 &&
                        request.width % 32 == 0 && request.height % 32 == 0 &&
                        static_cast<int64_t>(request.width) * request.height <=
                            768ll * 1344ll && request.steps == 4,
                    "streaming_workload_invalid: H3 public canvas/steps are unsupported");

        const auto component = root_ / "FL2VA";
        const auto transformer = component / "transformer";
        const auto manifest = transformer / "h3-turbo-merge-manifest.json";
        std::error_code error;
        tc::require(std::filesystem::is_regular_file(manifest, error) && !error,
                    "streaming_route_unsupported: H3 provenance manifest is missing");
        const auto manifest_digest = sha256_file(manifest);
        tc::require(!manifest_digest.empty(),
                    "streaming_source_identity: cannot hash H3 provenance manifest");
        std::vector<tc::streaming::SourceFileIdentity> files;
        h3_append_public_tree(root_, "FL2VA/transformer", files);
        h3_append_public_tree(root_, "FL2VA/text_encoder", files);
        h3_append_public_tree(root_, "FL2VA/video_vae/source", files);
        h3_append_public_tree(root_, "FL2VA/audio_vae", files);
        tc::streaming::SourceFileIdentity tokenizer;
        tokenizer.logical_id = "FL2VA/tokenizer/tokenizer.json";
        tokenizer.path = component / "tokenizer/tokenizer.json";
        files.push_back(std::move(tokenizer));
        auto lease = tc::streaming::SourceLease::capture(std::move(files));
        const auto transformer_ids = h3_public_transformer_logical_ids(*lease);

        const auto& tokenizer_file = lease->file("FL2VA/tokenizer/tokenizer.json");
        auto tokenizer_fd = lease->duplicate_fd(tokenizer_file.logical_id);
        char detail[512] = {};
        std::unique_ptr<h3_tokenizer, decltype(&h3_tokenizer_free)> tokenizer_value(
            h3_tokenizer_load_fd(tokenizer_file.path.c_str(), tokenizer_fd.get(),
                                 detail, sizeof(detail)), h3_tokenizer_free);
        tc::require(tokenizer_value != nullptr,
                    detail[0] ? detail : "cannot load H3 tokenizer from lease");
        uint32_t* token_ids = nullptr;
        size_t token_count = 0;
        tc::require(h3_tokenizer_encode(tokenizer_value.get(), request.prompt.c_str(),
                                        1, &token_ids, &token_count,
                                        detail, sizeof(detail)) != 0 &&
                        token_count > 0 && token_count <= UINT32_MAX,
                    detail[0] ? detail : "cannot tokenize H3 public prompt");
        h3_tokenizer_ids_free(token_ids);

        tc::h3::StreamingMetadata metadata(lease, transformer_ids);
        tc::h3::StreamingWorkload workload;
        workload.width = static_cast<uint32_t>(request.width);
        workload.height = static_cast<uint32_t>(request.height);
        workload.frames = static_cast<uint32_t>(request.frames);
        workload.fps = static_cast<uint32_t>(request.fps);
        workload.text_rows = static_cast<uint32_t>(token_count);
        workload.steps = static_cast<uint32_t>(request.steps);
        workload.active_blocks = H3_DIT_BLOCKS;
        workload.audio = false;
        auto descriptor = metadata.describe(workload);
        tc::streaming::PresetWorkload identity;
        identity.model = request.model;
        identity.operation = request.operation;
        identity.execution = request.execution;
        identity.device_class = input.device.device_class;
        identity.execution_container = input.execution_container;
        identity.width = static_cast<uint32_t>(request.width);
        identity.height = static_cast<uint32_t>(request.height);
        identity.frames = static_cast<uint32_t>(request.frames);
        identity.fps = static_cast<uint32_t>(request.fps);
        identity.steps = static_cast<uint32_t>(request.steps);
        identity.batch = 1;
        identity.audio = false;
        identity.dynamic_text = request.dynamic_text;
        identity.approximation = false;
        identity.conditioning_revision = "h3-turbo-text-adaln-v1";
        identity.vae_policy_revision = "h3-video-vae-v1";
        identity.feature_digest = h3_public_feature_digest(
            request, static_cast<uint32_t>(token_count), manifest_digest);
        identity.token_shapes.push_back({
            "h3-bpe", "h3-tokenizer-v1", "h3-template-v1",
            static_cast<uint32_t>(token_count),
            static_cast<uint32_t>(token_count),
            static_cast<uint32_t>(token_count)});
        (void)descriptor;
        return std::make_shared<tc::streaming::ValueModelStreamingProbe>(
            tc::streaming::ValueModelStreamingProbe::Values{
                request.model,
                h3_public_source_identity(*lease, manifest_digest),
                std::move(identity), h3_public_runtime_identity(),
                kH3PublicComponentPolicy, std::move(lease)});
    }

    std::shared_ptr<const tc::streaming::ModelStreamingSnapshot>
    compile_public_streaming(
            std::shared_ptr<const tc::streaming::ModelStreamingProbe> probe,
            const tc::streaming::StreamingPresetRecord& record) const override {
        auto value_probe = std::dynamic_pointer_cast<
            const tc::streaming::ValueModelStreamingProbe>(probe);
        tc::require(value_probe != nullptr,
                    "streaming_public_probe_type_mismatch");
        tc::require(value_probe->model_id() == "minimax-h3-turbo" &&
                        value_probe->component_policy_revision() ==
                            record.plan.component_policy_revision &&
                        record.source == value_probe->source_identity() &&
                        record.workload == value_probe->workload_identity() &&
                        record.runtime == value_probe->runtime_identity(),
                    "streaming_record_identity_mismatch");
        const auto& workload = value_probe->workload_identity();
        tc::require(workload.token_shapes.size() == 1,
                    "streaming_workload_invalid: H3 token shape count");
        std::vector<std::string> transformer_ids;
        for (const auto& file : value_probe->lease().descriptor().files)
            if (file.logical_id.starts_with("FL2VA/transformer/") &&
                file.logical_id.ends_with(".safetensors"))
                transformer_ids.push_back(file.logical_id);
        tc::h3::StreamingWorkload descriptor_workload;
        descriptor_workload.width = workload.width;
        descriptor_workload.height = workload.height;
        descriptor_workload.frames = workload.frames;
        descriptor_workload.fps = workload.fps;
        descriptor_workload.text_rows = workload.token_shapes.front().padded_rows;
        descriptor_workload.steps = workload.steps;
        descriptor_workload.active_blocks = H3_DIT_BLOCKS;
        descriptor_workload.audio = false;
        auto plan = std::make_shared<tc::h3::StreamingPlanView>(
            value_probe->lease_ptr(), transformer_ids,
            record.plan.canonical_config, descriptor_workload, 1);
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
        if (!record.plan.layout_digest.empty())
#endif
            tc::require(plan->layout().digest == record.plan.layout_digest,
                        "streaming_layout_digest_mismatch");
        return std::make_shared<tc::streaming::ValueModelStreamingSnapshot>(
            tc::streaming::ValueModelStreamingSnapshot::Values{
                "minimax-h3-turbo", value_probe->source_identity(),
                value_probe->runtime_identity(), plan->descriptor(),
                plan->layout(),
                std::string(value_probe->component_policy_revision()),
                value_probe->lease_ptr()});
    }

    tc::RunResult generate_resolved(
            std::shared_ptr<const tc::streaming::ResolvedRequestExecution> execution,
            const tc::Event& event, std::atomic<bool>& cancelled) override {
        tc::require(execution && execution->probe && execution->model_snapshot,
                    "streaming_authority_mismatch");
        tc::require(execution->request.streaming.active(),
                    "streaming_actual_plan_mismatch");
        auto value_probe = std::dynamic_pointer_cast<
            const tc::streaming::ValueModelStreamingProbe>(execution->probe);
        tc::require(value_probe != nullptr,
                    "streaming_public_probe_type_mismatch");
        auto lease = value_probe->lease_ptr();
        tc::require(lease && execution->probe->source_lease() == lease.get() &&
                        execution->model_snapshot->source_lease() == lease.get(),
                    "streaming_source_lease_mismatch");
        tc::require(execution->selection.record.plan.layout_digest ==
                        execution->model_snapshot->layout().digest &&
                        execution->selection.record.source ==
                            execution->model_snapshot->source_identity(),
                    "streaming_authority_mismatch");
        const auto target = execution->selection.exact_selector
                                .target_request_memory_bytes;
        tc::require(target && tc::streaming::supported_streaming_target(*target),
                    "streaming_target_unsupported");
        tc::require(!public_streaming_active_,
                    "streaming_public_request_reentrant");
        public_stream_lease_ = std::move(lease);
        public_stream_target_bytes_ = *target;
        public_stream_layout_digest_ =
            execution->model_snapshot->layout().digest;
        public_stream_descriptors_.clear();
        public_stream_sources_.clear();
        for (const auto& file : public_stream_lease_->descriptor().files) {
            if (!file.logical_id.starts_with("FL2VA/transformer/") ||
                !file.logical_id.ends_with(".safetensors"))
                continue;
            public_stream_descriptors_.push_back(
                public_stream_lease_->duplicate_fd(file.logical_id));
            public_stream_sources_.push_back({
                file.path.c_str(), public_stream_descriptors_.back().get()});
        }
        public_streaming_active_ = true;
        try {
            auto result = generate(execution->request, event, cancelled);
            public_streaming_active_ = false;
            public_stream_target_bytes_ = 0;
            public_stream_layout_digest_.clear();
            public_stream_sources_.clear();
            public_stream_descriptors_.clear();
            public_stream_lease_.reset();
            return result;
        } catch (...) {
            public_streaming_active_ = false;
            public_stream_target_bytes_ = 0;
            public_stream_layout_digest_.clear();
            public_stream_sources_.clear();
            public_stream_descriptors_.clear();
            public_stream_lease_.reset();
            throw;
        }
    }
    void set_memory_admission(tc::MemoryAdmission* admission) override {
        memory_bridge_.admission = admission;
        if (!admission) return;
        ++memory_generation_;
        if (!memory_generation_) ++memory_generation_;
        memory_bridge_.allocator_domain = h3_allocator_domain(this);
        memory_bridge_.generation = memory_generation_;
        memory_bridge_.completion_failure.store(
            false, std::memory_order_release);
    }
    void bind_memory_context(tc::MemoryExecutionContext* context) override {
        tc::require(context != nullptr && memory_context_ == nullptr &&
                        memory_bridge_.admission == &context->admission(),
                    "memory_lifetime_violation: invalid H3 context binding");
        memory_context_ = context;
        memory_bridge_.context = context;
        memory_schedule_hooks_ = context->make_schedule_hooks();
    }
    void unbind_memory_context() noexcept override {
        memory_schedule_hooks_ = {};
        memory_bridge_.context = nullptr;
        memory_context_ = nullptr;
    }
    tc::MemoryDrainResult drain_memory_completions(
            tc::MemoryExecutionContext& context,
            std::chrono::milliseconds timeout) override {
        tc::require(memory_context_ == &context &&
                        memory_bridge_.admission == &context.admission(),
                    "memory_lifetime_violation: invalid H3 completion drain");
        tc::require(timeout.count() > 0,
                    "memory_policy_invalid: H3 completion drain timeout is zero");
        char error[512] = {};
        h3_drain_info native{};
        if (context_ && !h3_drain(
                context_.get(), &native, error, sizeof(error))) {
            return {false, 0,
                    static_cast<uint64_t>(context.scheduler().pending_count()),
                    error[0] ? error : "H3 native GPU drain failed"};
        }
        const auto mailbox = context.drain_completion_mailbox();
        const auto snapshot = context.admission().snapshot();
        const auto pending = std::max<uint64_t>(
            snapshot.pending_release_count, context.scheduler().pending_count());
        const bool callback_failed = memory_bridge_.completion_failure.load(
            std::memory_order_acquire);
        const bool completed = mailbox.ok() && !callback_failed && pending == 0;
        std::string failure;
        if (!mailbox.ok())
            failure = mailbox.failure.empty() ?
                "H3 completion mailbox drain failed" : mailbox.failure;
        else if (callback_failed)
            failure = "H3 completion callback rejected a memory token";
        else if (pending)
            failure = "H3 generate returned with pending GPU memory";
        return {completed,
                native.components_drained + mailbox.consumed,
                pending, std::move(failure)};
    }
    tc::RunResult generate(const tc::Request& r,const tc::Event& event,std::atomic<bool>& cancel) override {
        using namespace tc;
        require(r.model=="minimax-h3-turbo","request model differs from H3 session");
        if (r.memory_constrained.enabled)
            require(memory_bridge_.admission != nullptr &&
                        memory_context_ != nullptr &&
                        memory_bridge_.admission ==
                            &memory_context_->admission() &&
                        memory_bridge_.allocator_domain != 0 &&
                        memory_bridge_.generation != 0,
                    "memory_lifetime_violation: constrained H3 request has "
                    "no active memory admission");
        if (r.memory_constrained.enabled) {
            require(r.operation == "video.generate" && !r.audio &&
                        r.inputs.empty() && r.loras.empty() &&
                        r.quantized_cache.empty() && r.execution == "gpu" &&
                        r.residency == "streamed",
                    "memory_policy_unsupported: constrained H3 v1 supports "
                    "only GPU streamed text-to-video without audio, inputs, "
                    "runtime LoRA, or quantized cache");
        }
        auto plan=make_plan(r);
        const auto* exact_stage = h3_exact_streaming_stage(r);
        const bool exact_streaming = exact_stage &&
            exact_stage->residency &&
            *exact_stage->residency == "streamed";
        if (r.streaming.active()) {
            require(allow_experimental_streaming_ || public_streaming_active_,
                    "streaming_layout_not_certified: H3 exact adapter is "
                    "restricted to the private candidate constructor");
            require(exact_streaming && r.streaming.stages.size() == 1 &&
                        exact_stage->block_group_size == 1u &&
                        exact_stage->slot_count == 2u &&
                        exact_stage->resident_prefix_blocks &&
                        *exact_stage->resident_prefix_blocks <= 48u &&
                        exact_stage->prefetch_distance == 1u &&
                        exact_stage->io_workers == 1u &&
                        r.streaming.retention == "request" &&
                        r.operation == "video.generate" && !r.audio &&
                        r.inputs.empty() && r.loras.empty() &&
                        r.quantized_cache.empty() && r.execution == "gpu" &&
                        !r.allow_approximation &&
                        !r.memory_constrained.enabled,
                    "streaming_route_unsupported: H3 exact candidate requires "
                    "GPU text-to-video, original BF16, G1/K2/D1/Q1, request "
                    "retention, no approximation, and no bounded-memory guard");
        }
        require(!r.prompt.empty()&&!r.output.empty(),"prompt and output are required");
        require(std::filesystem::path(r.output).extension()==".mp4","H3 output must be .mp4");
        ConfigurationLease config(r);checkpoint(cancel);auto start=Clock::now();
        // The four-step schedule must belong to the actual selected component.
        // Ref2VA requires its own distilled artifact; FL2VA provenance cannot
        // authorize running an unmodified Ref2VA checkpoint at four steps.
        auto component=r.operation=="video.reference"?"Ref2VA":"FL2VA";
        std::filesystem::path active_root = root_;
        if (!r.loras.empty()) {
            require(std::filesystem::is_regular_file(
                        root_/component/"transformer/h3-turbo-merge-manifest.json"),
                    "H3 LoRA requires an offline-premerged model with a provenance manifest; "
                    "the app does not perform disk fusion");
        }
        const auto manifest_path = active_root / component /
            "transformer/h3-turbo-merge-manifest.json";
        auto manifest=read_json(manifest_path);
        require([manifest[@"schema"] isEqual:@"h3-turbo-merge-manifest-v2"],"H3 Turbo requires a merge provenance manifest");
        NSDictionary *identity=manifest[@"identity"];
        NSDictionary *mapping=manifest[@"mapping"],*source=manifest[@"source"],*shards=manifest[@"shards"];
        require([identity isKindOfClass:NSDictionary.class]&&[mapping isKindOfClass:NSDictionary.class]&&[source isKindOfClass:NSDictionary.class]&&[shards isKindOfClass:NSDictionary.class],"invalid H3 checkpoint provenance");
        require(string_value(identity,@"profile")=="lightx2v-4step"&&[identity[@"raw_tensors"] integerValue]==624&&[identity[@"raw_pairs"] integerValue]==312&&[identity[@"mapped_weights"] integerValue]==208&&string_value(source,@"repository")=="lightx2v/Minimax-h3-Turbo"&&[mapping[@"raw_tensors"] integerValue]==624&&[mapping[@"raw_pairs"] integerValue]==312&&[mapping[@"mapped_weights"] integerValue]==208&&shards.count==13,"untrusted H3 Turbo merge manifest");
        auto variant=string_value(identity,@"variant");
        auto revision=string_value(identity,@"source_revision"),source_revision=string_value(source,@"revision");
        auto lora_sha=string_value(identity,@"lora_sha256"),artifact=string_value(source,@"artifact");
        double strength=[identity[@"strength"] doubleValue];
        bool v01=(variant.empty()||variant=="v0.1-544p")&&revision=="050494d5fe05bd1b1140b8565ea51dc33a5085a5"&&source_revision==revision&&lora_sha=="5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29"&&[identity[@"lora_bytes"] unsignedLongLongValue]==1383677888ULL&&std::abs(strength-0.0625)<=1e-12&&(!identity[@"video_flow_shift"]||std::abs([identity[@"video_flow_shift"] doubleValue]-12.0)<=1e-12)&&(!identity[@"audio_flow_shift"]||std::abs([identity[@"audio_flow_shift"] doubleValue]-3.0)<=1e-12)&&(artifact.empty()||artifact=="v0.1-544p");
        bool v11=variant=="v1.1-768p"&&revision=="2f8ea0dc0a7e2b26c9a43124eb89673787189b4e"&&source_revision==revision&&lora_sha=="b5e25a59292d51bca3fc02b9a0b2284e11b4eb20921a9c5adc2db785956b8966"&&[identity[@"lora_bytes"] unsignedLongLongValue]==1383677808ULL&&std::abs(strength-1.0)<=1e-12&&std::abs([identity[@"video_flow_shift"] doubleValue]-6.0)<=1e-12&&std::abs([identity[@"audio_flow_shift"] doubleValue]-3.0)<=1e-12&&artifact=="v1.1-768p";
        require(v01||v11,"unsupported or untrusted H3 Turbo checkpoint variant");
        std::optional<VerifiedLoRAIdentity> requested_lora_identity;
        if(!r.loras.empty()) {
            const auto& requested=r.loras.front();
            auto lora_path=std::filesystem::path(requested.path);
            require(std::filesystem::is_regular_file(lora_path),
                    "H3 requested LoRA file is missing");
            std::error_code path_error;
            auto absolute_path=std::filesystem::absolute(lora_path,path_error);
            require(!path_error,"cannot resolve H3 requested LoRA path");
            auto lora_bytes=std::filesystem::file_size(absolute_path,path_error);
            require(!path_error,"cannot inspect H3 requested LoRA size");
            auto lora_mtime=std::filesystem::last_write_time(
                absolute_path,path_error);
            require(!path_error,"cannot inspect H3 requested LoRA timestamp");
            require(lora_bytes==[identity[@"lora_bytes"] unsignedLongLongValue],
                    "H3 requested LoRA size does not match the merge manifest");
            if(absolute_path!=verified_lora_path_||
               lora_bytes!=verified_lora_bytes_||
               lora_mtime!=verified_lora_mtime_) {
                verified_lora_sha256_=sha256_file(absolute_path);
                require(!verified_lora_sha256_.empty(),
                        "cannot hash H3 requested LoRA");
                verified_lora_path_=absolute_path;
                verified_lora_bytes_=lora_bytes;
                verified_lora_mtime_=lora_mtime;
            }
            require(verified_lora_sha256_==lora_sha,
                    "H3 requested LoRA SHA-256 does not match the merge manifest");
            require(std::abs(requested.strength-float(strength))<=1e-6f,
                    "H3 requested LoRA strength does not match the merged checkpoint");
            requested_lora_identity = make_verified_lora_identity(
                requested, absolute_path, lora_bytes,
                verified_lora_sha256_);
        }
        double video_shift=v11?6.0:12.0;
        // h3_load_dir owns a process-local prepared-DiT cache.  The model
        // root is intentionally stable across requests, so a root-only key
        // could retain FL2VA/Ref2VA or a newly replaced premerged LoRA
        // artifact after the sidecar changed.  Include the selected component
        // and the tiny provenance manifest digest; the native h3 resident key
        // separately includes the actual transformer index contents.
        const auto manifest_sha256 = sha256_file(manifest_path);
        require(!manifest_sha256.empty(),
                "cannot hash H3 checkpoint provenance manifest");
        auto context_identity = active_root.string() + ":" + component +
            ":manifest_sha256=" + manifest_sha256;
        if (requested_lora_identity)
            context_identity += ":" + requested_lora_identity->cache_key(
                "h3-premerged-manifest-v1");
        if(!context_ || loaded_context_identity_ != context_identity) {
            context_.reset();
            event("model_load",0,1);
            context_.reset(h3_load_dir(active_root.c_str()));
            require(bool(context_),"cannot load H3 model metadata");
            loaded_root_ = active_root;
            loaded_context_identity_ = context_identity;
            event("model_load",1,1);
        }
        const auto *info=h3_model(context_.get());const auto *device=h3_device(context_.get());
        uint64_t transformer_bytes=r.operation=="video.reference"?info->ref2va_transformer.tensor_bytes:info->fl2va_transformer.tensor_bytes;
        require(exact_streaming||r.residency=="streamed"||transformer_bytes<device->recommended_working_set,"H3 weights exceed this GPU working set; select streamed residency");
        require(!r.memory_budget_bytes || r.memory_budget_bytes <= device->physical_memory,
                "H3 memory budget exceeds physical memory");
        require(r.residency=="streamed" || !r.memory_budget_bytes,
                "H3 memory budget requires streamed residency");
        if (!r.quantized_cache.empty())
            require(std::filesystem::is_directory(r.quantized_cache),
                    "H3 quantized cache directory is missing: " +
                        r.quantized_cache);
        /* Streamed DiT sessions are reusable too.  The runtime's resident key
         * includes streaming mode, pinned prefix and memory budget, while a
         * failed/cancelled denoise detaches the cached DiT and reprepare
         * requires the refill cursor to be back at the first streamed block.
         * Keeping this cache enabled is what makes the measured retained-
         * session benefit observable through the TurboCider Session API. */
        h3_cache_set_enabled(context_.get(),
                             !exact_streaming &&
                             !r.memory_constrained.enabled &&
                             (r.residency == "resident" ||
                              r.residency == "streamed"));
        h3_cache_set_decoder_enabled(context_.get(),
                                     r.residency == "resident");
        h3_params parameters=H3_PARAMS_DEFAULT;
        h3_gpu_options gpu_options{};
        parameters.width=r.width;parameters.height=r.height;parameters.frames=r.frames;parameters.steps=r.steps;parameters.seed=r.seed;
        parameters.video_flow_shift=video_shift;parameters.audio_flow_shift=3;
        parameters.ssd_streaming=exact_streaming||r.residency=="streamed";
        /* A nonzero request budget lets the H3 runtime choose the largest
         * safe resident prefix after accounting for its actual activation
         * geometry and two BF16 stream slots. With no explicit budget keep the
         * original two-slot streaming behavior unchanged. */
        parameters.ssd_pinned_prefix=exact_streaming ?
            static_cast<int>(*exact_stage->resident_prefix_blocks) : 0;
        parameters.ssd_memory_budget_bytes =
            (!exact_streaming && r.residency == "streamed" &&
             r.memory_budget_bytes) ?
                r.memory_budget_bytes : 0;
        parameters.ssd_quantized_cache_directory =
            r.quantized_cache.empty() ? nullptr : r.quantized_cache.c_str();
        if (exact_streaming) {
            ++exact_generation_;
            if (!exact_generation_) ++exact_generation_;
            parameters.exact_streaming = 1;
            parameters.exact_streaming_generation = exact_generation_;
            parameters.exact_prefetch_distance =
                *exact_stage->prefetch_distance;
            parameters.exact_io_workers = *exact_stage->io_workers;
            parameters.exact_carry_first_group = 1;
            parameters.exact_cancel = h3_exact_cancel_query;
            parameters.exact_cancel_user = &cancel;
            if (public_streaming_active_) {
                require(public_stream_lease_ &&
                            public_stream_sources_.size() == 13 &&
                            public_stream_descriptors_.size() == 13 &&
                            public_stream_layout_digest_.size() == 64,
                        "streaming_source_lease_mismatch");
                parameters.exact_weight_sources =
                    public_stream_sources_.data();
                parameters.exact_weight_source_count =
                    public_stream_sources_.size();
                parameters.exact_receipt_source_generation =
                    public_stream_lease_->generation();
                parameters.exact_receipt_layout_digest =
                    public_stream_layout_digest_.c_str();
                parameters.exact_receipt_implementation =
                    kH3PublicImplementation;
            }
        }
        if (r.memory_constrained.enabled) {
            memory_hooks_.struct_size = sizeof(memory_hooks_);
            memory_hooks_.version = 1u;
            memory_hooks_.user = &memory_bridge_;
            memory_hooks_.reserve = h3_memory_reserve;
            memory_hooks_.commit = h3_memory_commit;
            memory_hooks_.cancel = h3_memory_cancel;
            memory_hooks_.release = h3_memory_release;
            memory_hooks_.retire = h3_memory_retire;
            memory_hooks_.complete = h3_memory_complete;
            gpu_options.struct_size = sizeof(gpu_options);
            gpu_options.version = 1u;
            gpu_options.memory_hooks = &memory_hooks_;
            gpu_options.memory_allocator_domain =
                memory_bridge_.allocator_domain;
            gpu_options.memory_generation = memory_bridge_.generation;
            parameters.gpu_options = &gpu_options;
            host_memory_hooks_.struct_size = sizeof(host_memory_hooks_);
            host_memory_hooks_.version = 1u;
            host_memory_hooks_.user = &memory_bridge_;
            host_memory_hooks_.reserve = h3_host_memory_reserve;
            host_memory_hooks_.commit = h3_host_memory_commit;
            host_memory_hooks_.cancel = h3_host_memory_cancel;
            host_memory_hooks_.release = h3_host_memory_release;
            parameters.host_memory_hooks = &host_memory_hooks_;
            parameters.memory_allocator_domain =
                memory_bridge_.allocator_domain;
            parameters.memory_generation = memory_bridge_.generation;
            if (memory_schedule_hooks_.emit)
                parameters.schedule_hooks = &memory_schedule_hooks_;
        }
        parameters.use_reference_rope=1;
        parameters.use_slower_bf16_mlp=!r.allow_approximation;
        parameters.use_slower_bf16_qkv=!r.allow_approximation;
        parameters.use_slower_bf16_attention_output=!r.allow_approximation;
        parameters.output_path=r.output.c_str();
        std::vector<h3_reference> references;
        for(auto& input:r.inputs) {
            require(std::filesystem::is_regular_file(input.path),"input asset missing: "+input.path);
            if(input.role=="first_frame")parameters.first_frame=input.path.c_str();
            else if(input.role=="last_frame")parameters.last_frame=input.path.c_str();
            else references.push_back({input.kind=="image"?H3_REFERENCE_IMAGE:input.kind=="audio"?H3_REFERENCE_AUDIO:input.audio_path.empty()?H3_REFERENCE_VIDEO:H3_REFERENCE_VIDEO_AUDIO,input.path.c_str(),input.audio_path.empty()?nullptr:input.audio_path.c_str(),int(input.include_audio)});
        }
        parameters.references=references.data();parameters.reference_count=references.size();
        Progress progress{event,cancel,{}};parameters.on_progress=Progress::receive;parameters.callback_opaque=&progress;
        std::unique_ptr<h3_result,decltype(&h3_result_free)> result(h3_generate(context_.get(),r.prompt.c_str(),&parameters),h3_result_free);
        if(progress.failure)std::rethrow_exception(progress.failure);
        if(!result){checkpoint(cancel);throw std::runtime_error(h3_last_error(context_.get()));}
        h3_cache_info cache_info{};
        h3_cache_get_info(context_.get(), &cache_info);
        id plan_value = to_dictionary(plan);
        if (exact_streaming) {
            const uint32_t streamed_blocks = static_cast<uint32_t>(
                result->ssd_streamed_blocks);
            const uint64_t expected_groups =
                static_cast<uint64_t>(streamed_blocks) *
                static_cast<uint64_t>(r.steps);
            require(result->exact_streaming &&
                        result->exact_streaming_finished &&
                        !result->exact_streaming_poisoned &&
                        result->exact_completed_passes ==
                            static_cast<uint32_t>(r.steps) &&
                        result->exact_pool_creates == 1u &&
                        result->exact_slot_bundles == 2u &&
                        result->exact_fills == expected_groups &&
                        result->exact_groups_submitted == expected_groups,
                    "H3 exact execution counters differ from the sealed layout");
        }
        const double request_seconds =
            std::chrono::duration<double>(Clock::now()-start).count();
        auto value = @{ @"schema_version":@1,@"model":@(r.model.c_str()),
                  @"operation":@(r.operation.c_str()),@"output":@(r.output.c_str()),
                  @"width":@(result->width),@"height":@(result->height),
                  @"frames":@(result->frames),@"fps":@(result->fps),@"audio":@(r.audio),
                  @"ssd_streaming":@(result->ssd_streaming),
                  @"ssd_quantized":@(result->ssd_quantized),
                  @"ssd_pinned_blocks":@(result->ssd_pinned_blocks),
                  @"ssd_streamed_blocks":@(result->ssd_streamed_blocks),
                  @"ssd_memory_budget_bytes":@(result->ssd_memory_budget_bytes),
                  @"ssd_block_bytes":@(result->ssd_block_bytes),
                  @"ssd_activation_reserve_bytes":@(result->ssd_activation_reserve_bytes),
                  @"ssd_bytes_read":@(result->ssd_bytes_read),
                  @"ssd_read_seconds":@(result->ssd_read_seconds),
                  @"ssd_wait_seconds":@(result->ssd_wait_seconds),
                  @"ssd_request_bytes_read":@(result->ssd_request_bytes_read),
                  @"ssd_request_read_seconds":@(result->ssd_request_read_seconds),
                  @"ssd_request_wait_seconds":@(result->ssd_request_wait_seconds),
                  @"exact_streaming":@(result->exact_streaming),
                  @"exact_streaming_finished":@(result->exact_streaming_finished),
                  @"exact_streaming_poisoned":@(result->exact_streaming_poisoned),
                  @"exact_completed_passes":@(result->exact_completed_passes),
                  @"exact_pool_creates":@(result->exact_pool_creates),
                  @"exact_slot_bundles":@(result->exact_slot_bundles),
                  @"exact_fills":@(result->exact_fills),
                  @"exact_content_bytes_loaded":@(result->exact_content_bytes_loaded),
                  @"exact_groups_submitted":@(result->exact_groups_submitted),
                  @"exact_refill_load_seconds":@(result->exact_refill_load_seconds),
                  @"exact_max_refill_seconds":@(result->exact_max_refill_seconds),
                  @"exact_max_refill_block":@(result->exact_max_refill_block),
                  @"exact_wait_seconds":@(result->exact_wait_seconds),
                  @"cache_prepared_dit":@(cache_info.prepared_dit),
                  @"cache_video_decoder":@(cache_info.video_decoder),
                  @"cache_embedding_entries":@(cache_info.embedding_entries),
                  @"plan":plan_value,
                  @"seconds":@(request_seconds),
                  @"timings_seconds":@{
                      @"request_wall":@(request_seconds),
                      @"denoise":@(result->denoise_seconds),
                  },
                  @"validation": @"native_executor_manifest_verified",
                  @"lora_fusion": r.loras.empty() ? @"none" :
                      @"sidecar_manifest_verified" };
        auto run = native_run_result(value, r, plan);
        if (result->ssd_streaming && exact_streaming) {
            const unsigned active_blocks = static_cast<unsigned>(
                result->ssd_pinned_blocks + result->ssd_streamed_blocks);
            const uint64_t resident_blocks = static_cast<uint64_t>(
                result->ssd_pinned_blocks) + 2u;
            require(!result->ssd_block_bytes ||
                        resident_blocks <=
                            (std::numeric_limits<uint64_t>::max() -
                             result->ssd_activation_reserve_bytes) /
                                result->ssd_block_bytes,
                    "H3 exact working-set estimate overflow");
            BlockResidencyMetrics metrics;
            metrics.enabled = true;
            metrics.active_blocks = active_blocks;
            metrics.pinned_blocks = static_cast<unsigned>(
                result->ssd_pinned_blocks);
            metrics.streamed_blocks = static_cast<unsigned>(
                result->ssd_streamed_blocks);
            metrics.refill_slots = 2;
            metrics.memory_budget_bytes = public_streaming_active_ ?
                public_stream_target_bytes_ : 0;
            metrics.activation_reserve_bytes =
                result->ssd_activation_reserve_bytes;
            metrics.block_bytes = result->ssd_block_bytes;
            metrics.estimated_working_set_bytes =
                result->ssd_activation_reserve_bytes +
                resident_blocks * result->ssd_block_bytes;
            metrics.request_bytes_loaded =
                result->exact_content_bytes_loaded;
            metrics.request_slot_allocations = result->exact_slot_bundles;
            metrics.request_slot_refills = result->exact_fills;
            metrics.request_slot_fills = result->exact_fills;
            metrics.request_load_seconds = result->exact_refill_load_seconds;
            metrics.request_wait_seconds = result->exact_wait_seconds;
            metrics.request_refill_load_seconds =
                result->exact_refill_load_seconds;
            metrics.request_max_refill_seconds =
                result->exact_max_refill_seconds;
            metrics.request_max_refill_block =
                result->exact_max_refill_block;
            run.block_residency = metrics;
        } else if (result->ssd_streaming) {
            const unsigned active_blocks = static_cast<unsigned>(
                result->ssd_pinned_blocks + result->ssd_streamed_blocks);
            const auto residency = make_block_residency_plan(
                result->ssd_memory_budget_bytes,
                result->ssd_activation_reserve_bytes,
                result->ssd_block_bytes, active_blocks, 0u, 2u, false, false);
            require(residency.pinned_blocks ==
                        static_cast<unsigned>(result->ssd_pinned_blocks) &&
                    residency.streamed_blocks ==
                        static_cast<unsigned>(result->ssd_streamed_blocks),
                    "H3 native and framework block residency plans differ");
            require(!r.steps || residency.streamed_blocks <=
                        std::numeric_limits<uint64_t>::max() /
                            static_cast<uint64_t>(r.steps),
                    "H3 legacy logical fill count overflow");
            const uint64_t logical_fills =
                static_cast<uint64_t>(residency.streamed_blocks) *
                static_cast<uint64_t>(r.steps);
            BlockResidencyMetrics metrics;
            metrics.enabled = true;
            metrics.quantized = result->ssd_quantized != 0;
            metrics.active_blocks = active_blocks;
            metrics.pinned_blocks = residency.pinned_blocks;
            metrics.streamed_blocks = residency.streamed_blocks;
            metrics.refill_slots = residency.refill_slots;
            metrics.memory_budget_bytes = residency.memory_budget_bytes;
            metrics.activation_reserve_bytes =
                residency.activation_reserve_bytes;
            metrics.block_bytes = residency.block_bytes;
            metrics.estimated_working_set_bytes =
                residency.estimated_working_set_bytes;
            metrics.request_bytes_loaded = result->ssd_request_bytes_read;
            metrics.request_slot_allocations = residency.refill_slots;
            metrics.request_slot_refills = logical_fills;
            metrics.request_slot_fills = logical_fills;
            metrics.request_load_seconds = result->ssd_request_read_seconds;
            metrics.request_wait_seconds = result->ssd_request_wait_seconds;
            metrics.request_refill_load_seconds =
                result->ssd_request_read_seconds;
            run.block_residency = metrics;
        }
        if (result->ssd_streaming)
            run.streaming_runtime = h3_streaming_runtime_metrics(
                *result, r, exact_streaming);
        if (public_streaming_active_) {
            require(result->exact_receipt != nullptr,
                    "streaming_actual_plan_mismatch: H3 receipt is missing");
            auto stage = tc::streaming::actual_stage_receipt_from_c_v2(
                *result->exact_receipt);
            require(stage.layout_digest == public_stream_layout_digest_ &&
                        stage.implementation == kH3PublicImplementation &&
                        stage.source_generation ==
                            public_stream_lease_->generation(),
                    "streaming_actual_plan_mismatch: H3 receipt identity differs");
            auto receipt = tc::streaming::make_actual_execution_receipt(
                kH3PublicImplementation, public_stream_layout_digest_,
                kH3PublicComponentPolicy, {std::move(stage)});
            run.streaming_receipt = std::make_shared<
                const tc::streaming::ActualExecutionReceipt>(
                    std::move(receipt));
            require(run.streaming_runtime.has_value(),
                    "streaming_actual_plan_mismatch: H3 runtime metrics missing");
            auto& runtime = *run.streaming_runtime;
            runtime.implementation = kH3PublicImplementation;
            runtime.layout_digest = public_stream_layout_digest_;
            runtime.component_policy_revision = kH3PublicComponentPolicy;
            runtime.multi_pool_policy = "serial";
            runtime.pool_count = 1;
            runtime.slot_bundle_count = 2;
            runtime.refill_worker_count = 1;
            runtime.source_lease_verified = true;
            runtime.drained = result->exact_streaming_finished != 0;
            run.streaming_stages = {{0, runtime}};
        }
        return run;
    }
};
}
extern "C" const char *h3_runtime_getenv(const char *key) {
    auto found = h3_configuration.find(key);
    return found == h3_configuration.end() ? nullptr : found->second.c_str();
}
namespace tc {
std::unique_ptr<ModelSession> create_h3(const std::filesystem::path& root) {
    return std::make_unique<H3Session>(root, false);
}
std::unique_ptr<ModelSession> create_h3_candidate(
        const std::filesystem::path& root) {
    return std::make_unique<H3Session>(root, true);
}
}

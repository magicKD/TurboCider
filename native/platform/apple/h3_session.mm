#include "bridge.hpp"
#include "platform.hpp"
#include "../../runtime/lora_identity.hpp"
#include "../../runtime/memory_execution.hpp"
#include "memory_probe.hpp"
#include "../../runtime/residency.hpp"
#include "../../models/h3_runtime/h3.h"
#include "../../models/h3_runtime/h3_runtime_config.h"
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <new>

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
class H3Session final:public tc::ModelSession {
    std::filesystem::path root_;
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
    tc::MemoryExecutionContext* memory_context_=nullptr;
    mutable tc::MemoryCheckpointHashCache memory_probe_hash_cache_;
public:
    explicit H3Session(const std::filesystem::path& root):root_(root) {
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
        auto plan=make_plan(r);require(!r.prompt.empty()&&!r.output.empty(),"prompt and output are required");
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
        require(r.residency=="streamed"||transformer_bytes<device->recommended_working_set,"H3 weights exceed this GPU working set; select streamed residency");
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
                             !r.memory_constrained.enabled &&
                             (r.residency == "resident" ||
                              r.residency == "streamed"));
        h3_cache_set_decoder_enabled(context_.get(),
                                     r.residency == "resident");
        h3_params parameters=H3_PARAMS_DEFAULT;
        h3_gpu_options gpu_options{};
        parameters.width=r.width;parameters.height=r.height;parameters.frames=r.frames;parameters.steps=r.steps;parameters.seed=r.seed;
        parameters.video_flow_shift=video_shift;parameters.audio_flow_shift=3;
        parameters.ssd_streaming=r.residency=="streamed";
        /* A nonzero request budget lets the H3 runtime choose the largest
         * safe resident prefix after accounting for its actual activation
         * geometry and two BF16 stream slots. With no explicit budget keep the
         * original two-slot streaming behavior unchanged. */
        parameters.ssd_pinned_prefix=0;
        parameters.ssd_memory_budget_bytes =
            (r.residency == "streamed" && r.memory_budget_bytes) ?
                r.memory_budget_bytes : 0;
        parameters.ssd_quantized_cache_directory =
            r.quantized_cache.empty() ? nullptr : r.quantized_cache.c_str();
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
                  @"cache_prepared_dit":@(cache_info.prepared_dit),
                  @"cache_video_decoder":@(cache_info.video_decoder),
                  @"cache_embedding_entries":@(cache_info.embedding_entries),
                  @"plan":to_dictionary(plan),
                  @"seconds":@(std::chrono::duration<double>(Clock::now()-start).count()),
                  @"validation": @"native_executor_manifest_verified",
                  @"lora_fusion": r.loras.empty() ? @"none" :
                      @"sidecar_manifest_verified" };
        auto run = native_run_result(value, r, plan);
        if (result->ssd_streaming) {
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
            run.block_residency = BlockResidencyMetrics{
                true,
                false,
                result->ssd_quantized != 0,
                active_blocks,
                residency.pinned_blocks,
                residency.streamed_blocks,
                residency.refill_slots,
                residency.memory_budget_bytes,
                residency.activation_reserve_bytes,
                residency.block_bytes,
                residency.estimated_working_set_bytes,
                result->ssd_request_bytes_read,
                0,
                0,
                result->ssd_request_read_seconds,
                result->ssd_request_wait_seconds,
            };
        }
        return run;
    }
};
}
extern "C" const char *h3_runtime_getenv(const char *key) {
    auto found = h3_configuration.find(key);
    return found == h3_configuration.end() ? nullptr : found->second.c_str();
}
namespace tc {std::unique_ptr<ModelSession> create_h3(const std::filesystem::path& root){return std::make_unique<H3Session>(root);}}

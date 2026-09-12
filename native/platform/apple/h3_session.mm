#include "bridge.hpp"
#include "platform.hpp"
#include "../../runtime/lora_identity.hpp"
#include "../../runtime/residency.hpp"
#include "../../models/h3_runtime/h3.h"
#include "../../models/h3_runtime/h3_runtime_config.h"
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>

namespace {
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
public:
    explicit H3Session(const std::filesystem::path& root):root_(root) {
        tc::require(std::filesystem::is_directory(root),"H3 model directory missing: "+root.string());
    }
    void unload() override { context_.reset(); loaded_root_.clear(); loaded_context_identity_.clear(); }
    tc::RunResult generate(const tc::Request& r,const tc::Event& event,std::atomic<bool>& cancel) override {
        using namespace tc;
        require(r.model=="minimax-h3-turbo","request model differs from H3 session");
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
                             r.residency == "resident" ||
                             r.residency == "streamed");
        h3_cache_set_decoder_enabled(context_.get(),
                                     r.residency == "resident");
        h3_params parameters=H3_PARAMS_DEFAULT;
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

#include "runtime.hpp"
#include "h3_runtime/h3.h"
#include "h3_runtime/h3_runtime_config.h"
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
    NSDictionary *generate(const tc::Request& r,const tc::Event& event,std::atomic<bool>& cancel) override {
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
        tc::RuntimeLoRACache runtime_cache;
        bool used_runtime_cache = false;
        if (!r.loras.empty()) {
            bool sidecar_matches = false;
            try {
                auto sidecar = root_/component/"transformer/h3-turbo-merge-manifest.json";
                if (std::filesystem::is_regular_file(sidecar)) {
                    auto candidate = read_json(sidecar);
                    id identity = candidate[@"identity"];
                    auto requested = std::filesystem::absolute(r.loras.front().path);
                    sidecar_matches = [identity isKindOfClass:NSDictionary.class] &&
                        std::filesystem::is_regular_file(requested) &&
                        std::abs([identity[@"strength"] doubleValue] -
                                 r.loras.front().strength) <= 1e-6 &&
                        sha256_file(requested) == string_value(identity, @"lora_sha256") &&
                        [identity[@"lora_bytes"] unsignedLongLongValue] ==
                            std::filesystem::file_size(requested);
                }
            } catch (...) {
                sidecar_matches = false;
            }
            if (!sidecar_matches) {
                std::filesystem::path base;
                if (const char* configured = std::getenv("TURBOCIDER_H3_LORA_BASE"))
                    if (*configured) base = configured;
                auto installed_transformer = root_/component/"transformer";
                if (base.empty() && !std::filesystem::is_regular_file(
                        installed_transformer/"h3-turbo-merge-manifest.json"))
                    base = installed_transformer;
                require(!base.empty(),
                        "H3 model root already contains a different merged adapter; "
                        "set TURBOCIDER_H3_LORA_BASE to the unmerged "
                        "COMPONENT/transformer directory");
                runtime_cache = ensure_runtime_lora_cache(
                    "h3", base, r.loras.front(),
                    "lightx2v-4step");
                active_root = runtime_cache.artifact;
                used_runtime_cache = true;
            }
        }
        auto manifest=read_json(active_root/component/"transformer/h3-turbo-merge-manifest.json");
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
        }
        double video_shift=v11?6.0:12.0;
        const auto context_identity = active_root.string() + ":" +
            (used_runtime_cache ? runtime_cache.cache_key : "installed");
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
        h3_cache_set_enabled(context_.get(),r.residency=="resident");
        h3_params parameters=H3_PARAMS_DEFAULT;
        parameters.width=r.width;parameters.height=r.height;parameters.frames=r.frames;parameters.steps=r.steps;parameters.seed=r.seed;
        parameters.video_flow_shift=video_shift;parameters.audio_flow_shift=3;
        parameters.ssd_streaming=r.residency=="streamed";
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
        return @{ @"schema_version":@1,@"model":@(r.model.c_str()),
                  @"operation":@(r.operation.c_str()),@"output":@(r.output.c_str()),
                  @"width":@(result->width),@"height":@(result->height),
                  @"frames":@(result->frames),@"fps":@(result->fps),@"audio":@(r.audio),
                  @"plan":plan,
                  @"seconds":@(std::chrono::duration<double>(Clock::now()-start).count()),
                  @"validation": used_runtime_cache ? @"native_executor_runtime_lora_cache_verified" : @"native_executor_manifest_verified",
                  @"lora_fusion": r.loras.empty() ? @"none" :
                      (used_runtime_cache ? @"runtime_bake_cache" : @"sidecar_manifest_verified"),
                  @"lora_cache_key": runtime_cache.cache_key.empty() ?
                      (id)[NSNull null] : @(runtime_cache.cache_key.c_str()),
                  @"lora_cache_hit": @(runtime_cache.cache_hit) };
    }
};
}
extern "C" const char *h3_runtime_getenv(const char *key) {auto found=h3_configuration.find(key);return found==h3_configuration.end()?nullptr:found->second.c_str();}
namespace tc {std::unique_ptr<ModelSession> create_h3(const std::filesystem::path& root){return std::make_unique<H3Session>(root);}}

#include "runtime.hpp"
#include "h3/vendor/h3.h"
#include "h3/vendor/h3_runtime_config.h"
#include <exception>

namespace {
// Published only inside the process-wide inference lease. Worker threads may
// read this immutable table; neither private ANE flags nor shell environment
// can silently alter a request's math or placement.
std::map<std::string,std::string> h3_configuration;
struct ConfigurationLease {
    explicit ConfigurationLease(const tc::Request& r) {
        h3_configuration.clear();
        if(!r.audio)h3_configuration.emplace("H3_OUTPUT_SILENT","1");
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
    std::unique_ptr<h3_ctx,decltype(&h3_free)> context_{nullptr,h3_free};
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
        auto manifest=read_json(root_/component/"transformer/h3-turbo-merge-manifest.json");
        require([manifest[@"schema"] isEqual:@"h3-turbo-merge-manifest-v2"],"H3 Turbo requires a merge provenance manifest");
        NSDictionary *identity=manifest[@"identity"];
        require([identity isKindOfClass:NSDictionary.class],"invalid H3 checkpoint identity");
        auto variant=string_value(identity,@"variant");
        double video_shift=0;
        if(variant=="v1.1-768p") {
            require([identity[@"strength"] doubleValue]==1.0&&[identity[@"video_flow_shift"] doubleValue]==6.0,"v1.1 requires strength 1 and video shift 6");video_shift=6;
        } else if(variant=="v0.1-544p") {
            require([identity[@"strength"] doubleValue]==.0625,"v0.1 requires strength 0.0625");video_shift=12;
        } else throw std::invalid_argument("unsupported H3 Turbo checkpoint variant");
        if(!context_) {event("model_load",0,1);context_.reset(h3_load_dir(root_.c_str()));require(bool(context_),"cannot load H3 model metadata");event("model_load",1,1);}
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
        return @{@"schema_version":@1,@"model":@(r.model.c_str()),@"operation":@(r.operation.c_str()),@"output":@(r.output.c_str()),@"width":@(result->width),@"height":@(result->height),@"frames":@(result->frames),@"fps":@(result->fps),@"audio":@(r.audio),@"plan":plan,@"seconds":@(std::chrono::duration<double>(Clock::now()-start).count()),@"validation":@"native_executor_weights_pending"};
    }
};
}
extern "C" const char *h3_runtime_getenv(const char *key) {auto found=h3_configuration.find(key);return found==h3_configuration.end()?nullptr:found->second.c_str();}
namespace tc {std::unique_ptr<ModelSession> create_h3(const std::filesystem::path& root){return std::make_unique<H3Session>(root);}}

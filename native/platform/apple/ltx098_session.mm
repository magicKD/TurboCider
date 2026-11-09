#include "bridge.hpp"
#include "../../core/unigram_tokenizer.hpp"
#include "../../media/image.hpp"
#include "../../media/video.hpp"
#include "../../models/ltx098/pipeline.hpp"

namespace tc {
namespace {

class LTX098Session final : public ModelSession {
    std::filesystem::path root_;
    std::unique_ptr<UnigramTokenizer> tokenizer_;
    std::unique_ptr<ltx098::Pipeline> pipeline_;

  public:
    explicit LTX098Session(const std::filesystem::path &root) : root_(std::filesystem::absolute(root)) {
        for (const auto *path : {"ltxv-2b-0.9.8-distilled.safetensors", "text_encoders/t5xxl_fp16.safetensors", "tokenizer/tokenizer.json"})
            require(std::filesystem::is_regular_file(root_ / path), "LTX 0.9.8 model component missing: " + std::string(path));
    }

    void unload() override { pipeline_.reset(); tokenizer_.reset(); }

    RunResult generate(const Request &request, const Event &event, std::atomic<bool> &cancelled) override {
        require(request.model == "ltx-video-0.9.8-distilled", "request model differs from LTX 0.9.8 session");
        const auto plan = make_plan(request);
        require(!request.output.empty() && std::filesystem::path(request.output).extension() == ".mp4",
                "LTX 0.9.8 output must be an .mp4 file");
        require(std::filesystem::is_regular_file(request.inputs[0].path), "LTX 0.9.8 first-frame image is missing");
        const auto destination = std::filesystem::absolute(request.output);
        require(!std::filesystem::exists(destination), "LTX 0.9.8 output already exists");
        const auto staging = destination.parent_path() /
            (destination.filename().string() + ".tmp." + NSUUID.UUID.UUIDString.UTF8String + ".mp4");
        const auto started = Clock::now();
        checkpoint(cancelled);
        try {
            if (!tokenizer_) tokenizer_ = std::make_unique<UnigramTokenizer>(root_ / "tokenizer", UnigramFlavor::t5_v1_1);
            if (!pipeline_) pipeline_ = std::make_unique<ltx098::Pipeline>(root_);
            auto tokens = tokenizer_->t5_prompt(request.prompt);
            auto image = load_image_center_crop_tensor(request.inputs[0].path, request.width, request.height);
            mx::eval(image);
            ltx098::GenerateOptions options;
            options.width = request.width; options.height = request.height; options.frames = request.frames;
            options.fps = request.fps; options.seed = request.seed; options.strength = request.inputs[0].strength; options.steps = request.steps;
            options.resize_to_base = request.ltx098_resize_to_base;
            auto [pixels, metrics] = pipeline_->generate(tokens, image, options, event, cancelled);
            require(mx::all(mx::isfinite(pixels)).item<bool>(), "LTX 0.9.8 decoder produced nonfinite pixels");
            auto scaled = (mx::astype(pixels, mx::float32) + Tensor(1.f, mx::float32)) * Tensor(127.5f, mx::float32);
            auto rgb = mx::astype(mx::contiguous(mx::clip(scaled, Tensor(0.f), Tensor(255.f))), mx::uint8);
            mx::eval(rgb);
            checkpoint(cancelled);
            event("export", 0, 1);
            write_video_rgb24(staging, rgb.data<uint8_t>(), request.frames, metrics.output_width, metrics.output_height, request.fps);
            const auto info = probe_video(staging);
            require(info.frames == request.frames && info.width == metrics.output_width && info.height == metrics.output_height && info.fps == request.fps,
                    "LTX 0.9.8 encoded video failed validation");
            checkpoint(cancelled);
            require(!std::filesystem::exists(destination), "LTX 0.9.8 output appeared during generation");
            std::error_code error;
            std::filesystem::rename(staging, destination, error);
            require(!error, "cannot publish LTX 0.9.8 output: " + error.message());
            event("export", 1, 1);
            const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
            NSDictionary *result = @{
                @"schema_version":@1, @"model":@(request.model.c_str()), @"operation":@"video.image",
                @"output":@(destination.c_str()), @"width":@(info.width), @"height":@(info.height),
                @"requested_base_width":@(request.width), @"requested_base_height":@(request.height),
                @"stage1_width":@(metrics.stage1_width), @"stage1_height":@(metrics.stage1_height),
                @"stage2_width":@(metrics.stage2_width), @"stage2_height":@(metrics.stage2_height),
                @"workflow":@(request.steps == 10 ? "multiscale-7+3" : "single-scale-8"),
                @"ltx098_resize_to_base":@(options.resize_to_base),
                @"ltx098_adain_after_upscale":@(request.steps == 10),
                @"frames":@(request.frames), @"fps":@(request.fps), @"seed":@(request.seed),
                @"steps":@(request.steps), @"actual_denoise_steps":@(metrics.denoise_steps),
                @"execution":@"gpu", @"runtime_backend":@"ltx098-mlx-metal", @"runtime_dependency":@"native",
                @"runtime_precision":@"bf16-dit-vae+t5-bf16-weights-fp32-activations+fp32-scheduler", @"persistent_model":@YES,
                @"prompt_cache_hit":@(metrics.text_cache_hit), @"text_tokens":@(metrics.text_tokens),
                @"cross_attention_cache_hit":@(metrics.cross_attention_cache_hit),
                @"valid_text_tokens":@(metrics.valid_text_tokens), @"first_frame_strength":@(options.strength),
                @"seconds":@(seconds), @"mlx_active_bytes":@(mx::get_active_memory()), @"mlx_peak_bytes":@(mx::get_peak_memory()),
                @"timings_seconds":@{@"request_wall":@(seconds), @"text":@(metrics.text_seconds),
                    @"image":@(metrics.image_seconds), @"transformer_load":@(metrics.transformer_load_seconds),
                    @"cross_attention_prepare":@(metrics.cross_attention_prepare_seconds),
                    @"denoise":@(metrics.denoise_seconds), @"latent_upscale":@(metrics.upscale_seconds), @"decoder_load":@(metrics.decoder_load_seconds),
                    @"decode":@(metrics.decode_seconds)}
            };
            return native_run_result(result, request, plan);
        } catch (...) {
            std::error_code ignored;
            std::filesystem::remove(staging, ignored);
            unload();
            throw;
        }
    }
};
} // namespace

std::unique_ptr<ModelSession> create_ltx098(const std::filesystem::path &root) {
    return std::make_unique<LTX098Session>(root);
}
} // namespace tc

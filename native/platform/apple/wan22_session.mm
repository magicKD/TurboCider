#include "bridge.hpp"
#include "platform.hpp"
#include "../../core/unigram_tokenizer.hpp"
#include "../../media/image.hpp"
#include "../../media/video.hpp"
#include "../../models/wan22/pipeline.hpp"
#include "../../models/wan22/text.hpp"
#include "../../models/wan22/identifiers.hpp"

#include <cerrno>
#include <cstring>
#include <stdio.h>
#include <mlx/version.h>

namespace tc {
namespace {

class Wan22Session final : public ModelSession {
    std::filesystem::path root_;
    std::string model_id_;
    bool quanhaol_ = false;
    std::unique_ptr<UnigramTokenizer> tokenizer_;
    std::optional<Tensor> cached_text_;
    std::vector<int> cached_ids_;
    int cached_valid_ = 0;

  public:
    explicit Wan22Session(const std::filesystem::path &root, bool quanhaol)
        : root_(std::filesystem::absolute(root)),
          model_id_(quanhaol ? wan22::quanhaol_id : wan22::fastwan_id), quanhaol_(quanhaol) {
        const char *transformer = quanhaol_ ? "diffusion_models/quanhaol_wan22_ti2v_5b_turbo_fp16.safetensors" :
                                              "diffusion_models/fastwan2.2_ti2v_5b_fullattn.safetensors";
        for (const auto *path : {transformer, "text_encoders/umt5_xxl_fp16.safetensors", "vae/wan2.2_vae.safetensors",
                                 "tokenizer/tokenizer.json"})
            require(std::filesystem::is_regular_file(root_ / path), "Wan2.2 component missing: " + std::string(path));
    }

    void unload() override {
        cached_text_.reset();
        cached_ids_.clear();
        cached_valid_ = 0;
        tokenizer_.reset();
    }

    RunResult generate(const Request &request, const Event &event, std::atomic<bool> &cancelled) override {
        require(request.model == model_id_, "request model differs from Wan2.2 session");
        const auto plan = make_plan(request);
        require(!request.output.empty() && std::filesystem::path(request.output).extension() == ".mp4",
                "Wan2.2 output must be an .mp4 file");
        require(std::filesystem::is_regular_file(request.inputs[0].path), "Wan2.2 first-frame image is missing");
        const auto destination = std::filesystem::absolute(request.output);
        require(std::filesystem::is_directory(destination.parent_path()) && !std::filesystem::exists(destination) &&
                    !std::filesystem::is_symlink(destination), "Wan2.2 output directory is invalid or destination already exists");
        const auto staging = destination.parent_path() /
            (destination.filename().string() + ".tmp." + NSUUID.UUID.UUIDString.UTF8String + ".mp4");
        const auto started = Clock::now();
        checkpoint(cancelled);
        try {
            if (!tokenizer_) tokenizer_ = std::make_unique<UnigramTokenizer>(root_ / "tokenizer");
            auto tokens = tokenizer_->prompt_extended(request.prompt);
            const auto device = device_info();
            const auto os = NSProcessInfo.processInfo.operatingSystemVersion;
            const int text_alignment = wan22::text_padding_alignment(device.gpu, device.physical_memory,
                int(os.majorVersion), int(os.minorVersion), int(os.patchVersion), mx::version(), tokens.valid, tokens.ids.size());
            const auto text_started = Clock::now();
            const bool text_cache_hit = cached_text_ && cached_ids_ == tokens.ids && cached_valid_ == tokens.valid;
            if (!text_cache_hit) {
                wan22::TextEncoder encoder(root_ / "text_encoders/umt5_xxl_fp16.safetensors", root_ / "tokenizer");
                auto text = encoder.encode(tokens, event, cancelled, text_alignment);
                mx::eval(text);
                checkpoint(cancelled);
                cached_text_ = std::move(text);
                cached_ids_ = tokens.ids;
                cached_valid_ = tokens.valid;
                mx::clear_cache();
            } else event("umt5_prompt_cache", 1, 1);
            const double text_seconds = std::chrono::duration<double>(Clock::now() - text_started).count();
            auto image = load_image_center_crop_tensor(request.inputs[0].path, request.width, request.height);
            image = mx::astype(mx::expand_dims(mx::transpose(image, {0, 3, 1, 2}), 2), mx::bfloat16);
            mx::eval(image);
            const char *transformer = quanhaol_ ? "diffusion_models/quanhaol_wan22_ti2v_5b_turbo_fp16.safetensors" :
                                                  "diffusion_models/fastwan2.2_ti2v_5b_fullattn.safetensors";
            wan22::Pipeline pipeline(root_ / transformer,
                                     root_ / "vae/wan2.2_vae.safetensors");
            wan22::GenerateOptions options;
            options.frames = request.frames;
            options.seed = request.seed;
            if (quanhaol_) options.distillation = wan22::Distillation::quanhaol_4;
            // Keep the previous low-memory path on smaller systems.
            options.cache_cross_attention = device.physical_memory >= (64ull << 30);
            const bool tiny_decoder = request.tiny_vae && request.tiny_vae->decoder == "tiny";
            if (tiny_decoder) options.tiny_decoder_checkpoint = request.tiny_vae->path;
            auto result = pipeline.generate(image, *cached_text_, options, event, cancelled);
            require(result.pixels.shape() == mx::Shape{1, 3, request.frames, request.height, request.width} &&
                        mx::all(mx::isfinite(result.pixels)).item<bool>(), "Wan2.2 decoder produced invalid pixels");
            auto rgb = mx::squeeze(mx::transpose(mx::astype(result.pixels, mx::float32), {0, 2, 3, 4, 1}), 0);
            rgb = mx::astype(mx::round(mx::clip((rgb + 1.f) * 127.5f, Tensor(0.f), Tensor(255.f))), mx::uint8);
            rgb = mx::contiguous(rgb);
            mx::eval(rgb);
            checkpoint(cancelled);
            event("export", 0, 1);
            write_video_rgb24(staging, rgb.data<uint8_t>(), request.frames, request.width, request.height, request.fps);
            const auto info = probe_video(staging);
            require(info.frames == request.frames && info.width == request.width && info.height == request.height &&
                        info.fps == request.fps, "Wan2.2 encoded video failed validation");
            checkpoint(cancelled);
            require(::renamex_np(staging.c_str(), destination.c_str(), RENAME_EXCL) == 0,
                    "cannot publish Wan2.2 output without overwriting: " + std::string(std::strerror(errno)));
            event("export", 1, 1);
            const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
            NSDictionary *metadata = @{
                @"schema_version":@1, @"model":@(request.model.c_str()), @"operation":@"video.image",
                @"output":@(destination.c_str()), @"width":@(info.width), @"height":@(info.height),
                @"frames":@(info.frames), @"fps":@(info.fps), @"seed":@(request.seed), @"steps":@(quanhaol_ ? 4 : 3),
                @"actual_denoise_steps":@(quanhaol_ ? 4 : 3), @"first_frame_strength":@1.0,
                @"scheduler":quanhaol_ ? @"quanhaol-dmd-shift8-bf16" : @"fastvideo-mlx-dmd-shift5-warped", @"execution":@"gpu",
                @"runtime_backend":@"wan22-mlx-metal", @"runtime_dependency":@"native",
                @"runtime_precision":quanhaol_ ? (tiny_decoder ? @"mixed-source-bf16-dit-vae-encoder+fp16-taew22-super+umt5-bf16-weights-fp32-activations+bf16-renoise" : @"mixed-source-bf16-dit-vae+umt5-bf16-weights-fp32-activations+bf16-renoise") :
                    (tiny_decoder ? @"bf16-dit-vae-encoder+fp16-taew22-super+umt5-bf16-weights-fp32-activations+fp32-scheduler" : @"bf16-dit-vae+umt5-bf16-weights-fp32-activations+fp32-scheduler"),
                @"video_decoder":tiny_decoder ? @"taew2_2_super" : @"wan2.2_vae",
                @"stage_seconds":@{@"first_frame_vae_load_and_encode":@(result.first_frame_vae_seconds),
                                    @"transformer_load_and_denoise":@(result.transformer_and_denoise_seconds),
                                    @"video_vae_load_and_decode":@(result.video_vae_seconds)},
                @"persistent_model":@NO, @"prompt_cache_hit":@(text_cache_hit), @"valid_text_tokens":@(tokens.valid),
                @"cross_attention_cache":@(options.cache_cross_attention),
                @"text_padding_alignment":@(text_alignment),
                @"text_encoder_sequence_tokens":@(text_cache_hit ? 0 : (text_alignment ? 256 : tokens.ids.size())),
                @"text_encode_seconds":@(text_seconds),
                @"text_sequence_tokens":@(tokens.ids.size()), @"text_context_limit":@1024,
                @"seconds":@(seconds), @"mlx_active_bytes":@(mx::get_active_memory()), @"mlx_peak_bytes":@(mx::get_peak_memory())
            };
            return native_run_result(metadata, request, plan);
        } catch (...) {
            std::error_code ignored;
            std::filesystem::remove(staging, ignored);
            unload();
            throw;
        }
    }
};
} // namespace

std::unique_ptr<ModelSession> create_wan22(const std::filesystem::path &root, bool quanhaol) {
    return std::make_unique<Wan22Session>(root, quanhaol);
}
} // namespace tc

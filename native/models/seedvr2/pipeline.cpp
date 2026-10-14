#include "pipeline.hpp"
#include "input_lifetime.hpp"

namespace tc::seedvr2 {
namespace {
double seconds(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}
Tensor time_slice(const Tensor &x, int start, int stop) {
    return mx::slice(x, {0, 0, start, 0, 0}, {x.shape(0), x.shape(1), stop, x.shape(3), x.shape(4)});
}
} // namespace

void Pipeline::load(const std::filesystem::path &root, const Event &event, std::atomic<bool> &cancelled,
                    mx::Dtype dit_dtype) {
    if (loaded_) return;
    checkpoint(cancelled);
    try {
        dit_path_ = std::filesystem::absolute(root / "diffusion_models/seedvr2_3b_fp16.safetensors");
        dit_dtype_ = dit_dtype;
        if (stage_dit_) {
            dit_source_bytes_ = std::filesystem::file_size(dit_path_);
            dit_source_modified_ = std::filesystem::last_write_time(dit_path_);
        }
        if (event) event("seedvr2_load", 0, 2);
        vae_.load(root / "vae/seedvr2_ema_vae_fp16.safetensors");
        checkpoint(cancelled);
        if (event) event("seedvr2_load", 1, 2);
        dit_.load(dit_path_,dit_dtype);
        stable_profile_ = dit_.stable_fp16_profile();
        checkpoint(cancelled);
        if (event) event("seedvr2_load", 2, 2);
        loaded_ = true;
    } catch (...) {
        clear();
        throw;
    }
}
Restoration Pipeline::restore(const Tensor &pixels, const Tensor &noise, const Event &event,
                              std::atomic<bool> &cancelled, const std::filesystem::path &dump, const FFNRoute &ffn) {
    return restore_impl(pixels,noise,event,cancelled,dump,ffn,nullptr);
}
Restoration Pipeline::restore_owned(Tensor pixels, const Tensor &noise, const Event &event,
                                    std::atomic<bool> &cancelled, const std::filesystem::path &dump, const FFNRoute &ffn,
                                    const DecodedFrameSink &sink) {
    require(pixels.dtype() == mx::float32, "SeedVR2 owned restoration requires FP32 normalized RGB");
    require(!sink || dump.empty(), "streaming restoration cannot retain complete decoded pixel dumps");
    return restore_impl(pixels,noise,event,cancelled,dump,ffn,&pixels,sink);
}
Restoration Pipeline::restore_impl(const Tensor &pixels, const Tensor &noise, const Event &event,
                                  std::atomic<bool> &cancelled, const std::filesystem::path &dump,
                                  const FFNRoute &ffn, Tensor *input_owner, const DecodedFrameSink &sink) {
    require(loaded_, "SeedVR2 pipeline is not loaded");
    require(pixels.ndim() == 5 && pixels.shape(0) == 1 && pixels.shape(1) == 3 &&
            pixels.shape(3) % 16 == 0 && pixels.shape(4) % 16 == 0,
            "SeedVR2 requires resized batch-one RGB BCTHW pixels and dimensions divisible by sixteen");
    const int original_frames = pixels.shape(2), frames = padded_frames(original_frames);
    auto input = pixels;
    if (frames != original_frames) input = mx::concatenate({pixels,
        mx::repeat(time_slice(pixels, original_frames - 1, original_frames), frames - original_frames, 2)}, 2);
    require(noise.shape() == mx::Shape{1, 16, (frames - 1) / 4 + 1, pixels.shape(3) / 8, pixels.shape(4) / 8} &&
            noise.dtype() == mx::float32, "SeedVR2 requires matching FP32 sampling noise");
    const auto restore_started = Clock::now();
    double staging_seconds = 0;
    auto release_dit = [&] {
        const auto started = Clock::now();
        mx::synchronize();
        dit_.clear();
        mx::clear_cache();
        staging_seconds += seconds(started,Clock::now());
    };
    if (stage_dit_ && dit_.bytes()) release_dit();
    const auto started = Clock::now();
    auto condition = vae_.encode(input, event, cancelled);
    const auto encoded = Clock::now();
    double input_release_seconds = 0;
    uint64_t released_input_logical_bytes = 0;
    if (input_owner) {
        const auto releasing = Clock::now();
        released_input_logical_bytes = release_encoded_input(input,*input_owner,condition);
        input_release_seconds = seconds(releasing,Clock::now());
    }
    if (stage_dit_) {
        checkpoint(cancelled);
        const auto reloading = Clock::now();
        auto source_unchanged = [&] {
            require(std::filesystem::is_regular_file(dit_path_) &&
                    std::filesystem::file_size(dit_path_) == dit_source_bytes_ &&
                    std::filesystem::last_write_time(dit_path_) == dit_source_modified_,
                    "SeedVR2 DiT checkpoint changed during staged loading");
        };
        source_unchanged();
        if (event) event("seedvr2_dit_stage_load",0,1);
        dit_.load(dit_path_,dit_dtype_);
        require(!stable_profile_ || dit_.stable_fp16_profile(),
                "SeedVR2 DiT staged reload changed the qualified stable precision profile");
        source_unchanged();
        checkpoint(cancelled);
        if (event) event("seedvr2_dit_stage_load",1,1);
        staging_seconds += seconds(reloading,Clock::now());
    }
    const auto denoise_started = Clock::now();
    auto velocity = dit_.forward(noise, condition, 1000.f, event, cancelled, dump, ffn);
    // Flow/Euler schedule [1,0], CFG=1, condition-noise scale=0. Model inputs
    // follow the selected projection recipe; Euler retains FP32 noise.
    auto latent = noise - mx::astype(velocity, mx::float32);
    mx::eval(latent);
    const auto denoised = Clock::now();
    if (stage_dit_) release_dit();
    const auto decode_started = Clock::now();
    Tensor restored(0);
    if (sink) vae_.decode_stream(latent, original_frames, sink, event, cancelled);
    else {
        restored = vae_.decode(latent, event, cancelled);
        restored = time_slice(restored, 0, original_frames);
        mx::eval(restored);
    }
    const auto decoded = Clock::now();
    checkpoint(cancelled);
    return {restored, condition, velocity, latent, seconds(started, encoded),
            seconds(denoise_started, denoised), seconds(decode_started, decoded),
            staging_seconds,seconds(restore_started,decoded),input_release_seconds,released_input_logical_bytes,bool(sink)};
}
Restoration Pipeline::restore_source(int source_frames, int height, int width, const Tensor &noise,
                                    const EncodedFrameSource &source, const std::function<void()> &verify_complete_source,
                                    const Event &event, std::atomic<bool> &cancelled,
                                    const FFNRoute &ffn, const DecodedFrameSink &sink) {
    require(loaded_ && source_frames >= 1 && source_frames <= 121 && height > 0 && width > 0 &&
            height % 16 == 0 && width % 16 == 0 && bool(source) && bool(verify_complete_source),
            "invalid loaded SeedVR2 streaming source restoration");
    const int frames = padded_frames(source_frames);
    require(noise.shape() == mx::Shape{1,16,(frames-1)/4+1,height/8,width/8} && noise.dtype() == mx::float32,
            "SeedVR2 streaming source requires matching FP32 sampling noise");
    const auto restore_started = Clock::now();
    double staging_seconds = 0;
    auto release_dit = [&] {
        const auto started = Clock::now();
        mx::synchronize(); dit_.clear(); mx::clear_cache();
        staging_seconds += seconds(started,Clock::now());
    };
    if (stage_dit_ && dit_.bytes()) release_dit();
    const auto started = Clock::now();
    auto condition = vae_.encode_stream(source_frames,height,width,source,event,cancelled);
    // A declared frame count is not EOF proof. Reject extra/short/changed
    // source sequences before DiT work, and release import/codec owners now.
    verify_complete_source();
    checkpoint(cancelled);
    const auto encoded = Clock::now();
    if (stage_dit_) {
        const auto reloading = Clock::now();
        auto source_unchanged = [&] {
            require(std::filesystem::is_regular_file(dit_path_) &&
                    std::filesystem::file_size(dit_path_) == dit_source_bytes_ &&
                    std::filesystem::last_write_time(dit_path_) == dit_source_modified_,
                    "SeedVR2 DiT checkpoint changed during staged loading");
        };
        source_unchanged();
        if (event) event("seedvr2_dit_stage_load",0,1);
        dit_.load(dit_path_,dit_dtype_);
        require(!stable_profile_ || dit_.stable_fp16_profile(),
                "SeedVR2 DiT staged reload changed the qualified stable precision profile");
        source_unchanged(); checkpoint(cancelled);
        if (event) event("seedvr2_dit_stage_load",1,1);
        staging_seconds += seconds(reloading,Clock::now());
    }
    const auto denoise_started = Clock::now();
    auto velocity = dit_.forward(noise,condition,1000.f,event,cancelled,{},ffn);
    auto latent = noise-mx::astype(velocity,mx::float32);
    mx::eval(latent);
    const auto denoised = Clock::now();
    if (stage_dit_) release_dit();
    const auto decode_started = Clock::now();
    Tensor restored(0);
    if (sink) vae_.decode_stream(latent,source_frames,sink,event,cancelled);
    else {
        restored = time_slice(vae_.decode(latent,event,cancelled),0,source_frames);
        mx::eval(restored);
    }
    const auto decoded = Clock::now(); checkpoint(cancelled);
    return {restored,condition,velocity,latent,seconds(started,encoded),
            seconds(denoise_started,denoised),seconds(decode_started,decoded),
            staging_seconds,seconds(restore_started,decoded),0,0,bool(sink)};
}
} // namespace tc::seedvr2

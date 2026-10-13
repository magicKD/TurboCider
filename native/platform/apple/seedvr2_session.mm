#include "bridge.hpp"
#include "platform.hpp"
#include "seedvr2_memory_sampler.hpp"
#include "../../models/seedvr2/pipeline.hpp"
#include "../../models/seedvr2/media.hpp"
#include "../../models/seedvr2/memory.hpp"
#include "../../models/seedvr2/precision.hpp"
#include "../../media/image.hpp"
#include "../../media/video.hpp"
#include "../../media/video_input.hpp"
#include "../../runtime/residency.hpp"
#include <mlx/random.h>

namespace tc {
namespace {
constexpr const char *kDiTSha = "98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660";
constexpr const char *kVAESha = "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1";

struct SourceStamp {
    std::filesystem::path path;
    uintmax_t bytes;
    std::filesystem::file_time_type modified;
    void check() const {
        require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) == bytes &&
                std::filesystem::last_write_time(path) == modified,
                "SeedVR2 checkpoint changed since verification: " + path.string());
    }
};
SourceStamp verify_source(const std::filesystem::path &path, uintmax_t bytes, const char *sha) {
    require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) == bytes,
            "missing or incomplete original SeedVR2 checkpoint: " + path.string());
    SourceStamp stamp{path, bytes, std::filesystem::last_write_time(path)};
    require(sha256_file(path) == sha, "SeedVR2 checkpoint SHA-256 does not match the original FP16 release");
    stamp.check();
    return stamp;
}
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
Tensor channels(const Tensor &pixels, int first, int last) {
    return mx::slice(pixels, {0, 0, 0, first}, {pixels.shape(0), pixels.shape(1), pixels.shape(2), last});
}

class SeedVR2Session final : public ModelSession {
    std::filesystem::path root_;
    SourceStamp dit_source_, vae_source_;
    seedvr2::Pipeline pipeline_;
    bool loaded_ = false;
  public:
    explicit SeedVR2Session(const std::filesystem::path &root)
        : root_(std::filesystem::absolute(root)),
          dit_source_(verify_source(root_ / "diffusion_models/seedvr2_3b_fp16.safetensors", 6784268336, kDiTSha)),
          vae_source_(verify_source(root_ / "vae/seedvr2_ema_vae_fp16.safetensors", 501324814, kVAESha)) {}
    void unload() override { pipeline_.clear(); loaded_ = false; mx::clear_cache(); }
    LoadResult load(const Event &event, std::atomic<bool> &cancelled) override {
        checkpoint(cancelled);
        try {
            dit_source_.check(); vae_source_.check();
            pipeline_.load(root_, event, cancelled);
            require(pipeline_.dit().stable_fp16_profile(),
                    "SeedVR2 public inference requires the stable FP16-projection/FP32-state profile; use offline probes for other DiT recipes");
            dit_source_.check(); vae_source_.check();
            loaded_ = true;
        } catch (...) { unload(); throw; }
        return {pipeline_.bytes(), mx::get_active_memory()};
    }
    RunResult prepare(const Request &r, bool warmup, const Event &event, std::atomic<bool> &cancelled) override {
        require(r.model == "seedvr2-3b", "request model differs from SeedVR2 session");
        require(!warmup, "SeedVR2 preparation warmup is not yet supported");
        const auto plan = make_plan(r);
        const auto started = Clock::now();
        load(event, cancelled);
        return native_run_result(@{@"schema_version": @1, @"prepared": @YES, @"warmup": @NO,
            @"model": @(r.model.c_str()), @"operation": @(r.operation.c_str()),
            @"runtime_backend": @"seedvr2-mlx", @"runtime_dependency": @"native",
            @"seconds": @(elapsed(started)), @"mlx_active_bytes": @(mx::get_active_memory()),
            @"mlx_peak_bytes": @(mx::get_peak_memory())}, r, plan);
    }
    RunResult generate(const Request &r, const Event &event, std::atomic<bool> &cancelled) override {
        require(r.model == "seedvr2-3b", "request model differs from SeedVR2 session");
        const auto plan = make_plan(r);
        ResidencyPolicy::validate_budget(plan, device_info().physical_memory);
        const bool video = r.operation == "video.upscale";
        const auto output = std::filesystem::absolute(r.output);
        const auto source = std::filesystem::absolute(r.inputs[0].path);
        require(!r.output.empty() && output.extension() == (video ? ".mp4" : ".png"),
                "SeedVR2 output must be PNG for images or MP4 for videos");
        require(std::filesystem::weakly_canonical(output) != std::filesystem::weakly_canonical(source),
                "SeedVR2 will not overwrite the source media");
        checkpoint(cancelled);
        const bool reused = loaded_;
        mx::synchronize();
        mx::reset_peak_memory();
        const auto allocator_cache_bytes = seedvr2::allocator_cache_limit();
        seedvr2::AllocatorCacheScope allocator(allocator_cache_bytes);
        const auto started = Clock::now();
        SeedVR2MemorySampler memory;
        load(event, cancelled);
        const double load_seconds = elapsed(started);
        const auto import_started = Clock::now();
        auto unit = video ? load_video_tensor(source, r.frames, r.fps, event, cancelled) : load_rgba_image_tensor(source);
        const int source_height = unit.shape(1), source_width = unit.shape(2);
        unit = seedvr2::resize_bicubic(unit, r.height, r.width);
        auto rgb = channels(unit, 0, 3);
        auto pixels = mx::expand_dims(mx::transpose(rgb * 2.f - 1.f, {3, 0, 1, 2}), 0);
        mx::eval(pixels);
        const double import_seconds = elapsed(import_started);
        const int frames = seedvr2::padded_frames(r.frames);
        const mx::Shape noise_shape{1, 16, (frames - 1) / 4 + 1, r.height / 8, r.width / 8};
        auto noise = mx::random::normal(noise_shape, mx::float32, mx::random::key(r.seed));
        if (!r.noise_path.empty()) {
            auto values = mx::load_safetensors(r.noise_path).first;
            require(values.count("noise") == 1, "SeedVR2 noise fixture must contain a noise tensor");
            noise = values.at("noise");
            require(noise.shape() == noise_shape && noise.dtype() == mx::float32,
                    "SeedVR2 noise override must be matching FP32 BCTHW");
        }
        mx::eval(noise);
        require(mx::all(mx::isfinite(noise)).item<bool>(), "SeedVR2 sampling noise must be finite");
        if (!r.dump.empty()) {
            std::filesystem::create_directories(r.dump);
            mx::save_safetensors((std::filesystem::path(r.dump) / "input.safetensors").string(),
                                {{"pixels", pixels}, {"noise", noise}});
        }
        const auto temporary = output.parent_path() /
            (output.stem().string() + ".seedvr2-" + NSUUID.UUID.UUIDString.UTF8String + output.extension().string());
        try {
            auto result = pipeline_.restore(pixels, noise, event, cancelled, r.dump);
            require(result.pixels.shape() == mx::Shape{1, 3, r.frames, r.height, r.width} &&
                    mx::all(mx::isfinite(result.pixels)).item<bool>(), "SeedVR2 produced invalid restored pixels");
            auto restored = mx::transpose(mx::squeeze(mx::astype(result.pixels, mx::float32), 0), {1, 2, 3, 0});
            mx::eval(restored);
            if (!r.dump.empty()) mx::save_safetensors((std::filesystem::path(r.dump) / "restoration.safetensors").string(),
                {{"pixels", result.pixels}, {"condition", result.condition}, {"velocity", result.velocity}, {"latent", result.latent}});
            checkpoint(cancelled);
            if (event) event("export", 0, 1);
            const auto export_started = Clock::now();
            if (video) {
                auto bytes = mx::astype(mx::round(mx::clip((restored + 1.f) * .5f, Tensor(0.f), Tensor(1.f)) * 255.f), mx::uint8);
                mx::eval(bytes);
                write_video_rgb24(temporary, bytes.data<uint8_t>(), r.frames, r.width, r.height, r.fps);
                const auto info = probe_video(temporary);
                require(info.frames == r.frames && info.width == r.width && info.height == r.height && info.fps == r.fps,
                        "SeedVR2 MP4 failed frame/FPS/dimension validation");
            } else {
                auto alpha = channels(unit, 3, 4) * 2.f - 1.f;
                save_rgba_png(mx::concatenate({restored, alpha}, -1), temporary);
            }
            checkpoint(cancelled);
            std::filesystem::create_directories(output.parent_path());
            std::filesystem::rename(temporary, output);
            const double export_seconds = elapsed(export_started);
            if (event) event("export", 1, 1);
            memory.stop();
            return native_run_result(@{
                @"schema_version": @1, @"model": @(r.model.c_str()), @"operation": @(r.operation.c_str()),
                @"output": @(output.c_str()), @"width": @(r.width), @"height": @(r.height),
                @"frames": @(r.frames), @"padded_frames": @(frames), @"fps": @(r.fps),
                @"source_width": @(source_width), @"source_height": @(source_height),
                @"steps": @1, @"actual_denoise_steps": @1, @"seed": @(r.seed), @"execution": @"gpu",
                @"warmup": @NO, @"prepared": @NO,
                @"runtime_backend": @"seedvr2-mlx", @"runtime_dependency": @"native",
                @"runtime_precision": @(seedvr2::kStablePrecision),
                @"numerical_recipe": @(seedvr2::kStableRecipe),
                @"dit_projection_dtype": @"fp16", @"dit_residual_dtype": @"fp32",
                @"dit_rms_dtype": @"fp32", @"dit_time_dtype": @"fp32",
                @"dit_velocity_dtype": @"fp32", @"dit_attention_dtype": @"fp16",
                @"dit_rope_angle_dtype": @"original_fp16_angles_then_fp32_trig",
                @"persistent_model": @YES, @"model_reused": @(reused), @"prompt_cache_hit": @NO,
                @"conditioning_cache_hit": @NO, @"resize_method": @"bicubic_align_corners_false_no_antialias",
                @"color_correction": @"none", @"audio_output": @NO,
                @"vae_conv_recipe": @(pipeline_.vae_conv_recipe().c_str()),
                @"vae_norm_recipe": @(pipeline_.vae_norm_recipe().c_str()),
                @"vae_history_recipe": @(pipeline_.vae_history_recipe().c_str()),
                @"vae_pointwise_recipe": @(pipeline_.vae_pointwise_recipe().c_str()),
                @"vae_conv_tile_rows_limit": @(pipeline_.vae_conv_tile_rows()),
                @"vae_attention_recipe": @"full_spatial_keys_fp32_query512",
                @"vae_history_policy": video ? @"causal_temporal_history" : @"no_unused_single_frame_history",
                @"allocator_cache_limit_bytes": @(allocator_cache_bytes),
                @"allocator_cache_retention": @"bounded_during_request_cleared_on_exit",
                @"mlx_peak_scope": @"request_active_allocations_excludes_allocator_cache_and_non_mlx",
                @"sampled_process_footprint_peak_bytes": memory.footprint_available() ? (id)@(memory.footprint_peak()) : (id)NSNull.null,
                @"sampled_metal_allocation_peak_bytes": memory.metal_available() ? (id)@(memory.metal_peak()) : (id)NSNull.null,
                @"memory_sampling_interval_ms": @20,
                @"memory_sample_count": @(memory.samples()), @"memory_sample_failures": @(memory.failures()),
                @"sampled_memory_scope": @"process_task_vm_info.phys_footprint_and_MTLDevice.currentAllocatedSize_overlap_do_not_add",
                @"sampled_peaks_are_hard_bounds": @NO,
                @"checkpoint_sha256": @(kDiTSha), @"vae_sha256": @(kVAESha),
                @"checkpoint_sha256_verified": @YES, @"seconds": @(elapsed(started)),
                @"timings_seconds": @{@"request_wall": @(elapsed(started)), @"load": @(load_seconds),
                    @"media_import": @(import_seconds), @"vae_encode": @(result.encode_seconds),
                    @"denoise": @(result.denoise_seconds), @"vae_decode": @(result.decode_seconds),
                    @"export": @(export_seconds)},
                @"mlx_active_bytes": @(mx::get_active_memory()), @"mlx_peak_bytes": @(mx::get_peak_memory())
            }, r, plan);
        } catch (...) {
            std::error_code error;
            std::filesystem::remove(temporary, error);
            unload();
            throw;
        }
    }
};
} // namespace
std::unique_ptr<ModelSession> create_seedvr2(const std::filesystem::path &root) {
    return std::make_unique<SeedVR2Session>(root);
}
} // namespace tc

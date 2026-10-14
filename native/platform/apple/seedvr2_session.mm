#include "bridge.hpp"
#include "platform.hpp"
#include "seedvr2_memory_sampler.hpp"
#include "../../models/seedvr2/pipeline.hpp"
#include "../../models/seedvr2/media.hpp"
#include "../../models/seedvr2/memory.hpp"
#include "../../models/seedvr2/precision.hpp"
#include "../../models/seedvr2/hybrid.hpp"
#include "../../media/image.hpp"
#include "../../media/video.hpp"
#include "../../media/video_input.hpp"
#include "../../media/rgb_video_reader.hpp"
#include "../../runtime/residency.hpp"
#include <mlx/random.h>
#include <CommonCrypto/CommonDigest.h>
#include <cstdlib>

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
    return mx::slice(pixels,{0,0,0,first},{pixels.shape(0),pixels.shape(1),pixels.shape(2),last});
}
// Offline diagnostic only. Hash the exact bytes handed to the encoder, so
// codec variability is distinguishable from restoration/quantization errors.
struct VideoPixelHash {
    bool enabled = false;
    CC_SHA256_CTX context{};
    explicit VideoPixelHash(bool video) {
        if (const char *flag = std::getenv("TURBOCIDER_SEEDVR2_VIDEO_PIXEL_HASH")) {
            require(std::string(flag) == "0" || std::string(flag) == "1", "video pixel hash flag must be 0 or 1");
            enabled = video && std::string(flag) == "1";
        }
        if (enabled) require(CC_SHA256_Init(&context) == 1,"cannot initialize video pixel hash");
    }
    void add(const uint8_t *data, size_t bytes) {
        if (!enabled) return;
        while (bytes) {
            const auto count = CC_LONG(std::min<size_t>(bytes,1u << 28));
            require(CC_SHA256_Update(&context,data,count) == 1,"cannot update video pixel hash");
            data += count; bytes -= count;
        }
    }
    std::string finish() {
        if (!enabled) return {};
        unsigned char digest[CC_SHA256_DIGEST_LENGTH];
        require(CC_SHA256_Final(digest,&context) == 1,"cannot finish video pixel hash");
        constexpr char hex[] = "0123456789abcdef";
        std::string result(CC_SHA256_DIGEST_LENGTH*2,'0');
        for (size_t i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
            result[i*2] = hex[digest[i] >> 4]; result[i*2+1] = hex[digest[i] & 15];
        }
        return result;
    }
};
class SeedVR2Session final : public ModelSession {
    std::filesystem::path root_;
    SourceStamp dit_source_, vae_source_;
    seedvr2::Pipeline pipeline_;
    std::unique_ptr<seedvr2::HybridFFN> hybrid_;
    std::filesystem::path hybrid_manifest_;
    std::vector<SourceStamp> hybrid_files_;
    std::vector<std::pair<std::filesystem::path,std::filesystem::file_time_type>> hybrid_directories_;
    bool loaded_ = false;
    void check_hybrid_source() const {
        for (const auto &file : hybrid_files_) file.check();
        for (const auto &[path, modified] : hybrid_directories_)
            require(std::filesystem::is_directory(std::filesystem::symlink_status(path)) &&
                    std::filesystem::last_write_time(path) == modified,
                    "SeedVR2 compiled artifact directory changed since verification");
    }
    void configure_hybrid(const Request &r, const Event &event, std::atomic<bool> &cancelled) {
        if (r.execution != "gpu_ane") {
            hybrid_.reset(); hybrid_manifest_.clear(); hybrid_files_.clear(); hybrid_directories_.clear();
            return;
        }
        const auto manifest = std::filesystem::canonical(r.ane_manifest);
        if (hybrid_ && hybrid_manifest_ == manifest) { check_hybrid_source(); return; }
        hybrid_.reset(); hybrid_files_.clear(); hybrid_directories_.clear(); hybrid_manifest_.clear();
        try {
            seedvr2::validate_public_hybrid_manifest(manifest);
            auto config = seedvr2::read_hybrid_config(manifest);
            require(config.source_operand_recipe == "original-fp16" && config.rows == 512 &&
                    config.channels == 2048 && config.blocks.size() == 32,
                    "SeedVR2 qualified hybrid partition geometry changed");
            for (int block = 0; block < 32; ++block)
                require(config.blocks[block] == block, "SeedVR2 qualified hybrid block coverage changed");
            auto stamp = [&](const std::filesystem::path &path) {
                hybrid_files_.push_back({path,std::filesystem::file_size(path),std::filesystem::last_write_time(path)});
            };
            stamp(manifest);
            for (const auto &root : config.artifacts) {
                hybrid_directories_.emplace_back(root,std::filesystem::last_write_time(root));
                for (const auto &entry : std::filesystem::recursive_directory_iterator(root)) {
                    if (entry.is_directory()) hybrid_directories_.emplace_back(entry.path(),entry.last_write_time());
                    else stamp(entry.path());
                }
            }
            hybrid_ = std::make_unique<seedvr2::HybridFFN>(pipeline_.dit(),std::move(config),event,cancelled);
            check_hybrid_source();
            hybrid_manifest_ = manifest;
        } catch (...) {
            hybrid_.reset(); hybrid_files_.clear(); hybrid_directories_.clear();
            throw;
        }
    }
    void configure_residency(const Request &r) {
        const bool staged = r.residency == "component_staged";
        if (pipeline_.stages_dit() == staged) return;
        // The same public session may alternate policies. Never retain a
        // partially staged bank under a resident request (or vice versa).
        unload();
        pipeline_.set_stage_dit(staged);
    }
  public:
    explicit SeedVR2Session(const std::filesystem::path &root)
        : root_(std::filesystem::absolute(root)),
          dit_source_(verify_source(root_ / "diffusion_models/seedvr2_3b_fp16.safetensors", 6784268336, kDiTSha)),
          vae_source_(verify_source(root_ / "vae/seedvr2_ema_vae_fp16.safetensors", 501324814, kVAESha)) {}
    void unload() override {
        hybrid_.reset(); hybrid_manifest_.clear(); hybrid_files_.clear(); hybrid_directories_.clear();
        pipeline_.clear(); loaded_ = false; mx::clear_cache();
    }
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
        configure_residency(r);
        load(event, cancelled);
        pipeline_.configure_vae(r.seedvr2_vae);
        configure_hybrid(r,event,cancelled);
        return native_run_result(@{@"schema_version": @1, @"prepared": @YES, @"warmup": @NO,
            @"model": @(r.model.c_str()), @"operation": @(r.operation.c_str()),
            @"runtime_backend": @"seedvr2-mlx", @"runtime_dependency": @"native",
            @"weight_residency_recipe": @(pipeline_.weight_residency_recipe()),
            @"resident_parameter_bytes": @(pipeline_.bytes()),
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
        const bool reused = loaded_ && pipeline_.stages_dit() == (r.residency == "component_staged");
        mx::synchronize();
        mx::reset_peak_memory();
        const auto allocator_cache_bytes = r.seedvr2_vae ? size_t(r.seedvr2_vae->allocator_cache_bytes)
                                                       : seedvr2::allocator_cache_limit();
        seedvr2::AllocatorCacheScope allocator(allocator_cache_bytes);
        const auto started = Clock::now();
        SeedVR2MemorySampler memory;
        const bool hybrid_reused = hybrid_ && r.execution == "gpu_ane" &&
            hybrid_manifest_ == std::filesystem::weakly_canonical(r.ane_manifest) &&
            pipeline_.stages_dit() == (r.residency == "component_staged");
        configure_residency(r);
        load(event, cancelled);
        pipeline_.configure_vae(r.seedvr2_vae);
        configure_hybrid(r,event,cancelled);
        const double load_seconds = elapsed(started);
        const auto import_started = Clock::now();
        const bool stream_encode = video && r.seedvr2_vae && r.seedvr2_vae->encoder == "streaming";
        std::unique_ptr<RGB24VideoReader> source_reader;
        RGBVideoFrame source_frame;
        int source_height = 0, source_width = 0, streamed_input_frames = 0;
        size_t source_rgb_capacity_peak = 0;
        double frame_import_seconds = 0;
        Tensor unit(0);
        if (stream_encode) source_reader = std::make_unique<RGB24VideoReader>(source,r.frames,r.fps,event,cancelled);
        else {
            unit = video ? load_video_tensor(source, r.frames, r.fps, event, cancelled) : load_rgba_image_tensor(source);
            source_height = unit.shape(1); source_width = unit.shape(2);
            unit = seedvr2::resize_bicubic(unit, r.height, r.width);
        }
        std::optional<Tensor> resident_rgb;
        std::optional<seedvr2::NormalizedMediaInput> media_input;
        auto pixels = [&] {
            if (stream_encode) return Tensor(0);
            if (!video && r.residency == "resident") {
                // Keep both original source/RGB owners and defer the alpha
                // view until export, exactly as in the speed-first baseline.
                // Creating another source consumer before eval can change
                // normalization graph fusion even without copying alpha.
                resident_rgb = channels(unit,0,3);
                auto value = mx::expand_dims(mx::transpose(*resident_rgb*2.f-1.f,{3,0,1,2}),0);
                mx::eval(value);
                return value;
            }
            media_input = seedvr2::consume_resized_input(unit,video,true);
            return std::move(media_input->pixels);
        }();
        const bool owned_alpha = media_input && media_input->alpha_is_owned;
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
            const bool stream_decode = video && r.seedvr2_vae && r.seedvr2_vae->decoder == "streaming";
            VideoPixelHash pixel_hash(video);
            std::unique_ptr<RGB24VideoWriter> stream_writer;
            double frame_export_seconds = 0;
            seedvr2::DecodedFrameSink sink;
            if (stream_decode) {
                stream_writer = std::make_unique<RGB24VideoWriter>(temporary,r.frames,r.width,r.height,r.fps,&cancelled);
                sink = [&](int index, const Tensor &frame) {
                    checkpoint(cancelled);
                    require(frame.shape() == mx::Shape{1,3,1,r.height,r.width} && frame.dtype() == mx::float16 &&
                            mx::all(mx::isfinite(frame)).item<bool>(), "SeedVR2 streaming decoder produced an invalid frame");
                    const auto copying = Clock::now();
                    auto rgb = mx::astype(mx::reshape(mx::transpose(frame,{0,2,3,4,1}),{r.height,r.width,3}),mx::float32);
                    auto bytes = mx::astype(mx::round(mx::clip((rgb+1.f)*.5f,Tensor(0.f),Tensor(1.f))*255.f),mx::uint8);
                    mx::eval(bytes);
                    pixel_hash.add(bytes.data<uint8_t>(),bytes.nbytes());
                    stream_writer->append(index,bytes.data<uint8_t>(),bytes.nbytes());
                    frame_export_seconds += elapsed(copying);
                    if (event) event("seedvr2_vae_export_frame",index+1,r.frames);
                };
            }
            const auto calls_before = hybrid_ ? hybrid_->calls() : 0;
            const auto blocks_before = hybrid_ ? hybrid_->hybrid_blocks() : 0;
            const auto copies_before = hybrid_ ? hybrid_->copied_bytes() : 0;
            const auto batches_before = hybrid_ ? hybrid_->batches() : 0;
            const auto route = hybrid_ ? seedvr2::FFNRoute([&](int block, const Tensor &input) {
                return hybrid_->predict(block,input);
            }) : seedvr2::FFNRoute{};
            seedvr2::EncodedFrameSource frame_source = [&](int index) {
                require(source_reader && index == streamed_input_frames,"streaming source/frame order mismatch");
                const auto importing = Clock::now();
                require(source_reader->next(source_frame) && source_frame.index == index,"source video ended before the complete declared frame count");
                source_height = source_frame.height; source_width = source_frame.width;
                source_rgb_capacity_peak = std::max(source_rgb_capacity_peak,source_frame.pixels.capacity());
                auto frame_unit = mx::astype(Tensor(source_frame.pixels.data(),{1,source_height,source_width,3},mx::uint8),mx::float32)/255.f;
                // Match the buffered import's FP32 conversion/evaluation
                // boundary before resize and normalized-RGB arithmetic.
                mx::eval(frame_unit);
                frame_unit = seedvr2::resize_bicubic(frame_unit,r.height,r.width);
                auto normalized = seedvr2::consume_resized_input(frame_unit,true).pixels;
                ++streamed_input_frames;
                frame_import_seconds += elapsed(importing);
                return normalized;
            };
            auto verify_source = [&] {
                const auto finishing = Clock::now();
                require(source_reader && streamed_input_frames == r.frames &&
                        source_reader->frames_read() == r.frames && !source_reader->next(source_frame),
                        "streaming import did not verify complete source EOF");
                source_reader.reset(); source_frame = RGBVideoFrame{};
                frame_import_seconds += elapsed(finishing);
            };
            auto result = stream_encode
                ? pipeline_.restore_source(r.frames,r.height,r.width,noise,frame_source,verify_source,event,cancelled,route,sink)
                : (media_input ? pipeline_.restore_owned(std::move(pixels),noise,event,cancelled,r.dump,route,sink)
                               : pipeline_.restore(pixels,noise,event,cancelled,r.dump,route));
            const uint64_t coreml_calls = hybrid_ ? hybrid_->calls()-calls_before : 0;
            const uint64_t hybrid_blocks = hybrid_ ? hybrid_->hybrid_blocks()-blocks_before : 0;
            if (hybrid_) {
                const uint64_t tokens = uint64_t(noise_shape[2])*(r.height/16)*(r.width/16);
                require(coreml_calls == 32*((tokens+511)/512) && hybrid_blocks == 32,
                        "SeedVR2 hybrid did not execute its declared all-block prefix partition");
                check_hybrid_source();
            }
            Tensor restored(0);
            if (stream_decode) require(result.pixels_streamed && stream_writer->frames_written() == r.frames,
                                       "SeedVR2 streaming decoder did not export the complete source frame count");
            else {
                require(!result.pixels_streamed && result.pixels.shape() == mx::Shape{1, 3, r.frames, r.height, r.width} &&
                        mx::all(mx::isfinite(result.pixels)).item<bool>(), "SeedVR2 produced invalid restored pixels");
                restored = mx::transpose(mx::squeeze(mx::astype(result.pixels,mx::float32),0),{1,2,3,0});
                mx::eval(restored);
            }
            if (!r.dump.empty()) mx::save_safetensors((std::filesystem::path(r.dump) / "restoration.safetensors").string(),
                {{"pixels", result.pixels}, {"condition", result.condition}, {"velocity", result.velocity}, {"latent", result.latent}});
            checkpoint(cancelled);
            if (event) event("export", 0, 1);
            const auto export_started = Clock::now();
            if (stream_decode) stream_writer->finish();
            else if (video) {
                auto bytes = mx::astype(mx::round(mx::clip((restored + 1.f) * .5f, Tensor(0.f), Tensor(1.f)) * 255.f), mx::uint8);
                mx::eval(bytes);
                pixel_hash.add(bytes.data<uint8_t>(),bytes.nbytes());
                write_video_rgb24(temporary, bytes.data<uint8_t>(), r.frames, r.width, r.height, r.fps);
                const auto info = probe_video(temporary);
                require(info.frames == r.frames && info.width == r.width && info.height == r.height && info.fps == r.fps,
                        "SeedVR2 MP4 failed frame/FPS/dimension validation");
            } else {
                auto alpha = (media_input ? media_input->alpha : channels(unit,3,4)) * 2.f - 1.f;
                save_rgba_png(mx::concatenate({restored, alpha}, -1), temporary);
            }
            checkpoint(cancelled);
            std::filesystem::create_directories(output.parent_path());
            std::filesystem::rename(temporary, output);
            const double export_seconds = elapsed(export_started) + frame_export_seconds;
            if (event) event("export", 1, 1);
            memory.stop();
            const auto video_pixel_sha = pixel_hash.finish();
            const bool file_rows = pipeline_.vae_spatial_recipe() == "file_rows";
            const std::string numerical_recipe = std::string(hybrid_ ? seedvr2::kHybridRecipe : seedvr2::kStableRecipe) +
                (file_rows ? "+vae-globalstats-file-rows-v1" : "");
            const auto spatial = result.spatial_stats;
            return native_run_result(@{
                @"schema_version": @1, @"model": @(r.model.c_str()), @"operation": @(r.operation.c_str()),
                @"output": @(output.c_str()), @"width": @(r.width), @"height": @(r.height),
                @"frames": @(r.frames), @"padded_frames": @(frames), @"fps": @(r.fps),
                @"source_width": @(source_width), @"source_height": @(source_height),
                @"steps": @1, @"actual_denoise_steps": @1, @"seed": @(r.seed), @"execution": hybrid_ ? @"gpu_ane" : @"gpu",
                @"warmup": @NO, @"prepared": @NO,
                @"runtime_backend": @"seedvr2-mlx", @"runtime_dependency": @"native",
                @"runtime_precision": @(hybrid_ ? seedvr2::kHybridPrecision : seedvr2::kStablePrecision),
                @"numerical_recipe": @(numerical_recipe.c_str()),
                @"coreml_prefix_compute_units": hybrid_ ? @"cpuAndNeuralEngine" : @"not_applicable",
                @"coreml_prefix_activation_dtype": hybrid_ ? @"fp16" : @"not_applicable",
                @"coreml_prefix_reused": @(hybrid_reused), @"coreml_prefix_calls": @(coreml_calls),
                @"coreml_prefix_video_blocks": @(hybrid_blocks),
                @"coreml_prefix_batches": @(hybrid_ ? hybrid_->batches()-batches_before : 0),
                @"coreml_output_copied_bytes": @(hybrid_ ? hybrid_->copied_bytes()-copies_before : 0),
                @"coreml_prefix_backing_bytes": @(hybrid_ ? hybrid_->prefix_backing_bytes() : 0),
                @"coreml_prefix_manifest_sha256": hybrid_ ? @(hybrid_->config().manifest_sha256.c_str()) : (id)NSNull.null,
                @"physical_ane_occupancy_measured": @NO,
                @"dit_projection_dtype": @"fp16", @"dit_residual_dtype": @"fp32",
                @"dit_rms_dtype": @"fp32", @"dit_time_dtype": @"fp32",
                @"dit_velocity_dtype": @"fp32", @"dit_attention_dtype": @"fp16",
                @"dit_rope_angle_dtype": @"original_fp16_angles_then_fp32_trig",
                @"persistent_model": @YES, @"model_reused": @(reused), @"prompt_cache_hit": @NO,
                @"vae_weights_reused": @(reused),
                @"dit_weights_reused": @(reused && !pipeline_.stages_dit()),
                @"dit_reloaded_for_sampling": @(pipeline_.stages_dit()),
                @"weight_residency_recipe": @(pipeline_.weight_residency_recipe()),
                @"resident_parameter_bytes": @(pipeline_.bytes()),
                @"resident_parameter_bytes_scope": @"MLX parameter bank; excludes Core ML parameters and buffers",
                @"resident_components": hybrid_ ? (pipeline_.dit().bytes() ? @[@"vae", @"dit", @"coreml_ffn_prefix"] : @[@"vae", @"coreml_ffn_prefix"]) : (pipeline_.dit().bytes() ? @[@"vae", @"dit"] : @[@"vae"]),
                @"conditioning_cache_hit": @NO, @"resize_method": @"bicubic_align_corners_false_no_antialias",
                @"media_input_retention_recipe": video ? (stream_encode ? @"bounded_rgb_reader_and_fp16_encoder_chunks" : @"normalized_rgb_only") :
                    (owned_alpha ? @"normalized_rgb_plus_owned_alpha" : @"normalized_rgb_plus_source_rgba"),
                @"media_retained_alpha_bytes": @(video ? 0 : uint64_t(r.width)*r.height*sizeof(float)),
                @"media_retained_source_rgba_bytes": @(!video && !owned_alpha ? uint64_t(r.width)*r.height*4*sizeof(float) : 0),
                @"normalized_input_release_recipe": stream_encode ? @"bounded_chunks_and_source_eof_before_sampling" :
                    (media_input ? @"after_vae_encode_owned" : @"borrowed"),
                @"normalized_input_released_logical_bytes": @(result.released_input_logical_bytes),
                @"normalized_input_release_bytes_scope": @"logical normalized RGB tensor bytes; not physical-memory or peak-reduction guarantee",
                @"color_correction": @"none", @"audio_output": @NO,
                @"vae_conv_recipe": @(pipeline_.vae_conv_recipe().c_str()),
                @"vae_spatial_recipe": @(pipeline_.vae_spatial_recipe().c_str()),
                @"vae_spatial_rows": @(r.seedvr2_vae ? r.seedvr2_vae->spatial_rows : 64),
                @"vae_spatial_threshold_bytes": @(256ull<<20),
                @"vae_spatial_disk_budget_bytes": @(r.seedvr2_vae ? r.seedvr2_vae->spatial_disk_budget_bytes : (8ull<<30)),
                @"vae_spatial_tile_convolutions": @(spatial ? spatial->tile_convolutions : 0),
                @"vae_spatial_global_normalizations": @(spatial ? spatial->global_normalizations : 0),
                @"vae_spatial_file_written_bytes": @(spatial ? spatial->written_bytes : 0),
                @"vae_spatial_file_read_bytes": @(spatial ? spatial->read_bytes : 0),
                @"vae_spatial_file_peak_bytes": @(spatial ? spatial->disk_peak_bytes : 0),
                @"vae_spatial_host_window_peak_bytes": @(spatial ? spatial->host_peak_bytes.load() : 0),
                @"vae_spatial_file_live_bytes_after_decode": @(spatial ? spatial->disk_live_bytes : 0),
                @"vae_spatial_host_live_bytes_after_decode": @(spatial ? spatial->host_live_bytes.load() : 0),
                @"vae_spatial_cleanup_failures": @(spatial ? spatial->cleanup_failures : 0),
                @"vae_spatial_scope": @"eligible decoder norm/conv row bands; global statistics/full attention preserved; disk budget is not RAM budget",
                @"vae_encoder_recipe": stream_encode ? @"streaming" : @"buffered",
                @"streamed_input_frames": @(streamed_input_frames),
                @"source_rgb24_buffer_capacity_peak_bytes": @(source_rgb_capacity_peak),
                @"source_rgb24_buffer_bytes_scope": @"caller-owned one-frame RGB24 capacity; excludes codec/model/normalized chunk storage",
                @"vae_encoder_input_schedule": stream_encode ? @"initial5_then4_with_last_frame_padding" : @"complete_normalized_rgb_tensor",
                @"vae_encoder_wall_includes_frame_import": @(stream_encode),
                @"vae_decoder_recipe": stream_decode ? @"streaming" : @"buffered",
                @"video_encoder_recipe": video ? (stream_decode ? @"h264_source_fps_frame_gop2s_no_duration_limit_v1" : @"h264_buffered_legacy") : @"not_applicable",
                @"decoded_output_retention_recipe": stream_decode ? @"frame_sink_no_full_rgb_tensor" : @"full_rgb_bcthw",
                @"streamed_output_frames": @(stream_writer ? stream_writer->frames_written() : 0),
                @"video_preencode_rgb24_sha256": pixel_hash.enabled ? (id)@(video_pixel_sha.c_str()) : (id)NSNull.null,
                @"video_preencode_pixel_hash_diagnostic": @(pixel_hash.enabled),
                @"vae_decoder_wall_includes_frame_export": @(stream_decode),
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
                    @"media_import": @(import_seconds+frame_import_seconds),
                    @"vae_encode": @(std::max(0.0,result.encode_seconds-frame_import_seconds)),
                    @"vae_encode_wall": @(result.encode_seconds), @"frame_import": @(frame_import_seconds),
                    @"denoise": @(result.denoise_seconds), @"vae_decode": @(std::max(0.0,result.decode_seconds-frame_export_seconds)),
                    @"vae_decode_wall": @(result.decode_seconds), @"frame_export": @(frame_export_seconds),
                    @"weight_management": @(result.staging_seconds), @"restoration_total": @(result.restore_seconds),
                    @"input_release": @(result.input_release_seconds),
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

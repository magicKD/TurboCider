#include "bridge.hpp"
#include "../../media/image.hpp"
#include "../../models/relayvsr/pipeline.hpp"
#include "relayvsr/video.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>

namespace tc {
namespace {
namespace fs = std::filesystem;

struct StagingFiles {
    fs::path video, muxed;
    ~StagingFiles() {
        std::error_code error;
        fs::remove(video, error);
        fs::remove(muxed, error);
    }
};

class RelayVSRSession final : public ModelSession {
    fs::path root_;

public:
    explicit RelayVSRSession(const fs::path &root) : root_(fs::absolute(root)) {
        require(fs::is_regular_file(root_ / rv::kWanFile) && fs::is_regular_file(root_ / rv::kFlashFile),
                "Select the local RelayVSR 2× folder containing diffusion_models and vae");
    }

    bool uses_parent_mlx(const Request &request) const override { return request.operation == "image.upscale"; }
    LoadResult load(const Event &, std::atomic<bool> &cancelled) override {
        checkpoint(cancelled);
        return {};
    }
    void unload() override {}

    RunResult generate(const Request &r, const Event &event, std::atomic<bool> &cancelled) override {
        const auto started = Clock::now();
        const auto plan = make_plan(r);
        const bool video = r.operation == "video.upscale";
        const auto &input = r.inputs.front();
        const auto source = fs::absolute(input.path);
        const auto path = fs::absolute(r.output);
        require(!r.output.empty() && path.extension() == (video ? ".mp4" : ".png"),
                "RelayVSR writes PNG images and MP4 videos");
        require(!fs::exists(path), "RelayVSR output already exists");
        checkpoint(cancelled);
        if (event) event("media_import", 0, 1);

        rv::Frame first;
        relay_media::VideoTiming timing;
        std::unique_ptr<relay_media::VideoReader> reader;
        std::vector<float> alpha;
        if (video) {
            reader = std::make_unique<relay_media::VideoReader>(source, cancelled);
            require(reader->next(first), "No pictures in RelayVSR source video");
            timing = reader->timing();
            if (r.fps)
                require(std::abs(timing.fps - r.fps) < 1e-4, "RelayVSR uses the source FPS");
        } else {
            auto image = mx::contiguous(load_rgba_image_tensor(source, rv::kMaxInputSide), false, mx::StreamOrDevice(mx::Device::cpu));
            mx::eval(image);
            first.width = image.shape(2);
            first.height = image.shape(1);
            const auto count = size_t(first.width) * first.height;
            const auto *rgba = image.data<float>();
            first.rgb.resize(count * 3);
            alpha.resize(count);
            for (size_t i = 0; i < count; ++i) {
                for (int channel = 0; channel < 3; ++channel)
                    first.rgb[i * 3 + channel] = std::lround(rgba[i * 4 + channel] * 255.f);
                alpha[i] = rgba[i * 4 + 3];
            }
        }
        const int width = first.width * 2, height = first.height * 2;
        require(!r.width || (r.width == width && r.height == height),
                "RelayVSR output dimensions must be twice the oriented source dimensions");
        const double pixel_aspect = double(timing.pixel_aspect_width) / timing.pixel_aspect_height;
        fs::create_directories(path.parent_path());
        const auto stage = path.parent_path() / (".relayvsr-" + std::string(NSUUID.UUID.UUIDString.UTF8String));
        StagingFiles temporary{stage.string() + path.extension().string(), stage.string() + "-audio.mp4"};
        std::unique_ptr<relay_media::VideoWriter> writer;
        if (video) writer = std::make_unique<relay_media::VideoWriter>(temporary.video, width, height, timing, cancelled);
        int frames = 0;
        if (event) {
            event("media_import", 1, 1);
            event("relayvsr_restore", 0, video ? r.frames : 1);
        }
        const auto restored = rv::restore(root_, std::move(first), pixel_aspect, r.seed,
            [&](rv::Frame &frame) { return reader && reader->next(frame); },
            [&](const rv::Frame &frame, const uint8_t *rgb) {
                require(!r.frames || frames < r.frames, "RelayVSR source contains more frames than declared");
                if (writer) writer->append(frame, rgb);
                else {
                    // Upscale the source alpha separately so RelayVSR never
                    // invents opacity or fills transparent regions.
                    std::vector<float> rgba(size_t(width) * height * 4);
                    for (int y = 0; y < height; ++y) {
                        const int sy = y / 2, ay = std::clamp(sy + (y % 2 ? 1 : -1), 0, frame.height - 1);
                        for (int x = 0; x < width; ++x) {
                            const int sx = x / 2, ax = std::clamp(sx + (x % 2 ? 1 : -1), 0, frame.width - 1);
                            const auto index = size_t(y) * width + x;
                            for (int channel = 0; channel < 3; ++channel)
                                rgba[index * 4 + channel] = rgb[index * 3 + channel] / 127.5f - 1.f;
                            const auto row = size_t(sy) * frame.width, adjacent = size_t(ay) * frame.width;
                            const float a = .75f * alpha[row + sx] + .25f * alpha[row + ax];
                            const float b = .75f * alpha[adjacent + sx] + .25f * alpha[adjacent + ax];
                            rgba[index * 4 + 3] = 2.f * (.75f * a + .25f * b) - 1.f;
                        }
                    }
                    save_rgba_png(Tensor(rgba.data(), {1, height, width, 4}, mx::float32), temporary.video);
                }
                ++frames;
                if (event) event("relayvsr_restore", frames, video ? r.frames : 1);
            }, cancelled);
        require(!r.frames || frames == r.frames, "RelayVSR source frame count differs from the declaration");
        if (event) event("export", 0, 1);
        if (writer) writer->finish();
        const bool audio = video && r.audio && input.include_audio &&
            relay_media::attach_audio(source, temporary.video, temporary.muxed, cancelled);
        checkpoint(cancelled);
        const auto &ready = audio ? temporary.muxed : temporary.video;
        require(renamex_np(ready.c_str(), path.c_str(), RENAME_EXCL) == 0,
                "Cannot publish RelayVSR output at the requested path");
        if (event) event("export", 1, 1);

        const auto wall = std::chrono::duration<double>(Clock::now() - started).count();
        return native_run_result(@{
            @"schema_version": @1, @"model": @(r.model.c_str()), @"operation": @(r.operation.c_str()),
            @"seed": @(r.seed), @"warmup": @NO,
            @"output": @(path.c_str()), @"width": @(width), @"height": @(height),
            @"frames": @(restored.frames), @"fps": @(timing.fps), @"steps": @1,
            @"audio_output": @(audio), @"residency": @"streamed", @"scale": @2,
            @"backend": @"libtorch_mps_metal",
            @"precision": restored.wan_tensorops ? @"convrot_w8a8" : @"convrot_w8a16",
            @"relayvsr": @{
                @"model_width": @(restored.model_width), @"model_height": @(restored.model_height),
                @"flash_backend": restored.flash_tensorops ? @"tensorops_window" : @"metal_window",
                @"streamed_weight_blocks": @(restored.blocks), @"weight_bytes_read": @(restored.bytes_read),
                @"prefetched_blocks": @(restored.prefetched_blocks), @"synchronous_blocks": @(restored.synchronous_blocks),
                @"prefetch_denied": @(restored.prefetch_denied),
                @"peak_weight_slots": @(restored.peak_weight_slots), @"peak_device_blocks": @(restored.peak_device_blocks),
                @"overlapped_read_ms": @(restored.overlapped_read_ms), @"peak_source_frames": @(restored.peak_source_frames),
                @"tensorops_quantizations": @(restored.quantizations), @"tensorops_projections": @(restored.projections),
                @"tensorops_fused_gelu": @(restored.fused_gelu), @"observed_mps_driver_peak_bytes": @(restored.peak_mps_bytes)
            },
            @"timings_seconds": @{@"request_wall": @(wall)}
        }, r, plan);
    }
};
} // namespace

std::unique_ptr<ModelSession> create_relayvsr(const std::filesystem::path &root) {
    return std::make_unique<RelayVSRSession>(root);
}
} // namespace tc

#include "video.hpp"
#include "asset_track.hpp"

#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <stdexcept>
#include <thread>

namespace tc {
namespace {

void require_status(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

enum class VideoWriterMode { Buffered, Streaming };

struct VideoEncoder {
    AVAssetWriter *writer = nil;
    AVAssetWriterInput *input = nil;
    AVAssetWriterInputPixelBufferAdaptor *adaptor = nil;

    void start(NSString *path, int width, int height, int fps, VideoWriterMode mode) {
        const bool streaming = mode == VideoWriterMode::Streaming;
        NSError *error = nil;
        writer = [[AVAssetWriter alloc] initWithURL:[NSURL fileURLWithPath:path]
                                          fileType:AVFileTypeMPEG4 error:&error];
        require_status(writer != nil, error ? error.localizedDescription.UTF8String :
                       streaming ? "cannot create streaming video writer" : "cannot create video writer");
        NSNumber *bitrate = streaming ? @(std::max<int64_t>(1000000, int64_t(width) * height * fps / 2))
                                     : @(std::max(1000000, width * height * fps / 2));
        NSMutableDictionary *compression = [@{
            AVVideoAverageBitRateKey: bitrate,
            AVVideoAllowFrameReorderingKey: @NO,
        } mutableCopy];
        if (streaming) {
            // Bound GOPs by source frames so slow producers keep a stable cadence.
            compression[AVVideoExpectedSourceFrameRateKey] = @(fps);
            compression[AVVideoMaxKeyFrameIntervalKey] = @(fps * 2);
            compression[AVVideoMaxKeyFrameIntervalDurationKey] = @0.0;
        }
        input = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:@{
            AVVideoCodecKey: AVVideoCodecTypeH264,
            AVVideoWidthKey: @(width),
            AVVideoHeightKey: @(height),
            AVVideoCompressionPropertiesKey: compression,
        }];
        // The default 600-tick timebase cannot represent 1/16 s.
        input.mediaTimeScale = fps;
        writer.movieTimeScale = fps;
        input.expectsMediaDataInRealTime = NO;
        adaptor = [AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:input
            sourcePixelBufferAttributes:@{
                (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
                (NSString *)kCVPixelBufferWidthKey: @(width),
                (NSString *)kCVPixelBufferHeightKey: @(height),
                (NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{},
            }];
        require_status([writer canAddInput:input],
                       streaming ? "unsupported streaming video settings" : "unsupported video settings");
        [writer addInput:input];
        require_status([writer startWriting],
                       streaming ? "streaming video writer start failed" : "video writer start failed");
        [writer startSessionAtSourceTime:kCMTimeZero];
    }
};

void fill_bgra_pixel_buffer(CVPixelBufferRef buffer, const uint8_t *rgb, int width, int height) {
    auto *destination = static_cast<uint8_t *>(CVPixelBufferGetBaseAddress(buffer));
    const size_t stride = CVPixelBufferGetBytesPerRow(buffer);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto *pixel = rgb + (size_t(y) * width + x) * 3;
            auto *out = destination + size_t(y) * stride + size_t(x) * 4;
            out[0] = pixel[2]; out[1] = pixel[1]; out[2] = pixel[0]; out[3] = 255;
        }
    }
}

}  // namespace

void write_video_rgb24(const std::filesystem::path& output,
                       const uint8_t* frames,
                       int frame_count,
                       int width,
                       int height,
                       int fps) {
    require_status(frames && frame_count > 0 && width > 0 && height > 0 && fps > 0,
                   "invalid video export geometry");
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());
    std::filesystem::path temporary_path;
    try {
    @autoreleasepool {
        temporary_path = output;
        temporary_path += "." + std::string(NSUUID.UUID.UUIDString.UTF8String) + ".mp4";
        VideoEncoder encoder;
        encoder.start([NSString stringWithUTF8String:temporary_path.c_str()],
                      width, height, fps, VideoWriterMode::Buffered);
        AVAssetWriter *writer = encoder.writer;
        AVAssetWriterInput *video = encoder.input;
        AVAssetWriterInputPixelBufferAdaptor *adaptor = encoder.adaptor;

        dispatch_queue_t queue = dispatch_queue_create(
            "org.turbocider.native-video-export", DISPATCH_QUEUE_SERIAL);
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        __block int index = 0;
        [video requestMediaDataWhenReadyOnQueue:queue usingBlock:^{
            while (video.readyForMoreMediaData && index < frame_count &&
                   writer.status == AVAssetWriterStatusWriting) {
                CVPixelBufferRef buffer = nullptr;
                if (CVPixelBufferPoolCreatePixelBuffer(nullptr,
                        adaptor.pixelBufferPool, &buffer) != kCVReturnSuccess) {
                    [writer cancelWriting];
                    break;
                }
                CVPixelBufferLockBaseAddress(buffer, 0);
                const uint8_t* source = frames +
                    static_cast<size_t>(index) * width * height * 3;
                fill_bgra_pixel_buffer(buffer, source, width, height);
                CVPixelBufferUnlockBaseAddress(buffer, 0);
                BOOL appended = [adaptor appendPixelBuffer:buffer
                                      withPresentationTime:CMTimeMake(index, fps)];
                CFRelease(buffer);
                if (!appended) {
                    [writer cancelWriting];
                    break;
                }
                ++index;
            }
            if (index == frame_count || writer.status != AVAssetWriterStatusWriting) {
                [video markAsFinished];
                dispatch_semaphore_signal(done);
            }
        }];
        require_status(dispatch_semaphore_wait(
            done, dispatch_time(DISPATCH_TIME_NOW, 180 * NSEC_PER_SEC)) == 0,
            "video writer timed out");
        require_status(index == frame_count &&
                       writer.status == AVAssetWriterStatusWriting,
                       "video writer did not accept all frames");
        [writer endSessionAtSourceTime:CMTimeMake(frame_count, fps)];
        dispatch_semaphore_t finalized = dispatch_semaphore_create(0);
        [writer finishWritingWithCompletionHandler:^{
            dispatch_semaphore_signal(finalized);
        }];
        require_status(dispatch_semaphore_wait(
            finalized, dispatch_time(DISPATCH_TIME_NOW, 180 * NSEC_PER_SEC)) == 0 &&
            writer.status == AVAssetWriterStatusCompleted,
            "video writer finalize failed");
        auto info = probe_video(temporary_path);
        require_status(info.frames == frame_count && info.width == width &&
                       info.height == height && info.fps == fps,
                       "video writer output geometry mismatch");
        std::filesystem::rename(temporary_path, output);
    }
    } catch (...) {
        if (!temporary_path.empty()) {
            std::error_code ignored;
            std::filesystem::remove(temporary_path, ignored);
        }
        throw;
    }
}

struct RGB24VideoWriter::Impl {
    std::filesystem::path output, staging;
    int frames, width, height, fps, next = 0;
    const std::atomic<bool> *cancelled;
    VideoEncoder encoder;
    bool published = false;
    Impl(const std::filesystem::path &path, int count, int w, int h, int rate,
         const std::atomic<bool> *cancel)
        : output(path), frames(count), width(w), height(h), fps(rate), cancelled(cancel) {}
    void check_cancel() const {
        if (cancelled && cancelled->load()) throw Cancelled();
    }
    void start() {
        require_status(frames > 0 && width > 0 && height > 0 && width <= 8192 && height <= 8192 &&
                       fps > 0 && fps <= 120, "invalid streaming video geometry");
        check_cancel();
        if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
        staging = output;
        staging += "." + std::string(NSUUID.UUID.UUIDString.UTF8String) + ".mp4";
        encoder.start(@(staging.c_str()), width, height, fps, VideoWriterMode::Streaming);
    }
    ~Impl() {
        if (!published) {
            if (encoder.writer && encoder.writer.status == AVAssetWriterStatusWriting) [encoder.writer cancelWriting];
            if (!staging.empty()) { std::error_code ignored; std::filesystem::remove(staging,ignored); }
        }
    }
    void append(int index, const uint8_t *rgb, size_t bytes) {
        check_cancel();
        require_status(!published && encoder.writer.status == AVAssetWriterStatusWriting && index == next && next < frames &&
                       rgb && bytes == size_t(width)*height*3, "invalid streaming frame order, size or writer state");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!encoder.input.readyForMoreMediaData) {
            check_cancel();
            require_status(encoder.writer.status == AVAssetWriterStatusWriting, "streaming video writer stopped accepting frames");
            require_status(std::chrono::steady_clock::now() < deadline, "streaming video writer backpressure timed out");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CVPixelBufferRef buffer = nullptr;
        require_status(CVPixelBufferPoolCreatePixelBuffer(nullptr,encoder.adaptor.pixelBufferPool,&buffer) == kCVReturnSuccess,
                       "cannot allocate streaming pixel buffer");
        if (CVPixelBufferLockBaseAddress(buffer,0) != kCVReturnSuccess) {
            CFRelease(buffer);
            throw std::runtime_error("cannot lock streaming pixel buffer");
        }
        fill_bgra_pixel_buffer(buffer, rgb, width, height);
        CVPixelBufferUnlockBaseAddress(buffer,0);
        const BOOL accepted = [encoder.adaptor appendPixelBuffer:buffer withPresentationTime:CMTimeMake(index,fps)];
        CFRelease(buffer);
        require_status(accepted, "streaming video writer append failed");
        ++next;
    }
    void finish() {
        check_cancel();
        require_status(!published && next == frames && encoder.writer.status == AVAssetWriterStatusWriting,
                       "streaming video writer requires the complete declared sequence");
        [encoder.input markAsFinished];
        [encoder.writer endSessionAtSourceTime:CMTimeMake(frames,fps)];
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        [encoder.writer finishWritingWithCompletionHandler:^{ dispatch_semaphore_signal(done); }];
        // Poll a specific live completion handle so cancellation is responsive.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
        while (dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,100*NSEC_PER_MSEC)) != 0) {
            check_cancel();
            require_status(std::chrono::steady_clock::now() < deadline, "streaming video finalization timed out");
        }
        require_status(encoder.writer.status == AVAssetWriterStatusCompleted, "streaming video finalization failed");
        check_cancel();
        const auto info = probe_video(staging);
        require_status(info.frames == frames && info.width == width && info.height == height && info.fps == fps,
                       "streaming video geometry mismatch");
        std::filesystem::rename(staging,output);
        published = true;
    }
};
RGB24VideoWriter::RGB24VideoWriter(const std::filesystem::path &path, int frames, int width, int height, int fps,
                                 const std::atomic<bool> *cancelled)
    : impl_(std::make_unique<Impl>(path,frames,width,height,fps,cancelled)) {
    @autoreleasepool { impl_->start(); }
}
RGB24VideoWriter::~RGB24VideoWriter() = default;
void RGB24VideoWriter::append(int index, const uint8_t *rgb, size_t bytes) {
    @autoreleasepool { impl_->append(index,rgb,bytes); }
}
void RGB24VideoWriter::finish() { @autoreleasepool { impl_->finish(); } }
int RGB24VideoWriter::frames_written() const { return impl_->next; }

VideoMediaInfo probe_video(const std::filesystem::path& path) {
    @autoreleasepool {
        AVURLAsset* asset = [AVURLAsset URLAssetWithURL:
            [NSURL fileURLWithPath:@(path.c_str())] options:nil];
        AVAssetTrack* track = media::load_first_track_or_throw(
            asset, AVMediaTypeVideo, "timed out while loading video tracks");
        require_status(track != nil, "video stream is missing");
        Float64 duration = CMTimeGetSeconds(asset.duration);
        float rate = track.nominalFrameRate;
        int frames = duration > 0 && rate > 0 ?
            static_cast<int>(std::llround(duration * rate)) : 0;
        return {static_cast<int>(track.naturalSize.width),
                static_cast<int>(track.naturalSize.height), frames,
                rate > 0 ? static_cast<int>(std::llround(rate)) : 0};
    }
}

}  // namespace tc

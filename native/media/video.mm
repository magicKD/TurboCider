#include "video.hpp"
#include "../core/common.hpp"

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

AVAssetTrack* load_first_track(AVAsset* asset, AVMediaType media_type) {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block NSArray<AVAssetTrack*>* tracks = nil;
    __block NSError* load_error = nil;
    [asset loadTracksWithMediaType:media_type
                 completionHandler:^(NSArray<AVAssetTrack*>* loaded,
                                     NSError* error) {
        tracks = loaded;
        load_error = error;
        dispatch_semaphore_signal(done);
    }];
    require_status(dispatch_semaphore_wait(
        done, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC)) == 0,
        "timed out while loading video tracks");
    require_status(load_error == nil,
                   load_error ? load_error.localizedDescription.UTF8String :
                                "cannot load video tracks");
    return tracks.firstObject;
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
        NSError* error = nil;
        temporary_path = output;
        temporary_path += "." + std::string(NSUUID.UUID.UUIDString.UTF8String) + ".mp4";
        NSString* temporary = [NSString stringWithUTF8String:temporary_path.c_str()];
        AVAssetWriter* writer = [[AVAssetWriter alloc]
            initWithURL:[NSURL fileURLWithPath:temporary]
            fileType:AVFileTypeMPEG4
            error:&error];
        require_status(writer != nil, error ? error.localizedDescription.UTF8String :
                       "cannot create video writer");
        NSDictionary* settings = @{
            AVVideoCodecKey: AVVideoCodecTypeH264,
            AVVideoWidthKey: @(width),
            AVVideoHeightKey: @(height),
            AVVideoCompressionPropertiesKey: @{
                AVVideoAverageBitRateKey: @(std::max(1000000, width * height * fps / 2)),
                AVVideoAllowFrameReorderingKey: @NO,
            },
        };
        AVAssetWriterInput* video = [AVAssetWriterInput
            assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:settings];
        // AVFoundation's default 600-tick timebase cannot represent 1/16 s.
        // Use one exact frame per media tick rather than rounding frame times.
        video.mediaTimeScale = fps;
        writer.movieTimeScale = fps;
        video.expectsMediaDataInRealTime = NO;
        NSDictionary* attributes = @{
            (NSString*)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
            (NSString*)kCVPixelBufferWidthKey: @(width),
            (NSString*)kCVPixelBufferHeightKey: @(height),
            (NSString*)kCVPixelBufferIOSurfacePropertiesKey: @{},
        };
        AVAssetWriterInputPixelBufferAdaptor* adaptor =
            [AVAssetWriterInputPixelBufferAdaptor
                assetWriterInputPixelBufferAdaptorWithAssetWriterInput:video
                sourcePixelBufferAttributes:attributes];
        require_status([writer canAddInput:video], "unsupported video settings");
        [writer addInput:video];
        require_status([writer startWriting], "video writer start failed");
        [writer startSessionAtSourceTime:kCMTimeZero];

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
                auto* destination = static_cast<uint8_t*>(
                    CVPixelBufferGetBaseAddress(buffer));
                size_t stride = CVPixelBufferGetBytesPerRow(buffer);
                const uint8_t* source = frames +
                    static_cast<size_t>(index) * width * height * 3;
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        const uint8_t* pixel = source +
                            (static_cast<size_t>(y) * width + x) * 3;
                        uint8_t* out = destination + static_cast<size_t>(y) * stride +
                            static_cast<size_t>(x) * 4;
                        out[0] = pixel[2];
                        out[1] = pixel[1];
                        out[2] = pixel[0];
                        out[3] = 255;
                    }
                }
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
    AVAssetWriter *writer = nil;
    AVAssetWriterInput *video = nil;
    AVAssetWriterInputPixelBufferAdaptor *adaptor = nil;
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
        NSError *error = nil;
        writer = [[AVAssetWriter alloc] initWithURL:[NSURL fileURLWithPath:@(staging.c_str())]
                                         fileType:AVFileTypeMPEG4 error:&error];
        require_status(writer != nil, error ? error.localizedDescription.UTF8String : "cannot create streaming video writer");
        NSDictionary *settings = @{
            AVVideoCodecKey: AVVideoCodecTypeH264, AVVideoWidthKey: @(width), AVVideoHeightKey: @(height),
            AVVideoCompressionPropertiesKey: @{
                AVVideoAverageBitRateKey: @(std::max<int64_t>(1000000,int64_t(width)*height*fps/2)),
                // Slow offline producers must not inherit a wall-time-driven
                // automatic keyframe cadence. Bound GOPs by source frames,
                // disable a duration limit and declare expected source FPS.
                AVVideoExpectedSourceFrameRateKey: @(fps),
                AVVideoMaxKeyFrameIntervalKey: @(fps*2),
                AVVideoMaxKeyFrameIntervalDurationKey: @0.0,
                AVVideoAllowFrameReorderingKey: @NO,
            },
        };
        video = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:settings];
        video.mediaTimeScale = fps;
        writer.movieTimeScale = fps;
        video.expectsMediaDataInRealTime = NO;
        adaptor = [AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:video
            sourcePixelBufferAttributes:@{
                (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
                (NSString *)kCVPixelBufferWidthKey: @(width), (NSString *)kCVPixelBufferHeightKey: @(height),
                (NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{},
            }];
        require_status([writer canAddInput:video], "unsupported streaming video settings");
        [writer addInput:video];
        require_status([writer startWriting], "streaming video writer start failed");
        [writer startSessionAtSourceTime:kCMTimeZero];
    }
    ~Impl() {
        if (!published) {
            if (writer && writer.status == AVAssetWriterStatusWriting) [writer cancelWriting];
            if (!staging.empty()) { std::error_code ignored; std::filesystem::remove(staging,ignored); }
        }
    }
    void append(int index, const uint8_t *rgb, size_t bytes) {
        check_cancel();
        require_status(!published && writer.status == AVAssetWriterStatusWriting && index == next && next < frames &&
                       rgb && bytes == size_t(width)*height*3, "invalid streaming frame order, size or writer state");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!video.readyForMoreMediaData) {
            check_cancel();
            require_status(writer.status == AVAssetWriterStatusWriting, "streaming video writer stopped accepting frames");
            require_status(std::chrono::steady_clock::now() < deadline, "streaming video writer backpressure timed out");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CVPixelBufferRef buffer = nullptr;
        require_status(CVPixelBufferPoolCreatePixelBuffer(nullptr,adaptor.pixelBufferPool,&buffer) == kCVReturnSuccess,
                       "cannot allocate streaming pixel buffer");
        if (CVPixelBufferLockBaseAddress(buffer,0) != kCVReturnSuccess) {
            CFRelease(buffer);
            throw std::runtime_error("cannot lock streaming pixel buffer");
        }
        auto *destination = static_cast<uint8_t *>(CVPixelBufferGetBaseAddress(buffer));
        const size_t stride = CVPixelBufferGetBytesPerRow(buffer);
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            const auto *pixel = rgb + (size_t(y)*width+x)*3;
            auto *out = destination + size_t(y)*stride + size_t(x)*4;
            out[0] = pixel[2]; out[1] = pixel[1]; out[2] = pixel[0]; out[3] = 255;
        }
        CVPixelBufferUnlockBaseAddress(buffer,0);
        const BOOL accepted = [adaptor appendPixelBuffer:buffer withPresentationTime:CMTimeMake(index,fps)];
        CFRelease(buffer);
        require_status(accepted, "streaming video writer append failed");
        ++next;
    }
    void finish() {
        check_cancel();
        require_status(!published && next == frames && writer.status == AVAssetWriterStatusWriting,
                       "streaming video writer requires the complete declared sequence");
        [video markAsFinished];
        [writer endSessionAtSourceTime:CMTimeMake(frames,fps)];
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        [writer finishWritingWithCompletionHandler:^{ dispatch_semaphore_signal(done); }];
        // Poll a specific live completion handle so cancellation is responsive.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
        while (dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,100*NSEC_PER_MSEC)) != 0) {
            check_cancel();
            require_status(std::chrono::steady_clock::now() < deadline, "streaming video finalization timed out");
        }
        require_status(writer.status == AVAssetWriterStatusCompleted, "streaming video finalization failed");
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
        AVAssetTrack* track = load_first_track(asset, AVMediaTypeVideo);
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

#include "video.hpp"

#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <stdexcept>

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

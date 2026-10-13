#include "video_input.hpp"
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#include <cmath>

namespace tc {
namespace {
bool near(double value, double target) { return std::abs(value - target) < 1e-4; }
}
Tensor load_video_tensor(const std::filesystem::path &path, int expected_frames,
                         int expected_fps, const Event &event, std::atomic<bool> &cancelled) {
    require(expected_frames >= 1 && expected_frames <= 121 && expected_fps >= 1 && expected_fps <= 120,
            "invalid declared video geometry");
    require(std::filesystem::is_regular_file(path), "video input is missing: " + path.string());
    checkpoint(cancelled);
    @autoreleasepool {
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:@(path.c_str())] options:nil];
        dispatch_semaphore_t ready = dispatch_semaphore_create(0);
        __block NSArray<AVAssetTrack *> *tracks = nil;
        __block NSError *track_error = nil;
        [asset loadTracksWithMediaType:AVMediaTypeVideo completionHandler:^(NSArray<AVAssetTrack *> *value, NSError *error) {
            tracks = value;
            track_error = error;
            dispatch_semaphore_signal(ready);
        }];
        require(dispatch_semaphore_wait(ready, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC)) == 0,
                "timed out loading video input tracks");
        require(track_error == nil && tracks.count > 0, "video input has no readable video track");
        AVAssetTrack *track = tracks.firstObject;
        require(near(track.nominalFrameRate, expected_fps),
                "declared FPS differs from source; SeedVR2 does not silently resample video");
        NSError *error = nil;
        AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
        require(reader != nil, error ? error.localizedDescription.UTF8String : "cannot create video reader");
        AVAssetReaderTrackOutput *output = [[AVAssetReaderTrackOutput alloc] initWithTrack:track outputSettings:@{
            (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)
        }];
        output.alwaysCopiesSampleData = NO;
        require([reader canAddOutput:output], "cannot decode the source video format");
        [reader addOutput:output];
        require([reader startReading], "cannot start video decoding");
        std::vector<uint8_t> pixels;
        int width = 0, height = 0, frames = 0;
        double first_time = 0;
        try {
            while (true) {
                checkpoint(cancelled);
                CMSampleBufferRef sample = [output copyNextSampleBuffer];
                if (!sample) break;
                CVPixelBufferRef buffer = CMSampleBufferGetImageBuffer(sample);
                bool locked = false;
                try {
                    require(frames < expected_frames, "source has more frames than declared; no implicit truncation is allowed");
                    require(buffer != nullptr, "video decoder produced no image buffer");
                    const int w = int(CVPixelBufferGetWidth(buffer)), h = int(CVPixelBufferGetHeight(buffer));
                    require(w > 0 && h > 0 && uint64_t(w) * h <= 16'777'216,
                            "video frame exceeds native import limits");
                    if (frames == 0) {
                        width = w; height = h;
                        const uint64_t total = uint64_t(width) * height * expected_frames * 3;
                        require(total <= (2ull << 30), "decoded source video exceeds 2 GiB import limit");
                        pixels.reserve(size_t(total));
                    }
                    require(w == width && h == height, "source video changes frame dimensions");
                    const double time = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample));
                    if (frames == 0) first_time = time;
                    require(std::isfinite(time) && std::abs(time - first_time - double(frames) / expected_fps) < 1e-4,
                            "variable frame-rate input is not supported; normalize the source explicitly");
                    require(CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly) == kCVReturnSuccess,
                            "cannot access decoded video pixels");
                    locked = true;
                    const auto *data = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddress(buffer));
                    const auto stride = CVPixelBufferGetBytesPerRow(buffer);
                    require(data != nullptr && stride >= size_t(width) * 4, "invalid video pixel storage");
                    const size_t offset = pixels.size();
                    pixels.resize(offset + size_t(width) * height * 3);
                    for (int y = 0; y < height; ++y)
                        for (int x = 0; x < width; ++x) {
                            const auto *source = data + size_t(y) * stride + size_t(x) * 4;
                            auto *target = pixels.data() + offset + (size_t(y) * width + x) * 3;
                            target[0] = source[2]; target[1] = source[1]; target[2] = source[0];
                        }
                    CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
                    locked = false;
                    ++frames;
                    if (event) event("seedvr2_video_import", frames, expected_frames);
                    CFRelease(sample);
                } catch (...) {
                    if (locked) CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
                    CFRelease(sample);
                    throw;
                }
            }
            require(reader.status == AVAssetReaderStatusCompleted,
                    reader.error ? reader.error.localizedDescription.UTF8String : "video decode did not complete");
            require(frames == expected_frames, "decoded source frame count differs from declaration");
        } catch (...) {
            [reader cancelReading];
            throw;
        }
        auto result = mx::astype(Tensor(pixels.data(), {frames, height, width, 3}, mx::uint8), mx::float32) / 255.f;
        const auto transform = track.preferredTransform;
        // Apply standard rotation/reflection metadata to the decoded frames.
        if (near(transform.a, 0) && near(transform.d, 0)) {
            require(near(std::abs(transform.b), 1) && near(std::abs(transform.c), 1),
                    "unsupported video display transform");
            result = mx::transpose(result, {0, 2, 1, 3});
            if (transform.b > 0) result = mx::flip(result, 2);
            if (transform.c > 0) result = mx::flip(result, 1);
        } else {
            require(near(transform.b, 0) && near(transform.c, 0) &&
                    near(std::abs(transform.a), 1) && near(std::abs(transform.d), 1),
                    "unsupported video display transform");
            if (transform.a < 0) result = mx::flip(result, 2);
            if (transform.d < 0) result = mx::flip(result, 1);
        }
        mx::eval(result);
        checkpoint(cancelled);
        return result;
    }
}
} // namespace tc

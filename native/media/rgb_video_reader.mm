#include "rgb_video_reader.hpp"
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#include <cmath>

namespace tc {
namespace {
bool near(double value, double target) { return std::abs(value-target) < 1e-4; }
}
struct RGB24VideoReader::Impl {
    std::filesystem::path path;
    int expected_frames, fps, frames = 0, source_width = 0, source_height = 0;
    Event event;
    std::atomic<bool> &cancelled;
    AVURLAsset *asset = nil;
    AVAssetReader *reader = nil;
    AVAssetReaderTrackOutput *output = nil;
    CGAffineTransform transform = CGAffineTransformIdentity;
    bool rotated = false, finished = false, failed = false;
    double first_time = 0;
    uintmax_t source_bytes = 0;
    std::filesystem::file_time_type source_modified;
    Impl(const std::filesystem::path &p, int count, int rate, const Event &callback, std::atomic<bool> &cancel)
        : path(p),expected_frames(count),fps(rate),event(callback),cancelled(cancel) {}
    void start() {
        require(expected_frames >= 1 && expected_frames <= 1'000'000 && fps >= 1 && fps <= 120,
                "invalid declared streaming video geometry");
        require(std::filesystem::is_regular_file(path),"video input is missing: "+path.string());
        source_bytes = std::filesystem::file_size(path);
        source_modified = std::filesystem::last_write_time(path);
        checkpoint(cancelled);
        asset = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:@(path.c_str())] options:nil];
        dispatch_semaphore_t ready = dispatch_semaphore_create(0);
        __block NSArray<AVAssetTrack *> *tracks = nil;
        __block NSError *track_error = nil;
        [asset loadTracksWithMediaType:AVMediaTypeVideo completionHandler:^(NSArray<AVAssetTrack *> *value, NSError *error) {
            tracks = value; track_error = error; dispatch_semaphore_signal(ready);
        }];
        const auto deadline = Clock::now()+std::chrono::seconds(30);
        while (dispatch_semaphore_wait(ready,dispatch_time(DISPATCH_TIME_NOW,50*NSEC_PER_MSEC))) {
            checkpoint(cancelled);
            require(Clock::now()<deadline,"timed out loading video input tracks");
        }
        checkpoint(cancelled);
        require(track_error == nil && tracks.count > 0,"video input has no readable video track");
        AVAssetTrack *track = tracks.firstObject;
        require(near(track.nominalFrameRate,fps),"declared FPS differs from source; no implicit resampling");
        transform = track.preferredTransform;
        rotated = near(transform.a,0) && near(transform.d,0);
        if (rotated) require(near(std::abs(transform.b),1) && near(std::abs(transform.c),1),"unsupported video display transform");
        else require(near(transform.b,0) && near(transform.c,0) && near(std::abs(transform.a),1) && near(std::abs(transform.d),1),
                     "unsupported video display transform");
        NSError *error = nil;
        reader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
        require(reader != nil,error ? error.localizedDescription.UTF8String : "cannot create video reader");
        output = [[AVAssetReaderTrackOutput alloc] initWithTrack:track outputSettings:@{
            (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)}];
        output.alwaysCopiesSampleData = NO;
        require([reader canAddOutput:output],"cannot decode source video format");
        [reader addOutput:output];
        require([reader startReading],"cannot start video decoding");
    }
    ~Impl() { if (reader.status == AVAssetReaderStatusReading) [reader cancelReading]; }
    bool next(RGBVideoFrame &frame) {
        checkpoint(cancelled);
        require(!failed,"video reader cannot resume after a failed frame");
        if (finished) return false;
        CMSampleBufferRef sample = [output copyNextSampleBuffer];
        if (!sample) {
            require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) == source_bytes &&
                    std::filesystem::last_write_time(path) == source_modified,"video source changed during streaming decode");
            if (reader.status != AVAssetReaderStatusCompleted || frames != expected_frames) {
                failed = true;
                require(reader.status == AVAssetReaderStatusCompleted,
                        reader.error ? reader.error.localizedDescription.UTF8String : "video decode did not complete");
                require(frames == expected_frames,"decoded source frame count differs from declaration");
            }
            finished = true;
            return false;
        }
        CVPixelBufferRef buffer = CMSampleBufferGetImageBuffer(sample);
        bool locked = false;
        try {
            require(frames < expected_frames,"source has more frames than declared; no implicit truncation");
            require(buffer != nullptr,"video decoder produced no image buffer");
            const int w = int(CVPixelBufferGetWidth(buffer)), h = int(CVPixelBufferGetHeight(buffer));
            require(w > 0 && h > 0 && uint64_t(w)*h <= 16'777'216,"video frame exceeds native import limits");
            if (frames == 0) { source_width = w; source_height = h; }
            require(w == source_width && h == source_height,"source video changes frame dimensions");
            const double time = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample));
            if (frames == 0) first_time = time;
            require(std::isfinite(time) && near(time-first_time,double(frames)/fps),
                    "variable frame-rate input is not supported; normalize source explicitly");
            require(CVPixelBufferLockBaseAddress(buffer,kCVPixelBufferLock_ReadOnly) == kCVReturnSuccess,"cannot access decoded video pixels");
            locked = true;
            const auto *data = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddress(buffer));
            const size_t stride = CVPixelBufferGetBytesPerRow(buffer);
            require(data && stride >= size_t(w)*4,"invalid decoded video pixel storage");
            frame.index = frames; frame.width = rotated ? h : w; frame.height = rotated ? w : h;
            frame.pixels.resize(size_t(frame.width)*frame.height*3);
            for (int y = 0; y < frame.height; ++y) {
                if (y%32 == 0) checkpoint(cancelled);
                for (int x = 0; x < frame.width; ++x) {
                    const int sx = rotated ? (transform.c > 0 ? w-1-y : y) : (transform.a < 0 ? w-1-x : x);
                    const int sy = rotated ? (transform.b > 0 ? h-1-x : x) : (transform.d < 0 ? h-1-y : y);
                    const auto *pixel = data+size_t(sy)*stride+size_t(sx)*4;
                    auto *target = frame.pixels.data()+(size_t(y)*frame.width+x)*3;
                    target[0] = pixel[2]; target[1] = pixel[1]; target[2] = pixel[0];
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer,kCVPixelBufferLock_ReadOnly); locked = false;
            CFRelease(sample); sample = nullptr;
            ++frames;
            if (event) event("seedvr2_video_import",frames,expected_frames);
            checkpoint(cancelled);
            return true;
        } catch (...) {
            if (locked) CVPixelBufferUnlockBaseAddress(buffer,kCVPixelBufferLock_ReadOnly);
            if (sample) CFRelease(sample);
            failed = true;
            [reader cancelReading];
            throw;
        }
    }
};
RGB24VideoReader::RGB24VideoReader(const std::filesystem::path &path, int frames, int fps,
                                 const Event &event, std::atomic<bool> &cancelled)
    : impl_(std::make_unique<Impl>(path,frames,fps,event,cancelled)) { @autoreleasepool { impl_->start(); } }
RGB24VideoReader::~RGB24VideoReader() = default;
bool RGB24VideoReader::next(RGBVideoFrame &frame) { @autoreleasepool { return impl_->next(frame); } }
int RGB24VideoReader::frames_read() const { return impl_->frames; }
} // namespace tc

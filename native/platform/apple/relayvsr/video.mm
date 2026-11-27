#include "video.hpp"
#include "../../../media/asset_track.hpp"

#import <CoreVideo/CoreVideo.h>

#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <thread>
#include <utility>

namespace tc::relay_media {
namespace {
bool near(double value, double expected) { return std::abs(value - expected) < 1e-4; }

template<class Ref>
struct OwnedCF {
    Ref value = nullptr;
    ~OwnedCF() { if (value) CFRelease(value); }
};

struct LockedPixels {
    CVPixelBufferRef buffer;
    CVPixelBufferLockFlags flags;
    LockedPixels(CVPixelBufferRef value, CVPixelBufferLockFlags mode) : buffer(value), flags(mode) {
        require(CVPixelBufferLockBaseAddress(buffer, flags) == kCVReturnSuccess,
                "Cannot lock RelayVSR video pixels");
    }
    ~LockedPixels() { CVPixelBufferUnlockBaseAddress(buffer, flags); }
};

int64_t exact_ticks(CMTime time, int scale) {
    require(CMTIME_IS_NUMERIC(time) && time.timescale > 0, "Invalid source video timing");
    const auto ticks = CMTimeConvertScale(time, scale, kCMTimeRoundingMethod_Default);
    require(CMTIME_IS_NUMERIC(ticks) && CMTimeCompare(ticks, time) == 0,
            "Source timing cannot be represented exactly by the video timebase");
    return ticks.value;
}

void include_time(int &scale, CMTime time) {
    if (!CMTIME_IS_NUMERIC(time) || time.timescale <= 0) return;
    auto reduced = time.timescale / std::gcd(time.value % time.timescale, int64_t(time.timescale));
    const auto combined = std::lcm<int64_t>(scale, reduced);
    require(combined <= std::numeric_limits<int32_t>::max(), "Unsupported source video timebase");
    scale = static_cast<int>(combined);
}

void copy_rgb(const uint8_t *bgra, uint8_t *rgb, int width) {
    int x = 0;
    for (; x + 16 <= width; x += 16) {
        const auto pixels = vld4q_u8(bgra + x * 4);
        const uint8x16x3_t rgb_chunk{pixels.val[2], pixels.val[1], pixels.val[0]};
        vst3q_u8(rgb + x * 3, rgb_chunk);
    }
    for (; x < width; ++x) {
        rgb[x * 3] = bgra[x * 4 + 2];
        rgb[x * 3 + 1] = bgra[x * 4 + 1];
        rgb[x * 3 + 2] = bgra[x * 4];
    }
}

void copy_bgra(const uint8_t *rgb, uint8_t *bgra, int width) {
    int x = 0;
    const auto alpha = vdupq_n_u8(255);
    for (; x + 16 <= width; x += 16) {
        const auto pixels = vld3q_u8(rgb + x * 3);
        const uint8x16x4_t bgra_chunk{pixels.val[2], pixels.val[1], pixels.val[0], alpha};
        vst4q_u8(bgra + x * 4, bgra_chunk);
    }
    for (; x < width; ++x) {
        bgra[x * 4] = rgb[x * 3 + 2];
        bgra[x * 4 + 1] = rgb[x * 3 + 1];
        bgra[x * 4 + 2] = rgb[x * 3];
        bgra[x * 4 + 3] = 255;
    }
}

void wait_for(dispatch_semaphore_t done, std::atomic<bool> &cancelled, const char *timeout) {
    const auto deadline = Clock::now() + std::chrono::minutes(3);
    while (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 50 * NSEC_PER_MSEC))) {
        checkpoint(cancelled);
        require(Clock::now() < deadline, timeout);
    }
    checkpoint(cancelled);
}
} // namespace

struct VideoReader::Impl {
    std::filesystem::path source;
    std::atomic<bool> &cancelled;
    uintmax_t source_bytes;
    std::filesystem::file_time_type source_mtime;
    VideoTiming timing;
    AVAssetTrack *track = nil;
    AVAssetReader *reader = nil;
    AVAssetReaderTrackOutput *output = nil;
    OwnedCF<CMSampleBufferRef> pending;
    NSArray<AVAssetTrackSegment *> *segments = nil;
    CGAffineTransform transform = CGAffineTransformIdentity;
    CMTime first_media = kCMTimeInvalid, end = kCMTimeInvalid;
    bool rotated = false, finished = false;

    Impl(const std::filesystem::path &path, std::atomic<bool> &cancel)
        : source(path), cancelled(cancel), source_bytes(std::filesystem::file_size(path)),
          source_mtime(std::filesystem::last_write_time(path)) {
        auto asset = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:@(path.c_str())]
                                         options:@{AVURLAssetPreferPreciseDurationAndTimingKey: @YES}];
        auto loaded = media::load_first_track(asset, AVMediaTypeVideo, &cancelled);
        require(loaded.track && !loaded.error && !loaded.timed_out, "Cannot load the RelayVSR source video track");
        track = loaded.track;
        transform = track.preferredTransform;
        rotated = near(transform.a, 0) && near(transform.d, 0);
        require(rotated ? near(std::abs(transform.b), 1) && near(std::abs(transform.c), 1)
                        : near(transform.b, 0) && near(transform.c, 0) &&
                          near(std::abs(transform.a), 1) && near(std::abs(transform.d), 1),
                "Unsupported source video display transform");

        timing.fps = track.nominalFrameRate;
        if (!std::isfinite(timing.fps) || timing.fps <= 0) {
            const auto step = CMTimeGetSeconds(track.minFrameDuration);
            timing.fps = std::isfinite(step) && step > 0 ? 1.0 / step : 30.0;
        }
        const auto range = track.timeRange;
        end = CMTimeRangeGetEnd(range);
        timing.scale = std::max(1, track.naturalTimeScale);
        include_time(timing.scale, range.start);
        include_time(timing.scale, range.duration);
        segments = track.segments;
        for (AVAssetTrackSegment *segment in segments) {
            const auto target = segment.timeMapping.target;
            include_time(timing.scale, target.start);
            include_time(timing.scale, target.duration);
            if (segments.firstObject.isEmpty && !segment.isEmpty && !CMTIME_IS_VALID(first_media))
                first_media = target.start;
        }
        const auto format = (__bridge CMFormatDescriptionRef)track.formatDescriptions.firstObject;
        if (format) {
            const auto ratio = (__bridge NSDictionary *)CMFormatDescriptionGetExtension(
                format, kCMFormatDescriptionExtension_PixelAspectRatio);
            auto h = [ratio[(__bridge NSString *)kCMFormatDescriptionKey_PixelAspectRatioHorizontalSpacing] intValue];
            auto v = [ratio[(__bridge NSString *)kCMFormatDescriptionKey_PixelAspectRatioVerticalSpacing] intValue];
            if (h > 0 && v > 0) {
                const auto divisor = std::gcd(h, v);
                if (rotated) std::swap(h, v);
                timing.pixel_aspect_width = h / divisor;
                timing.pixel_aspect_height = v / divisor;
            }
        }
        NSError *error = nil;
        reader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
        require(reader != nil, error ? error.localizedDescription.UTF8String : "Cannot create RelayVSR video reader");
        output = [[AVAssetReaderTrackOutput alloc] initWithTrack:track outputSettings:@{
            (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)
        }];
        output.alwaysCopiesSampleData = NO;
        require([reader canAddOutput:output], "Unsupported RelayVSR video input format");
        [reader addOutput:output];
        require([reader startReading], "Cannot start RelayVSR video decoding");
    }

    ~Impl() { if (reader.status == AVAssetReaderStatusReading) [reader cancelReading]; }

    CMTime last_duration(CMTime pts, CMTime fallback) {
        for (AVAssetTrackSegment *segment in segments) {
            const auto map = segment.timeMapping;
            if (segment.isEmpty || !CMTimeRangeContainsTime(map.target, pts)) continue;
            const auto timestamp = CMTimeMapTimeFromRangeToRange(pts, map.target, map.source);
            auto cursor = [track makeSampleCursorWithPresentationTimeStamp:timestamp];
            if (cursor && CMTimeCompare(cursor.presentationTimeStamp, timestamp) == 0 &&
                CMTIME_IS_NUMERIC(cursor.currentSampleDuration) && cursor.currentSampleDuration.value > 0)
                return CMTimeMapDurationFromRangeToRange(cursor.currentSampleDuration, map.source, map.target);
            break;
        }
        if (CMTIME_IS_NUMERIC(fallback) && fallback.value > 0) return fallback;
        return CMTimeSubtract(end, pts);
    }

    bool next(rv::Frame &frame) {
        checkpoint(cancelled);
        if (finished) return false;
        OwnedCF<CMSampleBufferRef> sample{std::exchange(pending.value, nullptr)};
        if (!sample.value) sample.value = [output copyNextSampleBuffer];
        // AVFoundation produces synthetic black samples for leading empty edits.
        while (sample.value && CMTIME_IS_NUMERIC(first_media) &&
               CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(sample.value), first_media) < 0) {
            CFRelease(sample.value);
            sample.value = [output copyNextSampleBuffer];
        }
        first_media = kCMTimeInvalid;
        if (!sample.value) {
            require(reader.status == AVAssetReaderStatusCompleted,
                    reader.error ? reader.error.localizedDescription.UTF8String : "RelayVSR video decode did not complete");
            require(std::filesystem::file_size(source) == source_bytes &&
                    std::filesystem::last_write_time(source) == source_mtime,
                    "RelayVSR video source changed during streaming decode");
            finished = true;
            return false;
        }

        auto pixels = CMSampleBufferGetImageBuffer(sample.value);
        require(pixels != nullptr, "Video decoder returned no pixels");
        const int width = int(CVPixelBufferGetWidth(pixels)), height = int(CVPixelBufferGetHeight(pixels));
        require(width > 0 && height > 0 && width <= rv::kMaxInputSide && height <= rv::kMaxInputSide,
                "RelayVSR 2× supports source sides up to 2048 pixels");
        frame.width = rotated ? height : width;
        frame.height = rotated ? width : height;
        auto pts = CMSampleBufferGetPresentationTimeStamp(sample.value);
        frame.pts = exact_ticks(pts, timing.scale);
        pending.value = [output copyNextSampleBuffer];
        auto duration = pending.value ? CMTimeSubtract(CMSampleBufferGetPresentationTimeStamp(pending.value), pts)
                                      : last_duration(pts, CMSampleBufferGetDuration(sample.value));
        frame.duration = exact_ticks(duration, timing.scale);
        require(frame.pts >= 0 && frame.duration > 0 &&
                frame.pts <= std::numeric_limits<int64_t>::max() - frame.duration,
                "Invalid RelayVSR source frame timing");

        frame.rgb.resize(size_t(width) * height * 3);
        LockedPixels lock(pixels, kCVPixelBufferLock_ReadOnly);
        const auto *base = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddress(pixels));
        const auto stride = CVPixelBufferGetBytesPerRow(pixels);
        require(base && stride >= size_t(width) * 4, "Invalid source video pixel storage");
        for (int y = 0; y < frame.height; ++y) {
            auto *rgb = frame.rgb.data() + size_t(y) * frame.width * 3;
            if (!rotated && transform.a > 0) {
                const auto sy = transform.d < 0 ? height - 1 - y : y;
                copy_rgb(base + size_t(sy) * stride, rgb, width);
                continue;
            }
            for (int x = 0; x < frame.width; ++x) {
                const auto sx = rotated ? (transform.c > 0 ? width - 1 - y : y) : width - 1 - x;
                const auto sy = rotated ? (transform.b > 0 ? height - 1 - x : x) :
                                          (transform.d < 0 ? height - 1 - y : y);
                const auto *bgra = base + size_t(sy) * stride + sx * 4;
                rgb[x * 3] = bgra[2];
                rgb[x * 3 + 1] = bgra[1];
                rgb[x * 3 + 2] = bgra[0];
            }
        }
        return true;
    }
};

VideoReader::VideoReader(const std::filesystem::path &path, std::atomic<bool> &cancelled) {
    @autoreleasepool { impl_ = std::make_unique<Impl>(path, cancelled); }
}
VideoReader::~VideoReader() = default;
bool VideoReader::next(rv::Frame &frame) { @autoreleasepool { return impl_->next(frame); } }
VideoTiming VideoReader::timing() const { return impl_->timing; }

struct VideoWriter::Impl {
    const int width, height;
    const VideoTiming timing;
    std::atomic<bool> &cancelled;
    AVAssetWriter *writer = nil;
    AVAssetWriterInput *input = nil;
    AVAssetWriterInputPixelBufferAdaptor *adaptor = nil;
    int64_t last_pts = -1, last_end = 0;

    Impl(const std::filesystem::path &path, int w, int h, VideoTiming clock, std::atomic<bool> &cancel)
        : width(w), height(h), timing(clock), cancelled(cancel) {
        NSError *error = nil;
        writer = [[AVAssetWriter alloc] initWithURL:[NSURL fileURLWithPath:@(path.c_str())]
                                           fileType:AVFileTypeMPEG4 error:&error];
        require(writer != nil, error ? error.localizedDescription.UTF8String : "Cannot create RelayVSR video writer");
        NSMutableDictionary *settings = [@{
            AVVideoCodecKey: AVVideoCodecTypeH264,
            AVVideoWidthKey: @(width), AVVideoHeightKey: @(height),
            AVVideoCompressionPropertiesKey: @{
                AVVideoAverageBitRateKey: @8000000,
                AVVideoExpectedSourceFrameRateKey: @(timing.fps),
                AVVideoMaxKeyFrameIntervalKey: @(int(std::ceil(timing.fps * 2))),
                AVVideoAllowFrameReorderingKey: @NO,
            }
        } mutableCopy];
        if (timing.pixel_aspect_width != timing.pixel_aspect_height)
            settings[AVVideoPixelAspectRatioKey] = @{
                AVVideoPixelAspectRatioHorizontalSpacingKey: @(timing.pixel_aspect_width),
                AVVideoPixelAspectRatioVerticalSpacingKey: @(timing.pixel_aspect_height),
            };
        require([writer canApplyOutputSettings:settings forMediaType:AVMediaTypeVideo],
                "RelayVSR H.264 output dimensions are unsupported");
        input = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:settings];
        input.expectsMediaDataInRealTime = NO;
        input.mediaTimeScale = timing.scale;
        writer.movieTimeScale = timing.scale;
        adaptor = [AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:input
            sourcePixelBufferAttributes:@{
                (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
                (NSString *)kCVPixelBufferWidthKey: @(width), (NSString *)kCVPixelBufferHeightKey: @(height),
            }];
        require([writer canAddInput:input], "Cannot create RelayVSR H.264 writer input");
        [writer addInput:input];
        require([writer startWriting], "Cannot start RelayVSR video writer");
        [writer startSessionAtSourceTime:kCMTimeZero];
    }

    ~Impl() { if (writer.status == AVAssetWriterStatusWriting) [writer cancelWriting]; }

    void append(const rv::Frame &frame, const uint8_t *rgb) {
        checkpoint(cancelled);
        require(frame.width * 2 == width && frame.height * 2 == height && frame.pts > last_pts,
                "Invalid RelayVSR output frame geometry or order");
        const auto deadline = Clock::now() + std::chrono::seconds(30);
        while (!input.readyForMoreMediaData) {
            checkpoint(cancelled);
            require(writer.status == AVAssetWriterStatusWriting && Clock::now() < deadline,
                    "RelayVSR H.264 writer stalled");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        OwnedCF<CVPixelBufferRef> buffer;
        require(CVPixelBufferPoolCreatePixelBuffer(nullptr, adaptor.pixelBufferPool, &buffer.value) == kCVReturnSuccess,
                "Cannot allocate RelayVSR output pixel buffer");
        {
            LockedPixels lock(buffer.value, 0);
            auto *base = static_cast<uint8_t *>(CVPixelBufferGetBaseAddress(buffer.value));
            const auto stride = CVPixelBufferGetBytesPerRow(buffer.value);
            for (int y = 0; y < height; ++y)
                copy_bgra(rgb + size_t(y) * width * 3, base + size_t(y) * stride, width);
        }
        OwnedCF<CMVideoFormatDescriptionRef> format;
        require(CMVideoFormatDescriptionCreateForImageBuffer(nullptr, buffer.value, &format.value) == noErr,
                "Cannot describe RelayVSR output pixels");
        CMSampleTimingInfo stamp{CMTimeMake(frame.duration, timing.scale),
                                CMTimeMake(frame.pts, timing.scale), kCMTimeInvalid};
        OwnedCF<CMSampleBufferRef> sample;
        require(CMSampleBufferCreateReadyWithImageBuffer(nullptr, buffer.value, format.value, &stamp, &sample.value) == noErr &&
                [input appendSampleBuffer:sample.value], "RelayVSR H.264 writer rejected a timed frame");
        last_pts = frame.pts;
        last_end = frame.pts + frame.duration;
    }
};

VideoWriter::VideoWriter(const std::filesystem::path &path, int w, int h, VideoTiming clock, std::atomic<bool> &cancelled) {
    @autoreleasepool { impl_ = std::make_unique<Impl>(path, w, h, clock, cancelled); }
}
VideoWriter::~VideoWriter() = default;
void VideoWriter::append(const rv::Frame &frame, const uint8_t *rgb) { @autoreleasepool { impl_->append(frame, rgb); } }
void VideoWriter::finish() {
    @autoreleasepool {
        auto &video = *impl_;
        require(video.last_end > 0 && video.writer.status == AVAssetWriterStatusWriting,
                "Cannot finish an empty or failed RelayVSR video");
        [video.writer endSessionAtSourceTime:CMTimeMake(video.last_end, video.timing.scale)];
        [video.input markAsFinished];
        auto done = dispatch_semaphore_create(0);
        [video.writer finishWritingWithCompletionHandler:^{ dispatch_semaphore_signal(done); }];
        wait_for(done, video.cancelled, "RelayVSR video finalization timed out");
        require(video.writer.status == AVAssetWriterStatusCompleted, "RelayVSR video finalization failed");
    }
}

bool attach_audio(const std::filesystem::path &source, const std::filesystem::path &video,
                  const std::filesystem::path &output, std::atomic<bool> &cancelled) {
    @autoreleasepool {
        auto original = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:@(source.c_str())] options:nil];
        auto audio = [original tracksWithMediaType:AVMediaTypeAudio];
        if (!audio.count) return false;
        auto restored = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:@(video.c_str())] options:nil];
        auto track = media::load_first_track_or_throw(restored, AVMediaTypeVideo, "Timed out reading restored video for audio mux");
        require(track != nil, "Missing RelayVSR video track for audio mux");
        auto composition = [AVMutableComposition composition];
        auto picture = [composition addMutableTrackWithMediaType:AVMediaTypeVideo preferredTrackID:kCMPersistentTrackID_Invalid];
        picture.naturalTimeScale = track.naturalTimeScale;
        NSError *error = nil;
        require([picture insertTimeRange:track.timeRange ofTrack:track atTime:track.timeRange.start error:&error],
                "Cannot mux the RelayVSR video track");
        bool inserted = false;
        for (AVAssetTrack *source_track in audio) {
            checkpoint(cancelled);
            const auto end = CMTimeRangeGetEnd(track.timeRange);
            const auto range = CMTimeRangeGetIntersection(source_track.timeRange, CMTimeRangeMake(kCMTimeZero, end));
            if (!CMTIMERANGE_IS_VALID(range) || CMTIMERANGE_IS_EMPTY(range)) continue;
            auto sound = [composition addMutableTrackWithMediaType:AVMediaTypeAudio preferredTrackID:kCMPersistentTrackID_Invalid];
            sound.naturalTimeScale = source_track.naturalTimeScale;
            require([sound insertTimeRange:range ofTrack:source_track atTime:range.start error:&error],
                    "Cannot mux a source audio track");
            inserted = true;
        }
        if (!inserted) return false;
        auto exporter = [[AVAssetExportSession alloc] initWithAsset:composition presetName:AVAssetExportPresetPassthrough];
        require(exporter && [exporter.supportedFileTypes containsObject:AVFileTypeMPEG4],
                "The source audio cannot be copied into the MP4 container");
        exporter.outputURL = [NSURL fileURLWithPath:@(output.c_str())];
        exporter.outputFileType = AVFileTypeMPEG4;
        auto done = dispatch_semaphore_create(0);
        [exporter exportAsynchronouslyWithCompletionHandler:^{ dispatch_semaphore_signal(done); }];
        try { wait_for(done, cancelled, "RelayVSR audio mux timed out"); }
        catch (...) { [exporter cancelExport]; throw; }
        require(exporter.status == AVAssetExportSessionStatusCompleted,
                exporter.error ? exporter.error.localizedDescription.UTF8String : "RelayVSR audio mux failed");
        return true;
    }
}
} // namespace tc::relay_media

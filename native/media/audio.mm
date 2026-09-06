#include "audio.hpp"
#include "video.hpp"

#import <AVFoundation/AVFoundation.h>
#import <AudioToolbox/AudioToolbox.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace tc {
namespace {

void require_status(bool ok, const std::string& message) {
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
        "timed out while loading media tracks");
    require_status(load_error == nil,
                   load_error ? load_error.localizedDescription.UTF8String :
                                "cannot load media tracks");
    return tracks.firstObject;
}

struct PcmBuffer {
    std::vector<int16_t> samples;
    int clipped = 0;
    int channels = 0;
    int sample_rate = 0;
};

PcmBuffer make_pcm16(const float* waveform, size_t samples,
                     int sample_rate, int channels, size_t target_samples) {
    require_status(waveform && samples > 0 && target_samples > 0,
                   "invalid audio waveform geometry");
    require_status(sample_rate > 0 && channels > 0 && channels <= 8,
                   "invalid audio sample rate or channel count");
    PcmBuffer result;
    result.channels = channels;
    result.sample_rate = sample_rate;
    result.samples.assign(target_samples * static_cast<size_t>(channels), 0);
    for (size_t index = 0;
         index < samples * static_cast<size_t>(channels); ++index) {
        require_status(std::isfinite(waveform[index]),
                       "audio waveform contains non-finite samples");
    }
    const size_t copied = std::min(samples, target_samples);
    for (size_t frame = 0; frame < copied; ++frame) {
        for (int channel = 0; channel < channels; ++channel) {
            const float value = waveform[frame * static_cast<size_t>(channels) +
                                         static_cast<size_t>(channel)];
            const float clipped = std::clamp(value, -1.0f, 1.0f);
            if (clipped != value) ++result.clipped;
            /* Match the Python finalizer's int16 conversion: truncation after
             * scaling, with normalized -1.0 mapping to -32767. */
            result.samples[frame * static_cast<size_t>(channels) +
                           static_cast<size_t>(channel)] =
                static_cast<int16_t>(clipped * 32767.0f);
        }
    }
    return result;
}

AudioMediaInfo info_for(size_t samples, int sample_rate, int channels,
                        int clipped) {
    return {sample_rate, channels, static_cast<int>(samples),
            static_cast<double>(samples) / static_cast<double>(sample_rate),
            clipped};
}

void write_le16(FILE* stream, uint16_t value) {
    uint8_t bytes[2] = {static_cast<uint8_t>(value & 0xffu),
                        static_cast<uint8_t>((value >> 8) & 0xffu)};
    require_status(std::fwrite(bytes, 1, sizeof(bytes), stream) == sizeof(bytes),
                   "cannot write WAV header");
}

void write_le32(FILE* stream, uint32_t value) {
    uint8_t bytes[4] = {
        static_cast<uint8_t>(value & 0xffu),
        static_cast<uint8_t>((value >> 8) & 0xffu),
        static_cast<uint8_t>((value >> 16) & 0xffu),
        static_cast<uint8_t>((value >> 24) & 0xffu),
    };
    require_status(std::fwrite(bytes, 1, sizeof(bytes), stream) == sizeof(bytes),
                   "cannot write WAV header");
}

void write_bytes(FILE* stream, const char* bytes, size_t count) {
    require_status(std::fwrite(bytes, 1, count, stream) == count,
                   "cannot write WAV header");
}

AudioMediaInfo write_pcm16_wav_impl(const std::filesystem::path& output,
                                    const PcmBuffer& pcm) {
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());
    const uint64_t data_bytes = static_cast<uint64_t>(pcm.samples.size()) *
                                sizeof(int16_t);
    require_status(data_bytes <= UINT32_MAX - 36u,
                   "WAV payload is too large for RIFF");
    std::filesystem::path temporary = output;
    temporary += std::string(".tmp-") + NSUUID.UUID.UUIDString.UTF8String;
    FILE* stream = std::fopen(temporary.c_str(), "wb");
    require_status(stream != nullptr, "cannot create WAV output");
    try {
        write_bytes(stream, "RIFF", 4);
        write_le32(stream, static_cast<uint32_t>(36u + data_bytes));
        write_bytes(stream, "WAVEfmt ", 8);
        write_le32(stream, 16u);
        write_le16(stream, 1u);
        write_le16(stream, static_cast<uint16_t>(pcm.channels));
        write_le32(stream, static_cast<uint32_t>(pcm.sample_rate));
        const uint32_t byte_rate = static_cast<uint32_t>(pcm.sample_rate) *
                                    static_cast<uint32_t>(pcm.channels) * 2u;
        write_le32(stream, byte_rate);
        write_le16(stream, static_cast<uint16_t>(pcm.channels * 2));
        write_le16(stream, 16u);
        write_bytes(stream, "data", 4);
        write_le32(stream, static_cast<uint32_t>(data_bytes));
        require_status(std::fwrite(pcm.samples.data(), sizeof(int16_t),
                                   pcm.samples.size(), stream) ==
                           pcm.samples.size(),
                       "cannot write WAV samples");
        require_status(std::fclose(stream) == 0, "cannot finalize WAV output");
        stream = nullptr;
        std::filesystem::rename(temporary, output);
    } catch (...) {
        if (stream) std::fclose(stream);
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
    return info_for(pcm.samples.size() / static_cast<size_t>(pcm.channels),
                    pcm.sample_rate, pcm.channels, pcm.clipped);
}

CMFormatDescriptionRef make_pcm_format(int sample_rate, int channels) {
    AudioStreamBasicDescription description = {};
    description.mSampleRate = sample_rate;
    description.mFormatID = kAudioFormatLinearPCM;
    description.mFormatFlags =
        static_cast<AudioFormatFlags>(kAudioFormatFlagIsSignedInteger) |
        static_cast<AudioFormatFlags>(kAudioFormatFlagsNativeEndian) |
        static_cast<AudioFormatFlags>(kAudioFormatFlagIsPacked);
    description.mBytesPerPacket = static_cast<uint32_t>(channels * 2);
    description.mFramesPerPacket = 1;
    description.mBytesPerFrame = static_cast<uint32_t>(channels * 2);
    description.mChannelsPerFrame = static_cast<uint32_t>(channels);
    description.mBitsPerChannel = 16;
    CMAudioFormatDescriptionRef format = nullptr;
    OSStatus status = CMAudioFormatDescriptionCreate(
        kCFAllocatorDefault, &description, 0, nullptr, 0, nullptr, nullptr,
        &format);
    require_status(status == noErr, "cannot create PCM audio format description");
    return format;
}

CMSampleBufferRef make_audio_sample(CMFormatDescriptionRef format,
                                    const int16_t* samples,
                                    size_t frames, size_t offset,
                                    int sample_rate, int channels) {
    const size_t bytes = frames * static_cast<size_t>(channels) * sizeof(int16_t);
    CMBlockBufferRef block = nullptr;
    OSStatus status = CMBlockBufferCreateWithMemoryBlock(
        kCFAllocatorDefault, nullptr, bytes, kCFAllocatorDefault, nullptr,
        0, bytes, 0, &block);
    require_status(status == kCMBlockBufferNoErr, "cannot allocate audio sample block");
    status = CMBlockBufferReplaceDataBytes(samples, block, 0, bytes);
    if (status != kCMBlockBufferNoErr) {
        CFRelease(block);
        throw std::runtime_error("cannot populate audio sample block");
    }
    CMSampleTimingInfo timing = {
        CMTimeMake(static_cast<int64_t>(offset), sample_rate),
        CMTimeMake(static_cast<int64_t>(frames), sample_rate),
        kCMTimeInvalid,
    };
    CMSampleBufferRef sample = nullptr;
    status = CMSampleBufferCreate(
        kCFAllocatorDefault, block, true, nullptr, nullptr, format, frames,
        1, &timing, 0, nullptr, &sample);
    CFRelease(block);
    require_status(status == noErr, "cannot create audio sample buffer");
    return sample;
}

void validate_muxed_audio(const std::filesystem::path& path,
                          double expected_duration,
                          int expected_sample_rate,
                          int expected_channels) {
    AVURLAsset* asset = [AVURLAsset URLAssetWithURL:
        [NSURL fileURLWithPath:@(path.c_str())] options:nil];
    AVAssetTrack* track = load_first_track(asset, AVMediaTypeAudio);
    require_status(track != nil, "muxed MP4 does not contain an audio stream");
    const double duration = CMTimeGetSeconds(track.timeRange.duration);
    require_status(std::isfinite(duration) &&
                       std::abs(duration - expected_duration) <=
                           1.0 / expected_sample_rate,
                   "muxed AAC duration does not match the video duration");
    CMFormatDescriptionRef format = (__bridge CMFormatDescriptionRef)
        track.formatDescriptions.firstObject;
    const AudioStreamBasicDescription* description =
        format ? CMAudioFormatDescriptionGetStreamBasicDescription(format) : nullptr;
    require_status(description &&
                       std::lround(description->mSampleRate) == expected_sample_rate &&
                       description->mChannelsPerFrame ==
                           static_cast<uint32_t>(expected_channels),
                   "muxed AAC format does not match the requested audio contract");
}

}  // namespace

AudioMediaInfo write_audio_pcm16_wav(const std::filesystem::path& output,
                                    const float* waveform, size_t samples,
                                    int sample_rate, int channels) {
    auto pcm = make_pcm16(waveform, samples, sample_rate, channels, samples);
    return write_pcm16_wav_impl(output, pcm);
}

AudioMediaInfo mux_video_with_audio(const std::filesystem::path& video,
                                    const std::filesystem::path& output,
                                    const float* waveform, size_t samples,
                                    int sample_rate, int channels) {
    require_status(std::filesystem::is_regular_file(video),
                   "source video does not exist");
    require_status(waveform && samples > 0, "missing audio waveform");
    std::filesystem::path temporary;
    try {
    @autoreleasepool {
        NSError* error = nil;
        AVURLAsset* asset = [AVURLAsset URLAssetWithURL:
            [NSURL fileURLWithPath:@(video.c_str())] options:nil];
        AVAssetTrack* video_track = load_first_track(asset, AVMediaTypeVideo);
        require_status(video_track != nil, "source video stream is missing");
        const Float64 duration = CMTimeGetSeconds(asset.duration);
        const int fps = video_track.nominalFrameRate > 0.0f ?
            static_cast<int>(std::lround(video_track.nominalFrameRate)) : 0;
        require_status(fps > 0 && std::isfinite(duration) && duration > 0.0,
                       "source video has invalid duration or frame rate");
        const size_t target_samples = static_cast<size_t>(std::llround(
            duration * static_cast<double>(sample_rate)));
        auto pcm = make_pcm16(waveform, samples, sample_rate, channels,
                              std::max<size_t>(target_samples, 1));

        AVAssetReader* reader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
        require_status(reader != nil, error.localizedDescription.UTF8String ?: "cannot create video reader");
        AVAssetReaderTrackOutput* reader_output = [[AVAssetReaderTrackOutput alloc]
            initWithTrack:video_track outputSettings:nil];
        require_status([reader canAddOutput:reader_output], "cannot add video reader output");
        [reader addOutput:reader_output];
        NSMutableArray* video_samples = [NSMutableArray array];
        require_status([reader startReading], reader.error.localizedDescription.UTF8String ?: "cannot start video reader");
        CMSampleBufferRef sample = nullptr;
        while ((sample = [reader_output copyNextSampleBuffer])) {
            [video_samples addObject:(__bridge id)sample];
            CFRelease(sample);
        }
        require_status(reader.status == AVAssetReaderStatusCompleted,
                       reader.error.localizedDescription.UTF8String ?: "cannot read source video");
        require_status(video_samples.count > 0, "source video contains no samples");

        if (!output.parent_path().empty())
            std::filesystem::create_directories(output.parent_path());
        temporary = output;
        temporary += std::string(".") + NSUUID.UUID.UUIDString.UTF8String + ".mp4";
        AVAssetWriter* writer = [[AVAssetWriter alloc]
            initWithURL:[NSURL fileURLWithPath:@(temporary.c_str())]
            fileType:AVFileTypeMPEG4 error:&error];
        require_status(writer != nil, error.localizedDescription.UTF8String ?: "cannot create MP4 writer");
        CMFormatDescriptionRef source_format = (__bridge CMFormatDescriptionRef)
            video_track.formatDescriptions.firstObject;
        AVAssetWriterInput* video_input = [[AVAssetWriterInput alloc]
            initWithMediaType:AVMediaTypeVideo outputSettings:nil
            sourceFormatHint:source_format];
        video_input.expectsMediaDataInRealTime = NO;
        require_status([writer canAddInput:video_input], "cannot add video writer input");
        [writer addInput:video_input];
        NSDictionary* audio_settings = @{
            AVFormatIDKey: @(kAudioFormatMPEG4AAC),
            AVSampleRateKey: @(sample_rate),
            AVNumberOfChannelsKey: @(channels),
            AVEncoderBitRateKey: @(192000),
        };
        id audio_format_owner = CFBridgingRelease(
            make_pcm_format(sample_rate, channels));
        CMFormatDescriptionRef audio_format = (__bridge CMFormatDescriptionRef)
            audio_format_owner;
        AVAssetWriterInput* audio_input = [[AVAssetWriterInput alloc]
            initWithMediaType:AVMediaTypeAudio outputSettings:audio_settings
            sourceFormatHint:audio_format];
        require_status(audio_input != nil, "cannot create AAC audio writer input");
        audio_input.expectsMediaDataInRealTime = NO;
        if (![writer canAddInput:audio_input]) {
            std::string detail = "cannot add audio writer input";
            if (writer.error.localizedDescription)
                detail += ": " + std::string(writer.error.localizedDescription.UTF8String);
            throw std::runtime_error(detail);
        }
        [writer addInput:audio_input];
        require_status([writer startWriting], writer.error.localizedDescription.UTF8String ?: "cannot start MP4 writer");
        [writer startSessionAtSourceTime:kCMTimeZero];
        dispatch_group_t appended = dispatch_group_create();
        dispatch_queue_t video_queue = dispatch_queue_create(
            "com.turbocider.audio-mux-video", DISPATCH_QUEUE_SERIAL);
        dispatch_queue_t audio_queue = dispatch_queue_create(
            "com.turbocider.audio-mux-audio", DISPATCH_QUEUE_SERIAL);
        __block NSUInteger video_index = 0;
        __block BOOL video_finished = NO;
        __block BOOL video_failed = NO;
        dispatch_group_enter(appended);
        [video_input requestMediaDataWhenReadyOnQueue:video_queue usingBlock:^{
            @autoreleasepool {
                if (video_finished) return;
                while (video_index < video_samples.count &&
                       video_input.readyForMoreMediaData &&
                       writer.status == AVAssetWriterStatusWriting) {
                    CMSampleBufferRef video_sample = (__bridge CMSampleBufferRef)
                        video_samples[video_index++];
                    if (![video_input appendSampleBuffer:video_sample]) {
                        video_failed = YES;
                        break;
                    }
                }
                if (video_index == video_samples.count || video_failed ||
                    writer.status != AVAssetWriterStatusWriting) {
                    video_finished = YES;
                    [video_input markAsFinished];
                    dispatch_group_leave(appended);
                }
            }
        }];

        /* Apple's AAC encoder contributes 1024 priming samples to the
         * container timeline. Feed that many fewer PCM frames so the
         * resulting AAC track duration matches the video track. */
        constexpr size_t kAacPrimingSamples = 1024;
        const size_t encoded_samples = target_samples > kAacPrimingSamples ?
            target_samples - kAacPrimingSamples : target_samples;
        __block size_t audio_offset = 0;
        __block BOOL audio_finished = NO;
        __block BOOL audio_failed = NO;
        const size_t max_frames = 1024;
        dispatch_group_enter(appended);
        [audio_input requestMediaDataWhenReadyOnQueue:audio_queue usingBlock:^{
            @autoreleasepool {
                if (audio_finished) return;
                try {
                    while (audio_offset < encoded_samples &&
                           audio_input.readyForMoreMediaData &&
                           writer.status == AVAssetWriterStatusWriting) {
                        const size_t count = std::min(
                            max_frames, encoded_samples - audio_offset);
                        CMSampleBufferRef audio_sample = make_audio_sample(
                            audio_format,
                            pcm.samples.data() + audio_offset *
                                static_cast<size_t>(channels),
                            count, audio_offset, sample_rate, channels);
                        const BOOL accepted =
                            [audio_input appendSampleBuffer:audio_sample];
                        CFRelease(audio_sample);
                        if (!accepted) {
                            audio_failed = YES;
                            break;
                        }
                        audio_offset += count;
                    }
                } catch (...) {
                    audio_failed = YES;
                }
                if (audio_offset == encoded_samples || audio_failed ||
                    writer.status != AVAssetWriterStatusWriting) {
                    audio_finished = YES;
                    [audio_input markAsFinished];
                    dispatch_group_leave(appended);
                }
            }
        }];
        const long append_wait = dispatch_group_wait(
            appended, dispatch_time(DISPATCH_TIME_NOW, 180 * NSEC_PER_SEC));
        if (append_wait != 0) {
            [writer cancelWriting];
            dispatch_group_wait(
                appended, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
            throw std::runtime_error(
                "audio/video writer timed out while appending samples");
        }
        require_status(!video_failed && !audio_failed &&
                       video_index == video_samples.count &&
                       audio_offset == encoded_samples &&
                       writer.status == AVAssetWriterStatusWriting,
                       writer.error.localizedDescription.UTF8String ?:
                           "audio/video writer rejected media samples");
        [writer endSessionAtSourceTime:asset.duration];
        dispatch_semaphore_t finished = dispatch_semaphore_create(0);
        [writer finishWritingWithCompletionHandler:^{ dispatch_semaphore_signal(finished); }];
        const long finish_wait = dispatch_semaphore_wait(
            finished, dispatch_time(DISPATCH_TIME_NOW, 180 * NSEC_PER_SEC));
        if (finish_wait != 0) [writer cancelWriting];
        require_status(finish_wait == 0 &&
                           writer.status == AVAssetWriterStatusCompleted,
                       writer.error.localizedDescription.UTF8String ?:
                           "cannot finalize MP4 writer");
        auto video_info = probe_video(temporary);
        require_status(video_info.frames > 0,
                       "muxed MP4 does not contain a valid video stream");
        validate_muxed_audio(temporary, duration, sample_rate, channels);
        std::filesystem::rename(temporary, output);
        return info_for(target_samples, sample_rate, channels, pcm.clipped);
    }
    } catch (...) {
        if (!temporary.empty()) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
        }
        throw;
    }
}

}  // namespace tc

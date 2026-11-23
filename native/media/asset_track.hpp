#pragma once

#import <AVFoundation/AVFoundation.h>

#include "../core/common.hpp"

namespace tc::media {

struct TrackLoad {
    AVAssetTrack *track = nil;
    NSError *error = nil;
    bool timed_out = false;
};

inline TrackLoad load_first_track(AVAsset *asset, AVMediaType media_type,
                                  const std::atomic<bool> *cancelled = nullptr) {
    dispatch_semaphore_t ready = dispatch_semaphore_create(0);
    __block NSArray<AVAssetTrack *> *tracks = nil;
    __block NSError *failure = nil;
    [asset loadTracksWithMediaType:media_type
                 completionHandler:^(NSArray<AVAssetTrack *> *loaded, NSError *error) {
        tracks = loaded;
        failure = error;
        dispatch_semaphore_signal(ready);
    }];
    if (cancelled) {
        const auto deadline = Clock::now() + std::chrono::seconds(30);
        while (dispatch_semaphore_wait(ready, dispatch_time(DISPATCH_TIME_NOW, 50 * NSEC_PER_MSEC))) {
            if (cancelled->load()) throw Cancelled();
            if (Clock::now() >= deadline) return {.timed_out = true};
        }
        if (cancelled->load()) throw Cancelled();
    } else if (dispatch_semaphore_wait(ready, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC))) {
        return {.timed_out = true};
    }
    return {tracks.firstObject, failure};
}

inline AVAssetTrack *load_first_track_or_throw(AVAsset *asset, AVMediaType media_type,
                                              const char *timeout_message) {
    const auto loaded = load_first_track(asset, media_type);
    if (loaded.timed_out) throw std::runtime_error(timeout_message);
    if (loaded.error) throw std::runtime_error(loaded.error.localizedDescription.UTF8String);
    return loaded.track;
}

} // namespace tc::media

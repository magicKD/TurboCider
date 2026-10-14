#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <atomic>
#include <memory>

namespace tc {

struct VideoMediaInfo {
    int width = 0;
    int height = 0;
    int frames = 0;
    int fps = 0;
};

/* Write RGB24 frames in frame-major, interleaved layout to an H.264 MP4.
 * The function owns no input memory and never truncates the requested frame
 * count.  Audio is intentionally a separate API so a failed audio finalizer
 * cannot silently shorten the video stream. */
void write_video_rgb24(const std::filesystem::path& output,
                       const uint8_t* frames,
                       int frame_count,
                       int width,
                       int height,
                       int fps);

VideoMediaInfo probe_video(const std::filesystem::path& path);

// Bounded synchronous producer: append owns/copies no RGB input after return,
// waits for encoder backpressure, and accepts exactly the declared sequence.
// Only finish publishes output. Destruction cancels and removes its private
// staging file, never an existing destination or source.
class RGB24VideoWriter {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    RGB24VideoWriter(const std::filesystem::path &, int frames, int width, int height, int fps,
                     const std::atomic<bool> *cancelled = nullptr);
    ~RGB24VideoWriter();
    RGB24VideoWriter(const RGB24VideoWriter &) = delete;
    RGB24VideoWriter &operator=(const RGB24VideoWriter &) = delete;
    void append(int index, const uint8_t *rgb, size_t bytes);
    void finish();
    int frames_written() const;
};

}  // namespace tc

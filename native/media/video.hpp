#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

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

}  // namespace tc

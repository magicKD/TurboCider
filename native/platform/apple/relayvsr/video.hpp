#pragma once

#include "../../../models/relayvsr/pipeline.hpp"
#include <memory>

namespace tc::relay_media {

struct VideoTiming {
    int scale = 1;
    double fps = 0;
    int pixel_aspect_width = 1, pixel_aspect_height = 1;
};

class VideoReader {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    VideoReader(const std::filesystem::path &, std::atomic<bool> &);
    ~VideoReader();
    bool next(rv::Frame &);
    VideoTiming timing() const;
};

class VideoWriter {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    VideoWriter(const std::filesystem::path &, int width, int height, VideoTiming, std::atomic<bool> &);
    ~VideoWriter();
    void append(const rv::Frame &source, const uint8_t *rgb);
    void finish();
};

bool attach_audio(const std::filesystem::path &source, const std::filesystem::path &video,
                  const std::filesystem::path &output, std::atomic<bool> &);

} // namespace tc::relay_media

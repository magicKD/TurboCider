#pragma once
#include "../core/common.hpp"

namespace tc {
struct RGBVideoFrame {
    int index = 0, width = 0, height = 0;
    std::vector<uint8_t> pixels;
};
// A complete-source CFR contract with one decoded sample at a time. Reuse the
// caller's RGB frame buffer; never collect the whole movie or truncate a tail.
// Returned pixels already include the supported display rotation/reflection.
class RGB24VideoReader {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    RGB24VideoReader(const std::filesystem::path &, int expected_frames, int expected_fps,
                     const Event &, std::atomic<bool> &);
    ~RGB24VideoReader();
    RGB24VideoReader(const RGB24VideoReader &) = delete;
    RGB24VideoReader &operator=(const RGB24VideoReader &) = delete;
    bool next(RGBVideoFrame &);
    int frames_read() const;
};
} // namespace tc

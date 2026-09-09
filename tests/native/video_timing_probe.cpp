#include "../../native/media/video.hpp"
#include <filesystem>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    try {
        if (argc != 2) return 2;
        std::vector<uint8_t> rgb(81 * 16 * 16 * 3, 128);
        for (int fps : {16, 24, 30}) {
            for (int frames : {5, 81}) {
                auto path = std::filesystem::path(argv[1]) /
                    (std::to_string(fps) + "-" + std::to_string(frames) + ".mp4");
                tc::write_video_rgb24(path, rgb.data(), frames, 16, 16, fps);
            }
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

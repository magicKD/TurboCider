#import <Foundation/Foundation.h>
#include "../../native/media/video.hpp"
#include "../../native/backends/mlx.hpp"
#include <iostream>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            tc::require(argc == 4 && !std::filesystem::exists(argv[2]),
                        "usage: ltx098-video-encode-probe RGB.safetensors NEW_OUTPUT.mp4 FPS");
            auto rgb = tc::mx::load_safetensors(argv[1]).first.at("rgb");
            tc::require(rgb.ndim() == 4 && rgb.shape(3) == 3 && rgb.dtype() == tc::mx::uint8,
                        "expected uint8 [frames,height,width,3] RGB");
            rgb = tc::mx::contiguous(rgb);
            tc::mx::eval(rgb);
            tc::write_video_rgb24(argv[2], rgb.data<uint8_t>(), rgb.shape(0), rgb.shape(2), rgb.shape(1), std::stoi(argv[3]));
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}

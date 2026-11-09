#import <Foundation/Foundation.h>
#include "../../native/backends/mlx.hpp"
#include "../../native/media/video.hpp"

#include <iostream>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            tc::require((argc == 4 || argc == 5) && !std::filesystem::exists(argv[2]) &&
                            !std::filesystem::is_symlink(argv[2]),
                        "usage: wan22-pixels-to-video INPUT.safetensors NEW_OUTPUT.mp4 FPS [TENSOR_KEY]");
            auto pixels = tc::mx::load_safetensors(argv[1]).first.at(argc == 5 ? argv[4] : "pixels");
            tc::require(pixels.ndim() == 5 && pixels.shape(0) == 1 && pixels.shape(1) == 3 &&
                            pixels.shape(2) > 0 && pixels.shape(3) > 0 && pixels.shape(4) > 0,
                        "expected decoder pixels [1,3,frames,height,width]");
            const int frames = pixels.shape(2), height = pixels.shape(3), width = pixels.shape(4);
            auto rgb = tc::mx::squeeze(tc::mx::transpose(tc::mx::astype(pixels, tc::mx::float32), {0, 2, 3, 4, 1}), 0);
            rgb = tc::mx::astype(tc::mx::round(tc::mx::clip((rgb + 1.f) * 127.5f,
                                                             tc::Tensor(0.f), tc::Tensor(255.f))), tc::mx::uint8);
            rgb = tc::mx::contiguous(rgb);
            tc::mx::eval(rgb);
            tc::write_video_rgb24(std::filesystem::absolute(argv[2]), rgb.data<uint8_t>(), frames, width, height,
                                  std::stoi(argv[3]));
            auto info = tc::probe_video(std::filesystem::absolute(argv[2]));
            tc::require(info.frames == frames && info.width == width && info.height == height,
                        "encoded Wan2.2 video geometry mismatch");
            std::cout << "Wrote " << frames << " frames at " << width << "x" << height << '\n';
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}

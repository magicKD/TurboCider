#import <Foundation/Foundation.h>
#include "../../native/media/image.hpp"
#include <iostream>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            tc::require(argc == 5, "usage: ltx098-image-probe IMAGE WIDTH HEIGHT OUTPUT.safetensors");
            tc::require(!std::filesystem::exists(argv[4]), "probe output already exists");
            auto pixels = tc::load_image_center_crop_tensor(argv[1], std::stoi(argv[2]), std::stoi(argv[3]));
            tc::mx::eval(pixels);
            tc::mx::save_safetensors(argv[4], {{"pixels", pixels}});
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}

#include "../../native/media/image.hpp"
#include "../../native/models/qwen21/conditioning.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3 || (argc == 4 && std::string(argv[3]) == "--processor"),
                    "usage: qwen35-image-decode-probe INPUT_IMAGE OUTPUT.safetensors [--processor]");
        tc::mx::set_default_device(tc::mx::Device::cpu);
        auto pixels = tc::load_pe_image_tensor(argv[1]);
        std::unordered_map<std::string, tc::Tensor> outputs{{"pixels", pixels}};
        if (argc == 4) {
            auto processed = tc::qwen21::pe_vision_input(pixels);
            outputs.emplace("patches0", processed.patches);
            int grid[] = {1, processed.grid_height, processed.grid_width};
            outputs.emplace("grids", tc::Tensor(grid, {1, 3}, tc::mx::int32));
        }
        tc::mx::save_safetensors(argv[2], outputs);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

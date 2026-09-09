#include "../../native/components/diffusion/wan.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 6, "usage: wan-inputs-probe F H W SEED OUTPUT.safetensors");
        tc::components::WanPatchGrid grid{std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3])};
        auto tables = tc::components::wan_rotary(grid);
        tc::mx::random::KeySequence sequence(std::stoull(argv[4]));
        const tc::mx::Shape shape{1, 16, grid.frames, grid.height * 2, grid.width * 2};
        auto first = tc::mx::random::normal(shape, tc::mx::float32, sequence.next());
        auto second = tc::mx::random::normal(shape, tc::mx::float32, sequence.next());
        tc::mx::save_safetensors(argv[5], {{"cosine", tables.first}, {"sine", tables.second},
                                         {"renoise_0", first}, {"renoise_1", second}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

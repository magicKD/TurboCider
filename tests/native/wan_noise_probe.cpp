#include "../../native/components/diffusion/seeded_noise.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3, "expected count and seed");
        auto values = tc::components::wan_initial_noise(std::stoull(argv[1]), std::stoull(argv[2]));
        std::cout.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

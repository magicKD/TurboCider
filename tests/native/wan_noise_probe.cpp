#include "../../native/components/diffusion/seeded_noise.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3 || argc == 4, "expected count, seed and optional draw count");
        tc::components::WanNoiseGenerator generator(std::stoull(argv[2]));
        const int draws = argc == 4 ? std::stoi(argv[3]) : 1;
        tc::require(draws > 0 && draws <= 8, "invalid draw count");
        for (int i = 0; i < draws; ++i) {
            auto values = generator.normal(std::stoull(argv[1]));
            std::cout.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

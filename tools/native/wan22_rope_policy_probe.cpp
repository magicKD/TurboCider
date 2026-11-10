#include "../../native/models/wan22/identifiers.hpp"

#include <iostream>

int main(int argc, char **argv) {
    if (argc != 10) {
        std::cerr << "usage: wan22-rope-policy GPU MEMORY_BYTES OS_MAJOR OS_MINOR OS_PATCH MLX_VERSION WIDTH HEIGHT FRAMES\n";
        return 1;
    }
    std::cout << tc::wan22::fused_rope_eligible(argv[1], std::stoull(argv[2]), std::stoi(argv[3]),
        std::stoi(argv[4]), std::stoi(argv[5]), argv[6], std::stoi(argv[7]), std::stoi(argv[8]), std::stoi(argv[9])) << '\n';
}

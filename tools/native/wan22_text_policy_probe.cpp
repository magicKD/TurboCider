#include "../../native/models/wan22/identifiers.hpp"

#include <iostream>

int main(int argc, char **argv) {
    if (argc != 9) {
        std::cerr << "usage: wan22-text-policy GPU MEMORY_BYTES OS_MAJOR OS_MINOR OS_PATCH MLX_VERSION VALID SEQUENCE\n";
        return 1;
    }
    std::cout << tc::wan22::text_padding_alignment(argv[1], std::stoull(argv[2]), std::stoi(argv[3]),
        std::stoi(argv[4]), std::stoi(argv[5]), argv[6], std::stoi(argv[7]), std::stoull(argv[8])) << '\n';
}

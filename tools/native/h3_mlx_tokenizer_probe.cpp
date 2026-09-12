#include "../../native/core/tokenizer.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3,
                    "usage: h3-mlx-tokenizer-probe TOKENIZER_ROOT PROMPT");
        tc::Tokenizer tokenizer(std::filesystem::absolute(argv[1]));
        auto tokens = tokenizer.raw(argv[2]);
        std::cout << '[';
        for (size_t index = 0; index < tokens.ids.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << tokens.ids[index];
        }
        std::cout << "]\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

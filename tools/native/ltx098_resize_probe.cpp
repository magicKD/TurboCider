#include "../../native/models/ltx098/resize.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 5, "usage: ltx098-resize-probe INPUT OUTPUT HEIGHT WIDTH");
        auto values = tc::mx::load_safetensors(argv[1]).first;
        auto output = tc::ltx098::resize_bilinear(values.at("pixels"), std::stoi(argv[3]), std::stoi(argv[4]));
        tc::mx::eval(output);
        tc::mx::save_safetensors(argv[2], {{"pixels", output}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

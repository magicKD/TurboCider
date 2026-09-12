#include "../../native/models/h3_mlx/conditioner.hpp"

#include <iostream>
#include <sstream>

namespace {
std::vector<int> parse_rows(const std::string &value) {
    std::vector<int> rows;
    std::stringstream stream(value);
    std::string piece;
    while (std::getline(stream, piece, ',')) {
        tc::require(!piece.empty() && piece.find_first_not_of("0123456789") == std::string::npos,
                    "row list must contain comma-separated nonnegative integers");
        rows.push_back(std::stoi(piece));
    }
    tc::require(!rows.empty(), "row list is empty");
    return rows;
}
}

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4 || argc == 5,
                    "usage: h3-mlx-tensor-probe COMPONENT TENSOR OUTPUT [ROWS]");
        tc::h3_mlx::ShardIndex index(std::filesystem::absolute(argv[1]));
        tc::require(index.has(argv[2]), "requested tensor is absent from the shard index");
        auto tensor = argc == 5 ? index.rows(argv[2], parse_rows(argv[4])) :
                                  index.tensor(argv[2]);
        auto fp32 = tc::mx::astype(tensor, tc::mx::float32);
        tc::mx::eval(fp32);
        tc::mx::save_safetensors(argv[3], {{"tensor", fp32}});
        std::cout << "{\"rank\":" << tensor.ndim() << ",\"elements\":"
                  << tensor.size() << ",\"bytes\":" << tensor.nbytes() << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

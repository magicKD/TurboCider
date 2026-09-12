#include "../../native/models/h3_mlx/checkpoint.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 5,
                    "usage: h3-mlx-linear-probe CHECKPOINT PREFIX INPUT OUTPUT");
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::h3_mlx::Checkpoint checkpoint;
        checkpoint.load(std::filesystem::absolute(argv[1]), event, cancelled);
        auto arrays = tc::mx::load_safetensors(std::filesystem::absolute(argv[3]).string()).first;
        auto found = arrays.find("input");
        tc::require(found != arrays.end(), "linear probe input tensor is missing");
        auto input = tc::mx::astype(found->second, checkpoint.projection_dtype(argv[2]));
        auto output = checkpoint.linear(input, argv[2], false);
        tc::mx::eval(output);
        tc::mx::save_safetensors(std::filesystem::absolute(argv[4]).string(), {{"output", output}});
        std::cout << "{\"rows\":" << output.size() / output.shape(-1)
                  << ",\"columns\":" << output.shape(-1)
                  << ",\"dtype\":\"" << output.dtype() << "\"}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

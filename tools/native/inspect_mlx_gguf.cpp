#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <mlx/io.h>
#include <mlx/utils.h>

namespace mx = mlx::core;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: inspect_mlx_gguf /path/to/model.gguf\n";
        return 2;
    }
    try {
        auto [arrays, metadata] = mx::load_gguf(argv[1]);
        std::vector<std::string> names;
        names.reserve(arrays.size());
        for (const auto &[name, _] : arrays)
            names.push_back(name);
        std::sort(names.begin(), names.end());
        std::cout << "arrays=" << names.size() << " metadata=" << metadata.size() << '\n';
        for (const auto &name : names) {
            const auto &value = arrays.at(name);
            std::cout << name << " shape=[";
            for (int index = 0; index < value.ndim(); ++index) {
                if (index)
                    std::cout << ',';
                std::cout << value.shape(index);
            }
            std::cout << "] dtype=" << value.dtype() << " bytes=" << value.nbytes() << '\n';
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "GGUF inspection failed: " << error.what() << '\n';
        return 1;
    }
}

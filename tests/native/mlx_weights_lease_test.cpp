#include "backends/mlx.hpp"
#include "runtime/streaming/source_lease.hpp"

#include <atomic>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <class Function>
void rejects(Function &&function, const char *part) {
    try {
        function();
    } catch (const std::exception &error) {
        if (std::string(error.what()).find(part) != std::string::npos)
            return;
        throw std::runtime_error(
            std::string("unexpected rejection: ") + error.what());
    }
    throw std::runtime_error(
        std::string("operation unexpectedly succeeded: ") + part);
}

void save_values(const std::filesystem::path &path,
                 const std::vector<float> &values) {
    auto tensor = tc::Tensor(
        values.data(), {static_cast<int>(values.size())}, tc::mx::float32);
    tc::mx::save_safetensors(path.string(), {{"value", tensor}});
}

void expect_values(const tc::Tensor &tensor,
                   const std::vector<float> &expected) {
    tc::mx::eval(tensor);
    assert(tensor.dtype() == tc::mx::float32);
    assert(tensor.size() == expected.size());
    const float *actual = tensor.data<float>();
    for (size_t index = 0; index < expected.size(); ++index)
        assert(std::abs(actual[index] - expected[index]) < 1e-6f);
}

} // namespace

int main(int argc, char **argv) {
    assert(argc == 2);
    try {
        const std::filesystem::path root(argv[1]);
        std::filesystem::create_directories(root);
        const auto path = root / "weights.safetensors";
        const std::vector<float> original{1.f, 2.f, 3.f, 4.f};
        const std::vector<float> replacement{9.f, 8.f, 7.f, 6.f};
        save_values(path, original);

        tc::streaming::SourceFileIdentity file;
        file.logical_id = "weights.safetensors";
        file.path = path;
        auto lease = tc::streaming::SourceLease::capture({std::move(file)});
        assert(lease && lease->file_count() == 1);

        std::atomic<bool> cancelled{false};
        const tc::Event event = [](const std::string &, int, int) {};
        tc::Weights weights;
        rejects([&] {
            weights.load_lease(
                lease, {"missing.safetensors"}, event, cancelled);
        }, "unknown logical id");

        weights.load_lease(
            lease, {"weights.safetensors"}, event, cancelled);
        assert(weights.has("value"));

        const auto moved = root / "weights.original.safetensors";
        std::filesystem::rename(path, moved);
        save_values(path, replacement);

        // The lazy MLX load remains bound to the descriptor captured by the
        // SourceLease, even after the named path is replaced.
        weights.materialize();
        expect_values(weights.at("value"), original);
        rejects([&] { lease->revalidate_paths(); }, "source");

        weights.clear();
        std::cout << "PASS MLX lease weights: fd-backed lazy safetensors "
                     "load, failure cleanup and path replacement lineage\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

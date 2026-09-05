#include <mlx/mlx.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <vector>

namespace mx = mlx::core;

int main(void) {
    try {
        const mx::Device gpu(mx::Device::gpu);
        if (!mx::is_available(gpu)) {
            std::fprintf(stderr, "MLX GPU is not available\n");
            return 77;
        }
        mx::set_default_device(gpu);

        const mx::Shape input_shape = {1, 13, 14, 22, 128};
        const mx::Shape weight_shape = {256, 3, 3, 3, 128};
        mx::array input = mx::ones(input_shape, mx::bfloat16);
        mx::array weight = mx::full(weight_shape, 0.001f, mx::bfloat16);
        mx::eval(input, weight);

        auto run = [&]() {
            mx::array output = mx::conv3d(
                input,
                weight,
                std::tuple<int, int, int>{1, 1, 1},
                std::tuple<int, int, int>{1, 1, 1});
            mx::eval(output);
            return output;
        };

        mx::array output = run();
        std::vector<double> timings;
        timings.reserve(5);
        for (int iteration = 0; iteration < 5; ++iteration) {
            const auto started = std::chrono::steady_clock::now();
            output = run();
            const auto stopped = std::chrono::steady_clock::now();
            timings.push_back(
                std::chrono::duration<double>(stopped - started).count());
        }
        std::sort(timings.begin(), timings.end());

        const auto &shape = output.shape();
        mx::array first_value = mx::take(
            mx::reshape(mx::astype(output, mx::float32), {-1}), 0);
        mx::eval(first_value);
        const float first = first_value.item<float>();
        std::printf(
            "mlx_version=%s output_shape=[%d,%d,%d,%d,%d] "
            "warm_p50_seconds=%.6f first=%g\n",
            mx::version(),
            shape[0], shape[1], shape[2], shape[3], shape[4],
            timings[timings.size() / 2],
            first);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "MLX C++ smoke failed: %s\n", error.what());
        return 1;
    }
}

#include "../../native/models/h3_mlx/vdn_mlx.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        tc::configure_streams();
        const std::vector<int> dimensions{7, 31, 32, 33, 64, 128};
        constexpr int batch = 3;
        float worst_max_abs = 0.f;
        float worst_relative_rmse = 0.f;
        float worst_residual = 0.f;

        for (int dim : dimensions) {
            std::vector<float> values(size_t(batch) * dim * dim);
            for (int b = 0; b < batch; ++b) {
                for (int row = 0; row < dim; ++row) {
                    for (int column = 0; column < dim; ++column) {
                        const float phase = float(1 + b * dim * dim +
                                                  row * dim + column);
                        values[(size_t(b) * dim + row) * dim + column] =
                            0.08f * std::sin(phase * 0.173f) +
                            0.03f * std::cos(phase * 0.071f);
                    }
                }
            }
            auto x = tc::Tensor(values.data(), {batch, dim, dim},
                                tc::mx::float32);
            auto a = tc::mx::matmul(tc::mx::transpose(x, {0, 2, 1}), x) /
                     float(dim);
            auto cpu = tc::h3_mlx::vdn_spd_inverse(a, true);
            auto gpu = tc::h3_mlx::vdn_spd_inverse(a, false);
            tc::mx::eval(cpu, gpu);

            auto difference = tc::mx::astype(gpu - cpu, tc::mx::float32);
            const float max_abs =
                tc::mx::max(tc::mx::abs(difference)).item<float>();
            const float rmse = tc::mx::sqrt(
                tc::mx::mean(tc::mx::square(difference))).item<float>();
            const float reference_rms = tc::mx::sqrt(
                tc::mx::mean(tc::mx::square(
                    tc::mx::astype(cpu, tc::mx::float32)))).item<float>();
            const float relative_rmse = rmse / std::max(reference_rms, 1e-12f);

            auto eye = tc::mx::reshape(tc::mx::eye(dim, tc::mx::float32),
                                       {1, dim, dim});
            auto residual = tc::mx::matmul(a + eye, gpu) - eye;
            const float max_residual =
                tc::mx::max(tc::mx::abs(residual)).item<float>();
            tc::require(tc::mx::all(tc::mx::isfinite(gpu)).item<bool>(),
                        "VDN GPU solve produced non-finite values at d=" +
                            std::to_string(dim));
            worst_max_abs = std::max(worst_max_abs, max_abs);
            worst_relative_rmse = std::max(worst_relative_rmse, relative_rmse);
            worst_residual = std::max(worst_residual, max_residual);
        }

        tc::require(worst_max_abs < 5e-4f,
                    "VDN GPU solve max absolute error exceeds 5e-4");
        tc::require(worst_relative_rmse < 5e-4f,
                    "VDN GPU solve relative RMSE exceeds 5e-4");
        tc::require(worst_residual < 1e-3f,
                    "VDN GPU solve residual exceeds 1e-3");
        std::cout << std::setprecision(9)
                  << "{\"dimensions\":6,\"batch\":" << batch
                  << ",\"max_abs\":" << worst_max_abs
                  << ",\"relative_rmse\":" << worst_relative_rmse
                  << ",\"max_residual\":" << worst_residual << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#include "../../native/backends/dense_gpu_projection.hpp"
#include "../../native/backends/mlx.hpp"

#include <iostream>

namespace {
using namespace tc;

float relative_l2(const Tensor &actual, const Tensor &expected) {
    auto a = mx::astype(actual, mx::float32), b = mx::astype(expected, mx::float32);
    return mx::sqrt(mx::sum(mx::square(a-b)) /
        mx::maximum(mx::sum(mx::square(b)), Tensor(1e-20f))).item<float>();
}
} // namespace

int main() {
    try {
        configure_streams();
        int cases = 0, rejected = 0;
        for (auto dtype : {mx::float16, mx::bfloat16}) {
            // Nonzero row/column offsets with a larger physical leading
            // dimension catch accidentally compacted/tightly packed indexing.
            auto weight = mx::astype(mx::reshape(
                mx::sin(mx::arange(160*256, mx::float32)*.0031f)*.05f,
                {160,256}), dtype);
            mx::eval(weight);
            for (auto geometry : {std::pair{64,128}, std::pair{33,129}, std::pair{65,79}}) {
                const auto [rows, columns] = geometry;
                constexpr int row_begin = 7, col_begin = 32, inner = 128;
                auto source = mx::astype(mx::reshape(
                    mx::cos(mx::arange(rows*192, mx::float32)*.013f)*.25f,
                    {1,rows,192}), dtype);
                auto input = slice_axis(source, -1, 0, inner);
                mx::eval(input);
                require(!input.flags().row_contiguous, "strided input fixture became contiguous");
                auto selected = slice_axis(slice_axis(weight, 0, row_begin, row_begin+columns),
                                           1, col_begin, col_begin+inner);
                auto oracle = mx::matmul(mx::astype(input, mx::float32),
                                        mx::transpose(mx::astype(selected, mx::float32)));
                mx::eval(oracle);
                for (bool keep_fp32 : {false, true}) {
                    auto baseline = dense_gpu::projection_range(input, weight, row_begin,
                        row_begin+columns, col_begin, col_begin+inner, 32, keep_fp32);
                    mx::eval(baseline);
                    require(baseline.dtype() == (keep_fp32 ? mx::float32 : dtype), "output boundary changed");
                    require(relative_l2(baseline, oracle) < (keep_fp32 ? 5e-5f : .005f),
                            "physical projection differs from independent FP32 matmul");
                    for (int bm : {16,32,64}) for (int bn : {64,128}) for (bool statics : {false,true}) {
                        auto candidate = dense_gpu::projection_range(input, weight, row_begin,
                            row_begin+columns, col_begin, col_begin+inner, bm, keep_fp32, bn, statics);
                        mx::eval(candidate);
                        require(candidate.shape() == mx::Shape({1,rows,columns}) &&
                                    candidate.dtype() == baseline.dtype(), "tile shape/dtype changed");
                        require(relative_l2(candidate, baseline) < 2e-6f, "tile recipe changed FP32 accumulation");
                        ++cases;
                    }
                }
            }
            auto input = mx::zeros({1,33,128}, dtype);
            auto reject = [&](auto &&operation) {
                try { operation(); } catch (const std::invalid_argument &) { ++rejected; return; }
                throw std::runtime_error("invalid dense projection admitted");
            };
            reject([&] { dense_gpu::projection_range(input, weight, -1,128,0,128); });
            reject([&] { dense_gpu::projection_range(input, weight, 0,161,0,128); });
            reject([&] { dense_gpu::projection_range(input, weight, 0,128,0,127); });
            reject([&] { dense_gpu::projection_range(input, weight, 0,128,129,257); });
            reject([&] { dense_gpu::projection_range(input, weight, 0,128,0,128,8); });
            reject([&] { dense_gpu::projection_range(input, weight, 0,128,0,128,32,false,32); });
            reject([&] { dense_gpu::projection_range(mx::astype(input,mx::float32),weight,0,128,0,128); });
            auto noncontiguous_weight = mx::transpose(weight);
            mx::eval(noncontiguous_weight);
            reject([&] { dense_gpu::projection_range(input,noncontiguous_weight,0,128,0,128); });
        }
        std::cout << "PASS dense GPU projection cases=" << cases << " rejected=" << rejected << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#include "../../native/backends/mlx.hpp"

#include <iostream>

namespace {

float relative_l2(const tc::Tensor &actual, const tc::Tensor &expected) {
    auto delta = tc::mx::astype(actual, tc::mx::float32) -
                 tc::mx::astype(expected, tc::mx::float32);
    auto relative = tc::mx::sqrt(
        tc::mx::sum(delta * delta) /
        tc::mx::maximum(tc::mx::sum(tc::mx::astype(expected, tc::mx::float32) *
                                    tc::mx::astype(expected, tc::mx::float32)),
                        tc::Tensor(1e-20f)));
    tc::mx::eval(relative);
    return relative.item<float>();
}

void require_finite(const tc::Tensor &value, const std::string &name) {
    auto finite = tc::mx::all(tc::mx::isfinite(value));
    tc::mx::eval(finite);
    tc::require(finite.item<bool>(), name + " contains nonfinite values");
}

} // namespace

int main() {
    try {
        tc::configure_streams();
        constexpr int input_width = 128;
        constexpr int output_width = 96;
        constexpr int token_rows = 7;
        constexpr int row_start = 16;
        constexpr int row_end = 80;
        constexpr int column_split = 64;

        auto weight = tc::mx::reshape(
            tc::mx::sin(tc::mx::arange(output_width * input_width,
                                       tc::mx::float32) *
                        tc::Tensor(0.013f)),
            {output_width, input_width});
        auto bias = tc::mx::cos(tc::mx::arange(output_width, tc::mx::float32) *
                                tc::Tensor(0.017f));
        auto input = tc::mx::reshape(
            tc::mx::cos(tc::mx::arange(token_rows * input_width,
                                       tc::mx::float32) *
                        tc::Tensor(0.019f)),
            {1, token_rows, input_width});

        tc::Weights dense;
        dense.bind_arrays({"dense.weight", "dense.bias"}, {weight, bias});
        auto dense_full = dense.project(input, "dense");
        auto dense_rows = dense.project_slice(input, "dense", row_start, row_end,
                                              0, input_width);
        auto dense_rows_reference =
            tc::slice_axis(dense_full, -1, row_start, row_end);
        auto dense_left = dense.project_slice(
            tc::slice_axis(input, -1, 0, column_split), "dense", 0,
            output_width, 0, column_split, true);
        auto dense_right = dense.project_slice(
            tc::slice_axis(input, -1, column_split, input_width), "dense", 0,
            output_width, column_split, input_width, false);
        auto dense_partitioned = dense_left + dense_right;
        tc::mx::eval(dense_full, dense_rows, dense_partitioned);
        require_finite(dense_partitioned, "dense partition");
        const float dense_row_relative =
            relative_l2(dense_rows, dense_rows_reference);
        const float dense_partition_relative =
            relative_l2(dense_partitioned, dense_full);
        std::cerr << "dense row=" << dense_row_relative
                  << " partition=" << dense_partition_relative << std::endl;
        tc::require(dense_row_relative < 1e-5f,
                    "dense output-row slice differs from full projection");
        tc::require(dense_partition_relative < 1e-5f,
                    "dense input-column partition differs from full projection");

        auto packed = tc::mx::quantize(tc::mx::astype(weight, tc::mx::float16),
                                       32, 4, "affine");
        tc::require(packed.size() == 3,
                    "affine quantize did not return weight/scales/biases");
        tc::Weights quantized;
        quantized.bind_arrays(
            {"q.weight", "q.scales", "q.biases", "q.bias"},
            {packed[0], packed[1], packed[2],
             tc::mx::astype(bias, tc::mx::float16)});
        auto q_input = tc::mx::astype(input, tc::mx::float16);
        auto q_full = quantized.project(q_input, "q");
        auto q_rows = quantized.project_slice(q_input, "q", row_start, row_end,
                                              0, input_width);
        auto q_rows_reference = tc::slice_axis(q_full, -1, row_start, row_end);
        auto q_left = quantized.project_slice(
            tc::slice_axis(q_input, -1, 0, column_split), "q", 0,
            output_width, 0, column_split, true);
        auto q_right = quantized.project_slice(
            tc::slice_axis(q_input, -1, column_split, input_width), "q", 0,
            output_width, column_split, input_width, false);
        auto q_partitioned = q_left + q_right;
        tc::mx::eval(q_full, q_rows, q_partitioned);
        require_finite(q_partitioned, "quantized partition");
        const float q_row_relative = relative_l2(q_rows, q_rows_reference);
        const float q_partition_relative = relative_l2(q_partitioned, q_full);
        std::cerr << "q4 row=" << q_row_relative
                  << " partition=" << q_partition_relative << std::endl;
        tc::require(q_row_relative < 1e-6f,
                    "quantized output-row slice differs from full projection");
        tc::require(q_partition_relative < 5e-3f,
                    "quantized input-column partition differs from full projection");

        bool rejected_unaligned = false;
        try {
            (void)quantized.project_slice(
                tc::slice_axis(q_input, -1, 0, 48), "q", 0, output_width,
                0, 48, false);
        } catch (const std::exception &) {
            rejected_unaligned = true;
        }
        tc::require(rejected_unaligned,
                    "quantized projection accepted an unaligned column slice");

        std::cout << "{\"dense_row_relative_l2\":" << dense_row_relative
                  << ",\"dense_partition_relative_l2\":"
                  << dense_partition_relative
                  << ",\"q4_row_relative_l2\":" << q_row_relative
                  << ",\"q4_partition_relative_l2\":"
                  << q_partition_relative
                  << ",\"unaligned_rejected\":true}" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

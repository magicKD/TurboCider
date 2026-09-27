#include "../../native/backends/mlx.hpp"

#include <iostream>
#include <unistd.h>

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

        // Exercise the actual safetensors binding path: Qwen21's separately
        // trained gate/proj adapters share one fused [gate; up] base matrix.
        // A row slice crossing that boundary must intersect each adapter's
        // output rows and its selected input columns independently.
        struct Fixture {
            std::filesystem::path path = std::filesystem::temp_directory_path() /
                ("turbocider-lora-slice-" + std::to_string(getpid()) + ".safetensors");
            ~Fixture() { std::error_code error; std::filesystem::remove(path, error); }
        } fixture;
        const auto adapter_down = tc::mx::reshape(
            tc::mx::sin(tc::mx::arange(8 * input_width, tc::mx::float32) *
                        tc::Tensor(0.047f)) * tc::Tensor(0.07f), {8, input_width});
        const auto adapter_up = tc::mx::reshape(
            tc::mx::cos(tc::mx::arange(output_width * 8, tc::mx::float32) *
                        tc::Tensor(0.031f)) * tc::Tensor(0.10f), {output_width, 8});
        const auto second_down = adapter_down * tc::Tensor(-0.7f);
        const auto second_up = adapter_up * tc::Tensor(1.3f);
        const std::string stem = "transformer_blocks.0.img_mlp.";
        tc::mx::save_safetensors(fixture.path.string(), {
            {stem + "gate_layer.lora_A.weight", adapter_down},
            {stem + "gate_layer.lora_B.weight", adapter_up},
            {stem + "proj.lora_A.weight", second_down},
            {stem + "proj.lora_B.weight", second_up},
        });
        tc::Weights fused;
        const auto fused_weight = tc::mx::concatenate({weight, weight * tc::Tensor(-0.8f)}, 0);
        fused.bind_arrays({stem + "gate_up.weight"}, {fused_weight});
        std::atomic<bool> cancelled{false};
        const auto applied = fused.apply_loras(
            {{fixture.path.string(), 1.f, "transformer"}}, "transformer",
            [](const std::string &, int, int) {}, cancelled, true);
        tc::require(applied == 2, "fused gate/up runtime LoRA did not bind both branches");
        const auto full = fused.project(input, stem + "gate_up");
        tc::Weights base_fused;
        base_fused.bind_arrays({stem + "gate_up.weight"}, {fused_weight});
        const auto base_full = base_fused.project(input, stem + "gate_up");
        tc::require(relative_l2(tc::slice_axis(full, -1, 0, output_width),
                                tc::slice_axis(base_full, -1, 0, output_width)) > 1e-4f &&
                    relative_l2(tc::slice_axis(full, -1, output_width, 2 * output_width),
                                tc::slice_axis(base_full, -1, output_width,
                                               2 * output_width)) > 1e-4f,
                    "runtime LoRA test fixture does not affect both gate and up branches");
        const auto rows = fused.project_slice(input, stem + "gate_up", 48, 144,
                                               0, input_width);
        const auto left = fused.project_slice(
            tc::slice_axis(input, -1, 0, column_split), stem + "gate_up",
            48, 144, 0, column_split);
        const auto right = fused.project_slice(
            tc::slice_axis(input, -1, column_split, input_width), stem + "gate_up",
            48, 144, column_split, input_width);
        const float lora_rows = relative_l2(rows, tc::slice_axis(full, -1, 48, 144));
        const float lora_columns = relative_l2(left + right, rows);
        tc::require(lora_rows < 1e-5f && lora_columns < 1e-5f,
                    "runtime LoRA row/column slices disagree with full fused projection");
        fused.set_runtime_lora_fp16(true);
        const float half_rank_rows = relative_l2(
            fused.project_slice(input, stem + "gate_up", 48, 144, 0, input_width),
            tc::slice_axis(fused.project(input, stem + "gate_up"), -1, 48, 144));
        tc::require(half_rank_rows < 1e-5f,
                    "FP16 runtime LoRA row slice disagrees with full fused projection");

        std::cout << "{\"dense_row_relative_l2\":" << dense_row_relative
                  << ",\"dense_partition_relative_l2\":"
                  << dense_partition_relative
                  << ",\"q4_row_relative_l2\":" << q_row_relative
                  << ",\"q4_partition_relative_l2\":"
                  << q_partition_relative
                  << ",\"lora_rows_relative_l2\":" << lora_rows
                  << ",\"lora_columns_relative_l2\":" << lora_columns
                  << ",\"fp16_lora_rows_relative_l2\":" << half_rank_rows
                  << ",\"unaligned_rejected\":true}" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

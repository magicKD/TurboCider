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

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--cpu")
            tc::mx::set_default_device(tc::mx::Device(tc::mx::Device::cpu));
        else {
            tc::require(argc == 1, "usage: mlx-project-slice-probe [--cpu]");
            tc::configure_streams();
        }
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
        const auto row_base = fused.project_base_slice(input, stem + "gate_up",
            48, 144, 0, input_width, false);
        const auto supplied_rows = fused.apply_runtime_lora_slice(row_base, input,
            stem + "gate_up", 48, 144, 0, input_width);
        tc::require(relative_l2(supplied_rows, rows) == 0.f,
                    "supplied-base adapter additions changed the sliced projection");
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
        const auto delta_gate = fused.lora_delta_slice(
            input, stem + "gate_up", 0, output_width, 0, input_width);
        const auto delta_up = fused.lora_delta_slice(
            input, stem + "gate_up", output_width, 2 * output_width, 0, input_width);
        const auto delta_crossing = fused.lora_delta_slice(
            input, stem + "gate_up", 48, 144, 0, input_width);
        const auto absent = base_fused.lora_delta_slice(
            input, stem + "gate_up", 0, output_width, 0, input_width);
        tc::mx::eval(delta_gate, delta_up, delta_crossing, absent);
        const float gate_delta_relative = relative_l2(
            tc::slice_axis(base_full, -1, 0, output_width) + delta_gate,
            tc::slice_axis(full, -1, 0, output_width));
        const float up_delta_relative = relative_l2(
            tc::slice_axis(base_full, -1, output_width, 2 * output_width) + delta_up,
            tc::slice_axis(full, -1, output_width, 2 * output_width));
        tc::require(gate_delta_relative < 1e-5f && up_delta_relative < 1e-5f,
                    "adapter-only gate/up deltas disagree with runtime LoRA projection");
        tc::require(relative_l2(delta_crossing,
                    tc::slice_axis(full - base_full, -1, 48, 144)) < 1e-5f &&
                    tc::mx::sum(tc::mx::abs(absent)).item<float>() == 0.f,
                    "crossing/absent runtime LoRA delta slice differs from reference");
        // Correction readiness materializes the shared rank-sized branch.
        // The later GPU channel head must reuse both A projections, while
        // retaining the ordinary per-adapter accumulation/rounding contract.
        tc::Weights::LoRAWorkspace workspace;
        const auto shared_gate = fused.lora_delta_slice(input, stem + "gate_up",
            0, output_width, 0, input_width, std::nullopt, &workspace);
        const auto shared_up = fused.lora_delta_slice(input, stem + "gate_up",
            output_width, 2 * output_width, 0, input_width, std::nullopt, &workspace);
        tc::mx::eval(shared_gate, shared_up);
        tc::require(workspace.low_rank_projections() == 2,
                    "correction did not prepare exactly two adapter A projections");
        const auto shared_rows = fused.project_slice(input, stem + "gate_up",
            48, 144, 0, input_width, true, &workspace);
        const float workspace_rows = relative_l2(shared_rows, rows);
        tc::require(workspace.low_rank_projections() == 2 && workspace_rows == 0.f,
                    "GPU range recomputed adapter A or changed projection rounding");
        const auto different_input = input + tc::Tensor(0.03f);
        const auto new_input_rows = fused.project_slice(different_input, stem + "gate_up",
            48, 144, 0, input_width, true, &workspace);
        tc::require(relative_l2(new_input_rows,
                    fused.project_slice(different_input, stem + "gate_up",
                        48, 144, 0, input_width)) == 0.f &&
                    workspace.low_rank_projections() == 4,
                    "workspace reused a previous input's adapter activation");
        const auto shared_left = fused.project_slice(
            tc::slice_axis(input, -1, 0, column_split), stem + "gate_up",
            48, 144, 0, column_split, true, &workspace);
        tc::require(relative_l2(shared_left, left) == 0.f &&
                    workspace.low_rank_projections() == 6,
                    "workspace reused an incompatible input-column projection");
        // The optional Core ML boundary must preserve the FP32 low-rank
        // accumulation until its FP16 output, without changing the standard
        // BF16-returning slice contract or fabricating an absent adapter.
        auto bf16_input = tc::mx::astype(input, tc::mx::bfloat16);
        const auto direct_half = fused.lora_delta_slice(
            bf16_input, stem + "gate_up", 0, output_width, 0, input_width,
            tc::mx::float16);
        const auto reference_float = fused.lora_delta_slice(
            bf16_input, stem + "gate_up", 0, output_width, 0, input_width,
            tc::mx::float32);
        const auto default_bf16 = fused.lora_delta_slice(
            bf16_input, stem + "gate_up", 0, output_width, 0, input_width);
        const auto absent_half = base_fused.lora_delta_slice(
            bf16_input, stem + "gate_up", 0, output_width, 0, input_width,
            tc::mx::float16);
        const float direct_half_relative = relative_l2(
            direct_half, tc::mx::astype(reference_float, tc::mx::float16));
        const float absent_half_sum = tc::mx::sum(
            tc::mx::astype(tc::mx::abs(absent_half), tc::mx::float32)).item<float>();
        std::cerr << "direct FP16 delta relative=" << direct_half_relative
                  << " absent=" << absent_half_sum
                  << " dtypes_correct="
                  << (direct_half.dtype() == tc::mx::float16) << '/'
                  << (default_bf16.dtype() == tc::mx::bfloat16) << '/'
                  << (absent_half.dtype() == tc::mx::float16) << std::endl;
        tc::require(direct_half.dtype() == tc::mx::float16 &&
                        default_bf16.dtype() == tc::mx::bfloat16 &&
                        absent_half.dtype() == tc::mx::float16 &&
                        direct_half_relative == 0.f && absent_half_sum == 0.f,
                    "FP16 Core ML LoRA delta changed its FP32 reference or default dtype");
        tc::Weights::LoRAWorkspace bf16_workspace;
        const auto bf16_gate = fused.lora_delta_slice(bf16_input, stem + "gate_up",
            0, output_width, 0, input_width, std::nullopt, &bf16_workspace);
        const auto bf16_up = fused.lora_delta_slice(bf16_input, stem + "gate_up",
            output_width, 2 * output_width, 0, input_width, std::nullopt, &bf16_workspace);
        tc::mx::eval(bf16_gate, bf16_up);
        const auto bf16_rows = fused.project_slice(bf16_input, stem + "gate_up",
            48, 144, 0, input_width, true, &bf16_workspace);
        tc::require(relative_l2(bf16_rows,
                    fused.project_slice(bf16_input, stem + "gate_up", 48, 144, 0, input_width)) == 0.f &&
                    bf16_workspace.low_rank_projections() == 2,
                    "workspace changed the BF16 projection rounding contract");
        fused.set_runtime_lora_fp16(true);
        const float half_rank_rows = relative_l2(
            fused.project_slice(input, stem + "gate_up", 48, 144, 0, input_width),
            tc::slice_axis(fused.project(input, stem + "gate_up"), -1, 48, 144));
        tc::require(half_rank_rows < 1e-5f,
                    "FP16 runtime LoRA row slice disagrees with full fused projection");
        const auto shared_half = fused.project_slice(input, stem + "gate_up",
            48, 144, 0, input_width, true, &workspace);
        tc::require(relative_l2(shared_half,
                    fused.project_slice(input, stem + "gate_up", 48, 144, 0, input_width)) == 0.f &&
                    workspace.low_rank_projections() == 8,
                    "workspace reused the previous LoRA rank precision");
        // Appending adapters on the same owner must miss the previous A
        // identity, even if the model/input geometry stays unchanged.
        const auto rebound = fused.apply_loras(
            {{fixture.path.string(), 0.5f, "transformer"}}, "transformer",
            [](const std::string &, int, int) {}, cancelled, true);
        tc::require(rebound == 2, "second runtime adapter did not bind");
        const auto shared_rebound = fused.project_slice(input, stem + "gate_up",
            48, 144, 0, input_width, true, &workspace);
        tc::require(relative_l2(shared_rebound,
                    fused.project_slice(input, stem + "gate_up", 48, 144, 0, input_width)) == 0.f &&
                    workspace.low_rank_projections() == 10,
                    "workspace reused stale adapter weights after rebinding");
        tc::require(relative_l2(fused.apply_runtime_lora_slice(row_base, input,
                    stem + "gate_up", 48, 144, 0, input_width, true, &workspace),
                    shared_rebound) == 0.f,
                    "supplied-base path changed stacked adapter rounding/order");
        tc::Weights bf16_fused;
        bf16_fused.bind_arrays({stem + "gate_up.weight"},
                              {tc::mx::astype(fused_weight, tc::mx::bfloat16)});
        tc::require(bf16_fused.apply_loras(
                    {{fixture.path.string(), 1.f, "transformer"},
                     {fixture.path.string(), 0.5f, "transformer"}}, "transformer",
                    [](const std::string &, int, int) {}, cancelled, true) == 4,
                    "stacked BF16 test adapters did not bind");
        const auto bf16_base = bf16_fused.project_base_slice(bf16_input,
            stem + "gate_up", 48, 144, 0, input_width, false);
        const auto bf16_supplied = bf16_fused.apply_runtime_lora_slice(bf16_base,
            bf16_input, stem + "gate_up", 48, 144, 0, input_width);
        tc::require(bf16_supplied.dtype() == tc::mx::bfloat16 &&
                    relative_l2(bf16_supplied, bf16_fused.project_slice(bf16_input,
                        stem + "gate_up", 48, 144, 0, input_width)) == 0.f,
                    "supplied-base path changed BF16 rounding between stacked adapters");
        // Channel Runtime prepares only the ANE tail, then projects the GPU
        // complement. Both must share A but keep their disjoint B rows. This
        // exercises the fused gate/up output mapping and stacked-adapter
        // rounding without retaining full correction matrices in a cache.
        auto check_disjoint_correction = [&](tc::Weights &owner, const tc::Tensor &x) {
            constexpr int gpu_channels = 64;
            tc::Weights::LoRAWorkspace split_workspace;
            auto gate_tail = owner.lora_delta_slice(x, stem + "gate_up",
                gpu_channels, output_width, 0, input_width, std::nullopt, &split_workspace);
            auto up_tail = owner.lora_delta_slice(x, stem + "gate_up",
                output_width + gpu_channels, 2 * output_width, 0, input_width,
                std::nullopt, &split_workspace);
            tc::mx::eval(gate_tail, up_tail);
            const auto a_projections = split_workspace.low_rank_projections();
            auto gate_head = owner.project_slice(x, stem + "gate_up", 0, gpu_channels,
                0, input_width, false, &split_workspace);
            auto up_head = owner.project_slice(x, stem + "gate_up", output_width,
                output_width + gpu_channels, 0, input_width, false, &split_workspace);
            tc::require(relative_l2(gate_tail, tc::slice_axis(owner.lora_delta_slice(x,
                            stem + "gate_up", 0, output_width, 0, input_width),
                            -1, gpu_channels, output_width)) == 0.f &&
                        relative_l2(up_tail, tc::slice_axis(owner.lora_delta_slice(x,
                            stem + "gate_up", output_width, 2 * output_width, 0, input_width),
                            -1, gpu_channels, output_width)) == 0.f &&
                        relative_l2(gate_head, owner.project_slice(x, stem + "gate_up",
                            0, gpu_channels, 0, input_width, false)) == 0.f &&
                        relative_l2(up_head, owner.project_slice(x, stem + "gate_up",
                            output_width, output_width + gpu_channels, 0, input_width, false)) == 0.f &&
                        a_projections > 0 && split_workspace.low_rank_projections() == a_projections,
                        "disjoint channel LoRA changed corrections or repeated shared A projections");
            return a_projections;
        };
        check_disjoint_correction(fused, input);
        check_disjoint_correction(bf16_fused, bf16_input);
        // A genuinely fused [2F, rank] B owns one A for both halves. The ANE
        // tail and GPU head must all reuse that single rank-sized projection.
        Fixture single_fused_fixture;
        single_fused_fixture.path = fixture.path.parent_path() /
            ("turbocider-lora-slice-fused-" + std::to_string(getpid()) + ".safetensors");
        tc::mx::save_safetensors(single_fused_fixture.path.string(), {
            {stem + "gate_up.lora_A.weight", adapter_down},
            {stem + "gate_up.lora_B.weight", tc::mx::concatenate({adapter_up, second_up}, 0)},
        });
        tc::Weights single_fused;
        single_fused.bind_arrays({stem + "gate_up.weight"},
                                 {tc::mx::astype(fused_weight, tc::mx::bfloat16)});
        tc::require(single_fused.apply_loras(
                    {{single_fused_fixture.path.string(), 1.f, "transformer"}}, "transformer",
                    [](const std::string &, int, int) {}, cancelled, true) == 1,
                    "single fused gate/up adapter did not bind exactly one A/B pair");
        tc::require(check_disjoint_correction(single_fused, bf16_input) == 1,
                    "fused gate/up channel slices did not share exactly one A projection");
        const auto dense_base_rows = dense.project_base_slice(input, "dense",
            row_start, row_end, 0, input_width, false);
        tc::require(relative_l2(dense.apply_runtime_lora_slice(dense_base_rows,
                    input, "dense", row_start, row_end, 0, input_width), dense_rows) == 0.f,
                    "supplied-base path omitted or duplicated the bias");
        bool rejected_base_shape = false;
        try {
            (void)fused.apply_runtime_lora_slice(row_base, input, stem + "gate_up",
                0, output_width / 2, 0, input_width);
        } catch (const std::exception &) { rejected_base_shape = true; }
        tc::require(rejected_base_shape,
                    "supplied-base path accepted mismatched output geometry");

        std::cout << "{\"dense_row_relative_l2\":" << dense_row_relative
                  << ",\"dense_partition_relative_l2\":"
                  << dense_partition_relative
                  << ",\"q4_row_relative_l2\":" << q_row_relative
                  << ",\"q4_partition_relative_l2\":"
                  << q_partition_relative
                  << ",\"lora_rows_relative_l2\":" << lora_rows
                  << ",\"lora_columns_relative_l2\":" << lora_columns
                  << ",\"fp16_lora_rows_relative_l2\":" << half_rank_rows
                  << ",\"gate_delta_relative_l2\":" << gate_delta_relative
                  << ",\"up_delta_relative_l2\":" << up_delta_relative
                  << ",\"workspace_rows_relative_l2\":" << workspace_rows
                  << ",\"workspace_identity_checks\":true"
                  << ",\"supplied_base_identity_checks\":true"
                  << ",\"disjoint_channel_correction_checks\":true"
                  << ",\"unaligned_rejected\":true}" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

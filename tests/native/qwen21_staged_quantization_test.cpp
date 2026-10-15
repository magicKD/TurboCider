#include "../../native/models/qwen21/staged_suffix_quantization.hpp"
#include "../../native/models/qwen21/hybrid_merge.hpp"

#include <atomic>
#include <cmath>
#include <iostream>
#include <optional>

namespace {
namespace mx = tc::mx;
using tc::Tensor;
using tc::qwen21::StagedQuantizedSuffix;

constexpr int hidden = 128;
constexpr int full_width = 256;
constexpr int ane_width = 128;
constexpr int suffix_width = full_width - ane_width;

// Attach the counter to the actual MLX backing buffer, rather than the raw
// pointer array constructor (which is allowed to copy its input immediately).
// A surviving lazy view or quantization dependency keeps this deleter alive.
Tensor tracked_dense(int rows, int columns, int seed,
                     const std::shared_ptr<std::atomic<int>> &released) {
    auto storage = mx::allocator::malloc(size_t(rows) * columns * sizeof(mx::bfloat16_t));
    auto *values = static_cast<mx::bfloat16_t *>(storage.raw_ptr());
    tc::require(values != nullptr, "fixture weight allocation failed");
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const float value = .035f * std::sin(float((row + 1) * (column + 3)) * .017f + seed)
                + .018f * std::cos(float(row * 11 + column * 7 + seed) * .031f)
                + .001f * (row % 7 - 3);
            values[size_t(row) * columns + column] = mx::bfloat16_t(value);
        }
    }
    return Tensor(storage, {rows, columns}, mx::bfloat16,
        [released](mx::allocator::Buffer buffer) {
            mx::allocator::free(buffer);
            released->fetch_add(1);
        });
}

Tensor activation(int rows) {
    std::vector<float> values(size_t(rows) * hidden);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = .2f * std::sin(float(i + 1) * .37f)
                  + .07f * std::cos(float(i + 3) * .11f);
    return mx::astype(Tensor(values.data(), {1, rows, hidden}, mx::float32), mx::bfloat16);
}

// Keep this oracle independent of the staged forward helper and its argument
// assembly. It uses separately quantized arrays with the same documented
// affine group-64/FP16 arithmetic, not a comparison against dense weights.
Tensor quantized_reference(const std::vector<Tensor> &a, int bits) {
    auto gate_up = mx::quantized_matmul(a.at(0), a.at(1), a.at(2), a.at(3),
                                       true, 64, bits, "affine");
    auto gate_and_up = mx::split(gate_up, 2, -1);
    auto hidden_state = tc::silu(gate_and_up.at(0)) * gate_and_up.at(1);
    return mx::quantized_matmul(hidden_state, a.at(4), a.at(5), a.at(6),
                                true, 64, bits, "affine");
}

Tensor dense_reference(const Tensor &input, const Tensor &gate_up, const Tensor &down) {
    auto pieces = mx::split(mx::matmul(input, mx::transpose(gate_up)), 2, -1);
    return mx::matmul(tc::silu(pieces[0]) * pieces[1], mx::transpose(down));
}

void same(const Tensor &actual, const Tensor &expected, const std::string &message) {
    tc::require(actual.shape() == expected.shape() && actual.dtype() == expected.dtype(),
                message + " (shape/dtype)");
    tc::require(mx::array_equal(actual, expected).item<bool>(), message);
}

void inspect_cache(const StagedQuantizedSuffix &cache,
                   const std::vector<Tensor> &reference, int bits) {
    tc::require(cache.bits == bits && cache.group_size == 64, "cache quantization metadata changed");
    const std::array<mx::Shape, 6> shapes{{
        {2 * suffix_width, hidden * bits / 32},
        {2 * suffix_width, hidden / 64},
        {2 * suffix_width, hidden / 64},
        {hidden, suffix_width * bits / 32},
        {hidden, suffix_width / 64},
        {hidden, suffix_width / 64},
    }};
    uint64_t actual_bytes = 0;
    for (size_t i = 0; i < cache.tensors.size(); ++i) {
        const auto &tensor = cache.tensors[i];
        const auto dtype = i % 3 == 0 ? mx::uint32 : mx::float16;
        tc::require(tensor.shape() == shapes[i] && tensor.dtype() == dtype,
                    "packed/scales/biases shape or dtype changed at array " + std::to_string(i));
        tc::require(tensor.flags().row_contiguous &&
                        tensor.data_size() * tensor.itemsize() == tensor.nbytes(),
                    "cache has a strided or oversized backing span");
        tc::require(tensor.inputs().empty(), "cache retains a lazy dense source dependency");
        same(tensor, reference.at(i), "cache differs from independent affine quantization");
        actual_bytes += tensor.data_size() * tensor.itemsize();
    }
    const uint64_t elements = uint64_t(3) * suffix_width * hidden;
    const uint64_t expected_bytes = elements * bits / 8 + elements / 64 * 2 * sizeof(uint16_t);
    tc::require(actual_bytes == expected_bytes && cache.bytes() == actual_bytes,
                "cache byte accounting omitted packed weights, scales or biases");
    tc::require(actual_bytes < elements * sizeof(mx::bfloat16_t),
                "quantized cache did not reduce BF16 dense bytes");
}

struct CaseResult {
    uint64_t bytes;
    float max_dense_error;
    int forwards;
};

CaseResult exercise(int bits) {
    auto released = std::make_shared<std::atomic<int>>(0);
    std::optional<StagedQuantizedSuffix> cache;
    std::vector<Tensor> independently_quantized;
    std::vector<Tensor> inputs, dense_outputs, ane_outputs;
    for (int rows : {1, 7, 17}) inputs.push_back(activation(rows));
    {
        auto fused = tracked_dense(2 * full_width, hidden, 13, released);
        auto down = tracked_dense(hidden, full_width, 29, released);
        auto original_fused = mx::copy(fused), original_down = mx::copy(down);
        mx::eval(original_fused, original_down);

        auto upper = mx::contiguous(mx::concatenate({
            tc::slice_axis(fused, 0, ane_width, full_width),
            tc::slice_axis(fused, 0, full_width + ane_width, 2 * full_width)}, 0));
        auto lower = mx::contiguous(tc::slice_axis(down, 1, ane_width, full_width));
        cache.emplace(tc::qwen21::quantize_staged_suffix(upper, lower, bits));

        auto expected_upper = mx::quantize(mx::astype(upper, mx::float16), 64, bits, "affine");
        auto expected_lower = mx::quantize(mx::astype(lower, mx::float16), 64, bits, "affine");
        independently_quantized = {expected_upper[0], expected_upper[1], expected_upper[2],
                                  expected_lower[0], expected_lower[1], expected_lower[2]};
        mx::eval(independently_quantized);
        inspect_cache(*cache, independently_quantized, bits);

        auto ane_upper = mx::contiguous(mx::concatenate({
            tc::slice_axis(fused, 0, 0, ane_width),
            tc::slice_axis(fused, 0, full_width, full_width + ane_width)}, 0));
        auto ane_lower = mx::contiguous(tc::slice_axis(down, 1, 0, ane_width));
        for (const auto &input : inputs) {
            dense_outputs.push_back(dense_reference(mx::astype(input, mx::float16),
                mx::astype(upper, mx::float16), mx::astype(lower, mx::float16)));
            // This is a deterministic stand-in for the borrowed FP16 Core ML
            // output. It checks the join ABI, not ANE placement or W8A8 quality.
            ane_outputs.push_back(mx::astype(dense_reference(input, ane_upper, ane_lower),
                                             mx::float16));
        }
        mx::eval(dense_outputs);
        mx::eval(ane_outputs);
        same(fused, original_fused, "suffix quantization mutated fused weights/ANE prefix");
        same(down, original_down, "suffix quantization mutated down weights/ANE prefix");

        // 0 is the existing dense BF16 HybridMLP branch; it must not enter
        // the new quantization helper. Check fail-closed helper behavior and
        // verify the original BF16 expression/source survive the rejection.
        auto before_zero = dense_reference(inputs[0], upper, lower);
        mx::eval(before_zero);
        bool rejected_zero = false;
        try {
            (void)tc::qwen21::quantize_staged_suffix(upper, lower, 0);
        } catch (const std::invalid_argument &) {
            rejected_zero = true;
        }
        tc::require(rejected_zero, "0-bit dense branch entered quantization");
        auto after_zero = dense_reference(inputs[0], upper, lower);
        tc::require(after_zero.dtype() == mx::bfloat16, "0-bit expression changed BF16 precision");
        same(after_zero, before_zero, "0-bit rejection changed original dense BF16 output");
    }
    tc::require(released->load() == 2,
                "packed suffix still owns full dense source storage after layer release");

    auto compiled_actual = mx::compile([bits](const std::vector<Tensor> &args) {
        return std::vector<Tensor>{tc::qwen21::staged_quantized_suffix_forward(args, bits)};
    });
    auto compiled_reference = mx::compile([bits](const std::vector<Tensor> &args) {
        return std::vector<Tensor>{quantized_reference(args, bits)};
    });
    CaseResult result{cache->bytes(), 0.f, 0};
    const std::array<std::uintptr_t, 6> cache_ids{{
        cache->tensors[0].id(), cache->tensors[1].id(), cache->tensors[2].id(),
        cache->tensors[3].id(), cache->tensors[4].id(), cache->tensors[5].id(),
    }};
    for (size_t case_index = 0; case_index < inputs.size(); ++case_index) {
        auto packed_input = mx::astype(inputs[case_index], mx::float16);
        const auto &q = independently_quantized;
        std::vector<Tensor> reference_args{packed_input, q[0], q[1], q[2], q[3], q[4], q[5]};
        auto expected_eager = quantized_reference(reference_args, bits);
        auto expected_compiled = compiled_reference(reference_args)[0];
        mx::eval(expected_eager, expected_compiled);
        for (int repeat = 0; repeat < 3; ++repeat) {
            auto actual = tc::qwen21::staged_quantized_suffix_forward(cache->arguments(packed_input), bits);
            auto compiled = compiled_actual(cache->arguments(packed_input))[0];
            mx::eval(actual, compiled);
            tc::require(actual.dtype() == mx::float16 && compiled.dtype() == mx::float16,
                        "W4/W8 suffix activation precision changed");
            same(actual, expected_eager, "eager cached suffix differs elementwise after source release");
            same(compiled, expected_compiled, "compiled cached suffix differs elementwise after source release");
            for (size_t i = 0; i < cache_ids.size(); ++i)
                tc::require(cache->tensors[i].id() == cache_ids[i], "forward replaced/requantized the cache");

            auto bf16_suffix = mx::astype(compiled, mx::bfloat16);
            for (float scale : {1.f, 1.25f}) {
                auto merged = tc::qwen21::merge_mlp_partitions(
                    bf16_suffix, ane_outputs[case_index], scale);
                auto expected_merged = mx::astype(expected_compiled, mx::bfloat16)
                    + mx::astype(ane_outputs[case_index], mx::bfloat16)
                      * Tensor(scale, mx::bfloat16);
                same(merged, expected_merged, "FP16 suffix cast back to BF16 changed the hybrid join");
                tc::require(merged.dtype() == inputs[case_index].dtype(), "hybrid join promoted BF16");
            }
            result.forwards += 2;
        }
        auto difference = mx::max(mx::abs(mx::astype(expected_compiled, mx::float32)
                                          - mx::astype(dense_outputs[case_index], mx::float32)));
        const float error = difference.item<float>();
        tc::require(std::isfinite(error), "quantized-vs-dense error is not finite");
        result.max_dense_error = std::max(result.max_dense_error, error);
    }
    tc::require(result.max_dense_error > 0.f,
                "fixture did not distinguish the quantized computation from dense FP16");
    tc::require(released->load() == 2, "repeated forward changed dense-source ownership");

    // A same-shape bank from a later request must remain a runtime argument,
    // rather than a constant captured by the first compiled invocation.
    std::optional<StagedQuantizedSuffix> replacement;
    {
        auto next_upper = tracked_dense(2 * suffix_width, hidden, 47, released);
        auto next_lower = tracked_dense(hidden, suffix_width, 61, released);
        replacement.emplace(tc::qwen21::quantize_staged_suffix(next_upper, next_lower, bits));
    }
    for (const auto &tensor : replacement->tensors)
        tc::require(tensor.inputs().empty() && tensor.flags().row_contiguous &&
                        tensor.data_size() * tensor.itemsize() == tensor.nbytes(),
                    "replacement bank retained a source graph or oversized backing");
    // Evaluated packed arrays are detached, but the completed Metal command
    // buffer can still own source arguments until the stream is drained. The
    // first bank's scalar comparisons already supplied this barrier; this
    // bank has no consumer yet. Count source deleters only after that drain.
    mx::synchronize();
    tc::require(released->load() == 4, "replacement bank retained its dense source");
    const auto next_args = replacement->arguments(mx::astype(inputs[0], mx::float16));
    auto next_output = compiled_actual(next_args)[0];
    auto next_expected = compiled_reference(next_args)[0];
    same(next_output, next_expected, "compiled suffix captured the previous request's bank");
    auto old_output = compiled_actual(cache->arguments(mx::astype(inputs[0], mx::float16)))[0];
    tc::require(!mx::array_equal(next_output, old_output).item<bool>(),
                "replacement fixture did not distinguish the two request banks");

    bool rejected_bf16_input = false;
    try {
        (void)tc::qwen21::staged_quantized_suffix_forward(cache->arguments(inputs[0]), bits);
    } catch (const std::invalid_argument &) {
        rejected_bf16_input = true;
    }
    tc::require(rejected_bf16_input, "suffix QMM accepted implicit BF16/FP16 promotion");
    return result;
}
} // namespace

int main() {
    try {
        tc::configure_streams();
        mx::set_cache_limit(0);
        const auto w4 = exercise(4), w8 = exercise(8);
        tc::require(w4.bytes < w8.bytes, "W4 cache must be smaller than W8 including metadata");
        std::cout << "PASS Qwen21 staged quantization: group=64 arrays=6 W4_bytes=" << w4.bytes
                  << " W8_bytes=" << w8.bytes << " repeated_forwards=" << w4.forwards + w8.forwards
                  << " W4_dense_max_abs=" << w4.max_dense_error
                  << " W8_dense_max_abs=" << w8.max_dense_error << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

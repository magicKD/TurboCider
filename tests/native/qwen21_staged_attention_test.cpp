#include "../../native/models/qwen21/staged_attention_quantization.hpp"
#include "../../native/models/qwen21/transformer.hpp"
#include "../../native/models/qwen21/metal/qk_norm_rope.hpp"
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>

namespace {
namespace mx = tc::mx;
using tc::Tensor;
constexpr int width = 128;

Tensor source_weight(int seed, const std::shared_ptr<std::atomic<int>> &released) {
    auto memory = mx::allocator::malloc(size_t(width) * width * sizeof(mx::bfloat16_t));
    auto *values = static_cast<mx::bfloat16_t *>(memory.raw_ptr());
    tc::require(values != nullptr, "attention fixture allocation failed");
    for (int row = 0; row < width; ++row)
        for (int column = 0; column < width; ++column)
            values[size_t(row) * width + column] = mx::bfloat16_t(
                .07f * std::sin(float(row * 17 + column * 11 + seed) * .019f) +
                .04f * std::cos(float(row * column + seed) * .011f));
    return Tensor(memory, {width, width}, mx::bfloat16,
        [released](mx::allocator::Buffer buffer) {
            mx::allocator::free(buffer);
            released->fetch_add(1);
        });
}

Tensor input_for(int rows) {
    std::vector<float> values(size_t(rows) * width);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = .23f * std::sin(float(i + 3) * .031f) + .12f * std::cos(float(i + 1) * .017f);
    return mx::astype(Tensor(values.data(), {1, rows, width}, mx::float32), mx::bfloat16);
}

void equal(const Tensor &actual, const Tensor &expected, const char *why) {
    tc::require(actual.shape() == expected.shape() && actual.dtype() == expected.dtype() &&
                    mx::array_equal(actual, expected).item<bool>(), why);
}

template<class F> void rejects(F fn, const char *why) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument &) { rejected = true; }
    tc::require(rejected, why);
}

void configuration() {
    tc::Weights empty;
    tc::qwen21::Transformer transformer(empty);
    for (int bits : {-1, 3, 16})
        rejects([&] { transformer.configure_staged_optimizations(bits, false); },
                "unsupported attention bit width accepted");
    for (int bits : {4, 8, 0}) {
        transformer.configure_staged_optimizations(bits, false);
        tc::require(transformer.staged_attention_bits() == bits &&
                        transformer.staged_attention_cached_layers() == 0 &&
                        transformer.staged_attention_cache_bytes() == 0 &&
                        transformer.staged_attention_pack_seconds() == 0,
                    "attention reconfiguration retained an old bank");
    }
    auto config = tc::qwen21::TransformerConfig{};
    config.layers = 1;
    tc::qwen21::Transformer nonstandard(empty, config);
    nonstandard.configure_staged_optimizations(0, false);
    rejects([&] { nonstandard.configure_staged_optimizations(4, false); },
            "attention cache accepted nonstandard transformer geometry");
    rejects([&] { nonstandard.configure_staged_optimizations(0, true); },
            "Q/K fusion accepted nonstandard transformer geometry");
}

void composed_qk(const Tensor &q_projection, const Tensor &k_projection) {
    // Repeat a tiny real QMM projection across 32 heads to exercise the
    // production Q/K kernel ABI without allocating full 4096-square weights.
    auto q = mx::concatenate(std::vector<Tensor>(32, q_projection), -1);
    auto k = mx::concatenate(std::vector<Tensor>(32, k_projection), -1);
    auto qw = mx::full({128}, 1.f, mx::bfloat16);
    auto kw = mx::full({128}, .75f, mx::bfloat16);
    const int rows = q_projection.shape(1);
    auto phase = mx::reshape(mx::arange(0, rows * 64, mx::float32) / 133.f, {rows, 64});
    auto cosine = mx::cos(phase), sine = mx::sin(phase);
    auto qq = mx::fast::rms_norm(tc::heads(q, 32, 128), qw, 1e-6f);
    auto kk = mx::fast::rms_norm(tc::heads(k, 32, 128), kw, 1e-6f);
    auto expected = tc::rope_pairs_pair(qq, kk, cosine, sine);
    auto actual = tc::qwen21::metal::prepare_qk(q, k, qw, kw, cosine, sine, 1e-6f);
    for (size_t part = 0; part < 2; ++part) {
        tc::require(actual[part].shape() == mx::Shape{1, 32, rows, 128} &&
                        actual[part].dtype() == mx::bfloat16,
                    "quantized attention and Q/K fusion changed the BF16 layout");
        auto a = mx::astype(actual[part], mx::float32);
        auto b = mx::astype(expected[part], mx::float32);
        auto relative = mx::sqrt(mx::sum(mx::square(a - b)) /
                                 mx::sum(mx::square(b))).item<float>();
        tc::require(std::isfinite(relative) && relative < .01f,
                    "quantized attention and Q/K fusion exceeded the kernel tolerance");
    }
}

std::vector<Tensor> independent_forward(const std::vector<Tensor> &args, int bits) {
    std::vector<Tensor> output;
    // Independent public-MLX oracle; do not call the implementation helper.
    for (size_t offset : {1u, 4u, 7u, 10u})
        output.push_back(mx::quantized_matmul(args[0], args[offset], args[offset + 1],
                                              args[offset + 2], true, 64, bits, "affine"));
    return output;
}

struct Result { uint64_t bytes; int forwards; float dense_error; };
Result exercise(int bits) {
    auto released = std::make_shared<std::atomic<int>>(0);
    std::optional<tc::qwen21::StagedAttentionWeights> saved;
    std::vector<Tensor> reference, dense_outputs;
    std::vector<Tensor> inputs{input_for(1), input_for(7), input_for(17)};
    {
        std::array<Tensor, 4> original{{source_weight(3, released), source_weight(7, released),
                                      source_weight(13, released), source_weight(19, released)}};
        std::vector<Tensor> snapshots;
        for (const auto &weight : original) snapshots.push_back(mx::copy(weight));
        mx::eval(snapshots);
        saved.emplace(tc::qwen21::quantize_staged_attention(original, bits));
        tc::require(saved->bits == bits && saved->arrays.size() == 12, "attention cache ABI changed");
        for (const auto &weight : original) {
            auto pieces = mx::quantize(weight, 64, bits, "affine");
            reference.insert(reference.end(), pieces.begin(), pieces.end());
        }
        mx::eval(reference);
        uint64_t backing_bytes = 0;
        for (size_t i = 0; i < saved->arrays.size(); ++i) {
            const auto &value = saved->arrays[i];
            const auto expected_shape = i % 3 == 0 ? mx::Shape{width, width * bits / 32}
                                                    : mx::Shape{width, width / 64};
            tc::require(value.shape() == expected_shape &&
                            value.dtype() == (i % 3 == 0 ? mx::uint32 : mx::bfloat16),
                        "attention packed/scales/biases shape or dtype mismatch");
            tc::require(value.flags().row_contiguous && value.inputs().empty() &&
                            value.data_size() * value.itemsize() == value.nbytes(),
                        "attention cache retains source graph or oversized buffer");
            equal(value, reference[i], "attention packing differs from independent BF16 affine quantization");
            backing_bytes += value.data_size() * value.itemsize();
        }
        const uint64_t elements = uint64_t(4) * width * width;
        tc::require(backing_bytes == elements * bits / 8 + elements / 64 * 4 &&
                        saved->bytes() == backing_bytes, "attention cache byte accounting mismatch");
        for (size_t i = 0; i < original.size(); ++i)
            equal(original[i], snapshots[i], "attention packing mutated a BF16 source");
        for (const auto &input : inputs)
            dense_outputs.push_back(mx::matmul(input, mx::transpose(original[0])));
        mx::eval(dense_outputs);
        for (int bad_bits : {0, 3, 16})
            rejects([&] { (void)tc::qwen21::quantize_staged_attention(original, bad_bits); },
                    "bad attention bit width accepted");
        for (auto bad : {mx::zeros({96, 96}, mx::bfloat16), mx::zeros({128, 64}, mx::bfloat16),
                         mx::zeros({64, 64}, mx::bfloat16),
                         mx::zeros({128, 128}, mx::float16)}) {
            auto invalid = original;
            invalid[2] = bad;
            rejects([&] { (void)tc::qwen21::quantize_staged_attention(invalid, bits); },
                    "bad attention weight geometry/dtype accepted");
        }
    }
    tc::require(released->load() == 4, "attention packed bank still owns released dense layer weights");
    const auto compiled = mx::compile([bits](const std::vector<Tensor> &args) {
        std::vector<Tensor> output;
        for (size_t offset : {1u, 4u, 7u, 10u})
            output.push_back(tc::qwen21::staged_attention_project(args[0], args, offset, bits));
        return output;
    });
    const auto oracle = mx::compile([bits](const std::vector<Tensor> &args) {
        return independent_forward(args, bits);
    });
    std::vector<std::uintptr_t> ids;
    for (const auto &value : saved->arrays) ids.push_back(value.id());
    Result result{saved->bytes(), 0, 0.f};
    for (size_t case_index = 0; case_index < inputs.size(); ++case_index) {
        std::vector<Tensor> actual_args{inputs[case_index]}, reference_args{inputs[case_index]};
        actual_args.insert(actual_args.end(), saved->arrays.begin(), saved->arrays.end());
        reference_args.insert(reference_args.end(), reference.begin(), reference.end());
        auto expected_eager = independent_forward(reference_args, bits);
        auto expected_compiled = oracle(reference_args);
        mx::eval(expected_eager);
        mx::eval(expected_compiled);
        composed_qk(expected_compiled[0], expected_compiled[1]);
        for (int repeat = 0; repeat < 3; ++repeat) {
            auto actual = compiled(actual_args);
            mx::eval(actual);
            for (size_t projection = 0; projection < 4; ++projection) {
                auto eager = tc::qwen21::staged_attention_project(inputs[case_index], saved->arrays,
                                                                 projection * 3, bits);
                equal(eager, expected_eager[projection], "eager attention QMM differs after source release");
                equal(actual[projection], expected_compiled[projection],
                      "compiled attention QMM differs after source release");
                tc::require(eager.dtype() == mx::bfloat16 && actual[projection].dtype() == mx::bfloat16,
                            "attention QMM promoted BF16 activation/output");
                result.forwards += 2;
            }
            for (size_t i = 0; i < ids.size(); ++i)
                tc::require(saved->arrays[i].id() == ids[i], "attention forward replaced the packed bank");
        }
        auto error = mx::max(mx::abs(mx::astype(expected_compiled[0], mx::float32) -
                                    mx::astype(dense_outputs[case_index], mx::float32))).item<float>();
        tc::require(std::isfinite(error), "attention quantization produced nonfinite error");
        result.dense_error = std::max(result.dense_error, error);
    }
    rejects([&] { (void)tc::qwen21::staged_attention_project(inputs[0], saved->arrays, 0, 0); },
            "attention projection accepted disabled quantization bits");
    rejects([&] { (void)tc::qwen21::staged_attention_project(inputs[0], saved->arrays, 11, bits); },
            "attention projection accepted missing scale/bias arrays");
    rejects([&] { (void)tc::qwen21::staged_attention_project(inputs[0], saved->arrays,
                    std::numeric_limits<size_t>::max(), bits); },
            "attention projection accepted an overflowing offset");
    rejects([&] { (void)tc::qwen21::staged_attention_project(mx::astype(inputs[0], mx::float16),
                    saved->arrays, 0, bits); }, "attention projection accepted FP16 input");
    auto invalid_scales = saved->arrays;
    invalid_scales[1] = mx::astype(invalid_scales[1], mx::float16);
    rejects([&] { (void)tc::qwen21::staged_attention_project(inputs[0], invalid_scales, 0, bits); },
            "attention projection accepted FP16 scales");
    tc::require(result.dense_error > 0 && released->load() == 4,
                "attention fixture did not observe lossy output or changed source lifetime");
    return result;
}
} // namespace

int main() {
    try {
        tc::configure_streams();
        mx::set_cache_limit(0);
        configuration();
        auto q4 = exercise(4), q8 = exercise(8);
        tc::require(q4.bytes < q8.bytes, "W4 attention bank must be smaller than W8");
        std::cout << "PASS Qwen21 staged attention: group=64 arrays=12 W4_bytes=" << q4.bytes
                  << " W8_bytes=" << q8.bytes << " repeated_projection_forwards=" << q4.forwards + q8.forwards
                  << " W4_dense_max_abs=" << q4.dense_error << " W8_dense_max_abs=" << q8.dense_error << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

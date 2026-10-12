#include "../../native/backends/mlx.hpp"
#include <iostream>

using namespace tc;

static bool same(const Tensor &first, const Tensor &second) {
    return first.shape() == second.shape() && first.dtype() == second.dtype() &&
        mx::all(mx::view(first, mx::uint8) == mx::view(second, mx::uint8)).item<bool>();
}

int main() { try {
    configure_streams();
    for (int bits : {4, 8}) for (int group : {32, 64, 128}) {
        auto source = mx::astype(mx::reshape(mx::sin(mx::arange(8 * 256, mx::float32) * .13f), {8, 256}), mx::bfloat16);
        auto planes = mx::quantize(source, group, bits, "affine");
        mx::eval(planes);
        Weights packed;
        if (group != 32) packed.set_affine_quantization(group, bits);
        packed.bind_arrays({"p.weight", "p.scales", "p.biases"}, planes);
        Weights historical = packed, explicit_bf16 = packed;
        auto expected_fp16 = mx::dequantize(planes[0], planes[1], planes[2], group, bits, "affine", std::nullopt, mx::float16);
        auto expected_bf16 = mx::dequantize(planes[0], planes[1], planes[2], group, bits, "affine", std::nullopt, mx::bfloat16);
        mx::eval({expected_fp16, expected_bf16});
        historical.dequantize({"p"});
        explicit_bf16.dequantize({"p"}, mx::bfloat16);
        require(same(historical.at("p.weight"), expected_fp16), "historical FP16 decode changed");
        require(same(explicit_bf16.at("p.weight"), expected_bf16), "explicit BF16 decode differs from stored planes");
        require(packed.quantized("p") && packed.has("p.biases") &&
                    !historical.quantized("p") && !explicit_bf16.has("p.scales") &&
                    !explicit_bf16.has("p.biases"), "staged decode mutated source or retained stale planes");
        bool rejected = false;
        try { packed.dequantize({"p"}, mx::float32); } catch (const std::exception &) { rejected = true; }
        require(rejected && packed.quantized("p"), "unsupported dtype must reject before mutation");
        require(packed.logical_input_width("p") == 256, "declared logical width differs");
        const int32_t indexes[] = {7, 0, 3};
        auto ids = Tensor(indexes, {3}, mx::int32);
        require(same(packed.embedding_rows(ids, "p", mx::bfloat16), mx::take(expected_bf16, ids, 0)),
                "row-first BF16 quantized embedding differs");
        require(same(packed.embedding_rows(ids, "p"), mx::take(expected_fp16, ids, 0)),
                "historical FP16 quantized embedding differs");
        auto x = mx::astype(mx::reshape(mx::cos(mx::arange(4 * 256, mx::float32) * .1f), {4, 256}), mx::bfloat16);
        require(same(packed.project(x, "p"), mx::quantized_matmul(x, planes[0], planes[1], planes[2], true, group, bits, "affine")),
                "packed projection differs");
        auto partial = slice_axis(x, -1, 128, 256);
        auto q = slice_axis(slice_axis(planes[0], 0, 1, 6), 1, 128 * bits / 32, 256 * bits / 32);
        auto s = slice_axis(slice_axis(planes[1], 0, 1, 6), 1, 128 / group, 256 / group);
        auto b = slice_axis(slice_axis(planes[2], 0, 1, 6), 1, 128 / group, 256 / group);
        require(same(packed.project_base_slice(partial, "p", 1, 6, 128, 256, false),
                     mx::quantized_matmul(partial, q, s, b, true, group, bits, "affine")),
                "sliced packed projection differs");
        auto base_slice = packed.project_base_slice(partial, "p", 1, 6, 128, 256, false);
        require(same(packed.apply_runtime_lora_slice(base_slice, partial, "p", 1, 6, 128, 256, false), base_slice),
                "supplied-base LoRA slice lost declared affine geometry");
        Weights cleared = packed;
        cleared.clear();
        cleared.set_affine_quantization(32, bits == 4 ? 8 : 4);
        if (group == 64 && bits == 4) {
            Weights wrong;
            wrong.set_affine_quantization(32, 4);
            wrong.bind_arrays({"p.weight", "p.scales", "p.biases"}, planes);
            rejected = false;
            try { wrong.embedding_rows(ids, "p", mx::bfloat16); } catch (const std::exception &) { rejected = true; }
            require(rejected, "ambiguous packed Q4/g64 cannot silently decode as Q8/g32");
        }
    }
    std::cout << "PASS affine dequant dtypes: six actual Q4/Q8 g32/g64/g128 cases, row-first embeddings and full/sliced projections exact, historical FP16 unchanged, copied source immutable\n";
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }

#include "../../native/backends/mlx.hpp"
#include <iostream>

using namespace tc;

static bool same(const Tensor &first, const Tensor &second) {
    return first.shape() == second.shape() && first.dtype() == second.dtype() &&
        mx::all(mx::view(first, mx::uint8) == mx::view(second, mx::uint8)).item<bool>();
}

int main() { try {
    configure_streams();
    for (int bits : {4, 8}) {
        auto source = mx::astype(mx::reshape(mx::sin(mx::arange(8 * 64, mx::float32) * .13f), {8, 64}), mx::bfloat16);
        auto planes = mx::quantize(source, 32, bits, "affine");
        mx::eval(planes);
        Weights packed;
        packed.bind_arrays({"p.weight", "p.scales", "p.biases"}, planes);
        Weights historical = packed, explicit_bf16 = packed;
        auto expected_fp16 = mx::dequantize(planes[0], planes[1], planes[2], 32, bits, "affine", std::nullopt, mx::float16);
        auto expected_bf16 = mx::dequantize(planes[0], planes[1], planes[2], 32, bits, "affine", std::nullopt, mx::bfloat16);
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
    }
    std::cout << "PASS affine dequant dtypes: two actual Q4/Q8 cases, historical FP16 unchanged, explicit BF16 exact, copied source immutable\n";
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }

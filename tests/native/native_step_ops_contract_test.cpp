#include "../../native/backends/metal_graph/kernels/step_ops.hpp"
#include <iostream>

namespace ops = tc::metal_graph::step_ops;
namespace {
void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F body) {
    try { body(); } catch (const std::invalid_argument &) { return; }
    throw std::runtime_error("invalid native step contract accepted");
}
ops::Layout matrix(const void *owner, uint32_t rows = 1024, uint32_t cols = 64,
                   size_t stride = 128, ops::Element type = ops::Element::BF16) {
    return {owner, 0, size_t(rows - 1) * stride + size_t(cols) * ops::element_bytes(type), stride, rows, cols, type};
}
}

int main() {
    try {
        int a = 1, b = 2, c = 3, f = 4;
        const auto input = matrix(&a), noise = matrix(&b), output = matrix(&c);
        check(ops::validate_layout(input) == 1024 * 64 * 2, "1024x64 extent changed");
        ops::validate_copy(input, output);
        ops::validate_euler(input, noise, -0.125f, output);
        ops::validate_euler(input, input, 0.f, output); // independent output, immutable inputs may alias
        auto padded = matrix(&b, 1024, 64, 192);
        ops::validate_copy(input, padded);
        check(ops::validate_layout(padded) == 1023 * 192 + 128, "row padding counted as visible data");
        ops::validate_copy(matrix(&a, 1, 1, 2), matrix(&b, 1, 1, 2));
        ops::validate_copy(matrix(&a, 3, 7, 32, ops::Element::FP32), matrix(&b, 3, 7, 64, ops::Element::FP32));
        auto adjacent = input; adjacent.offset = input.available;
        ops::validate_copy(input, adjacent);
        adjacent.offset -= 2;
        rejects([&] { ops::validate_copy(input, adjacent); });
        rejects([&] { ops::validate_copy(input, input); });
        rejects([&] { ops::validate_euler(input, noise, 1.f, noise); });
        rejects([&] { ops::validate_euler(input, noise, 1.f, input); });
        rejects([&] { ops::validate_copy(input, matrix(&b, 1023)); });
        rejects([&] { ops::validate_copy(input, matrix(&b, 1024, 64, 256, ops::Element::FP32)); });
        rejects([&] { ops::validate_euler(matrix(&a, 2, 64, 256, ops::Element::FP32),
                                        matrix(&b, 2, 64, 256, ops::Element::FP32), 1.f,
                                        matrix(&c, 2, 64, 256, ops::Element::FP32)); });
        for (int fault = 0; fault < 11; ++fault) {
            auto invalid = input;
            switch (fault) {
            case 0: invalid.allocation = nullptr; break;
            case 1: invalid.rows = 0; break;
            case 2: invalid.columns = 0; break;
            case 3: invalid.stride = 0; break;
            case 4: invalid.stride = 127; break;
            case 5: invalid.offset = 1; break;
            case 6: --invalid.available; break;
            case 7: invalid.type = static_cast<ops::Element>(99); break;
            case 8: invalid.columns = UINT32_MAX; break;
            case 9: invalid.stride = std::numeric_limits<size_t>::max() - 1; break;
            case 10: invalid.offset = std::numeric_limits<size_t>::max() - 1; break;
            }
            rejects([&] { ops::validate_layout(invalid); });
        }
        const ops::FlagLayout flag{&f, 0, 4};
        ops::validate_finite(input, flag);
        ops::validate_finite(input, {&a, input.available, 4});
        rejects([&] { ops::validate_finite(input, {&a, 0, 4}); });
        rejects([&] { ops::validate_finite(input, {&a, input.available - 4, 4}); });
        rejects([&] { ops::validate_flag({nullptr, 0, 4}); });
        rejects([&] { ops::validate_flag({&f, 1, 4}); });
        rejects([&] { ops::validate_flag({&f, 0, 3}); });

        auto rounded = [](float x) { return ops::bf16_value(ops::bf16_bits(x)); };
        check(rounded(1.00390625f) == 1.f && rounded(1.01171875f) == 1.015625f, "BF16 ties-to-even changed");
        check(ops::bf16_bits(-0.f) == 0x8000, "BF16 signed zero lost");
        check(ops::bf16_value(ops::reference_euler(ops::bf16_bits(1.f), ops::bf16_bits(2.f), -0.125f)) == 0.75f,
              "negative Euler dt changed");
        const auto rounded_dt = ops::bf16_value(ops::reference_euler(0, ops::bf16_bits(1.5f), 1.00390625f));
        check(rounded_dt == 1.5f && rounded(1.5f * 1.00390625f) == 1.5078125f,
              "Euler must narrow dt before multiplication");
        const auto rounded_product = ops::bf16_value(ops::reference_euler(ops::bf16_bits(-1.5f), ops::bf16_bits(1.5f), 1.0078125f));
        check(rounded_product == 0.015625f && rounded(-1.5f + 1.5f * 1.0078125f) == 0.01171875f,
              "Euler product rounding was fused away");
        check(ops::bf16_value(ops::reference_euler(ops::bf16_bits(1.f), ops::bf16_bits(1.f), 0.00390625f)) == 1.f,
              "Euler add final BF16 tie changed");
        for (float invalid : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::max()})
            rejects([&] { ops::checked_dt(invalid); });
        check(!ops::bf16_finite(ops::reference_euler(0, 0x7f80, 0.f)), "infinity times zero was hidden");
        check(!ops::bf16_finite(ops::reference_euler(0x7f7f, 0x7f7f, 2.f)), "Euler overflow was hidden");
        for (uint32_t bits = 0; bits <= UINT16_MAX; ++bits) {
            const auto narrow = uint16_t(bits);
            const float value = ops::bf16_value(narrow);
            check(ops::bf16_finite(narrow) == std::isfinite(value), "finite predicate disagrees with BF16 exponent");
            if (std::isfinite(value)) check(ops::bf16_bits(value) == narrow, "finite BF16 roundtrip failed");
            else check(!ops::bf16_finite(ops::bf16_bits(value)), "NaN/infinity became finite");
        }
        std::cout << "PASS native step CPU contract: pitched BF16/FP32 layouts, aliases, Euler rounding, dt admission, all BF16 finite patterns\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

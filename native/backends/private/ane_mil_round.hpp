#pragma once

#include "../ane_runtime.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace tc::ane::private_api {
// BF16-RNE VALUES represented in FP16, not native BF16 arithmetic/cast.
// The select lowering canonicalizes -0 to +0: callers must explicitly accept
// this numerical policy. Overflow/nonfinite reaches whole-operation fallback.
inline std::string emit_bf16_value_round(std::string &body, const std::string &input,
                                      const std::string &prefix, const std::string &shape) {
    const auto type = "tensor<fp16, " + shape + ">", boolean = "tensor<bool, " + shape + ">";
    auto power2 = [](int exponent) {
        std::ostringstream out; out << std::hexfloat << std::ldexp(1.f, exponent);
        return "fp16(" + out.str() + ")";
    };
    auto line = [&](const std::string &name, const std::string &expression, bool predicate = false) {
        const auto symbol = prefix + "_" + name;
        body += "        " + (predicate ? boolean : type) + " " + symbol + " = " + expression + ";\n";
        return symbol;
    };
    const auto magnitude = line("abs", "abs(x = " + input + ")");
    const auto zero = line("zero", "mul(x = " + magnitude + ", y = fp16(0))");
    const auto one = line("one", "add(x = " + zero + ", y = fp16(1))");
    std::array<std::string, 4> factors{one, one, one, one};
    for (int exponent = -16; exponent <= 15; ++exponent) {
        const auto suffix = std::to_string(exponent + 17);
        const auto condition = line("bin" + suffix, "greater_equal(x = " + magnitude + ", y = " + power2(exponent) + ")", true);
        const int powers[]{std::min(7 - exponent, 10), std::max(7 - exponent - 10, 0),
                           std::max(exponent - 7, -14), std::min(exponent + 7, 0)};
        for (size_t f = 0; f < factors.size(); ++f)
            factors[f] = line("factor" + std::to_string(f) + "_" + suffix,
                "select(cond = " + condition + ", a = " + power2(powers[f]) + ", b = " + factors[f] + ")");
    }
    const auto scaled_first = line("scaled_first", "mul(x = " + input + ", y = " + factors[0] + ")");
    const auto scaled = line("scaled", "mul(x = " + scaled_first + ", y = " + factors[1] + ")");
    const auto scaled_abs = line("scaled_abs", "abs(x = " + scaled + ")");
    const auto lower = line("lower", "floor(x = " + scaled_abs + ")");
    const auto fraction = line("fraction", "sub(x = " + scaled_abs + ", y = " + lower + ")");
    const auto lower_half = line("lower_half", "mul(x = " + lower + ", y = fp16(0.5))");
    const auto parity_floor = line("parity_floor", "floor(x = " + lower_half + ")");
    // Equivalent mul(floor,2) failed the actual driver oracle. Do not
    // simplify this tested addition lowering algebraically.
    const auto even_lower = line("even_lower", "add(x = " + parity_floor + ", y = " + parity_floor + ")");
    const auto parity = line("parity", "sub(x = " + lower + ", y = " + even_lower + ")");
    const auto above = line("above", "greater(x = " + fraction + ", y = fp16(0.5))", true);
    const auto tie = line("tie", "equal(x = " + fraction + ", y = fp16(0.5))", true);
    const auto above_add = line("above_add", "select(cond = " + above + ", a = fp16(1), b = " + zero + ")");
    const auto tie_add = line("tie_add", "select(cond = " + tie + ", a = " + parity + ", b = " + zero + ")");
    const auto round_first = line("round_first", "add(x = " + lower + ", y = " + above_add + ")");
    const auto round_abs = line("round_abs", "add(x = " + round_first + ", y = " + tie_add + ")");
    const auto round_negative = line("round_negative", "mul(x = " + round_abs + ", y = fp16(-1))");
    const auto negative = line("negative", "less(x = " + scaled + ", y = fp16(0))", true);
    const auto rounded = line("rounded", "select(cond = " + negative + ", a = " + round_negative + ", b = " + round_abs + ")");
    const auto restored_first = line("restored_first", "mul(x = " + rounded + ", y = " + factors[2] + ")");
    const auto restored = line("restored", "mul(x = " + restored_first + ", y = " + factors[3] + ")");
    const auto tiny = line("tiny", "less(x = " + magnitude + ", y = " + power2(-16) + ")", true);
    return line("out", "select(cond = " + tiny + ", a = " + input + ", b = " + restored + ")");
}
} // namespace tc::ane::private_api

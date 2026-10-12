// CPU-only LoRA numerical specification. No production, backend or Metal code.
// Every expected result below is a fixed value, independent of the evaluator.
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
size_t checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}

enum class Format { BF16, FP32 };
uint16_t round_bf16(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

// Byte storage makes the row padding and retained prefix/suffix observable.
struct Matrix {
    size_t rows, columns, stride;
    Format format;
    std::vector<std::byte> bytes;
    static constexpr size_t prefix = 8, padding = 8;
    Matrix(size_t m, size_t n, Format type)
        : rows(m), columns(n), stride(n * (type == Format::BF16 ? 2 : 4) + padding),
          format(type), bytes(prefix + m * stride + padding, std::byte{0xcd}) {
        for (size_t r = 0; r < rows; ++r)
            std::memset(bytes.data() + prefix + r * stride, 0, columns * item());
    }
    size_t item() const { return format == Format::BF16 ? 2 : 4; }
    size_t offset(size_t r, size_t c) const { return prefix + r * stride + c * item(); }
    void put_bits(size_t r, size_t c, uint16_t value) {
        if (format != Format::BF16) throw std::logic_error("BF16 write to FP32 matrix");
        std::memcpy(bytes.data() + offset(r, c), &value, sizeof(value));
    }
    void put_float(size_t r, size_t c, float value) {
        if (format != Format::FP32) throw std::logic_error("FP32 write to BF16 matrix");
        std::memcpy(bytes.data() + offset(r, c), &value, sizeof(value));
    }
    uint16_t bits(size_t r, size_t c) const {
        uint16_t value = 0;
        std::memcpy(&value, bytes.data() + offset(r, c), sizeof(value));
        return value;
    }
    float value(size_t r, size_t c) const {
        if (format == Format::BF16)
            return std::bit_cast<float>(uint32_t(bits(r, c)) << 16);
        float result = 0;
        std::memcpy(&result, bytes.data() + offset(r, c), sizeof(result));
        return result;
    }
    void fill_bits(uint16_t value) {
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < columns; ++c) put_bits(r, c, value);
    }
};

struct Adapter { const Matrix* a; const Matrix* b; float scale = 1; };

// A is [rank,K], B is [N,rank]. The low intermediate stays FP32, and each
// adapter rounds the existing BF16 output once. This is an ordinary scalar
// reference; it has no GPU tile, vector, scheduling or cache implementation.
std::vector<float> project(const Matrix& input, const Matrix& a) {
    if (input.columns != a.columns) throw std::logic_error("invalid reference K");
    std::vector<float> low(input.rows * a.rows);
    for (size_t m = 0; m < input.rows; ++m)
        for (size_t r = 0; r < a.rows; ++r) {
            float sum = 0;
            for (size_t k = 0; k < input.columns; ++k)
                sum += input.value(m, k) * a.value(r, k);
            low[m * a.rows + r] = sum;
        }
    return low;
}
void add_bias(Matrix& output, size_t first, size_t count, const Matrix& bias) {
    for (size_t m = 0; m < output.rows; ++m)
        for (size_t n = first; n < first + count; ++n)
            output.put_bits(m, n, round_bf16(output.value(m, n) + bias.value(0, n)));
}
void evaluate(const Matrix& input, Matrix& output, size_t first, size_t count,
              std::initializer_list<Adapter> adapters, const Matrix* bias = nullptr) {
    if (output.format != Format::BF16 || input.rows != output.rows ||
        first > output.columns || count > output.columns - first)
        throw std::logic_error("invalid reference output range");
    for (const auto& adapter : adapters) {
        if (!std::isfinite(adapter.scale) || adapter.b->rows != output.columns ||
            adapter.b->columns != adapter.a->rows)
            throw std::logic_error("invalid reference adapter");
        const auto low = project(input, *adapter.a); // A identity matters, even for the same input.
        for (size_t m = 0; m < input.rows; ++m)
            for (size_t n = first; n < first + count; ++n) {
                float delta = 0;
                for (size_t r = 0; r < adapter.a->rows; ++r)
                    delta += low[m * adapter.a->rows + r] * adapter.b->value(n, r);
                output.put_bits(m, n, round_bf16(output.value(m, n) + adapter.scale * delta));
            }
    }
    if (bias) add_bias(output, first, count, *bias);
}

void factor_value(Matrix& m, size_t r, size_t c, uint16_t bf16, float fp32) {
    if (m.format == Format::BF16) m.put_bits(r, c, bf16);
    else m.put_float(r, c, fp32);
}
void untouched(const Matrix& output, const std::vector<std::byte>& before,
               size_t first, size_t count) {
    bool same = output.bytes.size() == before.size();
    for (size_t i = 0; same && i < before.size(); ++i) {
        bool selected = false;
        for (size_t r = 0; r < output.rows; ++r)
            selected = selected || (i >= output.offset(r, first) &&
                i < output.offset(r, first) + count * output.item());
        if (!selected) same = output.bytes[i] == before[i];
    }
    check(same, "reference changed padding, guards or an unselected column");
}

void test_fp32_low() {
    Matrix input(1, 2, Format::BF16); input.fill_bits(0x3f80);
    const auto original_input = input.bytes;
    for (size_t rank : {size_t{16}, size_t{128}})
        for (Format a_type : {Format::BF16, Format::FP32})
            for (Format b_type : {Format::BF16, Format::FP32}) {
                Matrix a(rank, 2, a_type), b(1, rank, b_type), output(1, 1, Format::BF16);
                factor_value(a, 0, 0, 0x3f80, 1.f);
                factor_value(a, 0, 1, 0x3b80, 0.00390625f);
                factor_value(b, 0, 0, 0x4380, 256.f);
                output.put_bits(0, 0, 0xc380); // -256
                const auto old_a = a.bytes, old_b = b.bytes, before = output.bytes;
                const auto low = project(input, a);
                check(low[0] == 1.00390625f, "rank16/128 low was rounded before B");
                for (size_t r = 1; r < rank; ++r)
                    check(low[r] == 0.f, "zero rank slot contributed to low");
                evaluate(input, output, 0, 1, {{&a, &b, 1.f}});
                check(output.bits(0, 0) == 0x3f80, "FP32 low must retain the final value one");
                untouched(output, before, 0, 1);
                check(a.bytes == old_a && b.bytes == old_b, "reference mutated factors");
            }
    check(input.bytes == original_input, "reference mutated input");
    check(round_bf16(1.00390625f) == 0x3f80 &&
          round_bf16(256.f * 1.f - 256.f) == 0x0000,
          "premature BF16 low negative control must produce zero");
}

void test_order_and_bias() {
    Matrix input(1, 1, Format::BF16), a(1, 1, Format::BF16);
    input.fill_bits(0x3f80); a.fill_bits(0x3f80);
    Matrix small(1, 1, Format::BF16), large(1, 1, Format::BF16), combined(1, 1, Format::BF16);
    small.fill_bits(0x3b80); large.fill_bits(0x3c00); combined.fill_bits(0x3c40);
    Matrix ordered(1, 1, Format::BF16), reversed = ordered, merged = ordered;
    ordered.fill_bits(0x3f80); reversed.fill_bits(0x3f80); merged.fill_bits(0x3f80);
    evaluate(input, ordered, 0, 1, {{&a, &small}, {&a, &large}});
    evaluate(input, reversed, 0, 1, {{&a, &large}, {&a, &small}});
    evaluate(input, merged, 0, 1, {{&a, &combined}});
    check(ordered.bits(0, 0) == 0x3f81, "ordered adapters lost the BF16 tie-to-even boundary");
    check(reversed.bits(0, 0) == 0x3f82, "reversed adapters should round to the next even BF16");
    check(merged.bits(0, 0) == 0x3f82, "combined adapters should differ from ordered rounding");
    Matrix last(1, 1, Format::BF16), first = last;
    last.fill_bits(0x3f80); first.fill_bits(0x3f80);
    evaluate(input, last, 0, 1, {{&a, &large}}, &small);
    add_bias(first, 0, 1, small);
    check(first.bits(0, 0) == 0x3f80, "bias-first half-ULP must round to even one");
    evaluate(input, first, 0, 1, {{&a, &large}});
    check(last.bits(0, 0) == 0x3f82, "bias must be rounded after all adapters");
    check(first.bits(0, 0) == 0x3f81, "bias-first negative control changed");
}

void test_finite_scales_and_factor_precision() {
    Matrix input(1, 1, Format::BF16), a(1, 1, Format::BF16), b(1, 1, Format::BF16);
    input.fill_bits(0x3f80); a.fill_bits(0x3f80); b.fill_bits(0x4000);
    const std::array<float, 3> scales{-1.f, 0.f, -0.f};
    const std::array<uint16_t, 3> expected{0xbf80, 0x3f80, 0x3f80};
    for (size_t i = 0; i < scales.size(); ++i) {
        Matrix output(1, 1, Format::BF16); output.fill_bits(0x3f80);
        evaluate(input, output, 0, 1, {{&a, &b, scales[i]}});
        check(output.bits(0, 0) == expected[i], "finite negative/zero adapter scale changed");
    }
    // Raw FP32 factors retain a half-BF16-ULP that is lost in BF16 storage.
    for (bool tiny_a : {true, false}) {
        Matrix raw_a(1, 1, tiny_a ? Format::FP32 : Format::BF16);
        Matrix raw_b(1, 1, tiny_a ? Format::BF16 : Format::FP32);
        Matrix bf_a(1, 1, Format::BF16), bf_b(1, 1, Format::BF16);
        if (tiny_a) {
            raw_a.put_float(0, 0, 1.00390625f); raw_b.fill_bits(0x4380);
            bf_a.fill_bits(0x3f80); bf_b.fill_bits(0x4380);
        } else {
            raw_a.fill_bits(0x4380); raw_b.put_float(0, 0, 1.00390625f);
            bf_a.fill_bits(0x4380); bf_b.fill_bits(0x3f80);
        }
        Matrix raw(1, 1, Format::BF16), narrowed = raw;
        raw.fill_bits(0xc380); narrowed.fill_bits(0xc380);
        evaluate(input, raw, 0, 1, {{&raw_a, &raw_b}});
        evaluate(input, narrowed, 0, 1, {{&bf_a, &bf_b}});
        check(raw.bits(0, 0) == 0x3f80, "raw FP32 A/B factor tiny value was lost");
        check(narrowed.bits(0, 0) == 0x0000, "BF16 factor negative control must produce zero");
    }
}

void test_distinct_a() {
    Matrix input(1, 2, Format::BF16), a1(1, 2, Format::BF16), a2(1, 2, Format::BF16);
    Matrix b(1, 1, Format::BF16), output(1, 1, Format::BF16);
    input.fill_bits(0x3f80); a1.put_bits(0, 0, 0x3f80); a2.put_bits(0, 1, 0x4000); b.fill_bits(0x3f80);
    check(project(input, a1)[0] == 1.f && project(input, a2)[0] == 2.f,
          "distinct A factors must produce distinct lows for the same input");
    evaluate(input, output, 0, 1, {{&a1, &b}, {&a2, &b}});
    check(output.bits(0, 0) == 0x4040, "low reuse keyed only by input would incorrectly produce two");
}

void test_rank3_tail_and_padding() {
    Matrix input(3, 2, Format::BF16), a(3, 2, Format::BF16), b(5, 3, Format::BF16);
    const std::array<uint16_t, 6> x{0x3f80, 0x4000, 0xbf80, 0x4040, 0x4000, 0xc000};
    const std::array<uint16_t, 6> av{0x3f80, 0, 0, 0x3f80, 0x3f80, 0xbf80};
    const std::array<uint16_t, 15> bv{0x40e0, 0x40e0, 0x40e0, 0xc080, 0x40a0, 0x40c0,
                                    0x4000, 0, 0x3f80, 0, 0x3f80, 0xbf80, 0x3f80, 0x3f80, 0};
    for (size_t i = 0; i < x.size(); ++i) { input.put_bits(i / 2, i % 2, x[i]); a.put_bits(i / 2, i % 2, av[i]); }
    for (size_t i = 0; i < bv.size(); ++i) b.put_bits(i / 3, i % 3, bv[i]);
    const std::array<float, 9> known_low{1, 2, -1, -1, 3, -4, 2, -2, 4};
    const auto low = project(input, a);
    for (size_t i = 0; i < known_low.size(); ++i) check(low[i] == known_low[i], "rank3 low integer oracle changed");
    Matrix output(3, 5, Format::BF16); output.fill_bits(0x4180); // 16
    const auto before = output.bytes, old_x = input.bytes, old_a = a.bytes, old_b = b.bytes;
    evaluate(input, output, 2, 3, {{&a, &b}});
    const std::array<uint16_t, 9> expected{0x4188, 0x4198, 0x4198,
                                           0x4120, 0x41b8, 0x4190,
                                           0x41c0, 0x4120, 0x4180};
    for (size_t i = 0; i < expected.size(); ++i)
        check(output.bits(i / 3, 2 + i % 3) == expected[i], "rank3/M3/N5 tail integer oracle changed");
    untouched(output, before, 2, 3);
    check(input.bytes == old_x && a.bytes == old_a && b.bytes == old_b,
          "rank3 evaluation mutated input/factor bytes or padding");
}
} // namespace

int main() {
    try {
        test_fp32_low();
        test_order_and_bias();
        test_finite_scales_and_factor_precision();
        test_distinct_a();
        test_rank3_tail_and_padding();
        std::cout << "native_lora_math=PASS checks=" << checks
                  << " device_created=0 gpu_submissions=0 models=0\n";
    } catch (const std::exception& error) {
        std::cerr << "native_lora_math=FAIL " << error.what() << '\n';
        return 1;
    }
}

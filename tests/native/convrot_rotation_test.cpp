#include "../../native/backends/convrot_rotation.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

namespace mx = mlx::core;
using tc::convrot_kernel::Rotation;
namespace {
void check(bool ok, const char *reason) { if (!ok) throw std::runtime_error(reason); }
bool same_bits(const mx::array &a, const mx::array &b) {
    return mx::all(mx::view(a, mx::uint8) == mx::view(b, mx::uint8)).item<bool>();
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end()); return values[values.size() / 2];
}
}
int main(int argc, char **argv) {
  try {
    int cases = 0;
    for (auto dtype : {mx::float32, mx::float16, mx::bfloat16})
      for (int rows : {1, 33, 1056, 4128}) for (int columns : {256, 3840, 10240}) for (bool strided : {false, true}) {
        auto values = mx::random::normal({strided ? columns : rows, strided ? rows : columns}, mx::float32, mx::random::key(43));
        auto x = mx::astype(strided ? mx::transpose(values) : values, dtype);
        auto snapshot = mx::copy(x); mx::eval({x, snapshot});
        auto old = tc::convrot_kernel::rotate(x, Rotation::Shared);
        auto candidate = tc::convrot_kernel::rotate(x, Rotation::Simd);
        auto quad = tc::convrot_kernel::rotate(x, Rotation::SimdQuad);
        auto registers = tc::convrot_kernel::rotate(x, Rotation::SimdRegister);
        mx::eval({old, candidate, quad, registers});
        check(mx::all(mx::isfinite(candidate)).item<bool>() && mx::all(old == candidate).item<bool>() &&
              mx::all(old == quad).item<bool>() && same_bits(old, registers),
              "SIMD ConvRot changed shared-memory butterfly rounding");
        check(mx::all(x == snapshot).item<bool>(), "ConvRot mutated input");
        ++cases;
      }
    // Every basis row independently fixes the Comfy ordering and signs.
    std::vector<float> basis(256 * 256), expected(256 * 256);
    for (int r = 0; r < 256; ++r) {
        basis[r * 256 + r] = 1;
        for (int c = 0; c < 256; ++c) {
            int sign = 1;
            for (int digit = 0; digit < 4; ++digit)
                if (((r >> (2 * digit)) & 3) + ((c >> (2 * digit)) & 3) == 3) sign = -sign;
            expected[r * 256 + c] = float(sign) / 16;
        }
    }
    auto x = mx::array(basis.data(), {256, 256}, mx::float32);
    auto wanted = mx::array(expected.data(), {256, 256}, mx::float32);
    check(mx::all(tc::convrot_kernel::rotate(x, Rotation::Simd) == wanted).item<bool>(), "SIMD ConvRot has wrong Comfy H256 order");
    check(mx::all(tc::convrot_kernel::rotate(x, Rotation::SimdQuad) == wanted).item<bool>(), "quad ConvRot has wrong Comfy H256 order");
    check(mx::all(tc::convrot_kernel::rotate(x, Rotation::SimdRegister) == wanted).item<bool>(), "register ConvRot has wrong Comfy H256 order");
    // Exact cancellation, signed zero, tiny/large exponents and a partial
    // four-group tail. Keep inputs finite; classify NaN/infinity outputs by
    // raw bytes instead of calling approximate allclose on boundary cases.
    std::vector<float> adversarial(7 * 768);
    constexpr float values[] = {0.f, -0.f, 0x1p-20f, -0x1p-20f, 1.f, -1.f, 255.f, -255.f, 2048.f, -2048.f};
    for (size_t i = 0; i < adversarial.size(); ++i) adversarial[i] = values[(i * 7 + i / 256) % 10];
    for (auto dtype : {mx::float32, mx::float16, mx::bfloat16}) {
        auto edge = mx::astype(mx::array(adversarial.data(), {7, 768}, mx::float32), dtype);
        auto control = tc::convrot_kernel::rotate(edge, Rotation::Shared);
        auto result = tc::convrot_kernel::rotate(edge, Rotation::SimdRegister);
        check(same_bits(control, result), "register ConvRot changed cancellation/signed-zero/tail bytes");
    }
    std::cout << "PASS SIMD ConvRot rotation: " << cases << " typed/strided cases bit-exact to shared kernel, immutable input and 256 independent basis rows\n";
    if (argc == 2 && std::string(argv[1]) == "bench") {
        for (int rows : {1056, 4224}) for (int columns : {3840, 10240}) {
            auto input = mx::astype(mx::random::normal({1, rows, columns}, mx::float32, mx::random::key(19)), mx::bfloat16);
            mx::eval(input);
            auto run = [&](Rotation kind) {
                const auto start = std::chrono::steady_clock::now();
                auto y = tc::convrot_kernel::rotate(input, kind); mx::eval(y);
                return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            };
            for (int warm = 0; warm < 3; ++warm) { run(Rotation::Shared); run(Rotation::SimdQuad); run(Rotation::SimdRegister); }
            std::vector<double> old, candidate, registers;
            for (int sample = 0; sample < 27; ++sample) {
                // Cyclic serial arms, avoiding a permanently hot last arm.
                for (int position = 0; position < 3; ++position) switch ((sample + position) % 3) {
                    case 0: old.push_back(run(Rotation::Shared)); break;
                    case 1: candidate.push_back(run(Rotation::SimdQuad)); break;
                    default: registers.push_back(run(Rotation::SimdRegister)); break;
                }
            }
            std::cout << std::setprecision(12) << "COMPONENT rows=" << rows << " columns=" << columns
                      << " shared=" << median(old) << " quad=" << median(candidate)
                      << " speedup=" << median(old) / median(candidate)
                      << " register=" << median(registers) << " quad_over_register=" << median(candidate) / median(registers)
                      << "; isolated operator host span, not E2E/performance qualification shared_samples=[";
            for (size_t i = 0; i < old.size(); ++i) std::cout << (i ? "," : "") << old[i];
            std::cout << "] quad_samples=[";
            for (size_t i = 0; i < candidate.size(); ++i) std::cout << (i ? "," : "") << candidate[i];
            std::cout << "] register_samples=[";
            for (size_t i = 0; i < registers.size(); ++i) std::cout << (i ? "," : "") << registers[i];
            std::cout << "]\n";
        }
    }
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

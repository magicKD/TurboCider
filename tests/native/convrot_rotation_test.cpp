#include "../../native/backends/convrot_rotation.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

namespace mx = mlx::core;
using tc::convrot_kernel::Rotation;
namespace {
void check(bool ok, const char *reason) { if (!ok) throw std::runtime_error(reason); }
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
        mx::eval({old, candidate, quad});
        check(mx::all(mx::isfinite(candidate)).item<bool>() && mx::all(old == candidate).item<bool>() &&
              mx::all(old == quad).item<bool>(),
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
            for (int warm = 0; warm < 3; ++warm) { run(Rotation::Shared); run(Rotation::SimdQuad); }
            std::vector<double> old, candidate;
            for (int sample = 0; sample < 9; ++sample) {
                if (sample & 1) { candidate.push_back(run(Rotation::SimdQuad)); old.push_back(run(Rotation::Shared)); }
                else { old.push_back(run(Rotation::Shared)); candidate.push_back(run(Rotation::SimdQuad)); }
            }
            std::cout << std::setprecision(12) << "COMPONENT rows=" << rows << " columns=" << columns
                      << " shared=" << median(old) << " quad=" << median(candidate)
                      << " speedup=" << median(old) / median(candidate)
                      << "; isolated operator host span, not E2E/performance qualification shared_samples=[";
            for (size_t i = 0; i < old.size(); ++i) std::cout << (i ? "," : "") << old[i];
            std::cout << "] quad_samples=[";
            for (size_t i = 0; i < candidate.size(); ++i) std::cout << (i ? "," : "") << candidate[i];
            std::cout << "]\n";
        }
    }
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

#include "../../native/backends/ane_w8a8_math.hpp"
#include <iostream>

int main() {
    using namespace tc::ane;
    for (int size : {128, 512}) {
        std::vector<float> a(size), b(size);
        for (int i = 0; i < size; ++i) { a[i] = float(i % 7 - 3) / 8; b[i] = float(i % 11 - 5) / 8; }
        double before = 0; for (int i = 0; i < size; ++i) before += double(a[i]) * b[i];
        auto rotated = a; rotate_block(rotated, 20260930);
        // Independent dense Hadamard matrix, not the butterfly under test.
        const double norm = 1 / std::sqrt(double(size));
        for (int out = 0; out < size; ++out) {
            double expected = 0; for (int in = 0; in < size; ++in) expected += ((std::popcount(unsigned(out & in)) & 1) ? -1 : 1) * double(a[in]) * rotation_sign(20260930, in);
            expected *= norm;
            if (std::abs(rotated[out] - expected) > 2e-6) return 1;
        }
        rotate_block(a, 20260930); rotate_block(b, 20260930);
        double after = 0; for (int i = 0; i < size; ++i) after += double(a[i]) * b[i];
        if (std::abs(after - before) > 2e-5) return 1;
    }
    if (normalized_scale(0) != 0x5800 || normalized_scale(1e-30f) != 1) return 1;
    const auto scale = tc::gguf::float_to_fp16_rne(128.f);
    for (const auto [value, expected] : {std::pair{.5f, 0}, {1.5f, 2}, {2.5f, 2}, {-.5f, 0}, {-1.5f, -2}, {-2.5f, -2}, {1000.f, 127}, {-1000.f, -127}})
        if (quantize_rotated(value, scale) != expected) return 1;
    // All independent Comfy H4^4 basis vectors distinguish ordering/signs
    // from Sylvester; every coefficient is exactly representable in FP32.
    for (int basis=0;basis<256;++basis) {
        std::vector<float> x(256);x[basis]=1;rotate_comfy_block(x,DType::FP32);
        constexpr int h4[4][4]={{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
        for (int out=0;out<256;++out) {
            int sign=1;
            for (int shift=0;shift<8;shift+=2) sign*=h4[(out>>shift)&3][(basis>>shift)&3];
            if (x[out]!=sign/16.f || comfy_h256_sign(out,basis)!=sign) return 1;
        }
    }
    if (std::string(convrot_w8a8_recipe)==w8a8_recipe) return 1;
    std::cout << "PASS independent H128/H512 orthogonality/dense oracle, deterministic signs, scale tiny/zero, signed RNE ties/clipping\n";
    std::cout << "PASS independent 256 Comfy H4^4 basis rows and distinct direct-Q8 recipe\n";
}

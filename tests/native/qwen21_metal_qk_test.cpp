#include "../../native/models/qwen21/metal/qk_norm_rope.hpp"
#include "../../native/backends/mlx.hpp"
#include <cmath>
#include <iostream>

int main() {
    try {
        tc::configure_streams();
        namespace mx = tc::mx;
        for (int n : {17, 1024, 1048}) {
            auto q = mx::astype(mx::random::normal({1, n, 4096}, mx::float32,
                                mx::random::key(n)), mx::bfloat16);
            auto k = mx::astype(mx::random::normal({1, n, 4096}, mx::float32,
                                mx::random::key(n + 77)), mx::bfloat16);
            auto qw = mx::full({128}, 1.f, mx::bfloat16);
            auto kw = mx::full({128}, .75f, mx::bfloat16);
            auto phase = mx::reshape(mx::arange(0, n * 64, mx::float32) / 133.f, {n, 64});
            auto cosine = mx::cos(phase), sine = mx::sin(phase);
            auto qq = mx::fast::rms_norm(tc::heads(q, 32, 128), qw, 1e-6f);
            auto kk = mx::fast::rms_norm(tc::heads(k, 32, 128), kw, 1e-6f);
            auto expected = tc::rope_pairs_pair(qq, kk, cosine, sine);
            auto actual = tc::qwen21::metal::prepare_qk(q, k, qw, kw, cosine, sine, 1e-6f);
            tc::require(actual.size() == 2 && actual[0].shape() == expected[0].shape() &&
                        actual[0].dtype() == mx::bfloat16, "Qwen21 fused Q/K shape or dtype mismatch");
            for (int part = 0; part < 2; ++part) {
                auto a = mx::astype(actual[part], mx::float32);
                auto b = mx::astype(expected[part], mx::float32);
                mx::eval(a, b);
                auto rel = mx::sqrt(mx::sum(mx::square(a - b)) /
                                    mx::sum(mx::square(b))).item<float>();
                tc::require(std::isfinite(rel) && rel < .01f, "Qwen21 fused Q/K norm-RoPE error too large");
                std::cout << "n=" << n << " part=" << part << " relative_l2=" << rel << '\n';
            }
        }
        std::cout << "PASS Qwen21 fused Q/K RMSNorm-RoPE\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

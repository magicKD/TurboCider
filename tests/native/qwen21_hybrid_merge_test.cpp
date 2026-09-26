#include "../../native/models/qwen21/hybrid_merge.hpp"
#include <algorithm>
#include <iostream>

int main() {
    try {
        tc::configure_streams();
        namespace mx = tc::mx;
        int cases = 0;
        for (auto dtype : {mx::float32, mx::bfloat16}) {
            for (int rows : {1, 17, 1024}) {
                auto gpu = mx::astype(mx::random::normal({1, rows, 4096}, mx::float32,
                    mx::random::key(42 + rows)), dtype);
                auto ane = mx::astype(mx::random::normal(gpu.shape(), mx::float32,
                    mx::random::key(73 + rows)), mx::float16);
                mx::eval(gpu, ane);
                for (float scale : {1.f, 1.25f, 3.7f, 2.f, 16.f, 256.f}) {
                    auto expected = gpu + mx::astype(ane, dtype) * tc::Tensor(scale, dtype);
                    auto actual = tc::qwen21::merge_mlp_partitions(gpu, ane, scale);
                    mx::eval(expected, actual);
                    tc::require(actual.dtype() == dtype && actual.shape() == gpu.shape(), "merge ABI changed");
                    tc::require(mx::array_equal(actual, expected).item<bool>(), "merge rounding changed");
                    ++cases;
                }
                if (rows == 1024 && dtype == mx::bfloat16) {
                    std::vector<double> eager, fused;
                    for (int i = 0; i < 24; ++i) {
                        auto measure = [&](bool compile) {
                            auto start = tc::Clock::now();
                            auto value = compile ? tc::qwen21::merge_mlp_partitions(gpu, ane, 1.f)
                                : gpu + mx::astype(ane, dtype) * tc::Tensor(1.f, dtype);
                            mx::eval(value);
                            return std::chrono::duration<double>(tc::Clock::now() - start).count();
                        };
                        double a, b;
                        if (i % 2) { b = measure(true); a = measure(false); }
                        else { a = measure(false); b = measure(true); }
                        if (i >= 4) { eager.push_back(a); fused.push_back(b); }
                    }
                    std::sort(eager.begin(), eager.end()); std::sort(fused.begin(), fused.end());
                    std::cout << "merge eager median=" << (eager[9] + eager[10]) / 2
                              << " fused median=" << (fused[9] + fused[10]) / 2 << '\n';
                }
            }
        }
        std::cout << "PASS exact hybrid merge: " << cases << " geometry/dtype/scale cases\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

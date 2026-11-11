#include "ltx_prunavaed_fusion.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace mx = mlx::core;

int main() {
    mx::set_default_device(mx::Device(mx::Device::gpu));
    auto silu = mx::compile([](const std::vector<mx::array>& x) {
        return std::vector<mx::array>{x[0] * mx::sigmoid(x[0])};
    }, true);
    int cases = 0;
    for (int channels : {64, 128, 256, 384, 512}) {
        for (int rows : {1, 17, 8193}) {
            for (bool strided : {false, true}) {
                std::vector<mx::bfloat16_t> host(static_cast<size_t>(rows) * channels);
                for (size_t index = 0; index < host.size(); ++index) {
                    const float scale = index % 19 == 0 ? 1000.0f : index % 23 == 0 ? 0.00001f : 1.0f;
                    const float value = index / channels % 7 == 0 ? 0.0f :
                        static_cast<float>(std::sin(index * 0.071) * scale);
                    host[index] = mx::bfloat16_t(value);
                }
                auto input = strided ? mx::transpose(mx::array(host.data(), {channels, rows})) :
                    mx::array(host.data(), {rows, channels});
                auto expected = silu({mx::fast::rms_norm(input, std::nullopt, 1.0e-8f)})[0];
                auto actual = ltx_prunavaed::norm_silu(input);
                mx::eval({expected, actual});
                if (expected.nbytes() != actual.nbytes() ||
                    std::memcmp(expected.data<mx::bfloat16_t>(), actual.data<mx::bfloat16_t>(), expected.nbytes())) {
                    std::fprintf(stderr, "FAIL fused norm/SiLU rows=%d channels=%d strided=%d\n", rows, channels, strided);
                    return 1;
                }
                cases++;
            }
        }
    }
    std::printf("PASS PrunaVAED fused RMS/SiLU: %d real-channel, zero/extreme/tail/strided BF16 cases byte-exact\n", cases);
    return 0;
}

// Development-only numerical and shared-output lifetime check using real assets.
#include "../../native/models/wan/hybrid.hpp"
#include <cmath>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3, "usage: wan-hybrid-probe CHECKPOINT MANIFEST");
        namespace mx = tc::mx;
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        const auto root = std::filesystem::absolute(argv[1]);
        tc::wan::Checkpoint weights;
        weights.load(root, event, cancelled);
        auto hybrid = tc::wan::load_hybrid(std::filesystem::absolute(argv[2]), root, weights, event, cancelled);
        const mx::Shape shape{1, 32760, 1536};
        auto residual = mx::zeros(shape, mx::float16);
        auto gate = mx::ones({1, 1, 1536}, mx::float32);
        // Zero input isolates bias and the first-layer bias activation; random
        // input additionally exercises quantized channel slices and the join.
        for (int sample = 0; sample < 2; ++sample) {
            auto input = sample == 0 ? mx::zeros(shape, mx::float16)
                : mx::contiguous(mx::random::normal(shape, mx::float16, mx::random::key(42)) *
                                 tc::Tensor(.2f, mx::float16));
            mx::eval(input, residual, gate);
            for (int block = 0; block < weights.config().layers; ++block) {
                const auto prefix = "blocks." + std::to_string(block) + ".ffn.";
                auto x = weights.linear(input, prefix + "fc_in");
                auto scalar = [&](float value) { return tc::Tensor(value, x.dtype()); };
                auto activated = scalar(.5f) * x * (scalar(1.f) + mx::tanh(
                    scalar(float(std::sqrt(2. / M_PI))) *
                    (x + scalar(.044715f) * mx::power(x, scalar(3.f)))));
                auto reference = weights.linear(activated, prefix + "fc_out");
                mx::eval(reference);
                auto output = hybrid->predict(block, input, residual, gate);
                tc::require(mx::all(mx::isfinite(reference)).item<bool>() &&
                            mx::all(mx::isfinite(output)).item<bool>(), "nonfinite hybrid comparison");
                auto difference = mx::astype(output, mx::float32) - mx::astype(reference, mx::float32);
                auto rmse = mx::sqrt(mx::mean(mx::square(difference))).item<float>();
                auto rms = mx::sqrt(mx::mean(mx::square(mx::astype(reference, mx::float32)))).item<float>();
                auto maximum = mx::max(mx::abs(difference)).item<float>();
                auto retained = mx::add(output, tc::Tensor(0.f, output.dtype()));
                mx::eval(retained);
                auto repeated = hybrid->predict(block, input, residual, gate);
                tc::require(mx::all(mx::equal(retained, repeated)).item<bool>(), "hybrid repeat mismatch");
                std::cout << "{\"sample\":" << sample << ",\"block\":" << block
                          << ",\"rmse\":" << rmse << ",\"reference_rms\":" << rms
                          << ",\"max_abs\":" << maximum << "}" << std::endl;
            }
        }
        std::cout << "{\"calls\":" << hybrid->calls() << ",\"copied_bytes\":"
                  << hybrid->copied_bytes() << "}\n";
        // Metrics deliberately have no arbitrary parity threshold: a quantized
        // artifact is approximate and requires an independently qualified gate.
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

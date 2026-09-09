#include "hybrid.hpp"
#include <cmath>

namespace tc::wan {

HybridFFN::HybridFFN(const Checkpoint &weights, const std::vector<std::filesystem::path> &artifacts,
                     int rows, int ane_width, const Event &event, std::atomic<bool> &cancelled)
    : partitions_(artifacts, rows, weights.config().hidden(), event, cancelled), rows_(rows) {
    const auto &config = weights.config();
    require(artifacts.size() == size_t(config.layers) && ane_width > 0 &&
                ane_width < config.ffn_dim && ane_width % 64 == 0,
            "invalid Wan hybrid partition");
    for (int block = 0; block < config.layers; ++block) {
        checkpoint(cancelled);
        const auto prefix = "blocks." + std::to_string(block) + ".ffn.";
        auto first = weights.affine(prefix + "fc_in.weight").slice(ane_width, config.ffn_dim, 0, config.hidden());
        auto last = weights.affine(prefix + "fc_out.weight").slice(0, config.hidden(), ane_width, config.ffn_dim);
        auto bias = slice_axis(weights.at(prefix + "fc_in.bias"), 0, ane_width, config.ffn_dim);
        auto graph = [first, last, bias](const std::vector<Tensor> &args) {
            auto x = first.project(args.at(0)) + bias;
            auto scalar = [&](float value) { return Tensor(value, x.dtype()); };
            x = scalar(.5f) * x * (scalar(1.f) + mx::tanh(scalar(float(std::sqrt(2. / M_PI))) *
                (x + scalar(.044715f) * mx::power(x, scalar(3.f)))));
            // The exported Core ML graph owns the output bias. Add no bias on
            // the GPU suffix, otherwise the shared bias would be duplicated.
            return std::vector<Tensor>{last.project(x)};
        };
        gpu_.push_back(mx::compile(std::function<std::vector<Tensor>(const std::vector<Tensor> &)>(graph)));
    }
    join_ = mx::compile([](const std::vector<Tensor> &args) {
        return std::vector<Tensor>{mx::astype(args[0] + (args[1] + mx::astype(args[2], args[1].dtype())) *
                                             args[3], args[0].dtype())};
    });
}

Tensor HybridFFN::predict(int block, const Tensor &input, const Tensor &residual, const Tensor &gate) {
    require(block >= 0 && size_t(block) < gpu_.size() && input.shape() == mx::Shape{1, rows_, 1536} &&
                residual.shape() == input.shape() && gate.shape() == mx::Shape{1, 1, 1536} &&
                input.dtype() == mx::float16 && residual.dtype() == mx::float16 && gate.dtype() == mx::float32,
            "invalid Wan hybrid FFN input");
    auto packed = mx::contiguous(input);
    mx::eval(packed);
    auto gpu = gpu_[block]({packed}).at(0);
    mx::async_eval(gpu);
    auto ane = partitions_.predict(block, packed);
    auto output = join_({residual, gpu, ane, gate}).at(0);
    mx::eval(output);
    return output;
}

} // namespace tc::wan

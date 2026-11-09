#include "tae_super.hpp"

#include <array>

namespace tc::wan22 {
namespace {

Tensor relu(const Tensor &x) { return mx::maximum(x, Tensor(0.f, x.dtype())); }

Tensor pixel_shuffle(const Tensor &x) {
    require(x.ndim() == 4 && x.shape(3) % 4 == 0, "invalid TAEW2.2-Super pixel-shuffle input");
    const int n = x.shape(0), h = x.shape(1), w = x.shape(2), c = x.shape(3) / 4;
    auto out = mx::reshape(x, {n, h, w, c, 2, 2});
    out = mx::transpose(out, {0, 1, 4, 2, 5, 3});
    return mx::reshape(out, {n, h * 2, w * 2, c});
}

} // namespace

TAESuperDecoder::TAESuperDecoder(const std::filesystem::path &checkpoint) {
    require(checkpoint.is_absolute() && std::filesystem::is_regular_file(checkpoint),
            "TAEW2.2-Super checkpoint must be an absolute regular file");
    auto loaded = mx::load_safetensors(checkpoint.string()).first;
    require(loaded.count("decoder.0.weight") && loaded.at("decoder.0.weight").shape() == mx::Shape{512, 48, 1, 1} &&
                loaded.count("decoder.1.conv.0.weight") &&
                loaded.at("decoder.1.conv.0.weight").shape() == mx::Shape{1024, 1, 7, 7} &&
                loaded.count("decoder.22.weight") && loaded.at("decoder.22.weight").shape() == mx::Shape{12, 64, 3, 3},
            "checkpoint is not TAEW2.2-Super's 48-channel decoder");
    std::vector<Tensor> materialize;
    for (const auto &[name, tensor] : loaded) {
        if (!name.starts_with("decoder.")) continue;
        require(tensor.dtype() == mx::float32 || tensor.dtype() == mx::float16 || tensor.dtype() == mx::bfloat16,
                "TAEW2.2-Super weights must be floating point");
        auto value = mx::astype(tensor, mx::float16);
        if (name.ends_with(".weight") && value.ndim() == 4) value = mx::transpose(value, {0, 2, 3, 1});
        weights_.emplace(name, value);
        materialize.push_back(value);
    }
    require(weights_.size() >= 30, "TAEW2.2-Super decoder is incomplete");
    mx::eval(materialize);
}

Tensor TAESuperDecoder::conv(const Tensor &x, const std::string &prefix, int padding, int groups) const {
    auto found = weights_.find(prefix + ".weight");
    require(found != weights_.end(), "missing TAEW2.2-Super weight: " + prefix);
    auto value = mx::conv2d(x, found->second, {1, 1}, {padding, padding}, {1, 1}, groups);
    auto bias = weights_.find(prefix + ".bias");
    if (bias != weights_.end()) value = value + bias->second;
    return value;
}

Tensor TAESuperDecoder::memory_block(const Tensor &x, int index) const {
    require(x.ndim() == 4 && x.shape(0) > 0, "invalid TAEW2.2-Super memory block input");
    const auto prefix = "decoder." + std::to_string(index) + ".conv.";
    const int frames = x.shape(0), channels = x.shape(3);
    auto previous = mx::concatenate({mx::zeros_like(slice_axis(x, 0, 0, 1)), slice_axis(x, 0, 0, frames - 1)}, 0);
    auto value = mx::concatenate({x, previous}, 3);
    value = conv(value, prefix + "0", 3, channels * 2);
    value = relu(conv(value, prefix + "1"));
    value = conv(value, prefix + "3");
    return value + x;
}

Tensor TAESuperDecoder::temporal_grow(const Tensor &x, int index, int stride) const {
    const auto prefix = "decoder." + std::to_string(index) + ".conv";
    auto found = weights_.find(prefix + ".weight");
    require(found != weights_.end(), "missing TAEW2.2-Super temporal-grow weight");
    auto weight = found->second;
    const int channels = x.shape(3), count = channels * stride;
    require(weight.shape(0) >= count && weight.shape(3) == channels,
            "invalid TAEW2.2-Super temporal-grow geometry");
    // Upstream uses only the trailing output channels for stride one.
    weight = slice_axis(weight, 0, weight.shape(0) - count, weight.shape(0));
    auto value = mx::conv2d(x, weight);
    if (stride == 1) return value;
    const int frames = value.shape(0), height = value.shape(1), width = value.shape(2);
    value = mx::reshape(value, {frames, height, width, stride, channels});
    value = mx::transpose(value, {0, 3, 1, 2, 4});
    return mx::reshape(value, {frames * stride, height, width, channels});
}

Tensor TAESuperDecoder::decode(const Tensor &latents, const Event &event, std::atomic<bool> *cancelled) const {
    require(latents.ndim() == 5 && latents.shape(0) == 1 && latents.shape(1) == 48 && latents.shape(2) > 0 &&
                latents.shape(3) > 0 && latents.shape(4) > 0,
            "TAEW2.2-Super expects normalized [1,48,F,H,W] latents");
    const int frames = latents.shape(2), height = latents.shape(3), width = latents.shape(4);
    auto x = mx::squeeze(mx::transpose(mx::astype(latents, mx::float16), {0, 2, 3, 4, 1}), 0);
    x = conv(x, "decoder.0");
    int completed = 0;
    for (const auto &stage : {std::array<int, 5>{1, 2, 3, 4, 7},
                              std::array<int, 5>{8, 9, 10, 11, 14},
                              std::array<int, 5>{15, 16, 17, 18, 21}}) {
        for (int i = 0; i < 3; ++i) {
            if (cancelled) checkpoint(*cancelled);
            x = memory_block(x, stage[i]);
            mx::eval(x);
            if (event) event("wan22_tae_super", ++completed, 9);
        }
        x = pixel_shuffle(relu(conv(x, "decoder." + std::to_string(stage[3]), 1)));
        x = temporal_grow(x, stage[4], stage[4] == 7 ? 1 : 2);
        mx::eval(x);
    }
    x = pixel_shuffle(conv(x, "decoder.22", 1));
    require(x.shape() == mx::Shape{frames * 4, height * 16, width * 16, 3},
            "TAEW2.2-Super produced invalid video geometry");
    x = slice_axis(x, 0, 3, frames * 4);
    x = mx::clip(x, Tensor(0.f, x.dtype()), Tensor(1.f, x.dtype()));
    x = mx::expand_dims(mx::transpose(x, {3, 0, 1, 2}), 0);
    mx::eval(x);
    if (cancelled) checkpoint(*cancelled);
    return x;
}

} // namespace tc::wan22

#include "taehv.hpp"

namespace tc::components {
namespace {
const Tensor &get(const std::unordered_map<std::string, Tensor> &w, const std::string &key) {
    auto found = w.find(key);
    require(found != w.end(), "missing TAEHV tensor: " + key);
    return found->second;
}
Tensor relu(const Tensor &x) { return mx::maximum(x, Tensor(0.f, x.dtype())); }
Tensor upsample2(const Tensor &x) {
    auto shape = x.shape();
    require(shape.size() == 4, "TAEHV upsample expects NCHW");
    auto expanded = mx::reshape(x, {shape[0], shape[1], shape[2], 1, shape[3], 1});
    return mx::reshape(mx::broadcast_to(expanded,
        {shape[0], shape[1], shape[2], 2, shape[3], 2}),
        {shape[0], shape[1], shape[2] * 2, shape[3] * 2});
}
} // namespace

TAEHVDecoder::TAEHVDecoder(const std::filesystem::path &path, int latent_channels)
    : latent_channels_(latent_channels) {
    require(path.is_absolute() && std::filesystem::is_regular_file(path),
            "TAEHV checkpoint must be an absolute regular file");
    require(latent_channels == 16, "native TAEHV currently supports Wan2.1's 16-channel checkpoint only");
    auto checkpoint = mx::load_safetensors(path.string());
    require(checkpoint.second["architecture"] == "taew2_1" &&
                checkpoint.second["source_sha256"] ==
                    "d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e",
            "TAEHV checkpoint needs the offline converter's source/architecture metadata");
    auto &loaded = checkpoint.first;
    for (auto &[key, value] : loaded) {
        require(value.dtype() == mx::float32 || value.dtype() == mx::float16 || value.dtype() == mx::bfloat16,
                "TAEHV weights must be floating point");
        weights_.emplace(std::move(key), mx::astype(value, mx::float32));
    }
    require(weights_.size() >= 20, "TAEHV checkpoint is incomplete");
}

Tensor TAEHVDecoder::conv(const Tensor &x, const std::string &key, int stride) const {
    const auto &weight = get(weights_, key + ".weight");
    auto y = mx::conv2d(mx::transpose(x, {0, 2, 3, 1}),
                        mx::transpose(weight, {0, 2, 3, 1}), {stride, stride}, {1, 1});
    y = mx::transpose(y, {0, 3, 1, 2});
    if (weights_.count(key + ".bias")) y = y + mx::reshape(get(weights_, key + ".bias"), {1, -1, 1, 1});
    return y;
}
Tensor TAEHVDecoder::conv1(const Tensor &x, const std::string &key) const {
    const auto &weight = get(weights_, key + ".weight");
    auto y = mx::conv2d(mx::transpose(x, {0, 2, 3, 1}),
                        mx::transpose(weight, {0, 2, 3, 1}), {1, 1}, {0, 0});
    y = mx::transpose(y, {0, 3, 1, 2});
    if (weights_.count(key + ".bias")) y = y + mx::reshape(get(weights_, key + ".bias"), {1, -1, 1, 1});
    return y;
}
Tensor TAEHVDecoder::past(const Tensor &x, int batch) const {
    auto shape = x.shape();
    auto temporal = shape[0] / batch;
    auto grouped = mx::reshape(x, {batch, temporal, shape[1], shape[2], shape[3]});
    auto zero = mx::zeros_like(slice_axis(grouped, 1, 0, 1));
    auto previous = mx::concatenate({zero, slice_axis(grouped, 1, 0, temporal - 1)}, 1);
    return mx::reshape(previous, shape);
}
Tensor TAEHVDecoder::memory_block(const Tensor &x, const Tensor &previous, int index) const {
    auto key = "decoder." + std::to_string(index);
    auto y = relu(conv(mx::concatenate({x, previous}, 1), key + ".conv.0"));
    y = relu(conv(y, key + ".conv.2"));
    y = conv(y, key + ".conv.4");
    if (weights_.count(key + ".skip.weight")) y = y + conv1(x, key + ".skip"); else y = y + x;
    return relu(y);
}
Tensor TAEHVDecoder::temporal_grow(const Tensor &x, int index, int stride) const {
    const auto key = "decoder." + std::to_string(index) + ".conv.weight";
    auto weight = get(weights_, key);
    const int out_channels = x.shape(1) * stride;
    require(weight.ndim() == 4 && weight.shape(0) >= out_channels, "invalid TAEHV temporal-grow weight");
    // Upstream stores a wider temporal kernel; only its trailing output
    // channels implement the selected stride (including stride one).
    weight = slice_axis(weight, 0, weight.shape(0) - out_channels, weight.shape(0));
    auto y = mx::transpose(mx::conv2d(mx::transpose(x, {0, 2, 3, 1}),
                                     mx::transpose(weight, {0, 2, 3, 1}), {1, 1}, {0, 0}), {0, 3, 1, 2});
    if (stride == 1) return y;
    auto shape = y.shape();
    require(shape[1] % stride == 0, "invalid TAEHV temporal-grow channels");
    auto channels = shape[1] / stride;
    y = mx::reshape(y, {shape[0], stride, channels, shape[2], shape[3]});
    y = mx::transpose(y, {0, 1, 2, 3, 4});
    return mx::reshape(y, {shape[0] * stride, channels, shape[2], shape[3]});
}

Tensor TAEHVDecoder::decode_ntchw(const Tensor &input, const Event &event,
                                  std::atomic<bool> &cancelled) const {
    checkpoint(cancelled);
    require(input.ndim() == 5 && input.shape(0) > 0 && input.shape(1) > 0 &&
                input.shape(3) > 0 && input.shape(4) > 0 && input.shape(2) == latent_channels_,
            "TAEHV expects NTCHW input with configured latent channels");
    const int batch = input.shape(0);
    auto shape = input.shape();
    auto x = mx::reshape(mx::astype(input, mx::float32), {shape[0] * shape[1], shape[2], shape[3], shape[4]});
    x = mx::tanh(x / Tensor(3.f, x.dtype())) * Tensor(3.f, x.dtype());
    x = relu(conv(x, "decoder.1"));
    for (int index : {3, 4, 5}) { checkpoint(cancelled); event("taehv", index, 23); x = memory_block(x, past(x, batch), index); mx::eval(x); }
    x = upsample2(x); x = temporal_grow(x, 7, 1); x = conv(x, "decoder.8");
    for (int index : {9, 10, 11}) { checkpoint(cancelled); event("taehv", index, 23); x = memory_block(x, past(x, batch), index); mx::eval(x); }
    x = upsample2(x); x = temporal_grow(x, 13, 2); x = conv(x, "decoder.14");
    for (int index : {15, 16, 17}) { checkpoint(cancelled); event("taehv", index, 23); x = memory_block(x, past(x, batch), index); mx::eval(x); }
    x = upsample2(x); x = temporal_grow(x, 19, 2); x = relu(conv(x, "decoder.20"));
    x = conv(x, "decoder.22");
    auto out = mx::reshape(x, {batch, x.shape(0) / batch, 3, x.shape(2), x.shape(3)});
    if (trim_frames_ > 0 && out.shape(1) > trim_frames_)
        out = slice_axis(out, 1, trim_frames_, out.shape(1));
    out = mx::clip(out, Tensor(0.f, out.dtype()), Tensor(1.f, out.dtype()));
    mx::eval(out);
    checkpoint(cancelled);
    event("taehv", 23, 23);
    return out;
}

} // namespace tc::components

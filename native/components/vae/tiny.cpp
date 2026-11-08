#include "tiny.hpp"
#include "../../media/image.hpp"
#include "../../platform/apple/platform.hpp"
#include <algorithm>
#include <cmath>
#include <set>

namespace tc::components {
namespace {
Tensor relu(const Tensor &x) {
    return mx::maximum(x, Tensor(0.f, x.dtype()));
}
Tensor upsample(const Tensor &x) {
    const auto &s = x.shape(); // NHWC nearest-neighbor, exactly PyTorch's 2x repeat.
    return mx::reshape(mx::broadcast_to(mx::reshape(x, {s[0], s[1], 1, s[2], 1, s[3]}),
                                       {s[0], s[1], 2, s[2], 2, s[3]}),
                       {s[0], s[1] * 2, s[2] * 2, s[3]});
}
} // namespace

Tensor flow_preview_estimate(const Tensor &sample, const Tensor &velocity, float sigma) {
    require(sample.shape() == velocity.shape() && std::isfinite(sigma) && sigma >= 0.f,
            "invalid flow preview sample, velocity or sigma");
    // FP32 here is preview-only. Preserve the sampler's dtype and rounding.
    return mx::astype(sample, mx::float32) - mx::astype(velocity, mx::float32) * sigma;
}

TinyAutoencoder::TinyAutoencoder(const std::filesystem::path &path, const std::string &model,
                                 bool encoder, bool decoder)
    : qwen_(model == "qwen-image-2.1") {
    require(qwen_ || model == "z-image-turbo" || model == "z-image-turbo-gguf",
            "unsupported tiny VAE model");
    require(path.is_absolute() && std::filesystem::is_regular_file(path),
            "tiny VAE requires an absolute safetensors file");
    require(std::filesystem::file_size(path) <= (64ull << 20), "tiny VAE checkpoint exceeds 64 MiB");
    auto [values, metadata] = mx::load_safetensors(path.string());
    const bool converted = metadata["format"] == "turbocider-tiny-v1";
    if (converted) {
        require(metadata["architecture"] == (qwen_ ? "taeqi2_1" : "taef1") &&
                    metadata["latent_space"] == "diffusion_normalized",
                "tiny VAE architecture/format mismatch");
    } else {
        // Official HF assets use Diffusers layer names (decoder's Clamp is
        // outside layers). Accept only pinned, fully hashed source bytes,
        // never infer compatibility from dimensions or a renamed checkpoint.
        const auto expected_digest = qwen_
            ? "6578b31cc5e336233a2c842eba61b56425bbcac97c7088705f7c7db93291887e"
            : "47a6c2bff850da04b267cab70fe3553fef57255eb9a8e76852baa0a87850e54d";
        require(sha256_file(path) == expected_digest,
                "tiny VAE architecture/format mismatch: not a pinned official Hugging Face asset");
        std::unordered_map<std::string, Tensor> canonical;
        for (const auto &[key, value] : values) {
            const bool decode = key.starts_with("decoder.layers.");
            require(decode || key.starts_with("encoder.layers."), "unknown official tiny VAE tensor");
            const auto begin = std::string("decoder.layers.").size(), end = key.find('.', begin);
            require(end != std::string::npos, "invalid official tiny VAE key");
            const auto index = std::stoi(key.substr(begin, end - begin)) + (decode ? 1 : 0);
            const auto name = std::string(decode ? "decoder." : "encoder.") + std::to_string(index) +
                              key.substr(end);
            require(canonical.emplace(name, mx::astype(value, mx::float32)).second,
                    "duplicate official tiny VAE coefficient");
        }
        values = std::move(canonical);
    }
    // Validate the complete expected graph, including unused halves, before
    // dispatching. Do not infer architecture from an ambiguous tensor shape.
    std::set<std::string> expected;
    auto coefficient = [&](const std::string &key, const mx::Shape &shape) {
        auto found = values.find(key);
        require(found != values.end() && found->second.shape() == shape &&
                    found->second.dtype() == mx::float32,
                "invalid tiny VAE coefficient: " + key);
        expected.insert(key);
        if ((encoder && key.starts_with("encoder.")) || (decoder && key.starts_with("decoder."))) {
            auto value = found->second;
            if (value.ndim() == 4)
                value = mx::contiguous(mx::transpose(value, {0, 2, 3, 1}));
            weights_.emplace(key, value);
        }
    };
    auto convolution = [&](const std::string &key, int in, int out, int kernel = 3, bool bias = true) {
        coefficient(key + ".weight", {out, in, kernel, kernel});
        if (bias)
            coefficient(key + ".bias", {out});
    };
    auto residual = [&](const std::string &key, int width) {
        for (int i : {0, 2, 4})
            convolution(key + ".conv." + std::to_string(i), width, width);
    };
    const int c = channels();
    convolution("encoder." + std::to_string(qwen_ ? 1 : 0), qwen_ ? 16 : 3, 64);
    if (!qwen_)
        residual("encoder.1", 64);
    else
        residual("encoder.3", 64);
    for (int stage = 0; stage < 3; ++stage) {
        int i = (qwen_ ? 4 : 2) + 4 * stage;
        int in = qwen_ && stage == 2 ? 128 : 64, out = qwen_ ? 64 * (1 << stage) : 64;
        convolution("encoder." + std::to_string(i), in, out, 3, false);
        for (int j = 1; j <= 3; ++j)
            residual("encoder." + std::to_string(i + j), out);
    }
    convolution("encoder." + std::to_string(qwen_ ? 16 : 14), qwen_ ? 256 : 64, c);
    convolution("decoder.1", c, qwen_ ? 256 : 64);
    for (int stage = 0; stage < 3; ++stage) {
        int i = 3 + 5 * stage, in = qwen_ ? 256 / (1 << stage) : 64;
        int out = qwen_ && stage < 2 ? in / 2 : 64;
        for (int j = 0; j < 3; ++j)
            residual("decoder." + std::to_string(i + j), in);
        convolution("decoder." + std::to_string(i + 4), in, out, 3, false);
    }
    residual("decoder.18", 64);
    convolution("decoder.19", 64, qwen_ ? 16 : 3);
    require(values.size() == expected.size(), "unexpected tiny VAE checkpoint tensors");
    std::vector<Tensor> loaded, finite;
    for (const auto &[key, value] : weights_) {
        loaded.push_back(value);
        finite.push_back(mx::all(mx::isfinite(value)));
    }
    mx::eval(loaded);
    require(!finite.empty() && mx::all(mx::stack(finite)).item<bool>(), "nonfinite tiny VAE weights");
}

const Tensor &TinyAutoencoder::get(const std::string &key) const {
    auto it = weights_.find(key);
    require(it != weights_.end(), "tiny VAE component not loaded: " + key);
    return it->second;
}
Tensor TinyAutoencoder::conv(const Tensor &x, const std::string &key, int stride) const {
    auto out = mx::conv2d(x, get(key + ".weight"), {stride, stride}, {1, 1});
    if (weights_.contains(key + ".bias"))
        out = out + get(key + ".bias");
    return out;
}
Tensor TinyAutoencoder::block(const Tensor &x, const std::string &key) const {
    return relu(conv(relu(conv(relu(conv(x, key + ".conv.0")), key + ".conv.2")), key + ".conv.4") + x);
}
Tensor TinyAutoencoder::encode(const Tensor &input, const Event &event,
                               std::atomic<bool> &cancelled) const {
    checkpoint(cancelled);
    require(input.ndim() == 4 && input.shape(0) == 1 &&
                (input.shape(1) == 3 || (qwen_ && input.shape(1) == 4)) && input.shape(2) > 0 &&
                input.shape(3) > 0 && input.shape(2) % scale() == 0 && input.shape(3) % scale() == 0,
            "tiny VAE expects NCHW unit-range image, spatial size divisible by its scale");
    auto x = mx::astype(input, mx::float32);
    require(mx::all(mx::isfinite(x)).item<bool>(), "nonfinite tiny VAE image");
    if (qwen_ && x.shape(1) == 3)
        x = mx::concatenate({x, mx::ones({1, 1, x.shape(2), x.shape(3)}, mx::float32)}, 1);
    if (qwen_) { // PixelUnshuffle channel order: C, patch Y, patch X.
        int h = x.shape(2) / 2, w = x.shape(3) / 2;
        x = mx::reshape(mx::transpose(mx::reshape(x, {1, 4, h, 2, w, 2}), {0, 1, 3, 5, 2, 4}),
                        {1, 16, h, w});
    }
    x = mx::transpose(x, {0, 2, 3, 1});
    x = conv(x, qwen_ ? "encoder.1" : "encoder.0");
    if (qwen_)
        x = relu(x);
    x = block(x, qwen_ ? "encoder.3" : "encoder.1");
    mx::eval(x);
    for (int stage = 0; stage < 3; ++stage) {
        checkpoint(cancelled);
        int i = (qwen_ ? 4 : 2) + stage * 4;
        x = conv(x, "encoder." + std::to_string(i), 2);
        for (int j = 1; j <= 3; ++j)
            x = block(x, "encoder." + std::to_string(i + j));
        mx::eval(x);
        if (event)
            event("tiny_vae_encode", stage + 1, 3);
    }
    x = mx::transpose(conv(x, qwen_ ? "encoder.16" : "encoder.14"), {0, 3, 1, 2});
    mx::eval(x);
    require(mx::all(mx::isfinite(x)).item<bool>(), "nonfinite tiny VAE latent");
    checkpoint(cancelled);
    return x;
}
Tensor TinyAutoencoder::decode(const Tensor &input, const Event &event,
                               std::atomic<bool> &cancelled) const {
    checkpoint(cancelled);
    require(input.ndim() == 4 && input.shape(0) == 1 && input.shape(1) == channels() &&
                input.shape(2) > 0 && input.shape(3) > 0,
            "tiny VAE latent channel/shape mismatch");
    auto x = mx::astype(mx::transpose(input, {0, 2, 3, 1}), mx::float32);
    require(mx::all(mx::isfinite(x)).item<bool>(), "nonfinite tiny VAE latent");
    x = relu(conv(mx::tanh(x / 3.f) * 3.f, "decoder.1"));
    for (int stage = 0; stage < 3; ++stage) {
        checkpoint(cancelled);
        int i = 3 + stage * 5;
        for (int j = 0; j < 3; ++j)
            x = block(x, "decoder." + std::to_string(i + j));
        mx::eval(x);
        if (event)
            event("tiny_vae_decode", stage + 1, 4);
        x = conv(upsample(x), "decoder." + std::to_string(i + 4));
    }
    x = conv(block(x, "decoder.18"), "decoder.19");
    if (qwen_) { // PixelShuffle channel order: C, patch Y, patch X.
        int h = x.shape(1), w = x.shape(2);
        x = mx::reshape(mx::transpose(mx::reshape(x, {1, h, w, 4, 2, 2}), {0, 1, 4, 2, 5, 3}),
                        {1, h * 2, w * 2, 4});
    }
    x = mx::transpose(x, {0, 3, 1, 2});
    mx::eval(x);
    require(mx::all(mx::isfinite(x)).item<bool>(), "nonfinite tiny VAE pixels");
    checkpoint(cancelled);
    if (event)
        event("tiny_vae_decode", 4, 4);
    return x;
}

TinyPreview::TinyPreview(const Request &r, const Event &event, std::atomic<bool> &cancelled)
    : request_(r), event_(event), cancelled_(cancelled) {
    if (r.tiny_vae && r.tiny_vae->preview && event.preview) {
        checkpoint(cancelled);
        require(r.tiny_vae->interval >= 1 && r.tiny_vae->interval <= 50 && r.steps >= 1 &&
                    !r.output.empty(),
                "invalid tiny preview interval/steps/output");
        path_ = r.output + ".preview.png";
        require(!std::filesystem::exists(std::filesystem::symlink_status(path_)),
                "refusing to replace an existing tiny preview sidecar");
        decoder_ = std::make_unique<TinyAutoencoder>(r.tiny_vae->path, r.model, false, true);
    }
}
TinyPreview::~TinyPreview() {
    if (!path_.empty()) {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }
}
bool TinyPreview::due(int step) const {
    return decoder_ && step > 0 && step <= request_.steps &&
           (step % request_.tiny_vae->interval == 0 || step == request_.steps);
}
void TinyPreview::publish(const Tensor &latent, int completed) {
    if (!due(completed))
        return;
    require(latent.ndim() == 4 && int64_t(latent.shape(2)) * decoder_->scale() == request_.height &&
                int64_t(latent.shape(3)) * decoder_->scale() == request_.width,
            "tiny preview latent does not match requested output dimensions");
    // Do not decimate the latent: previews have the same pixel dimensions and
    // aspect ratio as the final image, including the full Qwen RGBA output.
    auto pixels = decoder_->decode(latent, {}, cancelled_);
    auto image = mx::transpose(mx::clip(pixels, Tensor(0.f), Tensor(1.f)) * 2.f - 1.f, {0, 2, 3, 1});
    checkpoint(cancelled_);
    if (pixels.shape(1) == 4)
        save_rgba_png(image, path_);
    else
        save_png(image, path_);
    event_.preview(path_.string(), completed, request_.steps);
}
} // namespace tc::components

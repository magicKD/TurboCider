#include "media.hpp"
#include <algorithm>
#include <cmath>

namespace tc::seedvr2 {
namespace {
float cubic(float distance) {
    constexpr float a = -.75f;
    const float x = std::abs(distance);
    if (x <= 1.f) return ((a + 2.f) * x - (a + 3.f)) * x * x + 1.f;
    if (x < 2.f) return ((a * x - 5.f * a) * x + 8.f * a) * x - 4.f * a;
    return 0.f;
}
Tensor resize_axis(const Tensor &input, int size, int axis) {
    const int original = input.shape(axis);
    if (original == size) return input;
    std::vector<Tensor> terms;
    for (int tap = 0; tap < 4; ++tap) {
        std::vector<int32_t> indices(size);
        std::vector<float> weights(size);
        for (int i = 0; i < size; ++i) {
            const float coordinate = (i + .5f) * (float(original) / size) - .5f;
            const int center = int(std::floor(coordinate));
            const int index = center - 1 + tap;
            indices[i] = std::clamp(index, 0, original - 1);
            weights[i] = cubic(coordinate - float(index));
        }
        mx::Shape shape(input.ndim(), 1);
        shape[axis] = size;
        auto gathered = mx::take(input, Tensor(indices.data(), {size}, mx::int32), axis);
        terms.push_back(gathered * mx::reshape(Tensor(weights.data(), {size}, mx::float32), shape));
    }
    return ((terms[0] + terms[1]) + terms[2]) + terms[3];
}
} // namespace
Tensor resize_bicubic(const Tensor &pixels, int height, int width) {
    require(pixels.ndim() == 4 && pixels.shape(0) > 0 && pixels.shape(1) > 0 && pixels.shape(2) > 0 &&
            (pixels.shape(3) == 3 || pixels.shape(3) == 4) && height > 0 && width > 0,
            "SeedVR2 resize requires nonempty NHWC RGB/RGBA pixels");
    auto y = resize_axis(mx::astype(pixels, mx::float32), width, 2);
    y = resize_axis(y, height, 1);
    return mx::clip(y, Tensor(0.f), Tensor(1.f));
}
NormalizedMediaInput consume_resized_input(Tensor &unit, bool video, bool own_image_alpha) {
    require(unit.ndim() == 4 && unit.dtype() == mx::float32 && unit.shape(0) > 0 &&
            unit.shape(1) > 0 && unit.shape(2) > 0 && unit.shape(3) == (video ? 3 : 4) &&
            (video || unit.shape(0) == 1), "invalid resized SeedVR2 source pixels");
    auto channels = [&](int first, int last) {
        return mx::slice(unit,{0,0,0,first},{unit.shape(0),unit.shape(1),unit.shape(2),last});
    };
    auto pixels = mx::expand_dims(mx::transpose(channels(0,3)*2.f-1.f,{3,0,1,2}),0);
    Tensor alpha(0);
    if (video) mx::eval(pixels);
    else if (own_image_alpha) {
        // Contiguity alone is not an ownership guarantee: force a copy so
        // even a singleton/stride-compatible view cannot retain source RGB.
        alpha = mx::copy(mx::contiguous(channels(3,4)));
        mx::eval({pixels,alpha});
    } else {
        // Preserve the original image residency/evaluation path when speed
        // is preferred. This view intentionally retains full source RGBA.
        alpha = channels(3,4);
        mx::eval(pixels);
    }
    unit = Tensor(0);
    return {std::move(pixels),std::move(alpha),!video && own_image_alpha};
}
} // namespace tc::seedvr2

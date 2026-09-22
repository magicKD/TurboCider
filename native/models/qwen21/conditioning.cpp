#include "conditioning.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tc::qwen21 {
namespace {
// Python's round uses ties-to-even; do not depend on the process rounding mode.
int round_even(double x) {
    double lo = std::floor(x), fraction = x - lo;
    return int(lo + (fraction > .5 || (fraction == .5 && int64_t(lo) % 2)));
}
Tensor resize_axis(const Tensor &input, int axis, int size) {
    int old = input.shape(axis);
    if (old == size) return input;
    auto coordinate = mx::maximum((mx::arange(size, mx::float32) + .5f) * (float(old) / size) - .5f, Tensor(0.f));
    auto lower = mx::astype(coordinate, mx::int32);
    auto upper = mx::minimum(lower + 1, Tensor(old - 1));
    mx::Shape shape(input.ndim(), 1);
    shape[axis] = size;
    auto fraction = mx::reshape(coordinate - mx::astype(lower, mx::float32), shape);
    return mx::take(input, lower, axis) * (1.f - fraction) + mx::take(input, upper, axis) * fraction;
}
}
Tensor resize_reference(const Tensor &input, int resolution) {
    require(input.ndim() == 4 && input.shape(0) == 1 && input.shape(3) == 4 &&
            input.shape(1) > 0 && input.shape(2) > 0, "Qwen21 reference must be NHWC RGBA");
    require(resolution >= 32 && resolution <= 2048, "invalid Qwen21 reference resolution");
    const int h = input.shape(1), w = input.shape(2);
    const double ratio = double(w) / h;
    const int rw = round_even(std::sqrt(double(resolution) * resolution * ratio) / 32) * 32;
    const int rh = round_even(std::sqrt(double(resolution) * resolution / ratio) / 32) * 32;
    require(rh >= 32 && rw >= 32 && int64_t(rh) * rw <= 12845056,
            "Qwen21 reference aspect ratio exceeds supported area");
    if (rh == h && rw == w) return input;
    auto pixels = mx::astype(input, mx::float32);
    mx::eval(pixels);
    std::vector<float> source(pixels.data<float>(), pixels.data<float>() + pixels.size());
    for (size_t i = 0; i < source.size(); i += 4)
        for (int c = 0; c < 3; ++c) source[i + c] *= source[i + 3];
    auto filter = [](double x) {
        x = std::abs(x);
        if (x == 0) return 1.;
        if (x >= 3) return 0.;
        constexpr double pi = 3.14159265358979323846;
        const double a = pi * x;
        return std::sin(a) / a * std::sin(a / 3) / (a / 3);
    };
    auto resize = [&](const std::vector<float> &src, int width, int height, int size, bool horizontal) {
        const int old = horizontal ? width : height;
        const int dw = horizontal ? size : width, dh = horizontal ? height : size;
        std::vector<float> dst(size_t(dw) * dh * 4);
        const double scale = double(old) / size, spread = std::max(1., scale);
        for (int target = 0; target < size; ++target) {
            const double center = (target + .5) * scale;
            const int begin = std::max(0, int(center - 3 * spread + .5));
            const int end = std::min(old, int(center + 3 * spread + .5));
            std::vector<double> weights(end - begin);
            double total = 0;
            for (int i = begin; i < end; ++i) total += weights[i - begin] = filter((i + .5 - center) / spread);
            for (int other = 0; other < (horizontal ? height : width); ++other) {
                const size_t dest = (size_t(horizontal ? other : target) * dw + (horizontal ? target : other)) * 4;
                for (int c = 0; c < 4; ++c) {
                    double value = 0;
                    for (int i = begin; i < end; ++i) {
                        const size_t index = (size_t(horizontal ? other : i) * width + (horizontal ? i : other)) * 4;
                        value += src[index + c] * weights[i - begin];
                    }
                    dst[dest + c] = float(value / total);
                }
            }
        }
        return dst;
    };
    auto output = resize(resize(source, w, h, rw, true), rw, h, rh, false);
    for (size_t i = 0; i < output.size(); i += 4) {
        const float alpha = std::clamp(output[i + 3], 0.f, 1.f);
        for (int c = 0; c < 3; ++c)
            output[i + c] = alpha > 1e-8f ? std::clamp(output[i + c] / alpha, 0.f, 1.f) : 0.f;
        output[i + 3] = alpha;
    }
    return Tensor(output.data(), {1, rh, rw, 4}, mx::float32);
}
VisionInput vision_input(const Tensor &pixels, int min_pixels, int max_pixels) {
    require(pixels.ndim() == 4 && pixels.shape(0) == 1 && pixels.shape(3) == 3 &&
            pixels.shape(1) > 0 && pixels.shape(2) > 0, "Qwen21 vision expects one NHWC RGB image");
    require(min_pixels >= 1024 && max_pixels >= min_pixels && max_pixels <= 12845056,
            "invalid Qwen21 visual pixel budget");
    const int h = pixels.shape(1), w = pixels.shape(2), factor = 32;
    int rh = round_even(double(h) / factor) * factor, rw = round_even(double(w) / factor) * factor;
    if (int64_t(rh) * rw > max_pixels) {
        double beta = std::sqrt(double(h) * w / max_pixels);
        rh = std::max(factor, int(std::floor(h / beta / factor)) * factor);
        rw = std::max(factor, int(std::floor(w / beta / factor)) * factor);
    } else if (int64_t(rh) * rw < min_pixels) {
        double beta = std::sqrt(double(min_pixels) / (double(h) * w));
        rh = int(std::ceil(h * beta / factor)) * factor;
        rw = int(std::ceil(w * beta / factor)) * factor;
    }
    require(rh > 0 && rw > 0 && int64_t(rh) * rw <= 2LL * max_pixels,
            "Qwen21 visual aspect ratio exceeds pixel budget");
    auto resized = resize_axis(resize_axis(mx::astype(pixels, mx::float32), 2, rw), 1, rh);
    auto normalized = (resized - .5f) / .5f;
    // [H/32,2,16,W/32,2,16,C] -> merge-group, channel, time, patch.
    auto patches = mx::reshape(normalized, {rh / 32, 2, 16, rw / 32, 2, 16, 3});
    patches = mx::transpose(patches, {0, 3, 1, 4, 6, 2, 5});
    patches = mx::repeat(mx::expand_dims(patches, 5), 2, 5);
    return {mx::reshape(patches, {rh / 16 * (rw / 16), 1536}), rh / 16, rw / 16};
}

std::string reference_prompt_template(const std::string &prompt, size_t count) {
    require(count <= 10, "Qwen21 supports at most ten reference images");
    std::string refs;
    for (size_t i = 0; i < count; ++i) {
        if (i) refs += " ";
        refs += "<image" + std::to_string(i + 1) + "><|vision_start|><|image_pad|><|vision_end|>";
    }
    return TextEncoder::prompt_template(refs + prompt);
}

MultimodalPrompt assemble_prompt(const Tokens &tokens, const Tensor &table,
                                const std::vector<VisualReference> &refs, int max_tokens) {
    require(table.ndim() == 2 && table.shape(1) > 0, "invalid Qwen21 embedding table");
    require(tokens.valid > 0 && tokens.valid == int(tokens.ids.size()), "multimodal prompt must be unpadded");
    require(refs.size() <= 10 && max_tokens > 0, "invalid Qwen21 reference/token limit");
    constexpr int image_pad = 151655, im_start = 151644;
    int64_t count = tokens.valid;
    size_t levels = refs.empty() ? 0 : refs[0].features.deepstack.size();
    for (const auto &ref : refs) {
        int64_t n = int64_t(ref.grid_height / 2) * (ref.grid_width / 2);
        require(ref.grid_height > 0 && ref.grid_width > 0 && ref.grid_height % 2 == 0 && ref.grid_width % 2 == 0 &&
                n <= max_tokens && ref.features.merged.shape() == mx::Shape{int(n), table.shape(1)},
                "Qwen21 visual features/grid mismatch");
        require(ref.features.deepstack.size() == levels, "Qwen21 inconsistent visual deepstack levels");
        for (const auto &delta : ref.features.deepstack)
            require(delta.shape() == ref.features.merged.shape(), "Qwen21 visual deepstack shape mismatch");
        count += n - 1;
    }
    require(count > 0 && count <= max_tokens, "Qwen21 expanded prompt exceeds token limit");
    require(std::count(tokens.ids.begin(), tokens.ids.end(), image_pad) == int(refs.size()),
            "Qwen21 image placeholders/reference count mismatch");
    int starts = 0, drop = -1;
    for (size_t i = 0; i < tokens.ids.size(); ++i) {
        int id = tokens.ids[i];
        require(id >= 0 && id < table.shape(0), "Qwen21 prompt token out of vocabulary");
        if (id == im_start && ++starts == 2) { drop = int(i); break; }
    }
    require(drop >= 0 && std::find(tokens.ids.begin(), tokens.ids.begin() + drop, image_pad) == tokens.ids.begin() + drop,
            "Qwen21 prompt must have a system turn before user reference images");
    std::vector<Tensor> chunks;
    std::vector<std::vector<Tensor>> deltas(levels);
    std::vector<int> positions(size_t(count) * 3), retained, slots;
    int cursor = 0, position = 0;
    size_t ref_index = 0;
    auto append_text = [&](int begin, int end) {
        if (begin == end) return;
        for (int i = begin; i < end; ++i)
            require(tokens.ids[i] >= 0 && tokens.ids[i] < table.shape(0), "Qwen21 prompt token out of vocabulary");
        auto ids = Tensor(tokens.ids.data() + begin, {end - begin}, mx::int32);
        chunks.push_back(mx::take(table, ids, 0));
        for (auto &level : deltas) level.push_back(mx::zeros({end - begin, table.shape(1)}, table.dtype()));
        for (int i = begin; i < end; ++i, ++cursor, ++position) {
            for (int axis = 0; axis < 3; ++axis) positions[size_t(axis) * count + cursor] = position;
            if (i >= drop) retained.push_back(cursor);
        }
    };
    int begin = 0;
    for (int i = 0; i < tokens.valid; ++i) {
        if (tokens.ids[i] != image_pad) continue;
        append_text(begin, i);
        const auto &ref = refs[ref_index++];
        int h = ref.grid_height / 2, w = ref.grid_width / 2;
        slots.push_back(int(retained.size()));
        chunks.push_back(mx::astype(ref.features.merged, table.dtype()));
        for (size_t level = 0; level < levels; ++level)
            deltas[level].push_back(mx::astype(ref.features.deepstack[level], table.dtype()));
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x, ++cursor) {
            positions[cursor] = position;
            positions[size_t(count) + cursor] = position + y;
            positions[size_t(count) * 2 + cursor] = position + x;
        }
        position += std::max(h, w);
        begin = i + 1;
    }
    append_text(begin, tokens.valid);
    std::vector<Tensor> expanded_deltas;
    for (auto &level : deltas) expanded_deltas.push_back(mx::expand_dims(mx::concatenate(level, 0), 0));
    return {mx::expand_dims(mx::concatenate(chunks, 0), 0),
            Tensor(positions.data(), {3, int(count)}, mx::int32), std::move(expanded_deltas),
            std::move(retained), std::move(slots)};
}

Tensor MultimodalPrompt::retain(const Tensor &hidden) const {
    require(hidden.shape() == embeddings.shape() && !retained_indices.empty(), "Qwen21 hidden/prompt shape mismatch");
    return mx::take(hidden, Tensor(retained_indices.data(), {int(retained_indices.size())}, mx::int32), 1);
}
} // namespace tc::qwen21

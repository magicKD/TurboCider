#include "conditioning.hpp"
#include <algorithm>
#include <cmath>

namespace tc::qwen21 {
namespace {
using Pixels = std::vector<uint8_t>;
enum class Filter { lanczos, bicubic };
double kernel(double x, Filter filter) {
    x = std::abs(x);
    if (filter == Filter::lanczos) {
        if (x == 0.) return 1.;
        if (x >= 3.) return 0.;
        constexpr double pi = 3.14159265358979323846;
        return std::sin(pi*x)/(pi*x) * std::sin(pi*x/3)/(pi*x/3);
    }
    // Antialiased bicubic uses a=-0.5 (not the non-AA a=-0.75 kernel).
    if (x < 1.) return ((1.5*x - 2.5)*x)*x + 1.;
    if (x < 2.) return ((-.5*x + 2.5)*x - 4.)*x + 2.;
    return 0.;
}
Pixels axis_resize(const Pixels &input, int width, int height, int size,
                   bool horizontal, Filter filter) {
    const int old = horizontal ? width : height;
    if (old == size) return input;
    const int dw = horizontal ? size : width, dh = horizontal ? height : size;
    Pixels output(size_t(dw) * dh * 3);
    const double scale = double(old) / size, spread = std::max(1., scale);
    const double radius = (filter == Filter::lanczos ? 3. : 2.) * spread;
    struct Row { int begin; std::vector<double> weights; };
    std::vector<Row> rows;
    double largest_weight = 0.;
    for (int target = 0; target < size; ++target) {
        const double center = (target + .5) * scale;
        const int begin = std::max(0, int(center - radius + .5));
        const int end = std::min(old, int(center + radius + .5));
        std::vector<double> coefficients(end - begin);
        double sum = 0.;
        for (int i = begin; i < end; ++i)
            sum += coefficients[i-begin] = kernel((i + .5 - center)/spread, filter);
        for (auto &value : coefficients) {
            value /= sum;
            largest_weight = std::max(largest_weight, value);
        }
        rows.push_back({begin, std::move(coefficients)});
    }
    // Pillow's source-cap Lanczos uses 22 bits. Torch's uint8 bicubic
    // selects one per-axis precision to keep all signed coefficients in int16.
    int precision = 22;
    if (filter == Filter::bicubic) {
        precision = 0;
        while (precision < 22 && int(.5 + largest_weight * (1 << (precision + 1))) < (1 << 15))
            ++precision;
    }
    for (int target = 0; target < size; ++target) {
        const int begin = rows[target].begin;
        const auto &coefficients = rows[target].weights;
        const int end = begin + int(coefficients.size());
        std::vector<int32_t> fixed(coefficients.size());
        for (size_t i = 0; i < fixed.size(); ++i) {
            const double value = coefficients[i] * (1 << precision);
            fixed[i] = int32_t(value < 0 ? value - .5 : value + .5);
        }
        for (int other = 0; other < (horizontal ? height : width); ++other)
            for (int c = 0; c < 3; ++c) {
                int64_t value = int64_t(1) << (precision - 1);
                for (int i = begin; i < end; ++i) {
                    const size_t index = (size_t(horizontal ? other : i) * width + (horizontal ? i : other))*3 + c;
                    value += int64_t(input[index]) * fixed[i-begin];
                }
                const size_t dest = (size_t(horizontal ? other : target)*dw + (horizontal ? target : other))*3 + c;
                output[dest] = uint8_t(std::clamp<int64_t>(value >> precision, 0, 255));
            }
    }
    return output;
}
Pixels resize(const Pixels &input, int w, int h, int rw, int rh, Filter filter) {
    return axis_resize(axis_resize(input, w, h, rw, true, filter), rw, h, rh, false, filter);
}
int round_even(double value) {
    const int lo = int(std::floor(value));
    const double fraction = value - lo;
    return lo + (fraction > .5 || (fraction == .5 && lo % 2));
}
}
VisionInput pe_vision_input(const Tensor &input, int min_pixels, int max_pixels) {
    require(input.ndim() == 4 && input.shape(0) == 1 && input.shape(1) > 0 && input.shape(2) > 0 &&
            (input.shape(3) == 3 || input.shape(3) == 4) && input.dtype() == mx::uint8,
            "PE processor requires nonempty uint8 NHWC RGB/RGBA");
    require(min_pixels >= 1024 && max_pixels >= min_pixels && max_pixels <= 16777216,
            "invalid PE processor pixel budget");
    int h = input.shape(1), w = input.shape(2);
    require(int64_t(h)*w <= 100000000, "PE input exceeds decoded image safety limit");
    auto packed = mx::contiguous(input);
    mx::eval(packed);
    Pixels pixels(size_t(h)*w*3);
    const auto *source = packed.data<uint8_t>();
    for (size_t i = 0; i < size_t(h)*w; ++i)
        for (int c = 0; c < 3; ++c) pixels[i*3+c] = source[i*input.shape(3)+c];
    // PIL RGB conversion drops alpha before resizing. No premultiplication,
    // white compositing, or 32-alignment is applied at this first stage.
    constexpr int source_cap = 1024*1024;
    if (int64_t(h)*w > source_cap) {
        const double scale = std::sqrt(double(source_cap)/(double(h)*w));
        const int rw = std::max(1, int(w*scale)), rh = std::max(1, int(h*scale));
        pixels = resize(pixels, w, h, rw, rh, Filter::lanczos);
        w = rw; h = rh;
    }
    require(double(std::max(h,w))/std::min(h,w) <= 200., "PE image aspect ratio exceeds 200");
    int rh = round_even(double(h)/32)*32, rw = round_even(double(w)/32)*32;
    if (int64_t(rh)*rw > max_pixels) {
        const double beta = std::sqrt(double(h)*w/max_pixels);
        rh = std::max(32, int(std::floor(h/beta/32))*32);
        rw = std::max(32, int(std::floor(w/beta/32))*32);
    } else if (int64_t(rh)*rw < min_pixels) {
        const double beta = std::sqrt(double(min_pixels)/(double(h)*w));
        rh = int(std::ceil(h*beta/32))*32;
        rw = int(std::ceil(w*beta/32))*32;
    }
    require(rh > 0 && rw > 0 && int64_t(rh)*rw <= 2LL*max_pixels, "PE resized image exceeds safe geometry");
    pixels = resize(pixels, w, h, rw, rh, Filter::bicubic);
    auto normalized = mx::astype(Tensor(pixels.data(), {rh,rw,3}, mx::uint8), mx::float32) * (1.f/255.f);
    normalized = (normalized - .5f) / .5f;
    auto patches = mx::reshape(normalized, {rh/32,2,16,rw/32,2,16,3});
    patches = mx::transpose(patches, {0,3,1,4,6,2,5});
    patches = mx::repeat(mx::expand_dims(patches,5),2,5);
    return {mx::reshape(patches,{rh/16*(rw/16),1536}),rh/16,rw/16};
}
}

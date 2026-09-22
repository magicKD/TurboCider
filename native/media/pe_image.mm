#include "../platform/apple/bridge.hpp"
#include "image.hpp"
#import <ImageIO/ImageIO.h>
#include <memory>
#include <type_traits>

namespace tc {
namespace {
template<class T> auto owned(T object) {
    return std::unique_ptr<std::remove_pointer_t<T>, decltype(&CFRelease)>(object, CFRelease);
}
}

Tensor load_pe_image_tensor(const std::filesystem::path &path) {
    auto source = owned(CGImageSourceCreateWithURL(
        (__bridge CFURLRef)[NSURL fileURLWithPath:@(path.c_str())], nullptr));
    require(source != nullptr, "cannot read PE image: " + path.string());
    NSDictionary *properties = CFBridgingRelease(CGImageSourceCopyPropertiesAtIndex(source.get(), 0, nullptr));
    const int64_t width = [properties[(__bridge NSString *)kCGImagePropertyPixelWidth] longLongValue];
    const int64_t height = [properties[(__bridge NSString *)kCGImagePropertyPixelHeight] longLongValue];
    require(width > 0 && height > 0 && width <= 100000000 / height,
            "PE image dimensions exceed import limit");
    auto image = owned(CGImageSourceCreateImageAtIndex(source.get(), 0, nullptr));
    require(image != nullptr, "PE image decode failed");
    require(CGImageGetWidth(image.get()) == size_t(width) && CGImageGetHeight(image.get()) == size_t(height),
            "PE decoded image dimensions differ from metadata");
    auto data = owned(CGDataProviderCopyData(CGImageGetDataProvider(image.get())));
    require(data != nullptr, "PE image has no decoded pixel data");
    const size_t bpc = CGImageGetBitsPerComponent(image.get());
    const size_t bpp = CGImageGetBitsPerPixel(image.get());
    const size_t row_bytes = CGImageGetBytesPerRow(image.get());
    const auto info = CGImageGetBitmapInfo(image.get());
    const auto alpha = CGImageGetAlphaInfo(image.get());
    const auto color = CGImageGetColorSpace(image.get());
    const auto model = CGColorSpaceGetModel(color);
    require(!(info & kCGBitmapFloatComponents), "PE floating-point file pixels are not supported");
    require(row_bytes > 0 && row_bytes <= size_t(CFDataGetLength(data.get())) / size_t(height),
            "PE decoded image has incomplete pixel rows");
    const bool indexed = model == kCGColorSpaceModelIndexed;
    const bool gray = model == kCGColorSpaceModelMonochrome;
    require(indexed || gray || model == kCGColorSpaceModelRGB,
            "PE image color model is not supported");
    require(bpc == 8 && bpp % 8 == 0 && bpp <= 32,
            "PE image requires supported 8-bit RGB/gray/indexed pixel layout");
    const size_t stride = bpp / 8;
    const size_t channels = indexed || gray ? 1 : 3;
    require(stride >= channels && stride <= channels + 1 &&
                size_t(width) <= row_bytes / stride,
            "invalid PE image channel layout");
    const auto order = info & kCGBitmapByteOrderMask;
    const bool reverse = order == kCGBitmapByteOrder32Little;
    require(order == kCGBitmapByteOrderDefault || order == kCGBitmapByteOrder32Big ||
                (reverse && stride == 4), "unsupported PE image byte order");
    const bool first = alpha == kCGImageAlphaFirst || alpha == kCGImageAlphaPremultipliedFirst ||
                       alpha == kCGImageAlphaNoneSkipFirst;
    const bool last = alpha == kCGImageAlphaLast || alpha == kCGImageAlphaPremultipliedLast ||
                      alpha == kCGImageAlphaNoneSkipLast;
    require((stride == channels && alpha == kCGImageAlphaNone) ||
                (stride == channels + 1 && (first || last)), "unsupported PE image alpha layout");
    const bool premultiplied = alpha == kCGImageAlphaPremultipliedFirst || alpha == kCGImageAlphaPremultipliedLast;
    std::vector<uint8_t> palette;
    if (indexed) {
        require(CGColorSpaceGetModel(CGColorSpaceGetBaseColorSpace(color)) == kCGColorSpaceModelRGB,
                "PE indexed image requires an RGB palette");
        palette.resize(CGColorSpaceGetColorTableCount(color) * 3);
        require(!palette.empty(), "PE image palette is empty");
        CGColorSpaceGetColorTable(color, palette.data());
    }
    const uint8_t *raw = CFDataGetBytePtr(data.get());
    std::vector<uint8_t> rgb(size_t(width * height) * 3);
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            const auto *pixel = raw + size_t(y) * row_bytes + size_t(x) * stride;
            auto sample = [&](size_t c) { return pixel[reverse ? stride - 1 - c : c]; };
            // A premultiplied decode has already lost invisible RGB. Never
            // silently pass that different image into the prompt enhancer.
            require(!premultiplied || sample(first ? 0 : stride - 1) == 255,
                    "PE image decoder discarded straight-alpha RGB; unsupported premultiplied input");
            const size_t offset = first ? 1 : 0;
            auto *out = rgb.data() + size_t(y * width + x) * 3;
            for (size_t c = 0; c < 3; ++c) {
                if (indexed) {
                    const size_t index = size_t(sample(offset)) * 3 + c;
                    require(index < palette.size(), "PE palette index out of range");
                    out[c] = palette[index];
                } else out[c] = sample(offset + (gray ? 0 : c));
            }
        }
    }
    return Tensor(rgb.data(), {1, int(height), int(width), 3}, mx::uint8);
}
} // namespace tc

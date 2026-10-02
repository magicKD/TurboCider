#include "reference_preparation.hpp"
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>

namespace tc {
namespace {
constexpr int64_t maximum_pixels = 80'000'000;

void check(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}
template<class T> auto owned(T object) {
    return std::unique_ptr<std::remove_pointer_t<T>, decltype(&CFRelease)>(object, CFRelease);
}
void validate_path(const std::filesystem::path &path, const char *label) {
    check(path.is_absolute() && path.native().find('\0') == std::string::npos,
          std::string(label) + " must be an absolute path without NUL");
}
NSString *path_string(const std::filesystem::path &path) {
    NSString *value = [[NSString alloc] initWithBytes:path.native().data()
        length:path.native().size() encoding:NSUTF8StringEncoding];
    check(value != nil, "image preparation paths must be valid UTF-8");
    return value;
}
std::filesystem::path input_path(NSDictionary *input, NSString *key) {
    id value = input[key];
    check([value isKindOfClass:NSString.class] && [value length] > 0,
          std::string(key.UTF8String) + " must be a nonempty string");
    NSData *bytes = [value dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
    check(bytes != nil, std::string(key.UTF8String) + " must be valid UTF-8");
    std::filesystem::path path(std::string(static_cast<const char *>(bytes.bytes), bytes.length));
    validate_path(path, key.UTF8String);
    return path;
}
std::pair<int, int> bounds_for(const std::string &preset) {
    if (preset == "original") return {0, 0};
    if (preset == "automatic") return {1024, 1024};
    if (preset == "fit512") return {512, 512};
    if (preset == "portrait512") return {512, 768};
    if (preset == "landscape512") return {768, 512};
    throw std::invalid_argument("unknown image preparation preset");
}
int pixel_dimension(NSDictionary *properties, CFStringRef key) {
    id value = properties[(__bridge NSString *)key];
    check([value isKindOfClass:NSNumber.class] &&
              CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
          "image has no valid pixel dimensions");
    const double dimension = [value doubleValue];
    check(std::isfinite(dimension) && dimension == std::floor(dimension) &&
              dimension > 0 && dimension <= maximum_pixels,
          "image dimensions exceed the 80000000 pixel import limit");
    return static_cast<int>(dimension);
}
std::runtime_error io_failure(const char *operation) {
    const int failure = errno;
    return std::runtime_error(std::string(operation) + ": " + std::strerror(failure));
}
struct OwnedFD {
    int value;
    explicit OwnedFD(int descriptor) : value(descriptor) {}
    OwnedFD(const OwnedFD &) = delete;
    OwnedFD &operator=(const OwnedFD &) = delete;
    ~OwnedFD() { if (value >= 0) close(value); }
};
struct TemporaryPNG {
    int directory;
    std::string name;
    OwnedFD file;
    struct stat identity{};
    bool published = false;

    explicit TemporaryPNG(int parent) : directory(parent),
        name(".tc-reference-preparation-" + std::string(NSUUID.UUID.UUIDString.UTF8String) + ".tmp"),
        file(openat(parent, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600)) {
        if (file.value < 0) throw io_failure("cannot create image preparation temporary file");
        if (fstat(file.value, &identity) != 0) {
            const auto error = io_failure("cannot inspect image preparation temporary file");
            unlinkat(directory, name.c_str(), 0);
            throw error;
        }
    }
    ~TemporaryPNG() {
        // Remove only the inode this call created, never a replaced path or the
        // destination. The directory fd also survives parent-path changes.
        if (!published) {
            struct stat current{};
            if (fstatat(directory, name.c_str(), &current, AT_SYMLINK_NOFOLLOW) == 0 &&
                current.st_dev == identity.st_dev && current.st_ino == identity.st_ino)
                unlinkat(directory, name.c_str(), 0);
        }
    }
};
void publish_png(const std::vector<uint8_t> &png, const std::filesystem::path &output) {
    const auto parent = output.parent_path();
    OwnedFD directory(open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (directory.value < 0) throw io_failure("output parent must be an existing directory");
    const std::string filename = output.filename().string();
    struct stat existing{};
    if (fstatat(directory.value, filename.c_str(), &existing, AT_SYMLINK_NOFOLLOW) == 0)
        throw std::invalid_argument("output_path already exists; image preparation never overwrites files");
    if (errno != ENOENT) throw io_failure("cannot inspect image preparation output_path");

    TemporaryPNG temporary(directory.value);
    size_t offset = 0;
    while (offset < png.size()) {
        const ssize_t written = write(temporary.file.value, png.data() + offset, png.size() - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written < 0) throw io_failure("cannot write prepared PNG");
        if (written == 0) throw std::runtime_error("prepared PNG write made no progress");
        offset += static_cast<size_t>(written);
    }
    if (fsync(temporary.file.value) != 0) throw io_failure("cannot flush prepared PNG");
    // The dirfd form of renamex_np(RENAME_EXCL) publishes atomically without
    // replacing a target created after the initial check, including symlinks.
    if (renameatx_np(directory.value, temporary.name.c_str(), directory.value,
                     filename.c_str(), RENAME_EXCL) != 0)
        throw io_failure("cannot publish prepared PNG without overwriting output_path");
    temporary.published = true;
}
}

ReferencePreparation prepare_reference_image(const std::filesystem::path &path, const std::string &preset) {
    validate_path(path, "source_path");
    const auto bounds = bounds_for(preset);
    NSDictionary *source_options = @{(__bridge NSString *)kCGImageSourceShouldCache: @NO};
    auto source = owned(CGImageSourceCreateWithURL(
        (__bridge CFURLRef)[NSURL fileURLWithPath:path_string(path)], (__bridge CFDictionaryRef)source_options));
    check(source != nullptr && CGImageSourceGetCount(source.get()) == 1,
          "cannot read a single-frame original image");
    auto properties_object = owned(CGImageSourceCopyPropertiesAtIndex(source.get(), 0, nullptr));
    check(properties_object != nullptr && CFGetTypeID(properties_object.get()) == CFDictionaryGetTypeID(),
          "cannot read original image dimensions");
    NSDictionary *properties = (__bridge NSDictionary *)properties_object.get();
    int width = pixel_dimension(properties, kCGImagePropertyPixelWidth);
    int height = pixel_dimension(properties, kCGImagePropertyPixelHeight);
    check(int64_t(width) * height <= maximum_pixels,
          "image dimensions exceed the 80000000 pixel import limit");
    id orientation_value = properties[(__bridge NSString *)kCGImagePropertyOrientation];
    const int orientation = [orientation_value isKindOfClass:NSNumber.class] ? [orientation_value intValue] : 1;
    if (orientation >= 5 && orientation <= 8) std::swap(width, height);

    ReferencePreparation result{width, height, width, height, {}};
    if (preset != "original") {
        const double scale = std::min({1.0, double(bounds.first) / width, double(bounds.second) / height});
        result.width = std::max(1, std::min({width, bounds.first, int(std::round(width * scale))}));
        result.height = std::max(1, std::min({height, bounds.second, int(std::round(height * scale))}));
    }
    const bool keeps_original = result.width == width && result.height == height;
    const int maximum = keeps_original ? std::min(128, std::max(width, height)) : std::max(result.width, result.height);
    NSDictionary *thumbnail_options = @{
        (__bridge NSString *)kCGImageSourceCreateThumbnailFromImageAlways: @YES,
        (__bridge NSString *)kCGImageSourceCreateThumbnailWithTransform: @YES,
        (__bridge NSString *)kCGImageSourceThumbnailMaxPixelSize: @(maximum),
        (__bridge NSString *)kCGImageSourceShouldCacheImmediately: @YES
    };
    auto decoded = owned(CGImageSourceCreateThumbnailAtIndex(source.get(), 0, (__bridge CFDictionaryRef)thumbnail_options));
    check(decoded != nullptr, "original image cannot be decoded");
    if (keeps_original) return result;

    auto color = owned(CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
    check(color != nullptr, "cannot create prepared image sRGB color space");
    auto context = owned(CGBitmapContextCreate(nullptr, result.width, result.height, 8,
        size_t(result.width) * 4, color.get(),
        CGBitmapInfo(kCGImageAlphaPremultipliedLast) | kCGBitmapByteOrder32Big));
    check(context != nullptr, "cannot allocate prepared RGBA image bitmap");
    const CGRect rectangle = CGRectMake(0, 0, result.width, result.height);
    CGContextClearRect(context.get(), rectangle);
    CGContextSetInterpolationQuality(context.get(), kCGInterpolationHigh);
    CGContextDrawImage(context.get(), rectangle, decoded.get());
    auto image = owned(CGBitmapContextCreateImage(context.get()));
    check(image != nullptr, "cannot create prepared image");
    NSMutableData *encoded = [NSMutableData data];
    auto destination = owned(CGImageDestinationCreateWithData((__bridge CFMutableDataRef)encoded,
        (__bridge CFStringRef)UTTypePNG.identifier, 1, nullptr));
    check(destination != nullptr, "cannot create prepared PNG encoder");
    NSDictionary *png_options = @{(__bridge NSString *)kCGImagePropertyOrientation: @1};
    CGImageDestinationAddImage(destination.get(), image.get(), (__bridge CFDictionaryRef)png_options);
    check(CGImageDestinationFinalize(destination.get()) && encoded.length > 0, "cannot encode prepared PNG");
    const auto *bytes = static_cast<const uint8_t *>(encoded.bytes);
    result.png.assign(bytes, bytes + encoded.length);
    return result;
}

NSDictionary *reference_preparation_metadata(const ReferencePreparation &value) {
    return @{@"original_width": @(value.original_width), @"original_height": @(value.original_height),
             @"width": @(value.width), @"height": @(value.height), @"changed": @(!value.png.empty())};
}

NSDictionary *prepare_image_file(NSDictionary *input) {
    check([input isKindOfClass:NSDictionary.class], "image preparation input must be an object");
    NSArray *allowed = @[@"schema_version", @"source_path", @"preset", @"output_path"];
    for (id key in input) {
        check([key isKindOfClass:NSString.class] && [allowed containsObject:key], "unknown image preparation field");
    }
    id version = input[@"schema_version"];
    if (version) {
        check([version isKindOfClass:NSNumber.class] &&
                  CFGetTypeID((__bridge CFTypeRef)version) != CFBooleanGetTypeID() && [version doubleValue] == 1,
              "image preparation schema_version must be 1");
    }
    const auto source = input_path(input, @"source_path");
    id preset_value = input[@"preset"];
    check([preset_value isKindOfClass:NSString.class], "image preparation preset must be a string");
    NSString *preset_string = preset_value;
    const char *preset_bytes = preset_string.UTF8String;
    check(preset_bytes != nullptr && std::strlen(preset_bytes) == [preset_string lengthOfBytesUsingEncoding:NSUTF8StringEncoding],
          "image preparation preset must be valid UTF-8 without NUL");
    const std::string preset(preset_bytes);
    (void)bounds_for(preset);
    std::filesystem::path output;
    if (input[@"output_path"]) {
        output = input_path(input, @"output_path");
        check(output.extension() == ".png", "output_path must have a .png extension");
    }
    const auto prepared = prepare_reference_image(source, preset);
    const bool changed = !prepared.png.empty();
    check(!changed || !output.empty(), "output_path is required when image preparation changes dimensions");
    if (changed) publish_png(prepared.png, output);
    NSMutableDictionary *metadata = [reference_preparation_metadata(prepared) mutableCopy];
    metadata[@"schema_version"] = @1;
    metadata[@"source_path"] = path_string(source);
    metadata[@"preset"] = preset_string;
    metadata[@"image_path"] = path_string(changed ? output : source);
    metadata[@"output_created"] = @(changed);
    return metadata;
}
}

#pragma once

#import <Foundation/Foundation.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <string>
#include <unistd.h>
#include <vector>

namespace tc::streaming::metadata {

using MetadataCheck = void (*)(bool, const std::string &);

inline NSDictionary *json_dictionary(NSData *data, const std::string &invalid,
                                     MetadataCheck require) {
    NSError *failure = nil;
    id parsed = [NSJSONSerialization JSONObjectWithData:data
                                                options:0
                                                  error:&failure];
    require([parsed isKindOfClass:NSDictionary.class],
            failure ? failure.localizedDescription.UTF8String : invalid);
    return (NSDictionary *)parsed;
}

inline void pread_exact(int descriptor, void *destination, size_t bytes,
                        uint64_t offset, const char *truncated, MetadataCheck require) {
    auto *output = static_cast<unsigned char *>(destination);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t count = ::pread(descriptor, output + done, bytes - done,
                                      static_cast<off_t>(offset + done));
        if (count < 0 && errno == EINTR)
            continue;
        require(count > 0, truncated);
        done += static_cast<size_t>(count);
    }
}

inline NSDictionary *read_json_dictionary(int descriptor, uint64_t bytes, uint64_t offset,
                                          const char *truncated, const std::string &invalid,
                                          MetadataCheck require) {
    std::vector<unsigned char> encoded(static_cast<size_t>(bytes));
    pread_exact(descriptor, encoded.data(), encoded.size(), offset, truncated, require);
    NSData *data = [NSData dataWithBytes:encoded.data() length:encoded.size()];
    return json_dictionary(data, invalid, require);
}

struct SafetensorsHeader {
    NSDictionary *root;
    uint64_t payload_offset;
};

inline SafetensorsHeader read_safetensors_header(int descriptor, uint64_t file_bytes,
                                                uint64_t max_header_bytes, const char *truncated,
                                                const char *invalid, MetadataCheck require) {
    std::array<unsigned char, 8> prefix{};
    pread_exact(descriptor, prefix.data(), prefix.size(), 0, truncated, require);
    uint64_t header_bytes = 0;
    for (size_t index = 0; index < prefix.size(); ++index)
        header_bytes |= static_cast<uint64_t>(prefix[index]) << (8 * index);
    require(header_bytes && header_bytes <= max_header_bytes &&
            header_bytes <= file_bytes - prefix.size(),
            "invalid safetensors header length");
    return {read_json_dictionary(descriptor, header_bytes, prefix.size(),
                                 truncated, invalid, require),
            prefix.size() + header_bytes};
}

inline uint64_t json_unsigned_integer(id value, const char *label, MetadataCheck require) {
    constexpr uint64_t max_exact_json_integer = 1ull << 53;
    require([value isKindOfClass:NSNumber.class] &&
            CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            std::string("non-integer ") + label);
    const double number = [value doubleValue];
    require(std::isfinite(number) && number >= 0 &&
            number == std::floor(number) &&
            number <= static_cast<double>(max_exact_json_integer),
            std::string("invalid ") + label);
    return static_cast<uint64_t>(number);
}

} // namespace tc::streaming::metadata

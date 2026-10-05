#pragma once

#include "../../core/common.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <sys/stat.h>
#include <unistd.h>

namespace tc::detail {
// Caller owns the descriptor. pread leaves its offset unchanged, and the
// bounded heap buffer is safe on libdispatch's small worker stacks.
inline std::string sha256_fd(int fd, const std::atomic<bool> &cancelled) {
    auto check = [&] { if (cancelled.load(std::memory_order_relaxed)) throw Cancelled(); };
    check();
    require(fd >= 0, "invalid SHA-256 source descriptor");
    struct stat initial{};
    require(fstat(fd, &initial) == 0 && S_ISREG(initial.st_mode) && initial.st_size >= 0,
            "SHA-256 source is not a regular file");
    CC_SHA256_CTX context;
    require(CC_SHA256_Init(&context) == 1, "cannot initialize SHA-256");
    std::vector<unsigned char> buffer(1 << 20);
    off_t offset = 0;
    // A growing file must not extend a pending owner's join indefinitely.
    // Read exactly the initial length and reject a changed generation below.
    while (offset < initial.st_size) {
        check();
        const auto length = std::min(static_cast<uint64_t>(initial.st_size - offset),
                                     static_cast<uint64_t>(buffer.size()));
        const auto count = pread(fd, buffer.data(), static_cast<size_t>(length), offset);
        if (count < 0 && errno == EINTR) continue;
        require(count > 0, "cannot read complete source descriptor for SHA-256");
        require(CC_SHA256_Update(&context, buffer.data(), static_cast<CC_LONG>(count)) == 1,
                "cannot update SHA-256");
        offset += count;
    }
    check();
    struct stat after{};
    require(fstat(fd, &after) == 0 && initial.st_dev == after.st_dev && initial.st_ino == after.st_ino &&
                initial.st_size == after.st_size && S_ISREG(after.st_mode) &&
                initial.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec &&
                initial.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec &&
                initial.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec &&
                initial.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec,
            "SHA-256 source generation changed while reading");
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest;
    require(CC_SHA256_Final(digest.data(), &context) == 1, "cannot finalize SHA-256");
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(digest.size() * 2, '0');
    for (size_t i = 0; i < digest.size(); ++i) {
        result[i * 2] = hex[digest[i] >> 4]; result[i * 2 + 1] = hex[digest[i] & 15];
    }
    return result;
}
} // namespace tc::detail

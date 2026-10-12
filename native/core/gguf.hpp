#pragma once

#include "gguf_directory.hpp"
#include <filesystem>
#include <fcntl.h>
#include <sys/stat.h>

namespace tc {
// The existing MLX execution gate remains narrow. The independent decoder's
// additional types do not grant MLX execution capability.
inline void validate_native_gguf(const std::filesystem::path &path) {
    struct File { int fd; ~File() { if (fd >= 0) ::close(fd); } };
    File file{::open(path.c_str(), O_RDONLY | O_CLOEXEC)};
    if (file.fd < 0) throw std::runtime_error("cannot open GGUF: " + path.string());
    struct stat before{}, after{};
    gguf::check(::fstat(file.fd, &before) == 0 && S_ISREG(before.st_mode) && before.st_size >= 0,
                "invalid source file");
    (void)gguf::read_directory(file.fd, uint64_t(before.st_size), {}, true);
    gguf::check(::fstat(file.fd, &after) == 0 && before.st_dev == after.st_dev &&
        before.st_ino == after.st_ino && before.st_size == after.st_size &&
        before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec &&
        before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec &&
        before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec &&
        before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec, "source changed during directory read");
}
} // namespace tc

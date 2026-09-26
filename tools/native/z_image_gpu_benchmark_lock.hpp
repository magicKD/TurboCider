#pragma once

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <unistd.h>

// Shared with benchmark_z_image_metal.py. Coordinates our probes and model
// benchmarks; unrelated applications still need to be kept off the GPU.
class ZImageGpuBenchmarkLock {
    int fd_ = -1;
public:
    ZImageGpuBenchmarkLock() {
        fd_ = open("/tmp/turbocider-z-image-gpu-benchmark.lock", O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (fd_ < 0) throw std::runtime_error("cannot open Z-Image GPU benchmark lock");
        while (flock(fd_, LOCK_EX) != 0) {
            if (errno == EINTR) continue;
            const std::string error = std::strerror(errno);
            close(fd_); fd_ = -1;
            throw std::runtime_error("cannot lock Z-Image GPU benchmark: " + error);
        }
    }
    ~ZImageGpuBenchmarkLock() { if (fd_ >= 0) close(fd_); }
    ZImageGpuBenchmarkLock(const ZImageGpuBenchmarkLock&) = delete;
    ZImageGpuBenchmarkLock& operator=(const ZImageGpuBenchmarkLock&) = delete;
};

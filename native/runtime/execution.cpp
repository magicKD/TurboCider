#include "execution.hpp"
#include "common.hpp"
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
namespace tc {
std::mutex &execution_mutex() {
    static std::mutex mutex;
    return mutex;
}
DeviceLease::DeviceLease() {
    auto path = std::string("/private/tmp/turbocider-gpu-") + std::to_string(geteuid()) + ".lock";
    fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0)
        throw std::runtime_error("cannot open GPU coordination lock");
    struct stat info {};
    if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_uid != geteuid() ||
        flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        fd = -1;
        throw std::runtime_error(
            "GPU is busy in another TurboCider process; submit through the shared service queue");
    }
}
DeviceLease::~DeviceLease() {
    if (fd >= 0) {
        flock(fd, LOCK_UN);
        close(fd);
    }
}
} // namespace tc

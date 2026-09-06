#pragma once
#include <mutex>
namespace tc {
std::mutex &execution_mutex();
class DeviceLease {
    int fd = -1;

  public:
    DeviceLease();
    ~DeviceLease();
    DeviceLease(const DeviceLease &) = delete;
    DeviceLease &operator=(const DeviceLease &) = delete;
};
} // namespace tc

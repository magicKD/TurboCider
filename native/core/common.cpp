#include "common.hpp"
namespace tc {
void require(bool b, const std::string &s) {
    if (!b)
        throw std::invalid_argument(s);
}
void checkpoint(std::atomic<bool> &c) {
    if (c.load())
        throw Cancelled();
}
} // namespace tc

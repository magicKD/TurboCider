#include "models/z_image/suffix_materialization.hpp"
#include "core/common.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace tc::z_image;
namespace {
void check(bool value) { if (!value) throw std::runtime_error("suffix oracle failed"); }
struct File {
    int fd;
    File() { char name[] = "/tmp/tc-suffix-test-XXXXXX"; fd = mkstemp(name); check(fd >= 0); check(!unlink(name)); }
    ~File() { close(fd); }
};
std::string fault;
unsigned reads = 0, writes = 0;
std::atomic<bool> cancelled{false};
void reset(std::string mode = {}) { fault = std::move(mode); reads = writes = 0; cancelled = false; }
template<class F> void rejects(F f, const std::string &message) {
    try { f(); } catch (const std::exception &e) {
        check(std::string(e.what()).find(message) != std::string::npos); return;
    }
    throw std::runtime_error("expected suffix rejection: " + message);
}
}
// Only the separately compiled test object renames these syscalls. Production
// uses ordinary pread/pwrite; no fault hook or environment override is shipped.
extern "C" ssize_t tc_test_pread(int fd, void *data, size_t size, off_t offset) {
    ++reads;
    if (fault == "short" && reads == 1) { errno = EINTR; return -1; }
    if (fault == "eof" && reads == 2) return 0;
    if (fault == "cancel") cancelled = true;
    if (fault == "short" || fault == "eof") size = std::min(size, size_t(7));
    return pread(fd, data, size, offset);
}
extern "C" ssize_t tc_test_pwrite(int fd, const void *data, size_t size, off_t offset) {
    ++writes;
    if (fault == "short" && writes == 1) { errno = EINTR; return -1; }
    if (fault == "disk" && writes == 2) { errno = ENOSPC; return -1; }
    if (fault == "zero") return 0;
    if (fault == "cancel_write") cancelled = true;
    if (fault == "short" || fault == "disk") size = std::min(size, size_t(5));
    return pwrite(fd, data, size, offset);
}

int main() {
    try {
        // Byte oracle is independent of geometry/packing code. Cover zero,
        // one and last-channel boundaries, BF16/I8 and multiple scratch batches.
        for (auto element : {1u, 2u}) for (auto a : {0u, 1u, 6u}) {
            File src, dst;
            constexpr size_t h = 5, m = 7, offset = 11, output_offset = 13;
            std::vector<unsigned char> input(offset + h * m * element);
            for (size_t i = 0; i < input.size(); ++i) input[i] = (i * 37 + 3) % 251;
            check(write(src.fd, input.data(), input.size()) == ssize_t(input.size()));
            auto g = suffix_geometry(h, m, a, element);
            check(g.up_skip_bytes == a * h * element);
            check(g.up_suffix_bytes == (m - a) * h * element);
            // Derived size fields are never trusted as an allocation authority.
            g.down_suffix_bytes = UINT64_MAX;
            reset("short");
            SuffixPackMetrics metrics;
            pack_suffix_rows(src.fd, offset, dst.fd, output_offset, g, cancelled, metrics);
            std::vector<unsigned char> output(h * (m - a) * element);
            check(pread(dst.fd, output.data(), output.size(), output_offset) == ssize_t(output.size()));
            size_t cursor = 0;
            for (size_t row = 0; row < h; ++row)
                for (size_t col = a; col < m; ++col)
                    for (size_t byte = 0; byte < element; ++byte)
                        check(output[cursor++] == input[offset + (row * m + col) * element + byte]);
            check(metrics.read_bytes == h * m * element && metrics.write_bytes == output.size());
        }
        {
            File src, dst;
            auto g = suffix_geometry(1200, 2048, 1000, 2);
            std::vector<unsigned char> data(g.down_source_bytes);
            for (size_t i = 0; i < data.size(); ++i) data[i] = (i * 13 + i / 4096) % 251;
            check(write(src.fd, data.data(), data.size()) == ssize_t(data.size()));
            reset(); SuffixPackMetrics metrics;
            pack_suffix_rows(src.fd, 0, dst.fd, 0, g, cancelled, metrics);
            check(reads == 2 && writes == 2);
            std::vector<unsigned char> output(g.down_suffix_bytes);
            check(pread(dst.fd, output.data(), output.size(), 0) == ssize_t(output.size()));
            for (size_t row = 0; row < 1200; ++row)
                for (size_t col = 1000; col < 2048; ++col)
                    for (size_t byte = 0; byte < 2; ++byte)
                        check(output[(row * 1048 + col - 1000) * 2 + byte] == data[(row * 2048 + col) * 2 + byte]);
        }
        rejects([] { suffix_geometry(0, 7, 1, 2); }, "geometry");
        rejects([] { suffix_geometry(5, 7, 7, 2); }, "geometry");
        rejects([] { suffix_geometry(5, 7, 1, 4); }, "geometry");
        rejects([] { suffix_geometry(UINT64_MAX, 7, 1, 2); }, "overflow");
        rejects([] { suffix_geometry(1, 1ull << 23, 1, 2); }, "packing limit");
        File src, dst;
        std::vector<char> bytes(70, 42);
        check(write(src.fd, bytes.data(), bytes.size()) == ssize_t(bytes.size()));
        const auto g = suffix_geometry(5, 7, 1, 2);
        for (auto mode : {"eof", "disk", "zero", "cancel", "cancel_write"}) {
            reset(mode); SuffixPackMetrics metrics;
            rejects([&] { pack_suffix_rows(src.fd, 0, dst.fd, 0, g, cancelled, metrics); },
                    fault == "eof" ? "truncated" : fault.starts_with("cancel") ? "generation cancelled" : "write");
            if (fault == "eof") check(metrics.read_bytes == 7 && metrics.write_bytes == 0);
            if (fault == "disk") check(metrics.read_bytes == 70 && metrics.write_bytes == 5);
            if (fault == "cancel") check(writes == 0);
            if (fault == "cancel_write") check(metrics.write_bytes == 60);
        }
        reset(); SuffixPackMetrics metrics;
        rejects([&] { pack_suffix_rows(src.fd, 1, dst.fd, 0, g, cancelled, metrics); }, "truncated");
        rejects([&] { pack_suffix_rows(src.fd, UINT64_MAX, dst.fd, 0, g, cancelled, metrics); }, "overflow");
        rejects([&] { pack_suffix_rows(src.fd, 0, src.fd, 0, g, cancelled, metrics); }, "distinct");
        rejects([&] { pack_suffix_rows(src.fd, 0, dst.fd, UINT64_MAX, g, cancelled, metrics); }, "overflow");
        metrics.read_bytes = UINT64_MAX;
        rejects([&] { pack_suffix_rows(src.fd, 0, dst.fd, 0, g, cancelled, metrics); }, "accounting overflow");
        metrics = {};
        check(reads == 0 && writes == 0);
        cancelled = true;
        rejects([&] { pack_suffix_rows(src.fd, 0, dst.fd, 0, g, cancelled, metrics); }, "generation cancelled");
        try {
            pack_suffix_rows(src.fd, 0, dst.fd, 0, g, cancelled, metrics);
            throw std::logic_error("cancellation type lost");
        } catch (const tc::Cancelled &) { /* preserve native cancellation mapping */ }
        reset();
        pack_suffix_rows(src.fd, 0, dst.fd, 0, g, cancelled, metrics);
        check(metrics.read_bytes == 70 && metrics.write_bytes == 60);
        std::cout << "PASS suffix byte oracle, bounds, batching, EINTR/short I/O, EOF, ENOSPC, cancellation and retry\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}

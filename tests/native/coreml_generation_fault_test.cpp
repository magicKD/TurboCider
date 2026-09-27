#include "models/z_image/coreml_generation.hpp"
#include "core/common.hpp"
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
using tc::z_image::CoreMLGeneration;
namespace {
std::string fault;
unsigned reads = 0, writes = 0;
std::atomic<bool> stop{false};
fs::path source;
void check(bool condition) { if (!condition) throw std::runtime_error("generation fault assertion: " + fault); }
size_t fd_count() { return size_t(std::distance(fs::directory_iterator("/dev/fd"), fs::directory_iterator())); }
std::string bytes(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}
void put(const fs::path &path, const std::string &data) { std::ofstream(path, std::ios::binary) << data; }
}
// Symbol replacement applies only to coreml_generation.cpp's test object.
// SourceLease hashing and the production implementation have no test switches.
extern "C" ssize_t tc_generation_pread(int fd, void *data, size_t size, off_t offset) {
    ++reads;
    if (fault == "short" && reads == 1) { errno = EINTR; return -1; }
    if (fault == "eof" && reads == 2) return 0;
    if (fault == "read_error" && reads == 2) { errno = EIO; return -1; }
    if (fault == "cancel_read") stop = true;
    if (fault == "replace_source" && reads == 1) {
        fs::rename(source / "weights", source / "original");
        put(source / "weights", "replacement");
    }
    if (fault == "mutate_source" && reads == 1) {
        std::fstream file(source / "weights", std::ios::binary | std::ios::in | std::ios::out);
        file.put('!');
    }
    if (fault == "short" || fault == "eof" || fault == "read_error") size = std::min(size, size_t(16381));
    return pread(fd, data, size, offset);
}
extern "C" ssize_t tc_generation_write(int fd, const void *data, size_t size) {
    ++writes;
    if (fault == "short" && writes == 1) { errno = EINTR; return -1; }
    if (fault == "disk" && writes == 2) { errno = ENOSPC; return -1; }
    if (fault == "zero") return 0;
    if (fault == "cancel_write") stop = true;
    if (fault == "short" || fault == "disk") size = std::min(size, size_t(8191));
    if (fault == "corrupt" && writes == 1) {
        std::string changed(static_cast<const char *>(data), size);
        changed[0] ^= 1;
        return write(fd, changed.data(), changed.size());
    }
    return write(fd, data, size);
}
extern "C" int tc_generation_fchmod(int fd, mode_t mode) {
    if (fault == "seal") { errno = EPERM; return -1; }
    if (fault == "cancel_seal") stop = true;
    return fchmod(fd, mode);
}
int main(int argc, char **argv) {
    check(argc == 2);
    fs::path root(argv[1]); source = root / "source";
    auto managed = root / "managed";
    fs::create_directories(source); fs::create_directory(managed);
    std::string expected(2 * 1024 * 1024 + 23, '\0');
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = char((i * 37 + i / 4096) % 251);
    for (const auto &mode : {"short", "eof", "read_error", "disk", "zero", "cancel_read", "cancel_write", "cancel_seal", "seal", "corrupt", "replace_source", "mutate_source"}) {
        fs::remove_all(source); fs::create_directory(source); put(source / "weights", expected);
        fault = mode; reads = writes = 0; stop = false;
        auto before = fd_count();
        bool threw = false, typed_cancel = false, returned = false;
        std::string message;
        try {
            auto generation = CoreMLGeneration::import_tree(source, managed, &stop);
            returned = true;
            generation->revalidate();
            check(generation->copied_bytes() == expected.size());
            check(bytes(generation->root() / "weights") == expected);
        } catch (const tc::Cancelled &e) { threw = typed_cancel = true; message = e.what(); }
          catch (const std::exception &e) { threw = true; message = e.what(); }
        check(returned == (fault == "short"));
        check(threw == (fault != "short"));
        check(typed_cancel == fault.starts_with("cancel_"));
        check(reads > 0);
        if (fault == "disk" || fault == "zero" || fault == "corrupt" || fault == "seal") check(writes > 0);
        if (fault == "corrupt") check(message.find("content digest mismatch") != std::string::npos);
        check(fd_count() == before); check(fs::is_empty(managed));
        // A clean retry must independently reproduce the original bytes.
        fault.clear(); stop = false;
        fs::remove_all(source); fs::create_directory(source); put(source / "weights", expected);
        auto retry = CoreMLGeneration::import_tree(source, managed, &stop);
        check(bytes(retry->root() / "weights") == expected); retry.reset();
        check(fd_count() == before); check(fs::is_empty(managed));
        std::cout << mode << ": pass, no leaked fd/directory, retry pass\n";
    }
}

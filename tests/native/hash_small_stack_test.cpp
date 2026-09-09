#include "platform/apple/platform.hpp"
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <unistd.h>

struct HashCase {
    std::filesystem::path file;
    std::string digest;
    std::string error;
};

static void *hash_on_worker(void *opaque) {
    auto &test = *static_cast<HashCase *>(opaque);
    try { test.digest = tc::sha256_file(test.file); }
    catch (const std::exception &error) { test.error = error.what(); }
    return nullptr;
}

int main() {
    char temporary[] = "/tmp/turbocider-hash-XXXXXX";
    int fd = mkstemp(temporary);
    if (fd < 0) return 1;
    close(fd);
    { std::ofstream stream(temporary, std::ios::binary); stream << "abc"; }
    HashCase test{temporary, {}, {}};
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    if (pthread_attr_setstacksize(&attributes, 256 * 1024) != 0) return 1;
    pthread_t thread;
    int status = pthread_create(&thread, &attributes, hash_on_worker, &test);
    pthread_attr_destroy(&attributes);
    if (status == 0) status = pthread_join(thread, nullptr);
    std::filesystem::remove(temporary);
    if (status || !test.error.empty() || test.digest !=
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
        std::cerr << "small-stack SHA-256 failed: " << test.error << '\n';
        return 1;
    }
    std::cout << "PASS: SHA-256 on a 256 KiB worker stack\n";
}

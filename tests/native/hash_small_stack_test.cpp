#include "platform/apple/platform.hpp"
#include <iostream>
#include <pthread.h>

struct HashCase {
    std::filesystem::path file;
    std::string digest;
    std::string provenance_digest;
    std::string error;
};

static void *hash_on_worker(void *opaque) {
    auto &test = *static_cast<HashCase *>(opaque);
    try {
        test.digest = tc::sha256_file(test.file);
        test.provenance_digest = tc::sha256_file_or_empty(test.file);
    } catch (const std::exception &error) {
        test.error = error.what();
    }
    return nullptr;
}

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    HashCase test{argv[1], {}, {}, {}};
    pthread_attr_t attributes;
    if (pthread_attr_init(&attributes) != 0) return 1;
    if (pthread_attr_setstacksize(&attributes, 256 * 1024) != 0) {
        pthread_attr_destroy(&attributes);
        return 1;
    }
    pthread_t thread;
    int status = pthread_create(&thread, &attributes, hash_on_worker, &test);
    pthread_attr_destroy(&attributes);
    if (status == 0) status = pthread_join(thread, nullptr);
    if (status || !test.error.empty() || test.digest != argv[2] ||
        test.provenance_digest != argv[2]) {
        std::cerr << "small-stack SHA-256 failed: " << test.error << '\n';
        return 1;
    }
    const auto missing = test.file.string() + ".missing";
    if (!tc::sha256_file_or_empty(missing).empty()) {
        std::cerr << "SHA-256 returned a digest for a missing file\n";
        return 1;
    }
    try {
        tc::sha256_file(missing);
        std::cerr << "SHA-256 accepted a missing file\n";
        return 1;
    } catch (const std::invalid_argument &error) {
        if (error.what() != "cannot open file for SHA-256: " + missing) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    std::cout << "PASS: SHA-256 on a 256 KiB worker stack\n";
}

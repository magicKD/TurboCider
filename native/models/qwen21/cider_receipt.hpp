#pragma once
#include <cstdint>
#include <string>
namespace tc::qwen21 {
struct CiderCacheReceipt {
    std::string directory, source_sha256, recipe;
    uint64_t hits = 0, created = 0, bytes_read = 0, bytes_written = 0;
    double read_seconds = 0, prepare_seconds = 0;
};
}

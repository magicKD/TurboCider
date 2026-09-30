#include "gguf.hpp"
#include "gguf_decode.hpp"
#include <iostream>
#include <span>
#include <string>
#include <sys/stat.h>

namespace {
thread_local std::string last_error;
tc::gguf::Directory directory(const char *path, const char *limit = nullptr) {
    struct File { int fd; ~File() { if (fd >= 0) ::close(fd); } };
    File file{::open(path, O_RDONLY | O_CLOEXEC)};
    struct stat status{};
    tc::gguf::check(file.fd >= 0 && ::fstat(file.fd, &status) == 0 && status.st_size >= 0,
                    "cannot open fixture");
    tc::gguf::ParseLimits limits;
    if (limit) {
        const std::string setting(limit);
        if (setting == "retained") limits.retained_bytes = 1;
        else if (setting == "directory") limits.directory_bytes = 24;
        else if (setting == "string") limits.string_bytes = 1;
        else if (setting == "count") limits.tensors = 1;
        else if (setting == "array") limits.array_elements = 1;
        else throw std::invalid_argument("unknown test limit");
    }
    return tc::gguf::read_directory(file.fd, uint64_t(status.st_size), limits);
}
}

extern "C" const char *tc_gguf_test_error() { return last_error.c_str(); }
extern "C" int tc_gguf_test_decode(uint32_t type, const void *source, uint64_t source_bytes,
        uint64_t source_rows, uint64_t source_columns, uint64_t row_begin, uint64_t rows,
        uint64_t column_begin, uint64_t columns, uint32_t dtype, void *target, uint64_t target_bytes,
        uint64_t row_stride, uint64_t column_stride, const uint64_t *gather, uint64_t gather_count,
        int cancel, uint64_t *stats) {
    try {
        const tc::gguf::PackedMatrix packed{{static_cast<const std::byte *>(source), size_t(source_bytes)},
                                           type, source_rows, source_columns};
        const tc::gguf::DecodeTarget output{{static_cast<std::byte *>(target), size_t(target_bytes)},
                                            tc::gguf::DecodeDType(dtype), row_stride, column_stride};
        std::atomic<bool> cancelled{bool(cancel)};
        const auto receipt = gather
            ? tc::gguf::decode_cpu_gather(packed, {gather, size_t(gather_count)}, column_begin, columns, output, &cancelled)
            : tc::gguf::decode_cpu_into(packed, {row_begin, rows, column_begin, columns}, output, &cancelled);
        stats[0] = receipt.source_bytes_processed; stats[1] = receipt.elements_written;
        stats[2] = receipt.bytes_written; stats[3] = receipt.scratch_bytes;
        last_error.clear(); return 0;
    } catch (const std::exception &error) { last_error = error.what(); return 1; }
}
extern "C" int tc_gguf_test_round(uint32_t dtype, float value, uint16_t *result) {
    try {
        *result = dtype == 1 ? tc::gguf::float_to_fp16_rne(value) : tc::gguf::float_to_bf16_rne(value);
        last_error.clear(); return 0;
    } catch (const std::exception &error) { last_error = error.what(); return 1; }
}
extern "C" float tc_gguf_test_half(uint16_t value) { return tc::gguf::fp16_to_float(value); }

int main(int argc, char **argv) {
    try {
        if (argc < 3) return 2;
        const std::string mode(argv[1]);
        if (mode == "split") {
            std::vector<tc::gguf::Directory> shards;
            for (int i = 2; i < argc; ++i) shards.push_back(directory(argv[i]));
            tc::gguf::validate_split_set(shards); std::cout << "split_ok\n"; return 0;
        }
        if (mode == "legacy") { tc::validate_native_gguf(argv[2]); return 0; }
        const auto result = directory(argv[2], mode == "limit" && argc == 4 ? argv[3] : nullptr);
        std::cout << "directory " << result.tensors.size() << ' ' << result.data_offset
                  << ' ' << result.retained_upper_bytes << '\n';
        if (mode == "tensor" && argc == 4) {
            const auto &tensor = result.tensor(argv[3]);
            std::cout << "tensor " << tensor.type << ' ' << tensor.rows() << ' ' << tensor.columns()
                      << ' ' << tensor.file_offset << ' ' << tensor.bytes << '\n';
        }
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

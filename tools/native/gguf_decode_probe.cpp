#include "../../native/core/gguf_decode.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <chrono>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

namespace {
std::string digest(std::span<const std::byte> bytes) {
    tc::gguf::check(bytes.size() <= UINT32_MAX, "digest input too large");
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> output{};
    CC_SHA256(bytes.data(), CC_LONG(bytes.size()), output.data());
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (auto byte : output) { result += hex[byte >> 4]; result += hex[byte & 15]; }
    return result;
}
void json_string(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::cout << '"';
    for (unsigned char byte : value) {
        if (byte == '"' || byte == '\\') std::cout << '\\' << char(byte);
        else if (byte < 32) std::cout << "\\u00" << hex[byte >> 4] << hex[byte & 15];
        else std::cout << char(byte);
    }
    std::cout << '"';
}
void samples(const std::vector<double> &values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
    std::cout << ']';
}
}

int main(int argc, char **argv) {
    try {
        if (argc != 5) {
            std::cerr << "usage: gguf-decode-probe checkpoint.gguf tensor iterations max-buffer-bytes\n";
            return 2;
        }
        const int iterations = std::stoi(argv[3]);
        const uint64_t maximum = std::stoull(argv[4]);
        tc::gguf::check(iterations >= 8 && iterations <= 100, "invalid benchmark iterations");
        tc::gguf::check(maximum && maximum <= (1ull << 30), "invalid buffer budget");
        struct File { int fd; ~File() { if (fd >= 0) ::close(fd); } };
        File file{::open(argv[1], O_RDONLY | O_CLOEXEC)};
        struct stat status{};
        tc::gguf::check(file.fd >= 0 && ::fstat(file.fd, &status) == 0 && S_ISREG(status.st_mode), "cannot open model");
        const auto directory = tc::gguf::read_directory(file.fd, uint64_t(status.st_size));
        const auto &tensor = directory.tensor(argv[2]);
        const uint64_t target_bytes = tc::gguf::checked_mul(tensor.elements, 2);
        const uint64_t required = tc::gguf::checked_add(tensor.bytes, tc::gguf::checked_mul(target_bytes, 2));
        tc::gguf::check(required <= maximum, "benchmark buffer budget insufficient");
        std::vector<std::byte> packed(size_t(tensor.bytes));
        std::vector<std::byte> scalar(static_cast<size_t>(target_bytes));
        std::vector<std::byte> simd(static_cast<size_t>(target_bytes));
        uint64_t read_bytes = 0;
        while (read_bytes < tensor.bytes) {
            const uint64_t amount = std::min<uint64_t>(tensor.bytes - read_bytes, 1ull << 20);
            const ssize_t n = ::pread(file.fd, packed.data() + read_bytes, size_t(amount), off_t(tensor.file_offset + read_bytes));
            if (n < 0 && errno == EINTR) continue;
            tc::gguf::check(n > 0, "short read or source changed"); read_bytes += uint64_t(n);
        }
        struct stat after{};
        tc::gguf::check(::fstat(file.fd, &after) == 0 && status.st_size == after.st_size &&
            status.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec &&
            status.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec &&
            status.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec &&
            status.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec, "model changed while reading");
        const tc::gguf::PackedMatrix source{packed, tensor.type, tensor.rows(), tensor.columns()};
        const tc::gguf::DecodeSlice slice{0, tensor.rows(), 0, tensor.columns()};
        tc::gguf::DecodeReceipt receipt;
        auto run = [&](bool use_simd) {
            auto &output = use_simd ? simd : scalar;
            const tc::gguf::DecodeTarget target{output, tc::gguf::DecodeDType::bf16, tensor.columns() * 2, 2};
            const auto start = std::chrono::steady_clock::now();
            receipt = tc::gguf::decode_cpu_into(source, slice, target, nullptr, {use_simd});
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        };
        for (int i = 0; i < 3; ++i) { run(false); run(true); }
        std::vector<double> a, b;
        a.reserve(size_t(iterations)); b.reserve(size_t(iterations));
        for (int i = 0; i < iterations; ++i) {
            if (i % 4 < 2) { a.push_back(run(false)); b.push_back(run(true)); }
            else { b.push_back(run(true)); a.push_back(run(false)); }
        }
        run(true);
        tc::gguf::check(scalar == simd, "SIMD/scalar full-tensor output mismatch");
        std::cout << std::setprecision(10) << "{\"kind\":\"real_tensor_cpu_decode\",\"tensor\":";
        json_string(tensor.name);
        std::cout << ",\"type\":"; json_string(tc::gguf::type_info(tensor.type).name);
        std::cout << ",\"rows\":" << tensor.rows() << ",\"columns\":" << tensor.columns()
                  << ",\"packed_bytes\":" << tensor.bytes << ",\"one_target_bytes\":" << target_bytes
                  << ",\"managed_buffer_capacity\":" << required << ",\"scratch_upper\":1024"
                  << ",\"source_identity_kind\":\"tensor_payload_sha256\",\"source_sha256\":";
        json_string(digest(packed)); std::cout << ",\"output_sha256\":"; json_string(digest(simd));
        std::cout << ",\"exact\":true,\"simd_blocks\":" << receipt.simd_blocks
                  << ",\"warmups_per_arm\":3,\"scalar_ms\":";
        samples(a); std::cout << ",\"simd_ms\":"; samples(b);
        std::cout << ",\"excludes\":[\"parse/read\",\"allocation\",\"GEMM\",\"full_request\",\"whole_process_memory\",\"ANE\",\"M5\"]}\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

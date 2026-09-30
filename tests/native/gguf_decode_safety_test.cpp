#include "gguf_decode.hpp"
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>

namespace {
thread_local bool forbid_allocation = false;
void expect(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template <class Function> void rejected(Function function) {
    bool failed = false;
    try { function(); } catch (const tc::gguf::DecodeError &) { failed = true; }
    expect(failed, "unsafe operation was accepted");
}
}
void *operator new(std::size_t bytes) {
    if (forbid_allocation) throw std::bad_alloc();
    if (void *pointer = std::malloc(bytes ? bytes : 1)) return pointer;
    throw std::bad_alloc();
}
void operator delete(void *pointer) noexcept { std::free(pointer); }

int main() {
    try {
        using namespace tc::gguf;
        alignas(16) std::array<std::byte, 34 * 8> source{};
        for (size_t block = 0; block < 8; ++block) {
            source[block * 34 + 1] = std::byte(0x3c); // FP16 1.0
            for (size_t i = 0; i < 32; ++i) source[block * 34 + 2 + i] = std::byte((i * 17) & 255);
        }
        std::array<std::byte, 512> scalar{}, simd{};
        const PackedMatrix matrix{source, 8, 2, 128};
        const DecodeSlice slice{0, 2, 0, 128};
        forbid_allocation = true;
        const auto scalar_receipt = decode_cpu_into(matrix, slice, {scalar, DecodeDType::bf16, 256, 2}, nullptr, {false});
        const auto simd_receipt = decode_cpu_into(matrix, slice, {simd, DecodeDType::bf16, 256, 2});
        forbid_allocation = false;
        expect(scalar == simd, "scalar/SIMD mismatch");
        expect(scalar_receipt.scratch_bytes == 1024 && simd_receipt.scratch_bytes == 1024, "wrong scratch upper");
        std::array<std::byte, 515> unaligned{};
        decode_cpu_into(matrix, slice, {{unaligned.data() + 1, 513}, DecodeDType::bf16, 257, 2});
        for (size_t row = 0; row < 2; ++row)
            expect(std::memcmp(unaligned.data() + 1 + row * 257, scalar.data() + row * 256, 256) == 0,
                   "unaligned target changed numerics");
        rejected([&] { decode_cpu_into(matrix, slice, {{source.data(), source.size()}, DecodeDType::bf16, 256, 2}); });
        alignas(uint64_t) std::array<std::byte, 64> target{};
        const std::array<uint64_t, 2> indices{1, 0};
        std::memcpy(target.data(), indices.data(), sizeof(indices));
        forbid_allocation = true;
        rejected([&] { decode_cpu_gather(matrix,
            {reinterpret_cast<const uint64_t *>(target.data()), 2}, 0, 2,
            {target, DecodeDType::f32, 16, 4}); });
        rejected([&] { decode_cpu_into({source, 8, UINT64_MAX, 128}, slice,
                                       {scalar, DecodeDType::bf16, 256, 2}); });
        std::atomic<bool> cancel{true};
        rejected([&] { decode_cpu_into(matrix, slice, {scalar, DecodeDType::bf16, 256, 2}, &cancel); });
        forbid_allocation = false;
        expect(scalar == simd, "pre-cancel wrote target");
        std::cout << "PASS GGUF decoder safety: allocation-free, alias, unaligned, overflow, cancellation\n";
        return 0;
    } catch (const std::exception &error) {
        forbid_allocation = false;
        std::cerr << error.what() << '\n'; return 1;
    }
}

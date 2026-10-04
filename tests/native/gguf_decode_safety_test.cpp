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
        // Every quantized type, all target dtypes, extreme/subnormal/signed-zero
        // scales. Strided/unaligned targets must safely retain the scalar path.
        for (const auto &type : types) {
            if (type.elements == 1) continue;
            const uint64_t columns = 4 * type.elements, elements = 2 * columns;
            alignas(16) std::array<std::byte, 1680> packed{};
            const size_t bytes = size_t(type.bytes) * 8;
            for (float scale : {0.f, -0.f, 0x1p-24f, .03125f, -.03125f, 65504.f, -65504.f}) {
                for (size_t b = 0; b < 8; ++b) {
                    auto *p = packed.data() + b * type.bytes;
                    for (uint32_t i = 0; i < type.bytes; ++i) p[i] = std::byte((i * 73 + b * 17 + 19) & 255);
                    const auto bits = float_to_fp16_rne(scale);
                    std::memcpy(p + (type.id == 14 ? 208 : 0), &bits, 2);
                    if (type.id == 3 || type.id == 7 || type.id == 12 || type.id == 13) {
                        const auto minimum = float_to_fp16_rne(-.015625f);
                        std::memcpy(p + 2, &minimum, 2);
                    }
                }
                for (auto dtype : {DecodeDType::f32, DecodeDType::f16, DecodeDType::bf16}) {
                    const size_t item = dtype_bytes(dtype), target_bytes = size_t(elements) * item;
                    alignas(16) std::array<std::byte, 8256> a{}, b{}, unaligned_output{};
                    a.fill(std::byte(0xa5)); b = a; unaligned_output = a;
                    const PackedMatrix input{{packed.data(), bytes}, type.id, 2, columns};
                    const DecodeSlice request{0, 2, 0, columns};
                    bool scalar_ok = true, vector_ok = true;
                    DecodeReceipt vector_receipt;
                    forbid_allocation = true;
                    try { decode_cpu_into(input, request, {{a.data() + 16, target_bytes}, dtype, columns * item, item}, nullptr, {false}); }
                    catch (const DecodeError &) { scalar_ok = false; }
                    try { vector_receipt = decode_cpu_into(input, request, {{b.data() + 16, target_bytes}, dtype, columns * item, item}); }
                    catch (const DecodeError &) { vector_ok = false; }
                    forbid_allocation = false;
                    expect(scalar_ok == vector_ok, "SIMD changed overflow acceptance");
                    for (size_t i = 0; i < 16; ++i) {
                        expect(a[i] == std::byte(0xa5) && b[i] == std::byte(0xa5), "prefix guard overwritten");
                        expect(a[16 + target_bytes + i] == std::byte(0xa5) && b[16 + target_bytes + i] == std::byte(0xa5), "suffix guard overwritten");
                    }
                    if (!scalar_ok) continue;
                    expect(a == b, "all-type SIMD/scalar bit mismatch");
#if defined(__aarch64__)
                    expect(vector_receipt.simd_blocks == 8, "registered type did not execute SIMD");
#else
                    (void)vector_receipt;
#endif
                    decode_cpu_into(input, request, {{unaligned_output.data() + 1, target_bytes + 7}, dtype, columns * item + 3, item});
                    for (size_t row = 0; row < 2; ++row)
                        expect(std::memcmp(unaligned_output.data() + 1 + row * (columns * item + 3),
                            a.data() + 16 + row * columns * item, columns * item) == 0, "unaligned all-type numerics changed");
                    const uint64_t gather[] = {1, 0, 1};
                    alignas(16) std::array<std::byte, 12352> ga{}, gb{};
                    forbid_allocation = true;
                    decode_cpu_gather(input, gather, 0, columns, {{ga.data(), 3 * columns * item}, dtype, columns * item, item}, nullptr, {false});
                    decode_cpu_gather(input, gather, 0, columns, {{gb.data(), 3 * columns * item}, dtype, columns * item, item});
                    forbid_allocation = false;
                    expect(ga == gb, "SIMD gather changed duplicate/order numerics");
                }
                // A nonfinite master scale must reject even zero/unused codes.
                const uint16_t infinity = 0x7c00;
                std::memcpy(packed.data() + (type.id == 14 ? 208 : 0), &infinity, 2);
                std::array<std::byte, 4096> invalid_output{};
                for (bool use_simd : {false, true}) rejected([&] {
                    decode_cpu_into({{packed.data(), bytes}, type.id, 2, columns}, {0, 2, 0, columns},
                        {{invalid_output.data(), size_t(elements * 2)}, DecodeDType::bf16, columns * 2, 2}, nullptr, {use_simd});
                });
            }
        }
        // Exhaustive finite FP16 sources through the actual vector conversion
        // API, not just the scalar bit-converter helper. Include subnormals.
        {
            alignas(16) std::array<uint16_t, 63488> halves{};
            alignas(16) std::array<std::byte, 63488 * 4> a{}, b{};
            size_t count = 0;
            for (uint32_t bits = 0; bits < 65536; ++bits)
                if ((bits & 0x7c00) != 0x7c00) halves[count++] = uint16_t(bits);
            expect(count == halves.size(), "wrong exhaustive FP16 source count");
            PackedMatrix input{{reinterpret_cast<const std::byte *>(halves.data()), count * 2}, 1, 1, count};
            for (auto dtype : {DecodeDType::f32, DecodeDType::bf16}) {
                const size_t bytes = count * dtype_bytes(dtype);
                forbid_allocation = true;
                decode_cpu_into(input, {0, 1, 0, count}, {{a.data(), bytes}, dtype, bytes, dtype_bytes(dtype)}, nullptr, {false});
                decode_cpu_into(input, {0, 1, 0, count}, {{b.data(), bytes}, dtype, bytes, dtype_bytes(dtype)});
                forbid_allocation = false;
                expect(a == b, "exhaustive finite FP16 vector conversion differs");
            }
            const float extremes[] = {0.f, -0.f, 0x1p-149f, -0x1p-149f, 1.f, -1.f, 65504.f, 65520.f,
                std::bit_cast<float>(uint32_t(0x7f7fffff)), std::bit_cast<float>(uint32_t(0xff7fffff)), 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
            input = {{reinterpret_cast<const std::byte *>(extremes), sizeof(extremes)}, 0, 1, 16};
            for (auto dtype : {DecodeDType::f16, DecodeDType::bf16}) for (bool simd : {false, true})
                rejected([&] { decode_cpu_into(input, {0, 1, 0, 16}, {{a.data(), 32}, dtype, 32, 2}, nullptr, {simd}); });
        }
        std::cout << "PASS GGUF decoder safety: allocation-free, alias, unaligned, overflow, cancellation\n";
        return 0;
    } catch (const std::exception &error) {
        forbid_allocation = false;
        std::cerr << error.what() << '\n'; return 1;
    }
}

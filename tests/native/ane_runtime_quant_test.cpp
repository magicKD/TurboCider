#include "../../native/backends/ane_runtime_quant.hpp"
#include <cassert>
#include <limits>
#include <vector>

using namespace tc::ane;

namespace {
void store(std::vector<char> &buffer, size_t offset, float value, DType dtype) {
    if (dtype == DType::FP32) std::memcpy(buffer.data() + offset, &value, 4);
    else {
        const uint16_t bits = dtype == DType::BF16 ? round_bf16(value) :
            std::bit_cast<uint16_t>(_Float16(value));
        std::memcpy(buffer.data() + offset, &bits, 2);
    }
}
template<class F> void rejects(F body) {
    bool rejected = false;
    try { body(); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
}
}

int main() {
    // All Q4/Q8 codes, signed metadata, independent strides and an unaligned
    // base pointer. Oracle uses the original logical codes, not packed reads.
    for (int bits : {4, 8}) for (int group : {32, 64, 128})
    for (auto dtype : {DType::FP16, DType::BF16, DType::FP32})
    for (bool biased : {false, true}) {
        constexpr int rows = 3, cols = 512;
        const size_t element = dtype == DType::FP32 ? 4 : 2;
        const size_t pitch = cols * bits / 8 + 3, metadata_pitch = cols / group * element + 5;
        std::vector<char> packed(1 + rows * pitch, 0), scales(1 + rows * metadata_pitch, 0);
        std::vector<char> offsets(scales.size(), 0);
        MatrixView metadata{scales.data() + 1, scales.size() - 1, rows, cols / group,
                            metadata_pitch, dtype};
        AffineView source{packed.data() + 1, packed.size() - 1, rows, cols, pitch,
                          group, bits, metadata, std::nullopt};
        if (biased) {
            auto m = metadata; m.data = offsets.data() + 1; source.offsets = m;
        }
        for (int row = 0; row < rows; ++row) {
            for (int g = 0; g < cols / group; ++g) {
                store(scales, 1 + row * metadata_pitch + g * element, (g % 2 ? -1.f : 1.f) / 128.f, dtype);
                store(offsets, 1 + row * metadata_pitch + g * element, float(g - 3) / 16.f, dtype);
            }
            for (int word = 0; word < cols / (32 / bits); ++word) {
                uint32_t value = 0;
                for (int lane = 0; lane < 32 / bits; ++lane) {
                    const unsigned code = (word * (32 / bits) + lane + row * 17) % (1 << bits);
                    value |= code << (lane * bits);
                }
                std::memcpy(packed.data() + 1 + row * pitch + word * 4, &value, 4);
            }
        }
        const auto original = packed;
        validate_affine_view(source);
        for (float factor : {1.f, .25f, 1.f / 4096.f}) for (int row = 0; row < rows; ++row) {
            std::vector<uint16_t> fast(cols + 2, 0xabcd), slow(cols + 2, 0xabcd);
            assert(affine_fp16_row(source, row, fast.data() + 1, false, factor));
            assert(affine_fp16_row(source, row, slow.data() + 1, true, factor));
            assert(fast == slow);
            assert(fast.front() == 0xabcd && fast.back() == 0xabcd);
            for (int col = 0; col < cols; ++col) {
                const int g = col / group;
                const float scale = (g % 2 ? -1.f : 1.f) / 128.f;
                const float bias = biased ? float(g - 3) / 16.f : 0.f;
                const float code = float((col + row * 17) % (1 << bits));
                assert(fast[col + 1] == std::bit_cast<uint16_t>(_Float16(std::fma(scale, code, bias) * factor)));
            }
        }
        assert(original == packed);
        for (int invalid : {0, 3, 6, 16}) {
            auto bad = source; bad.bits = invalid;
            rejects([&] { validate_affine_view(bad); });
        }
        for (int invalid : {0, 16, 33, 256}) {
            auto bad = source; bad.group_size = invalid;
            rejects([&] { validate_affine_view(bad); });
        }
        auto bad = source; bad.bytes = (rows - 1) * pitch + cols * bits / 8 - 1;
        rejects([&] { validate_affine_view(bad); });
        bad = source; bad.scales.bytes = (rows - 1) * metadata_pitch + cols / group * element - 1;
        rejects([&] { validate_affine_view(bad); });
        bad = source; bad.row_stride_bytes = SIZE_MAX;
        rejects([&] { validate_affine_view(bad); });
        bad = source; bad.scales.row_stride_bytes = SIZE_MAX;
        rejects([&] { validate_affine_view(bad); });
        bad = source; bad.scales.cols--;
        rejects([&] { validate_affine_view(bad); });
        bad = source; bad.scales.data = nullptr;
        rejects([&] { validate_affine_view(bad); });
        bad = source; bad.cols--;
        rejects([&] { validate_affine_view(bad); });
        std::vector<uint16_t> out(cols);
        for (float invalid : {std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
            store(scales, 1, invalid, dtype);
            assert(!affine_fp16_row(source, 0, out.data()));
            assert(!affine_fp16_row(source, 0, out.data(), true));
        }
    }
    // Unused Q4 LUT values can overflow; only actually selected codes count.
    std::vector<uint32_t> packed(4, 0);
    float scale = 5000.f;
    AffineView source{packed.data(), packed.size() * 4, 1, 32, 0, 32, 4,
                      {&scale, 4, 1, 1, 0, DType::FP32}, std::nullopt};
    validate_affine_view(source);
    uint16_t out[32];
    assert(affine_fp16_row(source, 0, out));
    packed[0] = 15;
    assert(!affine_fp16_row(source, 0, out));
    assert(affine_fp16_row(source, 0, out, false, .25f));
    assert(out[0] == std::bit_cast<uint16_t>(_Float16(18750.f)));
}

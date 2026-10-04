#include "../../native/backends/ane_runtime_packed.hpp"
#include <cassert>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

using namespace tc::ane;
namespace {
bool forbid_allocations = false;
template<class F> void rejects(F body) {
    bool rejected = false;
    try { body(); } catch (const std::exception &) { rejected = true; }
    assert(rejected);
}
void put_half(void *p, float value) {
    const auto bits = tc::gguf::float_to_fp16_rne(value);
    std::memcpy(p, &bits, 2);
}
// Independent dense Kronecker oracle: no calls to the butterfly under test.
int sign(int i, int j) {
    constexpr int h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1},
                            {1, -1, 1, 1}, {-1, 1, 1, 1}};
    int value = 1;
    for (int digit = 0; digit < 4; ++digit, i /= 4, j /= 4) value *= h4[i % 4][j % 4];
    return value;
}
}
void *operator new(size_t bytes) {
    assert(!forbid_allocations);
    if (auto *p = std::malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void *operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }

int main() {
    for (int basis = 0; basis < 256; ++basis) {
        std::array<int32_t, 256> a{}, b{};
        a[basis] = b[basis] = -128;
        convrot_integer_h256(a, false); convrot_integer_h256(b, true);
        for (int column = 0; column < 256; ++column)
            assert(a[column] == -128 * sign(basis, column) && a[column] == b[column]);
    }
    constexpr int rows = 3, columns = 768, pitch = columns + 3, scale_pitch = 9;
    std::vector<int8_t> codes(1 + rows * pitch);
    std::vector<char> scales(1 + rows * scale_pitch);
    const float row_scales[] = {.015625f, -.031f, 0.f};
    for (int r = 0; r < rows; ++r) {
        std::memcpy(scales.data() + 1 + r * scale_pitch, row_scales + r, 4);
        for (int c = 0; c < columns; ++c) codes[1 + r * pitch + c] = int8_t((c * 37 + r * 19) % 256 - 128);
    }
    const auto original_codes = codes;
    const auto original_scales = scales;
    ConvrotView rotated{codes.data() + 1, codes.size() - 1, rows, columns, pitch,
        {scales.data() + 1, scales.size() - 1, rows, 1, scale_pitch, DType::FP32}};
    validate_convrot_view(rotated);
    std::vector<uint16_t> fast(columns + 2, 0xabcd), slow(fast);
    for (float factor : {1.f, .25f, 1.f / 4096.f}) for (int r = 0; r < rows; ++r) {
        forbid_allocations = true;
        assert(convrot_fp16_row(rotated, r, fast.data() + 1, false, factor));
        assert(convrot_fp16_row(rotated, r, slow.data() + 1, true, factor));
        forbid_allocations = false;
        assert(fast == slow && fast.front() == 0xabcd && fast.back() == 0xabcd);
        for (int c = 0; c < columns; ++c) {
            int64_t dot = 0;
            for (int k = 0; k < 256; ++k) dot += int64_t(codes[1 + r * pitch + c / 256 * 256 + k]) * sign(k, c % 256);
            const float value = ((float(dot) / 16.f) * row_scales[r]) * factor;
            assert(fast[c + 1] == tc::gguf::float_to_fp16_rne(value));
        }
    }
    assert(codes == original_codes && scales == original_scales);
    for (int bad_columns : {0, 255, 257, 33024}) {
        auto bad = rotated; bad.cols = bad_columns;
        rejects([&] { validate_convrot_view(bad); });
    }
    auto bad = rotated; bad.bytes = (rows - 1) * pitch + columns - 1;
    rejects([&] { validate_convrot_view(bad); });
    bad = rotated; bad.row_scales.dtype = DType::BF16;
    rejects([&] { validate_convrot_view(bad); });
    bad = rotated; bad.row_scales.bytes = (rows - 1) * scale_pitch + 3;
    rejects([&] { validate_convrot_view(bad); });
    bad = rotated; bad.row_stride_bytes = SIZE_MAX;
    rejects([&] { validate_convrot_view(bad); });
    assert(!convrot_fp16_row(rotated, rows, fast.data()));
    assert(!convrot_fp16_row(rotated, 0, reinterpret_cast<uint16_t *>(reinterpret_cast<char *>(fast.data()) + 1)));
    assert(!convrot_fp16_row(rotated, 0, reinterpret_cast<uint16_t *>(codes.data() + 2)));
    assert(!convrot_fp16_row(rotated, 0, reinterpret_cast<uint16_t *>(scales.data() + 2)));
    for (float value : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), 1e10f}) {
        std::memcpy(scales.data() + 1, &value, 4);
        assert(!convrot_fp16_row(rotated, 0, fast.data(), false));
        assert(!convrot_fp16_row(rotated, 0, fast.data(), true));
    }
    float huge_scale = 1e5f;
    std::memcpy(scales.data() + 1, &huge_scale, 4);
    assert(!convrot_fp16_row(rotated, 0, fast.data(), false));
    assert(convrot_fp16_row(rotated, 0, fast.data(), false, 1.f / 4096.f));
    assert(convrot_fp16_row(rotated, 0, slow.data(), true, 1.f / 4096.f));
    assert(fast == slow);

    // Explicit legacy packed ConvRot: recover signed codes without a second
    // source bank, use the ACTUAL rounded metadata, and never treat ordinary
    // group-affine scales/offsets as a valid rotation recipe.
    {
        constexpr int groups=columns/32,meta_pitch=groups*2+3,packed_pitch=columns+5;
        std::vector<uint8_t> packed(1+rows*packed_pitch);
        std::vector<char> legacy_scales(1+rows*meta_pitch),legacy_biases(legacy_scales.size());
        std::array<float,rows> effective_scales;
        for (int r=0;r<rows;++r) {
            const uint16_t bits=round_bf16(row_scales[r]);
            effective_scales[r]=std::bit_cast<float>(uint32_t(bits)<<16);
            const uint16_t bias=round_bf16(-128.f*effective_scales[r]);
            for (int g=0;g<groups;++g) {
                std::memcpy(legacy_scales.data()+1+r*meta_pitch+g*2,&bits,2);
                std::memcpy(legacy_biases.data()+1+r*meta_pitch+g*2,&bias,2);
            }
            for (int c=0;c<columns;++c)
                packed[1+r*packed_pitch+c]=uint8_t(int(original_codes[1+r*pitch+c])+128);
        }
        ConvrotAffineView legacy{{packed.data()+1,packed.size()-1,rows,columns,packed_pitch,32,8,
            {legacy_scales.data()+1,legacy_scales.size()-1,rows,groups,meta_pitch,DType::BF16},
            MatrixView{legacy_biases.data()+1,legacy_biases.size()-1,rows,groups,meta_pitch,DType::BF16}}};
        validate_convrot_affine_view(legacy);
        ConvrotView exact_raw{original_codes.data()+1,original_codes.size()-1,rows,columns,pitch,
            {effective_scales.data(),sizeof(effective_scales),rows,1,0,DType::FP32}};
        const auto original_packed=packed;
        const auto original_metadata=legacy_scales;
        std::fill(fast.begin(),fast.end(),0xabcd);slow=fast;
        for (int r=0;r<rows;++r) for (float factor : {1.f,.25f,1.f/4096.f}) {
            forbid_allocations=true;
            assert(convrot_affine_fp16_row(legacy,r,fast.data()+1,false,factor));
            assert(convrot_affine_fp16_row(legacy,r,slow.data()+1,true,factor));
            forbid_allocations=false;assert(fast==slow && fast.front()==0xabcd && fast.back()==0xabcd);
            assert(convrot_fp16_row(exact_raw,r,slow.data()+1,true,factor));assert(fast==slow);
            for (int c=0;c<columns;++c) {
                int64_t dot=0;for (int k=0;k<256;++k)
                    dot+=int64_t(original_codes[1+r*pitch+c/256*256+k])*sign(k,c%256);
                assert(fast[c+1]==tc::gguf::float_to_fp16_rne((float(dot)*.0625f*effective_scales[r])*factor));
            }
        }
        assert(packed==original_packed && legacy_scales==original_metadata);
        auto wrong=legacy;wrong.packed.bits=4;rejects([&]{validate_convrot_affine_view(wrong);});
        wrong=legacy;wrong.packed.offsets.reset();rejects([&]{validate_convrot_affine_view(wrong);});
        wrong=legacy;wrong.packed.bytes=columns-1;rejects([&]{validate_convrot_affine_view(wrong);});
        wrong=legacy;wrong.packed.data=reinterpret_cast<const void *>(UINTPTR_MAX-16);
        rejects([&]{validate_convrot_affine_view(wrong);});
        assert(!convrot_affine_fp16_row(legacy,rows,fast.data()));
        assert(!convrot_affine_fp16_row(legacy,0,reinterpret_cast<uint16_t *>(packed.data()+2)));
        assert(!convrot_affine_fp16_row(legacy,0,reinterpret_cast<uint16_t *>(legacy_scales.data()+2)));
        assert(!convrot_affine_fp16_row(legacy,0,reinterpret_cast<uint16_t *>(legacy_biases.data()+2)));
        uint16_t bad_scale=round_bf16(.5f);
        std::memcpy(legacy_scales.data()+1+2,&bad_scale,2);
        assert(!convrot_affine_fp16_row(legacy,0,fast.data()));legacy_scales=original_metadata;
        uint16_t bad_bias=round_bf16(1.f);
        std::memcpy(legacy_biases.data()+1,&bad_bias,2);
        assert(!convrot_affine_fp16_row(legacy,0,fast.data()));
        for (float factor : {0.f,-1.f,std::numeric_limits<float>::infinity()})
            assert(!convrot_affine_fp16_row(legacy,0,fast.data(),false,factor));
    }

    // All registered GGUF types through the NEW staging bridge; decoder's
    // independent GGML D0 tests remain separate. Padded, unaligned source rows.
    for (const auto &type : tc::gguf::types) {
        constexpr int cols = 512;
        const size_t row_bytes = cols / type.elements * type.bytes, stride = row_bytes + 3;
        std::vector<std::byte> packed(1 + rows * stride);
        for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; c += type.elements) {
            auto *p = packed.data() + 1 + r * stride + c / type.elements * type.bytes;
            for (unsigned i = 0; i < type.bytes; ++i) p[i] = std::byte((i * 17 + c + r) % 256);
            if (type.id == 0) { float f = .25f * (r + 1); std::memcpy(p, &f, 4); }
            else if (type.id == 30) { const auto bits = round_bf16(.25f * (r + 1)); std::memcpy(p, &bits, 2); }
            else {
                put_half(p + (type.id == 14 ? 208 : 0), .03125f);
                if (type.id == 3 || type.id == 7 || type.id == 12 || type.id == 13) put_half(p + 2, -.015625f);
            }
        }
        const auto original = packed;
        GgufView raw{packed.data() + 1, packed.size() - 1, rows, cols, stride, type.id};
        validate_gguf_view(raw);
        std::vector<float> reference(cols);
        std::vector<uint16_t> a(cols + 2, 0xabcd), b(a);
        for (int r = 0; r < rows; ++r) for (float factor : {1.f, .25f, 1.f / 4096.f}) {
            tc::gguf::decode_cpu_into({{packed.data() + 1 + r * stride, row_bytes}, type.id, 1, cols},
                {0, 1, 0, cols}, {{reinterpret_cast<std::byte *>(reference.data()), cols * 4},
                tc::gguf::DecodeDType::f32, cols * 4, 4});
            forbid_allocations = true;
            assert(gguf_fp16_row(raw, r, a.data() + 1, false, factor));
            assert(gguf_fp16_row(raw, r, b.data() + 1, true, factor));
            forbid_allocations = false;
            assert(a == b && a.front() == 0xabcd && a.back() == 0xabcd);
            for (int c = 0; c < cols; ++c) assert(a[c + 1] == tc::gguf::float_to_fp16_rne(reference[c] * factor));
        }
        assert(packed == original);
        auto invalid = raw; invalid.bytes = (rows - 1) * stride + row_bytes - 1;
        rejects([&] { validate_gguf_view(invalid); });
        invalid = raw; invalid.row_stride_bytes = SIZE_MAX;
        rejects([&] { validate_gguf_view(invalid); });
        invalid = raw; invalid.ggml_type = 23;
        rejects([&] { validate_gguf_view(invalid); });
        assert(!gguf_fp16_row(raw, rows, a.data()));
        assert(!gguf_fp16_row(raw, 0, reinterpret_cast<uint16_t *>(reinterpret_cast<char *>(a.data()) + 1)));
        assert(!gguf_fp16_row(raw, 0, reinterpret_cast<uint16_t *>(packed.data() + 2)));
        if (type.elements > 1) {
            invalid = raw; invalid.cols--;
            rejects([&] { validate_gguf_view(invalid); });
        }
    }
    std::array<float, 256> too_large;
    too_large.fill(1e5f);
    GgufView raw{too_large.data(), sizeof(too_large), 1, 256, 0, 0};
    validate_gguf_view(raw);
    assert(!gguf_fp16_row(raw, 0, fast.data()));
    assert(gguf_fp16_row(raw, 0, fast.data(), false, .25f));
    assert(fast[0] == tc::gguf::float_to_fp16_rne(25000.f));
    too_large[0] = std::numeric_limits<float>::infinity();
    assert(!gguf_fp16_row(raw, 0, fast.data(), false, .25f));
    for (float factor : {0.f, -1.f, std::numeric_limits<float>::infinity()}) {
        assert(!gguf_fp16_row(raw, 0, fast.data(), false, factor));
        assert(!convrot_fp16_row(rotated, 0, fast.data(), false, factor));
    }
}

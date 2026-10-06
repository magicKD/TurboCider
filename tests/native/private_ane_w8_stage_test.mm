#include "../../native/backends/private/ane_program.hpp"
#include "../../native/backends/ane_w8a8_math.hpp"
#import <Metal/Metal.h>
#include <cstring>
#include <iostream>

#pragma clang fp contract(off)
using namespace tc::ane;
using namespace tc::ane::private_api;
struct Buffer {
    id<MTLBuffer> value;
    std::shared_ptr<void> owner;
    Buffer(id<MTLDevice> gpu, size_t bytes) {
        value = [gpu newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!value) throw std::runtime_error("test allocation failed");
        owner = {(__bridge_retained void *)value, [](void *p) { CFRelease(p); }};
        std::memset(value.contents, 0, bytes);
    }
};
uint32_t random_word(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16); }
void put_half(uint8_t *p, float value) { const auto h = tc::gguf::float_to_fp16_rne(value); std::memcpy(p, &h, 2); }
int main() {
    @autoreleasepool {
      try {
        auto gpu = MTLCreateSystemDefaultDevice();
        setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE","0",1);Device device;
        setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE","1",1);Device specialized;
        constexpr int rows = 5, cols = 1024;
        for (auto encoding : {DeviceWeightEncoding::Dense, DeviceWeightEncoding::AffineQ4, DeviceWeightEncoding::AffineQ8,
            DeviceWeightEncoding::GgufQ4_0, DeviceWeightEncoding::GgufQ4_K, DeviceWeightEncoding::GgufQ8_0, DeviceWeightEncoding::GgufQ6_K}) {
            for (auto dtype : {DType::FP16, DType::BF16, DType::FP32}) {
                if (encoding != DeviceWeightEncoding::Dense && dtype != DType::FP32) continue;
                const bool affine = encoding == DeviceWeightEncoding::AffineQ4 || encoding == DeviceWeightEncoding::AffineQ8;
                const int bits = encoding == DeviceWeightEncoding::AffineQ4 ? 4 : 8;
                const int ggml = encoding == DeviceWeightEncoding::GgufQ4_0 ? 2 : encoding == DeviceWeightEncoding::GgufQ4_K ? 12 :
                    encoding == DeviceWeightEncoding::GgufQ8_0 ? 8 : 14;
                const int item = dtype == DType::FP32 ? 4 : 2;
                const int raw_bytes = ggml == 2 ? 18 : ggml == 12 ? 144 : ggml == 8 ? 34 : 210;
                const int raw_group = ggml == 2 || ggml == 8 ? 32 : 256;
                const size_t packed = encoding == DeviceWeightEncoding::Dense ? cols * item : affine ? cols * bits / 8 : cols / raw_group * raw_bytes;
                const size_t pitch = packed + 32, offset = 256;
                Buffer src(gpu, offset + rows * pitch + 256), meta(gpu, rows * 256), bias(gpu, rows * 128);
                DeviceWeightView view{(__bridge void *)src.value, src.value.length, offset, pitch, rows, cols, encoding, dtype, 32,
                    std::nullopt, std::nullopt, src.owner};
                if (affine) {
                    view.scales = DeviceMatrixView{(__bridge void *)meta.value, meta.value.length, 0, rows, cols / 32, 256, DType::FP32, meta.owner};
                    view.offsets = DeviceMatrixView{(__bridge void *)bias.value, bias.value.length, 0, rows, cols / 32, 128, DType::BF16, bias.owner};
                }
                std::vector<float> dense(rows * cols);
                for (int r = 0; r < rows; ++r) {
                    auto *at = static_cast<uint8_t *>(src.value.contents) + offset + size_t(r) * pitch;
                    if (encoding == DeviceWeightEncoding::Dense) for (int c = 0; c < cols; ++c) {
                        float x = r == 0 ? 0 : r == 1 ? 1e-20f : (int(random_word(r * cols + c) % 2048) - 1024) / 1024.f;
                        if (dtype == DType::FP32) std::memcpy(at + c * 4, &x, 4);
                        else {
                            uint16_t h = dtype == DType::BF16 ? tc::gguf::float_to_bf16_rne(x) : tc::gguf::float_to_fp16_rne(x);
                            std::memcpy(at + c * 2, &h, 2); x = dtype == DType::BF16 ? std::bit_cast<float>(uint32_t(h) << 16) : tc::gguf::fp16_to_float(h);
                        }
                        dense[r * cols + c] = x;
                    } else if (affine) {
                        auto *words = reinterpret_cast<uint32_t *>(at);
                        for (int w = 0; w < cols / (32 / bits); ++w) words[w] = random_word(r * 567 + w);
                        auto *s = reinterpret_cast<float *>(static_cast<char *>(meta.value.contents) + r * 256);
                        auto *b = reinterpret_cast<uint16_t *>(static_cast<char *>(bias.value.contents) + r * 128);
                        for (int g = 0; g < cols / 32; ++g) { s[g] = r ? (g + 1) * .0003f : 0; b[g] = tc::gguf::float_to_bf16_rne(r ? -.003f * g : 0); }
                        for (int c = 0; c < cols; ++c) {
                            const auto q = (words[c / (32 / bits)] >> (c % (32 / bits) * bits)) & ((1u << bits) - 1);
                            dense[r * cols + c] = float(q) * s[c / 32] + std::bit_cast<float>(uint32_t(b[c / 32]) << 16);
                        }
                    } else {
                        for (int g = 0; g < cols / raw_group; ++g) {
                            auto *p = at + g * raw_bytes;
                            for (int i = 0; i < raw_bytes; ++i) p[i] = uint8_t(random_word(r * 127 + g * 31 + i));
                            if (ggml == 14) put_half(p + 208, r ? .002f * (g + 1) : 0);
                            else { put_half(p, r ? .001f * (g + 1) : 0); if (ggml == 12) put_half(p + 2, r ? .0005f * (g + 1) : 0); }
                        }
                        tc::gguf::decode_cpu_into({{reinterpret_cast<const std::byte *>(at), packed}, uint32_t(ggml), 1, cols},
                            {0, 1, 0, cols}, {{reinterpret_cast<std::byte *>(dense.data() + r * cols), size_t(cols) * 4}, tc::gguf::DecodeDType::f32, cols * 4, 4});
                    }
                }
                for (int block : {128, 512}) for (bool transpose : {false, true}) {
                    const W8StageSpec spec{0, rows, block, block == 128 ? 384 : 512, block, 20260930, transpose};
                    Surface codes(device, transpose ? spec.columns : rows, transpose ? rows : spec.columns, Element::I8);
                    Surface scales(device, transpose ? 1 : rows, transpose ? rows : 1, Element::FP16);
                    std::memset(codes.data(), 0x5a, codes.rows() * codes.pitch()); std::memset(scales.data(), 0x5a, scales.rows() * scales.pitch());
                    const auto old_value = device.value();
                    auto job = device.stage_w8(view, spec, codes, scales);
                    auto result = job.finish(); if (!result.ok) throw std::runtime_error(result.error);
                    Surface fast_codes(specialized,codes.rows(),codes.columns(),Element::I8),
                            fast_scales(specialized,scales.rows(),scales.columns(),Element::FP16);
                    std::memset(fast_codes.data(),0x5a,fast_codes.rows()*fast_codes.pitch());
                    std::memset(fast_scales.data(),0x5a,fast_scales.rows()*fast_scales.pitch());
                    auto fast=specialized.stage_w8(view,spec,fast_codes,fast_scales);
                    if(!fast.finish().ok || fast.validation_flags()!=job.validation_flags() ||
                       std::memcmp(codes.data(),fast_codes.data(),codes.rows()*codes.pitch()) ||
                       std::memcmp(scales.data(),fast_scales.data(),scales.rows()*scales.pitch()))
                        throw std::runtime_error("specialized stage differs from generic codes/scales/padding");
                    if (encoding == DeviceWeightEncoding::Dense) {
                        for (size_t extra_offset : {size_t(0),size_t(1)}) for (size_t extra_pitch : {size_t(0),size_t(1)}) {
                            const size_t shifted_offset=256+extra_offset,shifted_pitch=packed+32+extra_pitch;
                            Buffer shifted(gpu,shifted_offset+rows*shifted_pitch+256);
                            for (int row=0;row<rows;++row)
                                std::memcpy(static_cast<uint8_t*>(shifted.value.contents)+shifted_offset+row*shifted_pitch,
                                            static_cast<const uint8_t*>(src.value.contents)+offset+row*pitch,packed);
                            auto shifted_view=view;shifted_view.buffer=(__bridge void*)shifted.value;
                            shifted_view.buffer_bytes=shifted.value.length;shifted_view.owner=shifted.owner;
                            shifted_view.offset_bytes=shifted_offset;shifted_view.row_stride_bytes=shifted_pitch;
                            std::memset(fast_codes.data(),0x5a,fast_codes.rows()*fast_codes.pitch());
                            std::memset(fast_scales.data(),0x5a,fast_scales.rows()*fast_scales.pitch());
                            auto moved=specialized.stage_w8(shifted_view,spec,fast_codes,fast_scales);
                            if(!moved.finish().ok || moved.validation_flags()!=job.validation_flags() ||
                               std::memcmp(codes.data(),fast_codes.data(),codes.rows()*codes.pitch()) ||
                               std::memcmp(scales.data(),fast_scales.data(),scales.rows()*scales.pitch()))
                                throw std::runtime_error("typed-load alignment fallback changed codes/scales/padding");
                        }
                    }
                    if (device.value() != old_value || !job.ready_event() ||
                        ((__bridge id<MTLSharedEvent>)job.ready_event()).signaledValue != 1) throw std::runtime_error("weight staging advanced current ANE timeline");
                    for (int r = 0; r < rows; ++r) {
                        std::vector<float> rotated(dense.begin() + r * cols + spec.column_begin, dense.begin() + r * cols + spec.column_begin + spec.columns);
                        for (int c = 0; c < spec.columns; c += block) rotate_block({rotated.data() + c, size_t(block)}, spec.rotation_seed);
                        float peak = 0; for (float x : rotated) peak = std::max(peak, std::abs(x));
                        const auto expected = normalized_scale(peak);
                        const auto *s = reinterpret_cast<const uint16_t *>(static_cast<const char *>(scales.data()) + (transpose ? 0 : r * scales.pitch()));
                        if (s[transpose ? r : 0] != expected) throw std::runtime_error("GPU row scale differs from CPU recipe");
                        for (int c = 0; c < spec.columns; ++c) {
                            const auto *q = static_cast<const int8_t *>(codes.data());
                            const int8_t actual = q[transpose ? c * codes.pitch() + r : r * codes.pitch() + c];
                            if (actual != quantize_rotated(rotated[c], expected)) throw std::runtime_error("GPU code differs from CPU decode/rotation/RNE");
                        }
                    }
                    if(!transpose) {
                        auto immutable=view;immutable.immutable_generation=true;immutable.allocation_identity=src.owner;
                        if(immutable.scales)immutable.scales->allocation_identity=meta.owner;
                        if(immutable.offsets)immutable.offsets->allocation_identity=bias.owner;
                        const auto before=device.scale_cache_stats();
                        auto miss=device.stage_w8(immutable,spec,codes,scales);if(!miss.finish().ok)throw std::runtime_error("immutable scale cache miss failed");
                        std::vector<uint8_t> expected_codes(codes.rows()*codes.pitch()),expected_scales(scales.rows()*scales.pitch());
                        std::memcpy(expected_codes.data(),codes.data(),expected_codes.size());std::memcpy(expected_scales.data(),scales.data(),expected_scales.size());
                        std::memset(codes.data(),0x5a,expected_codes.size());std::memset(scales.data(),0x5a,expected_scales.size());
                        auto hit=device.stage_w8(immutable,spec,codes,scales);if(!hit.finish().ok)throw std::runtime_error("immutable scale cache hit failed");
                        if(std::memcmp(expected_codes.data(),codes.data(),expected_codes.size())||std::memcmp(expected_scales.data(),scales.data(),expected_scales.size()))
                            throw std::runtime_error("cached scales/codes are not bit-exact vs fresh staging");
                        const auto after=device.scale_cache_stats();
                        if(after.hits!=before.hits+1||after.misses!=before.misses+1||after.bytes>scale_cache_budget_bytes)
                            throw std::runtime_error("bounded scale cache counters missing");
                        auto changed=spec;changed.rotation_seed++;
                        auto other=device.stage_w8(immutable,changed,codes,scales);if(!other.finish().ok||device.scale_cache_stats().misses!=after.misses+1)
                            throw std::runtime_error("scale cache ignored rotation identity");
                    }
                    // Padding is not part of the representation and must not be overwritten.
                    if (transpose) for (int c = 0; c < spec.columns; ++c) for (size_t i = rows; i < codes.pitch(); ++i)
                        if (static_cast<const uint8_t *>(codes.data())[c * codes.pitch() + i] != 0x5a) throw std::runtime_error("A8 padding overwrite");
                    auto invalid = view; invalid.buffer_bytes = invalid.offset_bytes + 1;
                    try { device.stage_w8(invalid, spec, codes, scales); throw std::runtime_error("short buffer accepted"); }
                    catch (const CapabilityError &) {}
                }
                std::cout << "PASS GPU W8 source=" << unsigned(encoding) << " dtype=" << unsigned(dtype) << " H128/H512 W/A layouts, physical stride/slice, zero/tiny rows\n";
            }
        }
        // Invalid source must never be staged as successful; a clean refill recovers.
        Buffer source(gpu, 512 * 4);
        auto *f = static_cast<float *>(source.value.contents); f[0] = std::numeric_limits<float>::infinity();
        DeviceWeightView view{(__bridge void *)source.value, source.value.length, 0, 512 * 4, 1, 512, DeviceWeightEncoding::Dense, DType::FP32, 32, {}, {}, source.owner};
        Surface q(device, 1, 512, Element::I8), s(device, 1, 1, Element::FP16);
        auto bad = device.stage_w8(view, {0, 1, 0, 512, 128}, q, s); if (bad.finish().ok || !(bad.validation_flags() & 1)) throw std::runtime_error("nonfinite accepted");
        f[0] = 1e20f;
        auto huge = device.stage_w8(view, {0, 1, 0, 512, 128}, q, s); if (huge.finish().ok || !(huge.validation_flags() & 4)) throw std::runtime_error("scale overflow accepted");
        f[0] = 0;
        auto clean = device.stage_w8(view, {0, 1, 0, 512, 128}, q, s); if (!clean.finish().ok) throw std::runtime_error("fresh refill failed");
        std::cout << "PASS W8 nonfinite/scale overflow rejection and fresh-slot refill\n";
        auto generation=std::make_shared<int>(1);std::weak_ptr<void> lifetime=generation;
        view.immutable_generation=true;view.allocation_identity=generation;view.owner=generation;f[0]=.25f;
        const auto before=device.scale_cache_stats();
        {auto a=device.stage_w8(view,{0,1,0,512,128},q,s);if(!a.finish().ok)throw std::runtime_error("generation initial cache failed");}
        view.owner=source.owner;generation.reset();
        if(!lifetime.expired())throw std::runtime_error("scale cache retained a source allocation/model");
        auto replacement=std::make_shared<int>(2);view.allocation_identity=replacement;f[0]=-.5f;
        {auto b=device.stage_w8(view,{0,1,0,512,128},q,s);if(!b.finish().ok)throw std::runtime_error("new-generation cache failed");}
        if(device.scale_cache_stats().misses!=before.misses+2)throw std::runtime_error("raw address reuse hit stale scales");
        view.immutable_generation=false;
        const auto old=device.scale_cache_stats();f[0]=.75f;
        {auto m=device.stage_w8(view,{0,1,0,512,128},q,s);if(!m.finish().ok)throw std::runtime_error("mutable source path failed");}
        if(device.scale_cache_stats().hits!=old.hits||device.scale_cache_stats().misses!=old.misses)
            throw std::runtime_error("mutable source reused an immutable scale cache");
        view.immutable_generation=true;
        for(uint64_t seed=1;seed<=140;++seed) {
            auto churn=device.stage_w8(view,{0,1,0,512,128,seed},q,s);if(!churn.finish().ok)throw std::runtime_error("scale cache LRU churn failed");
        }
        const auto bounded=device.scale_cache_stats();
        if(bounded.entries>128||bounded.bytes>scale_cache_budget_bytes||!bounded.evictions)throw std::runtime_error("scale cache LRU/cap not enforced");
        const auto pipeline_stats=specialized.stage_pipeline_stats();
        if(!pipeline_stats.specialized||pipeline_stats.variants!=18||device.stage_pipeline_stats().specialized||
           device.stage_pipeline_stats().variants!=1)throw std::runtime_error("format/dtype/block pipeline cache is not bounded/complete");
        // Metadata seed replacement must preserve the exact unsigned 64-bit
        // recipe, including H128's prefix and H512, and must retain an older
        // immutable sign table while its GPU producer is still in flight.
        auto verify_signs=[&](const Surface &codes,const Surface &scales,uint64_t seed,int block) {
            std::vector<float> rotated(f,f+512);
            for(int col=0;col<512;col+=block)rotate_block({rotated.data()+col,size_t(block)},seed);
            float peak=0;for(float value:rotated)peak=std::max(peak,std::abs(value));
            const auto wanted=normalized_scale(peak);
            if(*static_cast<const uint16_t*>(scales.data())!=wanted)throw std::runtime_error("rotation-sign seed table changed scales");
            for(int col=0;col<512;++col)if(static_cast<const int8_t*>(codes.data())[col]!=quantize_rotated(rotated[col],wanted))
                throw std::runtime_error("rotation-sign seed table changed codes");
        };
        Surface other_q(device,1,512,Element::I8),other_s(device,1,1,Element::FP16);
        for(int block:{128,512}) {
            const uint64_t first_seed=0x1234567800000001ULL,second_seed=UINT64_MAX;
            auto first=device.stage_w8(view,{0,1,0,512,block,first_seed},q,s);
            auto second=device.stage_w8(view,{0,1,0,512,block,second_seed},other_q,other_s);
            if(!first.finish().ok||!second.finish().ok)throw std::runtime_error("in-flight sign seed replacement failed");
            verify_signs(q,s,first_seed,block);verify_signs(other_q,other_s,second_seed,block);
        }
        // Even cache hits must still reject invalid source data in the codes
        // pass; bypassing the first scale pass must not bypass finite checks.
        f[0]=std::numeric_limits<float>::infinity();
        auto invalid_hit=device.stage_w8(view,{0,1,0,512,128,140},q,s);
        if(invalid_hit.finish().ok||!(invalid_hit.validation_flags()&1))throw std::runtime_error("cached scale path hid nonfinite source");
        // One generic pipeline is shared across rotation blocks AND bases.
        // Dispatch dimensions must follow each operation, not its first key.
        view.immutable_generation=false;
        for(int c=0;c<512;++c)f[c]=float((c*13)%127-63)/64.f;
        Surface activation_q(device,512,1,Element::I8),activation_s(device,1,1,Element::FP16);
        for(int block:{512,128,256,128,512}) {
            const bool comfy=block==256;
            W8StageSpec spec{0,1,0,512,block,comfy?0u:20260930u,true,comfy?W8Basis::ComfyH256:W8Basis::SylvesterDH};
            auto mixed=device.stage_w8(view,spec,activation_q,activation_s);
            if(!mixed.finish().ok)throw std::runtime_error("mixed generic H128/H512/Comfy refill failed");
            std::vector<float> rotated(f,f+512);
            for(int c=0;c<512;c+=block) {
                if(comfy)rotate_comfy_block({rotated.data()+c,size_t(block)},DType::FP32);
                else rotate_block({rotated.data()+c,size_t(block)},spec.rotation_seed);
            }
            float peak=0;for(float x:rotated)peak=std::max(peak,std::abs(x));
            const auto expected=normalized_scale(peak);
            if(*static_cast<const uint16_t*>(activation_s.data())!=expected)
                throw std::runtime_error("cached generic pipeline used stale rotation geometry");
            for(int c=0;c<512;++c)
                if(static_cast<const int8_t*>(activation_q.data())[c*activation_q.pitch()]!=quantize_rotated(rotated[c],expected))
                    throw std::runtime_error("cached generic pipeline changed mixed-basis codes");
        }
        if(device.stage_pipeline_stats().variants!=1)throw std::runtime_error("mixed generic basis created extra pipelines");
        std::cout<<"PASS W8 shared generic pipeline: H512/H128/Comfy-H256/H128/H512 per-operation launch geometry and CPU code/scale oracle\n";
        std::cout<<"PASS W8 immutable sign metadata: H128/H512 unsigned64 seed oracle and in-flight table replacement\n";
        std::cout<<"PASS W8 pipeline specialization: 9 encodings/dtypes H128/H512 W/A bit-exact, bounded 18 variants\n";
        std::cout<<"PASS W8 dense typed loads: FP16/BF16/FP32 aligned and independently unaligned offset/pitch, H128/H512 W/A bit-exact\n";
        std::cout<<"PASS W8 compact scale cache: 9 encodings/dtypes H128/H512 bit-exact, recipe identity, weak generations/address reuse, mutable bypass, bounded metadata\n";
      } catch (const std::exception &error) { std::cerr << error.what() << "\n"; return 1; }
    }
}

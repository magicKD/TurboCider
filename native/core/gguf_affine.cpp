#include "gguf_affine.hpp"
#include <cstring>
#include <cmath>
#include <bit>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tc::gguf {
uint64_t pack_native_affine(const PackedMatrix &source, AffinePart part, std::span<std::byte> target,
                            const std::atomic<bool> *cancel) {
    if(part!=AffinePart::codes && part!=AffinePart::scales && part!=AffinePart::biases)
        throw DecodeError("invalid affine part");
    if (source.type != 2 && source.type != 3 && source.type != 8)
        throw DecodeError("native affine packing supports Q4_0/Q4_1/Q8_0 only");
    const auto &type = type_info(source.type);
    if (!source.rows || !source.columns || source.columns % 32) throw DecodeError("invalid affine source geometry");
    const uint64_t groups = checked_mul(source.rows, source.columns / 32);
    const uint32_t bits = source.type == 8 ? 8 : 4;
    const uint64_t bytes = part == AffinePart::codes ? checked_mul(groups, bits * 4) : checked_mul(groups, 2);
    if (source.bytes.size() < checked_mul(groups,type.bytes) || target.size() < bytes)
        throw DecodeError("affine source/target too short");
    const uintptr_t begin=reinterpret_cast<uintptr_t>(source.bytes.data()), end=reinterpret_cast<uintptr_t>(target.data());
    if (!begin || !end || source.bytes.size()>UINTPTR_MAX-begin || bytes>UINTPTR_MAX-end ||
        !(begin+source.bytes.size()<=end || end+bytes<=begin)) throw DecodeError("affine source overlaps target");
    const auto *input=reinterpret_cast<const uint8_t *>(source.bytes.data());
    for(uint64_t group=0;group<groups;++group) {
        if(cancel && cancel->load(std::memory_order_acquire)) throw DecodeError("gguf_decode_cancelled");
        const uint8_t *p=input+group*type.bytes;
        const uint16_t scale_bits=uint16_t(p[0])|uint16_t(p[1])<<8;
        const float scale=fp16_to_float(scale_bits);
        if(!std::isfinite(scale)) throw DecodeError("nonfinite affine scale");
        if(part==AffinePart::scales) { std::memcpy(target.data()+group*2,&scale_bits,2); continue; }
        if(part==AffinePart::biases) {
            const uint16_t bias=source.type==3 ? (uint16_t(p[2])|uint16_t(p[3])<<8)
                : float_to_fp16_rne(scale*(source.type==8 ? -128.f : -8.f));
            if(!std::isfinite(fp16_to_float(bias))) throw DecodeError("nonfinite affine bias");
            std::memcpy(target.data()+group*2,&bias,2);continue;
        }
        if(bits==8) {
            for(uint32_t i=0;i<32;++i) target[group*32+i]=std::byte(p[2+i]^0x80);
        } else {
            const uint8_t *q=p+(source.type==3 ? 4 : 2);
            for(uint32_t word=0;word<4;++word) {
                uint32_t packed=0;
                for(uint32_t i=0;i<8;++i) {
                    const uint32_t k=word*8+i;
                    const uint32_t code=k<16 ? q[k]&15 : q[k-16]>>4;
                    packed|=code<<(i*4);
                }
                std::memcpy(target.data()+group*16+word*4,&packed,4);
            }
        }
    }
    if(cancel && cancel->load(std::memory_order_acquire)) throw DecodeError("gguf_decode_cancelled");
    return bytes;
}

namespace {
// Multiplying a finite half by -2^shift is exact whenever representable.
// Handle subnormals in integer units of 2^-24, including signed zero; never
// flush, clamp, change rounding mode, or convert the whole weight to dense.
uint16_t negative_power2_half(uint16_t scale,unsigned shift) {
    const uint16_t magnitude=scale&0x7fff,sign=(scale^0x8000)&0x8000;
    if (magnitude>=0x7c00-shift*1024) throw DecodeError("FP16 conversion overflow");
    if (magnitude>=0x400) return sign|uint16_t(magnitude+shift*1024);
    const uint32_t units=uint32_t(magnitude)<<shift;
    if (units<1024) return sign|uint16_t(units);
    const unsigned exponent=unsigned(31-std::countl_zero(units))-9;
    return sign|uint16_t((exponent<<10)|((units>>(exponent-1))&1023));
}
void check_cancel(const std::atomic<bool> *cancel) {
    if (cancel && cancel->load(std::memory_order_acquire)) throw DecodeError("gguf_decode_cancelled");
}
}

uint64_t pack_native_affine_all(const PackedMatrix &source,const std::array<std::span<std::byte>,3> &target,
                               const std::atomic<bool> *cancel,DecodeOptions options) {
    if (source.type!=2 && source.type!=3 && source.type!=8)
        throw DecodeError("native affine packing supports Q4_0/Q4_1/Q8_0 only");
    if (!source.rows || !source.columns || source.columns%32) throw DecodeError("invalid affine source geometry");
    const auto &type=type_info(source.type);
    const uint64_t groups=checked_mul(source.rows,source.columns/32),bits=source.type==8 ? 8 : 4;
    const std::array<uint64_t,3> bytes{checked_mul(groups,bits*4),checked_mul(groups,2),checked_mul(groups,2)};
    if (source.bytes.size()<checked_mul(groups,type.bytes)) throw DecodeError("affine source/target too short");
    const auto begin=reinterpret_cast<uintptr_t>(source.bytes.data());
    if (!begin || source.bytes.size()>UINTPTR_MAX-begin) throw DecodeError("invalid affine source span");
    std::array<uintptr_t,3> starts{};
    for (size_t i=0;i<3;++i) {
        starts[i]=reinterpret_cast<uintptr_t>(target[i].data());
        if (!starts[i] || target[i].size()<bytes[i] || bytes[i]>UINTPTR_MAX-starts[i])
            throw DecodeError("affine source/target too short");
        if (!(begin+source.bytes.size()<=starts[i] || starts[i]+bytes[i]<=begin))
            throw DecodeError("affine source overlaps target");
        for (size_t j=0;j<i;++j) if (!(starts[j]+bytes[j]<=starts[i] || starts[i]+bytes[i]<=starts[j]))
            throw DecodeError("affine targets overlap");
    }
    check_cancel(cancel);
    const auto *input=reinterpret_cast<const uint8_t *>(source.bytes.data());
    for (uint64_t group=0;group<groups;++group) {
        // At most 256 blocks (8.5 KiB for Q8) between cancellation checks.
        if (!(group%256)) check_cancel(cancel);
        const auto *p=input+group*type.bytes;
        const uint16_t scale=uint16_t(p[0])|uint16_t(p[1])<<8;
        if ((scale&0x7c00)==0x7c00) throw DecodeError("nonfinite affine scale");
        const uint16_t bias=source.type==3 ? uint16_t(p[2])|uint16_t(p[3])<<8
            : negative_power2_half(scale,source.type==8 ? 7 : 3);
        if ((bias&0x7c00)==0x7c00) throw DecodeError("nonfinite affine bias");
        std::memcpy(target[1].data()+group*2,&scale,2);
        std::memcpy(target[2].data()+group*2,&bias,2);
        auto *codes=reinterpret_cast<uint8_t *>(target[0].data()+group*bits*4);
#if defined(__aarch64__)
        if (options.use_simd) {
            if (source.type==8) {
                const auto sign=vdupq_n_u8(0x80);
                vst1q_u8(codes,veorq_u8(vld1q_u8(p+2),sign));
                vst1q_u8(codes+16,veorq_u8(vld1q_u8(p+18),sign));
            } else {
                const auto q=vld1q_u8(p+(source.type==3 ? 4 : 2));
                const auto even=vuzp1q_u8(q,q),odd=vuzp2q_u8(q,q);
                const auto low=vorrq_u8(vandq_u8(even,vdupq_n_u8(15)),vshlq_n_u8(odd,4));
                const auto high=vorrq_u8(vshrq_n_u8(even,4),vandq_u8(odd,vdupq_n_u8(0xf0)));
                vst1q_u8(codes,vcombine_u8(vget_low_u8(low),vget_low_u8(high)));
            }
            continue;
        }
#else
        (void)options;
#endif
        if (source.type==8) for (uint32_t i=0;i<32;++i) codes[i]=p[2+i]^0x80;
        else {
            const auto *q=p+(source.type==3 ? 4 : 2);
            for (uint32_t word=0;word<4;++word) {
                uint32_t packed=0;
                for (uint32_t i=0;i<8;++i) {
                    const uint32_t k=word*8+i,code=k<16 ? q[k]&15 : q[k-16]>>4;
                    packed|=code<<(i*4);
                }
                std::memcpy(codes+word*4,&packed,4);
            }
        }
    }
    check_cancel(cancel);
    return checked_add(checked_add(bytes[0],bytes[1]),bytes[2]);
}
} // namespace tc::gguf

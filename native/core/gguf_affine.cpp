#include "gguf_affine.hpp"
#include <cstring>
#include <cmath>

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
} // namespace tc::gguf

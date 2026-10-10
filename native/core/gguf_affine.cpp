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

uint64_t pack_k_affine_all(const PackedMatrix &source,const std::array<std::span<std::byte>,3> &target,
                          const std::atomic<bool> *cancel,DecodeDType metadata,DecodeOptions options) {
    if(metadata!=DecodeDType::f16 && metadata!=DecodeDType::bf16)
        throw DecodeError("K affine metadata requires FP16 or BF16");
    if(source.type!=12 && source.type!=13 && source.type!=14)
        throw DecodeError("K affine packing supports Q4_K/Q5_K/Q6_K only");
    if(!source.rows || !source.columns || source.columns%256)
        throw DecodeError("invalid K affine source geometry");
    const auto &type=type_info(source.type);
    const uint64_t blocks=checked_mul(source.rows,source.columns/256);
    const uint64_t groups=checked_mul(blocks,8),bits=source.type==12 ? 4 : 8;
    const std::array<uint64_t,3> bytes{checked_mul(groups,bits*4),checked_mul(groups,2),checked_mul(groups,2)};
    if(source.bytes.size()<checked_mul(blocks,type.bytes)) throw DecodeError("K affine source too short");
    const uintptr_t begin=reinterpret_cast<uintptr_t>(source.bytes.data());
    if(!begin || source.bytes.size()>UINTPTR_MAX-begin) throw DecodeError("invalid K affine source span");
    std::array<uintptr_t,3> starts{};
    for(size_t i=0;i<3;++i) {
        starts[i]=reinterpret_cast<uintptr_t>(target[i].data());
        if(!starts[i] || target[i].size()<bytes[i] || bytes[i]>UINTPTR_MAX-starts[i])
            throw DecodeError("K affine target too short");
        if(!(begin+source.bytes.size()<=starts[i] || starts[i]+bytes[i]<=begin))
            throw DecodeError("K affine source overlaps target");
        for(size_t j=0;j<i;++j) if(!(starts[j]+bytes[j]<=starts[i] || starts[i]+bytes[i]<=starts[j]))
            throw DecodeError("K affine targets overlap");
    }
    check_cancel(cancel);
    const auto *input=reinterpret_cast<const uint8_t *>(source.bytes.data());
    auto half=[](const uint8_t *p) {return fp16_to_float(uint16_t(p[0])|uint16_t(p[1])<<8);};
    auto store_half=[metadata](std::byte *p,float value) {
        if(!std::isfinite(value)) throw DecodeError("nonfinite K affine metadata");
        const uint16_t packed=metadata==DecodeDType::f16 ? float_to_fp16_rne(value) : float_to_bf16_rne(value);
        const float decoded=metadata==DecodeDType::f16 ? fp16_to_float(packed) : std::bit_cast<float>(uint32_t(packed)<<16);
        if(!std::isfinite(decoded)) throw DecodeError("K affine typed metadata overflow");
        std::memcpy(p,&packed,2);return decoded;
    };
    for(uint64_t block=0;block<blocks;++block) {
        if(!(block%128))check_cancel(cancel);
        const auto *p=input+block*type.bytes;
        const float d=half(p+(source.type==14 ? 208 : 0));
        const float minimum=source.type!=14 ? half(p+2) : 0.f;
        if(!std::isfinite(d) || !std::isfinite(minimum)) throw DecodeError("nonfinite K affine source scale");
        for(uint32_t group=0;group<8;++group) {
            const auto index=block*8+group;
            auto *codes=target[0].data()+index*bits*4;
            if(source.type!=14) {
                const auto *scales=p+4;
                const uint8_t scale=group<4 ? scales[group]&63 :
                    (scales[group+4]&15)|((scales[group-4]>>6)<<4);
                const uint8_t bias=group<4 ? scales[group+4]&63 :
                    (scales[group+4]>>4)|((scales[group]>>6)<<4);
                store_half(target[1].data()+index*2,d*scale);
                store_half(target[2].data()+index*2,-minimum*bias);
                const auto *q=p+(source.type==13 ? 48 : 16)+(group/2)*32;
#if defined(__aarch64__)
                if(options.use_simd) {
                    const auto first=vld1q_u8(q),second=vld1q_u8(q+16);
                    if(source.type==12) {
                        const auto even=vuzp1q_u8(first,second),odd=vuzp2q_u8(first,second);
                        const auto packed=(group&1) ? vorrq_u8(vshrq_n_u8(even,4),vandq_u8(odd,vdupq_n_u8(0xf0))) :
                            vorrq_u8(vandq_u8(even,vdupq_n_u8(15)),vshlq_n_u8(odd,4));
                        vst1q_u8(reinterpret_cast<uint8_t *>(codes),packed);
                    } else {
                        const auto mask=vdupq_n_u8(uint8_t(1)<<group);
                        const auto shift=vdupq_n_s8(int8_t(4-int(group)));
                        const auto low0=(group&1) ? vshrq_n_u8(first,4) : vandq_u8(first,vdupq_n_u8(15));
                        const auto low1=(group&1) ? vshrq_n_u8(second,4) : vandq_u8(second,vdupq_n_u8(15));
                        vst1q_u8(reinterpret_cast<uint8_t *>(codes),vorrq_u8(low0,vshlq_u8(vandq_u8(vld1q_u8(p+16),mask),shift)));
                        vst1q_u8(reinterpret_cast<uint8_t *>(codes)+16,vorrq_u8(low1,vshlq_u8(vandq_u8(vld1q_u8(p+32),mask),shift)));
                    }
                    continue;
                }
#else
                (void)options;
#endif
                if(source.type==13) {
                    for(uint32_t j=0;j<32;++j) {
                        const uint8_t low=(group&1) ? q[j]>>4 : q[j]&15;
                        codes[j]=std::byte(low|((p[16+j]&(uint8_t(1)<<group)) ? 16 : 0));
                    }
                    continue;
                }
                for(uint32_t word=0;word<4;++word) {
                    uint32_t packed=0;
                    for(uint32_t j=0;j<8;++j) {
                        const uint8_t code=(group&1) ? q[word*8+j]>>4 : q[word*8+j]&15;
                        packed|=uint32_t(code)<<(j*4);
                    }
                    std::memcpy(codes+word*4,&packed,4);
                }
            } else {
                // At most 128 bytes of dense stack scratch, never a row/model.
                std::array<float,32> values{};float maximum=0.f;
                const uint32_t half_index=group/4,local=group%4;
                const auto *low=p+half_index*64,*high=p+128+half_index*32;
                const auto *scales=p+192+half_index*8;
#if defined(__aarch64__)
                if(options.use_simd) {
                    for(uint32_t half=0;half<2;++half) {
                        const auto q=vld1q_u8(low+half*16+(local&1)*32);
                        const auto qh=vld1q_u8(high+half*16);
                        const auto low_codes=local<2 ? vandq_u8(q,vdupq_n_u8(15)) : vshrq_n_u8(q,4);
                        const auto high_codes=vshlq_n_u8(vandq_u8(vshlq_u8(qh,vdupq_n_s8(-int8_t(2*local))),vdupq_n_u8(3)),4);
                        const auto signed_codes=vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(low_codes,high_codes),vdupq_n_u8(32)));
                        const uint8_t raw_scale=scales[half+local*2];
                        const int signed_scale=raw_scale<128 ? int(raw_scale) : int(raw_scale)-256;
                        const auto multiplier=vdupq_n_f32(d*float(signed_scale));
                        const auto lo=vmovl_s8(vget_low_s8(signed_codes)),hi=vmovl_s8(vget_high_s8(signed_codes));
                        const int16x4_t lanes[]{vget_low_s16(lo),vget_high_s16(lo),vget_low_s16(hi),vget_high_s16(hi)};
                        for(uint32_t lane=0;lane<4;++lane) {
                            const auto decoded=vmulq_f32(vcvtq_f32_s32(vmovl_s16(lanes[lane])),multiplier);
                            vst1q_f32(values.data()+half*16+lane*4,decoded);
                            maximum=std::max(maximum,vmaxvq_f32(vabsq_f32(decoded)));
                        }
                    }
                    // Finite half d, signed-byte scales and bounded six-bit
                    // codes cannot overflow F32; max magnitude is <2^29.
                } else
#endif
                for(uint32_t j=0;j<32;++j) {
                    const uint8_t q=low[j+(local&1)*32];
                    const int code=int((local<2 ? q&15 : q>>4)|(((high[j]>>(2*local))&3)<<4))-32;
                    const uint8_t raw_scale=scales[j/16+local*2];
                    const int scale=raw_scale<128 ? int(raw_scale) : int(raw_scale)-256;
                    values[j]=(d*float(scale))*float(code);
                    if(!std::isfinite(values[j])) throw DecodeError("nonfinite K affine decoded source");
                    maximum=std::max(maximum,std::abs(values[j]));
                }
                const float scale=store_half(target[1].data()+index*2,maximum/127.f);
                const float precision=metadata==DecodeDType::f16 ? .002f : .005f;
                if(maximum>0 && (scale==0 || std::abs(scale-maximum/127.f)/(maximum/127.f)>precision))
                    throw DecodeError("Q6_K affine scale precision insufficient; no publication");
                store_half(target[2].data()+index*2,-128.f*scale);
#if defined(__aarch64__)
                if(options.use_simd) {
                    for(uint32_t half=0;half<2;++half) {
                        uint16x4_t narrowed[4];
                        for(uint32_t lane=0;lane<4;++lane) {
                            const auto v=vld1q_f32(values.data()+half*16+lane*4);
                            const auto shifted=scale==0 ? vdupq_n_f32(128.f) : vaddq_f32(vdivq_f32(v,vdupq_n_f32(scale)),vdupq_n_f32(128.f));
                            narrowed[lane]=vqmovun_s32(vcvtnq_s32_f32(shifted));
                        }
                        const auto lo=vqmovn_u16(vcombine_u16(narrowed[0],narrowed[1]));
                        const auto hi=vqmovn_u16(vcombine_u16(narrowed[2],narrowed[3]));
                        vst1q_u8(reinterpret_cast<uint8_t *>(codes)+half*16,vcombine_u8(lo,hi));
                    }
                    continue;
                }
#endif
                // Explicit ties-to-even, independent of process rounding mode.
                for(uint32_t j=0;j<32;++j) {
                    const float value=scale==0 ? 128.f : values[j]/scale+128.f;
                    const float floor=std::floor(value),fraction=value-floor;
                    int code=int(floor)+(fraction>.5f || (fraction==.5f && (int(floor)&1)));
                    codes[j]=std::byte(std::clamp(code,0,255));
                }
            }
        }
    }
    check_cancel(cancel);
    return checked_add(checked_add(bytes[0],bytes[1]),bytes[2]);
}
} // namespace tc::gguf

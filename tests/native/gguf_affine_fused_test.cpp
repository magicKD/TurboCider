#include "gguf_affine.hpp"
#include <cstring>
#include <iostream>
#include <vector>

using namespace tc::gguf;
void insist(bool value,const char *message) {if (!value) throw std::runtime_error(message);}
using Fields=std::array<std::vector<std::byte>,3>;
Fields fields(uint64_t groups,uint32_t type) {
    return {std::vector<std::byte>(groups*(type==8 ? 32 : 16)),
        std::vector<std::byte>(groups*2),std::vector<std::byte>(groups*2)};
}
std::array<std::span<std::byte>,3> spans(Fields &f) {return {f[0],f[1],f[2]};}
template<class F> bool accepted(F function) {try {function();return true;} catch (const std::exception &) {return false;}}
int main() {
    try {
        for (uint32_t type:{2u,3u,8u}) {
            const auto bytes=type_info(type).bytes;
            // Preserve the pre-optimization API as an independent scalar
            // oracle. Exhaustive bit patterns test finite, signed zero,
            // subnormal normalization and overflow, not just typical scales.
            for (uint32_t bits=0;bits<65536;++bits) {
                std::vector<std::byte> raw(bytes,std::byte(0x53));const uint16_t scale=uint16_t(bits),bias=0x3400;
                std::memcpy(raw.data(),&scale,2);if(type==3)std::memcpy(raw.data()+2,&bias,2);
                PackedMatrix source{raw,type,1,32};auto expected=fields(1,type);
                const bool legacy=accepted([&]{for(size_t part=0;part<3;++part)pack_native_affine(source,AffinePart(part),expected[part]);});
                for (bool simd:{false,true}) {
                    auto actual=fields(1,type);
                    const bool fused=accepted([&]{pack_native_affine_all(source,spans(actual),nullptr,{simd});});
                    insist(legacy==fused,"fused finite/overflow admission differs from legacy scalar");
                    if (legacy) insist(actual==expected,"fused half metadata/code bit patterns differ");
                }
            }
            if (type==3) for (uint32_t bits=0;bits<65536;++bits) {
                std::vector<std::byte> raw(bytes,std::byte(0x95));const uint16_t scale=0x3000,bias=uint16_t(bits);
                std::memcpy(raw.data(),&scale,2);std::memcpy(raw.data()+2,&bias,2);
                PackedMatrix source{raw,type,1,32};auto expected=fields(1,type),actual=fields(1,type);
                const bool legacy=accepted([&]{for(size_t part=0;part<3;++part)pack_native_affine(source,AffinePart(part),expected[part]);});
                const bool fused=accepted([&]{pack_native_affine_all(source,spans(actual));});
                insist(legacy==fused && (!legacy || expected==actual),"Q4_1 source bias identity/admission changed");
            }
            for (uint64_t rows:{1u,7u,33u,257u}) for(uint64_t columns:{32u,64u,256u}) {
                const uint64_t groups=rows*columns/32;
                std::vector<std::byte> backing(groups*bytes+1);auto raw=std::span<std::byte>(backing).subspan(1);
                for (uint64_t g=0;g<groups;++g) {
                    const uint16_t scale=uint16_t((g%23)*91),bias=uint16_t(0xb400+g%17);
                    std::memcpy(raw.data()+g*bytes,&scale,2);if(type==3)std::memcpy(raw.data()+g*bytes+2,&bias,2);
                    for (uint64_t i=type==3 ? 4 : 2;i<bytes;++i) raw[g*bytes+i]=std::byte((g*13+i*37)%256);
                }
                PackedMatrix source{raw,type,rows,columns};auto expected=fields(groups,type);
                for(size_t part=0;part<3;++part)pack_native_affine(source,AffinePart(part),expected[part]);
                for(bool simd:{false,true}) {
                    std::array<std::vector<std::byte>,3> guards;
                    std::array<std::span<std::byte>,3> target;
                    for (size_t part=0;part<3;++part) {guards[part].assign(expected[part].size()+13,std::byte(0x6a));target[part]={guards[part].data()+3,expected[part].size()};}
                    const auto written=pack_native_affine_all(source,target,nullptr,{simd});
                    insist(written==expected[0].size()+expected[1].size()+expected[2].size(),"fused receipt byte count changed");
                    for(size_t part=0;part<3;++part) {
                        insist(std::memcmp(target[part].data(),expected[part].data(),expected[part].size())==0,"unaligned multirow SIMD code permutation changed");
                        for (size_t i=0;i<guards[part].size();++i) if(i<3 || i>=3+expected[part].size())
                            insist(guards[part][i]==std::byte(0x6a),"fused packing wrote target guard bytes");
                    }
                }
            }
        }
        std::vector<std::byte> raw(34);const uint16_t scale=0x3000;std::memcpy(raw.data(),&scale,2);
        PackedMatrix source{raw,8,1,32};auto output=fields(1,8);const auto untouched=output;
        auto target=spans(output);
        std::atomic<bool> cancel{true};insist(!accepted([&]{pack_native_affine_all(source,target,&cancel);}),"cancelled packing accepted");
        insist(output==untouched,"pre-cancelled packing wrote outputs");
        auto short_target=target;short_target[0]=short_target[0].first(31);
        insist(!accepted([&]{pack_native_affine_all(source,short_target);}),"short codes target accepted");
        auto overlap=target;overlap[1]={output[0].data()+1,2};
        insist(!accepted([&]{pack_native_affine_all(source,overlap);}),"overlapping output fields accepted");
        auto source_alias=target;source_alias[0]=raw;
        insist(!accepted([&]{pack_native_affine_all(source,source_alias);}),"source/output alias accepted");
        for(uint32_t type:{0u,12u}) {
            auto bad=source;bad.type=type;insist(!accepted([&]{pack_native_affine_all(bad,target);}),"unsupported type admitted");
        }
        auto bad=source;bad.columns=31;insist(!accepted([&]{pack_native_affine_all(bad,target);}),"invalid block tail admitted");
        std::cout<<"PASS fused affine: exhaustive 65536 scales per Q4/Q8 type and Q4_1 biases; scalar/SIMD, 36 geometries, unaligned/guards, alias/capacity/type/cancel negatives\n";
    } catch (const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}

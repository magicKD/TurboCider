#include "gguf_affine.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace tc::gguf;
using Fields=std::array<std::vector<std::byte>,3>;
void insist(bool valid,const char *message) {if(!valid)throw std::runtime_error(message);}
template<class F> bool rejected(F fn) {try{fn();return false;}catch(const std::exception &){return true;}}
std::array<std::span<std::byte>,3> spans(Fields &f) {return {f[0],f[1],f[2]};}

int main() {try {
    int cases=0;
    for(bool simd:{false,true})for(auto metadata:{DecodeDType::f16,DecodeDType::bf16})
    for(uint32_t type:{12u,13u,14u})for(uint64_t rows:{1u,3u,17u})for(uint64_t columns:{256u,512u,1024u}) {
        const uint64_t blocks=rows*columns/256,groups=blocks*8,bits=type==12 ? 4 : 8;
        std::vector<std::byte> backing(blocks*type_info(type).bytes+1);
        auto raw=std::span<std::byte>(backing).subspan(1);
        for(uint64_t b=0;b<blocks;++b) {
            auto *p=raw.data()+b*type_info(type).bytes;
            for(uint32_t i=0;i<type_info(type).bytes;++i)p[i]=std::byte((b*41+i*17)%256);
            const uint16_t d=float_to_fp16_rne(.00031f*float(b%11+1)),minimum=float_to_fp16_rne(.00017f);
            std::memcpy(p+(type==14 ? 208 : 0),&d,2);
            if(type!=14)std::memcpy(p+2,&minimum,2);
        }
        PackedMatrix source{raw,type,rows,columns};
        Fields f{std::vector<std::byte>(groups*bits*4),std::vector<std::byte>(groups*2),std::vector<std::byte>(groups*2)};
        insist(pack_k_affine_all(source,spans(f),nullptr,metadata,{simd})==f[0].size()+f[1].size()+f[2].size(),"K affine byte receipt differs");
        auto scalar=f;
        pack_k_affine_all(source,spans(scalar),nullptr,metadata,{false});
        insist(scalar==f,"K scalar/SIMD coefficient or code bytes differ");
        std::vector<float> expected(rows*columns);
        decode_cpu_into(source,{0,rows,0,columns},{{reinterpret_cast<std::byte *>(expected.data()),expected.size()*4},DecodeDType::f32,columns*4,4});
        double error=0,energy=0;
        for(uint64_t i=0;i<rows*columns;++i) {
            const uint64_t group=i/32,j=i%32;
            uint16_t s,m;std::memcpy(&s,f[1].data()+group*2,2);std::memcpy(&m,f[2].data()+group*2,2);
            const auto *codes=reinterpret_cast<const uint8_t *>(f[0].data());
            const uint32_t q=bits==8 ? codes[i] : (codes[group*16+j/2]>>((j%2)*4))&15;
            auto scalar=[metadata](uint16_t b) {return metadata==DecodeDType::f16 ? fp16_to_float(b) : std::bit_cast<float>(uint32_t(b)<<16);};
            const float value=scalar(s)*float(q)+scalar(m);
            insist(std::isfinite(value),"K affine published nonfinite coefficient/value");
            error+=double(value-expected[i])*double(value-expected[i]);energy+=double(expected[i])*double(expected[i]);
        }
        const double budget=type==14 ? .02 : metadata==DecodeDType::f16 ? .002 : .01;
        insist(std::sqrt(error/std::max(energy,1e-30))<budget,"K affine source approximation exceeded disclosed budget");
        auto target=spans(f);auto before=f;
        std::atomic<bool> cancel{true};insist(rejected([&]{pack_k_affine_all(source,target,&cancel);}) && before==f,"pre-cancelled K pack wrote outputs");
        auto short_target=target;short_target[0]=short_target[0].first(short_target[0].size()-1);
        insist(rejected([&]{pack_k_affine_all(source,short_target);}),"short K codes accepted");
        auto alias=target;alias[1]={target[0].data()+1,target[1].size()};
        insist(rejected([&]{pack_k_affine_all(source,alias);}),"aliased K fields accepted");
        auto malformed=source;malformed.columns=columns-1;
        insist(rejected([&]{pack_k_affine_all(malformed,target);}),"K tail accepted");
        const uint16_t infinity=0x7c00;std::memcpy(raw.data()+(type==14 ? 208 : 0),&infinity,2);
        insist(rejected([&]{pack_k_affine_all(source,target);}),"nonfinite K scale accepted");
        ++cases;
    }
    std::cout<<"PASS 108 Q4_K/Q5_K/Q6_K typed affine cases: independent F32 decoder, scalar/SIMD, unaligned source, finite/alias/size/tail/cancel guards\n";
    insist(cases==108,"wrong K affine numeric case count");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}

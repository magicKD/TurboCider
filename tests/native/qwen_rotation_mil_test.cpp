#include "../../tools/profiling/qwen_base/rotation_mil.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include "../../native/backends/ane_w8a8_math.hpp"
#include <cassert>
#include <iostream>

template<class T>T read(const std::vector<uint8_t>&v,size_t p){T x;assert(p+sizeof(T)<=v.size());std::memcpy(&x,v.data()+p,sizeof(T));return x;}
int main(){
    namespace lab=tc::qwen_rotation_lab;using namespace tc::ane;
    for(int c:{512,6144})for(int m:{32,1024}) {
        const auto full=lab::program(c,m,512);
        const auto original=private_api::w8_swiglu_program({Kind::SwiGLU,m,4096,c,4096,512,false},20260930,1.f);
        assert(full.constants==original.constants);
        for(int policy:{16,32,64,128,-32,-64,10032,10064}){
            const bool flat=policy>10000;const int b=flat?policy-10000:std::abs(policy);
            const int a=512/b;auto p=lab::program(c,m,policy);
            assert(read<uint32_t>(p.constants,0)==2&&read<uint32_t>(p.constants,4)==2);
            const auto first=read<uint64_t>(p.constants,80),second_record=128+uint64_t(c)*b*2;
            assert(read<uint32_t>(p.constants,second_record)==0xdeadbeef);
            const auto second=read<uint64_t>(p.constants,second_record+16);
            assert(second+uint64_t(flat?c/b:c)*a*2==p.constants.size());
            for(int out=0;out<512;++out)for(int in=0;in<512;++in){
                const int outer_out=out/b,inner_out=out%b,outer_in=in/b,inner_in=in%b;
                const auto f=std::bit_cast<_Float16>(read<uint16_t>(p.constants,first+(uint64_t(outer_in*b+inner_out)*b+inner_in)*2));
                const auto g=std::bit_cast<_Float16>(read<uint16_t>(p.constants,second+(uint64_t(flat?outer_out:inner_out*a+outer_out)*a+outer_in)*2));
                const auto expected=std::bit_cast<_Float16>(read<uint16_t>(full.constants,128+(uint64_t(out)*512+in)*2));
                assert(float(f)*float(g)==float(expected));
            }
            assert((p.mil.find("perm = tensor<int32, [4]>([0, 2, 1, 3])")!=std::string::npos)!=flat);
        }
    }
    for(int i=0;i<512;++i)assert(lab::sign(20260930,i)==rotation_sign(20260930,i));
    for(const auto v:{std::array{511,32,64},std::array{6144,33,64},std::array{6144,1024,256},std::array{6144,1024,-1}}){
        bool bad=false;try{(void)lab::program(v[0],v[1],v[2]);}catch(const std::invalid_argument&){bad=true;}assert(bad);
    }
    std::cout<<"PASS: original rotation constants exact; two-factor serialized coefficient product/permutation/domain\n";
}

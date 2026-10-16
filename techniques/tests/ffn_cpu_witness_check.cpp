// Portable unit fixture, not a device/model benchmark.
#include "../labs/ffn_cpu_witness.hpp"
#include <iomanip>
#include <iostream>
uint16_t code(float value){uint32_t bits=0;std::memcpy(&bits,&value,4);if(bits&0xffff)throw std::runtime_error("exact BF16 fixture required");return uint16_t(bits>>16);}
int main(){
    std::vector<uint16_t>x(16),gu(40),down(20);
    for(size_t i=0;i<x.size();++i)x[i]=code(float(int(i%9)-4)/8);
    for(size_t i=0;i<gu.size();++i)gu[i]=code(float(int(i%7)-3)/16);
    for(size_t i=0;i<down.size();++i)down[i]=code(float(int(i%5)-2)/8);
    auto full=tc_witness::selected(x,gu,down,4,4,5,0,5),prefix=tc_witness::selected(x,gu,down,4,4,5,0,2);
    std::cout<<std::setprecision(17)<<"{\"full\":[";
    for(size_t i=0;i<16;++i)std::cout<<(i?",":"")<<full.output[i];
    std::cout<<"],\"prefix\":[";for(size_t i=0;i<16;++i)std::cout<<(i?",":"")<<prefix.output[i];
    std::cout<<"]}\n";
    try{(void)tc_witness::selected(x,gu,down,4,4,5,0,6);return 2;}catch(const std::invalid_argument&){}
    try{(void)tc_witness::bf16(0x7f80);return 3;}catch(const std::invalid_argument&){}
}

// Actual driver graph ablation, no checkpoint or inference default mutation.
#include "../../native/backends/private/ane_program.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include "../../native/core/gguf_decode.hpp"
#include "private_ane_qdq_share.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>

using namespace tc::ane;
using namespace tc::ane::private_api;
namespace {
uint32_t random_word(uint32_t x) {x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;return x^(x>>16);}
void fill_codes(Surface &surface,int seed) {
    std::memset(surface.data(),0,surface.rows()*surface.pitch());
    for(uint32_t row=0;row<surface.rows();++row)for(uint32_t col=0;col<surface.columns();++col)
        static_cast<int8_t*>(surface.data())[size_t(row)*surface.pitch()+col]=
            int8_t(int(random_word(row*surface.columns()+col+seed)%63)-31);
}
void fill_half(Surface &surface,float value) {
    std::memset(surface.data(),0,surface.rows()*surface.pitch());
    const auto bits=tc::gguf::float_to_fp16_rne(value);
    for(uint32_t row=0;row<surface.rows();++row)for(uint32_t col=0;col<surface.columns();++col)
        reinterpret_cast<uint16_t*>(static_cast<char*>(surface.data())+size_t(row)*surface.pitch())[col]=bits;
}
double median(std::vector<double> values) {std::sort(values.begin(),values.end());return values[values.size()/2];}
void samples(const std::vector<double> &values) {
    std::cout<<'[';for(size_t i=0;i<values.size();++i)std::cout<<(i?",":"")<<values[i];std::cout<<']';
}
}
int main(int argc,char **argv) {
    if(argc!=3)return 2;
    try {
        const int repeats=std::stoi(argv[2]);if(repeats<7 || repeats>101 || repeats%2==0)return 2;
        Device device;uint64_t timeline=0;
        for(int rows:{1056,4224})for(bool lora:{false,true}) {
            const int hidden=4096,width=rows==1056?5120:7168;
            const GraphShape shape{Kind::SwiGLU,rows,hidden,width,1024,512,lora};
            const auto emitted=w8_swiglu_program(shape,20260930,1.f);
            const auto shared=tc::ane::research::share_gate_up_qdq(emitted.mil);
            Program original(device,emitted.mil,emitted.constants,argv[1]);
            Program candidate(device,shared,emitted.constants,argv[1]);
            if(original.cache_key()==candidate.cache_key())throw std::runtime_error("graph variants share cache identity");
            Surface x(device,hidden,rows,Element::I8),tx(device,1,rows,Element::FP16),
                wg(device,width,hidden,Element::I8),sg(device,width,1,Element::FP16),
                wu(device,width,hidden,Element::I8),su(device,width,1,Element::FP16),
                wd(device,hidden,width,Element::I8),dg(device,width,rows,Element::FP16),du(device,width,rows,Element::FP16),
                y0(device,emitted.packed_rows,rows,Element::FP16),y1(device,emitted.packed_rows,rows,Element::FP16);
            fill_codes(x,7);fill_codes(wg,11);fill_codes(wu,13);fill_codes(wd,17);
            fill_half(tx,.25f);fill_half(sg,.125f);fill_half(su,.125f);fill_half(dg,.02f);fill_half(du,-.03f);
            std::memset(y0.data(),0x5a,y0.rows()*y0.pitch());std::memset(y1.data(),0x5a,y1.rows()*y1.pitch());
            std::vector<std::pair<std::string,Surface>> inputs{{"x",x},{"tx",tx},{"wg",wg},{"sg",sg},{"wu",wu},{"su",su},{"wd",wd}};
            if(lora){inputs.emplace_back("dg",dg);inputs.emplace_back("du",du);}
            auto run=[&](Program &program,Surface &output) {
                if(timeline>UINT64_MAX-2)throw std::runtime_error("benchmark timeline exhausted");
                const uint64_t ready=++timeline,done=++timeline;
                const std::pair<std::string,Surface> outputs[]{{"y",output}};
                auto prepared=program.prepare(inputs,outputs,ready,done);
                device.release_prepared(ready); // outside timer; no GPU handoff hidden in an alone sample
                const auto start=std::chrono::steady_clock::now();
                auto request=prepared.submit();const auto result=request.finish();
                const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                if(!result.ok)throw std::runtime_error(result.error);
                return seconds;
            };
            auto exact=[&] {
                if(std::memcmp(y0.data(),y1.data(),y0.rows()*y0.pitch()))
                    throw std::runtime_error("shared QDQ changed packed output/scales/hidden/padding");
                for(uint32_t row=0;row<y0.rows();++row)for(uint32_t col=0;col<y0.columns();++col) {
                    const uint16_t bits=reinterpret_cast<const uint16_t*>(static_cast<const char*>(y0.data())+row*y0.pitch())[col];
                    if((bits&0x7c00)==0x7c00)throw std::runtime_error("nonfinite shared/original output");
                }
            };
            for(int i=0;i<5;++i){run(original,y0);run(candidate,y1);}exact();
            std::vector<double> before,after;
            for(int i=0;i<repeats;++i) {
                if(i%2){after.push_back(run(candidate,y1));before.push_back(run(original,y0));}
                else {before.push_back(run(original,y0));after.push_back(run(candidate,y1));}
                exact();
            }
            std::cout<<std::setprecision(17)<<"{\"scope\":\"actual Private ANE graph host-submit/finish, frozen normalized inputs; no staging/restore/GPU/model/E2E qualification\",\"rows\":"<<rows
                <<",\"hidden\":"<<hidden<<",\"width\":"<<width<<",\"lora_inputs\":"<<(lora?"true":"false")
                <<",\"packed_output_scale_hidden_padding_exact\":true,\"warmups\":5,\"original_cache_key\":\""<<original.cache_key()
                <<"\",\"shared_cache_key\":\""<<candidate.cache_key()<<"\",\"original_median\":"<<median(before)
                <<",\"shared_median\":"<<median(after)<<",\"speedup\":"<<median(before)/median(after)<<",\"original_samples\":";
            samples(before);std::cout<<",\"shared_samples\":";samples(after);std::cout<<"}"<<std::endl;
        }
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}

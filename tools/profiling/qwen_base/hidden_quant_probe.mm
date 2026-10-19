#include "hidden_quant_mil.hpp"
#include "../../../native/backends/private/ane_program.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

#ifndef TC_HIDDEN_QUANT_CPP_SHA
#define TC_HIDDEN_QUANT_CPP_SHA "unbound"
#endif
#ifndef TC_HIDDEN_QUANT_HEADER_SHA
#define TC_HIDDEN_QUANT_HEADER_SHA "unbound"
#endif
#ifndef TC_HIDDEN_QUANT_ROTATION_SHA
#define TC_HIDDEN_QUANT_ROTATION_SHA "unbound"
#endif
#ifndef TC_HIDDEN_QUANT_NATIVE_ID
#define TC_HIDDEN_QUANT_NATIVE_ID "unbound"
#endif
namespace {
using namespace tc;namespace pa=ane::private_api;namespace lab=qwen_hidden_quant_lab;
const char *names[]{"divide","reciprocal-divide","reciprocal-inverse"};
const char *patterns[]{"random-a","zero","tiny","wide","overflow","random-a-restored"};
NSString *ns(const std::string&s){return [NSString stringWithUTF8String:s.c_str()];}
void save(const std::filesystem::path&p,const std::vector<_Float16>&v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*2);require(bool(f),"hidden quant array write failed");}
std::string hash(const std::vector<_Float16>&v){return memory_sha256_hex({reinterpret_cast<const char*>(v.data()),v.size()*2});}
std::vector<_Float16> input(int channels,int rows,int pattern) {
    std::vector<_Float16> v(size_t(channels)*rows);
    for(int c=0;c<channels;++c)for(int r=0;r<rows;++r) {
        const uint32_t index=uint32_t(c)*rows+r;
        uint32_t q=index*747796405u+2891336453u;q=((q>>((q>>28)+4))^q)*277803737u;q=(q>>22)^q;
        float x=(int(q&65535)-32768)/32768.f;
        if(pattern==1)x=0;
        else if(pattern==2)x=float((c*7+r*3)%17-8)*0x1p-24f;
        else if(pattern==4)x=float(qwen_rotation_lab::sign(20260930,c%512))*4096.f;
        else x*=pattern==3?16.f:.125f;
        v[size_t(c)*rows+r]=_Float16(x==0?0:x);
    }
    return v;
}
struct Arm {
    pa::Surface y;std::unique_ptr<pa::Program> program;std::string first;
    Arm(pa::Device&d,int c,int m):y(d,c+1,m,pa::Element::FP16){}
};
}
extern "C" __attribute__((visibility("default")))
int tc_hidden_quant_probe(const char *output,int repeats,void(*scope)(int)) {@autoreleasepool{
    const std::filesystem::path out(output?output:"");bool fresh=false;
    NSMutableDictionary *report=[@{@"schema":@"tc-hidden-quant-probe-v1",@"status":@"starting",@"synthetic_only":@YES,
        @"model_weights_loaded":@NO,@"complete_request":@NO,@"image_export":@NO,@"cpp_sha256":@TC_HIDDEN_QUANT_CPP_SHA,
        @"header_sha256":@TC_HIDDEN_QUANT_HEADER_SHA,@"compiled_native_id":@TC_HIDDEN_QUANT_NATIVE_ID,
        @"rotation_header_sha256":@TC_HIDDEN_QUANT_ROTATION_SHA,
        @"repeats":@(repeats),@"warmups":@2,@"dtype":@"float16-le",@"layout":@"channels_tokens_packed_scale"} mutableCopy];
    try {
        require(output&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"fresh hidden quant output required");fresh=true;
        require(repeats==12||repeats==24,"bounded hidden quant repetitions required");
        char *raw=tc_runtime_build_identity();require(raw,"native ID missing");std::string id(raw);tc_string_free(raw);
        report[@"loaded_runtime_id"]=ns(id);require(id==TC_HIDDEN_QUANT_NATIVE_ID,"hidden quant helper/native ABI mismatch");
        pa::Device device(false,false,0);uint64_t timeline=0;NSMutableArray *programs=[NSMutableArray array],*arrays=[NSMutableArray array],*samples=[NSMutableArray array];
        report[@"programs"]=programs;report[@"arrays"]=arrays;report[@"samples"]=samples;
        int group=0;
        for(const auto &[rows,channels]:{std::pair{32,512},std::pair{1024,6144}})for(bool rotate:{false,true}) {
            const auto prefix="m"+std::to_string(rows)+"-c"+std::to_string(channels)+(rotate?"-rotate":"-raw");
            pa::Surface x(device,channels,rows,pa::Element::FP16);require(x.pitch()==size_t(rows)*2,"hidden quant fixed input pitch mismatch");
            std::vector<std::unique_ptr<Arm>> arms;
            for(int policy=0;policy<3;++policy) {
                const int ordinal=group*3+policy;if(scope)scope(ordinal*1000);
                report[@"active_policy"]=@(names[policy]);report[@"active_rows"]=@(rows);report[@"active_rotate"]=@(rotate);
                auto spec=lab::program(channels,rows,rotate,policy);auto arm=std::make_unique<Arm>(device,channels,rows);
                try{arm->program=std::make_unique<pa::Program>(device,spec.mil,spec.constants,out/"cache");}
                catch(const std::exception&e){
                    [programs addObject:@{@"ordinal":@(ordinal),@"policy":@(names[policy]),@"rotate":@(rotate),@"rows":@(rows),@"channels":@(channels),
                        @"status":@"compile-rejected",@"error":ns(e.what())}];
                    if(policy==0)throw;arms.push_back(nullptr);continue;
                }
                require(arm->program->inputs()==std::vector<std::string>{"x"}&&arm->program->outputs()==std::vector<std::string>{"y"},"hidden quant single-output ABI mismatch");
                [programs addObject:@{@"ordinal":@(ordinal),@"policy":@(names[policy]),@"rotate":@(rotate),@"rows":@(rows),@"channels":@(channels),
                    @"status":@"ready",@"key":ns(arm->program->cache_key()),@"compiled_now":@(arm->program->compiled_now()),@"constant_bytes":@(spec.constants.size())}];
                arms.push_back(std::move(arm));
            }
            auto predict=[&](int policy) {
                auto &a=*arms[policy];const auto ready=++timeline,done=++timeline;
                auto io=device.prepare_transfer({},{{a.y,std::nullopt,0,ane::DType::FP16,1.f}},ready,done);
                std::pair<std::string,pa::Surface> ins[]{{"x",x}},outs[]{{"y",a.y}};
                auto request=a.program->enqueue(ins,outs,ready,done,io.failure_callback());
                try{io.submit();}catch(...){(void)request.finish();throw;}
                const auto e=request.finish(),g=io.finish();
                require(e.ok&&e.physical_complete&&!e.retired&&!e.timed_out&&g.ok&&!g.timed_out,"hidden quant physical completion failed");
                return io.validation_flags();
            };
            for(int pattern=0;pattern<6;++pattern) {
                auto feed=input(channels,rows,pattern);std::memcpy(x.data(),feed.data(),feed.size()*2);
                const auto name=prefix+"-"+patterns[pattern]+"-input.f16";save(out/name,feed);const auto feed_hash=hash(feed);
                for(int policy=0;policy<3;++policy) {
                    if(!arms[policy])continue;
                    const int ordinal=group*3+policy,phase=ordinal*1000+pattern+1;if(scope)scope(phase);
                    const auto flags=predict(policy);const bool rejected=rotate&&pattern==4;
                    require(rejected?flags!=0:flags==0,"hidden quant finite/overflow negative control differed");
                    require(std::memcmp(x.data(),feed.data(),feed.size()*2)==0,"hidden quant modified source");
                    std::vector<_Float16> result(size_t(channels+1)*rows);std::memcpy(result.data(),arms[policy]->y.data(),result.size()*2);
                    const auto digest=hash(result);if(pattern==0)arms[policy]->first=digest;
                    if(pattern==5)require(digest==arms[policy]->first,"hidden quant A/zero/tiny/wide/overflow/A did not restore bits");
                    const auto output_name=prefix+"-"+names[policy]+"-"+patterns[pattern]+"-output.f16";save(out/output_name,result);
                    [arrays addObject:@{@"ordinal":@(ordinal),@"scope":@(phase),@"policy":@(names[policy]),@"rotate":@(rotate),@"pattern":@(patterns[pattern]),
                        @"rows":@(rows),@"channels":@(channels),@"validation_flags":@(flags),@"expected_rejection":@(rejected),
                        @"input_file":ns(name),@"input_sha256":ns(feed_hash),@"output_file":ns(output_name),@"output_sha256":ns(digest)}];
                }
            }
            std::vector<int> selected;for(int policy=0;policy<3;++policy)if(arms[policy])selected.push_back(policy);
            for(int rep=-2;rep<repeats;++rep) {
                auto feed=input(channels,rows,(rep&1)?3:0);std::memcpy(x.data(),feed.data(),feed.size()*2);
                for(int j=0;j<int(selected.size());++j) {
                    const int policy=selected[(j+rep+int(selected.size()))%selected.size()],ordinal=group*3+policy,phase=ordinal*1000+102+rep;
                    if(scope)scope(phase);const auto start=Clock::now();const auto flags=predict(policy);
                    const auto seconds=std::chrono::duration<double>(Clock::now()-start).count();require(flags==0,"timed hidden quant rejected output");
                    [samples addObject:@{@"ordinal":@(ordinal),@"scope":@(phase),@"policy":@(names[policy]),@"rotate":@(rotate),@"rows":@(rows),@"channels":@(channels),
                        @"rep":@(rep),@"warmup":@(rep<0),@"input_variant":@(bool(rep&1)),@"validation_flags":@(flags),@"seconds":@(seconds)}];
                }
                require(std::memcmp(x.data(),feed.data(),feed.size()*2)==0,"timed hidden quant source changed");
            }
            ++group;
        }
        report[@"source_unchanged"]=@YES;report[@"same_program_restoration_exact"]=@YES;report[@"overflow_negative_checked"]=@YES;report[@"status"]=@"complete";
    }catch(const std::exception&e){report[@"status"]=@"failed";report[@"error"]=ns(e.what());std::cerr<<e.what()<<'\n';}
    if(scope)scope(-3);if(fresh){std::ofstream f(out/"report.json");f<<json(report)<<'\n';}
    return [report[@"status"] isEqual:@"complete"]?0:1;
}}

#include "rotation_mil.hpp"
#include "../../../native/backends/private/ane_program.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

#ifndef TC_ROTATION_CPP_SHA
#define TC_ROTATION_CPP_SHA "unbound"
#endif
#ifndef TC_ROTATION_HEADER_SHA
#define TC_ROTATION_HEADER_SHA "unbound"
#endif
#ifndef TC_ROTATION_NATIVE_ID
#define TC_ROTATION_NATIVE_ID "unbound"
#endif
namespace {
using namespace tc;namespace pa=ane::private_api;namespace lab=qwen_rotation_lab;
constexpr int K=10,blocks[]{0,512,64,32,16,128,-64,-32,10064,10032};
const char *names[]{"identity","conv512","conv64x8","conv32x16","conv16x32","conv128x4","matmul64x8","matmul32x16","flat64x8","flat32x16"};
const char *patterns[]{"random-a","signed-basis","wide","random-a-restored"};
NSString *ns(const std::string&s){return [NSString stringWithUTF8String:s.c_str()];}
void save(const std::filesystem::path&p,const std::vector<_Float16>&v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*2);require(bool(f),"rotation array write failed");}
std::string hash(const std::vector<_Float16>&v){return memory_sha256_hex({reinterpret_cast<const char*>(v.data()),v.size()*2});}
std::vector<_Float16> input(int channels,int rows,int pattern) {
    std::vector<_Float16> values(size_t(channels)*rows);
    for(int c=0;c<channels;++c)for(int r=0;r<rows;++r) {
        const uint32_t index=uint32_t(c)*uint32_t(rows)+uint32_t(r);
        uint32_t v=index*747796405u+2891336453u;v=((v>>((v>>28)+4))^v)*277803737u;v=(v>>22)^v;
        float x=(int(v&65535)-32768)/32768.f;
        if(pattern==1)x=(c%512==(r*13)%512)?float(lab::sign(20260930,c%512))*.5f:0.f;
        else x*=pattern==2?16.f:.125f;
        values[size_t(c)*rows+r]=_Float16(x==0?0:x);
    }
    return values;
}
struct Arm {
    pa::Surface y;std::unique_ptr<pa::Program> program;std::string first;
    Arm(pa::Device&device,int channels,int rows):y(device,channels,rows,pa::Element::FP16){}
};
}
extern "C" __attribute__((visibility("default")))
int tc_rotation_probe(const char *output,int repeats,void(*scope)(int)) {@autoreleasepool {
    const std::filesystem::path out(output?output:"");bool fresh=false;
    NSMutableDictionary *report=[@{@"schema":@"tc-h512-rotation-probe-v1",@"status":@"starting",@"synthetic_only":@YES,
        @"model_weights_loaded":@NO,@"complete_request":@NO,@"image_export":@NO,@"cpp_sha256":@TC_ROTATION_CPP_SHA,
        @"header_sha256":@TC_ROTATION_HEADER_SHA,@"compiled_native_id":@TC_ROTATION_NATIVE_ID,
        @"repeats":@(repeats),@"warmups":@2,@"dtype":@"float16-le",@"layout":@"channels_tokens"} mutableCopy];
    try {
        require(output&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"fresh rotation diagnostic directory required");fresh=true;
        require(repeats==12||repeats==24,"bounded balanced repetitions required");
        char *raw=tc_runtime_build_identity();require(raw,"native identity missing");std::string id(raw);tc_string_free(raw);
        report[@"loaded_runtime_id"]=ns(id);require(id==TC_ROTATION_NATIVE_ID,"rotation helper/native ABI mismatch");
        pa::Device device(false,false,0);uint64_t timeline=0;NSMutableArray *programs=[NSMutableArray array],*records=[NSMutableArray array],*samples=[NSMutableArray array];
        report[@"programs"]=programs;report[@"arrays"]=records;report[@"samples"]=samples;
        int geometry=0;
        for(const auto &[rows,channels]:{std::pair{32,512},std::pair{1024,6144}}) {
            pa::Surface x(device,channels,rows,pa::Element::FP16);require(x.pitch()==size_t(rows)*2,"fixed input pitch mismatch");
            std::vector<std::unique_ptr<Arm>> arms;
            for(int arm=0;arm<K;++arm) {
                const int ordinal=geometry*K+arm;if(scope)scope(ordinal*1000);
                report[@"active_arm"]=@(names[arm]);report[@"active_rows"]=@(rows);report[@"active_channels"]=@(channels);
                auto spec=lab::program(channels,rows,blocks[arm]);auto value=std::make_unique<Arm>(device,channels,rows);
                try{value->program=std::make_unique<pa::Program>(device,spec.mil,spec.constants,out/"cache");}
                catch(const std::exception&e){
                    [programs addObject:@{@"ordinal":@(ordinal),@"arm":@(names[arm]),@"block":@(blocks[arm]),@"rows":@(rows),@"channels":@(channels),
                        @"status":@"compile-rejected",@"error":ns(e.what()),@"constant_bytes":@(spec.constants.size())}];
                    if(arm<2)throw;arms.push_back(nullptr);continue;
                }
                require(value->program->inputs()==std::vector<std::string>{"x"}&&value->program->outputs()==std::vector<std::string>{"y"},"rotation model ABI mismatch");
                [programs addObject:@{@"ordinal":@(ordinal),@"arm":@(names[arm]),@"block":@(blocks[arm]),@"rows":@(rows),@"channels":@(channels),
                    @"status":@"ready",@"key":ns(value->program->cache_key()),@"compiled_now":@(value->program->compiled_now()),@"constant_bytes":@(spec.constants.size())}];
                arms.push_back(std::move(value));
            }
            auto predict=[&](int arm) {
                const auto ready=++timeline,done=++timeline;
                auto &a=*arms[arm];auto io=device.prepare_transfer({},{{a.y,std::nullopt,0,ane::DType::FP16,1.f}},ready,done);
                std::pair<std::string,pa::Surface> ins[]{{"x",x}},outs[]{{"y",a.y}};
                auto ticket=a.program->enqueue(ins,outs,ready,done,io.failure_callback());
                try{io.submit();}catch(...){(void)ticket.finish();throw;}
                const auto e=ticket.finish(),g=io.finish();
                require(e.ok&&e.physical_complete&&!e.retired&&!e.timed_out&&g.ok&&!g.timed_out&&io.validation_flags()==0,
                    "rotation actual physical completion/finite transfer failed");
            };
            // Qualification patterns/old-output ownership are outside timing.
            for(int pattern=0;pattern<4;++pattern) {
                auto feed=input(channels,rows,pattern);std::memcpy(x.data(),feed.data(),feed.size()*2);
                const auto prefix="m"+std::to_string(rows)+"-c"+std::to_string(channels);
                const auto input_name=prefix+"-"+patterns[pattern]+"-input.f16";save(out/input_name,feed);
                for(int arm=0;arm<K;++arm) {
                    if(!arms[arm])continue;
                    const int phase=(geometry*K+arm)*1000+pattern+1;if(scope)scope(phase);predict(arm);
                    std::vector<_Float16> result(feed.size());std::memcpy(result.data(),arms[arm]->y.data(),result.size()*2);
                    require(std::memcmp(x.data(),feed.data(),feed.size()*2)==0,"rotation modified source");
                    if(arm==0)require(std::memcmp(result.data(),feed.data(),feed.size()*2)==0,"rotation identity bits/layout failed");
                    const auto digest=hash(result);
                    if(pattern==0)arms[arm]->first=digest;
                    if(pattern==3)require(digest==arms[arm]->first,"rotation A/B/wide/A restoration failed");
                    const auto name=prefix+"-"+names[arm]+"-"+patterns[pattern]+"-output.f16";save(out/name,result);
                    [records addObject:@{@"ordinal":@(geometry*K+arm),@"scope":@(phase),@"arm":@(names[arm]),@"pattern":@(patterns[pattern]),
                        @"rows":@(rows),@"channels":@(channels),@"input_file":ns(input_name),@"input_sha256":ns(hash(feed)),
                        @"output_file":ns(name),@"output_sha256":ns(digest)}];
                }
            }
            std::vector<int> selected;for(int arm=0;arm<K;++arm)if(arms[arm])selected.push_back(arm);
            for(int rep=-2;rep<repeats;++rep) {
                auto feed=input(channels,rows,(rep&1)?1:0);std::memcpy(x.data(),feed.data(),feed.size()*2);
                for(int j=0;j<int(selected.size());++j) {
                    const int arm=selected[(j+rep+int(selected.size()))%selected.size()],phase=(geometry*K+arm)*1000+10+rep+2;if(scope)scope(phase);
                    const auto start=Clock::now();predict(arm);const auto seconds=std::chrono::duration<double>(Clock::now()-start).count();
                    [samples addObject:@{@"ordinal":@(geometry*K+arm),@"scope":@(phase),@"arm":@(names[arm]),@"rows":@(rows),@"channels":@(channels),
                        @"rep":@(rep),@"warmup":@(rep<0),@"input_variant":@(bool(rep&1)),@"seconds":@(seconds)}];
                }
                require(std::memcmp(x.data(),feed.data(),feed.size()*2)==0,"timed rotation modified source");
            }
            ++geometry;
        }
        report[@"programs"]=programs;report[@"arrays"]=records;report[@"samples"]=samples;
        report[@"source_unchanged"]=@YES;report[@"identity_bits_exact"]=@YES;report[@"same_program_restoration_exact"]=@YES;report[@"status"]=@"complete";
    }catch(const std::exception&e){report[@"status"]=@"failed";report[@"error"]=ns(e.what());std::cerr<<e.what()<<'\n';}
    if(scope)scope(-3);if(fresh){std::ofstream f(out/"report.json");f<<json(report)<<'\n';}
    return [report[@"status"] isEqual:@"complete"]?0:1;
}}

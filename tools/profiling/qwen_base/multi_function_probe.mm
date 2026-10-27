#include "multi_function.hpp"
#include "../../../native/backends/private/ane_program.hpp"
#include "turbocider/turbocider.h"
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <algorithm>
#include <cmath>
#include <fstream>

#ifndef TC_MULTI_FUNCTION_CPP_SHA
#define TC_MULTI_FUNCTION_CPP_SHA "unbound"
#endif
#ifndef TC_MULTI_FUNCTION_HEADER_SHA
#define TC_MULTI_FUNCTION_HEADER_SHA "unbound"
#endif

namespace {
using namespace tc::ane::private_api;
void require(bool ok,const std::string &message) {if(!ok)throw tc::ane::CapabilityError(message);}
NSString *ns(const std::string &s) {return [NSString stringWithUTF8String:s.c_str()];}
void raw(const std::filesystem::path &path,const Surface &s) {
    std::ofstream f(path,std::ios::binary);f.write(static_cast<const char*>(s.data()),s.bytes());
    require(bool(f),"capability raw array export failed");
}
void json(const std::filesystem::path &path,NSDictionary *value) {
    NSError *error=nil;NSData *data=[NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingPrettyPrinted error:&error];
    require(data!=nil,"capability JSON encoding failed");
    std::ofstream f(path,std::ios::binary);f.write(static_cast<const char*>(data.bytes),data.length);
    require(bool(f),"capability JSON export failed");
}
NSArray *numbers(const std::vector<uint32_t> &values) {
    NSMutableArray *array=[NSMutableArray new];for(auto value:values)[array addObject:@(value)];return array;
}
NSArray *names(const std::vector<std::string> &values) {
    NSMutableArray *array=[NSMutableArray new];for(const auto &value:values)[array addObject:ns(value)];return array;
}
}

extern "C" __attribute__((visibility("default")))
int tc_ane_multi_function_probe(const char *output,void(*scope)(int)) {
 @autoreleasepool {
    const std::filesystem::path out(output?output:"");
    NSMutableDictionary *report=[@{@"schema":@"tc-ane-multiple-procedure-capability-v1",@"status":@"starting",
        @"cpp_sha256":@TC_MULTI_FUNCTION_CPP_SHA,@"mil_header_sha256":@TC_MULTI_FUNCTION_HEADER_SHA,
        @"synthetic_only":@YES,@"complete_image_request":@NO,
        @"physical_placement_qualified":@NO,@"performance_qualified":@NO} mutableCopy];
    try {
        require(output && std::filesystem::is_directory(out) && std::filesystem::is_empty(out),"fresh capability output required");
        char *identity=tc_runtime_build_identity();require(identity,"runtime build ID missing");
        report[@"loaded_runtime_id"]=@(identity);tc_string_free(identity);
        Device device;
        Surface input(device,128,64,Element::FP16),weight(device,128,128,Element::FP16),output_surface(device,128,64,Element::FP16);
        require(input.pitch()==128 && weight.pitch()==256 && output_surface.pitch()==128,"actual capability surface pitch differs");
        report[@"surfaces"]=@{@"input_id":@(IOSurfaceGetID((IOSurfaceRef)input.native_iosurface())),
            @"weight_id":@(IOSurfaceGetID((IOSurfaceRef)weight.native_iosurface())),
            @"output_id":@(IOSurfaceGetID((IOSurfaceRef)output_surface.native_iosurface())),
            @"input_pitch_bytes":@(input.pitch()),@"weight_pitch_bytes":@(weight.pitch()),
            @"output_pitch_bytes":@(output_surface.pitch()),@"max_rows":@64,@"channels":@128};
        const auto mil=tc::profiling::multi_function_matmul_mil();
        {std::ofstream f(out/"model.mil");f<<mil;require(bool(f),"MIL export failed");}
        if(scope)scope(0);
        Program program(device,mil,{},out/"cache",Program::MultipleProcedures{});
        report[@"cache_key"]=ns(program.cache_key());report[@"compiled_now"]=@(program.compiled_now());
        NSMutableArray *table=[NSMutableArray new];
        for(const auto &p:program.procedures())
            [table addObject:@{@"function":ns(p.function),@"id":@(p.id),@"inputs":names(p.inputs),@"outputs":names(p.outputs),
                @"input_indices":numbers(p.input_indices),@"output_indices":numbers(p.output_indices)}];
        report[@"procedures"]=table;json(out/"report.json",report);
        require(program.procedures().size()==2,"exactly two procedures required");
        bool unknown_rejected=false,implicit_rejected=false;
        try {(void)named_procedure(program.procedures(),"absent");}catch(const tc::ane::CapabilityError&){unknown_rejected=true;}
        try {(void)program.inputs();}catch(const tc::ane::CapabilityError&){implicit_rejected=true;}
        require(unknown_rejected && implicit_rejected,"unknown/implicit function was admitted");
        report[@"unknown_function_rejected"]=@YES;report[@"implicit_binding_rejected"]=@YES;
        NSMutableArray *runs=[NSMutableArray new];report[@"runs"]=runs;
        uint64_t cursor=device.value();
        for(int iteration=0;iteration<3;++iteration) {
            const int rows=iteration==1?64:32;
            const std::string function="rows"+std::to_string(rows),prefix="run-"+std::to_string(iteration);
            auto *x=static_cast<_Float16*>(input.data()),*w=static_cast<_Float16*>(weight.data()),*y=static_cast<_Float16*>(output_surface.data());
            std::fill_n(w,128*128,_Float16(0));std::fill_n(y,128*64,_Float16(1234));
            for(int c=0;c<128;++c) {
                w[c*128+c]=_Float16(iteration==1?-.25f:.5f);
                w[c*128+(c+1)%128]=_Float16(iteration==1?-.0625f:.125f);
                for(int r=0;r<64;++r)x[c*64+r]=_Float16(float(c%17-8)*.125f+float(r)*.25f);
            }
            raw(out/(prefix+"-input.bin"),input);raw(out/(prefix+"-weight.bin"),weight);
            const auto &p=named_procedure(program.procedures(),function);
            require(p.inputs==std::vector<std::string>({"x","w"}) || p.inputs==std::vector<std::string>({"w","x"}),
                    "loaded function input names differ from the declared x/w ports");
            require(p.outputs==std::vector<std::string>({"y"}),"loaded function output name differs from y");
            std::vector<std::pair<std::string,Surface>> ins{{"x",input},{"w",weight}},outs{{"y",output_surface}};
            const auto ready=cursor+1,done=cursor+2;
            if(scope)scope(iteration+1);
            auto prepared=program.prepare_function(function,ins,outs,ready,done);
            device.signal(ready);auto ticket=prepared.submit();const auto completion=ticket.finish();
            require(completion.ok && completion.physical_complete && !completion.retired && !completion.timed_out,
                    "function did not physically complete: "+completion.error);
            require(device.wait(done,std::chrono::seconds(30)),"function done event missing");cursor=done;
            raw(out/(prefix+"-output.bin"),output_surface);
            float max_error=0;bool tail_unchanged=true;
            for(int c=0;c<128;++c)for(int r=0;r<64;++r) {
                if(r>=rows)tail_unchanged=tail_unchanged && y[c*64+r]==_Float16(1234);
                else {
                    const float reference=float(w[c*128+c])*float(x[c*64+r])+
                        float(w[c*128+(c+1)%128])*float(x[((c+1)%128)*64+r]);
                    const float actual=float(y[c*64+r]);require(std::isfinite(actual),"nonfinite function output");
                    max_error=std::max(max_error,std::abs(actual-float(_Float16(reference))));
                }
            }
            require(max_error<=.015625f && tail_unchanged,"shared-pitch output values/tail canary differ");
            [runs addObject:@{@"index":@(iteration),@"function":ns(function),@"procedure_id":@(p.id),@"rows":@(rows),
                @"input_indices":numbers(p.input_indices),@"output_indices":numbers(p.output_indices),
                @"same_surface_set":@YES,@"physical_complete":@(completion.physical_complete),
                @"max_half_reference_error":@(max_error),@"tail_canary_unchanged":@(tail_unchanged)}];
            json(out/"report.json",report);
        }
        report[@"status"]=@"complete";json(out/"report.json",report);return 0;
    }catch(const std::exception &e) {
        report[@"status"]=@"failed";report[@"error"]=@(e.what());
        if(std::filesystem::is_directory(out))json(out/"report.json",report);return 1;
    }
 }
}

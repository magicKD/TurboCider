#include "../../../native/backends/private/ane_program.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "../../../native/runtime/memory_manifest.hpp"
#include "turbocider/turbocider.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#ifndef TC_ACTIVATION_OP_CPP_SHA
#define TC_ACTIVATION_OP_CPP_SHA "unbound"
#endif

namespace {
using namespace tc;
namespace pa=ane::private_api;
static_assert(std::endian::native==std::endian::little);
NSString *ns(const std::string&s){return [NSString stringWithUTF8String:s.c_str()];}
void write_json(const std::filesystem::path&p,NSDictionary*v){std::ofstream f(p);f<<json(v)<<'\n';require(bool(f),"operator diagnostic JSON write failed");}
std::string digest(const std::vector<_Float16>&v){return memory_sha256_hex({reinterpret_cast<const char*>(v.data()),v.size()*2});}
void save(const std::filesystem::path&p,const std::vector<_Float16>&v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*2);require(bool(f),"operator array write failed");}
std::string mil(int channels,int rows,int mode){
    const auto shape="[1, 1, "+std::to_string(channels)+", "+std::to_string(rows)+"]";
    const auto plane=std::to_string(uint64_t(channels)*rows),pitch=std::to_string(rows);
    const auto tensor="tensor<fp16, "+shape+">";
    const auto buffer="tensor_buffer<fp16, shape="+shape+", strides=["+plane+", "+plane+", "+pitch+", 1], interleave_factors=[1, 1, 1, 1]>";
    std::string body="        "+tensor+" xt = tensor_buffer_to_tensor<ios17>(input = x);\n";
    if(mode==0)body+="        "+tensor+" value = mul(x = xt, y = fp16(1));\n";
    else if(mode==1)body+="        "+tensor+" neg = mul(x = xt, y = fp16(-1));\n        "+tensor+" eg = exp(x = neg);\n        "+tensor+
        " denom = add(x = eg, y = fp16(1));\n        "+tensor+" value = real_div(x = xt, y = denom);\n";
    else if(mode==4)body+="        "+tensor+" sigmoid = sigmoid(x = xt);\n        "+tensor+" value = mul(x = xt, y = sigmoid);\n";
    else if(mode==5)body+="        "+tensor+" neg = mul(x = xt, y = fp16(-1));\n        "+tensor+
        " sigmoid_negative = sigmoid(x = neg);\n        "+tensor+" complement = sub(x = fp16(1), y = sigmoid_negative);\n        "+tensor+
        " value = mul(x = xt, y = complement);\n";
    else body+="        "+tensor+" value = "+std::string(mode==2?"sigmoid":"silu")+"(x = xt);\n";
    body+="        "+buffer+" y = tensor_to_tensor_buffer<ios17>(input = value, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>(["+
        plane+", "+plane+", "+pitch+", 1]));\n";
    return "program(1.3)\n{\n    func main_ane<ios18>("+buffer+" x) {\n"+body+"    } -> (y);\n}\n";
}
NSDictionary *quality(const std::vector<_Float16>&input,const std::vector<_Float16>&output,int mode){
    double error=0,energy=0,maximum=0;uint64_t changed=0;
    for(size_t i=0;i<input.size();++i){const double x=float(input[i]);
        const double e=std::exp(x<0?x:-x),sigmoid=x<0?e/(1+e):1/(1+e);
        const double expected=mode==0?x:mode==2?sigmoid:x*sigmoid,actual=float(output[i]);
        require(std::isfinite(actual),"nonfinite diagnostic activation output");
        const double d=actual-expected;error+=d*d;energy+=expected*expected;maximum=std::max(maximum,std::abs(d));
        changed+=std::bit_cast<uint16_t>(_Float16(expected))!=std::bit_cast<uint16_t>(output[i]);
    }
    return @{@"relative_l2":@(std::sqrt(error/std::max(energy,1e-300))),@"max_abs":@(maximum),
             @"changed_vs_fp16_oracle":@(changed),@"elements":@(input.size())};
}
}

extern "C" __attribute__((visibility("default")))
int tc_activation_op_probe(const char *output,void(*scope)(int)){
 @autoreleasepool {
    const std::filesystem::path out(output?output:"");bool fresh=false;
    NSMutableDictionary *report=[@{@"schema":@"tc-activation-operator-diagnostic-v1",@"status":@"starting",
        @"synthetic_only":@YES,@"model_weights_loaded":@NO,@"complete_request":@NO,@"image_export":@NO,
        @"cpp_sha256":@TC_ACTIVATION_OP_CPP_SHA,@"dtype":@"float16-le",@"layout":@"channels_tokens"} mutableCopy];
    try{
        require(output&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"fresh operator output directory required");fresh=true;
        char *id=tc_runtime_build_identity();require(id,"native identity missing");report[@"loaded_runtime_id"]=ns(id);tc_string_free(id);
        pa::Device device(false,false,0);uint64_t timeline=0;int ordinal=0;NSMutableArray *records=[NSMutableArray array],*programs=[NSMutableArray array];
        const char *modes[]{"identity","exp-divide","sigmoid","silu","sigmoid-mul","sigmoid-complement"};
        const char *patterns[]{"small0125","negative025","wide","small0125-restored"};
        for(const auto &[rows,channels]:{std::pair{32,128},std::pair{1024,6144}})for(int mode=0;mode<6;++mode,++ordinal){
            if(scope)scope(ordinal*10);
            pa::Surface x(device,channels,rows,pa::Element::FP16),y(device,channels,rows,pa::Element::FP16);
            require(x.pitch()==size_t(rows)*2&&y.pitch()==x.pitch(),"operator pitch differs from fixed MIL ABI");
            pa::Program program(device,mil(channels,rows,mode),std::span<const uint8_t>{},out/"cache");
            require(program.inputs()==std::vector<std::string>{"x"}&&program.outputs()==std::vector<std::string>{"y"},"operator model ABI mismatch");
            [programs addObject:@{@"ordinal":@(ordinal),@"operator":@(modes[mode]),@"rows":@(rows),@"channels":@(channels),
                @"cache_key":ns(program.cache_key()),@"compiled_now":@(program.compiled_now())}];
            std::vector<_Float16> first_input,first_output;
            for(int pattern=0;pattern<4;++pattern){
                std::vector<_Float16> input(size_t(channels)*rows),value(input.size());
                for(int c=0;c<channels;++c)for(int r=0;r<rows;++r){const size_t i=size_t(c)*rows+r;
                    const int hidden=channels==128?128:4096;
                    float v=pattern==2?(int(i%4096)-2048)/128.f:((r*3+(c%hidden)*7)%17-8)/8.f*(pattern==1?-.25f:.125f);
                    if(v==0)v=0;input[i]=_Float16(v);
                }
                std::memcpy(x.data(),input.data(),input.size()*2);
                if(scope)scope(ordinal*10+pattern+1);
                const auto ready=++timeline,done=++timeline;
                // A real read-only output validation binding supplies the
                // native handoff contract; empty transfers are not admitted.
                auto transfer=device.prepare_transfer({},{{y,std::nullopt,0,ane::DType::FP16,1.f}},ready,done);
                std::pair<std::string,pa::Surface> inputs[]{{"x",x}},outputs[]{{"y",y}};
                auto ticket=program.enqueue(inputs,outputs,ready,done,transfer.failure_callback());
                transfer.submit();const auto completed=ticket.finish(),gpu=transfer.finish();
                require(completed.ok&&completed.physical_complete&&!completed.retired&&!completed.timed_out&&gpu.ok&&!gpu.timed_out&&
                    transfer.validation_flags()==0,"operator diagnostic physical completion/transfer failed");
                std::memcpy(value.data(),y.data(),value.size()*2);
                require(std::memcmp(x.data(),input.data(),input.size()*2)==0,"operator modified input surface");
                if(mode==0)require(std::memcmp(input.data(),value.data(),input.size()*2)==0,"identity failed FP16 bits/layout control");
                if(pattern==0){first_input=input;first_output=value;}
                if(pattern==3)require(first_input==input&&std::memcmp(first_output.data(),value.data(),value.size()*2)==0,"operator A/B/wide/A state restoration failed");
                const auto name="m"+std::to_string(rows)+"-c"+std::to_string(channels)+"-"+modes[mode]+"-"+patterns[pattern];
                save(out/(name+"-input.f16"),input);save(out/(name+"-output.f16"),value);
                [records addObject:@{@"program":@(ordinal),@"scope":@(ordinal*10+pattern+1),@"operator":@(modes[mode]),@"pattern":@(patterns[pattern]),
                    @"rows":@(rows),@"channels":@(channels),@"input_file":ns(name+"-input.f16"),@"output_file":ns(name+"-output.f16"),
                    @"input_sha256":ns(digest(input)),@"output_sha256":ns(digest(value)),@"quality":quality(input,value,mode)}];
            }
        }
        if(scope)scope(-3);
        report[@"programs"]=programs;report[@"samples"]=records;report[@"source_unchanged"]=@YES;
        report[@"identity_bits_exact"]=@YES;report[@"same_program_restoration_exact"]=@YES;report[@"status"]=@"complete";
        write_json(out/"report.json",report);return 0;
    }catch(const std::exception&e){report[@"status"]=@"failed";report[@"error"]=ns(e.what());
        if(fresh)write_json(out/"report.json",report);std::cerr<<e.what()<<'\n';return 1;}
 }
}

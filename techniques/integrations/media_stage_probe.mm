// Original public vImage/MLX client. Product PE decoder is linked from source.
// No model inference, VideoToolbox decode/encode, color conversion or system changes.
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <mlx/mlx.h>
#include <mlx/compile.h>
#include <mlx/version.h>
#include "native/media/image.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;
namespace {
constexpr double budget = 1e-6;
constexpr size_t guard_bytes = 64;
void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
uint64_t now() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
NSString *sha(const void *pointer, size_t size) {
    check(size <= UINT32_MAX, "bounded hash required");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(pointer, CC_LONG(size), digest);
    NSMutableString *result = [NSMutableString string];
    for (auto byte : digest) [result appendFormat:@"%02x", byte];
    return result;
}
void save(NSDictionary *report, const char *path) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];
    check(data && std::filesystem::space(std::filesystem::path(path).parent_path()).available >= 128ull*1024*1024 + data.length*2 + (1<<20),
          "128MiB reserve plus atomic report/metadata required; nothing deleted");
    check([data writeToFile:[NSString stringWithUTF8String:path] options:NSDataWritingAtomic error:nil], "report write failed");
}
struct Frame {
    int width, height, pitch;
    std::vector<uint8_t> rgb, bgra;
    Frame(int w, int h, std::vector<uint8_t> data) : width(w), height(h), pitch(((w*4+63)/64+1)*64),
        rgb(std::move(data)), bgra(size_t(pitch)*h + guard_bytes) {
        check(w>0 && h>0 && w<=2048 && h<=2048 && rgb.size()==size_t(w)*h*3, "bounded complete RGB frame required");
    }
    void generation(int value) {
        std::fill(bgra.begin(),bgra.end(),0xa5);
        for(int y=0;y<height;++y) for(int x=0;x<width;++x) {
            auto *target=bgra.data()+size_t(y)*pitch+x*4; const auto *p=rgb.data()+(size_t(y)*width+x)*3;
            target[0]=p[2]^value;target[1]=p[1]^value;target[2]=p[0]^value;target[3]=(x*11+y*3)%256;
        }
        std::fill(bgra.end()-guard_bytes,bgra.end(),0x6b);
    }
};
Frame file_frame(NSDictionary *config) {
    auto image=tc::load_pe_image_tensor([config[@"source_png"] UTF8String]); image.eval();mx::synchronize();
    check(image.ndim()==4 && image.shape(0)==1 && image.shape(3)==3 && image.dtype()==mx::uint8, "actual product decoder RGB contract differs");
    check([sha(image.data<uint8_t>(),image.nbytes()) isEqual:config[@"expected_file_rgb_sha256"]], "product PE decoder differs from independent PNG sample bytes");
    return Frame(image.shape(2),image.shape(1),std::vector<uint8_t>(image.data<uint8_t>(),image.data<uint8_t>()+image.size()));
}
Frame synthetic_frame(int seed) {
    constexpr int w=513,h=257;std::vector<uint8_t> rgb(size_t(w)*h*3);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int c=0;c<3;++c)
        rgb[(size_t(y)*w+x)*3+c]=(x*13+y*7+c*43+(x*y)%31+seed*17)%256;
    return Frame(w,h,std::move(rgb));
}
struct Workspace {
    int w,h;size_t pixels;
    std::vector<uint8_t> rgb, first, second;
    explicit Workspace(int width,int height):w(width),h(height),pixels(size_t(w)*h),
        rgb(pixels*3+guard_bytes),first(pixels*4+guard_bytes),second(pixels*4+guard_bytes) {}
    void poison() {
        for(auto *value:{&rgb,&first,&second}) {
            std::fill(value->begin(),value->end(),0xee);std::fill(value->end()-guard_bytes,value->end(),0x6b);
        }
    }
    bool guards() const {
        for(const auto *value:{&rgb,&first,&second})
            if(!std::all_of(value->end()-guard_bytes,value->end(),[](auto v){return v==0x6b;}))return false;
        return true;
    }
};
// Source-bound inverse gather from RGB24VideoReader::Impl::next, not a new video decoder.
void product_loop(const Frame &frame,int code,uint8_t *rgb) {
    const bool rotated=code&4,fx=code&1,fy=code&2;
    const int width=rotated?frame.height:frame.width,height=rotated?frame.width:frame.height;
    std::atomic<bool> cancelled{false};
    for(int y=0;y<height;++y) { if(y%32==0)tc::checkpoint(cancelled);for(int x=0;x<width;++x) {
        const int sx=rotated?(fy?frame.width-1-y:y):(fx?frame.width-1-x:x);
        const int sy=rotated?(fx?frame.height-1-x:x):(fy?frame.height-1-y:y);
        const auto *p=frame.bgra.data()+size_t(sy)*frame.pitch+size_t(sx)*4;
        auto *out=rgb+(size_t(y)*width+x)*3;out[0]=p[2];out[1]=p[1];out[2]=p[0];
    }}
}
int accelerate(const Frame &frame,int code,Workspace &work) {
    vImage_Buffer source{const_cast<uint8_t*>(frame.bgra.data()),size_t(frame.height),size_t(frame.width),size_t(frame.pitch)};
    int calls=0;bool first=true;
    auto destination=[&](int w,int h) { return vImage_Buffer{first?work.first.data():work.second.data(),size_t(h),size_t(w),size_t(w)*4}; };
    auto advance=[&](vImage_Buffer next,vImage_Error error){check(error==kvImageNoError,"public vImage transform returned an error");source=next;first=!first;++calls;};
    if(code&4) {
        auto next=destination(frame.height,frame.width);Pixel_8888 background={0,0,0,0};
        advance(next,vImageRotate90_ARGB8888(&source,&next,kRotate90DegreesClockwise,background,kvImageNoFlags));
    }
    if(bool(code&1)!=bool(code&4)) { auto next=destination(work.w,work.h);advance(next,vImageHorizontalReflect_ARGB8888(&source,&next,kvImageNoFlags)); }
    if(code&2) { auto next=destination(work.w,work.h);advance(next,vImageVerticalReflect_ARGB8888(&source,&next,kvImageNoFlags)); }
    vImage_Buffer output{work.rgb.data(),size_t(work.h),size_t(work.w),size_t(work.w)*3};
    check(vImageConvert_BGRA8888toRGB888(&source,&output,kvImageNoFlags)==kvImageNoError,"public vImage BGRA/RGB conversion returned an error");
    return calls+1;
}
mx::array normalize(mx::array rgb,int width,int height) {
    // Original video_input unit normalization + SeedVR2 consume_resized_input at identity resize.
    auto pixels=(mx::astype(rgb,mx::float32)/255.f)*2.f-1.f;
    return mx::contiguous(mx::reshape(mx::transpose(pixels,{2,0,1}),{1,3,1,height,width}));
}
using Program=std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;
Program gpu_program(int width,int height,int code,int negative=0) {
    return mx::compile(Program([=](const std::vector<mx::array>&a){
        auto bgra=negative==1?mx::reshape(mx::slice(mx::reshape(a[0],{-1}),{0},{width*height*4}),{height,width,4}):
            mx::slice(a[0],{0,0,0},{height,width,4});
        if(code&4)bgra=mx::transpose(bgra,{1,0,2});
        if(code&1)bgra=mx::flip(bgra,1);
        if(code&2)bgra=mx::flip(bgra,0);
        auto rgb=mx::take(bgra,negative==2?mx::array({0,1,2},mx::int32):mx::array({2,1,0},mx::int32),2);
        const int ow=code&4?height:width,oh=code&4?width:height;
        return std::vector<mx::array>{normalize(rgb,ow,oh)};
    }));
}
NSDictionary *metrics(const float *values,const std::vector<uint8_t>&rgb,int width,int height) {
    size_t nonfinite=0,mismatches=0;double maximum=0,error=0,aa=0,bb=0,dot=0;
    std::vector<uint8_t> recovered(rgb.size());
    for(size_t i=0;i<size_t(width)*height;++i)for(size_t c=0;c<3;++c) {
        const size_t index=i*3+c;const double expected=double(rgb[index])/255.*2.-1.,actual=values[c*size_t(width)*height+i];
        nonfinite+=!std::isfinite(actual);
        if(std::isfinite(actual)) {
            maximum=std::max(maximum,std::abs(actual-expected));error+=(actual-expected)*(actual-expected);aa+=expected*expected;bb+=actual*actual;dot+=actual*expected;
            recovered[index]=uint8_t(std::clamp<long>(std::lround((actual+1)*127.5),0,255));
        }
        mismatches+=!std::isfinite(actual)||recovered[index]!=rgb[index];
    }
    return @{@"finite":@(nonfinite==0),@"max_abs":@(maximum),@"relative_l2":@(std::sqrt(error/aa)),@"cosine":@(dot/std::sqrt(aa*bb)),
        @"recovered_rgb_mismatches":@(mismatches),@"recovered_rgb_sha256":sha(recovered.data(),recovered.size())};
}
NSDictionary *metrics(mx::array &output,const std::vector<uint8_t>&rgb,int width,int height) {
    check(output.dtype()==mx::float32&&output.shape()==mx::Shape({1,3,1,height,width})&&output.flags().row_contiguous,"complete contiguous model-input consumer required");
    return metrics(output.data<float>(),rgb,width,height);
}
bool passes(NSDictionary *metric) {
    return [metric[@"finite"] boolValue]&&[metric[@"recovered_rgb_mismatches"] unsignedLongLongValue]==0&&
        [metric[@"max_abs"] doubleValue]<=budget&&[metric[@"relative_l2"] doubleValue]<=1e-6&&[metric[@"cosine"] doubleValue]>=.999999;
}
void exercise(Frame &frame,NSString *kind,int code,int repeats,NSDictionary *config,NSMutableDictionary *report,const char *path) {
    const int ow=code&4?frame.height:frame.width,oh=code&4?frame.width:frame.height;
    Workspace work(ow,oh);std::array<std::vector<uint8_t>,2> references;
    NSMutableArray *input_hashes=[NSMutableArray array],*rgb_hashes=[NSMutableArray array];
    for(int generation=0;generation<2;++generation) {
        frame.generation(generation);references[generation].resize(size_t(ow)*oh*3);
        product_loop(frame,code,references[generation].data());
        [input_hashes addObject:sha(frame.bgra.data(),size_t(frame.pitch)*frame.height)];
        [rgb_hashes addObject:sha(references[generation].data(),references[generation].size())];
    }
    NSDictionary *expected=config[@"source_identities"][kind];
    check([input_hashes isEqual:expected[@"bgra_sha256_by_generation"]],"complete padded input differs from independent source bytes");
    check([rgb_hashes isEqual:expected[@"rgb_sha256_by_orientation"][@(code).stringValue]],"product inverse gather differs from independent forward scatter");
    NSMutableDictionary *scenario=[@{@"id":[NSString stringWithFormat:@"%@-o%d",kind,code],@"source":kind,@"orientation":@(code),
        @"source_wh":@[@(frame.width),@(frame.height)],@"source_pitch":@(frame.pitch),@"output_wh":@[@(ow),@(oh)],
        @"rgb_sha256_by_generation":rgb_hashes,@"bgra_sha256_by_generation":input_hashes,@"samples":[NSMutableArray array],
        @"product_source_formula_bound":@YES,@"vimage_scratch_capacity_bytes":@(work.first.size()+work.second.size()),
        @"rgb_capacity_bytes":@(work.rgb.size())} mutableCopy];
    [report[@"scenarios"] addObject:scenario];save(report,path);
    auto normal=mx::compile(Program([=](const std::vector<mx::array>&a){return std::vector<mx::array>{normalize(a[0],ow,oh)};}));
    auto direct=gpu_program(frame.width,frame.height,code);
    std::vector<float> old_output;
    const std::array<NSString*,3> arms={@"product-loop",@"vimage",@"gpu-fused"};
    for(int group=0;group<repeats+3;++group) {
        const int generation=group%2;frame.generation(generation);
        NSString *phase=group==0?@"first":group<3?@"warmup":@"measured";
        const int trial=group==0?0:group<3?group-1:group-3,row=group<3?group:trial;
        for(int position=0;position<3;++position) {
            const int arm=(row%3+(row/3%2?2-position:position))%3;
            work.poison();mx::synchronize();NSMutableArray *events=[NSMutableArray array];
            std::optional<mx::array> input,output;int calls=0;
            const uint64_t start=now();uint64_t t=start;
            auto event=[&](NSString *name,uint64_t begin){[events addObject:@{@"name":name,@"start_ns":@(begin-start),@"duration_ns":@(now()-begin),@"clock_domain":@"same-process-uptime-raw-host"}];};
            if(arm==0)product_loop(frame,code,work.rgb.data());else if(arm==1)calls=accelerate(frame,code,work);
            event(@"cpu-transform",t);t=now();
            if(arm==2)input.emplace(frame.bgra.data(),mx::Shape{frame.height,frame.pitch/4,4},mx::uint8);
            else input.emplace(work.rgb.data(),mx::Shape{oh,ow,3},mx::uint8);
            event(@"mlx-import",t);t=now();output.emplace((arm==2?direct:normal)({*input})[0]);output->eval();mx::synchronize();
            event(@"consumer-ready",t);const uint64_t wall=now()-start;
            auto metric=metrics(*output,references[generation],ow,oh);
            const bool guards=work.guards()&&std::all_of(frame.bgra.end()-guard_bytes,frame.bgra.end(),[](auto v){return v==0x6b;});
            const bool unchanged=[sha(frame.bgra.data(),size_t(frame.pitch)*frame.height) isEqual:input_hashes[generation]];
            const bool cpu_exact=arm==2||[sha(work.rgb.data(),size_t(ow)*oh*3) isEqual:rgb_hashes[generation]];
            NSDictionary *sample=@{@"phase":phase,@"trial":@(trial),@"generation":@(generation),@"arm":arms[arm],@"wall_ns":@(wall),
                @"events":events,@"metrics":metric,@"numeric_pass":@(passes(metric)&&cpu_exact),@"host_guards_pass":@(guards),@"source_unchanged":@(unchanged),
                @"mlx_stream_drained":@YES,@"output_dtype":@"float32",@"output_elements":@(output->size()),
                @"output_sha256":sha(output->data<float>(),output->nbytes()),@"rgb_sha256":metric[@"recovered_rgb_sha256"],
                @"logical_import_bytes":@(arm==2?size_t(frame.pitch)*frame.height:size_t(ow)*oh*3),@"vimage_api_calls":@(calls)};
            [scenario[@"samples"] addObject:sample];check(passes(metric)&&cpu_exact&&guards&&unchanged,"complete normalization/RGB/source/guard gate failed");
            if(group==0&&arm==0)old_output.assign(output->data<float>(),output->data<float>()+output->size());
        }
        if(group==1) {
            NSMutableDictionary *control=[metrics(old_output.data(),references[1],ow,oh) mutableCopy];
            control[@"new_gpu_dispatch"]=@NO;control[@"used_for_speed_ratio"]=@NO;control[@"rejected"]=@(!passes(control));
            scenario[@"stale_actual_output_control"]=control;check([control[@"rejected"] boolValue],"actual cached-output control not rejected");
        }
    }
    frame.generation(1);
    for(int negative=1;negative<=2;++negative) {
        auto program=gpu_program(frame.width,frame.height,code,negative);
        mx::array source(frame.bgra.data(),{frame.height,frame.pitch/4,4},mx::uint8);auto output=program({source})[0];output.eval();mx::synchronize();
        NSMutableDictionary *control=[metrics(output,references[1],ow,oh) mutableCopy];
        control[@"new_gpu_dispatch"]=@YES;control[@"used_for_speed_ratio"]=@NO;control[@"rejected"]=@(!passes(control));
        scenario[negative==1?@"wrong_stride_control":@"wrong_channels_control"]=control;
        check([control[@"finite"] boolValue]&&[control[@"rejected"] boolValue],"finite wrong stride/channels must be rejected");
    }
    save(report,path);
}
}
int main(int argc,const char **argv) {
    @autoreleasepool {
        if(argc!=3){std::fprintf(stderr,"usage: media-stage new-config.json new-native.json\n");return 2;}
        NSMutableDictionary *report=[@{@"schema":@"tc-media-stage-native-v1",@"status":@"starting",@"errors":[NSMutableArray array],
            @"scenarios":[NSMutableArray array],@"production_qualified":@NO,@"whole_request_qualified":@NO,@"physical_overlap_qualified":@NO,
            @"resize":@NO,@"output_layout":@"contiguous-FP32-1C1HW",@"gate":@{@"max_abs":@(budget),@"relative_l2":@1e-6,@"cosine_min":@.999999},
            @"source_pixel_contract":@"PNG RGB bytes; synthetic padding/alpha and XOR1 freshness control; not actual video samples",
            @"timing_boundary":@"prepared padded CPU BGRA -> complete contiguous normalized FP32 model-input GPU output; no decode/resize/VAE/codec"} mutableCopy];
        try {
            check(!std::filesystem::exists(argv[2]),"new native report required");
            NSData *config_data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];
            NSDictionary *config=[NSJSONSerialization JSONObjectWithData:config_data options:0 error:nil];check(config!=nil,"explicit config required");
            const int seed=[config[@"seed"] intValue],repeats=[config[@"measured_groups"] intValue];
            check(seed>=0&&seed<=10000&&(repeats==10||repeats==30),"bounded protocol required");
            report[@"source_config_sha256"]=sha(config_data.bytes,config_data.length);report[@"seed"]=@(seed);report[@"measured_groups"]=@(repeats);
            mx::set_default_device(mx::Device::gpu);mx::enable_compile();
            report[@"mlx_version"]=[NSString stringWithUTF8String:mx::version()];id<MTLDevice> device=MTLCreateSystemDefaultDevice();check(device!=nil,"actual Metal GPU required");report[@"gpu_name"]=device.name;
            auto file=file_frame(config),synthetic=synthetic_frame(seed);
            report[@"product_pe_decoder_matches_independent_png"]=@YES;report[@"file_rgb_sha256"]=sha(file.rgb.data(),file.rgb.size());
            report[@"source_file_sha256"]=config[@"source_file_sha256"];save(report,argv[2]);
            for(int code=0;code<8;++code){exercise(file,@"file",code,repeats,config,report,argv[2]);exercise(synthetic,@"synthetic",code,repeats,config,report,argv[2]);}
            mx::synchronize();report[@"all_owned_gpu_work_completed"]=@YES;report[@"status"]=@"measured";
        } catch(const std::exception &error) {
            report[@"status"]=@"failed";[report[@"errors"] addObject:[NSString stringWithUTF8String:error.what()]];
            try{mx::synchronize();report[@"all_owned_gpu_work_completed"]=@YES;}catch(...){report[@"all_owned_gpu_work_completed"]=@NO;}
        }
        try{save(report,argv[2]);}catch(const std::exception &error){std::fprintf(stderr,"%s\n",error.what());return 4;}
        return [report[@"status"] isEqual:@"measured"]?0:3;
    }
}

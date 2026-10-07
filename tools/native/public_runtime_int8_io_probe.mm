// Public API-only macOS26 compressed-IO feasibility, not a model/E2E benchmark.
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#include <dispatch/dispatch.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <functional>
#include <stdexcept>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
void check(bool value,const char *why) {if(!value)throw std::runtime_error(why);}
struct Slot {
    CVPixelBufferRef pixel=nullptr;
    MLMultiArray *array=nil;
    int rows,cols,item;
    Slot(int r,int c,bool i8):rows(r),cols(c),item(i8?1:2) {
        NSDictionary *attributes=@{(id)kCVPixelBufferIOSurfacePropertiesKey:@{}};
        check(CVPixelBufferCreate(kCFAllocatorDefault,c,r,i8?kCVPixelFormatType_OneComponent8:kCVPixelFormatType_OneComponent16Half,
              (__bridge CFDictionaryRef)attributes,&pixel)==kCVReturnSuccess && pixel,"public pixel buffer allocation failed");
        array=[[MLMultiArray alloc] initWithPixelBuffer:pixel shape:@[@(r),@(c)]];
        if(!array || !CVPixelBufferGetIOSurface(pixel) || array.dataType!=(i8?MLMultiArrayDataTypeInt8:MLMultiArrayDataTypeFloat16)) {
            CVPixelBufferRelease(pixel);pixel=nullptr;throw std::runtime_error("public INT8/FP16 IOSurface array binding failed");
        }
    }
    ~Slot(){if(pixel)CVPixelBufferRelease(pixel);}
    Slot(const Slot &)=delete;
    size_t bytes()const{return CVPixelBufferGetBytesPerRow(pixel)*size_t(rows);}
    template<class F>void access(F function) {
        check(CVPixelBufferLockBaseAddress(pixel,0)==kCVReturnSuccess,"public input lock failed");
        try {function(CVPixelBufferGetBaseAddress(pixel),CVPixelBufferGetBytesPerRow(pixel));}
        catch(...) {CVPixelBufferUnlockBaseAddress(pixel,0);throw;}
        check(CVPixelBufferUnlockBaseAddress(pixel,0)==kCVReturnSuccess,"public input unlock failed");
    }
};
int code(int feature,int row){return (feature*13+row*17)%256-128;}
NSDictionary *compute_plan(NSURL *url,MLModelConfiguration *configuration) {
    // Public anticipated placement only. Never label this an execution trace.
    __block MLComputePlan *plan=nil;
    dispatch_semaphore_t ready=dispatch_semaphore_create(0);
    [MLComputePlan loadContentsOfURL:url configuration:configuration completionHandler:^(MLComputePlan *value,NSError *) {
        plan=value;dispatch_semaphore_signal(ready);
    }];
    if(dispatch_semaphore_wait(ready,dispatch_time(DISPATCH_TIME_NOW,15*NSEC_PER_SEC)))
        return @{@"available":@NO,@"reason":@"deadline",@"scope":@"anticipated placement; no execution trace"};
    if(!plan || !plan.modelStructure.program)
        return @{@"available":@NO,@"reason":@"plan/program unavailable",@"scope":@"anticipated placement; no execution trace"};
    MLComputePlan *completed_plan=plan;
    NSMutableArray *operations=[NSMutableArray array];
    std::function<void(MLModelStructureProgramBlock*,int)> visit=[&](MLModelStructureProgramBlock *block,int depth) {
        check(depth<16 && operations.count<10000,"unexpected compute-plan extent");
        for(MLModelStructureProgramOperation *op in block.operations) {
            auto device=[completed_plan computeDeviceUsageForMLProgramOperation:op].preferredComputeDevice;
            NSString *kind=[device isKindOfClass:MLNeuralEngineComputeDevice.class]?@"neural_engine":
                [device isKindOfClass:MLCPUComputeDevice.class]?@"cpu":@"other_or_unknown";
            [operations addObject:@{@"operator":op.operatorName,@"preferred_device":kind}];
            for(MLModelStructureProgramBlock *nested in op.blocks)visit(nested,depth+1);
        }
    };
    for(MLModelStructureProgramFunction *function in plan.modelStructure.program.functions.allValues)visit(function.block,0);
    return @{@"available":@YES,@"operations":operations,@"scope":@"public anticipated placement; not actual residency or INT8 MAC proof"};
}
}
int main(int argc,char **argv) {
    if(argc!=3)return 2;
    @autoreleasepool {
        try {
            const std::string policy=argv[2];check(policy=="cpu" || policy=="ne","policy requires cpu or ne");
            NSError *error=nil;
            NSData *data=[NSData dataWithContentsOfFile:@(argv[1])];
            check(data && data.length<(1u<<20),"bounded probe manifest missing");
            NSDictionary *spec=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            check([spec isKindOfClass:NSDictionary.class] && [spec[@"schema"] isEqual:@"tc-public-runtime-int8-io-probe-v1"],"public INT8 probe manifest schema mismatch");
            const int rows=[spec[@"rows"] intValue],hidden=[spec[@"hidden"] intValue],width=[spec[@"width"] intValue];
            const bool i8=[spec[@"arm"] isEqual:@"int8"];
            check(i8 || [spec[@"arm"] isEqual:@"fp16"],"public probe arm invalid");
            check(rows>0 && rows<=4224 && hidden>0 && hidden<=4096 && width>0 && width<=16384,"public probe geometry invalid");
            check(2*(uint64_t(hidden)*rows+uint64_t(width)*hidden+uint64_t(width)*rows)<=(512ull<<20),"public probe slots exceed bounded fixture allowance");
            Slot x(hidden,rows,i8),w(width,hidden,i8),y(width,rows,false);
            MLModelConfiguration *config=[MLModelConfiguration new];
            config.computeUnits=policy=="cpu"?MLComputeUnitsCPUOnly:MLComputeUnitsCPUAndNeuralEngine;
            NSString *directory=[@(argv[1]) stringByDeletingLastPathComponent];
            NSURL *model_url=[NSURL fileURLWithPath:[directory stringByAppendingPathComponent:@"graph.mlmodelc"]];
            auto start=Clock::now();
            MLModel *model=[MLModel modelWithContentsOfURL:model_url
                                           configuration:config error:&error];
            if(!model)throw std::runtime_error(std::string("public model load failed: ")+(error?error.localizedDescription.UTF8String:"unknown"));
            const double load=std::chrono::duration<double>(Clock::now()-start).count();
            auto planned=compute_plan(model_url,config);
            NSData *encoded_plan=[NSJSONSerialization dataWithJSONObject:planned options:NSJSONWritingSortedKeys error:&error];
            check(encoded_plan!=nil,"compute plan receipt serialization failed");
            NSString *plan_json=[[NSString alloc] initWithData:encoded_plan encoding:NSUTF8StringEncoding];
            const auto expected=i8?MLMultiArrayDataTypeInt8:MLMultiArrayDataTypeFloat16;
            check(model.modelDescription.inputDescriptionsByName.count==2 && model.modelDescription.outputDescriptionsByName.count==1,"actual public feature count differs");
            for(NSString *name in @[@"x",@"w"])
                check(model.modelDescription.inputDescriptionsByName[name].multiArrayConstraint.dataType==expected,"actual public input datatype differs");
            check(model.modelDescription.outputDescriptionsByName[@"y"].multiArrayConstraint.dataType==MLMultiArrayDataTypeFloat16,"actual public output datatype differs");
            check([model.modelDescription.inputDescriptionsByName[@"x"].multiArrayConstraint.shape isEqual:x.array.shape] &&
                  [model.modelDescription.inputDescriptionsByName[@"w"].multiArrayConstraint.shape isEqual:w.array.shape] &&
                  [model.modelDescription.outputDescriptionsByName[@"y"].multiArrayConstraint.shape isEqual:y.array.shape],"actual public geometry differs");
            MLDictionaryFeatureProvider *features=[[MLDictionaryFeatureProvider alloc] initWithDictionary:
                @{@"x":[MLFeatureValue featureValueWithMultiArray:x.array],@"w":[MLFeatureValue featureValueWithMultiArray:w.array]} error:&error];
            check(features!=nil,"public feature provider failed");
            MLPredictionOptions *options=[MLPredictionOptions new];options.outputBackings=@{@"y":y.array};
            x.access([&](void *base,size_t pitch) {
                std::memset(base,0,size_t(hidden)*pitch);
                for(int c=0;c<hidden;++c)for(int r=0;r<rows;++r) {
                    auto *at=static_cast<char*>(base)+size_t(c)*pitch;
                    if(i8)reinterpret_cast<int8_t*>(at)[r]=int8_t(code(c,r));
                    else reinterpret_cast<_Float16*>(at)[r]=_Float16(code(c,r)/128.f);
                }
            });
            uint64_t calls=0,checked=0,backing_hits=0;
            double max_abs=0;std::vector<double> samples;
            for(int variant:{64,-64,64}) {
                w.access([&](void *base,size_t pitch) {
                    std::memset(base,0,size_t(width)*pitch);
                    for(int n=0;n<width;++n) {
                        auto *at=static_cast<char*>(base)+size_t(n)*pitch;
                        if(i8)reinterpret_cast<int8_t*>(at)[n%hidden]=int8_t(variant);
                        else reinterpret_cast<_Float16*>(at)[n%hidden]=_Float16(variant/128.f);
                    }
                });
                for(int repetition=0;repetition<7;++repetition) {
                    start=Clock::now();
                    id<MLFeatureProvider> result=[model predictionFromFeatures:features options:options error:&error];
                    const double seconds=std::chrono::duration<double>(Clock::now()-start).count();
                    if(!result)throw std::runtime_error(std::string("public prediction failed: ")+(error?error.localizedDescription.UTF8String:"unknown"));
                    ++calls;if(repetition>=2)samples.push_back(seconds);
                    MLMultiArray *output=[result featureValueForName:@"y"].multiArrayValue;
                    check(output && output.dataType==MLMultiArrayDataTypeFloat16 && [output.shape isEqual:y.array.shape],"public prediction output interface changed");
                    if(output.pixelBuffer==y.pixel)++backing_hits;
                    const size_t rs=output.strides[0].unsignedLongLongValue,cs=output.strides[1].unsignedLongLongValue;
                    check(rs && cs,"public output strides invalid");
                    __block bool valid=true;__block double error_max=0;
                    [output getBytesWithHandler:^(const void *bytes,NSInteger size) {
                        const size_t extent=size_t(width-1)*rs+size_t(rows-1)*cs+1;
                        if(!bytes || size<0 || size_t(size)/2<extent){valid=false;return;}
                        const auto *values=static_cast<const _Float16*>(bytes);
                        for(int n=0;n<width;++n)for(int r=0;r<rows;++r) {
                            const float actual=float(values[size_t(n)*rs+size_t(r)*cs]);
                            const float reference=float(_Float16(code(n%hidden,r)*variant/16384.f));
                            if(!std::isfinite(actual)){valid=false;return;}
                            error_max=std::max(error_max,double(std::abs(actual-reference)));
                        }
                    }];
                    check(valid && error_max==0,"public signed INT8/FP16 dense-source sparse oracle differs");
                    max_abs=std::max(max_abs,error_max);checked+=uint64_t(width)*rows;
                }
            }
            auto sorted=samples;std::sort(sorted.begin(),sorted.end());
            std::cout<<std::setprecision(17)<<"{\"schema\":\"tc-public-runtime-int8-io-observation-v1\",\"arm\":\""<<(i8?"int8":"fp16")<<"\",\"policy\":\""<<policy
                <<"\",\"rows\":"<<rows<<",\"hidden\":"<<hidden<<",\"width\":"<<width<<",\"actual_input_dtype\":\""<<(i8?"int8":"fp16")
                <<"\",\"calls\":"<<calls<<",\"checked_values\":"<<checked<<",\"max_abs\":"<<max_abs<<",\"signed_minus128_checked\":true,\"weight_switch_aba_checked\":true"
                <<",\"output_backing_identity_hits\":"<<backing_hits<<",\"slot_bytes\":"<<x.bytes()+w.bytes()+y.bytes()
                <<",\"load_seconds\":"<<load<<",\"prediction_median_seconds\":"<<sorted[sorted.size()/2]<<",\"samples\":[";
            for(size_t i=0;i<samples.size();++i)std::cout<<(i?",":"")<<samples[i];
            std::cout<<"],\"compute_plan\":"<<plan_json.UTF8String<<",\"arithmetic_evidence\":\"unknown\",\"observed_ane_residency\":\"unknown\",\"production_qualified\":false,\"scope\":\"actual public compressed IOSurface IO and sparse source oracle; no hardware arithmetic/overlap/model qualification\"}"<<std::endl;
        }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
    }
}

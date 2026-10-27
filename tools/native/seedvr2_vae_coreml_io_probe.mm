// Offline public-Core-ML fixed-convolution IO screen, never a VAE backend.
// Reuses the provider and buffers; verifies every prediction against FP32.
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#include <mach/mach.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
void check(bool value, const char *why) { if (!value) throw std::runtime_error(why); }
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
double median(std::vector<double> values) {
    check(!values.empty(), "empty timing cohort");
    std::sort(values.begin(),values.end());
    return (values[(values.size()-1)/2]+values[values.size()/2])/2;
}
uint64_t footprint() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    check(task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count)==KERN_SUCCESS,
          "process footprint unavailable");
    return info.phys_footprint;
}
NSArray<NSNumber *> *shape(int channels,int rows,int cols) { return @[@1,@(channels),@(rows),@(cols)]; }
struct Slot {
    std::vector<_Float16> storage;
    MLMultiArray *array = nil;
    CVPixelBufferRef pixel = nullptr;
    int channels, rows, cols;
    Slot(int c,int h,int w,const std::string &mode):channels(c),rows(h),cols(w) {
        NSError *error = nil;
        auto dimensions = shape(c,h,w);
        if (mode=="surface") {
            NSDictionary *attributes = @{(id)kCVPixelBufferIOSurfacePropertiesKey:@{}};
            check(CVPixelBufferCreate(kCFAllocatorDefault,w,c*h,kCVPixelFormatType_OneComponent16Half,
                  (__bridge CFDictionaryRef)attributes,&pixel)==kCVReturnSuccess && pixel,
                  "FP16 IOSurface allocation failed");
            array = [[MLMultiArray alloc] initWithPixelBuffer:pixel shape:dimensions];
            if (!array || !CVPixelBufferGetIOSurface(pixel)) {
                CVPixelBufferRelease(pixel); pixel=nullptr;
                throw std::runtime_error("FP16 IOSurface array binding failed");
            }
        } else if (mode=="wrapped") {
            storage.resize(size_t(c)*h*w);
            array = [[MLMultiArray alloc] initWithDataPointer:storage.data() shape:dimensions
                dataType:MLMultiArrayDataTypeFloat16 strides:@[@(size_t(c)*h*w),@(h*w),@(w),@1]
                deallocator:^(void *) {} error:&error];
        } else {
            check(mode=="array", "unknown IO mode");
            array = [[MLMultiArray alloc] initWithShape:dimensions dataType:MLMultiArrayDataTypeFloat16 error:&error];
        }
        check(array && array.dataType==MLMultiArrayDataTypeFloat16 && [array.shape isEqual:dimensions],
              "FP16 slot allocation/binding failed");
    }
    ~Slot() { array=nil; if(pixel)CVPixelBufferRelease(pixel); }
    Slot(const Slot &)=delete;
    uint64_t bytes() const {
        if(pixel)return CVPixelBufferGetBytesPerRow(pixel)*uint64_t(channels)*rows;
        return (array.strides[1].unsignedLongLongValue*(channels-1)+
            array.strides[2].unsignedLongLongValue*(rows-1)+
            array.strides[3].unsignedLongLongValue*(cols-1)+1)*2;
    }
    void fill(const _Float16 *input,float sign) {
        if(pixel)check(CVPixelBufferLockBaseAddress(pixel,0)==kCVReturnSuccess,"input surface lock failed");
        auto *base = pixel ? static_cast<_Float16 *>(CVPixelBufferGetBaseAddress(pixel)) :
            static_cast<_Float16 *>(array.dataPointer);
        const size_t cs=array.strides[1].unsignedLongLongValue, rs=array.strides[2].unsignedLongLongValue,
                     ws=array.strides[3].unsignedLongLongValue;
        check(base && cs && rs && ws, "invalid input strides");
        for(int c=0;c<channels;++c)for(int r=0;r<rows;++r)for(int w=0;w<cols;++w)
            base[size_t(c)*cs+size_t(r)*rs+size_t(w)*ws]=_Float16(sign*float(input[(size_t(c)*rows+r)*cols+w]));
        if(pixel)check(CVPixelBufferUnlockBaseAddress(pixel,0)==kCVReturnSuccess,"input surface unlock failed");
    }
};
NSDictionary *compute_plan(NSURL *url,MLModelConfiguration *configuration) {
    __block MLComputePlan *plan=nil;
    dispatch_semaphore_t ready=dispatch_semaphore_create(0);
    [MLComputePlan loadContentsOfURL:url configuration:configuration completionHandler:^(MLComputePlan *value,NSError *) {
        plan=value; dispatch_semaphore_signal(ready);
    }];
    if(dispatch_semaphore_wait(ready,dispatch_time(DISPATCH_TIME_NOW,15*NSEC_PER_SEC)) || !plan.modelStructure.program)
        return @{@"available":@NO,@"scope":@"anticipated placement only"};
    MLComputePlan *completed=plan;
    NSMutableArray *operations=[NSMutableArray array];
    std::function<void(MLModelStructureProgramBlock *,int)> visit=[&](MLModelStructureProgramBlock *block,int depth) {
        check(depth<16 && operations.count<10000,"unbounded compute plan");
        for(MLModelStructureProgramOperation *operation in block.operations) {
            auto usage=[completed computeDeviceUsageForMLProgramOperation:operation];
            auto device=usage.preferredComputeDevice;
            NSString *kind=[device isKindOfClass:MLNeuralEngineComputeDevice.class]?@"neural_engine":
                [device isKindOfClass:MLCPUComputeDevice.class]?@"cpu":@"other_or_unknown";
            [operations addObject:@{@"operator":operation.operatorName,@"preferred":kind}];
            for(MLModelStructureProgramBlock *nested in operation.blocks)visit(nested,depth+1);
        }
    };
    for(MLModelStructureProgramFunction *function in plan.modelStructure.program.functions.allValues)visit(function.block,0);
    return @{@"available":@YES,@"operations":operations,@"scope":@"anticipated; not physical ANE occupancy"};
}
}

int main(int argc,char **argv) {
    if(argc!=5){std::cerr<<"usage: seedvr2-vae-coreml-io-probe FIXTURE.json cpu|ne array|wrapped|surface backing|none\n";return 2;}
    @autoreleasepool {
        try {
            const std::string policy=argv[2], mode=argv[3], backing=argv[4];
            check(policy=="cpu" || policy=="ne","invalid compute policy");
            check(backing=="backing" || backing=="none","invalid backing policy");
            NSError *error=nil;
            NSData *fixture_data=[NSData dataWithContentsOfFile:@(argv[1])];
            check(fixture_data && fixture_data.length<65536,"bounded fixture missing");
            NSDictionary *spec=[NSJSONSerialization JSONObjectWithData:fixture_data options:0 error:&error];
            check([spec isKindOfClass:NSDictionary.class] &&
                [spec[@"schema"] isEqual:@"seedvr2-vae-native-io-fixture-v1"],"fixture schema mismatch");
            const int cin=[spec[@"cin"] intValue], cout=[spec[@"cout"] intValue],
                hi=[spec[@"input_rows"] intValue], wi=[spec[@"input_cols"] intValue],
                ho=[spec[@"rows"] intValue], wo=[spec[@"cols"] intValue], warm=[spec[@"iterations"] intValue];
            check(cin>0 && cin<=1536 && cout>0 && cout<=4096 && hi>0 && hi<=258 && wi>0 && wi<=258 &&
                  ho>0 && ho<=256 && wo>0 && wo<=256 && warm>=3 && warm<=30,"invalid bounded geometry");
            check((uint64_t(cin)*hi*wi*2+uint64_t(cout)*ho*wo*6)<=(512ull<<20),"fixture exceeds 512-MiB bound");
            NSData *input=[NSData dataWithContentsOfFile:spec[@"input"]];
            NSData *reference=[NSData dataWithContentsOfFile:spec[@"reference"]];
            check(input.length==uint64_t(cin)*hi*wi*2 && reference.length==uint64_t(cout)*ho*wo*4,
                  "fixture bytes differ from fixed geometry");
            auto config=[MLModelConfiguration new];
            config.computeUnits=policy=="cpu"?MLComputeUnitsCPUOnly:MLComputeUnitsCPUAndNeuralEngine;
            NSURL *url=[NSURL fileURLWithPath:spec[@"compiled"]];
            auto start=Clock::now();
            MLModel *model=[MLModel modelWithContentsOfURL:url configuration:config error:&error];
            if(!model)throw std::runtime_error(std::string("Core ML load failed: ")+(error?error.localizedDescription.UTF8String:"unknown"));
            const double load_seconds=elapsed(start);
            auto plan=compute_plan(url,config);
            check(model.modelDescription.inputDescriptionsByName.count==1 && model.modelDescription.outputDescriptionsByName.count==1,
                  "unexpected model feature count");
            for(NSString *name in @[@"x",@"y"]) {
                auto constraint=[name isEqual:@"x"]?model.modelDescription.inputDescriptionsByName[name].multiArrayConstraint:
                    model.modelDescription.outputDescriptionsByName[name].multiArrayConstraint;
                check(constraint.dataType==MLMultiArrayDataTypeFloat16 &&
                    [constraint.shape isEqual:[name isEqual:@"x"]?shape(cin,hi,wi):shape(cout,ho,wo)],"model ABI changed");
            }
            Slot x(cin,hi,wi,mode), y(cout,ho,wo,mode);
            auto features=[[MLDictionaryFeatureProvider alloc] initWithDictionary:
                @{@"x":[MLFeatureValue featureValueWithMultiArray:x.array]} error:&error];
            check(features!=nil,"provider allocation failed");
            auto options=[MLPredictionOptions new];
            if(backing=="backing")options.outputBackings=@{@"y":y.array};
            NSMutableArray *times=[NSMutableArray array], *reads=[NSMutableArray array], *fills=[NSMutableArray array];
            uint64_t calls=0,hits=0,checked=0,peak=footprint();
            double worst_l2=0,minimum_cosine=1,maximum_abs=0,cold=0;
            std::vector<double> seconds;
            // Each mutation follows completion of every prior consumer. This
            // also detects stale input/output buffers after provider reuse.
            for(float sign:{1.f,-1.f,1.f}) {
                start=Clock::now();x.fill(static_cast<const _Float16 *>(input.bytes),sign);
                [fills addObject:@(elapsed(start))];
                for(int repetition=0;repetition<warm+2;++repetition) {
                    @autoreleasepool {
                        start=Clock::now();
                        id<MLFeatureProvider> result=[model predictionFromFeatures:features options:options error:&error];
                        const double prediction=elapsed(start);
                        if(!result)throw std::runtime_error(std::string("Core ML predict failed: ")+(error?error.localizedDescription.UTF8String:"unknown"));
                        if(calls==0)cold=prediction;
                        ++calls;
                        auto output=[result featureValueForName:@"y"].multiArrayValue;
                        check(output && output.dataType==MLMultiArrayDataTypeFloat16 && [output.shape isEqual:shape(cout,ho,wo)],
                              "output ABI changed");
                        if(backing=="backing" && (y.pixel?output.pixelBuffer==y.pixel:output.dataPointer==y.array.dataPointer))++hits;
                        const size_t cs=output.strides[1].unsignedLongLongValue,rs=output.strides[2].unsignedLongLongValue,
                            ws=output.strides[3].unsignedLongLongValue;
                        __block bool valid=cs && rs && ws;
                        __block double nx=0,ny=0,dot=0,error_sum=0,max_abs=0,read_seconds=0;
                        start=Clock::now();
                        [output getBytesWithHandler:^(const void *bytes,NSInteger size) {
                            // Callback entry measures only making output host
                            // accessible; the CPU quality scan is not timed.
                            read_seconds=elapsed(start);
                            const size_t extent=size_t(cout-1)*cs+size_t(ho-1)*rs+size_t(wo-1)*ws+1;
                            if(!bytes || size<0 || size_t(size)/2<extent){valid=false;return;}
                            const auto *actual=static_cast<const _Float16 *>(bytes);
                            const auto *expected=static_cast<const float *>(reference.bytes);
                            for(int c=0;c<cout;++c)for(int r=0;r<ho;++r)for(int w=0;w<wo;++w) {
                                const double a=double(actual[size_t(c)*cs+size_t(r)*rs+size_t(w)*ws]),
                                    b=sign*double(expected[(size_t(c)*ho+r)*wo+w]);
                                if(!std::isfinite(a) || !std::isfinite(b)){valid=false;continue;}
                                nx+=b*b;ny+=a*a;dot+=a*b;error_sum+=(a-b)*(a-b);max_abs=std::max(max_abs,std::abs(a-b));
                            }
                        }];
                        const double l2=std::sqrt(error_sum/std::max(nx,1e-20)),cosine=dot/std::max(std::sqrt(nx*ny),1e-20);
                        check(valid && l2<=.05 && cosine>=.999,"unchanged FP32 component quality gate failed");
                        worst_l2=std::max(worst_l2,l2);minimum_cosine=std::min(minimum_cosine,cosine);maximum_abs=std::max(maximum_abs,max_abs);
                        checked+=uint64_t(cout)*ho*wo;peak=std::max(peak,footprint());
                        if(repetition>=2){seconds.push_back(prediction);[times addObject:@(prediction)];[reads addObject:@(read_seconds)];}
                    }
                }
            }
            NSDictionary *report=@{@"schema":@"seedvr2-vae-native-io-observation-v1",@"policy":@(policy.c_str()),
                @"io":@(mode.c_str()),@"output_backing_requested":@(backing=="backing"),@"calls":@(calls),
                @"checked_values":@(checked),@"output_backing_identity_hits":@(hits),@"slot_bytes":@(x.bytes()+y.bytes()),
                @"sampled_process_footprint_max_bytes":@(peak),@"load_seconds":@(load_seconds),@"cold_seconds":@(cold),
                @"warm_seconds":times,@"warm_median_seconds":@(median(seconds)),@"host_access_seconds":reads,
                @"input_fill_seconds":fills,@"compute_plan":plan,@"quality":@{@"passed":@YES,@"finite":@YES,
                @"relative_l2":@(worst_l2),@"cosine":@(minimum_cosine),@"max_abs":@(maximum_abs)},
                @"signed_input_aba_checked":@YES,@"physical_ane_occupancy_measured":@NO,@"qualified":@NO,
                @"scope":@"prepared original-FP16 convolution, native reusable IO; no GPU pack/sync/merge, whole VAE or runtime occupancy claim"};
            NSData *encoded=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingSortedKeys error:&error];
            check(encoded!=nil,"report serialization failed");
            std::cout<<[[[NSString alloc] initWithData:encoded encoding:NSUTF8StringEncoding] UTF8String]<<'\n';
        } catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
    }
}

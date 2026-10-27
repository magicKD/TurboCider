#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <iostream>

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2) { std::cerr << "compiled Core ML model path required\n"; return 1; }
        auto configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        dispatch_semaphore_t ready = dispatch_semaphore_create(0);
        __block MLComputePlan *plan = nil;
        __block NSError *failure = nil;
        [MLComputePlan loadContentsOfURL:[NSURL fileURLWithPath:@(argv[1])] configuration:configuration
            completionHandler:^(MLComputePlan *value, NSError *error) {
                plan = value; failure = error; dispatch_semaphore_signal(ready);
            }];
        if (dispatch_semaphore_wait(ready, dispatch_time(DISPATCH_TIME_NOW, 60 * NSEC_PER_SEC)) != 0 || !plan) {
            std::cerr << (failure ? failure.localizedDescription.UTF8String : "compute plan unavailable") << '\n'; return 1;
        }
        NSMutableArray *operations = [NSMutableArray array];
        for (MLModelStructureProgramOperation *operation in plan.modelStructure.program.functions[@"main"].block.operations) {
            auto usage = [plan computeDeviceUsageForMLProgramOperation:operation];
            const bool neural = [usage.preferredComputeDevice isKindOfClass:MLNeuralEngineComputeDevice.class];
            [operations addObject:@{@"operator": operation.operatorName,
                @"preferred_device": neural ? @"NeuralEngine" : @"CPU_or_unknown"}];
        }
        NSDictionary *record = @{@"compute_units": @"cpuAndNeuralEngine", @"operations": operations,
            @"scope": @"anticipated device assignment, not measured runtime occupancy"};
        auto data = [NSJSONSerialization dataWithJSONObject:record options:NSJSONWritingSortedKeys error:nil];
        std::cout.write((const char *)data.bytes, data.length); std::cout << '\n';
        return 0;
    }
}

#include "models/z_image/coreml_generation.hpp"
#include "core/common.hpp"
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <cstring>

namespace fs = std::filesystem;
int main(int argc, char **argv) {
    tc::require(argc == 3, "expected artifact and private parent");
    auto generation = tc::z_image::CoreMLGeneration::import_tree(argv[1], argv[2]);
    auto path = generation->root();
    generation->revalidate();
    @autoreleasepool {
        NSError *error = nil;
        auto config = [MLModelConfiguration new];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        auto model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(path.c_str())]
                                     configuration:config error:&error];
        tc::require(model != nil, "cannot load imported compiled model: " + std::string(error ? error.localizedDescription.UTF8String : "unknown"));
        generation->revalidate();
        auto inputConstraint = model.modelDescription.inputDescriptionsByName[@"x"].multiArrayConstraint;
        auto outputConstraint = model.modelDescription.outputDescriptionsByName[@"y"].multiArrayConstraint;
        NSArray *shape = @[@1, @3840, @1, @1088];
        tc::require([inputConstraint.shape isEqual:shape] && [outputConstraint.shape isEqual:shape] &&
                    inputConstraint.dataType == MLMultiArrayDataTypeFloat16 && outputConstraint.dataType == MLMultiArrayDataTypeFloat16,
                    "unexpected fixed FFN feature ABI");
        auto input = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat16 error:&error];
        tc::require(input != nil, "input allocation failed");
        // The fixture uses the previously exported bias-free FP16 FFN. Zero
        // input has an independent exact-zero oracle; this is not image quality.
        std::memset(input.dataPointer, 0, size_t(input.count) * 2);
        auto provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x":input} error:&error];
        tc::require(provider != nil, "feature provider failed");
        auto result = [model predictionFromFeatures:provider error:&error];
        tc::require(result != nil, "prediction from imported generation failed");
        auto output = [result featureValueForName:@"y"].multiArrayValue;
        tc::require(output != nil && [output.shape isEqual:shape] && output.dataType == MLMultiArrayDataTypeFloat16,
                    "unexpected output ABI");
        for (NSInteger i = 0; i < output.count; ++i)
            tc::require(output[i].doubleValue == 0.0, "zero-input FFN produced nonzero/nonfinite output");
        generation->revalidate();
    }
    // Core ML objects leave the autorelease pool before the generation owner.
    generation->revalidate();
    std::cout << "{\"predictions\":1,\"warmups\":0,\"zero_output_passed\":true,"
                 "\"compute_units\":\"CPU_AND_NE\",\"observed_ane_residency\":\"unknown\","
                 "\"copied_bytes\":" << generation->copied_bytes() << ",\"content_digest\":\""
              << generation->content_digest() << "\"}" << std::endl;
    generation.reset();
    tc::require(!fs::exists(path), "generation survived final owner release");
}

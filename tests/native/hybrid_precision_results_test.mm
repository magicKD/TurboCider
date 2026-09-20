#include "platform/apple/bridge.hpp"
#include <iostream>
using namespace tc;
int main() {
 @autoreleasepool {
    Request request; request.model="z-image-turbo";request.prompt="fox";request.audio=false;
    request.width=512;request.height=512;request.steps=9;request.execution="gpu_ane";
    request.ane_manifest="/nonexistent/metadata-only.json";request.allow_approximation=true;
    const auto plan=make_plan(request);
    auto proposed=to_dictionary(plan);
    require([proposed[@"precision"] isEqual:@"bf16_gpu+coreml_mlp_fp16_io"],"plan guessed unloaded weight precision");
    for (const std::string variant : {"unknown","fp16","int8_pc"}) {
        HybridMetrics metrics;metrics.weight_variant=variant;
        auto encoded=to_dictionary(metrics);
        require([encoded[@"weight_variant"] isEqual:@(variant.c_str())],"metrics variant lost");
        const std::string expected=variant=="fp16"?"bf16_gpu+fp16_mlp_fp16_io":
            variant=="int8_pc"?"bf16_gpu+int8_mlp_fp16_io":"bf16_gpu+coreml_mlp_fp16_io";
        RunResult result;result.request=request;result.plan=plan;
        result.hybrid=metrics;result.encoder_hybrid=metrics;result.backend="mlx_cpp_metal+coreml";
        result.precision=hybrid_precision_label(metrics);
        for (bool prepared : {false,true}) {
            result.prepared=prepared;auto output=to_dictionary(result);
            require([output[@"runtime_precision"] isEqual:@(expected.c_str())] &&
                    [output[@"encoder_runtime_precision"] isEqual:@(expected.c_str())] &&
                    [output[@"plan"][@"precision"] isEqual:@(expected.c_str())] &&
                    [output[@"plan"][@"encoder_precision"] isEqual:@(expected.c_str())],"runtime precision mismatch");
        }
    }
    std::cout<<"hybrid plan/metrics/generated/prepared precision PASS\n";
 }
}

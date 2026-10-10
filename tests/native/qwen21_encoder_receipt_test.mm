#include "../../native/platform/apple/bridge.hpp"
#include <iostream>

int main() {
    @autoreleasepool {
        for (const char *backend : {"public_coreml","private_ane"})
        for (const char *path : {"fp16","w8a8_hadamard"})
        for (int calls : {0,36}) {
            tc::RunResult result;
            result.request.model="qwen-image-2.1";result.request.operation="image.edit";
            result.request.width=result.request.height=512;result.request.steps=40;
            result.request.execution="gpu";result.request.residency="resident";
            result.plan.request=result.request;result.plan.recipe.model=result.request.model;
            tc::HybridMetrics metrics;metrics.block_count=36;
            metrics.mlp_output_kind="runtime_weight_swiglu";
            metrics.runtime_weight_backend=backend;metrics.runtime_weight_data_path=path;
            metrics.runtime_calls=calls;metrics.session_released_after_encoding=true;
            result.encoder_hybrid=metrics;
            NSDictionary *data=tc::to_dictionary(result),*plan=data[@"plan"];
            NSString *execution=calls?@"gpu_ane_experimental":@"gpu";
            NSString *runtime=calls ? (std::string(backend)=="private_ane" ?
                @"mlx_cpp_metal+private_ane_runtime_weight_experimental" : @"mlx_cpp_metal+coreml_runtime_weight") : @"mlx_cpp_metal";
            NSString *precision=calls ? (std::string(path)=="w8a8_hadamard" ?
                @"bf16_gpu+runtime_w8a8_ffn_bf16_io" : @"bf16_gpu+runtime_fp16_ffn_bf16_io") : @"bf16";
            tc::require([data[@"encoder_execution"] isEqual:execution] && [plan[@"encoder_execution"] isEqual:execution] &&
                [data[@"encoder_runtime_backend"] isEqual:runtime] && [plan[@"encoder_backend"] isEqual:runtime] &&
                [data[@"encoder_runtime_precision"] isEqual:precision] && [plan[@"encoder_precision"] isEqual:precision],
                "Qwen encoder receipt intent/actual/backend/precision mismatch");
            tc::require([data[@"encoder_hybrid"][@"runtime_calls_session_total"] intValue]==calls &&
                [data[@"encoder_hybrid"][@"block_count"] intValue]==36 &&
                [data[@"encoder_hybrid"][@"session_released_after_encoding"] boolValue],"encoder evidence lost");
            // Cache hits carry no new attempt/calls, even when the plan used
            // an explicit encoder manifest to produce the cached condition.
            result.encoder_hybrid.reset();result.prompt_cache_hit=true;
            data=tc::to_dictionary(result);
            tc::require([data[@"encoder_execution"] isEqual:@"gpu"] &&
                [data[@"encoder_hybrid"] count]==0,"cache hit replayed encoder execution");
            // A reused graph's historical counters are not this request's
            // execution. Exercise both zero-call decline and real execution.
            result.prompt_cache_hit=false;result.encoder_hybrid=metrics;
            result.encoder_hybrid->runtime_calls=72;
            result.encoder_runtime_reuse=tc::EncoderRuntimeReuseMetrics{true,true,true,0,123456};
            data=tc::to_dictionary(result);
            tc::require([data[@"encoder_execution"] isEqual:@"gpu"] &&
                [data[@"plan"][@"encoder_execution"] isEqual:@"gpu"] &&
                [data[@"encoder_runtime_reuse"][@"actual_calls_this_request"] intValue]==0 &&
                [data[@"encoder_hybrid"][@"runtime_calls_session_total"] intValue]==72,
                "cumulative encoder calls masqueraded as new execution");
            result.encoder_runtime_reuse->calls_this_request=36;
            data=tc::to_dictionary(result);
            tc::require([data[@"encoder_execution"] isEqual:@"gpu_ane_experimental"] &&
                [data[@"encoder_runtime_reuse"][@"executor_reused"] boolValue] &&
                [data[@"encoder_runtime_reuse"][@"executor_retained"] boolValue] &&
                [data[@"encoder_runtime_reuse"][@"retained_estimated_bytes"] intValue]==123456,
                "retained executor request evidence lost");
        }
        std::cout << "PASS Qwen encoder receipts: actual calls, Private/Public, W8/FP16, release, cache hits\n";
    }
}

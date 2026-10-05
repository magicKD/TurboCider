#include "../../native/platform/apple/bridge.hpp"
#include <iostream>

int main() {
    @autoreleasepool {
        for (const auto &variant : {"runtime_fp16", "runtime_w8a8", "future_representation"}) {
            for (const auto &kind : {"runtime_weight_swiglu", "runtime_weight_swiglu_lora_inputs"}) {
                tc::HybridMetrics metrics;
                metrics.weight_variant = variant;
                metrics.mlp_output_kind = kind;
                metrics.runtime_weight_backend = "private_ane";
                metrics.runtime_weight_io_path = "gpu_iosurface";
                metrics.runtime_weight_data_path = "w8a8_hadamard";
                metrics.runtime_weight_source_recipe = "sylvester-dh-b128-b512-rne-norm-f16-v2";
                metrics.runtime_weight_device_io_calls = 3;
                metrics.runtime_weight_lora_channel_range_calls = 1;
                metrics.runtime_weight_lora_channel_full_calls = 2;
                metrics.runtime_weight_a8_lookahead_enabled = true;
                metrics.runtime_weight_a8_prefetches = 2;
                metrics.runtime_weight_a8_wait_seconds = .001;
                metrics.runtime_weight_stage_specialized = true;
                metrics.runtime_weight_stage_pipeline_variants = 4;
                metrics.runtime_calls = 3;
                metrics.runtime_weight_deferred_join_enabled=true;
                metrics.runtime_weight_channel_blocks=2;
                metrics.runtime_weight_deferred_join_blocks=2;
                auto serialized = tc::to_dictionary(metrics);
                id runtime = serialized[@"runtime_weight"];
                if (![runtime isKindOfClass:[NSDictionary class]] ||
                    ![runtime[@"data_path"] isEqual:@"w8a8_hadamard"] ||
                    ![runtime[@"executor_backend"] isEqual:@"private_ane"] ||
                    ![runtime[@"io_path"] isEqual:@"gpu_iosurface"] ||
                    ![runtime[@"source_recipe"] isEqual:@"sylvester-dh-b128-b512-rne-norm-f16-v2"] ||
                    [runtime[@"device_io_calls_session_total"] unsignedLongLongValue] != 3 ||
                    [runtime[@"lora_channel_range_calls_session_total"] unsignedLongLongValue] != 1 ||
                    [runtime[@"lora_channel_full_calls_session_total"] unsignedLongLongValue] != 2 ||
                    ![runtime[@"a8_lookahead_enabled"] boolValue] ||
                    [runtime[@"a8_prefetches_session_total"] unsignedLongLongValue] != 2 ||
                    [runtime[@"a8_wait_seconds_session_total"] doubleValue] != .001 ||
                    ![runtime[@"stage_specialized"] boolValue] ||
                    [runtime[@"stage_pipeline_variants"] unsignedLongLongValue] != 4 ||
                    ![runtime[@"deferred_channel_join_enabled"] boolValue] ||
                    [runtime[@"deferred_channel_join_blocks_session_total"] unsignedLongLongValue] != 2 ||
                    ![runtime[@"post_join_scope"] isEqual:@"host_graph_construction_deferred_gpu_consumption"] ||
                    ![serialized[@"provenance"] hasPrefix:@"checkpoint-independent"])
                    return 1;
                for(uint64_t count:{0u,1u,2u}) {
                    metrics.runtime_weight_deferred_join_blocks=count;
                    serialized=tc::to_dictionary(metrics);
                    NSString *scope=count==0?@"evaluated_join_host_span":count==2?
                        @"host_graph_construction_deferred_gpu_consumption":@"mixed_evaluated_and_deferred_join_spans";
                    if(![serialized[@"runtime_weight"][@"post_join_scope"] isEqual:scope])return 1;
                }
                // Capability/memory fallback still requires a runtime receipt
                // even when no executor was successfully selected.
                metrics.runtime_weight_backend.clear();
                metrics.runtime_failed = true;
                serialized = tc::to_dictionary(metrics);
                if (![serialized[@"runtime_failed"] boolValue] || ![serialized[@"runtime_weight"] isKindOfClass:[NSDictionary class]]) return 1;
                metrics.mlp_output_kind = "fused_lora";
                serialized = tc::to_dictionary(metrics);
                if (serialized[@"runtime_weight"] != [NSNull null] ||
                    [serialized[@"provenance"] hasPrefix:@"checkpoint-independent"]) return 1;
            }
        }
        for (const auto &backend : {"public_coreml", "private_ane", ""}) {
            for (bool w8 : {false,true}) for (bool channels : {false,true}) {
                tc::RunResult result;
                result.request.model="z-image-turbo";result.request.operation="image.generate";
                result.request.execution="gpu_ane";result.request.hybrid_mlp_mode="runtime";
                result.request.compile_gpu=true;
                result.request.width=result.request.height=512;result.request.steps=8;
                result.plan.request=result.request;result.plan.recipe.model=result.request.model;
                tc::HybridMetrics metrics;
                metrics.mlp_output_kind="runtime_weight_swiglu";
                metrics.runtime_weight_backend=backend;
                metrics.runtime_weight_data_path=w8?"w8a8_hadamard":"fp16";
                metrics.runtime_weight_partition_axis=channels?"intermediate_channels":"rows";
                result.hybrid=metrics;
                NSDictionary *data=tc::to_dictionary(result);NSDictionary *plan=data[@"plan"];
                NSString *marker=w8?(channels?@"runtime_weight_w8a8_hadamard_channel_ffn":@"runtime_weight_w8a8_hadamard_token_row_ffn"):
                    (channels?@"runtime_weight_fp16_channel_ffn":@"runtime_weight_fp16_token_row_ffn");
                NSArray *labels=plan[@"algorithm_approximations"];
                if(std::string(backend).empty()) {
                    if([labels containsObject:marker] || ![plan[@"execution"] isEqual:@"gpu"] ||
                       ![data[@"gpu_graph"] isEqual:@"compiled_fused_blocks"])return 1;
                } else {
                    if(![labels containsObject:marker] || ![plan[@"runtime_weight_contract"][@"data_path"] isEqual:@(metrics.runtime_weight_data_path.c_str())])return 1;
                    if(channels && ![data[@"gpu_graph"] isEqual:@"runtime_weight_intermediate_channel_ffn"])return 1;
                    if(w8 && [labels containsObject:@"runtime_weight_fp16_token_row_ffn"])return 1;
                }
            }
        }
        std::cout << "PASS runtime receipt: FP16/W8A8/future representation, base/LoRA, capability failure and frozen provenance separation\n";
    }
}

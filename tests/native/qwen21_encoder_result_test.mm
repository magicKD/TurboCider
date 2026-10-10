#include "../../native/platform/apple/bridge.hpp"
#include <iostream>

int main() {
    @autoreleasepool {
        try {
            tc::Request request;request.model="qwen-image-2.1";request.width=request.height=512;
            request.execution="gpu";request.operation="image.generate";request.audio=false;
            tc::RunResult result;result.request=request;result.plan=tc::make_plan(request);
            result.prepared=true;result.backend="mlx_cpp_metal";result.precision="bf16";
            tc::HybridMetrics metrics;metrics.runtime_weight_backend="private_ane";
            metrics.runtime_weight_data_path="w8a8_hadamard";metrics.weight_variant="runtime_w8a8";
            result.encoder_hybrid=metrics;
            for(bool completed:{false,true}) {
                result.encoder_hybrid->runtime_weight_hybrid_blocks=completed?36:0;
                NSDictionary* value=tc::to_dictionary(result);NSDictionary* plan=value[@"plan"];
                tc::require([value[@"encoder_execution"] isEqual:completed?@"gpu_ane_experimental":@"gpu"],"incorrect actual encoder execution");
                tc::require([value[@"encoder_runtime_backend"] isEqual:completed?@"mlx_cpp_metal+private_ane_runtime_weight_experimental":@"mlx_cpp_metal"],"private encoder reported Core ML or fallback as ANE");
                tc::require([value[@"runtime_backend"] isEqual:@"mlx_cpp_metal"],"encoder route changed DiT execution");
                NSString* precision=value[@"encoder_runtime_precision"];
                if(completed) {
                    tc::require([precision containsString:@"private_ane_runtime_w8a8_hadamard"] && ![precision containsString:@"coreml"],"incorrect private encoder precision");
                    tc::require([plan[@"algorithm_approximations"] containsObject:@"qwen21_encoder_runtime_w8a8_channel_split"],"missing approximation disclosure");
                } else {
                    tc::require([precision isEqual:@"bf16"],"fallback precision changed");
                    tc::require(![plan[@"algorithm_approximations"] containsObject:@"qwen21_encoder_runtime_w8a8_channel_split"],"fallback claimed successful approximation");
                }
                tc::require([plan[@"encoder_precision"] isEqual:precision],"plan and result encoder precision disagree");
            }
            result.encoder_hybrid.reset();result.prepared=false;
            tc::NativeEncoderExecutionMetrics native;
            native.precision="bf16_language+bf16_gemm_fp32_vision_experimental";
            native.vision_layers=27;native.language_layers=36;native.graph_submissions=33;
            native.physically_complete=true;result.native_encoder=native;
            NSDictionary* value=tc::to_dictionary(result);NSDictionary* plan=value[@"plan"];
            tc::require([value[@"encoder_runtime_backend"] isEqual:@"native_metal_graph_qwen3_vl_experimental"],"native encoder backend mislabeled");
            tc::require([value[@"encoder_gpu_graph"] isEqual:@"native_vision27_language36_two_layer_graph"] && [plan[@"encoder_gpu_graph"] isEqual:value[@"encoder_gpu_graph"]],"native encoder graph mislabeled");
            tc::require([value[@"native_encoder_execution"][@"physically_complete"] boolValue] && ![value[@"native_encoder_execution"][@"release_qualified"] boolValue],"native encoder qualification/proof mislabeled");
            tc::require([value[@"runtime_backend"] isEqual:@"mlx_cpp_metal"],"native encoder changed denoiser backend");
            result.native_encoder->ane_ffns=36;result.native_encoder->ane_chunks=36;result.native_encoder->ane_padded_rows=3096;
            value=tc::to_dictionary(result);plan=value[@"plan"];
            tc::require([value[@"encoder_execution"] isEqual:@"gpu_ane_experimental"] && [value[@"encoder_runtime_backend"] isEqual:@"native_metal_graph_qwen3_vl_private_ane_w1_experimental"],"native ANE encoder mislabeled GPU/Core ML");
            tc::require([value[@"encoder_gpu_graph"] isEqual:@"native_vision_gpu_language_private_ane_w1_padded_ffn"] && [plan[@"encoder_gpu_graph"] isEqual:value[@"encoder_gpu_graph"]],"native ANE graph mislabeled");
            result.native_prefill=tc::NativePrefillExecutionMetrics{.calls=32,.chunks=64,.valid_rows=164736,.padded_rows=7296,.bucket_rows=2688,.physically_complete=true};
            value=tc::to_dictionary(result);plan=value[@"plan"];
            tc::require([plan[@"execution"] isEqual:@"gpu_ane_experimental"] && [value[@"native_prefill_execution"][@"first_step_only"] boolValue] && [value[@"native_prefill_execution"][@"base_only"] boolValue],"first step route/provenance lost");
            std::cout<<"qwen21_encoder_result=PASS device=0 models=0\n";
            return 0;
        } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
    }
}

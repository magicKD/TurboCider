#include "../../native/platform/apple/bridge.hpp"
#include <iostream>

int main() {
    @autoreleasepool {
        tc::RunResult result;
        result.request.model = "qwen-image-2.1";
        result.request.operation = "image.edit";
        result.request.width = result.request.height = 512;
        result.plan.request = result.request;
        result.plan.recipe.model = result.request.model;
        tc::require(!tc::to_dictionary(result)[@"shared_lora_ranks"], "ordinary receipt gained shared rank intent");
        for (bool enabled : {false,true}) for (int variant : {0,1,2}) {
            result.shared_lora_ranks = tc::SharedLoraRankMetrics{enabled,
                enabled ? 193u : 0u,enabled ? 192u : 0u,enabled ? 384u : 0u};
            result.prepared = variant == 1;
            result.native_json = variant == 2 ? "{\"model\":\"qwen-image-2.1\"}" : "";
            NSDictionary *metrics = tc::to_dictionary(result)[@"shared_lora_ranks"];
            tc::require(metrics && [metrics[@"enabled"] boolValue] == enabled &&
                [metrics[@"prepared_sets_this_request"] unsignedLongLongValue] == (enabled ? 193u : 0u) &&
                [metrics[@"completed_hybrid_blocks_this_request"] unsignedLongLongValue] == (enabled ? 192u : 0u) &&
                [metrics[@"completed_adapter_rank_arrays_this_request"] unsignedLongLongValue] == (enabled ? 384u : 0u),
                "shared rank request-local counters changed during serialization");
            tc::require([metrics[@"scope"] containsString:@"not physical kernel counts"],
                        "shared rank receipt lost its observation scope");
        }
        std::cout << "PASS shared LoRA rank receipts: off/on, normal/prepared/native, request-local scope\n";
    }
}

#include "bridge.hpp"
#include "platform.hpp"
#import <Metal/Metal.h>
namespace tc {
DeviceInfo device_info() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return {device ? std::string(device.name.UTF8String) : "unavailable",
            NSProcessInfo.processInfo.physicalMemory};
}
void validate_flux_configuration(const std::filesystem::path &root) {
    auto t = read_json(root / "transformer/config.json"),
         q = read_json(root / "text_encoder/config.json"), v = read_json(root / "vae/config.json");
    require([t[@"num_attention_heads"] intValue] == 24 &&
                [t[@"attention_head_dim"] intValue] == 128 && [t[@"num_layers"] intValue] == 5 &&
                [t[@"num_single_layers"] intValue] == 20 &&
                [t[@"joint_attention_dim"] intValue] == 7680 &&
                [t[@"in_channels"] intValue] == 128 && ![t[@"guidance_embeds"] boolValue],
            "only official FLUX.2 Klein 4B configuration supported");
    require([q[@"hidden_size"] intValue] == 2560 && [q[@"num_hidden_layers"] intValue] == 36 &&
                [q[@"num_attention_heads"] intValue] == 32 &&
                [q[@"num_key_value_heads"] intValue] == 8,
            "unsupported Qwen3 configuration");
    require([v[@"latent_channels"] intValue] == 32, "unsupported Flux VAE");
}
} // namespace tc

#include "../../backends/mlx.hpp"
#include <mlx/version.h>
namespace tc {
NSDictionary *system_info() {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    return @{
        @"abi" : @1,
        @"engine_version" : @"0.2.0-native-dev",
        @"gpu_available" : @(d != nil),
        @"gpu" : d.name ?: @"unavailable",
        @"physical_memory_bytes" : @([NSProcessInfo processInfo].physicalMemory),
        @"recommended_working_set_bytes" : @(d ? d.recommendedMaxWorkingSetSize : 0),
        @"os" : [NSProcessInfo processInfo].operatingSystemVersionString,
        @"mlx_version" : @(mx::version()),
        @"runtime_dependencies" : @[ @"libmlx", @"Metal", @"Foundation", @"ImageIO" ],
        @"python_runtime_required" : @NO
    };
}
} // namespace tc

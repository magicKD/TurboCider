#include "bridge.hpp"
#include "platform.hpp"
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <vector>
#include <fstream>
namespace tc {
DeviceInfo device_info() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return {device ? std::string(device.name.UTF8String) : "unavailable",
            NSProcessInfo.processInfo.physicalMemory};
}
FluxConfiguration flux_configuration(const std::filesystem::path &root,
                                     const std::string &model) {
    auto t = read_json(root / "transformer/config.json"),
         q = read_json(root / "text_encoder/config.json"), v = read_json(root / "vae/config.json");
    FluxConfiguration result;
    result.heads = [t[@"num_attention_heads"] intValue];
    result.hidden = result.heads * [t[@"attention_head_dim"] intValue];
    result.dual_layers = [t[@"num_layers"] intValue];
    result.single_layers = [t[@"num_single_layers"] intValue];
    require([t[@"attention_head_dim"] intValue] == 128 &&
                [t[@"in_channels"] intValue] == 128 && ![t[@"guidance_embeds"] boolValue],
            "unsupported FLUX.2 configuration");
    require((model == "flux2-klein-4b" && result.heads == 24 && result.dual_layers == 5 &&
             result.single_layers == 20) ||
            (model == "flux2-klein-9b" && result.heads == 32 && result.dual_layers == 8 &&
             result.single_layers == 24),
            "FLUX configuration does not match the selected model module");
    require([q[@"hidden_size"] intValue] > 0 && [q[@"num_hidden_layers"] intValue] == 36 &&
                [q[@"num_attention_heads"] intValue] == 32 &&
                [q[@"num_key_value_heads"] intValue] == 8 &&
                [t[@"joint_attention_dim"] intValue] == [q[@"hidden_size"] intValue] * 3,
            "unsupported Qwen3 configuration");
    require([v[@"latent_channels"] intValue] == 32, "unsupported Flux VAE");
    return result;
}
std::string sha256_file(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "cannot open file for SHA-256: " + path.string());
    CC_SHA256_CTX context;
    require(CC_SHA256_Init(&context) == 1, "cannot initialize SHA-256");
    // App inference runs on libdispatch workers with a roughly 512 KiB stack.
    // A 1 MiB automatic buffer crashes before hashing LoRA/Core ML provenance.
    std::vector<char> buffer(1 << 20);
    while (stream.good()) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        auto count = stream.gcount();
        if (count > 0)
            require(CC_SHA256_Update(&context, buffer.data(), static_cast<CC_LONG>(count)) == 1,
                    "cannot update SHA-256");
    }
    require(stream.eof(), "cannot read file for SHA-256: " + path.string());
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    require(CC_SHA256_Final(digest, &context) == 1, "cannot finalize SHA-256");
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(CC_SHA256_DIGEST_LENGTH * 2, '0');
    for (size_t i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
        result[i * 2] = hex[digest[i] >> 4];
        result[i * 2 + 1] = hex[digest[i] & 15];
    }
    return result;
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

#import <Foundation/Foundation.h>

#include "../../models/h3_mlx/vae_weights.hpp"

#include <cmath>

namespace tc::h3_mlx {
namespace {
NSDictionary *object_at(const std::filesystem::path &path) {
    NSData *data = [NSData dataWithContentsOfFile:@(path.c_str())];
    require(data != nil, "H3 video VAE config is missing: " + path.string());
    NSError *error = nil;
    id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    require([value isKindOfClass:NSDictionary.class],
            "H3 video VAE config must be a JSON object");
    return (NSDictionary *)value;
}

double number(NSDictionary *object, NSString *key) {
    id value = object[key];
    require([value isKindOfClass:NSNumber.class] &&
                CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            "H3 video VAE config field must be numeric: " +
                std::string(key.UTF8String));
    return [value doubleValue];
}

int integer(NSDictionary *object, NSString *key) {
    double value = number(object, key);
    require(value >= 0 && value <= INT32_MAX && std::floor(value) == value,
            "invalid H3 video VAE integer: " + std::string(key.UTF8String));
    return static_cast<int>(value);
}

void expected_array(NSDictionary *object, NSString *key,
                    const std::vector<int> &expected) {
    id raw = object[key];
    require([raw isKindOfClass:NSArray.class] &&
                [(NSArray *)raw count] == expected.size(),
            "invalid H3 video VAE array: " + std::string(key.UTF8String));
    for (NSUInteger index = 0; index < expected.size(); ++index)
        require([raw[index] intValue] == expected[index],
                "unsupported H3 video VAE architecture: " +
                    std::string(key.UTF8String));
}

template <size_t N>
void floats(NSDictionary *object, NSString *key, std::array<float, N> &out,
            float fallback) {
    id raw = object[key];
    if (!raw) {
        out.fill(fallback);
        return;
    }
    require([raw isKindOfClass:NSArray.class] && [(NSArray *)raw count] == N,
            "invalid H3 video VAE statistic: " + std::string(key.UTF8String));
    for (NSUInteger index = 0; index < N; ++index) {
        require([raw[index] isKindOfClass:NSNumber.class],
                "non-numeric H3 video VAE statistic");
        out[index] = [raw[index] floatValue];
        require(std::isfinite(out[index]) &&
                    (fallback == 0.f || out[index] > 0.f),
                "invalid H3 video VAE statistic value");
    }
}
} // namespace

VideoVAEConfig load_video_vae_config(const std::filesystem::path &path) {
    auto object = object_at(path);
    require(integer(object, @"in_channels") == 3 &&
                integer(object, @"out_channels") == 3,
            "H3 video VAE requires RGB input/output");
    expected_array(object, @"block_out_channels", {128, 256, 256, 512, 512, 1024});
    expected_array(object, @"spatial_downsample_factors", {2, 2, 2, 2, 1, 1});
    expected_array(object, @"temporal_downsample_factors", {1, 2, 2, 1, 1, 1});
    VideoVAEConfig result;
    result.latent_channels = integer(object, @"latent_channels");
    result.decoder_layers = integer(object, @"decoder_num_layers");
    result.num_heads = integer(object, @"decoder_num_attention_heads");
    result.head_dim = integer(object, @"decoder_attention_head_dim");
    result.register_tokens = integer(object, @"decoder_num_register_tokens");
    result.clip_length = integer(object, @"clip_length");
    result.token_drop = integer(object, @"token_drop");
    result.rope_theta = static_cast<float>(number(object, @"decoder_rope_theta"));
    result.rope_ratio = static_cast<float>(number(object, @"decoder_rope_dim_ratio"));
    result.norm_epsilon = static_cast<float>(number(object, @"decoder_norm_eps"));
    floats(object, @"latents_mean", result.latent_mean, 0.f);
    floats(object, @"latents_std", result.latent_std, 1.f);
    require(result.latent_channels == 24 && result.decoder_layers == 36 &&
                result.num_heads == 32 && result.head_dim == 64 &&
                result.register_tokens == 4 && result.clip_length == 17 &&
                result.token_drop == 3 &&
                std::abs(result.rope_theta - 100.f) < 1e-6f &&
                std::abs(result.rope_ratio - .75f) < 1e-6f,
            "unsupported H3 video VAE decoder configuration");
    return result;
}

} // namespace tc::h3_mlx

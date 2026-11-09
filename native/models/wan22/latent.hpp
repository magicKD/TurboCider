#pragma once

#include "../../backends/mlx.hpp"

namespace tc::wan22 {

// Wan2.2's full VAE outputs raw latents; the DiT and TAEW2.2-Super use the
// normalized diffusion space. Keep the conversion explicit at the boundary.
inline constexpr float latent_mean[] = {
        -.2289f,-.0052f,-.1323f,-.2339f,-.2799f,.0174f,.1838f,.1557f,
        -.1382f,.0542f,.2813f,.0891f,.1570f,-.0098f,.0375f,-.1825f,
        -.2246f,-.1207f,-.0698f,.5109f,.2665f,-.2108f,-.2158f,.2502f,
        -.2055f,-.0322f,.1109f,.1567f,-.0729f,.0899f,-.2799f,-.1230f,
        -.0313f,-.1649f,.0117f,.0723f,-.2839f,-.2083f,-.0520f,.3748f,
        .0152f,.1957f,.1433f,-.2944f,.3573f,-.0548f,-.1681f,-.0667f};
inline constexpr float latent_stddev[] = {
        .4765f,1.0364f,.4514f,1.1677f,.5313f,.4990f,.4818f,.5013f,
        .8158f,1.0344f,.5894f,1.0901f,.6885f,.6165f,.8454f,.4978f,
        .5759f,.3523f,.7135f,.6804f,.5833f,1.4146f,.8986f,.5659f,
        .7069f,.5338f,.4889f,.4917f,.4069f,.4999f,.6866f,.4093f,
        .5709f,.6065f,.6415f,.4944f,.5726f,1.2042f,.5458f,1.6887f,
        .3971f,1.0600f,.3943f,.5537f,.5444f,.4089f,.7468f,.7744f};
inline Tensor normalize_vae_latents(const Tensor &raw) {
    require(raw.ndim() == 5 && raw.shape(1) == 48, "Wan2.2 latent must have 48 channels");
    return (mx::astype(raw, mx::float32) - Tensor(latent_mean, {1, 48, 1, 1, 1}, mx::float32)) /
           Tensor(latent_stddev, {1, 48, 1, 1, 1}, mx::float32);
}

inline Tensor denormalize_vae_latents(const Tensor &normalized, mx::Dtype dtype = mx::bfloat16) {
    require(normalized.ndim() == 5 && normalized.shape(1) == 48, "Wan2.2 latent must have 48 channels");
    return mx::astype(mx::astype(normalized, mx::float32) * Tensor(latent_stddev, {1, 48, 1, 1, 1}, mx::float32) +
                      Tensor(latent_mean, {1, 48, 1, 1, 1}, mx::float32), dtype);
}

} // namespace tc::wan22

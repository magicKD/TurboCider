#pragma once
#include "../../native/models/seedvr2/vae.hpp"
#if !defined(TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_EXPERIMENT) || !TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_EXPERIMENT
#error "Causal ANE convolution is an unqualified offline experiment"
#endif
namespace tc::seedvr2 {
struct ExperimentalCausalConvStats {
    std::atomic<uint64_t> frames{0},coreml_calls{0},backing_hits{0};
    uint64_t norm_pack_frames=0;
    uint64_t slot_bytes=0,pack_checked_words=0,consume_checked_words=0;
    int tile_rows=0,tile_cols=0;
    std::array<uint64_t,4> input_strides{},output_strides{};
    int output_channels=0;
    std::string layer;
    double load_seconds=0,boundary_seconds=0;
    bool cpu_control=false;
};
// Caller binds the original checkpoint and actual compiled-tree identities.
// Runtime validates model descriptors/geometry, nonaliasing slot ownership and
// completed GPU/Core ML boundaries. verify_io is CPU-intensive unit QA only.
ExperimentalCausalConv make_experimental_ane_causal_conv(const std::filesystem::path &manifest,bool cpu_control,
    std::atomic<bool> &cancelled,std::shared_ptr<ExperimentalCausalConvStats> &stats,bool verify_io=false);
// Separate original 256->256 spatial-upsample causal convolution bank/hook.
// Never treat the older 128-output bank as a complete implementation of it.
ExperimentalCausalConv make_experimental_ane_upscale_causal_conv(const std::filesystem::path &manifest,bool cpu_control,
    std::atomic<bool> &cancelled,std::shared_ptr<ExperimentalCausalConvStats> &stats,bool verify_io=false);
ExperimentalNormCausalConv make_experimental_ane_norm_causal_conv(const std::filesystem::path &manifest,bool cpu_control,
    std::atomic<bool> &cancelled,std::shared_ptr<ExperimentalCausalConvStats> &stats,bool verify_io=false);
}

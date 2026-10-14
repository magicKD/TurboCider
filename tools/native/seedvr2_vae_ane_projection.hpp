#pragma once
#include "../../native/models/seedvr2/vae.hpp"
#if !defined(TURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT) || !TURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT
#error "ANE projection is an unqualified offline VAE experiment"
#endif
namespace tc::seedvr2 {
struct ExperimentalProjectionStats {
    std::atomic<uint64_t> frames{0},coreml_calls{0},backing_hits{0};
    uint64_t slot_bytes=0;
    int tile_rows=0,tile_cols=0;
    double boundary_seconds=0,load_seconds=0;
    bool cpu_control=false;
};
// The caller verifies checkpoint/manifest/compiled identities before execution.
// This factory additionally validates the exact compiled model and IO geometry.
ExperimentalUpscaleProjection make_experimental_ane_projection(
    const std::filesystem::path &manifest,bool cpu_control,std::atomic<bool> &cancelled,
    std::shared_ptr<ExperimentalProjectionStats> &stats);
}

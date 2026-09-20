#pragma once
#include "coreml_bundle.hpp"
#include "streaming_descriptor.hpp"
namespace tc::z_image {
// Pure planning result, not executable authority. The GPU recipe remains a
// recipe; materialized bytes and actual execution receipts are separate.
struct HybridStreamingPlan {
    GpuSuffixPlan gpu;
    streaming::Layout layout;
};
// Requires native source proof and a bound bundle, but only reads metadata and
// checks their generations. Does not materialize, load Core ML or allocate GPU.
HybridStreamingPlan describe_hybrid_streaming(
    const StreamingMetadata &, const VerifiedCoreMLBundleLease &,
    const StreamingConfig &, const StreamingWorkload &);
} // namespace tc::z_image

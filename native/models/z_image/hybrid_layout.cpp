#include "hybrid_layout.hpp"
#include <bit>
namespace tc::z_image {
HybridStreamingPlan describe_hybrid_streaming(
        const StreamingMetadata &metadata, const VerifiedCoreMLBundleLease &bundle,
        const StreamingConfig &config, const StreamingWorkload &workload) {
    require(metadata.lease().has_verified_content(), "hybrid layout requires verified GPU source");
    bundle.revalidate(); metadata.check_unchanged();
    require(workload.width == 512 && workload.height == 512 && workload.caption_rows == 64 && workload.steps == 9,
            "hybrid layout requires frozen HY-M0 512/64/9 scope");
    const auto &partition = bundle.partition();
    HybridStreamingPlan plan;
    plan.gpu = metadata.describe_gpu_suffix(workload, partition.ane_end);
    auto &descriptor = plan.gpu.descriptor;
    require(descriptor.artifacts.size() == 2 && descriptor.stages.size() == 1 &&
            descriptor.artifacts[0].identity_kind == streaming::SourceIdentityKind::content_sha256 &&
            descriptor.artifacts[0].identity == partition.parent_checkpoint_digest,
            "hybrid layout GPU/ANE parent mismatch");
    require(partition.hidden == 3840 && partition.mlp_width == 10240 && partition.ane_begin == 0 &&
            partition.bucket_rows == 1088 && descriptor.workload.at("unified_rows") == "1088" &&
            descriptor.workload.at("image_rows") == "1024" && plan.gpu.packing.size() == 32,
            "hybrid layout partition/workload mismatch");
    // Independent from exact-GPU and the earlier planning-only revision.
    // Execution still requires the internal owner and separate public authority.
    descriptor.backend_revision = "z-image-verified-hybrid-stage-v1";
    descriptor.stages[0].adapter_revision = "z-image-verified-hybrid-stage-v1";
    auto &identity = descriptor.workload;
    identity["execution"] = "gpu_ane";
    identity["hybrid_block_kernel"] = "uncompiled-attention-compiled-ffn-v1";
    identity["component_lifecycle"] = "owned-inputs-stage-drain-v1";
    identity["approximation"] = "true";
    identity["hybrid_partition_identity"] = partition.identity;
    identity["hybrid_bundle_content"] = partition.artifact_content_digest;
    identity["hybrid_precision"] = partition.precision_revision;
    identity["hybrid_bucket_rows"] = std::to_string(partition.bucket_rows);
    identity["hybrid_activation_scale_f32_bits"] = std::to_string(std::bit_cast<uint32_t>(partition.activation_scale));
    identity["hybrid_output_scale_f32_bits"] = std::to_string(std::bit_cast<uint32_t>(partition.output_scale));
    identity["gpu_suffix_kernel"] = "z-image-bf16-matmul-silu-v1";
    identity["join_policy"] = "fp16-to-gpu-dtype-scale-add-v1";
    identity["output_backing_policy"] = "single-shared-ffn-output-consumer-drain-before-reuse-v1";
    plan.layout = streaming::compile_layout(config, descriptor);
    bundle.revalidate(); metadata.check_unchanged();
    return plan;
}
} // namespace tc::z_image

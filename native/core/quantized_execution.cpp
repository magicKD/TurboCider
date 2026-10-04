#include "quantized_execution.hpp"
#include "common.hpp"
#include "quantized_execution_profiles.hpp"

namespace tc {
void validate_quantized_execution(const QuantizedExecutionConfig &c) {
    if (!c.specified()) return;
    require(c.enabled.has_value() && c.schema_version.value_or(0) == 1,
            "quantized_execution requires enabled and schema_version=1");
    if (!c.active()) {
        require(!c.mode && !c.source_residency && !c.decode_backend && !c.precision_profile && !c.granularity &&
                    !c.prefetch_layers && !c.persistent_dense_layers && !c.oversized_layer_policy &&
                    !c.ane_compute && !c.allow_requantization, "disabled quantized_execution accepts only version/enabled");
        return;
    }
    const bool native_affine = c.precision_profile == "z-source-native-affine-v1" || c.precision_profile == "z-mlx-compat-affine-v1";
    const bool dense_bf16 = c.precision_profile == "z-dense-bf16-v1";
    const bool raw_gpu=gguf_raw_gpu_profile(c.precision_profile.value_or(""));
    const auto mode=raw_gpu ? "bounded_raw_packed" : native_affine ? "bounded_packed" : "bounded_dequant";
    require(c.mode.value_or(mode) == mode &&
                (c.source_residency.value_or("packed_resident") == "packed_resident" || c.source_residency == "packed_streamed") &&
                c.decode_backend.value_or(raw_gpu ? "cpu_io_gpu_affine" : "cpu_simd") == (raw_gpu ? "cpu_io_gpu_affine" : "cpu_simd") &&
                (c.precision_profile.value_or("z-source-mixed-v1") == "z-source-mixed-v1" ||
                 c.precision_profile == "z-source-mixed-f16-v1" || c.precision_profile == "z-source-exact-f32-v1" ||
                 c.precision_profile == "z-mlx-compat-f16-v1" || c.precision_profile == "z-mlx-compat-f32-v1" || native_affine || dense_bf16 || raw_gpu) &&
                c.granularity.value_or("layer") == "layer" &&
                c.oversized_layer_policy.value_or("reject") == "reject" &&
                c.ane_compute.value_or("off") == "off", "quantized execution candidate unsupported");
    require(c.prefetch_layers.value_or(1) <= 2 && c.persistent_dense_layers.value_or(0) == 0 &&
                !c.allow_requantization.value_or(false), "quantized execution layout/approximation unsupported");
    require(!dense_bf16 || c.source_residency == "packed_streamed",
            "qe_config_conflict: dense BF16 candidate requires explicit packed_streamed source");
    require(!raw_gpu || c.source_residency=="packed_streamed","qe_config_conflict: raw GPU candidate requires explicit packed_streamed source");
}
} // namespace tc

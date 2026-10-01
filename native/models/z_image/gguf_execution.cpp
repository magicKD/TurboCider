#include "gguf_execution.hpp"

#include <map>

namespace tc::z_image {
namespace {
using Shape = std::vector<uint64_t>;
uint64_t capacity(uint64_t bytes) {
    constexpr uint64_t alignment = streaming::GgufWeightPager::buffer_alignment;
    return gguf::checked_add(bytes, alignment - 1) & ~(alignment - 1);
}
std::map<std::string, Shape> required_shapes() {
    const std::map<std::string, Shape> fields{
        {"attention.qkv.weight", {11520,3840}}, {"attention.out.weight", {3840,3840}},
        {"attention.q_norm.weight", {128}}, {"attention.k_norm.weight", {128}},
        {"attention_norm1.weight", {3840}}, {"attention_norm2.weight", {3840}},
        {"ffn_norm1.weight", {3840}}, {"ffn_norm2.weight", {3840}},
        {"feed_forward.w1.weight", {10240,3840}}, {"feed_forward.w3.weight", {10240,3840}},
        {"feed_forward.w2.weight", {3840,10240}}};
    std::map<std::string, Shape> result;
    for (const auto *group : {"layers", "noise_refiner", "context_refiner"}) {
        const uint32_t count = std::string_view(group) == "layers" ? 30 : 2;
        for (uint32_t i = 0; i < count; ++i) {
            const std::string prefix = std::string(group) + "." + std::to_string(i) + ".";
            for (const auto &[key, shape] : fields) result.emplace(prefix + key, shape);
            if (std::string_view(group) != "context_refiner") {
                result.emplace(prefix + "adaLN_modulation.0.weight", Shape{15360,256});
                result.emplace(prefix + "adaLN_modulation.0.bias", Shape{15360});
            }
        }
    }
    for (const auto &[key, shape] : std::map<std::string, Shape>{
        {"x_embedder.weight", {3840,64}}, {"x_embedder.bias", {3840}},
        {"cap_embedder.0.weight", {2560}}, {"cap_embedder.1.weight", {3840,2560}}, {"cap_embedder.1.bias", {3840}},
        {"t_embedder.mlp.0.weight", {1024,256}}, {"t_embedder.mlp.0.bias", {1024}},
        {"t_embedder.mlp.2.weight", {256,1024}}, {"t_embedder.mlp.2.bias", {256}},
        {"final_layer.adaLN_modulation.1.weight", {3840,256}}, {"final_layer.adaLN_modulation.1.bias", {3840}},
        {"final_layer.linear.weight", {64,3840}}, {"final_layer.linear.bias", {64}},
        {"x_pad_token", {1,3840}}, {"cap_pad_token", {1,3840}}}) result.emplace(key, shape);
    return result;
}
}

GgufExecutionPlan describe_gguf_execution(std::shared_ptr<const streaming::SourceLease> lease,
        uint32_t prefetch, uint32_t width, uint32_t height, uint32_t caption, uint32_t steps,
        const std::string &profile, const std::string &residency) {
    require(lease && lease->has_verified_content(), "gguf execution requires native content proof");
    require(prefetch <= 2 && width && height && width % 16 == 0 && height % 16 == 0 &&
                caption && steps && steps <= 50, "gguf execution workload unsupported");
    require(profile == "z-source-mixed-v1" || profile == "z-source-mixed-f16-v1" || profile == "z-source-exact-f32-v1" || profile == "z-source-native-affine-v1" ||
            profile == "z-mlx-compat-affine-v1" || profile == "z-mlx-compat-f16-v1" || profile == "z-mlx-compat-f32-v1",
            "unknown GGUF precision profile");
    require(residency == "packed_resident" || residency == "packed_streamed", "unknown GGUF source residency");
    const bool legacy_float=profile.starts_with("z-mlx-compat-");
    const auto &file = lease->file("transformer");
    auto fd = lease->duplicate_fd("transformer");
    const auto directory = gguf::read_directory(fd.get(), file.bytes);
    auto expected = required_shapes();
    require(directory.tensors.size() == expected.size(), "gguf Z tensor count mismatch");
    GgufExecutionPlan result;
    auto &d = result.descriptor;
    d.model = "z-image-turbo-gguf";
    d.checkpoint_identity = std::string(lease->artifact_digest());
    d.backend_revision = "gguf-cpu-rne-mlx-v1:" + profile;
    d.artifacts.push_back({"transformer", file.content_digest, file.bytes, streaming::SourceIdentityKind::content_sha256});
    d.workload = {{"width", std::to_string(width)}, {"height", std::to_string(height)},
        {"source_residency", residency},
        {"caption_rows", std::to_string(caption)}, {"steps", std::to_string(steps)},
        {"precision", profile}, {"fixed_policy", "source-float-alias-v1"}};
    if(legacy_float)d.workload["float_loader"]="mlx-bf16-to-f16-v1";
    streaming::StageDescriptor stage;
    stage.id = "denoiser"; stage.adapter_revision = d.backend_revision;
    stage.min_slots = 1; stage.max_slots = 3; stage.max_group_size = 1;
    stage.pass_count = steps;
    for (uint32_t step = 0; step < steps; ++step) stage.passes.push_back({step, "denoise", {width,height,caption}});
    for (uint32_t i = 0; i < 30; ++i) {
        streaming::BlockSpec block;
        block.id = i; block.layout_class = "z-gguf-main-source-mixed-v1";
        stage.blocks.push_back(std::move(block));
    }
    for (const auto &tensor : directory.tensors) {
        const auto found = expected.find(tensor.name);
        require(found != expected.end(), "unexpected GGUF Z tensor: " + tensor.name);
        const auto logical = tensor.logical_shape();
        const bool pad = (tensor.name == "x_pad_token" || tensor.name == "cap_pad_token") && logical == Shape{3840};
        require(logical == found->second || pad, "GGUF Z tensor geometry mismatch: " + tensor.name);
        const bool main = tensor.name.starts_with("layers.");
        const bool floating = gguf::type_info(tensor.type).elements == 1;
        require(main || floating, "GGUF quantized refiners require a later streamed-refiner revision");
        const char *format = floating ? (tensor.type == 0 ? "F32" : tensor.type == 1 || legacy_float ? "F16" : "BF16")
            : profile == "z-source-exact-f32-v1" || profile == "z-mlx-compat-f32-v1" ? "F32"
            : profile == "z-source-mixed-f16-v1" || profile == "z-mlx-compat-f16-v1" ? "F16" : "BF16";
        const uint32_t item = std::string_view(format) == "F32" ? 4 : 2;
        streaming::Materialization materialization;
        materialization.format = format; materialization.storage_mode = "mlx-metal-shared";
        materialization.conversion = main ? "gguf-cpu-rne-v1" : legacy_float && tensor.type==30 ? "gguf-mlx-float-alias-v1" : "gguf-native-alias-v1";
        materialization.shape = found->second;
        materialization.reads.push_back({0,tensor.file_offset,tensor.bytes,tensor.name,
                                       gguf::type_info(tensor.type).name,logical});
        streaming::FieldSpec field;
        field.bytes = gguf::checked_mul(tensor.elements, item);
        field.alignment = streaming::GgufWeightPager::buffer_alignment;
        field.materialization = std::move(materialization);
        if (main) {
            const auto separator = tensor.name.find('.', 7);
            require(separator != std::string::npos, "invalid layer name");
            const std::string number = tensor.name.substr(7,separator - 7);
            const auto layer = std::stoul(number);
            require(layer < 30 && number == std::to_string(layer), "invalid layer index");
            field.name = tensor.name.substr(separator + 1);
            field.storage_id = "gguf-layer-" + std::to_string(layer) + ":" + field.name;
            if (!floating && (profile == "z-source-native-affine-v1" || profile == "z-mlx-compat-affine-v1")) {
                require(tensor.type == 2 || tensor.type == 3 || tensor.type == 8, "native affine profile only supports Q4_0/Q4_1/Q8_0");
                const uint32_t bits = tensor.type == 8 ? 8 : 4;
                const std::string stem = field.name.substr(0,field.name.size()-7);
                for (const auto *part : {"codes","scales","biases"}) {
                    auto packed = field;
                    const bool codes = std::string_view(part) == "codes";
                    packed.name = codes ? field.name : stem + "." + part;
                    packed.storage_id = "gguf-layer-" + std::to_string(layer) + ":" + packed.name;
                    auto &m = *packed.materialization;
                    m.conversion = "gguf-affine-" + std::string(part) + "-v1";
                    m.format = codes ? "U32" : "F16";
                    m.shape = {tensor.rows(),codes ? tensor.columns()*bits/32 : tensor.columns()/32};
                    packed.bytes = gguf::checked_mul(gguf::checked_mul(m.shape[0],m.shape[1]),codes ? 4 : 2);
                    stage.blocks[layer].fields.push_back(std::move(packed));
                }
            } else stage.blocks[layer].fields.push_back(std::move(field));
        } else {
            field.name = tensor.name; field.storage_id = "gguf-source:" + tensor.name;
            stage.resident_fields.push_back(std::move(field));
        }
        if (residency == "packed_resident" || !main)
            result.packed_capacity_upper = gguf::checked_add(result.packed_capacity_upper, capacity(tensor.bytes));
    }
    for (auto &block : stage.blocks)
        std::sort(block.fields.begin(), block.fields.end(), [](const auto &a, const auto &b) { return a.name < b.name; });
    d.stages.push_back(std::move(stage));
    auto &c = result.config;
    c.enabled = true; c.schema_version = 1; c.selection = "manual"; c.retention = "request";
    c.stages["denoiser"] = {"streamed", 1, 1 + prefetch, 0, prefetch, 1};
    result.layout = streaming::compile_layout(c, d);
    result.dense_capacity_upper = result.layout.stages.front().peak_pool_bytes;
    if (residency == "packed_streamed") result.read_capacity_upper = streaming::GgufWeightPager::default_read_buffer_bytes;
    lease->revalidate_open_files(); lease->revalidate_paths();
    return result;
}
} // namespace tc::z_image

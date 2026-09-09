#include "bridge.hpp"
#include "platform.hpp"
#include "../../models/wan/hybrid.hpp"

namespace tc::wan {

std::unique_ptr<HybridFFN> load_hybrid(const std::filesystem::path &file,
                                     const std::filesystem::path &model_root,
                                     const Checkpoint &weights, const Event &event,
                                     std::atomic<bool> &cancelled) {
    require(file.is_absolute() && model_root.is_absolute(), "Wan hybrid asset paths must be absolute");
    checkpoint(cancelled);
    auto manifest = read_json(file);
    require(string_value(manifest, @"schema") == "turbocider-wan-ane-mlp-v1",
            "unsupported Wan hybrid manifest schema");
    NSDictionary *shape = manifest[@"shape"];
    require([shape isKindOfClass:NSDictionary.class], "Wan hybrid shape missing");
    auto integer = [&](NSString *key, int value) {
        id number = shape[key];
        require([number isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)number) != CFBooleanGetTypeID() &&
                    [number doubleValue] == value, "unsupported Wan hybrid shape");
    };
    integer(@"rows", 32760); integer(@"hidden", 1536); integer(@"intermediate", 8960);
    integer(@"ane_intermediate", 4096); integer(@"gpu_intermediate", 4864);
    const auto variant = string_value(manifest, @"variant");
    require(variant == "int8_pc" || variant == "fp16", "unsupported Wan hybrid variant");
    const auto source_sha = string_value(manifest, @"checkpoint_sha256");
    const auto config_sha = string_value(manifest, @"mlx_dit_json_sha256");
    require(source_sha == "a48f7370cab9664ebc71afefa6cbc2ea2ab1970b06aa9ad77712179cf52213ff" &&
                config_sha == "db5603223f17a03051d36e4a3a218477dd48a7723117743d5d916d1db3f87691",
            "unqualified Wan hybrid source identity");
    // Verify the actual model selected by this session. Never follow a
    // developer's absolute checkpoint path embedded in a copied manifest.
    require(sha256_file(model_root / "mlx_dit.safetensors") == source_sha &&
                sha256_file(model_root / "mlx_dit.json") == config_sha,
            "Wan hybrid artifacts do not match the selected checkpoint");
    checkpoint(cancelled);
    NSArray *blocks = manifest[@"blocks"];
    NSDictionary *artifacts = manifest[@"artifacts"];
    require([blocks isKindOfClass:NSArray.class] && blocks.count == 30 &&
                [artifacts isKindOfClass:NSDictionary.class] && artifacts.count == 30,
            "Wan hybrid manifest must contain exactly 30 blocks");
    std::vector<std::filesystem::path> paths;
    const auto root = std::filesystem::canonical(file.parent_path());
    for (int block = 0; block < 30; ++block) {
        require([blocks[block] isKindOfClass:NSNumber.class] && [blocks[block] doubleValue] == block,
                "Wan hybrid blocks must be ordered 0...29");
        id name = artifacts[[NSString stringWithFormat:@"%d", block]];
        require([name isKindOfClass:NSString.class], "Wan Core ML artifact name missing");
        auto relative = std::filesystem::path([name UTF8String]);
        require(!relative.empty() && relative == relative.filename() && relative.extension() == ".mlmodelc",
                "Wan Core ML artifact must be a compiled filename");
        auto path = std::filesystem::canonical(root / relative);
        require(path.parent_path() == root, "Wan Core ML artifact escapes manifest directory");
        paths.push_back(path);
    }
    return std::make_unique<HybridFFN>(weights, paths, 32760, 4096, event, cancelled);
}

} // namespace tc::wan

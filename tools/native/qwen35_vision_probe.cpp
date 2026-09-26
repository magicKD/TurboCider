#include "../../native/models/qwen21/vision.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4 || (argc == 5 && std::string(argv[4]) == "--gpu"),
                    "usage: qwen35-vision-probe weights inputs output [--gpu]; default CPU");
        if (argc == 5) tc::configure_streams();
        else tc::mx::set_default_device(tc::mx::Device::cpu);
        tc::Weights weights;
        weights.load_file(argv[1], "model.visual.");
        weights.remap_keys([](const std::string &key) { return "model.visual." + key; });
        auto [inputs, meta] = tc::mx::load_safetensors(argv[2]);
        tc::qwen21::VisionConfig config;
        config.pe_qwen35 = true;
        config.pe_fp32 = meta.contains("fp32_visual") && meta.at("fp32_visual") == "1";
        config.pe_compensated_projection = meta.contains("compensated_projection") &&
            meta.at("compensated_projection") == "1";
        config.deepstack_layers.clear();
        config.patch = std::stoi(meta.at("patch"));
        config.hidden = std::stoi(meta.at("hidden"));
        config.heads = std::stoi(meta.at("heads"));
        config.layers = std::stoi(meta.at("layers"));
        config.position_side = std::stoi(meta.at("position_side"));
        const int height = std::stoi(meta.at("height")), width = std::stoi(meta.at("width"));
        tc::qwen21::VisionEncoder encoder(weights, config);
        std::atomic<bool> cancelled{false};
        std::unordered_map<std::string, tc::Tensor> outputs;
        auto features = encoder.encode(inputs.at("patches"), height, width, {}, cancelled, &outputs);
        tc::require(features.deepstack.empty(), "PE vision unexpectedly emitted DeepStack features");
        tc::mx::eval(features.merged);
        tc::require(tc::mx::all(tc::mx::isfinite(features.merged)).item<bool>(), "nonfinite PE vision output");
        if (config.pe_fp32) {
            tc::require(features.merged.dtype() == tc::mx::float32, "FP32 vision returned reduced precision");
            // Compare runtime promotion against an independently preconverted
            // visual weight map, without changing the source checkpoint.
            tc::Weights converted;
            auto keys = weights.sorted_keys();
            std::vector<tc::Tensor> arrays;
            std::vector<tc::mx::Dtype> original_dtypes;
            for (const auto &key : keys) {
                original_dtypes.push_back(weights.at(key).dtype());
                arrays.push_back(tc::mx::astype(weights.at(key), tc::mx::float32));
            }
            converted.bind_arrays(keys, arrays);
            auto converted_config = config;
            converted_config.pe_fp32 = false;
            tc::qwen21::VisionEncoder converted_encoder(converted, converted_config);
            auto equivalent = converted_encoder.encode(tc::mx::astype(inputs.at("patches"), tc::mx::float32),
                height, width, {}, cancelled);
            tc::require(tc::mx::all(features.merged == equivalent.merged).item<bool>(),
                        "runtime FP32 vision differs from preconverted FP32 visual weights");
            for (size_t i = 0; i < keys.size(); ++i)
                tc::require(weights.at(keys[i]).dtype() == original_dtypes[i], "FP32 vision mutated checkpoint dtype");
        }
        outputs.emplace("merged", features.merged);
        tc::mx::save_safetensors(argv[3], outputs, {
            {"fp32_visual", config.pe_fp32 ? "1" : "0"},
            {"compensated_projection", config.pe_compensated_projection ? "1" : "0"}});
        cancelled = true;
        bool rejected = false;
        try { encoder.encode(inputs.at("patches"), height, width, {}, cancelled); }
        catch (const tc::Cancelled &) { rejected = true; }
        tc::require(rejected, "PE vision ignored cancellation");
        cancelled = false;
        rejected = false;
        try { encoder.encode(inputs.at("patches"), height + 1, width, {}, cancelled); }
        catch (const std::exception &) { rejected = true; }
        tc::require(rejected, "PE vision accepted invalid grid");
        std::cout << "PASS PE vision finite output, cancellation, invalid grid\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

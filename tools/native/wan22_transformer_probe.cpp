#include "../../native/models/wan22/transformer.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc >= 12 && argc <= 16, "usage: wan22-transformer-probe WEIGHTS INPUT OUTPUT CHANNELS HIDDEN HEADS INTERMEDIATE LAYERS TEXT_DIM FREQ_DIM TIMESTEP [TRACE] [--comfy] [--cache-cross] [--fused-rope]");
        auto input = tc::mx::load_safetensors(argv[2]).first;
        tc::wan22::TransformerConfig config;
        config.channels = std::stoi(argv[4]); config.hidden = std::stoi(argv[5]); config.heads = std::stoi(argv[6]);
        config.intermediate = std::stoi(argv[7]); config.layers = std::stoi(argv[8]);
        config.text_dim = std::stoi(argv[9]); config.frequency_dim = std::stoi(argv[10]);
        std::string trace_path;
        bool cache_cross = false;
        for (int i = 12; i < argc; ++i) {
            if (std::string(argv[i]) == "--comfy") config.layout = tc::wan22::CheckpointLayout::comfy;
            else if (std::string(argv[i]) == "--cache-cross") cache_cross = true;
            else if (std::string(argv[i]) == "--fused-rope") config.fused_rope = true;
            else { tc::require(trace_path.empty(), "only one trace path is allowed"); trace_path = argv[i]; }
        }
        tc::wan22::Transformer model;
        model.load(std::filesystem::absolute(argv[1]), config);
        std::unordered_map<std::string, tc::Tensor> trace_values;
        tc::wan22::Transformer::Trace trace;
        if (!trace_path.empty()) trace = [&](const std::string &name, const tc::Tensor &value) {
            tc::mx::eval(value);
            trace_values.emplace(name, value);
        };
        auto context = model.prepare_text(input.at("text"), trace);
        tc::mx::eval(context);
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        std::vector<tc::wan22::CrossAttentionKV> cross;
        if (cache_cross) cross = model.prepare_cross_attention(context, cancelled);
        auto output = model.forward(input.at("latents"), context, std::stof(argv[11]), event, cancelled, trace,
                                    cache_cross ? &cross : nullptr);
        tc::mx::save_safetensors(argv[3], {{"output", output}});
        if (!trace_path.empty()) tc::mx::save_safetensors(trace_path, trace_values);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

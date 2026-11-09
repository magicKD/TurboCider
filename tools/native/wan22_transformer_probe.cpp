#include "../../native/models/wan22/transformer.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 12 || argc == 13, "usage: wan22-transformer-probe WEIGHTS INPUT OUTPUT CHANNELS HIDDEN HEADS INTERMEDIATE LAYERS TEXT_DIM FREQ_DIM TIMESTEP [TRACE]");
        auto input = tc::mx::load_safetensors(argv[2]).first;
        tc::wan22::TransformerConfig config;
        config.channels = std::stoi(argv[4]); config.hidden = std::stoi(argv[5]); config.heads = std::stoi(argv[6]);
        config.intermediate = std::stoi(argv[7]); config.layers = std::stoi(argv[8]);
        config.text_dim = std::stoi(argv[9]); config.frequency_dim = std::stoi(argv[10]);
        tc::wan22::Transformer model;
        model.load(std::filesystem::absolute(argv[1]), config);
        std::unordered_map<std::string, tc::Tensor> trace_values;
        tc::wan22::Transformer::Trace trace;
        if (argc == 13) trace = [&](const std::string &name, const tc::Tensor &value) {
            tc::mx::eval(value);
            trace_values.emplace(name, value);
        };
        auto context = model.prepare_text(input.at("text"), trace);
        tc::mx::eval(context);
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        auto output = model.forward(input.at("latents"), context, std::stof(argv[11]), event, cancelled, trace);
        tc::mx::save_safetensors(argv[3], {{"output", output}});
        if (argc == 13) tc::mx::save_safetensors(argv[12], trace_values);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

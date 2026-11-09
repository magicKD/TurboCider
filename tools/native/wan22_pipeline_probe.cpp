#include "../../native/models/wan22/pipeline.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 8 || argc == 9,
                    "usage: wan22-pipeline-probe TRANSFORMER VAE PIXELS TEXT OUTPUT FRAMES SEED [TRACE]");
        auto image = tc::mx::load_safetensors(argv[3]).first.at("pixels");
        auto text = tc::mx::load_safetensors(argv[4]).first.at("text");
        tc::wan22::Pipeline pipeline(std::filesystem::absolute(argv[1]), std::filesystem::absolute(argv[2]));
        tc::wan22::GenerateOptions options;
        options.frames = std::stoi(argv[6]);
        options.seed = std::stoull(argv[7]);
        std::unordered_map<std::string, tc::Tensor> trace_values;
        tc::wan22::Pipeline::Trace trace;
        if (argc == 9) trace = [&](const std::string &name, const tc::Tensor &value) {
            tc::mx::eval(value);
            trace_values.emplace(name, value);
        };
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &phase, int current, int total) {
            if (phase == "wan22_denoise") std::cerr << phase << " " << current << "/" << total << '\n';
        };
        auto output = pipeline.generate(image, text, options, event, cancelled, trace);
        tc::mx::save_safetensors(argv[5], {{"latents", output.latents}, {"pixels", output.pixels}});
        if (argc == 9) tc::mx::save_safetensors(argv[8], trace_values);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

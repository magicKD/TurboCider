#include "../../native/models/qwen21/transformer.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4 || argc == 5, "usage: qwen21-transformer-probe weights.safetensors inputs.safetensors OUTPUT [single|trace|trajectory|calibration|benchmark_iterations]; calibration OUTPUT is a new directory");
        tc::configure_streams();
        tc::Weights weights;
        weights.load_file(argv[1]);
        auto [inputs, metadata] = tc::mx::load_safetensors(argv[2]);
        const bool calibration = argc == 5 && std::string(argv[4]) == "calibration";
        const auto &source_latents = inputs.at(calibration && inputs.count("initial") ? "initial" : "latents");
        tc::qwen21::TransformerConfig config;
        if (!calibration || metadata.count("layers")) {
            config.layers = std::stoi(metadata.at("layers"));
            config.heads = std::stoi(metadata.at("heads"));
            config.head_dim = std::stoi(metadata.at("head_dim"));
            config.rope_axes = {std::stoi(metadata.at("axis0")), std::stoi(metadata.at("axis1")),
                                std::stoi(metadata.at("axis2"))};
        }
        config.channels = source_latents.shape(2);
        config.context_dim = inputs.at("text").shape(2);
        int height = calibration && !metadata.count("height") ? 32 : std::stoi(metadata.at("height"));
        int width = calibration && !metadata.count("width") ? 32 : std::stoi(metadata.at("width"));
        if (calibration) tc::require(height == 32 && width == 32 && config.layers == 32 &&
                                    source_latents.shape(1) == 1024 && config.channels == 64,
                                    "Qwen21 calibration needs real full-model 512px generation inputs");
        tc::qwen21::Transformer model(weights, config);
        auto text = inputs.at("text"), latents = source_latents;
        std::vector<tc::qwen21::ReferenceLatents> references;
        if (metadata.count("reference_count")) {
            int count = std::stoi(metadata.at("reference_count"));
            for (int i = 0; i < count; ++i) {
                auto key = "reference" + std::to_string(i);
                references.push_back({inputs.at(key), {std::stoi(metadata.at(key + "_height")),
                    std::stoi(metadata.at(key + "_width")), std::stoi(metadata.at(key + "_slot"))}});
            }
        }
        if (argc == 5 && std::string(argv[4]) == "trace") {
            std::unordered_map<std::string, tc::Tensor> trace;
            auto output = model.forward(latents, text, 0.8f, height, width, false, &trace, references);
            trace.emplace("output", output);
            tc::mx::save_safetensors(argv[3], trace);
            return 0;
        }
        if (calibration) {
            tc::require(inputs.count("sigmas"), "calibration requires the generation schedule");
            auto schedule = tc::mx::astype(inputs.at("sigmas"), tc::mx::float32);
            tc::require(schedule.ndim() == 1 && schedule.size() >= 3,
                        "calibration requires at least two denoising steps");
            tc::mx::eval(schedule);
            // The prefix is populated by an actual prefill at step zero.
            // Capture the second step, where production decoding reuses KV.
            auto first = model.forward(latents, text, schedule.data<float>()[0],
                                       height, width, true, nullptr, references);
            latents = latents + first * tc::Tensor(schedule.data<float>()[1] -
                                                   schedule.data<float>()[0], latents.dtype());
            tc::mx::eval(latents);
            std::unordered_map<std::string, tc::Tensor> trace;
            auto output = model.forward(latents, text, schedule.data<float>()[1],
                                        height, width, true, &trace, references);
            tc::mx::eval(output);
            // Capture all target rows, but not the large attention/KV and
            // block-output traces, for the per-block SmoothQuant exporter.
            const int image_rows = int(latents.shape(1));
            const auto directory = std::filesystem::absolute(argv[3]);
            tc::require(!std::filesystem::exists(directory), "calibration output directory already exists");
            std::filesystem::create_directories(directory);
            for (int block = 0; block < config.layers; ++block) {
                auto key = "mlp_input" + std::to_string(block);
                auto input = trace.at(key);
                const int prefix = int(input.shape(1)) - image_rows;
                auto sample = tc::mx::astype(
                    tc::mx::squeeze(tc::slice_axis(input, 1, prefix, input.shape(1)), 0),
                    tc::mx::float32);
                tc::require(tc::mx::all(tc::mx::isfinite(sample)).item<bool>(),
                            "nonfinite Qwen21 calibration activation");
                auto folder = directory / ("block" + std::to_string(block));
                std::filesystem::create_directory(folder);
                tc::mx::save((folder / "step1.npy").string(), sample);
                std::cerr << "calibration block " << block + 1 << '/' << config.layers << '\n';
            }
            return 0;
        }
        if (argc == 5 && std::string(argv[4]) == "single") {
            float timestep = metadata.count("timestep") ? std::stof(metadata.at("timestep")) : 1.f;
            auto output = model.forward(latents, text, timestep, height, width, false, nullptr, references);
            tc::mx::save_safetensors(argv[3], {{"output", output}});
            return 0;
        }
        if (argc == 5 && std::string(argv[4]) == "trajectory") {
            auto sigmas = tc::mx::astype(inputs.at("sigmas"), tc::mx::float32);
            tc::require(sigmas.ndim() == 1 && sigmas.size() >= 2,
                        "trajectory requires at least two sigmas");
            tc::mx::eval(sigmas);
            std::unordered_map<std::string, tc::Tensor> outputs;
            // Match the official pipeline's BF16 timestep and FP32 Euler
            // policy, isolating DiT/cache errors from scheduler arithmetic.
            for (int step = 0; step + 1 < int(sigmas.size()); ++step) {
                float sigma = sigmas.data<float>()[step];
                auto rounded = tc::mx::astype(tc::Tensor(sigma * 1000.f), latents.dtype());
                rounded = rounded / tc::Tensor(1000.f, latents.dtype());
                float timestep = tc::mx::astype(rounded, tc::mx::float32).item<float>();
                auto noise = model.forward(latents, text, timestep, height, width, true, nullptr, references);
                float dt = sigmas.data<float>()[step + 1] - sigma;
                latents = tc::mx::astype(tc::mx::astype(latents, tc::mx::float32) +
                    tc::mx::astype(noise, tc::mx::float32) * dt, latents.dtype());
                tc::mx::eval(latents, noise);
                outputs.emplace("noise_" + std::to_string(step), noise);
                outputs.emplace("latent_" + std::to_string(step), latents);
            }
            tc::mx::save_safetensors(argv[3], outputs);
            return 0;
        }
        auto first = model.forward(latents, text, 0.8f, height, width, true, nullptr, references);
        tc::mx::eval(first);
        tc::require(first.dtype() == latents.dtype(), "Qwen21 activation dtype promoted");
        tc::require(model.cached_layers() == size_t(config.layers), "prefix cache did not populate");
        auto cached = model.forward(latents, text, 0.3f, height, width, true, nullptr, references);
        tc::mx::eval(cached);
        model.set_decode_mlp([&](int block, const tc::Tensor &input) {
            const auto p = "transformer_blocks." + std::to_string(block) + ".img_mlp.";
            if (weights.has(p + "gate_up.weight")) {
                auto gate_up = tc::mx::split(tc::linear(input, weights, p + "gate_up"), 2, -1);
                return tc::linear(tc::silu(gate_up[0]) * gate_up[1], weights, p + "out");
            }
            return tc::linear(tc::silu(tc::linear(input, weights, p + "gate_layer")) *
                              tc::linear(input, weights, p + "proj"), weights, p + "out");
        });
        auto split_cached = model.forward(latents, text, 0.3f, height, width, true, nullptr, references);
        tc::mx::eval(split_cached);
        model.set_decode_mlp({});
        auto uncached = model.forward(latents, text, 0.3f, height, width, false, nullptr, references);
        tc::mx::eval(uncached);
        tc::require(model.cached_layers() == 0, "cache-disable did not clear cache");
        auto warm = model.forward(latents, text, 0.8f, height, width, true, nullptr, references);
        tc::mx::eval(warm);
        auto changed = model.forward(latents, inputs.at("text_changed"), 0.3f, height, width, true, nullptr, references);
        tc::mx::eval(changed);
        auto changed_reference = model.forward(latents, inputs.at("text_changed"), 0.3f, height, width, false, nullptr, references);
        tc::mx::eval(changed_reference);
        std::unordered_map<std::string, tc::Tensor> outputs{{"first", first}, {"cached", cached}, {"split_cached", split_cached},
                                          {"uncached", uncached}, {"changed", changed},
                                          {"changed_reference", changed_reference}};
        if (!references.empty()) {
            tc::mx::eval(model.forward(latents, text, 0.8f, height, width, true, nullptr, references));
            references[0].latents = inputs.at("reference_changed");
            auto changed_image = model.forward(latents, text, 0.3f, height, width, true, nullptr, references);
            tc::mx::eval(changed_image);
            auto changed_image_uncached = model.forward(latents, text, 0.3f, height, width, false, nullptr, references);
            tc::mx::eval(changed_image_uncached);
            outputs.emplace("changed_image", changed_image);
            outputs.emplace("changed_image_uncached", changed_image_uncached);
        }
        tc::mx::save_safetensors(argv[3], outputs);
        if (argc == 5) {
            int iterations = std::stoi(argv[4]);
            tc::require(iterations > 0, "benchmark iterations must be positive");
            for (bool cache : {false, true}) {
                model.reset();
                for (int iteration = -2; iteration < iterations; ++iteration) {
                    auto start = tc::Clock::now();
                    auto output = model.forward(latents, text, 0.3f, height, width, cache, nullptr, references);
                    tc::mx::eval(output);
                    double elapsed = std::chrono::duration<double>(tc::Clock::now() - start).count();
                    if (iteration >= 0)
                        std::cout << "{\"cache\":" << (cache ? "true" : "false")
                                  << ",\"iteration\":" << iteration << ",\"seconds\":" << elapsed << "}\n";
                }
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

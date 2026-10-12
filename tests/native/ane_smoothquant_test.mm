#include "../../native/backends/ane_smoothquant.hpp"
#include <fstream>
#include <iostream>

namespace sq = tc::ane::smoothquant;
namespace cal = tc::ane::calibration;
static void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
static sq::Binding binding(const std::filesystem::path &checkpoint) {
    sq::Binding result;
    result.experimental = true; result.checkpoint = checkpoint;
    result.model_id = "qwen-image-2.1"; result.hidden = 8; result.total_steps = 6;
    result.reference_size = 512; result.reference_count = 1;
    result.runtime_recipe = "w8a8:channels=4096"; result.configured_backend = "private";
    result.loras = {{"a.safetensors",sq::fingerprint_adapter("existing-bound-adapter-identity",0),.7},
                    {"b.safetensors",sq::fingerprint_adapter("existing-bound-adapter-identity",1),-.2}};
    return result;
}
int main(int argc, char **argv) {
    try {
        check(argc >= 3, "expected mode and CPU fake source");
        const std::string mode = argv[1];
        const auto source = std::filesystem::path(argv[2]);
        if (mode == "fingerprints") {
            std::cout << sq::checkpoint_fingerprint(source) << '\n'
                      << sq::fingerprint_adapter("existing-bound-adapter-identity",0) << '\n'
                      << sq::fingerprint_adapter("existing-bound-adapter-identity",1) << '\n';
            return 0;
        }
        check(argc >= 4, "expected profile or capture directory");
        auto expected = binding(source);
        if (mode == "capture") {
            cal::Config config; config.enabled = true;
            config.layers = expected.layers; config.steps = {0,3,5}; config.rows_per_point = 8;
            cal::Metadata metadata;
            metadata.request_id = "cpu-full-s1-fixture"; metadata.model_id = expected.model_id;
            metadata.model_fingerprint = sq::checkpoint_fingerprint(source);
            metadata.identity_kind = "canonical-stat-identity"; metadata.recipe = cal::capture_recipe;
            metadata.loras = expected.loras; metadata.seed = 42;
            metadata.width = metadata.height = 512; metadata.total_steps = 6;
            metadata.reference_count = 1; metadata.reference_size = 512;
            metadata.runtime_recipe = expected.runtime_recipe; metadata.configured_backend = expected.configured_backend;
            metadata.execution_route = "requested-runtime-ffn"; metadata.actual_execution = "unknown-candidate-trajectory";
            cal::Sampler sampler(config,metadata);
            std::vector<float> maxima(8,2.f); maxima[3] = 4.f;
            for (const auto layer : config.layers) {
                sampler.set_weight_channel_max(layer,maxima,metadata.model_fingerprint);
                for (const auto step : config.steps)
                    sampler.observe_rows({layer,step,step == 0 ? "prefill" : "denoise",
                        {{"text-0",0,2},{"reference-0",2,8},{"target",8,17}}},17,8,
                        [](size_t first, size_t count, auto values) {
                            std::fill(values.begin(),values.end(),uint16_t(0x3c00));
                            if (first <= 12 && first + count > 12) values[(12-first)*8+3] = 0x4c00;
                        });
            }
            check(sampler.complete() && sampler.records().size() == 96 && sampler.input_bytes() == 12288 &&
                  sampler.statistics_bytes() == 4096, "full tiny calibration accounting mismatch");
            sampler.write(argv[3]); std::cout << "PASS CPU full32x3x8 capture\n"; return 0;
        }
        check(mode == "load", "unknown CPU test mode");
        if (argc >= 5) {
            const std::string mutation = argv[4];
            if (mutation == "no-opt-in") expected.experimental = false;
            else if (mutation == "model") expected.model_id = "z-image-turbo";
            else if (mutation == "hidden") expected.hidden = 9;
            else if (mutation == "steps") expected.total_steps = 7;
            else if (mutation == "reference-count") expected.reference_count = 2;
            else if (mutation == "reference-size") expected.reference_size = 1024;
            else if (mutation == "recipe") expected.runtime_recipe = "fp16:channels=4096";
            else if (mutation == "backend") expected.configured_backend = "auto";
            else if (mutation == "adapter-order") std::reverse(expected.loras.begin(),expected.loras.end());
            else if (mutation == "adapter-strength") expected.loras[0].strength = .8;
            else if (mutation == "adapter-count") expected.loras.pop_back();
            else if (mutation == "partial") expected.layers.pop_back();
            else if (mutation == "alpha") expected.alpha = .4;
            else throw std::runtime_error("unknown binding mutation");
        }
        const auto profile = sq::ExperimentalProfile::load(argv[3],expected);
        check(profile.layers().size() == 32 && profile.find(-1) == nullptr && profile.find(32) == nullptr &&
              profile.scales_for(0).size() == 8 && profile.scales_for(31).size() == 8,
              "loaded full S1 layer shape mismatch");
        std::cout << profile.content_digest() << '\n' << profile.calibration_digest() << '\n'
                  << profile.layers().size() << '\n' << profile.alpha() << '\n' << profile.scales_for(0)[3] << '\n';
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

// Bounded local-model check: three encoder-only prepares, no denoising.
#include "../../native/models/qwen21/pipeline.hpp"
#include "../../native/platform/apple/bridge.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            tc::require(argc == 4, "usage: qwen21-reference-cache-probe MODEL REFERENCE OUTPUT_DIRECTORY");
            const auto output = std::filesystem::absolute(argv[3]);
            tc::require(std::filesystem::create_directory(output), "probe output already exists");
            tc::configure_streams();
            std::atomic<bool> cancelled{false};
            tc::Event event = [](const std::string &phase, int step, int total) {
                std::cerr << phase << ' ' << step << '/' << total << '\n';
            };
            tc::Request r;
            r.model = "qwen-image-2.1"; r.operation = "image.edit";
            r.width = r.height = 512; r.steps = 20; r.audio = false;
            r.execution = "gpu"; r.residency = "component_staged";
            r.prompt = "Change the teapot in <image1> to cobalt blue.";
            r.inputs.push_back({.kind="image", .role="reference",
                               .path=std::filesystem::absolute(argv[2]).string()});
            tc::qwen21::Session session(std::filesystem::absolute(argv[1]));
            auto prepare = [&](const char *name) {
                r.dump = (output / name).string();
                auto result = session.prepare(r, false, event, cancelled);
                std::ofstream report(output / (std::string(name) + ".json"));
                report << tc::json(tc::to_dictionary(result)) << '\n';
                tc::require(bool(report), "cannot write probe report");
                return result;
            };
            prepare("first");
            // Content identity survives a path change; language conditioning
            // must still be recomputed for the new prompt.
            const auto alias = output / "same-reference.png";
            std::filesystem::copy_file(r.inputs[0].path, alias);
            r.inputs[0].path = alias.string();
            r.prompt = "Change the teapot in <image1> to emerald green, preserve its shape.";
            auto cached = prepare("reused");
            tc::require(!cached.prompt_cache_hit &&
                cached.selection.find("reused reference encodings=1") != std::string::npos,
                "changed prompt did not reuse only the reference encoders");
            session.unload();
            auto cold = prepare("cold");
            tc::require(cold.selection.find("reused reference encodings") == std::string::npos,
                        "unload retained the reference cache");
            for (const char *name : {"qwen21_text", "qwen21_reference_0"}) {
                auto [a, am] = tc::mx::load_safetensors((output / "reused" / (std::string(name)+".safetensors")).string());
                auto [b, bm] = tc::mx::load_safetensors((output / "cold" / (std::string(name)+".safetensors")).string());
                tc::require(tc::mx::array_equal(a.at("tensor"), b.at("tensor")).item<bool>(),
                            std::string("cached/cold tensor differs: ") + name);
            }
            session.unload();
            std::cout << "{\"scope\":\"encoder-only; three prepares; no image generation\","
                "\"bit_identical_text_and_reference\":true,\"reused_encoder_seconds\":"
                << cached.timings.text + cached.timings.image << ",\"cold_encoder_seconds\":"
                << cold.timings.text + cold.timings.image << "}\n";
            return 0;
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n'; return 1;
        }
    }
}

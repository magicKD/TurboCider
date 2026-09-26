// Native production-Session lifecycle and warm-request benchmark. No Python.
#include "../../native/models/qwen21/pipeline.hpp"
#include "../../native/platform/apple/bridge.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            tc::require(argc >= 6 && argc <= 8,
                "usage: qwen21-session-probe MODEL OUTPUT_DIRECTORY STEPS REPEATS MANIFEST_OR_DASH [PROMPT] [--dump-tensors]");
            tc::configure_streams();
            const auto directory = std::filesystem::absolute(argv[2]);
            tc::require(!std::filesystem::exists(directory), "benchmark directory already exists");
            std::filesystem::create_directories(directory);
            tc::Request r;
            r.model = "qwen-image-2.1";
            r.audio = false; r.width = r.height = 512; r.residency = "resident";
            r.steps = std::stoi(argv[3]);
            const int repeats = std::stoi(argv[4]);
            tc::require(repeats >= 1 && repeats <= 10, "repeats must be 1...10");
            const bool dump_tensors = argc == 8;
            tc::require(!dump_tensors || std::string(argv[7]) == "--dump-tensors", "unknown session probe option");
            r.prompt = argc >= 7 ? argv[6] : "A ceramic teapot on a wooden table, warm sunlight, detailed photography.";
            if (std::string(argv[5]) != "-") {
                r.execution = "gpu_ane"; r.allow_approximation = true;
                r.ane_manifest = std::filesystem::absolute(argv[5]).string();
            }
            tc::make_plan(r);
            tc::qwen21::Session session(std::filesystem::absolute(argv[1]));
            std::atomic<bool> cancelled{false};
            tc::Event event = [](const std::string &phase, int step, int total) {
                std::cerr << phase << ' ' << step << '/' << total << std::endl;
            };
            auto save = [&](const char *name, const tc::RunResult &result) {
                auto value = tc::to_dictionary(result);
                std::ofstream file(directory / name);
                file << tc::json(value) << '\n';
                tc::require(bool(file), "failed writing benchmark report");
            };
            r.output = (directory / "must-not-export.png").string();
            auto prepared = session.prepare(r, false, event, cancelled);
            save("prepare.json", prepared);
            tc::require(!std::filesystem::exists(r.output), "prepare exported an image");
            auto warm = r; warm.steps = 2;
            auto warmed = session.prepare(warm, true, event, cancelled);
            save("warmup.json", warmed);
            tc::require(!std::filesystem::exists(r.output), "warmup exported an image");
            uint64_t previous_calls = warmed.hybrid ? warmed.hybrid->runtime_calls : 0;
            for (int iteration = 0; iteration < repeats; ++iteration) {
                auto name = "run-" + std::to_string(iteration);
                r.output = (directory / (name + ".png")).string();
                // Persist request-owned text/noise/latent tensors per repeat so
                // the external comparison can prove identical inputs rather
                // than relying only on request metadata.
                r.dump = dump_tensors ? (directory / (name + "-dump")).string() : "";
                auto result = session.generate(r, event, cancelled);
                tc::require(result.prompt_cache_hit, "resident prompt cache was not reused");
                tc::require(std::filesystem::is_regular_file(r.output), "generation did not export");
                if (r.execution == "gpu_ane") {
                    tc::require(result.hybrid.has_value() && result.request.execution == "gpu_ane",
                                "hybrid selection/report was lost");
                    tc::require(result.hybrid->runtime_calls - previous_calls == uint64_t(r.steps - 1) * 32,
                                "wrong decode-only hybrid call count");
                    previous_calls = result.hybrid->runtime_calls;
                    tc::require(result.hybrid->load_seconds == warmed.hybrid->load_seconds,
                                "resident Core ML session was reloaded");
                }
                save((name + ".json").c_str(), result);
                std::cout << "{\"iteration\":" << iteration << ",\"wall_seconds\":" << result.timings.wall
                          << ",\"denoise_seconds\":" << result.timings.denoise
                          << ",\"setup_seconds\":" << result.timings.hybrid << "}" << std::endl;
            }
            // Cancellation happens between prefill and decode. It must not
            // export or leave a partially consumed hybrid prefix for a retry.
            warm.output = (directory / "cancelled.png").string();
            bool caught = false;
            try {
                session.generate(warm, [&](const std::string &phase, int completed, int) {
                    if (phase == "denoise" && completed == 1) cancelled = true;
                }, cancelled);
            } catch (const tc::Cancelled &) { caught = true; }
            tc::require(caught && !std::filesystem::exists(warm.output), "cancellation/export contract failed");
            cancelled = false;
            // Switching to GPU must release the Core ML bank and report GPU.
            warm.execution = "gpu"; warm.ane_manifest.clear(); warm.allow_approximation = false;
            auto gpu = session.prepare(warm, false, event, cancelled);
            tc::require(!gpu.hybrid && gpu.request.execution == "gpu", "GPU switch retained hybrid route");
            session.unload();
            std::cout << "{\"lifecycle\":\"passed\",\"quality\":\"requires visual inspection\"}" << std::endl;
            return 0;
        } catch (const std::exception &error) {
            std::cerr << error.what() << std::endl;
            return 1;
        }
    }
}

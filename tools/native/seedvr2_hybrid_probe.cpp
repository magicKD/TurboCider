#include "../../native/models/seedvr2/hybrid.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <mlx/random.h>
#include <fstream>
#include <iostream>
#include <cmath>

namespace mx = mlx::core;
using tc::Tensor;

int main(int argc, char **argv) {
    try {
        tc::require(argc == 5 || argc == 6, "usage: seedvr2-hybrid-probe MODEL_ROOT MANIFEST FRESH_OUTPUT_DIR TOKEN_ROWS [cpu-only]");
        const bool cpu_only_control = argc == 6;
        tc::require(!cpu_only_control || std::string(argv[5]) == "cpu-only",
                    "unknown prefix compute-unit diagnostic control");
        const std::filesystem::path root = argv[2], output = argv[3];
        tc::require(!std::filesystem::exists(output), "fresh hybrid probe output directory required");
        const int tokens = std::stoi(argv[4]);
        tc::require(tokens > 0 && tokens <= 131072, "hybrid probe token rows must be 1...131072");
        const auto config = tc::seedvr2::read_hybrid_config(root);
        mx::set_default_device(mx::Device::gpu);
        tc::seedvr2::AllocatorCacheScope allocator(4ull << 30);
        tc::seedvr2::DiT dit;
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        const auto started = tc::Clock::now();
        const auto dtype = config.source_operand_recipe == "original-fp16" ? mx::float16 : mx::bfloat16;
        dit.load(std::filesystem::path(argv[1]) / "diffusion_models/seedvr2_3b_fp16.safetensors",dtype);
        tc::seedvr2::HybridFFN hybrid(dit, config, event, cancelled,cpu_only_control);
        const auto loaded = tc::Clock::now();
        const int block = config.blocks.front();
        Tensor input(0);
        std::filesystem::create_directories(output);
        std::ofstream report(output / "report.json");
        report << "{\"schema\":\"seedvr2-hybrid-ffn-probe-v1\",\"block\":" << block
               << ",\"tokens\":" << tokens << ",\"ane_channels\":" << config.channels
               << ",\"manifest_sha256\":\"" << config.manifest_sha256 << "\""
               << ",\"source_operand_recipe\":\"" << config.source_operand_recipe << "\""
               << ",\"schedule\":\"" << hybrid.schedule() << "\""
               << ",\"prefix_compute_units\":\"" << hybrid.prefix_compute_units() << "\""
               << ",\"cpu_only_control\":" << (cpu_only_control ? "true" : "false")
               << ",\"input_recipe\":\"independent_row_chunks_and_seed42_plus_iteration\""
               << ",\"load_seconds\":" << std::chrono::duration<double>(loaded-started).count()
               << ",\"iterations\":[";
        Tensor reference(0), actual(0);
        bool passed=true;
        for (int iteration = 0; iteration < 4; ++iteration) {
            // Vary every chunk and every invocation. Identical repeated data
            // can hide a stale output even if the pool overwrites unsafely.
            input = mx::random::normal({tokens,2560},dtype,mx::random::key(42+iteration));
            mx::eval(input);
            auto begin = tc::Clock::now();
            reference = dit.ffn_full(input,block); mx::eval(reference);
            const auto gpu_end = tc::Clock::now();
            const auto calls = hybrid.calls();
            actual = hybrid.predict(block,input);
            const auto hybrid_end = tc::Clock::now();
            tc::require(hybrid.calls() > calls && mx::all(mx::isfinite(actual)).item<bool>(), "hybrid prefix did not execute or produced nonfinite output");
            auto x = mx::astype(reference,mx::float32), y = mx::astype(actual,mx::float32);
            const double nx = mx::sum(x*x).item<float>(), ny = mx::sum(y*y).item<float>();
            const double l2 = std::sqrt(mx::sum((x-y)*(x-y)).item<float>()/nx);
            const double cosine = mx::sum(x*y).item<float>()/std::sqrt(nx*ny);
            double worst_chunk_l2 = 0, worst_chunk_cosine = 1;
            bool chunks_passed = true;
            for (int first = 0; first < tokens; first += config.rows) {
                const int stop = std::min(tokens, first + config.rows);
                auto a = mx::slice(x,{first,0},{stop,2560});
                auto b = mx::slice(y,{first,0},{stop,2560});
                const double ax = mx::sum(a*a).item<float>(), bx = mx::sum(b*b).item<float>();
                const double chunk_l2 = std::sqrt(mx::sum((a-b)*(a-b)).item<float>()/ax);
                const double chunk_cosine = mx::sum(a*b).item<float>()/std::sqrt(ax*bx);
                worst_chunk_l2 = std::max(worst_chunk_l2,chunk_l2);
                worst_chunk_cosine = std::min(worst_chunk_cosine,chunk_cosine);
                chunks_passed &= std::isfinite(chunk_l2) && std::isfinite(chunk_cosine) &&
                    chunk_l2<=.01 && chunk_cosine>=.99999;
            }
            const bool ok=std::isfinite(l2) && std::isfinite(cosine) && l2<=.01 && cosine>=.99999 && chunks_passed;
            passed &= ok;
            if (iteration) report << ',';
            report << "{\"iteration\":" << iteration << ",\"gpu_seconds\":" << std::chrono::duration<double>(gpu_end-begin).count()
                   << ",\"hybrid_seconds\":" << std::chrono::duration<double>(hybrid_end-gpu_end).count()
                   << ",\"coreml_calls\":" << hybrid.calls()-calls << ",\"relative_l2\":" << l2 << ",\"cosine\":" << cosine
                   << ",\"seed\":" << 42+iteration << ",\"worst_chunk_relative_l2\":" << worst_chunk_l2
                   << ",\"worst_chunk_cosine\":" << worst_chunk_cosine
                   << ",\"chunk_gates_passed\":" << (chunks_passed ? "true" : "false")
                   << ",\"parity_passed\":" << (ok ? "true" : "false") << '}';
            std::cout << "iteration=" << iteration << " gpu=" << std::chrono::duration<double>(gpu_end-begin).count()
                      << " hybrid=" << std::chrono::duration<double>(hybrid_end-gpu_end).count()
                      << " relative_l2=" << l2 << " cosine=" << cosine << '\n';
        }
        report << "],\"prefix_snapshot_copied_bytes\":" << hybrid.snapshot_bytes()
               << ",\"prefix_backing_slots\":" << hybrid.prefix_backing_slots()
               << ",\"prefix_backing_bytes\":" << hybrid.prefix_backing_bytes()
               << ",\"coreml_output_copied_bytes\":" << hybrid.copied_bytes()
               << ",\"parity_passed\":" << (passed ? "true" : "false")
               << ",\"scope\":\"one-block synthetic FFN; explicit Core ML compute-unit setting, not physical ANE occupancy or complete model qualification\"}\n";
        report.close();
        mx::save_safetensors((output/"ffn.safetensors").string(), {{"input",input},{"gpu",reference},{"hybrid",actual}});
        tc::require(passed,"SeedVR2 FFN numerical gate failed");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

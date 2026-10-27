#include "../../native/models/seedvr2/hybrid.hpp"
#include "../../native/models/seedvr2/pipeline.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <fstream>
#include <iostream>
#include <charconv>
#include <set>

namespace mx = mlx::core;
int main(int argc, char **argv) {
    try {
        tc::require(argc >= 5 && argc <= 8, "usage: seedvr2-hybrid-restore-probe MODEL_ROOT MANIFEST INPUT.safetensors FRESH_OUTPUT [BLOCKS_CSV|gpu] [REPEATS] [stage-dit]");
        const bool stage_dit = argc == 8;
        tc::require(!stage_dit || std::string(argv[7]) == "stage-dit", "unknown weight residency diagnostic control");
        const std::filesystem::path destination = argv[4];
        tc::require(!std::filesystem::exists(destination), "fresh hybrid restoration directory required");
        auto config = tc::seedvr2::read_hybrid_config(argv[2]);
        const bool gpu_only = argc >= 6 && std::string(argv[5]) == "gpu";
        int repeats = 1;
        if (argc >= 7) {
            const std::string value = argv[6];
            const auto parsed = std::from_chars(value.data(),value.data()+value.size(),repeats);
            tc::require(parsed.ec == std::errc{} && parsed.ptr == value.data()+value.size() &&
                repeats > 0 && repeats <= 20, "diagnostic repeats must be 1...20");
        }
        if (argc >= 6 && !gpu_only) {
            // Validate the complete bank first, then restrict this diagnostic's
            // coverage. Never relabel a partial-bank run as all-32 execution.
            std::set<int> selected;
            const std::string requested = argv[5];
            for (size_t first = 0; first < requested.size();) {
                const auto comma = requested.find(',', first);
                const auto stop = comma == std::string::npos ? requested.size() : comma;
                int block = -1;
                const auto parsed = std::from_chars(requested.data()+first, requested.data()+stop, block);
                tc::require(parsed.ec == std::errc{} && parsed.ptr == requested.data()+stop &&
                    block >= 0 && block < 32 && selected.insert(block).second, "invalid diagnostic hybrid block list");
                first = stop + 1;
            }
            tc::require(!selected.empty() && requested.back() != ',', "nonempty diagnostic hybrid block list required");
            std::vector<int> blocks;
            std::vector<std::filesystem::path> artifacts;
            for (size_t index = 0; index < config.blocks.size(); ++index) {
                const int block = config.blocks[index];
                if (!selected.erase(block)) continue;
                blocks.push_back(block); artifacts.push_back(config.artifacts[index]);
            }
            tc::require(selected.empty(), "requested hybrid block is absent from the verified bank");
            config.blocks = std::move(blocks); config.artifacts = std::move(artifacts);
        }
        mx::set_default_device(mx::Device::gpu);
        tc::seedvr2::AllocatorCacheScope allocator(tc::seedvr2::allocator_cache_limit());
        std::atomic<bool> cancelled{false};
        auto event = [](const std::string &name, int step, int total) { std::cerr << name << ' ' << step << '/' << total << '\n'; };
        const auto begin = tc::Clock::now();
        tc::seedvr2::Pipeline pipeline;
        pipeline.set_stage_dit(stage_dit);
        const auto dtype = config.source_operand_recipe == "original-fp16" ? mx::float16 : mx::bfloat16;
        pipeline.load(argv[1], event, cancelled,dtype);
        std::unique_ptr<tc::seedvr2::HybridFFN> hybrid;
        if (!gpu_only) hybrid = std::make_unique<tc::seedvr2::HybridFFN>(pipeline.dit(), config, event, cancelled);
        const auto loaded = tc::Clock::now();
        auto inputs = mx::load_safetensors(argv[3]).first;
        std::filesystem::create_directories(destination);
        std::ofstream report(destination/"report.json");
        report << "{\"schema\":\"seedvr2-hybrid-restoration-probe-v2\",\"load_seconds\":"
               << std::chrono::duration<double>(loaded-begin).count()
               << ",\"source_operand_recipe\":\"" << config.source_operand_recipe << "\""
               << ",\"weight_residency_recipe\":\"" << pipeline.weight_residency_recipe() << "\""
               << ",\"route\":\"" << (gpu_only ? "gpu" : "gpu_cpu_ne") << "\",\"iterations\":[";
        tc::seedvr2::Restoration result{tc::Tensor(0),tc::Tensor(0),tc::Tensor(0),tc::Tensor(0)};
        const tc::seedvr2::FFNRoute route = gpu_only ? tc::seedvr2::FFNRoute{} : tc::seedvr2::FFNRoute(
            [&](int block, const tc::Tensor &x) { return hybrid->predict(block,x); });
        for (int iteration = 0; iteration < repeats; ++iteration) {
            result = {tc::Tensor(0),tc::Tensor(0),tc::Tensor(0),tc::Tensor(0)};
            mx::synchronize(); mx::reset_peak_memory();
            const auto calls = hybrid ? hybrid->calls() : 0;
            result = pipeline.restore(inputs.at("pixels"), inputs.at("noise"), event, cancelled, {},route);
            const auto peak = mx::get_peak_memory();
            tc::require((gpu_only || hybrid->calls() > calls) && mx::all(mx::isfinite(result.pixels)).item<bool>(),
                        "hybrid restoration did not execute a prefix or produced nonfinite output");
            if (iteration) report << ',';
            report << "{\"iteration\":" << iteration << ",\"encode_seconds\":" << result.encode_seconds
                   << ",\"denoise_seconds\":" << result.denoise_seconds << ",\"decode_seconds\":" << result.decode_seconds
                   << ",\"staging_seconds\":" << result.staging_seconds << ",\"restore_seconds\":" << result.restore_seconds
                   << ",\"resident_parameter_bytes_at_end\":" << pipeline.bytes()
                   << ",\"active_peak_bytes\":" << peak << ",\"coreml_calls\":" << (hybrid ? hybrid->calls()-calls : 0) << '}';
        }
        mx::save_safetensors((destination/"restoration.safetensors").string(),
            {{"pixels",result.pixels},{"condition",result.condition},{"velocity",result.velocity},{"latent",result.latent}});
        report << "],\"coreml_calls\":" << (hybrid ? hybrid->calls() : 0)
               << ",\"schedule\":\"" << (hybrid ? hybrid->schedule() : "gpu_full") << "\""
               << ",\"prefix_compute_units\":\"" << (hybrid ? hybrid->prefix_compute_units() : "not_applicable") << "\""
               << ",\"prefix_batches\":" << (hybrid ? hybrid->batches() : 0)
               << ",\"prefix_snapshot_copied_bytes\":" << (hybrid ? hybrid->snapshot_bytes() : 0)
               << ",\"prefix_backing_slots\":" << (hybrid ? hybrid->prefix_backing_slots() : 0)
               << ",\"prefix_backing_bytes\":" << (hybrid ? hybrid->prefix_backing_bytes() : 0)
               << ",\"video_ffn_hybrid_blocks\":" << (hybrid ? hybrid->hybrid_blocks() : 0)
               << ",\"video_ffn_gpu_blocks\":" << (hybrid ? hybrid->gpu_blocks() : 32*repeats)
               << ",\"coreml_output_copied_bytes\":" << (hybrid ? hybrid->copied_bytes() : 0)
               << ",\"manifest_sha256\":\"" << config.manifest_sha256 << "\",\"selected_blocks\":[";
        for (size_t index = 0; !gpu_only && index < config.blocks.size(); ++index) {
            if (index) report << ',';
            report << config.blocks[index];
        }
        report << "],\"scope\":\"full model component restoration; text FFN stays GPU; CPU+NE configuration does not prove physical ANE occupancy\"}\n";
        std::cout << "restoration saved; coreml_calls=" << (hybrid ? hybrid->calls() : 0) << '\n';
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

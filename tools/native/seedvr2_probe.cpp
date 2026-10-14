#include "../../native/models/seedvr2/model.hpp"
#include "../../native/models/seedvr2/vae.hpp"
#include "../../native/models/seedvr2/pipeline.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <fstream>
#include <iostream>
#include <sys/resource.h>

namespace mx = mlx::core;
using tc::Tensor;

int main(int argc, char **argv) {
    try {
        tc::require(argc >= 2, "usage: seedvr2-probe windows T H W SHIFT | dit MODEL INPUT OUTPUT [bf16|fp16|fp32] | vae MODEL INPUT OUTPUT [REPEATS]");
        const std::string command = argv[1];
        if (command == "windows") {
            tc::require(argc == 6, "windows requires T H W SHIFT");
            const auto windows = tc::seedvr2::make_windows(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
            std::cout << '[';
            for (size_t i = 0; i < windows.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << '[';
                for (int j = 0; j < 3; ++j) {
                    if (j) std::cout << ',';
                    std::cout << '[' << windows[i].start[j] << ',' << windows[i].stop[j] << ']';
                }
                std::cout << ']';
            }
            std::cout << "]\n";
            return 0;
        }
        tc::require(argc >= 5, "model, input safetensors and output directory required");
        const std::filesystem::path model = argv[2], destination = argv[4];
        tc::require(!std::filesystem::exists(destination), "a fresh SeedVR2 probe output directory is required");
        mx::set_default_device(mx::Device::gpu);
        auto inputs = mx::load_safetensors(argv[3]).first;
        std::filesystem::create_directories(destination);
        std::atomic<bool> cancelled{false};
        auto event = [](const std::string &stage, int step, int total) {
            std::cerr << stage << ' ' << step << '/' << total << '\n';
        };
        const auto started = tc::Clock::now();
        if (command == "dit") {
            tc::seedvr2::DiT dit;
            const auto mode = argc > 5 ? std::string(argv[5]) : "bf16";
            tc::require(mode == "bf16" || mode == "fp16" || mode == "fp32", "invalid probe compute dtype");
            auto dtype = mode == "bf16" ? mx::bfloat16 : mode == "fp16" ? mx::float16 : mx::float32;
            dit.load(model / "diffusion_models/seedvr2_3b_fp16.safetensors", dtype);
            auto loaded = tc::Clock::now();
            auto velocity = dit.forward(inputs.at("noise"), inputs.at("condition"), 1000.f, event, cancelled, destination);
            auto finished = tc::Clock::now();
            auto restored = mx::astype(inputs.at("noise"), mx::float32) - mx::astype(velocity, mx::float32);
            mx::save_safetensors((destination / "sampling.safetensors").string(), {{"restored", restored}});
            std::cout << "{\"load_seconds\":" << std::chrono::duration<double>(loaded - started).count()
                      << ",\"inference_seconds_with_dumps\":" << std::chrono::duration<double>(finished - loaded).count()
                      << ",\"weight_bytes\":" << dit.bytes()
                      << ",\"rms_recipe\":\"" << dit.rms_recipe() << "\""
                      << ",\"time_recipe\":\"" << dit.time_recipe() << "\""
                      << ",\"attention_recipe\":\"" << dit.attention_recipe() << "\""
                      << ",\"rope_recipe\":\"" << dit.rope_recipe() << "\""
                      << ",\"linear_recipe\":\"" << dit.linear_recipe() << "\""
                      << ",\"residual_recipe\":\"" << dit.residual_recipe() << "\"}\n";
        } else if (command == "vae") {
            tc::require(argc <= 6, "vae accepts only an optional positive repeat count");
            const int repeats = argc == 6 ? std::stoi(argv[5]) : 1;
            tc::require(repeats > 0 && repeats <= 20, "VAE repeats must be between one and twenty");
            size_t cache_bytes = tc::seedvr2::allocator_cache_limit();
            if (const char *value = std::getenv("TURBOCIDER_SEEDVR2_VAE_CACHE_BYTES"))
                cache_bytes = std::stoull(value);
            tc::seedvr2::AllocatorCacheScope allocator(cache_bytes);
            tc::seedvr2::VAE vae;
            vae.load(model / "vae/seedvr2_ema_vae_fp16.safetensors");
            auto loaded = tc::Clock::now();
            const auto load_peak = mx::get_peak_memory();
            Tensor latent(0), output(0);
            std::ofstream report(destination / "report.json");
            tc::require(bool(report), "cannot write VAE probe report");
            report << "{\"schema\":\"seedvr2-vae-probe-v2\",\"load_seconds\":"
                   << std::chrono::duration<double>(loaded - started).count()
                   << ",\"weight_bytes\":" << vae.bytes()
                   << ",\"load_peak_bytes\":" << load_peak
                   << ",\"allocator_cache_limit_bytes\":" << cache_bytes
                   << ",\"attention_recipe\":\"full_spatial_keys_fp32_query512\""
                   << ",\"conv_tile_rows_limit\":" << vae.conv_tile_rows()
                   << ",\"norm_recipe\":\"" << vae.norm_recipe() << "\""
                   << ",\"history_recipe\":\"" << vae.history_recipe() << "\""
                   << ",\"pointwise_recipe\":\"" << vae.pointwise_recipe() << "\""
                   << ",\"conv_recipe\":\"" << vae.conv_recipe() << "\",\"iterations\":[";
            for (int iteration = 0; iteration < repeats; ++iteration) {
                // No retained decoded output or cached latent from an earlier
                // iteration. The model and allocator caches may stay warm.
                latent = Tensor(0); output = Tensor(0);
                mx::synchronize();
                mx::reset_peak_memory();
                const auto encode_start = tc::Clock::now();
                latent = vae.encode(inputs.at("pixels"), event, cancelled);
                const auto encoded = tc::Clock::now();
                const auto encode_peak = mx::get_peak_memory();
                tc::require(mx::all(mx::isfinite(latent)).item<bool>(), "VAE encoded a nonfinite latent");
                mx::reset_peak_memory();
                output = vae.decode(latent, event, cancelled);
                const auto decoded = tc::Clock::now();
                const auto decode_peak = mx::get_peak_memory();
                tc::require(mx::all(mx::isfinite(output)).item<bool>(), "VAE decoded nonfinite pixels");
                struct rusage usage{};
                getrusage(RUSAGE_SELF, &usage);
                if (iteration) report << ',';
                report << "{\"iteration\":" << iteration
                       << ",\"encode_seconds\":" << std::chrono::duration<double>(encoded - encode_start).count()
                       << ",\"decode_seconds\":" << std::chrono::duration<double>(decoded - encoded).count()
                       << ",\"encode_peak_bytes\":" << encode_peak
                       << ",\"decode_peak_bytes\":" << decode_peak
                       << ",\"mlx_active_bytes\":" << mx::get_active_memory()
                       << ",\"mlx_cache_bytes\":" << mx::get_cache_memory()
                       << ",\"process_peak_rss_bytes\":" << usage.ru_maxrss << '}';
                std::cerr << "VAE iteration " << iteration << " encode="
                          << std::chrono::duration<double>(encoded - encode_start).count()
                          << " decode=" << std::chrono::duration<double>(decoded - encoded).count()
                          << " peak=" << std::max(encode_peak, decode_peak) << '\n';
            }
            mx::save_safetensors((destination / "vae.safetensors").string(), {{"latent", latent}, {"decoded", output}});
            report << "]}\n";
            std::cout << "VAE report written to " << destination / "report.json" << '\n';
        } else if (command == "restore" || command == "restore-owned") {
            tc::require(argc <= 7,"restore accepts an optional dit dtype and no-dumps control");
            const bool owned_input = command == "restore-owned";
            const auto mode = argc >= 6 ? std::string(argv[5]) : (owned_input ? "fp16" : "bf16");
            const bool dump_intermediates = argc != 7;
            tc::require(dump_intermediates || std::string(argv[6]) == "no-dumps","unknown restoration dump control");
            tc::require(mode=="bf16" || mode=="fp16" || mode=="fp32","invalid restore compute dtype");
            const auto dtype = mode=="bf16" ? mx::bfloat16 : mode=="fp16" ? mx::float16 : mx::float32;
            tc::seedvr2::AllocatorCacheScope allocator(tc::seedvr2::allocator_cache_limit());
            tc::seedvr2::Pipeline pipeline;
            pipeline.load(model, event, cancelled,dtype);
            auto loaded = tc::Clock::now();
            const auto dump = dump_intermediates ? destination : std::filesystem::path{};
            auto result = owned_input
                ? pipeline.restore_owned(std::move(inputs.at("pixels")),inputs.at("noise"),event,cancelled,dump)
                : pipeline.restore(inputs.at("pixels"), inputs.at("noise"), event, cancelled,dump);
            mx::save_safetensors((destination / "sampling.safetensors").string(), {{"restored", result.latent}});
            mx::save_safetensors((destination / "restoration.safetensors").string(), {
                {"pixels", result.pixels}, {"condition", result.condition}, {"velocity", result.velocity}, {"latent", result.latent}});
            std::ofstream report(destination / "report.json");
            report << "{\"load_seconds\":" << std::chrono::duration<double>(loaded - started).count()
                   << ",\"encode_seconds\":" << result.encode_seconds
                   << ",\"denoise_seconds_with_dumps\":" << result.denoise_seconds
                   << ",\"decode_seconds\":" << result.decode_seconds
                   << ",\"restore_total_seconds\":" << result.restore_seconds
                   << ",\"intermediates_dumped\":" << (dump_intermediates ? "true" : "false")
                   << ",\"input_release_recipe\":\"" << (owned_input ? "after_encode_owned" : "borrowed") << "\""
                   << ",\"input_release_seconds\":" << result.input_release_seconds
                   << ",\"released_input_logical_bytes\":" << result.released_input_logical_bytes
                   << ",\"weight_bytes\":" << pipeline.bytes()
                   << ",\"mlx_peak_bytes\":" << mx::get_peak_memory()
                   << ",\"dit_compute_dtype\":\"" << mode << "\""
                   << ",\"dit_residual_recipe\":\"" << pipeline.dit().residual_recipe() << "\""
                   << ",\"dit_time_recipe\":\"" << pipeline.dit().time_recipe() << "\""
                   << ",\"dit_attention_recipe\":\"" << pipeline.dit().attention_recipe() << "\""
                   << ",\"dit_rope_recipe\":\"" << pipeline.dit().rope_recipe() << "\"}\n";
            report.close();
            std::cout << "restoration written to " << destination << '\n';
        } else {
            throw std::invalid_argument("unknown SeedVR2 probe operation");
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

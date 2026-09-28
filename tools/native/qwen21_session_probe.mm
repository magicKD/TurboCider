// Native production-Session lifecycle and warm-request benchmark. No Python.
#include "../../native/models/qwen21/pipeline.hpp"
#include "../../native/models/qwen21/diagnostic_options.hpp"
#include "../../native/platform/apple/bridge.hpp"
#include "../../native/platform/apple/platform.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string_view>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            tc::require(argc >= 6 && argc <= 8,
                "usage: qwen21-session-probe MODEL OUTPUT_DIRECTORY STEPS REPEATS MANIFEST_OR_DASH [PROMPT|--request=JSON] [--dump-tensors]");
            tc::configure_streams();
            const auto directory = std::filesystem::absolute(argv[2]);
            tc::require(!std::filesystem::exists(directory), "benchmark directory already exists");
            std::filesystem::create_directories(directory);
            tc::Request r;
            r.model = "qwen-image-2.1";
            r.audio = false; r.width = r.height = 512; r.residency = "resident";
            r.steps = std::stoi(argv[3]);
            const bool db_cache = tc::qwen21::option_enabled(
                std::getenv("TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC"));
            const bool rectangular_w8a8 = tc::qwen21::option_enabled(
                std::getenv("TURBOCIDER_QWEN21_RECT_W8A8_DIAGNOSTIC"));
            const int repeats = std::stoi(argv[4]);
            tc::require(repeats >= 1 && repeats <= 10, "repeats must be 1...10");
            const bool dump_tensors = argc == 8 && std::string(argv[7]) == "--dump-tensors";
            const bool test_edit_cache = argc == 8 && std::string(argv[7]) == "--test-edit-cache";
            const char *prefix_flag = std::getenv("TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV");
            bool prefix_probe = prefix_flag && std::string_view(prefix_flag) == "1";
            tc::require(argc != 8 || dump_tensors || test_edit_cache, "unknown session probe option");
            const std::string prompt_arg = argc >= 7 ? argv[6] : "A ceramic teapot on a wooden table, warm sunlight, detailed photography.";
            if (prompt_arg.starts_with("--request=")) {
                const std::filesystem::path request_path = prompt_arg.substr(10);
                r = tc::request_from_json(tc::read_json(request_path));
                tc::require(r.model == "qwen-image-2.1" &&
                            ((r.width == 512 && r.height == 512) ||
                             ((r.width == 768 && r.height == 512) ||
                              (r.width == 512 && r.height == 768)) ||
                             (r.width == 1024 && r.height == 1024 &&
                              r.operation == "image.generate" && r.inputs.empty())) &&
                            r.steps == std::stoi(argv[3]) && r.residency == "component_staged",
                            "session probe needs a matching 512px or diagnostic 1024px component-staged Qwen21 request");
                r.residency = "resident";
            } else {
                r.prompt = prompt_arg;
            }
            const bool lora_base_ane = tc::qwen21::lora_base_ane(r);
            const bool gate_up_ane = tc::qwen21::gate_up_ane(r);
            const bool prefix_lora_guard = prefix_probe && !r.loras.empty();
            prefix_probe = prefix_probe && r.loras.empty();
            std::filesystem::path mutable_reference;
            if (test_edit_cache) {
                tc::require(r.inputs.size() >= 2,
                            "edit cache mutation probe needs two distinct reference images");
                mutable_reference = directory / "mutable-reference.png";
                tc::require(tc::sha256_file(r.inputs[0].path) != tc::sha256_file(r.inputs[1].path),
                            "edit cache mutation probe needs different reference bytes");
                std::filesystem::copy_file(r.inputs[0].path, mutable_reference);
                r.inputs[0].path = mutable_reference.string();
            }
            if (std::string(argv[5]) != "-") {
                r.execution = "gpu_ane"; r.allow_approximation = true;
                r.ane_manifest = std::filesystem::absolute(argv[5]).string();
            } else {
                r.execution = "gpu"; r.ane_manifest.clear();
                tc::require(!r.qwen21_w8a8 && r.qwen21_gpu_full_ffn_blocks.empty(),
                            "GPU request cannot retain W8A8 hybrid options");
            }
            if (db_cache) r.allow_approximation = true; // explicit diagnostic probe opt-in
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
            const char *hybrid_final = std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC");
            const char *hybrid_last16 =
                std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC");
            const char *hybrid_half =
                std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC");
            const char *full_warm = std::getenv("TURBOCIDER_QWEN21_PROBE_MATCHED_FULL_WARMUP");
            const char *tiled_prefix_flag =
                std::getenv("TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC");
            const bool tiled_prefix_probe = prefix_probe && tiled_prefix_flag &&
                std::string_view(tiled_prefix_flag) == "1";
            const char *target_only_flag =
                std::getenv("TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC");
            const bool lossy_prefix_hit = tiled_prefix_probe && target_only_flag &&
                std::string_view(target_only_flag) == "1";
            auto warm = r;
            warm.steps = r.loras.empty() &&
                !(hybrid_final && std::string_view(hybrid_final) == "1") &&
                !(hybrid_last16 && std::string_view(hybrid_last16) == "1") &&
                !db_cache && !rectangular_w8a8 &&
                !(full_warm && std::string_view(full_warm) == "1") ? 2 : r.steps;
            // Keep the full five-step kernel warmup but do not populate the
            // experimental prefix bank: measured run 0 must be a miss, run 1
            // a hit against its captured FFN tile tails.
            if (tiled_prefix_probe) {
                tc::require(setenv("TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV", "0", 1) == 0 &&
                            setenv("TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC", "0", 1) == 0,
                            "cannot isolate tiled prefix warmup");
                if (lossy_prefix_hit)
                    tc::require(setenv("TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC", "0", 1) == 0,
                                "cannot isolate lossy tiled prefix warmup");
            }
            auto warmed = session.prepare(warm, true, event, cancelled);
            if (tiled_prefix_probe) {
                tc::require(setenv("TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV", "1", 1) == 0 &&
                            setenv("TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC", "1", 1) == 0,
                            "cannot restore tiled prefix experiment after warmup");
                if (lossy_prefix_hit)
                    tc::require(setenv("TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC", "1", 1) == 0,
                                "cannot restore lossy tiled prefix after warmup");
            }
            save("warmup.json", warmed);
            tc::require(!std::filesystem::exists(r.output), "warmup exported an image");
            // A matching full-step warmup can populate the resident prefix
            // before the first measured request; a short warmup cannot.
            const bool prefix_warmed = prefix_probe && !r.prompt_enhance &&
                warm.steps == r.steps && r.steps > 2 && !tiled_prefix_probe;
            uint64_t previous_calls = warmed.hybrid ? warmed.hybrid->runtime_calls : 0;
            for (int iteration = 0; iteration < repeats; ++iteration) {
                auto name = "run-" + std::to_string(iteration);
                r.output = (directory / (name + ".png")).string();
                // Persist request-owned text/noise/latent tensors per repeat so
                // the external comparison can prove identical inputs rather
                // than relying only on request metadata.
                r.dump = dump_tensors ? (directory / (name + "-dump")).string() : "";
                auto result = session.generate(r, event, cancelled);
                if (prefix_lora_guard)
                    tc::require(result.selection.find("resident prefix KV hit") == std::string::npos &&
                                    result.selection.find("resident prefix KV miss") == std::string::npos,
                                "LoRA request bypassed the resident prefix KV safety guard");
                if (prefix_probe && r.width == 512 && r.height == 512 && r.steps > 2 &&
                    !r.prompt_enhance && r.loras.empty()) {
                    const auto marker = iteration == 0 && !prefix_warmed
                        ? "resident prefix KV miss" : "resident prefix KV hit";
                    tc::require(result.selection.find(marker) != std::string::npos,
                                "cross-request prefix KV cache did not follow the expected miss/hit lifecycle");
                    if (iteration > 0 && !lossy_prefix_hit)
                        tc::require(tc::sha256_file(r.output) ==
                                        tc::sha256_file(directory / "run-0.png"),
                                    "cross-request prefix KV changed the same-seed output");
                    if (iteration > 1 && lossy_prefix_hit)
                        tc::require(tc::sha256_file(r.output) ==
                                        tc::sha256_file(directory / "run-1.png"),
                                    "lossy prefix hit did not write a reproducible output");
                }
                tc::require(result.prompt_cache_hit,
                            "repeated resident prompt/references were not cached");
                tc::require(std::filesystem::is_regular_file(r.output), "generation did not export");
                if (r.execution == "gpu_ane") {
                    tc::require(result.hybrid.has_value() && result.request.execution == "gpu_ane",
                                "hybrid selection/report was lost");
                    const char *tiled_flag = std::getenv("TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC");
                    const int tiled_layers = tc::qwen21::tiled_prefill_layer_count(tiled_flag ? tiled_flag : "0");
                    const auto prefix_tokens = uint64_t(result.text_tokens) + result.reference_tokens;
                    const tc::qwen21::W8A8CallBudget budget{
                        .steps = r.steps,
                        .decode_layers = int(32 - r.qwen21_gpu_full_ffn_blocks.size()),
                        .decode_tiles = rectangular_w8a8 ? 2 : 1,
                        .db_cached_steps = result.db_cache_steps,
                        .db_skipped_layers = 32 - tc::qwen21::Transformer::db_front_blocks -
                            tc::qwen21::Transformer::db_back_blocks,
                        .final_reuse_layers = tc::qwen21::option_enabled(hybrid_final) ? 32 :
                            tc::qwen21::option_enabled(hybrid_last16) ? 16 : 0,
                        .penultimate_reuse_layers = tc::qwen21::option_enabled(hybrid_half) ? 16 : 0,
                        .tiled_prefill_layers = tiled_layers,
                        .prefix_tokens = prefix_tokens,
                        .total_tokens = uint64_t(result.total_tokens),
                        .tile_rows = rectangular_w8a8 ? 1024 : r.width / 16 * (r.height / 16),
                        .prefix_hit = result.selection.find("resident prefix KV hit") != std::string::npos,
                        .tiled_prefix_reuse = tiled_prefix_probe,
                        .prefix_target_only = lossy_prefix_hit,
                        .last_target_only = tc::qwen21::option_enabled(std::getenv(
                            "TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC")),
                    };
                    tc::require(result.hybrid->runtime_calls - previous_calls ==
                                    tc::qwen21::expected_w8a8_calls(budget),
                                "wrong hybrid FFN call count");
                    previous_calls = result.hybrid->runtime_calls;
                    tc::require(result.hybrid->load_seconds == warmed.hybrid->load_seconds,
                                "resident Core ML session was reloaded");
                }
                save((name + ".json").c_str(), result);
                std::cout << "{\"iteration\":" << iteration << ",\"wall_seconds\":" << result.timings.wall
                          << ",\"denoise_seconds\":" << result.timings.denoise
                          << ",\"setup_seconds\":" << result.timings.hybrid << "}" << std::endl;
            }
            if (prefix_probe && r.steps > 2) {
                auto another_seed = r;
                another_seed.seed += 1;
                another_seed.dump.clear();
                another_seed.output = (directory / "other-seed.png").string();
                auto changed_noise = session.generate(another_seed, event, cancelled);
                tc::require(changed_noise.selection.find("resident prefix KV hit") != std::string::npos &&
                                tc::sha256_file(another_seed.output) !=
                                    tc::sha256_file(directory / "run-0.png"),
                            "different seed failed to reuse the independent conditioning prefix");
                save("other-seed.json", changed_noise);
                tc::require(setenv("TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV", "0", 1) == 0,
                            "cannot disable prefix cache for other-seed GPU oracle");
                if (tiled_prefix_probe)
                    tc::require(setenv("TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC", "0", 1) == 0,
                                "cannot disable tiled prefix mode for other-seed oracle");
                if (lossy_prefix_hit)
                    tc::require(setenv("TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC", "0", 1) == 0,
                                "cannot disable lossy tiled prefix for other-seed oracle");
                another_seed.output = (directory / "other-seed-gpu-oracle.png").string();
                auto oracle = session.generate(another_seed, event, cancelled);
                if (!lossy_prefix_hit)
                    tc::require(tc::sha256_file(another_seed.output) ==
                                    tc::sha256_file(directory / "other-seed.png"),
                                "cross-seed prefix KV output differs from uncached request");
                save("other-seed-gpu-oracle.json", oracle);
                tc::require(setenv("TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV", "1", 1) == 0,
                            "cannot restore prefix cache experiment");
                if (tiled_prefix_probe)
                    tc::require(setenv("TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC", "1", 1) == 0,
                                "cannot restore tiled prefix experiment");
                if (lossy_prefix_hit)
                    tc::require(setenv("TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC", "1", 1) == 0,
                                "cannot restore lossy tiled prefix experiment");
            }
            if (test_edit_cache) {
                std::filesystem::copy_file(r.inputs[1].path, mutable_reference,
                                           std::filesystem::copy_options::overwrite_existing);
                r.output = (directory / "changed-reference.png").string();
                auto changed = session.generate(r, event, cancelled);
                tc::require(!changed.prompt_cache_hit && std::filesystem::is_regular_file(r.output),
                            "overwritten Qwen21 reference reused stale conditioning");
                if (prefix_probe)
                    tc::require(changed.selection.find("resident prefix KV miss") != std::string::npos,
                                "changed reference reused a stale prefix KV bank");
                save("changed-reference.json", changed);
                r.output = (directory / "changed-reference-repeat.png").string();
                auto repeated = session.generate(r, event, cancelled);
                tc::require(repeated.prompt_cache_hit && std::filesystem::is_regular_file(r.output),
                            "unchanged Qwen21 reference failed to repopulate conditioning cache");
                if (prefix_probe)
                    tc::require(repeated.selection.find("resident prefix KV hit") != std::string::npos,
                                "unchanged reference did not repopulate prefix KV bank");
                save("changed-reference-repeat.json", repeated);
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
            if (prefix_probe && r.loras.empty() && r.steps >= 2) {
                r.output = (directory / "after-cancellation.png").string();
                auto retried = session.generate(r, event, cancelled);
                const auto retry_oracle = test_edit_cache ? "changed-reference.png" : "run-0.png";
                tc::require(retried.selection.find("resident prefix KV miss") != std::string::npos &&
                                tc::sha256_file(r.output) == tc::sha256_file(directory / retry_oracle),
                            "cancelled cross-request prefix KV was reused or retry output changed");
                save("after-cancellation.json", retried);
            }
            if (!r.loras.empty()) {
                // Resident requests may alternate between a student and the
                // unchanged base checkpoint. Neither path may retain the
                // other's runtime adapter or silently skip rebinding it.
                auto plain = warm;
                plain.loras.clear(); plain.lora_strategy = "auto";
                const bool lora_ref512 = tc::qwen21::option_enabled(
                    std::getenv("TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC"));
                if (lora_ref512)
                    tc::require(setenv("TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC", "0", 1) == 0,
                                "cannot disable LoRA/512px diagnostic before base rebind");
                if (lora_base_ane) {
                    tc::require(setenv("TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC", "0", 1) == 0,
                                "cannot disable LoRA/base ANE before testing base GPU rebind");
                    if (gate_up_ane)
                        tc::require(setenv("TURBOCIDER_QWEN21_LORA_GATE_UP_DIAGNOSTIC", "0", 1) == 0,
                                    "cannot disable gate/up ANE before testing base GPU rebind");
                    plain.execution = "gpu"; plain.ane_manifest.clear(); plain.qwen21_w8a8 = false;
                    plain.hybrid_mlp_mode = "auto";
                }
                auto without_adapter = session.prepare(plain, false, event, cancelled);
                tc::require(without_adapter.lora_applied_projections == 0,
                            "resident base-model switch retained Viggle LoRA");
                if (lora_base_ane)
                    tc::require(setenv("TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC", "1", 1) == 0,
                                "cannot restore LoRA/base ANE before testing rebind");
                if (gate_up_ane)
                    tc::require(setenv("TURBOCIDER_QWEN21_LORA_GATE_UP_DIAGNOSTIC", "1", 1) == 0,
                                "cannot restore gate/up ANE before testing rebind");
                if (lora_ref512)
                    tc::require(setenv("TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC", "1", 1) == 0,
                                "cannot restore LoRA/512px diagnostic before testing rebind");
                auto restored = session.prepare(warm, false, event, cancelled);
                tc::require(restored.lora_applied_projections == 227,
                            "resident Viggle switch did not restore all adapter projections");
                if (prefix_probe) {
                    r.output = (directory / "after-lora-rebind.png").string();
                    r.dump.clear();
                    auto rebound = session.generate(r, event, cancelled);
                    tc::require(rebound.selection.find("resident prefix KV miss") != std::string::npos &&
                                    tc::sha256_file(r.output) == tc::sha256_file(directory / "run-0.png"),
                                "resident LoRA rebind reused stale prefix KV or changed the output");
                    save("after-lora-rebind.json", rebound);
                }
            }
            // Switching to GPU must release the Core ML bank and report GPU.
            warm.execution = "gpu"; warm.ane_manifest.clear();
            warm.hybrid_mlp_mode = "auto";
            if (lora_base_ane)
                tc::require(setenv("TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC", "0", 1) == 0,
                            "cannot disable LoRA/base ANE before GPU lifecycle switch");
            if (gate_up_ane)
                tc::require(setenv("TURBOCIDER_QWEN21_LORA_GATE_UP_DIAGNOSTIC", "0", 1) == 0,
                            "cannot disable gate/up ANE before GPU lifecycle switch");
            if (rectangular_w8a8)
                tc::require(setenv("TURBOCIDER_QWEN21_RECT_W8A8_DIAGNOSTIC", "0", 1) == 0,
                            "cannot disable rectangular tiling before GPU lifecycle switch");
            warm.qwen21_w8a8 = false; warm.qwen21_gpu_w8a16 = false;
            warm.qwen21_gpu_full_ffn_blocks.clear();
            warm.allow_approximation = warm.allow_approximation || warm.qwen21_reference_size != 1024;
            if (std::getenv("TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC"))
                tc::require(setenv("TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC", "0", 1) == 0,
                            "cannot disable tiled prefill before GPU lifecycle switch");
            if (std::getenv("TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC"))
                tc::require(setenv("TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC", "0", 1) == 0,
                            "cannot disable tiled prefix before GPU lifecycle switch");
            if (std::getenv("TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC"))
                tc::require(setenv("TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC", "0", 1) == 0,
                            "cannot disable lossy tiled prefix before GPU lifecycle switch");
            if (std::getenv("TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC"))
                tc::require(setenv("TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC", "0", 1) == 0,
                            "cannot disable last-block target-only before GPU lifecycle switch");
            if (std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC"))
                tc::require(setenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC", "0", 1) == 0,
                            "cannot disable hybrid FFN cache before GPU lifecycle switch");
            if (std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC"))
                tc::require(setenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC", "0", 1) == 0,
                            "cannot disable hybrid last-16 FFN cache before GPU lifecycle switch");
            if (std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC"))
                tc::require(setenv("TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC", "0", 1) == 0,
                            "cannot disable hybrid penultimate FFN cache before GPU lifecycle switch");
            const char *fused_qkv = std::getenv("TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC");
            const bool switch_fused_qkv = fused_qkv && std::string_view(fused_qkv) == "1";
            if (switch_fused_qkv)
                tc::require(setenv("TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC", "0", 1) == 0,
                            "cannot disable fused QKV before GPU lifecycle switch");
            const char *local_flag = std::getenv("TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION");
            const bool switch_local = local_flag && std::string_view(local_flag) == "3";
            if (switch_local)
                tc::require(setenv("TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION", "0", 1) == 0,
                            "cannot disable last-16 reference locality before GPU lifecycle switch");
            auto gpu = session.prepare(warm, false, event, cancelled);
            tc::require(!gpu.hybrid && gpu.request.execution == "gpu", "GPU switch retained hybrid route");
            if (switch_fused_qkv) {
                auto switched_request = r;
                switched_request.execution = "gpu";
                switched_request.ane_manifest.clear();
                switched_request.qwen21_w8a8 = false;
                switched_request.qwen21_gpu_w8a16 = false;
                switched_request.qwen21_gpu_full_ffn_blocks.clear();
                switched_request.output = (directory / "post-fused-qkv-switch.png").string();
                switched_request.dump.clear();
                auto switched = session.generate(switched_request, event, cancelled);
                tc::require(switched.selection.find("diagnostic Metal fused QKV") == std::string::npos &&
                                switched.request.execution == "gpu" &&
                                std::filesystem::is_regular_file(switched_request.output),
                            "Qwen21 fused QKV switch did not restore the ordinary GPU route");
                save("post-fused-qkv-switch.json", switched);
            }
            if (switch_local) {
                auto switched_request = r;
                switched_request.execution = "gpu";
                switched_request.ane_manifest.clear();
                switched_request.qwen21_w8a8 = false;
                switched_request.qwen21_gpu_w8a16 = false;
                switched_request.qwen21_gpu_full_ffn_blocks.clear();
                switched_request.output = (directory / "post-local-switch.png").string();
                switched_request.dump.clear();
                auto switched = session.generate(switched_request, event, cancelled);
                tc::require(switched.selection.find("reference-local prefill attention") == std::string::npos &&
                            std::filesystem::is_regular_file(switched_request.output) &&
                            switched.request.execution == "gpu",
                            "Qwen21 reference-local switch did not restore ordinary GPU");
                save("post-local-switch.json", switched);
            }
            session.unload();
            std::cout << "{\"lifecycle\":\"passed\",\"quality\":\"requires visual inspection\"}" << std::endl;
            return 0;
        } catch (const std::exception &error) {
            std::cerr << error.what() << std::endl;
            return 1;
        }
    }
}

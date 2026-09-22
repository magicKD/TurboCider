#include "../../native/models/qwen21/text_encoder.hpp"
#include "../../native/models/qwen21/transformer.hpp"
#include "../../native/models/qwen21/vae.hpp"
#include "../../native/models/qwen21/scheduler.hpp"
#include "../../native/models/qwen21/conditioning.hpp"
#include "../../native/models/qwen21/hybrid.hpp"
#include "../../native/media/image.hpp"
#include <mlx/random.h>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc >= 8, "usage: qwen21-generate MODEL_ROOT PROMPT OUTPUT.png WIDTH HEIGHT STEPS SEED [--raw-text] [--normalized-edit] [--ane-manifest=PATH] [REFERENCE.png ... up to 10]");
        bool normalized_edit = false, raw_text = false;
        std::string ane_manifest;
        int first_reference = 8;
        for (; first_reference < argc; ++first_reference) {
            std::string option(argv[first_reference]);
            if (option == "--raw-text") raw_text = true;
            else if (option == "--normalized-edit") normalized_edit = true;
            else if (option.starts_with("--ane-manifest=")) ane_manifest = option.substr(15);
            else { tc::require(!option.starts_with("--"), "unknown Qwen21 diagnostic option"); break; }
        }
        tc::require(argc - first_reference <= 10 && (!normalized_edit || argc > first_reference), "invalid diagnostic reference options");
        tc::configure_streams();
        std::filesystem::path root(argv[1]), output(argv[3]);
        int width = std::stoi(argv[4]), height = std::stoi(argv[5]), steps = std::stoi(argv[6]);
        tc::require(ane_manifest.empty() || (width == 512 && height == 512 && first_reference == argc),
                    "experimental Qwen21 hybrid is restricted to 512x512 text-to-image");
        auto seed = std::stoull(argv[7]);
        auto schedule = tc::qwen21::sigmas(width, height, steps);
        tc::mx::eval(schedule);
        std::atomic<bool> cancelled{false};
        auto event = [](const std::string &phase, int i, int total) {
            std::cerr << phase << ' ' << i << '/' << total << '\n';
        };
        tc::Tensor conditioning(0.f);
        std::vector<tc::Tensor> reference_pixels;
        std::vector<tc::qwen21::ReferenceLatents> reference_latents;
        std::vector<int> image_slots;
        auto total_start = tc::Clock::now();
        for (int i = first_reference; i < argc; ++i) {
            auto pixels = tc::load_rgba_image_tensor(argv[i]);
            tc::require(pixels.shape(1) % 32 == 0 && pixels.shape(2) % 32 == 0 &&
                        int64_t(pixels.shape(1)) * pixels.shape(2) <= 12845056,
                        "diagnostic reference images must have dimensions divisible by 32 and fit the vision budget");
            reference_pixels.push_back(pixels);
        }
        {
            tc::Weights weights;
            weights.load_file(root / "text_encoders/qwen3vl_8b_bf16.safetensors");
            tc::Tokenizer tokenizer(root / "processor");
            auto tokens = tokenizer.raw_bounded(tc::qwen21::reference_prompt_template(argv[2], reference_pixels.size()), tc::Tokenizer::qwen21_limit);
            std::vector<tc::qwen21::VisualReference> refs;
            if (!reference_pixels.empty()) {
                tc::qwen21::VisionEncoder vision(weights);
                for (const auto &pixels : reference_pixels) {
                    auto alpha = tc::slice_axis(pixels, 3, 3, 4);
                    auto rgb = tc::slice_axis(pixels, 3, 0, 3) * alpha + (1.f - alpha);
                    auto input = tc::qwen21::vision_input(rgb);
                    refs.push_back({vision.encode(input.patches, input.grid_height, input.grid_width, event, cancelled),
                                    input.grid_height, input.grid_width});
                }
            }
            auto assembled = tc::qwen21::assemble_prompt(tokens, weights.at("model.embed_tokens.weight"), refs);
            tc::qwen21::TextConfig config;
            config.final_norm = !raw_text && (reference_pixels.empty() || normalized_edit);
            tc::qwen21::TextEncoder encoder(weights, config);
            conditioning = assembled.retain(encoder.encode_embeddings(assembled.embeddings, assembled.positions,
                assembled.embeddings.shape(1), event, cancelled, assembled.deepstack_deltas));
            image_slots = assembled.image_slots;
            tc::mx::eval(conditioning);
        }
        tc::mx::clear_cache();
        if (!reference_pixels.empty()) {
            tc::Weights weights;
            weights.load_file(root / "vae/qwen_image_2.1_vae_bf16.safetensors");
            tc::qwen21::VAE vae(weights);
            for (size_t i = 0; i < reference_pixels.size(); ++i) {
                const auto &pixels = reference_pixels[i];
                auto encoded = vae.encode(tc::mx::transpose(pixels * 2.f - 1.f, {0, 3, 1, 2}), event, cancelled);
                encoded = tc::mx::astype(tc::mx::reshape(tc::mx::transpose(encoded, {0, 2, 3, 1}), {1, -1, 64}), tc::mx::bfloat16);
                tc::mx::eval(encoded);
                reference_latents.push_back({encoded, {pixels.shape(1) / 16, pixels.shape(2) / 16, image_slots[i]}});
            }
        }
        reference_pixels.clear();
        tc::mx::clear_cache();
        auto latents = tc::mx::astype(tc::mx::random::normal({1, width / 16 * (height / 16), 64},
                                       tc::mx::float32, tc::mx::random::key(seed)), tc::mx::bfloat16);
        auto initial = latents;
        std::vector<float> durations;
        {
            tc::Weights weights;
            weights.load_file(root / "diffusion_models/qwen_image_2.1_bf16.safetensors");
            std::unique_ptr<tc::HybridSession> ane;
            std::unique_ptr<tc::qwen21::HybridMLP> hybrid;
            if (!ane_manifest.empty()) {
                ane = std::make_unique<tc::HybridSession>(std::filesystem::absolute(ane_manifest), root, 1024,
                    event, cancelled, 1, root / "diffusion_models/qwen_image_2.1_bf16.safetensors",
                    std::vector<tc::LoRAAsset>{}, 1024, 32);
                hybrid = std::make_unique<tc::qwen21::HybridMLP>(weights, *ane);
            }
            tc::qwen21::Transformer transformer(weights);
            if (hybrid) transformer.set_decode_mlp([&](int block, const tc::Tensor &input) { return (*hybrid)(block, input); });
            for (int step = 0; step < steps; ++step) {
                auto start = tc::Clock::now();
                auto noise = transformer.forward(latents, conditioning, schedule.data<float>()[step], height / 16, width / 16,
                                                 true, nullptr, reference_latents);
                auto dt = tc::Tensor(schedule.data<float>()[step + 1] - schedule.data<float>()[step], latents.dtype());
                latents = latents + noise * dt;
                tc::mx::eval(latents);
                float elapsed = std::chrono::duration<float>(tc::Clock::now() - start).count();
                durations.push_back(elapsed);
                std::cout << "{\"step\":" << step + 1 << ",\"seconds\":" << elapsed << "}" << std::endl;
            }
            if (ane) {
                const auto metrics = ane->metrics();
                std::cout << "{\"hybrid_runtime_calls\":" << metrics.runtime_calls
                          << ",\"hybrid_prediction_seconds\":" << metrics.prediction_seconds
                          << ",\"hybrid_load_seconds\":" << metrics.load_seconds
                          << ",\"hybrid_copied_bytes\":" << metrics.copied_bytes << "}" << std::endl;
            }
        }
        tc::mx::clear_cache();
        tc::Weights vae_weights;
        vae_weights.load_file(root / "vae/qwen_image_2.1_vae_bf16.safetensors");
        tc::qwen21::VAE vae(vae_weights);
        vae_weights.clear();
        auto spatial = tc::mx::transpose(tc::mx::reshape(latents, {1, height / 16, width / 16, 64}), {0, 3, 1, 2});
        auto pixels = tc::mx::transpose(vae.decode(spatial, event, cancelled), {0, 2, 3, 1});
        tc::mx::eval(pixels);
        tc::save_rgba_png(pixels, output);
        std::unordered_map<std::string, tc::Tensor> artifacts{{"initial", initial}, {"latents", latents},
            {"text", conditioning}, {"sigmas", schedule}, {"pixels", pixels},
            {"step_seconds", tc::Tensor(durations.data(), {steps}, tc::mx::float32)}};
        for (size_t i = 0; i < reference_latents.size(); ++i)
            artifacts.emplace("reference" + std::to_string(i), reference_latents[i].latents);
        if (!image_slots.empty()) artifacts.emplace("image_slots", tc::Tensor(image_slots.data(), {int(image_slots.size())}, tc::mx::int32));
        tc::mx::save_safetensors(output.string() + ".safetensors", artifacts,
            {{"prompt", argv[2]}, {"seed", std::to_string(seed)},
             {"text_final_norm", (!raw_text && (reference_latents.empty() || normalized_edit)) ? "true" : "false"},
             {"experimental_ane_manifest", ane_manifest}});
        std::cout << "{\"total_seconds\":" << std::chrono::duration<double>(tc::Clock::now() - total_start).count() << "}" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

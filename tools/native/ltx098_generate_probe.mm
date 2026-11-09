#import <Foundation/Foundation.h>

#include "../../native/core/unigram_tokenizer.hpp"
#include "../../native/media/image.hpp"
#include "../../native/media/video.hpp"
#include "../../native/models/ltx098/pipeline.hpp"
#ifdef TURBOCIDER_LTX098_RGB_CAST_EXPERIMENT
#include "ltx098_rgb_cast_candidate.hpp"
#endif

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            tc::require(argc >= 8, "usage: ltx098-generate-probe ROOT IMAGE NEW_OUTPUT_DIR SIZE FRAMES COUNT PROMPT [--trace] [--seed N] [--steps 8|10] [--strength S]");
            const auto began = std::chrono::steady_clock::now();
            const auto root = std::filesystem::absolute(argv[1]);
            const auto image_path = std::filesystem::absolute(argv[2]);
            const auto output = std::filesystem::absolute(argv[3]);
            const int size = std::stoi(argv[4]), frames = std::stoi(argv[5]), count = std::stoi(argv[6]);
            tc::require(count == 1 || count == 2, "probe count must be 1 or 2");
            bool tracing = false;
            uint64_t seed = 42;
            int steps = 8;
            float strength = 1.f;
            bool resize_to_base = false;
            for (int i = 8; i < argc; ++i) {
                const std::string arg(argv[i]);
                if (arg == "--trace" && !tracing) tracing = true;
                else if (arg == "--seed" && i + 1 < argc) {
                    const std::string value(argv[++i]);
                    tc::require(!value.empty() && value.front() != '-', "seed must be nonnegative");
                    size_t parsed = 0;
                    seed = std::stoull(value, &parsed);
                    tc::require(parsed == value.size(), "invalid seed");
                } else if (arg == "--steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
                else if (arg == "--strength" && i + 1 < argc) strength = std::stof(argv[++i]);
                else if (arg == "--resize-to-base") resize_to_base = true;
                else tc::require(false, "invalid probe option");
            }
            tc::require(seed <= std::numeric_limits<uint64_t>::max() - uint64_t(count - 1), "seed sequence overflow");
            tc::require(!std::filesystem::exists(output), "probe output directory must not already exist");
            tc::require(std::filesystem::create_directories(output), "cannot create probe output directory");
            tc::UnigramTokenizer tokenizer(root / "tokenizer", tc::UnigramFlavor::t5_v1_1);
            tc::ltx098::Pipeline pipeline(root);
            auto image = tc::load_image_center_crop_tensor(image_path, size, size);
            tc::mx::eval(image);
            auto tokens = tokenizer.t5_prompt(argv[7]);
            NSMutableArray *runs = [NSMutableArray array];
            std::atomic<bool> cancelled{false};
            tc::Event event = [](const std::string &phase, int done, int total) {
                if (phase == "denoise" || phase == "video_vae")
                    std::cerr << phase << " " << done << "/" << total << '\n';
            };
            for (int index = 0; index < count; ++index) {
                const auto start = index == 0 ? began : std::chrono::steady_clock::now();
                tc::ltx098::GenerateOptions options;
                options.width = size; options.height = size; options.frames = frames; options.fps = 24; options.seed = seed + index;
                options.steps = steps; options.strength = strength; options.resize_to_base = resize_to_base;
                if (tracing) {
                    const auto trace = output / ("trace-" + std::to_string(index));
                    tc::require(std::filesystem::create_directory(trace), "cannot create LTX trace directory");
                    options.trace = [trace](const std::string &name, const tc::Tensor &value) {
                        tc::mx::eval(value);
                        tc::mx::save_safetensors((trace / (name + ".safetensors")).string(), {{"value", value}});
                    };
                }
                auto [pixels, metrics] = pipeline.generate(tokens, image, options, event, cancelled);
                tc::require(tc::mx::all(tc::mx::isfinite(pixels)).item<bool>(), "LTX 0.9.8 decoder produced nonfinite pixels");
#ifdef TURBOCIDER_LTX098_RGB_CAST_EXPERIMENT
                auto rgb = tc::ltx098::rgb_cast_probe::fused(pixels);
#else
                auto scaled = (tc::mx::astype(pixels, tc::mx::float32) + tc::Tensor(1.f, tc::mx::float32)) * tc::Tensor(127.5f, tc::mx::float32);
                auto rgb = tc::mx::astype(tc::mx::contiguous(tc::mx::clip(scaled, tc::Tensor(0.f), tc::Tensor(255.f))), tc::mx::uint8);
#endif
                tc::mx::eval(rgb);
                auto movie = output / (std::to_string(index) + ".mp4");
                tc::write_video_rgb24(movie, rgb.data<uint8_t>(), frames, metrics.output_width, metrics.output_height, 24);
                auto info = tc::probe_video(movie);
                tc::require(info.frames == frames && info.width == metrics.output_width && info.height == metrics.output_height && info.fps == 24,
                            "LTX 0.9.8 MP4 verification failed");
                const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                [runs addObject:@{@"index":@(index), @"seed":@(options.seed), @"wall_seconds":@(wall),
                    @"text_seconds":@(metrics.text_seconds), @"image_seconds":@(metrics.image_seconds),
                    @"transformer_load_seconds":@(metrics.transformer_load_seconds), @"denoise_seconds":@(metrics.denoise_seconds),
                    @"decoder_load_seconds":@(metrics.decoder_load_seconds), @"decode_seconds":@(metrics.decode_seconds),
                    @"text_cache_hit":@(metrics.text_cache_hit), @"text_tokens":@(metrics.text_tokens),
                    @"cross_attention_cache_hit":@(metrics.cross_attention_cache_hit),
                    @"cross_attention_prepare_seconds":@(metrics.cross_attention_prepare_seconds),
                    @"valid_text_tokens":@(metrics.valid_text_tokens), @"actual_steps":@(metrics.denoise_steps),
                    @"workflow":@(steps == 10 ? "multiscale-7+3" : "single-scale-8"),
                    @"first_frame_strength":@(strength), @"upscale_seconds":@(metrics.upscale_seconds),
                    @"stage1_width":@(metrics.stage1_width), @"stage1_height":@(metrics.stage1_height),
                    @"stage2_width":@(metrics.stage2_width), @"stage2_height":@(metrics.stage2_height),
                    @"output_width":@(metrics.output_width), @"output_height":@(metrics.output_height),
                    @"resize_to_base":@(resize_to_base), @"adain_after_upscale":@(steps == 10),
                    @"mlx_peak_bytes":@(tc::mx::get_peak_memory()), @"output":@(movie.c_str())}];
                NSDictionary *report = @{@"schema":@"ltx098-native-probe-v1", @"execution":@"gpu", @"runs":runs};
                NSData *json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];
                tc::require(json != nil && [json writeToURL:[NSURL fileURLWithPath:@((output / "report.json").c_str())] atomically:YES],
                            "cannot write native LTX probe report");
                std::cout << "run " << index << " wall_seconds=" << wall << '\n' << std::flush;
            }
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}

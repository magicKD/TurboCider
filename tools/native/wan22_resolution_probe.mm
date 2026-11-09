#import <Foundation/Foundation.h>
#include "../../native/core/unigram_tokenizer.hpp"
#include "../../native/media/image.hpp"
#include "../../native/media/video.hpp"
#include "../../native/models/wan22/pipeline.hpp"
#include "../../native/models/wan22/text.hpp"

#include <mach/mach.h>
#include <iostream>

// Isolated development probe: reproduce one saved App job at a new canvas
// size before widening the public resolution limits. This bypasses the App's
// admission policy and is not an alternative production entry point.
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            tc::require(argc >= 7 && argc <= 9, "usage: wan22-resolution-probe MODEL_ROOT JOBS_JSON OUTPUT_DIR WIDTH HEIGHT super|full [fastwan|quanhaol] [cross-kv]");
            const auto root = std::filesystem::absolute(argv[1]);
            const auto output = std::filesystem::absolute(argv[3]);
            const int width = std::stoi(argv[4]), height = std::stoi(argv[5]);
            const std::string decoder(argv[6]);
            const std::string variant = argc >= 8 ? argv[7] : "fastwan";
            bool cache_cross = false;
            for (int i = 8; i < argc; ++i) {
                if (std::string(argv[i]) == "cross-kv") cache_cross = true;
                else tc::require(false, "invalid experimental probe option");
            }
            tc::require(width >= 128 && height >= 128 && width <= 768 && height <= 768 &&
                            width % 32 == 0 && height % 32 == 0 &&
                            (decoder == "super" || decoder == "full") &&
                            (variant == "fastwan" || variant == "quanhaol") && !std::filesystem::exists(output),
                        "invalid test canvas, decoder or existing output directory");
            NSData *data = [NSData dataWithContentsOfFile:@(argv[2])];
            id jobs = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
            tc::require([jobs isKindOfClass:NSArray.class] && [jobs count] > 0, "invalid jobs JSON");
            NSDictionary *request = jobs[0][@"request"];
            tc::require([request isKindOfClass:NSDictionary.class] &&
                            [request[@"model"] isEqual:@"wan2.2-ti2v-5b-fastwan"] &&
                            [request[@"prompt"] isKindOfClass:NSString.class] &&
                            [request[@"inputs"] isKindOfClass:NSArray.class] && [request[@"inputs"] count] == 1,
                        "first job must contain a Wan2.2 image-to-video request");
            NSString *image_path = request[@"inputs"][0][@"path"];
            tc::require([image_path isKindOfClass:NSString.class] &&
                            std::filesystem::is_regular_file(image_path.UTF8String), "first-frame image missing");
            const int frames = [request[@"frames"] intValue], fps = [request[@"fps"] intValue];
            tc::require(frames >= 5 && frames <= 97 && frames % 4 == 1 && fps == 24,
                        "invalid saved video geometry");
            std::filesystem::create_directory(output);
            const auto started = tc::Clock::now();
            std::atomic<bool> cancelled{false};
            uint64_t peak_footprint = 0;
            tc::Event event = [&](const std::string &phase, int current, int total) {
                task_vm_info_data_t info{};
                mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
                if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
                    peak_footprint = std::max<uint64_t>(peak_footprint, info.phys_footprint);
                    tc::require(info.phys_footprint < (uint64_t(56) << 30),
                                "test stopped after exceeding 56 GiB process footprint");
                }
                if ((phase == "wan22_denoise" || phase == "wan22_dit_block") &&
                    (phase == "wan22_denoise" || current % 5 == 0))
                    std::cerr << phase << " " << current << "/" << total << " footprint=" << peak_footprint << '\n';
            };
            tc::UnigramTokenizer tokenizer(root / "tokenizer");
            auto tokens = tokenizer.prompt_extended([request[@"prompt"] UTF8String]);
            tc::wan22::TextEncoder text(root / "text_encoders/umt5_xxl_fp16.safetensors", root / "tokenizer");
            auto context = text.encode(tokens, event, cancelled);
            tc::mx::eval(context);
            tc::mx::clear_cache();
            auto image = tc::load_image_center_crop_tensor(image_path.UTF8String, width, height);
            image = tc::mx::astype(tc::mx::expand_dims(tc::mx::transpose(image, {0, 3, 1, 2}), 2), tc::mx::bfloat16);
            tc::mx::eval(image);
            const auto checkpoint = variant == "quanhaol" ? "diffusion_models/quanhaol_wan22_ti2v_5b_turbo_fp16.safetensors" :
                                                          "diffusion_models/fastwan2.2_ti2v_5b_fullattn.safetensors";
            tc::wan22::Pipeline pipeline(root / checkpoint,
                                         root / "vae/wan2.2_vae.safetensors");
            tc::wan22::GenerateOptions options;
            options.frames = frames;
            options.seed = [request[@"seed"] unsignedLongLongValue];
            if (variant == "quanhaol") options.distillation = tc::wan22::Distillation::quanhaol_4;
            options.cache_cross_attention = cache_cross;
            if (decoder == "super") options.tiny_decoder_checkpoint = root / "vae/taew2_2_super.safetensors";
            std::optional<tc::Tensor> first_latent;
            tc::wan22::Pipeline::Trace trace = [&](const std::string &name, const tc::Tensor &value) {
                if (name == "first_latent") {
                    tc::mx::eval(value);
                    first_latent = value;
                }
            };
            auto result = pipeline.generate(image, context, options, event, cancelled, trace);
            tc::require(result.pixels.shape() == tc::mx::Shape{1, 3, frames, height, width} &&
                            tc::mx::all(tc::mx::isfinite(result.pixels)).item<bool>() && first_latent,
                        "invalid decoded video or missing first-frame condition");
            tc::mx::save_safetensors((output / "latent.safetensors").string(), {{"latents", result.latents}});
            tc::mx::save_safetensors((output / "conditioning.safetensors").string(),
                                    {{"text", context}, {"first_latent", *first_latent}});
            auto rgb = tc::mx::squeeze(tc::mx::transpose(tc::mx::astype(result.pixels, tc::mx::float32),
                                                          {0, 2, 3, 4, 1}), 0);
            rgb = tc::mx::astype(tc::mx::round(tc::mx::clip((rgb + 1.f) * 127.5f,
                                      tc::Tensor(0.f), tc::Tensor(255.f))), tc::mx::uint8);
            rgb = tc::mx::contiguous(rgb);
            tc::mx::eval(rgb);
            const auto movie = output / "video.mp4";
            tc::write_video_rgb24(movie, rgb.data<uint8_t>(), frames, width, height, fps);
            const auto video = tc::probe_video(movie);
            tc::require(video.frames == frames && video.width == width && video.height == height && video.fps == fps,
                        "encoded video geometry mismatch");
            const double elapsed = std::chrono::duration<double>(tc::Clock::now() - started).count();
            NSDictionary *report = @{@"width":@(width), @"height":@(height), @"frames":@(frames), @"fps":@(fps),
                @"seed":@(options.seed), @"decoder":@(decoder.c_str()), @"variant":@(variant.c_str()),
                @"cross_attention_cache":@(cache_cross),
                @"valid_text_tokens":@(tokens.valid),
                @"wall_seconds":@(elapsed), @"first_frame_vae_seconds":@(result.first_frame_vae_seconds),
                @"transformer_and_denoise_seconds":@(result.transformer_and_denoise_seconds),
                @"video_vae_seconds":@(result.video_vae_seconds), @"sampled_peak_footprint_bytes":@(peak_footprint),
                @"scope":@"isolated native resolution probe; footprint is sampled between phases and blocks"};
            NSData *json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];
            tc::require([json writeToFile:@((output / "report.json").c_str()) options:NSDataWritingWithoutOverwriting error:nil],
                        "cannot write report");
            std::cout << [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String << '\n';
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}

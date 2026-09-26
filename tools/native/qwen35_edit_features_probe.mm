#import <Foundation/Foundation.h>
#include "../../native/models/qwen21/conditioning.hpp"
#include "../../native/models/qwen21/pe_generation.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            tc::require(argc >= 5 && argc <= 8,
                "usage: qwen35-edit-features-probe PE_DIR PIXELS.safetensors PROMPT OUTPUT_JSON [--image-api] [--full] [--fp32-visual]");
            bool image_api = false, full = false, fp32_visual = false;
            for (int i = 5; i < argc; ++i) {
                if (std::string(argv[i]) == "--image-api") image_api = true;
                else if (std::string(argv[i]) == "--full") full = true;
                else if (std::string(argv[i]) == "--fp32-visual") fp32_visual = true;
                else tc::require(false, "unknown PE edit feature probe option");
            }
            tc::configure_streams();
            const auto root = std::filesystem::path(argv[1]);
            std::ifstream system_file(root / "system_prompt.txt");
            tc::require(system_file.good(), "missing PE system prompt");
            std::string system((std::istreambuf_iterator<char>(system_file)), std::istreambuf_iterator<char>());
            auto [inputs, metadata] = tc::mx::load_safetensors(argv[2]);
            tc::require(inputs.size() >= 1 && inputs.size() <= 10, "invalid PE feature probe image count");
            std::atomic<bool> cancelled{false};
            tc::Weights weights;
            weights.load(root, [](const std::string &, int step, int total) { std::cerr << "load " << step << '/' << total << '\n'; }, cancelled);
            tc::Tokenizer tokenizer(root);
            tc::qwen21::VisionConfig config; config.pe_qwen35 = true; config.deepstack_layers.clear();
            config.pe_fp32 = fp32_visual;
            tc::qwen21::VisionEncoder vision(weights, config);
            std::vector<tc::qwen21::pe::ImageGrid> grids;
            std::vector<tc::Tensor> features;
            std::vector<tc::Tensor> images;
            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto name = "pixels" + std::to_string(i);
                tc::require(inputs.count(name), "PE image keys must be consecutive pixels0...pixelsN");
                images.push_back(inputs.at(name));
                if (image_api) continue;
                auto processed = tc::qwen21::pe_vision_input(inputs.at(name));
                auto patches = tc::mx::astype(processed.patches, fp32_visual ? tc::mx::float32 :
                    weights.at("model.visual.patch_embed.proj.weight").dtype());
                auto output = vision.encode(patches, processed.grid_height, processed.grid_width, {}, cancelled);
                tc::mx::eval(output.merged);
                grids.push_back({processed.grid_height, processed.grid_width});
                features.push_back(output.merged);
            }
            auto sampling = tc::qwen21::pe::SamplingProfile::for_edit();
            sampling.max_new_tokens = full ? 24000 : 2;
            tc::Event event = [](const std::string &stage, int current, int total) {
                    std::cerr << stage << ' ' << current << '/' << total << '\n';
                };
            auto generate = [&] {
                if (image_api) return tc::qwen21::pe::generate_edit_images(weights, tokenizer, system, argv[3],
                    images, sampling, 42, event, cancelled, {}, fp32_visual);
                return tc::qwen21::pe::generate_edit_features(weights, tokenizer, system, argv[3], grids,
                    features, sampling, 42, event, cancelled);
            };
            auto result = generate();
            if (!full)
                tc::require(result.generated_tokens == 2 && !result.complete() && !result.stopped_eos,
                    "shared PE edit feature API unexpectedly completed two-token probe");
            cancelled = true;
            bool cancellation_observed = false;
            try {
                generate();
            } catch (const tc::Cancelled &) { cancellation_observed = true; }
            tc::require(cancellation_observed, "shared PE edit API ignored cancellation");
            NSMutableArray *sampled_ids = [NSMutableArray new];
            for (int token : result.generated_ids) [sampled_ids addObject:@(token)];
            NSDictionary *report = @{
                @"passed": @(!full || result.complete()), @"complete": @(result.complete()), @"stopped_eos": @(result.stopped_eos),
                @"prompt_tokens": @(result.input_prompt_tokens), @"expanded_tokens": @(result.prompt_tokens),
                @"generated_tokens": @(result.generated_tokens), @"generated_ids": sampled_ids,
                @"eos_token_id": @248044, @"max_new_tokens": @(sampling.max_new_tokens),
                @"next_position": @(result.next_position),
                @"rope_delta": @(result.next_position - result.prompt_tokens),
                @"cached_tokens": @(result.cached_tokens),
                @"chunked_prefill": @(result.chunked_prefill),
                @"thinking": @(result.rewrite.thinking.c_str()),
                @"positive_prompt": @(result.rewrite.positive_prompt.c_str()),
                @"wh_ratio": @(result.rewrite.wh_ratio.c_str()),
                @"ratio_follow": @(result.rewrite.ratio_follow.c_str()),
                @"prefill_seconds": @(result.prefill_seconds), @"decode_seconds": @(result.decode_seconds),
                @"parse_ok": @(result.rewrite.parse_ok), @"raw": @(result.raw.c_str()),
                @"image_api": @(image_api),
                @"fp32_visual": @(fp32_visual),
                @"quality_accepted": @NO,
                @"scope": full ?
                    (image_api ? @"shared native generate_edit_images API full rewrite" : @"shared native generate_edit_features API full rewrite") :
                    (image_api ? @"shared native generate_edit_images API including processor/vision; two-token prefill/decode and pre-call cancellation only; no quality acceptance" :
                                 @"shared native generate_edit_features API; two-token prefill/decode and pre-call cancellation only; no quality acceptance")
            };
            NSData *json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];
            tc::require(json != nil && [json writeToFile:@(argv[4]) atomically:YES], "cannot write shared API report");
            tc::require(!full || result.complete(), "full PE-I2I rewrite did not complete; report saved");
            std::cout << "PASS shared PE-I2I API, finite visual conditioning, prefill/cache, cancellation and "
                      << (full ? "complete rewrite\n" : "truncation contract\n");
            return 0;
        } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
    }
}

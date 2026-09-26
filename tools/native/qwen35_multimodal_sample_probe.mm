#import <Foundation/Foundation.h>
#include "../../native/models/qwen21/conditioning.hpp"
#include "../../native/models/qwen21/pe_conditioning.hpp"
#include "../../native/models/qwen21/pe_generation.hpp"
#include "../../native/models/qwen21/prompt_rewrite.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            tc::require(argc >= 6 && argc <= 8,
                "usage: qwen35-multimodal-sample-probe PE_DIR PIXELS.safetensors PROMPT OUTPUT_JSON MAX_NEW_TOKENS [--chunked|--recurrent] [--vision-fp32]; default chunked/checkpoint vision dtype");
            bool chunked = true, prefill_option = false, vision_fp32 = false;
            for (int i = 6; i < argc; ++i) {
                const std::string option = argv[i];
                if (option == "--chunked" || option == "--recurrent") {
                    tc::require(!prefill_option, "duplicate/conflicting PE prefill mode");
                    prefill_option = true;
                    chunked = option == "--chunked";
                } else if (option == "--vision-fp32") {
                    tc::require(!vision_fp32, "duplicate PE vision precision option");
                    vision_fp32 = true;
                } else tc::require(false, "unknown PE sampler option: " + option);
            }
            tc::configure_streams();
            const auto root = std::filesystem::path(argv[1]);
            std::ifstream system_file(root / "system_prompt.txt");
            tc::require(system_file.good(), "missing PE system prompt");
            std::string system((std::istreambuf_iterator<char>(system_file)), std::istreambuf_iterator<char>());
            auto [inputs, metadata] = tc::mx::load_safetensors(argv[2]);
            tc::require(!inputs.empty() && inputs.size() <= 10, "PE probe requires 1...10 decoded image tensors");
            const int max_new_tokens = std::stoi(argv[5]);
            tc::require(max_new_tokens > 0 && max_new_tokens <= 24000, "invalid PE edit token budget");
            std::vector<tc::qwen21::VisionInput> processed_images;
            std::vector<tc::qwen21::pe::ImageGrid> grids;
            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto name = "pixels" + std::to_string(i);
                tc::require(inputs.count(name), "PE image keys must be consecutive pixels0...pixelsN");
                auto processed = tc::qwen21::pe_vision_input(inputs.at(name));
                grids.push_back({processed.grid_height, processed.grid_width});
                processed_images.push_back(std::move(processed));
            }
            tc::Tokenizer tokenizer(root);
            const auto tokens = tokenizer.raw_bounded(
                tc::qwen21::prompt_rewrite_chat(system, argv[3], inputs.size()), 32768 - max_new_tokens);
            auto layout = tc::qwen21::pe::image_layout(tokens, grids, 32768 - max_new_tokens);
            std::atomic<bool> cancelled{false};
            tc::Weights weights;
            weights.load(root, [](const std::string &, int step, int total) {
                std::cerr << "load " << step << '/' << total << '\n';
            }, cancelled);
            tc::qwen21::VisionConfig visual_config;
            visual_config.pe_qwen35 = true;
            visual_config.deepstack_layers.clear();
            // Experimental precision policy: only the visual tower is promoted.
            // Keep language weights and embedding insertion at checkpoint dtype.
            tc::Weights promoted_vision;
            if (vision_fp32) {
                std::vector<std::string> names;
                std::vector<tc::Tensor> tensors;
                for (const auto &name : weights.sorted_keys()) {
                    if (!name.starts_with("model.visual.")) continue;
                    names.push_back(name);
                    tensors.push_back(tc::mx::astype(weights.at(name), tc::mx::float32));
                }
                tc::require(!names.empty(), "PE checkpoint has no visual weights");
                promoted_vision.bind_arrays(names, tensors);
            }
            const auto &visual_weights = vision_fp32 ? promoted_vision : weights;
            const auto visual_dtype = visual_weights.at("model.visual.patch_embed.proj.weight").dtype();
            tc::qwen21::VisionEncoder vision(visual_weights, visual_config);
            std::vector<tc::Tensor> features;
            for (const auto &processed : processed_images) {
                auto patches = tc::mx::astype(processed.patches, visual_dtype);
                auto feature = vision.encode(patches, processed.grid_height, processed.grid_width, {}, cancelled);
                tc::mx::eval(feature.merged);
                tc::require(tc::mx::all(tc::mx::isfinite(feature.merged)).item<bool>(), "nonfinite PE visual feature");
                features.push_back(feature.merged);
            }
            const auto &table = weights.at("model.language_model.embed_tokens.weight");
            auto embeddings = tc::qwen21::pe::image_embeddings(layout, table, features);
            auto positions = layout.position_tensor();
            tc::qwen21::pe::LanguageConfig language;
            language.chunked_prefill = chunked;
            tc::qwen21::pe::LanguageModel model(weights, language);
            tc::Tensor hidden(0.f);
            const auto prefill_start = tc::Clock::now();
            for (int begin = 0; begin < int(layout.ids.size()); begin += 128) {
                const int end = std::min(begin + 128, int(layout.ids.size()));
                hidden = model.forward_embeddings(tc::slice_axis(embeddings, 1, begin, end),
                    tc::slice_axis(positions, 1, begin, end), {}, cancelled);
                std::cerr << "prefill " << end << '/' << layout.ids.size() << '\n';
            }
            const auto prefill_seconds = std::chrono::duration<double>(tc::Clock::now() - prefill_start).count();
            tc::qwen21::pe::SamplingProfile sampling = tc::qwen21::pe::SamplingProfile::for_edit();
            sampling.max_new_tokens = max_new_tokens;
            std::mt19937_64 rng(42);
            std::unordered_set<int> generated;
            std::vector<int> answer;
            constexpr int eos = 248044;
            const auto decode_start = tc::Clock::now();
            auto save_partial = [&] {
                // Diagnostic output only, not resumable model state or an
                // accepted rewrite. Atomic replacement survives interruption.
                NSMutableArray *ids = [NSMutableArray arrayWithCapacity:answer.size()];
                for (int id : answer) [ids addObject:@(id)];
                const auto text = tokenizer.decode(answer);
                NSData *bytes = [NSData dataWithBytes:text.data() length:text.size()];
                NSString *decoded = [[NSString alloc] initWithData:bytes encoding:NSUTF8StringEncoding];
                NSDictionary *partial = @{
                    @"status": @"running", @"complete": @NO, @"quality_accepted": @NO,
                    @"vision_fp32": @(vision_fp32), @"chunked_prefill": @(chunked),
                    @"generated_tokens": @(answer.size()), @"generated_ids": ids,
                    @"expanded_tokens": @(layout.ids.size()), @"max_new_tokens": @(max_new_tokens),
                    @"next_position": @(layout.next_position), @"cached_tokens": @(model.cached_tokens()),
                    @"raw": decoded ?: (id)NSNull.null,
                    @"raw_utf8_base64": [bytes base64EncodedStringWithOptions:0],
                    @"scope": @"partial diagnostic output, not resumable state or completed rewrite"
                };
                NSData *data = [NSJSONSerialization dataWithJSONObject:partial options:NSJSONWritingPrettyPrinted error:nil];
                NSString *path = [@(argv[4]) stringByAppendingString:@".partial.json"];
                tc::require(data != nil && [data writeToFile:path atomically:YES], "cannot save PE partial output");
            };
            for (int step = 0; step < max_new_tokens; ++step) {
                auto logits = tc::mx::astype(tc::linear(tc::slice_axis(hidden, 1, hidden.shape(1)-1, hidden.shape(1)),
                    weights, "lm_head"), tc::mx::float32);
                tc::mx::eval(logits);
                tc::require(logits.ndim() == 3 && logits.shape(0) == 1 && logits.shape(1) == 1,
                    "PE sampling requires last-token logits");
                std::vector<float> values(logits.data<float>(), logits.data<float>() + logits.size());
                const int id = tc::qwen21::pe::draw_token(
                    tc::qwen21::pe::token_distribution(values, generated, sampling), rng);
                answer.push_back(id); generated.insert(id);
                if (id == eos) break;
                if (step + 1 < max_new_tokens) {
                    // The first generated token occupies next_position, not
                    // expanded prompt length (image spans compress mRoPE).
                    hidden = model.forward_embeddings(
                        tc::mx::take(table, tc::Tensor(&id, {1,1}, tc::mx::int32), 0),
                        tc::mx::full({3,1}, layout.next_position + step, tc::mx::int32), {}, cancelled);
                }
                if ((step + 1) % 32 == 0) {
                    if (step + 1 < max_new_tokens) save_partial();
                    std::cerr << "decode " << step + 1 << '/' << max_new_tokens << '\n';
                }
            }
            const auto decode_seconds = std::chrono::duration<double>(tc::Clock::now() - decode_start).count();
            const auto raw = tokenizer.decode(answer);
            tc::require(model.cached_tokens() == int(layout.ids.size() + answer.size() - 1),
                "PE multimodal generation cache count mismatch");
            const auto parsed = tc::qwen21::parse_prompt_rewrite_answer("<think>\n" + raw, true);
            NSString *raw_ns = [[NSString alloc] initWithBytes:raw.data() length:raw.size() encoding:NSUTF8StringEncoding];
            tc::require(raw_ns != nil, "PE output has incomplete/invalid UTF-8");
            NSMutableArray *sampled_ids = [NSMutableArray arrayWithCapacity:answer.size()];
            for (int id : answer) [sampled_ids addObject:@(id)];
            NSDictionary *out = @{
                @"prompt_tokens": @(tokens.valid), @"expanded_tokens": @(layout.ids.size()),
                @"generated_tokens": @(answer.size()), @"stopped_eos": @(answer.size() > 0 && answer.back() == eos),
                @"max_new_tokens": @(max_new_tokens), @"seed": @42, @"quality_accepted": @NO,
                @"vision_fp32": @(vision_fp32),
                @"generated_ids": sampled_ids, @"eos_token_id": @(eos),
                @"next_position": @(layout.next_position), @"rope_delta": @(layout.rope_delta()),
                @"cached_tokens": @(model.cached_tokens()),
                @"complete": @((answer.size() > 0 && answer.back() == eos) && parsed.parse_ok),
                @"chunked_prefill": @(language.chunked_prefill), @"prefill_seconds": @(prefill_seconds),
                @"decode_seconds": @(decode_seconds), @"raw": raw_ns, @"parse_ok": @(parsed.parse_ok),
                @"positive_prompt": @(parsed.positive_prompt.c_str()), @"wh_ratio": @(parsed.wh_ratio.c_str()),
                @"ratio_follow": @(parsed.ratio_follow.c_str()), @"thinking": @(parsed.thinking.c_str()),
                @"scope": @"native PE-I2I multimodal rewrite sampling; visual parity and downstream edit not accepted"
            };
            NSData *json = [NSJSONSerialization dataWithJSONObject:out options:NSJSONWritingPrettyPrinted error:nil];
            tc::require(json != nil && [json writeToFile:@(argv[4]) atomically:YES], "cannot write PE multimodal sample report");
            std::cout << raw << '\n';
            return 0;
        } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
    }
}

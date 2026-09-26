// Experimental PE visual conditioning/prefill, not an accepted edit backend.
#include "../../native/models/qwen21/conditioning.hpp"
#include "../../native/models/qwen21/pe_conditioning.hpp"
#include "../../native/models/qwen21/pe_language.hpp"
#include "../../native/models/qwen21/prompt_rewrite.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 5 || (argc == 6 && std::string(argv[5]) == "--prefill"),
            "usage: qwen35-multimodal-probe PE_DIR PIXELS.safetensors PROMPT OUTPUT.safetensors [--prefill]");
        tc::configure_streams();
        const auto root = std::filesystem::path(argv[1]);
        std::ifstream system_file(root / "system_prompt.txt");
        tc::require(system_file.good(), "missing PE system prompt");
        std::string system((std::istreambuf_iterator<char>(system_file)), std::istreambuf_iterator<char>());
        auto [inputs, metadata] = tc::mx::load_safetensors(argv[2]);
        tc::require(!inputs.empty() && inputs.size() <= 10, "PE probe requires 1...10 decoded image tensors");
        std::atomic<bool> cancelled{false};
        tc::Weights weights;
        weights.load(root, [](const std::string &, int step, int total) {
            std::cerr << "load " << step << '/' << total << '\n';
        }, cancelled);
        tc::Tokenizer tokenizer(root);
        const auto tokens = tokenizer.raw_bounded(tc::qwen21::prompt_rewrite_chat(system, argv[3], inputs.size()), 32768);
        tc::qwen21::VisionConfig visual_config;
        visual_config.pe_qwen35 = true;
        visual_config.deepstack_layers.clear();
        tc::qwen21::VisionEncoder vision(weights, visual_config);
        std::vector<tc::qwen21::pe::ImageGrid> grids;
        std::vector<tc::Tensor> features;
        std::unordered_map<std::string, tc::Tensor> outputs;
        std::vector<int> grid_values;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto name = "pixels" + std::to_string(i);
            tc::require(inputs.count(name), "PE image keys must be consecutive pixels0...pixelsN");
            auto processed = tc::qwen21::pe_vision_input(inputs.at(name));
            auto patches = tc::mx::astype(processed.patches, weights.at("model.visual.patch_embed.proj.weight").dtype());
            auto feature = vision.encode(patches, processed.grid_height, processed.grid_width, {}, cancelled);
            tc::mx::eval(feature.merged);
            tc::require(tc::mx::all(tc::mx::isfinite(feature.merged)).item<bool>(), "nonfinite PE vision feature");
            tc::require(feature.deepstack.empty(), "PE vision must not emit DeepStack");
            grids.push_back({processed.grid_height, processed.grid_width});
            grid_values.insert(grid_values.end(), {1,processed.grid_height,processed.grid_width});
            features.push_back(feature.merged);
            outputs.emplace("patches"+std::to_string(i), processed.patches);
            outputs.emplace("features"+std::to_string(i), feature.merged);
            std::cerr << "vision " << i+1 << '/' << inputs.size() << '\n';
        }
        auto layout = tc::qwen21::pe::image_layout(tokens, grids);
        const auto &table = weights.at("model.language_model.embed_tokens.weight");
        auto embeddings = tc::qwen21::pe::image_embeddings(layout, table, features);
        auto positions = layout.position_tensor();
        outputs.emplace("embeddings", embeddings);
        outputs.emplace("positions", positions);
        outputs.emplace("ids", tc::Tensor(layout.ids.data(), {1,int(layout.ids.size())}, tc::mx::int32));
        outputs.emplace("token_types", tc::Tensor(layout.token_types.data(), {1,int(layout.ids.size())}, tc::mx::int32));
        outputs.emplace("grids", tc::Tensor(grid_values.data(), {int(grids.size()),3}, tc::mx::int32));
        outputs.emplace("rope_delta", tc::Tensor(layout.rope_delta(), tc::mx::int32));
        if (argc == 6) {
            tc::qwen21::pe::LanguageModel model(weights);
            tc::Tensor hidden(0.f);
            const int total = int(layout.ids.size());
            for (int begin = 0; begin < total; begin += 128) {
                const int end = std::min(begin+128, total);
                hidden = model.forward_embeddings(tc::slice_axis(embeddings,1,begin,end),
                    tc::slice_axis(positions,1,begin,end), {}, cancelled);
                std::cerr << "prefill " << end << '/' << total << '\n';
            }
            auto last = tc::slice_axis(hidden,1,hidden.shape(1)-1,hidden.shape(1));
            auto logits = tc::linear(last,weights,"lm_head");
            tc::mx::eval(logits);
            tc::require(tc::mx::all(tc::mx::isfinite(logits)).item<bool>(), "nonfinite PE multimodal logits");
            outputs.emplace("last_hidden", last);
            outputs.emplace("logits", logits);
            const int token = tc::mx::argmax(tc::mx::reshape(logits,{-1})).item<int>();
            auto next_embedding = tc::mx::take(table,tc::Tensor(&token,{1,1},tc::mx::int32),0);
            auto next_positions = tc::mx::full({3,1}, layout.next_position, tc::mx::int32);
            auto decoded = model.forward_embeddings(next_embedding,next_positions,{},cancelled);
            auto next_logits = tc::linear(decoded,weights,"lm_head");
            tc::mx::eval(next_logits);
            tc::require(tc::mx::all(tc::mx::isfinite(next_logits)).item<bool>(), "nonfinite PE cached decode");
            outputs.emplace("decode_token",tc::Tensor(token,tc::mx::int32));
            outputs.emplace("decode_logits",next_logits);
            tc::require(model.cached_tokens() == total+1, "PE multimodal cache count mismatch");
        }
        tc::mx::save_safetensors(argv[4], outputs, {
            {"scope", "experimental PE image conditioning; BF16 visual parity not accepted"},
            {"prefill", argc == 6 ? "true" : "false"},
            {"next_position",std::to_string(layout.next_position)}});
        std::cout << "PASS finite native PE multimodal conditioning"
                  << (argc == 6 ? ", prefill and one cached decode (not rewrite/quality acceptance)\n" : " (not rewrite/quality acceptance)\n");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}

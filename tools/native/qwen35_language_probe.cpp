#include "../../native/models/qwen21/pe_language.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4 || (argc == 5 && std::string(argv[4]) == "gpu"),
                    "usage: qwen35-language-probe weights inputs outputs [gpu]; default CPU");
        if (argc == 5) tc::configure_streams();
        else tc::mx::set_default_device(tc::mx::Device::cpu);
        auto [inputs, metadata] = tc::mx::load_safetensors(argv[2]);
        tc::qwen21::pe::LanguageConfig c;
        c.chunked_prefill = !metadata.count("chunked_prefill") || metadata.at("chunked_prefill") == "true";
        auto integer = [&](const char *key, int &value) { if (metadata.count(key)) value = std::stoi(metadata.at(key)); };
        integer("hidden", c.hidden); integer("layers", c.layers);
        integer("heads", c.heads); integer("kv_heads", c.kv_heads);
        integer("head_dim", c.head_dim); integer("rotary_dim", c.rotary_dim);
        integer("key_heads", c.linear_key_heads); integer("value_heads", c.linear_value_heads);
        integer("key_dim", c.linear_key_dim); integer("value_dim", c.linear_value_dim); integer("conv", c.conv_kernel);
        for (int axis = 0; axis < 3; ++axis) integer(("section" + std::to_string(axis)).c_str(), c.mrope_sections[axis]);
        if (metadata.count("full_attention")) {
            for (char flag : metadata.at("full_attention")) {
                tc::require(flag == '0' || flag == '1', "invalid layer type flag");
                c.full_attention.push_back(flag == '1');
            }
        }
        tc::Weights weights;
        std::atomic<bool> cancelled{false};
        if (std::filesystem::is_directory(argv[1])) weights.load(argv[1], [](auto &, int, int) {}, cancelled);
        else weights.load_file(argv[1]);
        tc::qwen21::pe::LanguageModel model(weights, c);
        auto ids = inputs.at("ids");
        const int count = ids.shape(1);
        auto all = model.forward_ids(ids, {}, cancelled);
        tc::require(model.cached_tokens() == count, "PE cache token count mismatch");
        std::unordered_map<std::string, tc::Tensor> outputs{{"all", all}};
        if (weights.has("lm_head.weight")) outputs.emplace("logits", tc::linear(all, weights, "lm_head"));
        if (count > 1) {
            model.reset();
            auto prefill = model.forward_ids(tc::slice_axis(ids, 1, 0, count - 1), {}, cancelled);
            auto decode = model.forward_ids(tc::slice_axis(ids, 1, count - 1, count), {}, cancelled);
            outputs.emplace("split", tc::mx::concatenate({prefill, decode}, 1));
            if (weights.has("lm_head.weight")) outputs.emplace("decode_logits", tc::linear(decode, weights, "lm_head"));
        }
        model.reset();
        outputs.emplace("reset", model.forward_ids(ids, {}, cancelled));
        if (inputs.count("positions")) {
            model.reset();
            const auto &embedding = weights.at("model.language_model.embed_tokens.weight");
            outputs.emplace("mrope", model.forward_embeddings(tc::mx::take(embedding, ids, 0), inputs.at("positions"), {}, cancelled));
        }
        tc::mx::save_safetensors(argv[3], outputs);
        cancelled = true;
        bool rejected = false;
        try { (void)model.forward_ids(ids, {}, cancelled); } catch (const tc::Cancelled &) { rejected = true; }
        tc::require(rejected && model.cached_tokens() == 0, "PE language ignored cancellation/reset");
        cancelled = false; rejected = false;
        try {
            (void)model.forward_ids(ids, [&](const std::string &, int, int) { cancelled = true; }, cancelled);
        } catch (const tc::Cancelled &) { rejected = true; }
        tc::require(rejected && model.cached_tokens() == 0, "PE partial-layer state survived cancellation");
        std::cout << "PASS native Qwen35 language forward/cache/reset/cancellation\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

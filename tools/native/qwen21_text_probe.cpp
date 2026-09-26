#include "../../native/models/qwen21/text_encoder.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "--prompt") {
            tc::configure_streams();
            tc::Weights weights;
            weights.load_file(argv[2]);
            tc::Tokenizer tokenizer(argv[3]);
            auto tokens = tokenizer.raw(tc::qwen21::TextEncoder::prompt_template(argv[4]));
            int drop = int(tokenizer.raw(tc::qwen21::TextEncoder::system_prefix()).ids.size());
            tc::require(tokens.valid > drop && tokens.valid <= 2048, "Qwen21 prompt length outside supported range");
            tc::qwen21::TextEncoder encoder(weights);
            std::atomic<bool> cancelled{false};
            auto start = tc::Clock::now();
            auto hidden = encoder.encode(tokens, {}, cancelled);
            auto conditioning = tc::slice_axis(hidden, 1, drop, tokens.valid);
            tc::mx::eval(conditioning);
            double elapsed = std::chrono::duration<double>(tc::Clock::now() - start).count();
            tc::mx::save_safetensors(argv[5], {{"text", conditioning}, {"hidden", hidden},
                {"ids", tc::Tensor(tokens.ids.data(), {1, int(tokens.ids.size())}, tc::mx::int32)}},
                {{"system_prefix_tokens", std::to_string(drop)}, {"encode_seconds", std::to_string(elapsed)}});
            std::cout << "{\"tokens\":" << tokens.valid << ",\"drop\":" << drop << ",\"seconds\":" << elapsed << "}\n";
            return 0;
        }
        tc::require(argc == 4, "usage: qwen21-text-probe weights inputs outputs");
        tc::configure_streams();
        tc::Weights weights;
        weights.load_file(argv[1]);
        auto [inputs, metadata] = tc::mx::load_safetensors(argv[2]);
        tc::qwen21::TextConfig config;
        config.layers = std::stoi(metadata.at("layers"));
        config.heads = std::stoi(metadata.at("heads"));
        config.kv_heads = std::stoi(metadata.at("kv_heads"));
        config.head_dim = std::stoi(metadata.at("head_dim"));
        for (int i = 0; i < 3; ++i) config.mrope_sections[i] = std::stoi(metadata.at("section" + std::to_string(i)));
        auto ids = tc::mx::astype(inputs.at("ids"), tc::mx::int32);
        tc::mx::eval(ids);
        tc::Tokens tokens;
        tokens.ids.assign(ids.data<int>(), ids.data<int>() + ids.size());
        tokens.valid = std::stoi(metadata.at("valid"));
        tc::qwen21::TextEncoder encoder(weights, config);
        std::atomic<bool> cancelled{false};
        auto output = encoder.encode(tokens, {}, cancelled);
        tc::mx::eval(output);
        const auto &embedding = weights.at(weights.has("model.language_model.embed_tokens.weight")
            ? "model.language_model.embed_tokens.weight" : "model.embed_tokens.weight");
        tc::require(output.dtype() == embedding.dtype(), "text activation dtype promoted");
        auto embeddings = tc::mx::take(embedding, ids, 0);
        auto visual_positions = encoder.encode_embeddings(embeddings, inputs.at("positions"), tokens.valid, {}, cancelled);
        std::unordered_map<std::string, tc::Tensor> outputs{{"text", output}, {"positions", visual_positions}};
        if (inputs.count("deep0")) {
            std::vector<tc::Tensor> deltas{inputs.at("deep0"), inputs.at("deep1")};
            outputs.emplace("deepstack", encoder.encode_embeddings(embeddings, inputs.at("positions"), tokens.valid, {}, cancelled, deltas));
            config.final_norm = false;
            tc::qwen21::TextEncoder raw_encoder(weights, config);
            outputs.emplace("raw_deepstack", raw_encoder.encode_embeddings(embeddings, inputs.at("positions"), tokens.valid, {}, cancelled, deltas));
        }
        tc::mx::save_safetensors(argv[3], outputs);
        cancelled = true;
        bool rejected = false;
        try { encoder.encode(tokens, {}, cancelled); } catch (const tc::Cancelled &) { rejected = true; }
        tc::require(rejected, "text cancellation was ignored");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

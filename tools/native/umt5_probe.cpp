// Development-only UMT5 stage probe, linked against the native encoder.
#include "../../native/components/text/umt5.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        const bool trace = argc == 5 && std::string(argv[4]) == "--trace-first-block";
        tc::require(argc == 4 || trace, "usage: umt5-probe WEIGHTS INPUT.safetensors OUTPUT.safetensors [--trace-first-block]");
        auto inputs = tc::mx::load_safetensors(argv[2]).first;
        auto ids = tc::mx::astype(inputs.at("ids"), tc::mx::int32);
        tc::mx::eval(ids);
        tc::require(ids.ndim() == 1 && ids.size() <= 512, "invalid probe token ids");
        auto valid = inputs.at("valid");
        tc::mx::eval(valid);
        tc::Tokens tokens{{ids.data<int>(), ids.data<int>() + ids.size()}, valid.item<int>()};
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::Weights weights;
        weights.load(std::filesystem::absolute(argv[1]), event, cancelled);
        tc::components::UMT5Encoder encoder(weights);
        std::unordered_map<std::string, tc::Tensor> outputs;
        auto save = [&](const std::string &name, const tc::Tensor &x) {
            tc::mx::eval(x);
            outputs.emplace(name, x);
        };
        auto x = encoder.embed(tokens);
        auto buckets = encoder.relative_buckets(int(tokens.ids.size()));
        save("embedding", x);
        save("buckets", buckets);
        for (int layer = 0; layer < 24; ++layer) {
            tc::components::UMT5Encoder::Trace inspect;
            if (trace && layer == 0) inspect = [&](const std::string &name, const tc::Tensor &value) {
                save("first_" + name, value);
            };
            x = encoder.block(x, buckets, tokens.valid, layer, inspect);
            save("block_" + std::to_string(layer), x);
        }
        save("output", encoder.finish(x, tokens.valid));
        tc::mx::save_safetensors(argv[3], outputs);
        std::cout << "Wrote " << outputs.size() << " UMT5 stages\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

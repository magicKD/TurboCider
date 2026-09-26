#include "../../native/models/qwen21/conditioning.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3, "usage: qwen21-conditioning-probe inputs outputs");
        tc::configure_streams();
        auto [inputs, metadata] = tc::mx::load_safetensors(argv[1]);
        if (metadata.count("reference_resolution")) {
            auto output = tc::qwen21::resize_reference(inputs.at("pixels"), std::stoi(metadata.at("reference_resolution")));
            tc::mx::save_safetensors(argv[2], {{"pixels", output}});
            return 0;
        }
        auto pixels = tc::qwen21::vision_input(inputs.at("pixels"),
            std::stoi(metadata.at("min_pixels")), std::stoi(metadata.at("max_pixels")));
        auto ids = tc::mx::astype(inputs.at("ids"), tc::mx::int32);
        tc::mx::eval(ids);
        tc::Tokens tokens;
        tokens.ids.assign(ids.data<int>(), ids.data<int>() + ids.size());
        tokens.valid = int(ids.size());
        std::vector<tc::qwen21::VisualReference> refs;
        int count = std::stoi(metadata.at("references"));
        for (int i = 0; i < count; ++i) {
            auto key = "ref" + std::to_string(i);
            refs.push_back({{inputs.at(key), {inputs.at(key + "_deep0"), inputs.at(key + "_deep1")}},
                std::stoi(metadata.at(key + "_h")), std::stoi(metadata.at(key + "_w"))});
        }
        auto prompt = tc::qwen21::assemble_prompt(tokens, inputs.at("table"), refs);
        std::unordered_map<std::string, tc::Tensor> outputs{
            {"patches", pixels.patches}, {"embeddings", prompt.embeddings}, {"positions", prompt.positions},
            {"retained", prompt.retain(prompt.embeddings)}};
        if (!prompt.image_slots.empty())
            outputs.emplace("slots", tc::Tensor(prompt.image_slots.data(), {int(prompt.image_slots.size())}, tc::mx::int32));
        for (size_t i = 0; i < prompt.deepstack_deltas.size(); ++i)
            outputs.emplace("deep" + std::to_string(i), prompt.deepstack_deltas[i]);
        tc::mx::save_safetensors(argv[2], outputs,
            {{"grid_h", std::to_string(pixels.grid_height)}, {"grid_w", std::to_string(pixels.grid_width)}});
        auto rejects = [](auto fn) {
            bool rejected = false;
            try { fn(); } catch (const std::exception &) { rejected = true; }
            tc::require(rejected, "invalid conditioning input was accepted");
        };
        rejects([&] { tc::qwen21::assemble_prompt(tokens, inputs.at("table"), refs, 1); });
        rejects([&] { tc::qwen21::reference_prompt_template("test", 11); });
        if (!refs.empty()) {
            auto bad = refs;
            bad[0].grid_height += 1;
            rejects([&] { tc::qwen21::assemble_prompt(tokens, inputs.at("table"), bad); });
            bad.pop_back();
            rejects([&] { tc::qwen21::assemble_prompt(tokens, inputs.at("table"), bad); });
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

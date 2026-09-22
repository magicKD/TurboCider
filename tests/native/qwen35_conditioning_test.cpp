#include "../../native/models/qwen21/pe_conditioning.hpp"
#include "../../native/models/qwen21/conditioning.hpp"
#include <cassert>
#include <iostream>

int main(int argc, char **argv) {
    using namespace tc;
    using namespace tc::qwen21::pe;
    mx::set_default_device(mx::Device::cpu);
    if (argc == 4 && std::string(argv[1]) == "--vision") {
        auto [values, metadata] = mx::load_safetensors(argv[2]);
        auto input = values.at("pixels");
        auto result = qwen21::pe_vision_input(input);
        mx::eval(result.patches);
        mx::save_safetensors(argv[3], {{"patches", result.patches}},
            {{"grid_height", std::to_string(result.grid_height)},
             {"grid_width", std::to_string(result.grid_width)}});
        std::cout << "PASS PE processor shape " << result.grid_height << 'x' << result.grid_width << "\n";
        return 0;
    }
    auto rejects = [](auto fn) {
        bool rejected = false;
        try { fn(); } catch (const std::exception &) { rejected = true; }
        assert(rejected);
    };
    rejects([] { qwen21::pe_vision_input(mx::zeros({1,32,32,3})); });
    rejects([] { qwen21::pe_vision_input(mx::zeros({32,32,3}, mx::uint8)); });
    rejects([] { qwen21::pe_vision_input(mx::zeros({2,32,32,3}, mx::uint8)); });
    rejects([] { qwen21::pe_vision_input(mx::zeros({1,0,32,3}, mx::uint8)); });
    rejects([] { qwen21::pe_vision_input(mx::zeros({1,32,32,2}, mx::uint8)); });
    rejects([] { qwen21::pe_vision_input(mx::zeros({1,1,201,3}, mx::uint8)); });
    rejects([] { qwen21::pe_vision_input(mx::zeros({1,32,32,3}, mx::uint8), 100); });
    rejects([] { qwen21::pe_vision_input(mx::zeros({1,32,32,3}, mx::uint8), 65536, 1024); });
    rejects([] { qwen21::pe_vision_input(mx::zeros({1,32,32,3}, mx::uint8), 65536, 16777217); });
    const uint8_t hidden_rgb[] = {255,128,0,0};
    auto solid = qwen21::pe_vision_input(Tensor(hidden_rgb,{1,1,1,4},mx::uint8));
    mx::eval(solid.patches);
    assert(solid.grid_height == 16 && solid.grid_width == 16);
    for (int channel = 0; channel < 3; ++channel)
        assert(std::abs(solid.patches.data<float>()[channel*512] - (float(hidden_rgb[channel])/255.f*2.f-1.f)) < 1e-6f);
    std::cout << "PASS PE processor input validation, singleton sizing and hidden RGB preservation\n";
    std::unordered_map<std::string, Tensor> outputs;
    int cases = 0;
    for (int n : {0, 1, 2, 10}) for (int variant : {0, 1, 2}) {
        Tokens tokens{{10, 11, 12}, 0};
        std::vector<ImageGrid> grids;
        std::vector<int> grid_values;
        for (int i = 0; i < n; ++i) {
            const int h = 2 * (1 + (i + variant) % 4), w = 2 * (1 + (2 * i + variant) % 5);
            grids.push_back({h, w});
            grid_values.insert(grid_values.end(), {1, h, w});
            tokens.ids.insert(tokens.ids.end(), {248053, 248056, 248054});
        }
        tokens.ids.insert(tokens.ids.end(), {13, 14, 15});
        tokens.valid = int(tokens.ids.size());
        auto layout = image_layout(tokens, grids);
        assert(layout.images.size() == grids.size());
        assert(layout.ids.front() == 10 && layout.ids.back() == 15);
        auto table = mx::reshape(mx::arange(248058 * 2, mx::float32), {248058, 2});
        std::vector<Tensor> features;
        for (size_t i = 0; i < layout.images.size(); ++i)
            features.push_back(mx::full({layout.images[i].count, 2}, -float(i + 1), mx::float32));
        auto embeddings = image_embeddings(layout, table, features);
        mx::eval(embeddings);
        const float *values = embeddings.data<float>();
        for (size_t t = 0; t < layout.ids.size(); ++t) {
            if (layout.token_types[t] == 0) {
                assert(values[t*2] == float(layout.ids[t]*2));
                assert(values[t*2+1] == float(layout.ids[t]*2+1));
            }
        }
        for (size_t i = 0; i < layout.images.size(); ++i) {
            const auto span = layout.images[i];
            for (int t = span.begin; t < span.begin + span.count; ++t)
                assert(values[t*2] == -float(i+1) && values[t*2+1] == -float(i+1));
        }
        if (!features.empty()) {
            auto wrong = features;
            wrong[0] = mx::zeros({1, 1});
            rejects([&] { image_embeddings(layout, table, wrong); });
        }
        const auto p = "case" + std::to_string(cases++) + ".";
        outputs.emplace(p+"ids", Tensor(layout.ids.data(), {1, int(layout.ids.size())}, mx::int32));
        outputs.emplace(p+"types", Tensor(layout.token_types.data(), {1, int(layout.ids.size())}, mx::int32));
        outputs.emplace(p+"positions", layout.position_tensor());
        outputs.emplace(p+"delta", Tensor(layout.rope_delta(), mx::int32));
        if (n) outputs.emplace(p+"grids", Tensor(grid_values.data(), {n, 3}, mx::int32));
        rejects([&] { image_layout(tokens, grids, int(layout.ids.size()) - 1); });
    }
    rejects([] { image_layout(Tokens{{}, 0}, {}); });
    rejects([] { image_layout(Tokens{{1}, 2}, {}); });
    rejects([] { image_layout(Tokens{{248056}, 1}, {{2,2}}); });
    rejects([] { image_layout(Tokens{{248057}, 1}, {}); });
    rejects([] { image_layout(Tokens{{248053,248056,248054}, 3}, {{3,2}}); });
    rejects([] { image_layout(Tokens{{248053,248056,248054}, 3}, {}); });
    rejects([] { image_layout(Tokens{{248053,248056,248054}, 3}, {{2147483646,2147483646}}); });
    rejects([] { image_layout(Tokens{{248053,1,248054}, 3}, {}); });
    rejects([] { image_layout(Tokens{{248054}, 1}, {}); });
    rejects([] { image_layout(Tokens{{1}, 1}, std::vector<ImageGrid>(11, {2,2})); });
    if (argc == 2) mx::save_safetensors(argv[1], outputs);
    std::cout << "PASS PE image layout/embedding replacement: " << cases
              << " cases, 0/1/2/10 images, token budgets and malformed inputs\n";
}

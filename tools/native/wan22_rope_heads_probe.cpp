#include "../../native/models/wan22/fused_rope.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 5, "usage: wan22-rope-heads-probe INPUT OUTPUT HEADS baseline|fused");
        tc::require(!std::filesystem::exists(argv[2]), "RoPE probe output must be new");
        auto input = tc::mx::load_safetensors(argv[1]).first;
        auto q = input.at("q"), k = input.at("k"), v = input.at("v");
        auto cosine = input.at("cosine"), sine = input.at("sine");
        const int heads = std::stoi(argv[3]);
        tc::require(q.ndim() == 3 && heads > 0 && q.shape(2) % heads == 0, "invalid RoPE probe geometry");
        const int rows = q.shape(1), dim = q.shape(2) / heads;
        const std::string mode(argv[4]);
        tc::require(dim % 2 == 0 &&
                        (mode == "baseline" || mode == "fused"), "invalid RoPE probe mode or geometry");
        std::vector<tc::Tensor> result;
        if (mode == "fused") {
            result = tc::wan22::fused_rope_heads(q, k, v, cosine, sine, heads);
        } else {
            auto rotate = [&](const tc::Tensor &value) {
                auto x = tc::mx::reshape(tc::mx::astype(value, tc::mx::float32), {1, rows, heads, dim});
                auto halves = tc::mx::split(tc::mx::reshape(x, {1, rows, heads, dim / 2, 2}), 2, -1);
                auto rotated = tc::mx::reshape(tc::mx::concatenate({-halves[1], halves[0]}, -1), {1, rows, heads, dim});
                auto c = tc::mx::reshape(cosine, {1, rows, 1, dim});
                auto s = tc::mx::reshape(sine, {1, rows, 1, dim});
                return tc::mx::transpose(tc::mx::astype(x * c + rotated * s, value.dtype()), {0, 2, 1, 3});
            };
            auto vh = tc::mx::transpose(tc::mx::reshape(v, {1, rows, heads, dim}), {0, 2, 1, 3});
            result = {rotate(q), rotate(k), vh};
        }
        tc::mx::eval(result);
        tc::mx::save_safetensors(argv[2], {{"q", result[0]}, {"k", result[1]}, {"v", result[2]}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#include "../../native/models/h3_mlx/prompt_cache.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4,
                    "usage: h3-mlx-prompt-cache-probe COMPONENT TOKENIZER CACHE");
        const auto component = std::filesystem::absolute(argv[1]);
        const auto tokenizer = std::filesystem::absolute(argv[2]);
        const auto cache = std::filesystem::absolute(argv[3]);
        auto identity = tc::h3_mlx::prompt_cache_identity(component, tokenizer, "cache probe");
        const float values[] = {1.f, -2.f, 3.5f, 4.f};
        tc::h3_mlx::ConditioningResult expected{
            tc::Tensor(const_cast<float *>(values), {2, 2}, tc::mx::float32),
            {1, 1}, 2};
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::h3_mlx::save_prompt_cache(cache, identity, expected, event, cancelled);
        auto actual = tc::h3_mlx::load_prompt_cache(cache, identity, event, cancelled);
        tc::require(actual.has_value(), "prompt cache did not round-trip");
        tc::mx::eval(actual->hidden_states);
        tc::require(actual->hidden_states.shape() == tc::mx::Shape({2, 2}) &&
                        actual->hidden_states.dtype() == tc::mx::float32 &&
                        actual->token_tags == expected.token_tags,
                    "prompt cache round-trip metadata mismatch");
        std::cout << "PASS: H3 prompt cache atomic round-trip\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

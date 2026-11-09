#include "../../native/models/wan22/text.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 5, "usage: wan22-text-probe CHECKPOINT TOKENIZER INPUT OUTPUT");
        auto input = tc::mx::load_safetensors(argv[3]).first;
        auto ids = tc::mx::astype(input.at("ids"), tc::mx::int32);
        auto valid = tc::mx::astype(input.at("valid"), tc::mx::int32);
        tc::mx::eval(ids, valid);
        tc::require(ids.ndim() == 1 && ids.size() >= 512 && ids.size() <= 1024,
                    "invalid Wan2.2 UMT5 fixture IDs");
        tc::Tokens tokens{{ids.data<int>(), ids.data<int>() + ids.size()}, valid.item<int>()};
        tc::wan22::TextEncoder encoder(std::filesystem::absolute(argv[1]), std::filesystem::absolute(argv[2]));
        tc::Event event = [](const std::string &, int, int) {};
        std::atomic<bool> cancelled{false};
        auto output = encoder.encode(tokens, event, cancelled);
        tc::mx::save_safetensors(argv[4], {{"output", output}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

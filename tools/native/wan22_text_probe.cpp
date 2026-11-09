#include "../../native/models/wan22/text.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 5 || argc == 6,
                    "usage: wan22-text-probe CHECKPOINT TOKENIZER INPUT OUTPUT [--align64|--align128|--align256]");
        int alignment = 0;
        if (argc == 6) {
            const std::string flag(argv[5]);
            if (flag == "--align64") alignment = 64;
            else if (flag == "--align128") alignment = 128;
            else if (flag == "--align256") alignment = 256;
            else tc::require(false, "invalid experimental text alignment");
        }
        tc::require(!std::filesystem::exists(argv[4]), "text probe output must be new");
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
        const auto started = tc::Clock::now();
        auto output = encoder.encode(tokens, event, cancelled, alignment);
        std::cerr << "text_encode_seconds=" << std::chrono::duration<double>(tc::Clock::now() - started).count() << '\n';
        tc::mx::save_safetensors(argv[4], {{"output", output}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#include "../../native/components/vae/taehv.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4, "usage: taehv-probe WEIGHTS.safetensors INPUT.safetensors OUTPUT.safetensors");
        auto inputs = tc::mx::load_safetensors(argv[2]).first;
        auto input = inputs.at("latent");
        tc::require(input.ndim() == 5, "probe expects NTCHW");
        tc::components::TAEHVDecoder decoder(std::filesystem::absolute(argv[1]), input.shape(2));
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        auto result = decoder.decode_ntchw(input, event, cancelled);
        tc::mx::save_safetensors(argv[3], {{"output", result}});
        // Same decoder, same input: previous requests must leave no memory.
        auto repeat = decoder.decode_ntchw(input, event, cancelled);
        tc::mx::eval(repeat);
        tc::require(tc::mx::array_equal(result, repeat).item<bool>(), "TAEHV state leaked across requests");
        cancelled = true;
        bool rejected = false;
        try { decoder.decode_ntchw(input, event, cancelled); }
        catch (const tc::Cancelled &) { rejected = true; }
        tc::require(rejected, "TAEHV ignored pre-cancellation");
        cancelled = false;
        tc::Event cancel_mid_decode = [&](const std::string &, int index, int) {
            if (index >= 9) cancelled = true;
        };
        rejected = false;
        try { decoder.decode_ntchw(input, cancel_mid_decode, cancelled); }
        catch (const tc::Cancelled &) { rejected = true; }
        tc::require(rejected, "TAEHV ignored mid-decode cancellation");
        cancelled = false;
        auto recovered = decoder.decode_ntchw(input, event, cancelled);
        tc::require(tc::mx::array_equal(result, recovered).item<bool>(), "TAEHV cancellation corrupted later decode");
        std::cout << "TAEHV decode, repeat and cancellation passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

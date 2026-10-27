#include "../../native/components/vae/tiny.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 6, "usage: tiny-vae-probe MODEL WEIGHTS INPUT OUTPUT encode|decode");
        auto input = tc::mx::load_safetensors(argv[3]).first.at("input");
        bool encode = std::string(argv[5]) == "encode";
        tc::components::TinyAutoencoder model(argv[2], argv[1], encode, !encode);
        std::atomic<bool> cancelled{false};
        auto run = [&](const tc::Event &event) {
            return encode ? model.encode(input, event, cancelled)
                          : model.decode(input, event, cancelled);
        };
        auto output = run({});
        tc::mx::save_safetensors(argv[4], {{"output", output}});
        tc::require(tc::mx::array_equal(output, run({})).item<bool>(), "tiny VAE repeat changed output");

        cancelled = true;
        bool rejected = false;
        try {
            run({});
        } catch (const tc::Cancelled &) {
            rejected = true;
        }
        tc::require(rejected, "tiny VAE ignored pre-cancellation");

        cancelled = false;
        rejected = false;
        tc::Event cancel = [&](const std::string &, int, int) { cancelled = true; };
        try {
            run(cancel);
        } catch (const tc::Cancelled &) {
            rejected = true;
        }
        tc::require(rejected, "tiny VAE ignored mid-cancellation");
        cancelled = false;
        tc::require(tc::mx::array_equal(output, run({})).item<bool>(),
                    "tiny VAE cancellation corrupted state");
        std::cout << "tiny VAE repeat/pre/mid-cancel/recovery passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

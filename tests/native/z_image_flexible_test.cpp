#include "../../native/backends/coreml.hpp"
#include <iostream>

// Real-artifact integration test; does not download weights or synthesize model metadata.
int main(int argc, char **argv) {
    try {
        tc::require(argc == 4, "z-image-flexible-test MODEL FIXED_MANIFEST FLEXIBLE_MANIFEST");
        std::atomic<bool> cancelled(false);
        auto event = [](const std::string &, int, int) {};
        auto checkpoint = std::filesystem::path(argv[1]) / "transformer/diffusion_pytorch_model.safetensors";
        {
            tc::HybridSession fixed(argv[2], argv[1], 1056, event, cancelled, 0, checkpoint);
            bool rejected = false;
            try { fixed.set_tokens(1088); }
            catch (const std::exception &e) {
                rejected = std::string(e.what()).find("request needs 1088 rows") != std::string::npos;
            }
            tc::require(rejected, "warm fixed cache did not explain insufficient capacity");
            tc::require(fixed.rows == 1056, "failed selection mutated active shape");
        }
        tc::HybridSession flexible(argv[3], argv[1], 1056, event, cancelled, 0, checkpoint);
        auto load = flexible.load_seconds;
        for (int rows = 1056; rows <= 1536; rows += 32) {
            flexible.set_tokens(rows);
            tc::require(flexible.rows == rows, "incorrect selected shape");
            auto x = tc::mx::full({1, rows, 3840}, 0.125f, tc::mx::float16);
            tc::mx::eval(x);
            auto y = flexible.predict(0, x);
            tc::mx::eval(y);
            tc::require(y.shape() == x.shape(), "variable output shape mismatch");
            tc::require(tc::mx::all(tc::mx::isfinite(y)).item<bool>(), "non-finite variable output");
        }
        flexible.set_tokens(1056);
        tc::require(flexible.load_seconds == load, "shape switch reloaded model session");
        tc::require(flexible.metrics().copied_bytes == 0, "variable shape declined output backing");
        bool rejected = false;
        try { flexible.set_tokens(1537); }
        catch (const std::exception &) { rejected = true; }
        tc::require(rejected && flexible.rows == 1056, "out-of-range input changed active shape");
        std::cout << "PASS: 16 real input shapes, output binding, warm capacity error, model reuse, upper boundary\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}

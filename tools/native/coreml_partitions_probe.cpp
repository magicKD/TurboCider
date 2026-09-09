#include "../../native/backends/coreml_partitions.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 5, "usage: coreml-partitions-probe MODEL.mlmodelc ROWS HIDDEN OUTPUT.safetensors");
        const int rows = std::stoi(argv[2]), hidden = std::stoi(argv[3]);
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::CoreMLPartitions partitions({std::filesystem::absolute(argv[1])}, rows, hidden, event, cancelled);
        auto input = tc::mx::contiguous(tc::mx::random::normal({1, rows, hidden}, tc::mx::float16,
                                                               tc::mx::random::key(42)));
        tc::mx::eval(input);
        auto output = partitions.predict(0, input);
        tc::mx::eval(output);
        tc::require(tc::mx::all(tc::mx::isfinite(output)).item<bool>(), "nonfinite Core ML output");
        tc::mx::save_safetensors(argv[4], {{"input", input}, {"output", output}});
        bool rejected = false;
        try { partitions.predict(1, input); }
        catch (const std::exception &) { rejected = true; }
        tc::require(rejected, "invalid Core ML block index accepted");
        rejected = false;
        try { partitions.predict(0, tc::mx::astype(input, tc::mx::float32)); }
        catch (const std::exception &) { rejected = true; }
        tc::require(rejected, "invalid Core ML input dtype accepted");
        std::cout << "Core ML calls=" << partitions.calls()
                  << " copied_bytes=" << partitions.copied_bytes() << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

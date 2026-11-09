#include "../../native/models/wan22/block.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

namespace {
tc::Tensor repeat_rows(const tc::Tensor &value, int axis, int rows) {
    auto shape = value.shape();
    std::vector<int> repeats(shape.size(), 1);
    tc::mx::Shape starts, ends = shape;
    starts.resize(shape.size(), 0);
    repeats[axis] = (rows + shape[axis] - 1) / shape[axis];
    ends[axis] = rows;
    return tc::mx::slice(tc::mx::tile(value, repeats), starts, ends);
}
} // namespace

int main(int argc, char **argv) {
    try {
        tc::require(argc == 5, "usage: wan22-block-profile CHECKPOINT CAPTURE ROWS FIRST_FRAME_TOKENS");
        const int rows = std::stoi(argv[3]), first = std::stoi(argv[4]);
        tc::require(rows > 0 && rows <= 14400 && first > 0 && first < rows, "invalid profile geometry");
        auto fixture = tc::mx::load_safetensors(argv[2]).first;
        auto hidden = repeat_rows(fixture.at("hidden"), 1, rows);
        auto cosine = repeat_rows(fixture.at("cosine"), 0, rows);
        auto sine = repeat_rows(fixture.at("sine"), 0, rows);
        auto context = fixture.at("context"), modulation = fixture.at("modulation");
        tc::mx::eval(hidden, cosine, sine, context, modulation);
        tc::wan22::Block block;
        block.load(std::filesystem::absolute(argv[1]), 0);
        auto run = [&]() { return block.forward(hidden, context, modulation, cosine, sine, first); };
        tc::mx::eval(run());
        tc::mx::eval(run());
        std::cout << std::fixed << std::setprecision(9);
        for (int iteration = 0; iteration < 3; ++iteration) {
            const auto start = tc::Clock::now();
            tc::mx::eval(run());
            std::cout << "untraced," << iteration << "," << std::chrono::duration<double>(tc::Clock::now() - start).count() << '\n';
        }
        for (int iteration = 0; iteration < 3; ++iteration) {
            auto last = tc::Clock::now();
            tc::wan22::Block::Trace trace = [&](const std::string &name, const tc::Tensor &value) {
                tc::mx::eval(value);
                const auto now = tc::Clock::now();
                std::cout << name << ',' << iteration << ',' << std::chrono::duration<double>(now - last).count() << '\n';
                last = now;
            };
            auto result = block.forward(hidden, context, modulation, cosine, sine, first, trace);
            tc::mx::eval(result);
            std::cout << "final_residual," << iteration << ',' << std::chrono::duration<double>(tc::Clock::now() - last).count() << '\n';
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

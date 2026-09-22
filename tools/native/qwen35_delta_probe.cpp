#include "../../native/models/qwen21/pe_delta.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3 || (argc == 4 && std::string(argv[3]) == "--gpu"),
                    "usage: qwen35-delta-probe input.safetensors output.safetensors [--gpu]");
        if (argc == 4) tc::configure_streams();
        else tc::mx::set_default_device(tc::mx::Device::cpu);
        auto [x, metadata] = tc::mx::load_safetensors(argv[1]);
        std::optional<tc::Tensor> initial;
        if (x.count("initial")) initial = x.at("initial");
        std::atomic<bool> cancelled{false};
        const bool normalize = metadata.at("normalize") == "true";
        const int chunk_size = metadata.count("chunk_size") ? std::stoi(metadata.at("chunk_size")) : 64;
        auto invoke = [&](int begin, int end, const std::optional<tc::Tensor> &state, bool chunked = false) {
            const auto q = tc::slice_axis(x.at("q"), 1, begin, end);
            if (chunked) return tc::qwen21::pe::chunked_delta(q,
                tc::slice_axis(x.at("k"), 1, begin, end), tc::slice_axis(x.at("v"), 1, begin, end),
                tc::slice_axis(x.at("g"), 1, begin, end), tc::slice_axis(x.at("beta"), 1, begin, end),
                state, normalize, cancelled, chunk_size);
            return tc::qwen21::pe::recurrent_delta(
                tc::slice_axis(x.at("q"), 1, begin, end), tc::slice_axis(x.at("k"), 1, begin, end),
                tc::slice_axis(x.at("v"), 1, begin, end), tc::slice_axis(x.at("g"), 1, begin, end),
                tc::slice_axis(x.at("beta"), 1, begin, end), state, normalize, cancelled);
        };
        const int length = x.at("q").shape(1);
        auto result = invoke(0, length, initial);
        tc::mx::eval({result.output, result.state});
        std::unordered_map<std::string, tc::Tensor> outputs{{"output", result.output}, {"state", result.state}};
        auto chunked = invoke(0, length, initial, true);
        tc::mx::eval({chunked.output, chunked.state});
        outputs.emplace("chunk_output", chunked.output);
        outputs.emplace("chunk_state", chunked.state);
        if (length > 1) {
            auto prefix = invoke(0, length - 1, initial);
            tc::mx::eval({prefix.output, prefix.state});
            auto decode = invoke(length - 1, length, prefix.state);
            outputs.emplace("split_output", tc::mx::concatenate({prefix.output, decode.output}, 1));
            outputs.emplace("split_state", decode.state);
            prefix = invoke(0, length - 1, initial, true);
            tc::mx::eval({prefix.output, prefix.state});
            decode = invoke(length - 1, length, prefix.state);
            outputs.emplace("chunk_split_output", tc::mx::concatenate({prefix.output, decode.output}, 1));
            outputs.emplace("chunk_split_state", decode.state);
        }
        if (metadata.count("benchmark") && metadata.at("benchmark") == "true") {
            std::vector<float> durations;
            for (bool chunk : {false, true}) {
                for (int repeat = 0; repeat < 4; ++repeat) {
                    const auto start = tc::Clock::now();
                    auto measured = invoke(0, length, initial, chunk);
                    tc::mx::eval({measured.output, measured.state});
                    if (repeat) durations.push_back(std::chrono::duration<float>(tc::Clock::now() - start).count());
                }
            }
            outputs.emplace("seconds", tc::Tensor(durations.data(), {2, 3}));
        }
        tc::mx::save_safetensors(argv[2], outputs);
        cancelled = true;
        bool rejected = false;
        try { (void)invoke(0, length, initial); } catch (const tc::Cancelled &) { rejected = true; }
        tc::require(rejected, "Qwen35 delta cancellation ignored");
        rejected = false;
        try { (void)invoke(0, length, initial, true); } catch (const tc::Cancelled &) { rejected = true; }
        tc::require(rejected, "Qwen35 chunk delta cancellation ignored");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

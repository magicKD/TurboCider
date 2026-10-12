#include "../../native/backends/mlx.hpp"

#include <fstream>
#include <iostream>

// CPU-only, reduced geometry of the installed ordinary Qwen21 adapter:
// 32 blocks, seven projections per block, BF16 A/B.default.weight pairs.
int main(int argc, char **argv) {
    try {
        tc::require(argc == 2, "usage: qwen21_lora_binding_probe FIXTURE_DIRECTORY");
        tc::mx::set_default_device(tc::mx::Device(tc::mx::Device::cpu));
        constexpr int hidden = 8, ffn = 12, rank = 2;
        std::vector<std::string> keys;
        std::vector<tc::Tensor> bases;
        std::unordered_map<std::string, tc::Tensor> adapter;
        auto base = [&](const std::string &stem, int output, int input) {
            keys.push_back(stem + ".weight");
            bases.push_back(tc::mx::zeros({output, input}, tc::mx::bfloat16));
        };
        auto pair = [&](const std::string &stem, int output, int input) {
            adapter.emplace(stem + ".lora_A.default.weight",
                            tc::mx::ones({rank, input}, tc::mx::bfloat16));
            adapter.emplace(stem + ".lora_B.default.weight",
                            tc::mx::ones({output, rank}, tc::mx::bfloat16));
        };
        for (int block = 0; block < 32; ++block) {
            const auto prefix = "transformer_blocks." + std::to_string(block);
            for (const auto *projection : {"to_q", "to_k", "to_v", "to_out.0"}) {
                const auto stem = prefix + ".attn." + projection;
                base(stem, hidden, hidden);
                pair(stem, hidden, hidden);
            }
            base(prefix + ".img_mlp.gate_up", 2 * ffn, hidden);
            pair(prefix + ".img_mlp.gate_layer", ffn, hidden);
            pair(prefix + ".img_mlp.proj", ffn, hidden);
            base(prefix + ".img_mlp.out", hidden, ffn);
            pair(prefix + ".img_mlp.out", hidden, ffn);
        }
        const auto path = std::filesystem::path(argv[1]) / "ordinary.safetensors";
        auto write = [&] { tc::mx::save_safetensors(path.string(), adapter); };
        write();
        tc::Weights weights;
        std::atomic<bool> cancelled{false};
        const tc::Event event = [](const std::string &, int, int) {};
        auto bind = [&](bool strict) {
            weights.bind_arrays(keys, bases);
            return weights.apply_loras({{path.string(), .75f, "transformer"}},
                                       "transformer", event, cancelled, true, strict);
        };
        tc::require(bind(true) == 224, "ordinary adapter must bind all 224 projections");
        tc::require(weights.has_runtime_loras(), "runtime adapter was not retained");
        // Verify the real fused gate/up coordinate mapping on the CPU.
        auto input = tc::mx::ones({1, hidden}, tc::mx::bfloat16);
        auto output = tc::mx::astype(weights.project(input, "transformer_blocks.0.img_mlp.gate_up"),
                                     tc::mx::float32);
        tc::mx::eval(output);
        tc::require(tc::mx::all(output == tc::Tensor(float(hidden * rank) * .75f)).item<bool>(),
                    "gate and up runtime branches did not both apply");
        auto rejects = [&](const std::string &reason) {
            bool rejected = false;
            try { (void)bind(true); }
            catch (const std::exception &error) {
                rejected = std::string(error.what()).find(reason) != std::string::npos;
            }
            return rejected && weights.sorted_keys().empty() && !weights.has_runtime_loras();
        };
        const std::string unsupported = "transformer_blocks.0.attn.to_q.lora_magnitude_vector.weight";
        adapter.emplace(unsupported, tc::mx::ones({hidden}, tc::mx::bfloat16));
        write();
        tc::require(rejects("unsupported LoRA tensor"),
                    "strict binding must reject unsupported extra tensors and discard partial state");
        tc::require(bind(false) == 224, "legacy ignored tensor semantics changed");
        adapter.erase(unsupported);
        const std::string orphan = "transformer_blocks.31.unknown_projection.alpha";
        adapter.emplace(orphan, tc::Tensor(float(rank)));
        write();
        tc::require(rejects("orphan LoRA alpha"), "strict binding must reject an orphan alpha");
        tc::require(bind(false) == 224, "legacy orphan alpha semantics changed");
        adapter.erase(orphan);
        const std::string known = "transformer_blocks.0.attn.to_q";
        const auto original_a = adapter.at(known + ".lora_A.default.weight");
        const auto original_b = adapter.at(known + ".lora_B.default.weight");
        adapter.insert_or_assign(known + ".lora_B.default.weight",
                                 tc::mx::ones({hidden + 1, rank}, tc::mx::bfloat16));
        write();
        tc::require(rejects("LoRA output rows"), "strict binding must reject unused B output rows");
        tc::require(bind(false) == 224, "legacy oversized B semantics changed");
        // MLX's writer rejects empty arrays before the loader can see them.
        // Encode this zero-rank pair directly using the safetensors layout.
        std::string zero_header = "{\"" + known + ".lora_A.default.weight\":{\"dtype\":\"BF16\","
            "\"shape\":[0," + std::to_string(hidden) + "],\"data_offsets\":[0,0]},\"" + known +
            ".lora_B.default.weight\":{\"dtype\":\"BF16\",\"shape\":[" + std::to_string(hidden) +
            ",0],\"data_offsets\":[0,0]}}";
        while (zero_header.size() % 8) zero_header.push_back(' ');
        const uint64_t zero_header_bytes = zero_header.size();
        {
            std::ofstream zero_file(path, std::ios::binary | std::ios::trunc);
            zero_file.write(reinterpret_cast<const char *>(&zero_header_bytes), sizeof(zero_header_bytes));
            zero_file.write(zero_header.data(), zero_header.size());
            tc::require(bool(zero_file), "failed to write zero-rank safetensors fixture");
        }
        tc::require(rejects("LoRA rank and dimensions must be positive"),
                    "strict binding must reject a zero-rank pair");
        adapter.insert_or_assign(known + ".lora_A.default.weight", original_a);
        adapter.insert_or_assign(known + ".lora_B.default.weight", original_b);
        pair("transformer_blocks.31.unknown_projection", hidden, hidden);
        write();
        bool rejected_unknown = false;
        try { (void)bind(true); }
        catch (const std::exception &error) {
            rejected_unknown = std::string(error.what()).find("unknown_projection") != std::string::npos;
        }
        tc::require(rejected_unknown && weights.sorted_keys().empty() && !weights.has_runtime_loras(),
                    "strict binding must reject unknown pairs and discard partial state");
        tc::require(bind(false) == 224, "legacy non-strict binding semantics changed");
        adapter.erase("transformer_blocks.31.unknown_projection.lora_B.default.weight");
        write();
        bool rejected_incomplete = false;
        try { (void)bind(true); }
        catch (const std::exception &error) {
            rejected_incomplete = std::string(error.what()).find("incomplete LoRA pair") != std::string::npos;
        }
        tc::require(rejected_incomplete && weights.sorted_keys().empty(),
                    "incomplete pair must fail closed");
        std::cout << "{\"device\":\"cpu\",\"projections\":224,\"unknown_rejected\":true,"
                     "\"unsupported_tensor_rejected\":true,\"orphan_alpha_rejected\":true,"
                     "\"oversized_b_rejected\":true,\"zero_rank_rejected\":true,"
                     "\"incomplete_rejected\":true,\"legacy_subset_preserved\":true}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

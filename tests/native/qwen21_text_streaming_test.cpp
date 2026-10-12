#include "../../native/models/qwen21/text_encoder.hpp"
#include "../../native/models/qwen21/transformer.hpp"
#include "../../native/runtime/streaming/source_lease.hpp"
#include <mlx/random.h>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 2, "expected fixture directory");
        tc::configure_streams();
        tc::mx::set_cache_limit(0);
        tc::qwen21::TextConfig config;
        config.layers = 6; config.heads = 4; config.kv_heads = 2;
        config.head_dim = 16; config.mrope_sections = {4, 2, 2};
        auto positions = tc::mx::broadcast_to(tc::mx::reshape(tc::mx::arange(7, tc::mx::int32), {1, 7}), {3, 7});
        int seed = 0;
        auto random = [&](tc::mx::Shape shape) {
            return tc::mx::astype(tc::mx::random::normal(shape, tc::mx::float32,
                tc::mx::random::key(++seed)) * .03f, tc::mx::bfloat16);
        };
        for (const std::string stem : {"model.", "model.language_model."}) {
            std::unordered_map<std::string, tc::Tensor> data;
            data.emplace(stem + "embed_tokens.weight", random({32, 64}));
            data.emplace(stem + "norm.weight", tc::mx::ones({64}, tc::mx::bfloat16));
            for (int i = 0; i < config.layers; ++i) {
                auto p = stem + "layers." + std::to_string(i);
                for (auto name : {"input_layernorm", "post_attention_layernorm"})
                    data.emplace(p + "." + name + ".weight", tc::mx::ones({64}, tc::mx::bfloat16));
                for (auto name : {"q_norm", "k_norm"})
                    data.emplace(p + ".self_attn." + name + ".weight", tc::mx::ones({16}, tc::mx::bfloat16));
                data.emplace(p + ".self_attn.q_proj.weight", random({64, 64}));
                data.emplace(p + ".self_attn.k_proj.weight", random({32, 64}));
                data.emplace(p + ".self_attn.v_proj.weight", random({32, 64}));
                data.emplace(p + ".self_attn.o_proj.weight", random({64, 64}));
                data.emplace(p + ".mlp.gate_proj.weight", random({128, 64}));
                data.emplace(p + ".mlp.up_proj.weight", random({128, 64}));
                data.emplace(p + ".mlp.down_proj.weight", random({64, 128}));
            }
            auto path = std::filesystem::path(argv[1]) / (stem + "safetensors");
            tc::mx::save_safetensors(path.string(), data);
            data.clear();
            auto embeddings = random({1, 7, 64});
            std::vector<tc::Tensor> deepstack{random({1, 7, 64}), random({1, 7, 64})};
            for (bool final_norm : {false, true}) {
                config.final_norm = final_norm;
                tc::Weights resident; resident.load_file(path);
                tc::qwen21::TextEncoder reference(resident, config);
                std::atomic<bool> cancelled{false};
                auto expected = reference.encode_embeddings(embeddings, positions, 5, {}, cancelled, deepstack);
                tc::mx::eval(expected);
                resident.clear();
                auto run = [&](bool cancel_midway) {
                    tc::Weights streamed; streamed.load_file(path);
                    tc::qwen21::TextEncoder encoder(streamed, config);
                    streamed.erase(stem + "embed_tokens.weight");
                    int released = 0;
                    auto release = [&](const std::string &prefix) {
                        tc::require(prefix == stem + "layers." + std::to_string(released) + ".", "layer release order changed");
                        streamed.erase_prefix(prefix);
                        tc::require(!streamed.has(prefix + "self_attn.q_proj.weight"), "consumed layer remains resident");
                        ++released;
                        if (cancel_midway && released == 2) cancelled.store(true);
                    };
                    try {
                        auto actual = encoder.encode_embeddings(embeddings, positions, 5, {}, cancelled, deepstack, release);
                        tc::mx::eval(actual);
                        tc::require(!cancel_midway && released == config.layers, "streaming coverage incomplete");
                        tc::require(tc::mx::array_equal(expected, actual).item<bool>(), "streamed BF16 output differs");
                    } catch (const tc::Cancelled &) {
                        tc::require(cancel_midway && released == 2, "unexpected streaming cancellation");
                    }
                };
                run(false);
                run(true);
                cancelled.store(false);
                run(false); // cancellation must permit a clean reload/retry
            }
        }
        // Exercise the actual compiled DiT blocks over prefill and two decode
        // passes. Restoring lazy weights must preserve the prefix cache and
        // never execute a graph captured against already-consumed arrays.
        tc::qwen21::TransformerConfig dit_config;
        dit_config.layers = 4; dit_config.heads = 2; dit_config.head_dim = 8;
        dit_config.channels = 4; dit_config.context_dim = 12;
        dit_config.rope_axes = {2, 2, 4};
        std::unordered_map<std::string, tc::Tensor> data;
        for (const auto &[name, shape] : std::vector<std::pair<std::string, tc::mx::Shape>>{
                {"time_text_embed.timestep_embedder.linear_1", {16, 256}},
                {"time_text_embed.timestep_embedder.linear_2", {16, 16}},
                {"modulation.1", {64, 16}}, {"img_in", {16, 4}},
                {"txt_in.in_layer", {24, 12}}, {"txt_in.out_layer", {16, 24}},
                {"norm_out.linear", {16, 16}}, {"proj_out", {4, 16}}})
            data.emplace(name + ".weight", random(shape));
        data.emplace("txt_in.text_norm.weight", random({12}));
        for (int i = 0; i < dit_config.layers; ++i) {
            auto p = "transformer_blocks." + std::to_string(i);
            for (auto name : {"to_q", "to_k", "to_v", "to_out.0"})
                data.emplace(p + ".attn." + name + ".weight", random({16, 16}));
            for (auto name : {"norm_q", "norm_k"})
                data.emplace(p + ".attn." + name + ".weight", tc::mx::ones({8}, tc::mx::bfloat16));
            data.emplace(p + ".img_mlp.gate_up.weight", random({64, 16}));
            data.emplace(p + ".img_mlp.out.weight", random({16, 32}));
        }
        auto dit_path = std::filesystem::path(argv[1]) / "dit.safetensors";
        tc::mx::save_safetensors(dit_path.string(), data);
        data.clear();
        auto initial = random({1, 4, 4}), text = random({1, 3, 12});
        {
            std::atomic<bool> cancelled{false};
            const tc::Event quiet = [](const std::string &, int, int) {};
            tc::Weights resident; resident.load_file(dit_path);
            tc::qwen21::Transformer reference(resident, dit_config);
            tc::streaming::SourceFileIdentity source;
            source.logical_id = "dit"; source.path = dit_path;
            auto lease = tc::streaming::SourceLease::capture({source});
            tc::Weights streamed; streamed.load_lease(lease, {"dit"}, quiet, cancelled);
            tc::qwen21::Transformer model(streamed, dit_config);
            int released = 0;
            model.set_layer_release([&](const std::string &prefix) {
                tc::require(prefix == "transformer_blocks." + std::to_string(released % dit_config.layers) + ".",
                            "DiT layer release order changed");
                streamed.erase_prefix(prefix); ++released;
            });
            auto actual = initial, expected = initial;
            for (int step = 0; step < 6; ++step) {
                if (step) {
                    lease->revalidate_after_drain();
                    streamed.clear();
                    streamed.load_lease(lease, {"dit"}, quiet, cancelled);
                }
                float sigma = 1.f - step * .15f;
                expected = expected + reference.forward(expected, text, sigma, 2, 2) * tc::Tensor(-.15f, tc::mx::bfloat16);
                tc::mx::eval(expected);
                actual = actual + model.forward(actual, text, sigma, 2, 2) * tc::Tensor(-.15f, tc::mx::bfloat16);
                tc::mx::eval(actual);
                tc::require(tc::mx::array_equal(expected, actual).item<bool>(), "streamed DiT trajectory differs");
                tc::require(model.cached_layers() == size_t(dit_config.layers), "streaming lost prefix KV");
                tc::require(released == (step + 1) * dit_config.layers, "streamed DiT skipped a layer");
            }
        }
        std::cout << "PASS Qwen21 text streaming: exact BF16, both key layouts, raw/final norm, deepstack, padding, cancellation/retry; DiT prefill/decode, six-step trajectory and prefix KV\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}

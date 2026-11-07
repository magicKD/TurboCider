#include "models/qwen21/bf16_streaming.hpp"
#include "models/qwen21/text_encoder.hpp"
#include "models/qwen21/encoder_convrot.hpp"
#include "models/qwen21/transformer.hpp"
#include <iostream>

using namespace tc;
using namespace tc::qwen21;

namespace {
constexpr uint32_t layers = 4;
constexpr uint64_t budget = 16ull << 20;

template <typename Function>
void rejects(Function &&fn, std::string_view reason) {
    try {
        fn();
    } catch (const std::exception &error) {
        require(std::string_view(error.what()).find(reason) != std::string_view::npos,
                std::string("wrong rejection: ") + error.what());
        return;
    }
    throw std::runtime_error("invalid streaming source or budget was accepted");
}

void check_plan(const Bf16StreamPlan &plan, bool encoder) {
    const auto &stage = plan.descriptor.stages.at(0);
    const auto &layout = plan.layout.stages.at(0);
    require(plan.prefix == 0 && stage.blocks.size() == layers && layout.groups.size() == layers &&
            layout.pools.at(0).slots.size() == 2, "unexpected ConvRot layer slot layout");
    require(stage.resident_fields.size() == (encoder ? 1 : 9), "unexpected fixed ConvRot fields");
    require(plan.embedding.has_value() == encoder, "sparse embedding source missing or attached to DiT");
    require(stage.blocks.at(0).fields.size() == (encoder ? 25 : 20), "unexpected typed layer source closure");
    for (const auto &field : stage.resident_fields)
        require(!field.name.starts_with("lm_head.") && !field.name.starts_with("model.visual.") &&
                !field.name.starts_with("model.embed_tokens."),
                "unused encoder weights became resident");
    for (const auto &block : stage.blocks)
        for (const auto &field : block.fields)
            require(field.name != "attn.comfy_attention.config", "unused attention config became a slot field");
}

void equal(const Tensor &actual, const Tensor &reference, const char *message) {
    mx::eval({actual, reference});
    require(actual.shape() == reference.shape() && actual.dtype() == reference.dtype() &&
            mx::all(actual == reference).item<bool>(), message);
}

Weights reference_weights(const std::filesystem::path &path, bool w8a8) {
    Weights weights;
    weights.load_file(path);
    if (w8a8) weights.set_convrot_w8a8(true);
    else weights.pack_convrot_q8(128, mx::float32);
    weights.materialize();
    return weights;
}

void check_bank(const QwenBf16StreamStageMetrics &metrics, uint32_t passes, bool w8a8) {
    require(metrics.drained && metrics.slots == 2 && metrics.peak_live_layer_owners == 1 &&
            metrics.completed_passes == passes && metrics.completed_layers == layers * passes &&
            metrics.fills == metrics.completed_layers && metrics.completed_reader_fences == metrics.fills &&
            metrics.reader_fences == metrics.fills, "ConvRot slot epochs or reader lifetime were not completed");
    require(metrics.source_precision == "original_i8_convrot" &&
            metrics.matmul_mode == (w8a8 ? "int8_tensorops_w8a8" : "q8_g128_float_a") &&
            (metrics.derived_weight_capacity_bytes == 0) == w8a8 &&
            metrics.managed_weight_capacity_bytes + metrics.derived_weight_capacity_bytes <= budget,
            "ConvRot route or managed-capacity accounting differs");
}

void check_encoder(const char *path, StreamWeightMode mode, bool w8a8, std::atomic<bool> &cancel) {
    auto weights = reference_weights(path, w8a8);
    TextConfig config;
    config.layers = layers;
    config.heads = config.kv_heads = 2;
    config.final_norm = false;
    Tokens tokens;
    tokens.ids = {4, 7, 11, 4};
    tokens.valid = 4;
    TextEncoder reference(weights, config);
    auto expected = reference.encode(tokens, {}, cancel);
    mx::eval(expected);

    Bf16StreamBank bank(bf16_stream_plan(path, true, layers, 0, 1, cancel, mode), budget, cancel);
    require(bank.embedding_vocabulary() == 128, "sparse embedding vocabulary changed");
    Tokens invalid=tokens;
    invalid.ids[0]=-1;
    rejects([&] { bank.embedding_rows(invalid); }, "out of vocabulary");
    invalid.ids[0]=128;
    rejects([&] { bank.embedding_rows(invalid); }, "out of vocabulary");
    invalid.ids.clear();
    rejects([&] { bank.embedding_rows(invalid); }, "bounded nonempty");
    invalid.ids.assign(Tokenizer::qwen21_limit+1,1);
    rejects([&] { bank.embedding_rows(invalid); }, "bounded nonempty");
    cancel.store(true);
    rejects([&] { bank.embedding_rows(tokens); }, "cancelled");
    cancel.store(false);
    auto embeddings=bank.embedding_rows(tokens);
    auto ids=Tensor(tokens.ids.data(),{int(tokens.ids.size())},mx::int32);
    equal(embeddings,encoder_embedding_rows(weights,ids,"model.embed_tokens"),
        "sparse embedding changed duplicate or out-of-order rows");
    rejects([&] { bank.embedding_rows(tokens); }, "exactly once");
    TextEncoder streamed(bank.resident(), config);
    streamed.set_layer_weights([&](int layer) -> const Weights & { return bank.acquire(layer); },
                              [&](int layer) { bank.retire(layer); });
    const auto before = convrot_w8a8_stats();
    bank.begin_pass(0);
    auto positions=mx::broadcast_to(mx::reshape(mx::arange(4,mx::int32),{1,4}),{3,4});
    equal(streamed.encode_embeddings(mx::expand_dims(embeddings,0),positions,tokens.valid,{},cancel),
        expected, "streamed ConvRot encoder changed hidden states");
    bank.finish_pass();
    bank.finish();
    check_bank(bank.metrics(), 1, w8a8);
    require(bank.metrics().resident_source_bytes == 512 && bank.metrics().gathered_embedding_rows == 4 &&
            bank.metrics().gathered_embedding_source_bytes == 4*(256+sizeof(float)),
            "sparse embedding did not drop the full vocabulary or record actual reads");
    require(convrot_w8a8_stats().projections - before.projections == (w8a8 ? 7 * layers : 0),
            "encoder lost its selected layer mode");
}

void check_denoiser(const char *path, StreamWeightMode mode, bool w8a8, std::atomic<bool> &cancel) {
    auto weights = reference_weights(path, w8a8);
    TransformerConfig config;
    config.layers = layers;
    config.heads = 2;
    config.context_dim = 256;
    Transformer reference(weights, config);
    auto input = mx::astype(mx::reshape(mx::sin(mx::arange(4 * 64, mx::float32) * .01f), {1, 4, 64}), mx::bfloat16);
    auto context = mx::astype(mx::reshape(mx::cos(mx::arange(3 * 256, mx::float32) * .013f), {1, 3, 256}), mx::bfloat16);
    mx::eval({input, context});
    constexpr uint32_t passes = 3;
    Bf16StreamBank bank(bf16_stream_plan(path, false, layers, 0, passes, cancel, mode), budget, cancel);
    Transformer streamed(bank.resident(), config);
    streamed.set_layer_weights([&](int layer) -> const Weights & { return bank.acquire(layer); },
                              [&](int layer) { bank.retire(layer); });
    for (uint32_t pass = 0; pass < passes; ++pass) {
        const float sigma = .9f - .2f * pass;
        auto expected = reference.forward(input, context, sigma, 2, 2, true);
        mx::eval(expected);
        const auto before = convrot_w8a8_stats();
        bank.begin_pass(pass);
        equal(streamed.forward(input, context, sigma, 2, 2, true), expected,
              "compiled DiT changed numerics across streamed slots");
        bank.finish_pass();
        require(convrot_w8a8_stats().projections - before.projections == (w8a8 ? 6 * layers : 0),
                "compiled DiT lost its selected layer mode");
        input = input + expected * Tensor(-.2f, mx::bfloat16);
        mx::eval(input);
    }
    bank.finish();
    check_bank(bank.metrics(), passes, w8a8);
}
} // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 8, "expected six owned safetensors fixtures and a mode");
        std::atomic<bool> cancel{false};
        const auto mode = StreamWeightMode::ConvRotFloat;
        auto encoder = bf16_stream_plan(argv[1], true, layers, 0, 1, cancel, mode);
        check_plan(encoder, true);
        check_plan(bf16_stream_plan(argv[2], false, layers, 0, 3, cancel, mode), false);
        rejects([&] { bf16_stream_plan(argv[3], true, layers, 0, 1, cancel, mode); }, "H256 marker");
        rejects([&] { bf16_stream_plan(argv[4], true, layers, 0, 1, cancel, mode); }, "layer source closure incomplete");
        rejects([&] { bf16_stream_plan(argv[5], false, layers, 0, 1, cancel, mode); }, "attention marker");
        rejects([&] { bf16_stream_plan(argv[6], true, layers, 0, 1, cancel, mode); }, "visual tensor");
        const auto &layout = encoder.layout.stages.at(0);
        rejects([&] { Bf16StreamBank bank(encoder, layout.resident_bytes + layout.peak_pool_bytes, cancel); },
                "managed weight budget floor");
        std::cout << "PASS ConvRot typed source closure and derived-capacity admission\n";
        if (std::string_view(argv[7]) == "gpu") {
            configure_streams();
            mx::set_cache_limit(0);
            check_encoder(argv[1], mode, false, cancel);
            check_denoiser(argv[2], mode, false, cancel);
            std::cout << "PASS ConvRot Float-A streamed/resident numerics and reader retirement\n";
            if (convrot_w8a8_available()) {
                check_encoder(argv[1], StreamWeightMode::ConvRotW8A8, true, cancel);
                check_denoiser(argv[2], StreamWeightMode::ConvRotW8A8, true, cancel);
                std::cout << "PASS ConvRot W8A8 streamed/resident numerics and projection coverage\n";
            } else std::cout << "SKIP ConvRot W8A8 GPU unavailable\n";
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

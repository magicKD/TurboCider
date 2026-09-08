#include "llada.hpp"

#include "../../backends/coreml.hpp"

#include <cmath>
#include <cstring>
#include <regex>

namespace tc {
namespace {

constexpr int kHidden = 3840;
constexpr int kHeads = 30;
constexpr int kHeadDim = 128;
constexpr int kSequenceMultiple = 32;

Tensor rms_plain(const Tensor &x, float eps = 1e-5f) {
    return mx::astype(mx::fast::rms_norm(mx::astype(x, mx::float32), {}, eps),
                      x.dtype());
}

Tensor layer_norm_plain(const Tensor &x, float eps = 1e-6f) {
    return mx::astype(mx::fast::layer_norm(mx::astype(x, mx::float32), {}, {}, eps),
                      x.dtype());
}

Tensor llada_rope(const Tensor &ids) {
    const int dimensions[] = {32, 48, 48};
    std::vector<Tensor> cosine;
    std::vector<Tensor> sine;
    for (int axis = 0; axis < 3; ++axis) {
        const int dim = dimensions[axis];
        auto inverse = 1.f /
                       mx::power(Tensor(256.f, mx::float32),
                                 mx::arange(0, dim, 2, mx::float32) / float(dim));
        auto positions = mx::astype(
            mx::squeeze(slice_axis(ids, 1, axis, axis + 1), 1), mx::float32);
        auto angle = mx::expand_dims(positions, -1) * mx::reshape(inverse, {1, dim / 2});
        cosine.push_back(mx::cos(angle));
        sine.push_back(mx::sin(angle));
    }
    return mx::stack({mx::concatenate(cosine, -1), mx::concatenate(sine, -1)}, -1);
}

Tensor timestep_embedding(float timestep, const Weights &weights) {
    constexpr int dimension = 256;
    auto frequencies = mx::exp(
        Tensor(-std::log(10000.f), mx::float32) *
        mx::arange(dimension / 2, mx::float32) / float(dimension / 2));
    auto arguments = Tensor(timestep, mx::float32) * frequencies;
    auto embedding = mx::reshape(mx::concatenate({mx::cos(arguments), mx::sin(arguments)}, -1),
                                 {1, dimension});
    embedding = mx::astype(embedding, mx::bfloat16);
    embedding = weights.project(embedding, "t_embedder.mlp.0");
    embedding = silu(embedding);
    return weights.project(embedding, "t_embedder.mlp.2");
}

Tensor attention(const Tensor &x, const Weights &weights, const std::string &prefix,
                 const Tensor &frequencies) {
    auto qkv = weights.project(x, prefix + ".attention.qkv");
    auto parts = mx::split(qkv, 3, -1);
    auto q = rms_plain(heads(parts[0], kHeads, kHeadDim));
    auto k = rms_plain(heads(parts[1], kHeads, kHeadDim));
    auto cos = mx::squeeze(slice_axis(frequencies, -1, 0, 1), -1);
    auto sin = mx::squeeze(slice_axis(frequencies, -1, 1, 2), -1);
    auto rotated = rope_pairs_pair(q, k, cos, sin);
    auto value = heads(parts[2], kHeads, kHeadDim);
    return weights.project(attend(rotated[0], rotated[1], value, false, {}, true),
                           prefix + ".attention.out");
}

Tensor feed_forward(const Tensor &x, const Weights &weights,
                    const std::string &prefix) {
    return weights.project(
        silu(weights.project(x, prefix + ".w1")) *
            weights.project(x, prefix + ".w3"),
        prefix + ".w2");
}

std::function<std::vector<Tensor>(const std::vector<Tensor> &)>
make_hybrid_gpu_graph(int mlp_width, int gpu_mlp_start) {
    require(mlp_width == 10240 && gpu_mlp_start > 0 &&
                gpu_mlp_start < mlp_width,
            "unsupported LLaDA hybrid FFN geometry");
    return mx::compile(
        [mlp_width, gpu_mlp_start](const std::vector<Tensor> &args) {
            require(args.size() == 4,
                    "invalid LLaDA hybrid GPU graph inputs");
            auto w1 = slice_axis(args[1], 0, gpu_mlp_start, mlp_width);
            auto w3 = slice_axis(args[2], 0, gpu_mlp_start, mlp_width);
            auto w2 = slice_axis(args[3], 1, gpu_mlp_start, mlp_width);
            auto gate = mx::matmul(args[0], mx::transpose(w1));
            auto up = mx::matmul(args[0], mx::transpose(w3));
            return std::vector<Tensor>{
                mx::matmul(silu(gate) * up, mx::transpose(w2))};
        });
}

std::function<std::vector<Tensor>(const std::vector<Tensor> &)> &compiled_block() {
    static auto graph = mx::compile([](const std::vector<Tensor> &args) {
        require(args.size() == 10, "invalid LLaDA compiled block inputs");
        auto fast_rms = [](const Tensor &x, float eps = 1e-5f) {
            return mx::astype(mx::fast::rms_norm(mx::astype(x, mx::float32), {}, eps),
                              x.dtype());
        };
        auto modulation = mx::expand_dims(
            mx::matmul(args[2], mx::transpose(args[3])) + args[4], 1);
        auto mod = mx::split(modulation, 4, -1);
        auto attention_input = fast_rms(args[0]) *
                               (Tensor(1.f, mod[0].dtype()) + mod[0]);
        auto qkv = mx::matmul(attention_input, mx::transpose(args[5]));
        auto qkv_parts = mx::split(qkv, 3, -1);
        auto q = fast_rms(heads(qkv_parts[0], kHeads, kHeadDim));
        auto k = fast_rms(heads(qkv_parts[1], kHeads, kHeadDim));
        auto cosine = mx::squeeze(slice_axis(args[1], -1, 0, 1), -1);
        auto sine = mx::squeeze(slice_axis(args[1], -1, 1, 2), -1);
        auto rotated = rope_pairs_pair(q, k, cosine, sine);
        auto attended = attend(rotated[0], rotated[1],
                               heads(qkv_parts[2], kHeads, kHeadDim),
                               false, {}, true);
        auto projected = mx::matmul(attended, mx::transpose(args[6]));
        auto value = args[0] + mx::tanh(mod[1]) * fast_rms(projected);
        auto feed_input = fast_rms(value) *
                          (Tensor(1.f, mod[2].dtype()) + mod[2]);
        auto gate = mx::matmul(feed_input, mx::transpose(args[7]));
        auto up = mx::matmul(feed_input, mx::transpose(args[8]));
        auto feed = mx::matmul(silu(gate) * up, mx::transpose(args[9]));
        return std::vector<Tensor>{value + mx::tanh(mod[3]) * fast_rms(feed)};
    });
    return graph;
}

Tensor modulated_block(const Tensor &x, const Weights &weights,
                       const std::string &prefix, const Tensor &frequencies,
                       const Tensor &embedding, HybridSession *hybrid,
                       int hybrid_block,
                       const std::function<std::vector<Tensor>(
                           const std::vector<Tensor> &)> *gpu_graph,
                       bool compile) {
    if (compile && !hybrid) {
        return compiled_block()(
            {x, frequencies, embedding,
             weights.at(prefix + ".adaLN_modulation.0.weight"),
             weights.at(prefix + ".adaLN_modulation.0.bias"),
             weights.at(prefix + ".attention.qkv.weight"),
             weights.at(prefix + ".attention.out.weight"),
             weights.at(prefix + ".feed_forward.w1.weight"),
             weights.at(prefix + ".feed_forward.w3.weight"),
             weights.at(prefix + ".feed_forward.w2.weight")})[0];
    }
    auto modulation = mx::expand_dims(
        weights.project(embedding, prefix + ".adaLN_modulation.0"), 1);
    auto pieces = mx::split(modulation, 4, -1);
    auto attention_output = attention(
        rms_plain(x) * (Tensor(1.f, pieces[0].dtype()) + pieces[0]),
        weights, prefix, frequencies);
    auto value = x + mx::tanh(pieces[1]) * rms_plain(attention_output);
    auto feed_input =
        rms_plain(value) * (Tensor(1.f, pieces[2].dtype()) + pieces[2]);
    Tensor feed = feed_input;
    const auto ffn = prefix + ".feed_forward";
    if (hybrid) {
        require(gpu_graph != nullptr,
                "LLaDA hybrid GPU complement graph is missing");
        auto packed = mx::astype(feed_input, mx::float16);
        const int actual_rows = packed.shape(1);
        if (actual_rows < hybrid->rows)
            packed = mx::concatenate(
                {packed,
                 mx::zeros({1, hybrid->rows - actual_rows, kHidden},
                           mx::float16)},
                1);
        mx::eval({feed_input, packed});
        auto gpu = (*gpu_graph)(
            {feed_input, weights.at(ffn + ".w1.weight"),
             weights.at(ffn + ".w3.weight"),
             weights.at(ffn + ".w2.weight")})[0];
        mx::async_eval({gpu});
        auto ane = slice_axis(hybrid->predict(hybrid_block, packed), 1, 0,
                              actual_rows);
        feed = gpu + mx::astype(ane, gpu.dtype()) *
                         Tensor(hybrid->output_scale, gpu.dtype());
    } else {
        feed = feed_forward(feed_input, weights, ffn);
    }
    return value + mx::tanh(pieces[3]) * rms_plain(feed);
}

Tensor context_block(const Tensor &x, const Weights &weights,
                     const std::string &prefix, const Tensor &frequencies) {
    auto value = x + rms_plain(attention(rms_plain(x), weights, prefix, frequencies));
    return value + rms_plain(feed_forward(rms_plain(value), weights,
                                          prefix + ".feed_forward"));
}

int padded_length(int value) {
    return ((value + kSequenceMultiple - 1) / kSequenceMultiple) * kSequenceMultiple;
}

void dump_stage(const std::string &directory, int step, const std::string &name,
                const Tensor &value) {
    if (directory.empty() || step != 0)
        return;
    std::filesystem::create_directories(directory);
    mx::save_safetensors(
        (std::filesystem::path(directory) / ("llada_stage_" + name + ".safetensors")).string(),
        {{"tensor", value}});
}

Tensor pad_features(const Tensor &features, int padded, const Tensor &pad_token) {
    const int original = features.shape(1);
    if (original == padded)
        return features;
    auto padding = mx::broadcast_to(mx::reshape(pad_token, {1, 1, kHidden}),
                                    {1, padded - original, kHidden});
    return mx::concatenate({features, padding}, 1);
}

Tensor caption_ids(int original, int padded) {
    auto valid = mx::stack(
        {mx::arange(1, original + 1, mx::int32),
         mx::zeros({original}, mx::int32),
         mx::zeros({original}, mx::int32)},
        1);
    if (original == padded)
        return valid;
    return mx::concatenate({valid, mx::zeros({padded - original, 3}, mx::int32)}, 0);
}

Tensor image_ids(int height, int width, int start, int padded) {
    const int count = height * width;
    auto first = mx::full({count}, start, mx::int32);
    auto y = mx::reshape(mx::repeat(mx::arange(height, mx::int32), width), {count});
    auto x = mx::tile(mx::arange(width, mx::int32), {height});
    auto valid = mx::stack({first, y, x}, 1);
    if (count == padded)
        return valid;
    return mx::concatenate({valid, mx::zeros({padded - count, 3}, mx::int32)}, 0);
}

} // namespace

std::function<std::vector<Tensor>(const std::vector<Tensor> &)>
make_llada_hybrid_gpu_graph(int mlp_width, int gpu_mlp_start) {
    return make_hybrid_gpu_graph(mlp_width, gpu_mlp_start);
}

void normalize_llada_transformer(Weights &weights) {
    if (!weights.has("x_embedder.weight")) {
        weights.remap_keys([](std::string key) {
            constexpr const char *x_prefix = "all_x_embedder.1-1.";
            constexpr const char *final_prefix = "all_final_layer.1-1.";
            if (key.starts_with(x_prefix))
                key.replace(0, std::strlen(x_prefix), "x_embedder.");
            else if (key.starts_with(final_prefix))
                key.replace(0, std::strlen(final_prefix), "final_layer.");
            key = std::regex_replace(key,
                                     std::regex("\\.attention\\.to_out\\.0\\."),
                                     ".attention.out.");
            return key;
        });
    }
    for (const char *group : {"noise_refiner", "context_refiner", "layers"}) {
        const int count = std::string(group) == "layers" ? 30 : 2;
        for (int index = 0; index < count; ++index) {
            auto prefix = std::string(group) + "." + std::to_string(index) +
                          ".attention.";
            if (!weights.has(prefix + "qkv.weight"))
                weights.fuse_keys(prefix + "qkv.weight",
                                  {prefix + "to_q.weight", prefix + "to_k.weight",
                                   prefix + "to_v.weight"},
                                  0);
        }
    }
}

LLaDATransformerContext llada_prepare_transformer_context(
    const Tensor &caption, const Weights &weights, const Event &event,
    std::atomic<bool> &cancelled, const std::string &dump_directory) {
    require(caption.ndim() == 2 && caption.shape(1) == 2560,
            "LLaDA caption geometry mismatch");
    const int caption_tokens = caption.shape(0);
    const int padded_caption = padded_length(caption_tokens);
    auto condition = weights.project(
        mx::expand_dims(rms_plain(mx::astype(caption, mx::bfloat16)), 0),
        "cap_embedder.1");
    dump_stage(dump_directory, 0, "cap_embedder", condition);
    condition = pad_features(condition, padded_caption, weights.at("cap_pad_token"));
    auto frequencies = llada_rope(caption_ids(caption_tokens, padded_caption));
    for (int layer = 0; layer < 2; ++layer) {
        checkpoint(cancelled);
        event("llada_context_refiner", layer, 2);
        condition = context_block(condition, weights,
                                  "context_refiner." + std::to_string(layer),
                                  frequencies);
        mx::eval(condition);
        dump_stage(dump_directory, 0,
                   "context_refiner_" + std::to_string(layer), condition);
    }
    event("llada_context_refiner", 2, 2);
    return {std::move(condition), std::move(frequencies), caption_tokens,
            padded_caption};
}

Tensor llada_transformer(const Tensor &latent,
                         const LLaDATransformerContext &context,
                         float sigma, int height, int width,
                         const Weights &weights, const Event &event,
                         std::atomic<bool> &cancelled, HybridSession *hybrid,
                         const std::function<std::vector<Tensor>(
                             const std::vector<Tensor> &)> *gpu_graph,
                         bool compile_blocks,
                         const std::string &dump_directory, int dump_step) {
    require(latent.ndim() == 4 && latent.shape(0) == 1 && latent.shape(1) == 128 &&
                latent.shape(2) == height / 16 && latent.shape(3) == width / 16,
            "LLaDA latent geometry mismatch");
    require(context.features.ndim() == 3 && context.features.shape(0) == 1 &&
                context.features.shape(2) == kHidden &&
                context.features.shape(1) == context.padded_tokens &&
                context.frequencies.shape(0) == context.padded_tokens,
            "LLaDA prepared context geometry mismatch");

    const int latent_height = height / 16;
    const int latent_width = width / 16;
    const int image_tokens = latent_height * latent_width;
    const int padded_image = padded_length(image_tokens);
    const int padded_caption = context.padded_tokens;

    auto image = mx::reshape(mx::transpose(mx::astype(latent, mx::bfloat16),
                                            {0, 2, 3, 1}),
                             {1, image_tokens, 128});
    image = weights.project(image, "x_embedder");
    dump_stage(dump_directory, dump_step, "x_embedder", image);
    image = pad_features(image, padded_image, weights.at("x_pad_token"));
    auto image_frequencies = llada_rope(
        image_ids(latent_height, latent_width, padded_caption + 1, padded_image));

    auto embedding = timestep_embedding(sigma * 1000.f, weights);
    for (int layer = 0; layer < 2; ++layer) {
        checkpoint(cancelled);
        event("llada_noise_refiner", layer, 2);
        image = modulated_block(image, weights,
                                "noise_refiner." + std::to_string(layer),
                                image_frequencies, embedding, hybrid, layer,
                                gpu_graph, compile_blocks);
        if (!compile_blocks || hybrid || !dump_directory.empty())
            mx::eval(image);
        dump_stage(dump_directory, dump_step,
                   "noise_refiner_" + std::to_string(layer), image);
    }
    event("llada_noise_refiner", 2, 2);

    auto hidden = mx::concatenate({image, context.features}, 1);
    auto frequencies = mx::concatenate({image_frequencies, context.frequencies}, 0);
    for (int layer = 0; layer < 30; ++layer) {
        checkpoint(cancelled);
        event("llada_transformer", layer, 30);
        hidden = modulated_block(hidden, weights,
                                 "layers." + std::to_string(layer),
                                 frequencies, embedding, hybrid, 2 + layer,
                                 gpu_graph, compile_blocks);
        // Compiled blocks keep their MLX dependency chain.  Pure GPU
        // execution can therefore submit the whole denoising step before the
        // sampler synchronizes, instead of forcing 30 host-visible barriers
        // per step.  Diagnostic dumps and the eager compatibility path retain
        // the old synchronization granularity.
        if (!compile_blocks || hybrid || !dump_directory.empty()) {
            mx::eval(hidden);
            checkpoint(cancelled);
        }
        dump_stage(dump_directory, dump_step, "layer_" + std::to_string(layer), hidden);
    }
    event("llada_transformer", 30, 30);

    auto scale = Tensor(1.f, mx::bfloat16) +
                 weights.project(silu(embedding), "final_layer.adaLN_modulation.1");
    hidden = layer_norm_plain(hidden) * mx::expand_dims(scale, 1);
    auto output = weights.project(hidden, "final_layer.linear");
    dump_stage(dump_directory, dump_step, "final_layer", output);
    output = slice_axis(output, 1, 0, image_tokens);
    output = mx::transpose(mx::reshape(output,
                                       {1, latent_height, latent_width, 128}),
                           {0, 3, 1, 2});
    return -mx::astype(output, mx::float32);
}

} // namespace tc

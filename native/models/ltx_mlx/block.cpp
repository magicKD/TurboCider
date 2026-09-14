#include "block.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace tc::ltx_mlx {
namespace {

Tensor scalar_like(float value, mx::Dtype dtype) {
    return Tensor(value, dtype);
}

Tensor gelu_approx(const Tensor &value) {
    // This is MLX nn.gelu_approx's tanh formulation.  Keep it in one graph so
    // the C++ path has the same operation boundaries as the Python reference.
    static auto graph = mx::compile([](const std::vector<Tensor> &args) {
        const auto &x = args[0];
        const auto dtype = x.dtype();
        return std::vector<Tensor>{
            scalar_like(0.5f, dtype) * x *
            (scalar_like(1.0f, dtype) + mx::tanh(
                scalar_like(std::sqrt(2.0f / float(M_PI)), dtype) *
                (x + scalar_like(0.044715f, dtype) * mx::power(x, scalar_like(3.0f, dtype)))))};
    }, true);
    return graph({value})[0];
}

Tensor reshape_params(const Tensor &params, int count, int dim) {
    require(params.ndim() == 2 && params.shape(0) == 1 &&
                params.shape(1) == count * dim,
            "LTX AdaLN parameters must be [1, count*dim]");
    return mx::reshape(params, {1, count, dim});
}

struct PhaseProfiler {
    using Clock = std::chrono::steady_clock;

    int block = -1;
    uint64_t invocation = 0;
    bool enabled = false;
    Clock::time_point started{};
    Clock::time_point phase_started{};

    PhaseProfiler(int block_index, uint64_t forward_index)
        : block(block_index), invocation(forward_index) {
        const char *configured =
            std::getenv("TURBOCIDER_LTX_PROFILE_BLOCK");
        if (!configured || !*configured) return;
        char *end = nullptr;
        const long selected = std::strtol(configured, &end, 10);
        if (!end || *end || selected != block_index) return;
        if (const char *only =
                std::getenv("TURBOCIDER_LTX_PROFILE_INVOCATION")) {
            char *invocation_end = nullptr;
            const auto selected_invocation =
                std::strtoull(only, &invocation_end, 10);
            if (!invocation_end || *invocation_end ||
                selected_invocation != forward_index) return;
        }
        enabled = true;
        started = phase_started = Clock::now();
    }

    void finish(const char *phase, const std::vector<Tensor> &values) {
        if (!enabled) return;
        mx::eval(values);
        const auto now = Clock::now();
        const double milliseconds =
            std::chrono::duration<double, std::milli>(now - phase_started)
                .count();
        const double total =
            std::chrono::duration<double, std::milli>(now - started).count();
        std::fprintf(stderr,
                     "ltx_mlx_profile block=%d invocation=%llu phase=%s "
                     "milliseconds=%.6f total_ms=%.6f\n",
                     block, static_cast<unsigned long long>(invocation), phase,
                     milliseconds, total);
        phase_started = now;
    }
};

} // namespace

void BasicAVTransformerBlock::load(const std::filesystem::path &checkpoint,
                                   int block_index, int convrot_group_size) {
    clear();
    require(block_index >= 0 && block_index < 48,
            "LTX block index is outside the 48-block transformer");
    block_index_ = block_index;
    bf16_fences_ = std::getenv("TURBOCIDER_LTX_MLX_BF16_FENCES") != nullptr;
    weights_.set_metal_convrot(
        std::getenv("TURBOCIDER_LTX_MLX_METAL_CONVROT") != nullptr);
    const auto prefix = "model.diffusion_model.transformer_blocks." +
                        std::to_string(block_index) + ".";
    weights_.load_file(checkpoint, prefix);
    // Keep a raw ConvRot route available while validating the operation graph.
    // The packed MLX path is selected by default once its group-layout probe is
    // verified; raw mode is useful for isolating quantizer and graph errors.
    const bool raw_mode = std::getenv("TURBOCIDER_LTX_RAW_CONVROT") != nullptr;
    const auto packed = raw_mode ? 0 :
        static_cast<int>(weights_.pack_convrot_q8(convrot_group_size, mx::float32));
    require(raw_mode || packed > 0, "LTX block contains no ConvRot INT8 projections");
    weights_.materialize();
    require(weights_.has("scale_shift_table") &&
                weights_.has("audio_scale_shift_table") &&
                weights_.has("prompt_scale_shift_table") &&
                weights_.has("audio_prompt_scale_shift_table") &&
                weights_.has("scale_shift_table_a2v_ca_video") &&
                weights_.has("scale_shift_table_a2v_ca_audio"),
            "LTX block is missing AdaLN tables");
}

void BasicAVTransformerBlock::clear() {
    weights_.clear();
    forward_count_ = 0;
}

Tensor BasicAVTransformerBlock::rms_norm(const Tensor &value, float eps) const {
    auto f = mx::astype(value, mx::float32);
    return mx::astype(mx::fast::rms_norm(f, std::nullopt, eps), value.dtype());
}

std::vector<Tensor> BasicAVTransformerBlock::unpack(
    const Tensor &params, const std::string &table_key, int count, int dim) const {
    auto table = weights_.at(table_key);
    require(table.ndim() == 2 && table.shape(0) >= count && table.shape(1) == dim,
            "LTX AdaLN table geometry mismatch");
    table = mx::slice(table, {0, 0}, {count, dim});
    auto packed = reshape_params(params, count, dim) + mx::expand_dims(table, 0);
    std::vector<Tensor> result;
    result.reserve(count);
    for (int i = 0; i < count; ++i)
        result.push_back(mx::slice(packed, {0, i, 0}, {1, i + 1, dim}));
    return result;
}

Tensor BasicAVTransformerBlock::apply_rope(const Tensor &value,
                                           const RopeFrequencies &frequencies) const {
    require(value.ndim() == 4 && value.shape(-1) % 2 == 0,
            "LTX RoPE expects [B,H,N,D] with even D");
    const int half = value.shape(-1) / 2;
    require(frequencies.cosine.shape() == frequencies.sine.shape() &&
                frequencies.cosine.shape() ==
                    mx::Shape{value.shape(0), value.shape(1), value.shape(2), half},
            "LTX RoPE frequency geometry mismatch");
    if (std::getenv("TURBOCIDER_LTX_MLX_METAL_ROPE")) {
        static auto kernel = mx::fast::metal_kernel(
            "tc_ltx_rope_split", {"x", "cosine", "sine", "pairs", "half_dim"},
            {"out"},
            "uint pair = thread_position_in_grid.x; "
            "if (pair < uint(pairs)) { "
            "  uint h = uint(half_dim); uint row = pair / h; uint d = pair % h; "
            "  uint left = row * h * 2 + d; uint right = left + h; "
            "  float c = float(cosine[pair]); float s = float(sine[pair]); "
            "  float a = float(x[left]); float b = float(x[right]); "
            "  out[left] = T(a * c - b * s); "
            "  out[right] = T(a * s + b * c); "
            "}");
        const int pairs = int(value.size() / 2);
        return kernel({value, frequencies.cosine, frequencies.sine,
                       Tensor(pairs), Tensor(half)},
                      {value.shape()}, {value.dtype()}, {pairs, 1, 1},
                      {256, 1, 1}, {{"T", value.dtype()}}, {}, false, {})[0];
    }
    auto x = mx::astype(value, mx::float32);
    auto left = mx::slice(x, {0, 0, 0, 0},
                          {x.shape(0), x.shape(1), x.shape(2), half});
    auto right = mx::slice(x, {0, 0, 0, half}, x.shape());
    auto cosine = mx::astype(frequencies.cosine, mx::float32);
    auto sine = mx::astype(frequencies.sine, mx::float32);
    auto rotated = mx::concatenate({left * cosine - right * sine,
                                    left * sine + right * cosine}, -1);
    return mx::astype(rotated, value.dtype());
}

std::pair<Tensor, Tensor> BasicAVTransformerBlock::apply_rope_pair(
    const Tensor &query, const Tensor &key,
    const RopeFrequencies &frequencies) const {
    require(query.shape() == key.shape() && query.ndim() == 4 &&
                query.shape(-1) % 2 == 0,
            "LTX paired RoPE expects matching [B,H,N,D] tensors");
    if (!std::getenv("TURBOCIDER_LTX_MLX_METAL_ROPE"))
        return {apply_rope(query, frequencies), apply_rope(key, frequencies)};
    const int half = query.shape(-1) / 2;
    require(frequencies.cosine.shape() == frequencies.sine.shape() &&
                frequencies.cosine.shape() ==
                    mx::Shape{query.shape(0), query.shape(1), query.shape(2), half},
            "LTX paired RoPE frequency geometry mismatch");
    static auto kernel = mx::fast::metal_kernel(
        "tc_ltx_rope_split_qk",
        {"q", "k", "cosine", "sine", "pairs", "half_dim"},
        {"q_out", "k_out"},
        "uint pair = thread_position_in_grid.x; "
        "if (pair < uint(pairs)) { "
        "  uint h = uint(half_dim); uint row = pair / h; uint d = pair % h; "
        "  uint left = row * h * 2 + d; uint right = left + h; "
        "  float c = float(cosine[pair]); float s = float(sine[pair]); "
        "  float qa = float(q[left]); float qb = float(q[right]); "
        "  float ka = float(k[left]); float kb = float(k[right]); "
        "  q_out[left] = T(qa * c - qb * s); "
        "  q_out[right] = T(qa * s + qb * c); "
        "  k_out[left] = T(ka * c - kb * s); "
        "  k_out[right] = T(ka * s + kb * c); "
        "}");
    const int pairs = int(query.size() / 2);
    auto output = kernel(
        {query, key, frequencies.cosine, frequencies.sine,
         Tensor(pairs), Tensor(half)},
        {query.shape(), key.shape()}, {query.dtype(), key.dtype()},
        {pairs, 1, 1}, {256, 1, 1}, {{"T", query.dtype()}}, {}, false, {});
    return {std::move(output[0]), std::move(output[1])};
}

Tensor BasicAVTransformerBlock::attention(
    const Tensor &query_input, const std::optional<Tensor> &encoder_input,
    const std::string &prefix, int query_dim, int kv_dim, int out_dim,
    int heads, int head_dim, const std::optional<RopeFrequencies> &query_rope,
    const std::optional<RopeFrequencies> &key_rope,
    const std::optional<Tensor> &mask, bool gated) const {
    require(query_input.ndim() == 3 && query_input.shape(0) == 1 &&
                query_input.shape(-1) == query_dim,
            "LTX attention query geometry mismatch");
    const Tensor &kv_input = encoder_input ? *encoder_input : query_input;
    require(kv_input.ndim() == 3 && kv_input.shape(0) == 1 &&
                kv_input.shape(-1) == kv_dim,
            "LTX attention key/value geometry mismatch");

    Tensor q = query_input;
    Tensor k = kv_input;
    Tensor v = kv_input;
    if (!encoder_input) {
        auto qkv = weights_.project_many(
            query_input, {prefix + ".to_q", prefix + ".to_k",
                          prefix + ".to_v"});
        q = std::move(qkv[0]);
        k = std::move(qkv[1]);
        v = std::move(qkv[2]);
    } else {
        q = weights_.project(query_input, prefix + ".to_q");
        auto kv = weights_.project_many(
            kv_input, {prefix + ".to_k", prefix + ".to_v"});
        k = std::move(kv[0]);
        v = std::move(kv[1]);
    }
    q = mx::fast::rms_norm(q, weights_.at(prefix + ".q_norm.weight"),
                           norm_eps_);
    k = mx::fast::rms_norm(k, weights_.at(prefix + ".k_norm.weight"),
                           norm_eps_);
    const int query_rows = q.shape(1);
    const int key_rows = k.shape(1);
    q = mx::transpose(mx::reshape(q, {1, query_rows, heads, head_dim}),
                      {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {1, key_rows, heads, head_dim}),
                      {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {1, key_rows, heads, head_dim}),
                      {0, 2, 1, 3});
    if (query_rope) {
        if (!encoder_input && !key_rope) {
            auto pair = apply_rope_pair(q, k, *query_rope);
            q = std::move(pair.first);
            k = std::move(pair.second);
        } else {
            q = apply_rope(q, *query_rope);
            k = apply_rope(k, key_rope ? *key_rope : *query_rope);
        }
    }
    // Match the existing MLX backend heuristic by default. A fused SDPA
    // override is intentionally opt-in: it is useful for profiling the long
    // stage-2 sequence, but can select a numerically different kernel for
    // masked cross-attention on some MLX releases.
    const bool force_fused =
        std::getenv("TURBOCIDER_LTX_FORCE_FUSED_SDPA") != nullptr;
    auto attended = tc::attend(q, k, v, false, mask, force_fused);
    // tc::attend returns [B,N,H*D].  The reference applies the per-head gate
    // before merging heads, so re-form the head layout for that operation.
    auto heads_out = mx::transpose(
        mx::reshape(attended, {1, query_rows, heads, head_dim}), {0, 2, 1, 3});
    if (gated) {
        auto logits = weights_.project(query_input, prefix + ".to_gate_logits");
        auto gate = mx::transpose(mx::reshape(2.0f * mx::sigmoid(logits),
                                               {1, query_rows, heads}),
                                  {0, 2, 1});
        heads_out = heads_out * mx::expand_dims(gate, -1);
    }
    auto merged = mx::reshape(mx::transpose(heads_out, {0, 2, 1, 3}),
                              {1, query_rows, heads * head_dim});
    require(out_dim == heads * head_dim || weights_.has(prefix + ".to_out.0.weight"),
            "LTX attention output geometry mismatch");
    return weights_.project(merged, prefix + ".to_out.0");
}

Tensor BasicAVTransformerBlock::feed_forward(const Tensor &input,
                                              const std::string &prefix) const {
    auto hidden = weights_.project(input, prefix + ".net.0.proj");
    return weights_.project(gelu_approx(hidden), prefix + ".net.2");
}

std::pair<Tensor, Tensor> BasicAVTransformerBlock::forward(
    const Tensor &video_hidden, const Tensor &audio_hidden,
    const Tensor &video_adaln_params, const Tensor &audio_adaln_params,
    const Tensor &video_prompt_adaln_params,
    const Tensor &audio_prompt_adaln_params, const Tensor &av_video_params,
    const Tensor &av_audio_params, const Tensor &a2v_gate_params,
    const Tensor &v2a_gate_params, const std::optional<Tensor> &video_text_embeds,
    const std::optional<Tensor> &audio_text_embeds,
    const std::optional<RopeFrequencies> &video_rope,
    const std::optional<RopeFrequencies> &audio_rope,
    const std::optional<RopeFrequencies> &video_cross_rope,
    const std::optional<RopeFrequencies> &audio_cross_rope,
    const std::optional<Tensor> &video_attention_mask,
    const std::optional<Tensor> &audio_attention_mask,
    const std::optional<Tensor> &video_cross_attention_mask) const {
    require(video_hidden.ndim() == 3 && audio_hidden.ndim() == 3 &&
                video_hidden.shape(0) == 1 && audio_hidden.shape(0) == 1 &&
                video_hidden.shape(-1) == video_dim_ &&
                audio_hidden.shape(-1) == audio_dim_,
            "LTX block hidden-state geometry mismatch");

    PhaseProfiler profiler(block_index_, forward_count_++);
    auto fence = [&](Tensor value) {
        return bf16_fences_ ? mx::astype(value, mx::bfloat16) : value;
    };
    auto v = video_hidden;
    auto a = audio_hidden;
    auto vp = unpack(video_adaln_params, "scale_shift_table", 9, video_dim_);
    auto ap = unpack(audio_adaln_params, "audio_scale_shift_table", 9, audio_dim_);
    auto vprompt = unpack(video_prompt_adaln_params,
                          "prompt_scale_shift_table", 2, video_dim_);
    auto aprompt = unpack(audio_prompt_adaln_params,
                          "audio_prompt_scale_shift_table", 2, audio_dim_);
    auto avv = unpack(av_video_params, "scale_shift_table_a2v_ca_video", 4,
                      video_dim_);
    auto ava = unpack(av_audio_params, "scale_shift_table_a2v_ca_audio", 4,
                      audio_dim_);
    profiler.finish("adaln_unpack", {vp[0], ap[0], vprompt[0], aprompt[0],
                                      avv[0], ava[0]});

    auto v_norm = rms_norm(v, norm_eps_) * (1.0f + vp[1]) + vp[0];
    v = fence(v + fence(attention(v_norm, {}, "attn1", video_dim_, video_dim_, video_dim_,
                      video_heads_, video_head_dim_, video_rope, {},
                      video_attention_mask, true)) * vp[2]);
    profiler.finish("video_self", {v});

    auto a_norm = rms_norm(a, norm_eps_) * (1.0f + ap[1]) + ap[0];
    a = fence(a + fence(attention(a_norm, {}, "audio_attn1", audio_dim_, audio_dim_, audio_dim_,
                      audio_heads_, audio_head_dim_, audio_rope, {},
                      audio_attention_mask, true)) * ap[2]);
    profiler.finish("audio_self", {a});

    if (video_text_embeds) {
        v_norm = rms_norm(v, norm_eps_) * (1.0f + vp[7]) + vp[6];
        auto text = *video_text_embeds * (1.0f + vprompt[1]) + vprompt[0];
        v = fence(v + fence(attention(v_norm, text, "attn2", video_dim_, video_dim_, video_dim_,
                          video_heads_, video_head_dim_, {}, {},
                          video_cross_attention_mask, true)) * vp[8]);
        profiler.finish("video_text", {v});
    }
    if (audio_text_embeds) {
        a_norm = rms_norm(a, norm_eps_) * (1.0f + ap[7]) + ap[6];
        auto text = *audio_text_embeds * (1.0f + aprompt[1]) + aprompt[0];
        a = fence(a + fence(attention(a_norm, text, "audio_attn2", audio_dim_, audio_dim_, audio_dim_,
                          audio_heads_, audio_head_dim_, {}, {},
                          video_cross_attention_mask, true)) * ap[8]);
        profiler.finish("audio_text", {a});
    }

    // The reference computes both cross-modal branches from the same pre-branch
    // RMS-normalized states.  Do not normalize again after A->V updates.
    auto v_norm3 = rms_norm(v, norm_eps_);
    auto a_norm3 = rms_norm(a, norm_eps_);
    auto v_q_a2v = v_norm3 * (1.0f + avv[0]) + avv[1];
    auto a_kv_a2v = a_norm3 * (1.0f + ava[0]) + ava[1];
    auto a2v = fence(attention(
        v_q_a2v, a_kv_a2v, "audio_to_video_attn", video_dim_, audio_dim_,
        video_dim_, av_heads_, av_head_dim_, video_cross_rope,
        audio_cross_rope, {}, true));
    auto a2v_gate = a2v_gate_params + mx::expand_dims(mx::slice(
        weights_.at("scale_shift_table_a2v_ca_video"),
        {4, 0}, {5, video_dim_}), 0);
    v = fence(v + a2v * a2v_gate);
    profiler.finish("audio_to_video", {v});

    auto a_q_v2a = a_norm3 * (1.0f + ava[2]) + ava[3];
    auto v_kv_v2a = v_norm3 * (1.0f + avv[2]) + avv[3];
    auto v2a = fence(attention(
        a_q_v2a, v_kv_v2a, "video_to_audio_attn", audio_dim_, video_dim_,
        audio_dim_, av_heads_, av_head_dim_, audio_cross_rope,
        video_cross_rope, {}, true));
    auto v2a_gate = v2a_gate_params + mx::expand_dims(mx::slice(
        weights_.at("scale_shift_table_a2v_ca_audio"),
        {4, 0}, {5, audio_dim_}), 0);
    a = fence(a + v2a * v2a_gate);
    profiler.finish("video_to_audio", {a});

    auto v_ff_norm = rms_norm(v, norm_eps_) * (1.0f + vp[4]) + vp[3];
    v = fence(v + fence(feed_forward(v_ff_norm, "ff")) * vp[5]);
    profiler.finish("video_ffn", {v});
    auto a_ff_norm = rms_norm(a, norm_eps_) * (1.0f + ap[4]) + ap[3];
    a = fence(a + fence(feed_forward(a_ff_norm, "audio_ff")) * ap[5]);
    profiler.finish("audio_ffn", {a});
    return {v, a};
}

CompiledBlockForward::CompiledBlockForward(
    const BasicAVTransformerBlock &prototype,
    const std::optional<Tensor> &video_text_embeds,
    const std::optional<Tensor> &audio_text_embeds,
    const std::optional<RopeFrequencies> &video_rope,
    const std::optional<RopeFrequencies> &audio_rope,
    const std::optional<RopeFrequencies> &video_cross_rope,
    const std::optional<RopeFrequencies> &audio_cross_rope,
    const std::optional<Tensor> &video_attention_mask,
    const std::optional<Tensor> &audio_attention_mask,
    const std::optional<Tensor> &video_cross_attention_mask)
    : weight_keys_(prototype.weights_.sorted_keys()),
      video_text_(video_text_embeds.has_value()),
      audio_text_(audio_text_embeds.has_value()),
      video_rope_(video_rope.has_value()),
      audio_rope_(audio_rope.has_value()),
      video_cross_rope_(video_cross_rope.has_value()),
      audio_cross_rope_(audio_cross_rope.has_value()),
      video_attention_mask_(video_attention_mask.has_value()),
      audio_attention_mask_(audio_attention_mask.has_value()),
      video_cross_attention_mask_(video_cross_attention_mask.has_value()),
      metal_convrot_(prototype.weights_.metal_convrot()),
      bf16_fences_(prototype.bf16_fences_) {
    auto keys = weight_keys_;
    const bool has_video_text = video_text_;
    const bool has_audio_text = audio_text_;
    const bool has_video_rope = video_rope_;
    const bool has_audio_rope = audio_rope_;
    const bool has_video_cross_rope = video_cross_rope_;
    const bool has_audio_cross_rope = audio_cross_rope_;
    const bool has_video_attention_mask = video_attention_mask_;
    const bool has_audio_attention_mask = audio_attention_mask_;
    const bool has_video_cross_attention_mask = video_cross_attention_mask_;
    const bool use_metal_convrot = metal_convrot_;
    const bool use_bf16_fences = bf16_fences_;

    graph_ = mx::compile(
        [keys = std::move(keys), has_video_text, has_audio_text,
         has_video_rope, has_audio_rope, has_video_cross_rope,
         has_audio_cross_rope, has_video_attention_mask,
         has_audio_attention_mask, has_video_cross_attention_mask,
         use_metal_convrot, use_bf16_fences](const std::vector<Tensor> &args) {
            size_t cursor = 0;
            auto take = [&]() -> Tensor {
                require(cursor < args.size(),
                        "compiled LTX block input list is truncated");
                return args[cursor++];
            };
            auto take_optional = [&](bool present) -> std::optional<Tensor> {
                if (!present) return {};
                return take();
            };
            auto take_rope = [&](bool present)
                    -> std::optional<RopeFrequencies> {
                if (!present) return {};
                return RopeFrequencies{take(), take()};
            };

            auto video_hidden = take();
            auto audio_hidden = take();
            auto video_adaln = take();
            auto audio_adaln = take();
            auto video_prompt = take();
            auto audio_prompt = take();
            auto av_video = take();
            auto av_audio = take();
            auto a2v_gate = take();
            auto v2a_gate = take();
            auto active_video_text = take_optional(has_video_text);
            auto active_audio_text = take_optional(has_audio_text);
            auto active_video_rope = take_rope(has_video_rope);
            auto active_audio_rope = take_rope(has_audio_rope);
            auto active_video_cross_rope = take_rope(has_video_cross_rope);
            auto active_audio_cross_rope = take_rope(has_audio_cross_rope);
            auto active_video_attention_mask =
                take_optional(has_video_attention_mask);
            auto active_audio_attention_mask =
                take_optional(has_audio_attention_mask);
            auto active_video_cross_attention_mask =
                take_optional(has_video_cross_attention_mask);

            BasicAVTransformerBlock block;
            block.weights_.set_metal_convrot(use_metal_convrot);
            // Keep the compiled experiment's precision policy identical to
            // the eager prototype.  Forcing extra BF16 fences here changed
            // the long-sequence numerical trajectory more than compiling the
            // graph itself and made the opt-in candidate strictly worse.
            block.set_bf16_fences(use_bf16_fences);
            block.weights_.bind_arrays(keys, args, cursor);
            cursor += keys.size();
            require(cursor == args.size(),
                    "compiled LTX block input list has trailing arrays");
            auto output = block.forward(
                video_hidden, audio_hidden, video_adaln, audio_adaln,
                video_prompt, audio_prompt, av_video, av_audio,
                a2v_gate, v2a_gate, active_video_text, active_audio_text,
                active_video_rope, active_audio_rope,
                active_video_cross_rope, active_audio_cross_rope,
                active_video_attention_mask, active_audio_attention_mask,
                active_video_cross_attention_mask);
            return std::vector<Tensor>{std::move(output.first),
                                       std::move(output.second)};
        });
}

bool CompiledBlockForward::compatible(
    const BasicAVTransformerBlock &block,
    const std::optional<Tensor> &video_text_embeds,
    const std::optional<Tensor> &audio_text_embeds,
    const std::optional<RopeFrequencies> &video_rope,
    const std::optional<RopeFrequencies> &audio_rope,
    const std::optional<RopeFrequencies> &video_cross_rope,
    const std::optional<RopeFrequencies> &audio_cross_rope,
    const std::optional<Tensor> &video_attention_mask,
    const std::optional<Tensor> &audio_attention_mask,
    const std::optional<Tensor> &video_cross_attention_mask) const {
    return metal_convrot_ == block.weights_.metal_convrot() &&
           bf16_fences_ == block.bf16_fences_ &&
           video_text_ == video_text_embeds.has_value() &&
           audio_text_ == audio_text_embeds.has_value() &&
           video_rope_ == video_rope.has_value() &&
           audio_rope_ == audio_rope.has_value() &&
           video_cross_rope_ == video_cross_rope.has_value() &&
           audio_cross_rope_ == audio_cross_rope.has_value() &&
           video_attention_mask_ == video_attention_mask.has_value() &&
           audio_attention_mask_ == audio_attention_mask.has_value() &&
           video_cross_attention_mask_ ==
               video_cross_attention_mask.has_value();
}

std::pair<Tensor, Tensor> CompiledBlockForward::forward(
    const BasicAVTransformerBlock &block,
    const Tensor &video_hidden,
    const Tensor &audio_hidden,
    const Tensor &video_adaln_params,
    const Tensor &audio_adaln_params,
    const Tensor &video_prompt_adaln_params,
    const Tensor &audio_prompt_adaln_params,
    const Tensor &av_ca_video_params,
    const Tensor &av_ca_audio_params,
    const Tensor &av_ca_a2v_gate_params,
    const Tensor &av_ca_v2a_gate_params,
    const std::optional<Tensor> &video_text_embeds,
    const std::optional<Tensor> &audio_text_embeds,
    const std::optional<RopeFrequencies> &video_rope,
    const std::optional<RopeFrequencies> &audio_rope,
    const std::optional<RopeFrequencies> &video_cross_rope,
    const std::optional<RopeFrequencies> &audio_cross_rope,
    const std::optional<Tensor> &video_attention_mask,
    const std::optional<Tensor> &audio_attention_mask,
    const std::optional<Tensor> &video_cross_attention_mask) const {
    require(compatible(block, video_text_embeds, audio_text_embeds,
                       video_rope, audio_rope, video_cross_rope,
                       audio_cross_rope, video_attention_mask,
                       audio_attention_mask, video_cross_attention_mask),
            "compiled LTX block signature changed");
    std::vector<Tensor> args{
        video_hidden, audio_hidden, video_adaln_params, audio_adaln_params,
        video_prompt_adaln_params, audio_prompt_adaln_params,
        av_ca_video_params, av_ca_audio_params, av_ca_a2v_gate_params,
        av_ca_v2a_gate_params};
    auto append_optional = [&](const std::optional<Tensor> &value) {
        if (value) args.push_back(*value);
    };
    auto append_rope = [&](const std::optional<RopeFrequencies> &value) {
        if (!value) return;
        args.push_back(value->cosine);
        args.push_back(value->sine);
    };
    append_optional(video_text_embeds);
    append_optional(audio_text_embeds);
    append_rope(video_rope);
    append_rope(audio_rope);
    append_rope(video_cross_rope);
    append_rope(audio_cross_rope);
    append_optional(video_attention_mask);
    append_optional(audio_attention_mask);
    append_optional(video_cross_attention_mask);
    args.reserve(args.size() + weight_keys_.size());
    for (const auto &key : weight_keys_)
        args.push_back(block.weights_.at(key));
    auto output = graph_(args);
    require(output.size() == 2,
            "compiled LTX block returned an invalid output list");
    return {std::move(output[0]), std::move(output[1])};
}

RopeFrequencies precompute_rope(const Tensor &positions, int inner_dim,
                                int num_heads, const std::vector<int> &max_pos,
                                float theta, bool double_precision_grid) {
    require(positions.ndim() == 3 && positions.shape(0) == 1 &&
                positions.shape(2) == int(max_pos.size()),
            "LTX RoPE positions must be [1,N,axes]");
    const int axes = positions.shape(2);
    const int num_freqs = inner_dim / (2 * axes);
    Tensor indices = mx::zeros({num_freqs}, mx::float32);
    if (double_precision_grid) {
        // MLX float64 is CPU-only. Build the log-spaced table there and cast
        // before it enters the device graph, matching ltx-2-mlx.
        mx::StreamContext context(mx::Device::cpu);
        {
            indices = mx::power(
                Tensor(theta, mx::float64),
                mx::linspace(0.0, 1.0, num_freqs, mx::float64)) *
                      Tensor(float(M_PI / 2.0), mx::float64);
            indices = mx::astype(indices, mx::float32);
        }
    } else {
        indices = mx::power(Tensor(theta, mx::float32),
                            mx::linspace(0.0f, 1.0f, num_freqs, mx::float32)) *
                  Tensor(float(M_PI / 2.0f), mx::float32);
    }
    std::vector<Tensor> fractional;
    fractional.reserve(axes);
    for (int axis = 0; axis < axes; ++axis) {
        auto one = mx::slice(positions, {0, 0, axis},
                             {1, positions.shape(1), axis + 1});
        auto normalized = mx::astype(one, mx::float32) /
                          Tensor(float(max_pos[axis]), mx::float32);
        fractional.push_back(normalized * 2.0f - 1.0f);
    }
    auto frac = mx::concatenate(fractional, -1);
    auto scaled = mx::expand_dims(frac, -1) * indices;
    // [B,N,axes,num_freqs] -> [B,N,num_freqs,axes] -> [B,N,num_freqs*axes]
    auto angles = mx::reshape(mx::transpose(scaled, {0, 1, 3, 2}),
                              {1, positions.shape(1), num_freqs * axes});
    const int expected = inner_dim / 2;
    if (angles.shape(-1) < expected)
        angles = mx::concatenate({mx::zeros({1, positions.shape(1),
                                             expected - angles.shape(-1)},
                                             mx::float32), angles}, -1);
    require(angles.shape(-1) == expected, "LTX RoPE angle width mismatch");
    auto cosine = mx::transpose(
        mx::reshape(mx::cos(angles), {1, positions.shape(1), num_heads,
                                      expected / num_heads}),
        {0, 2, 1, 3});
    auto sine = mx::transpose(
        mx::reshape(mx::sin(angles), {1, positions.shape(1), num_heads,
                                      expected / num_heads}),
        {0, 2, 1, 3});
    return {std::move(cosine), std::move(sine)};
}

} // namespace tc::ltx_mlx

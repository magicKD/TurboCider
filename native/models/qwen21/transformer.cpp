#include "transformer.hpp"
#include "diagnostic_options.hpp"
#include "metal/qk_norm_rope.hpp"
#include "metal/qkv_projection.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace tc::qwen21 {
namespace {
Tensor rotate(const Tensor &x, const Tensor &cosine, const Tensor &sine) {
    auto shape = x.shape();
    auto paired_shape = shape;
    paired_shape.back() /= 2;
    paired_shape.push_back(2);
    auto paired = mx::reshape(mx::astype(x, mx::float32), paired_shape);
    // Single-output slices avoid MLX compiled multi-output sibling cycles
    // retaining captured, materialized weights (upstream mlx issue #3932).
    auto a = mx::squeeze(slice_axis(paired, -1, 0, 1), -1);
    auto b = mx::squeeze(slice_axis(paired, -1, 1, 2), -1);
    auto c = mx::reshape(cosine, {1, 1, shape[2], shape[3] / 2});
    auto s = mx::reshape(sine, {1, 1, shape[2], shape[3] / 2});
    return mx::astype(mx::reshape(mx::stack({a * c - b * s, a * s + b * c}, -1), shape), x.dtype());
}
Tensor layer_norm(const Tensor &x, float eps) {
    return mx::fast::layer_norm(x, {}, {}, eps);
}
Tensor gelu(const Tensor &x) {
    static auto compiled = mx::compile([](const std::vector<Tensor> &args) {
        const auto &a = args[0];
        auto scalar = [&](float v) { return Tensor(v, a.dtype()); };
        return std::vector<Tensor>{scalar(0.5f) * a *
            (scalar(1.f) + mx::tanh(scalar(0.7978845608028654f) *
                                  (a + scalar(0.044715f) * mx::power(a, scalar(3.f)))))};
    }, true);
    return compiled({x})[0];
}
}

Transformer::Transformer(const Weights &weights, TransformerConfig config,
                         const std::vector<Tensor> *fused_qkv)
    : weights_(weights), config_(config), fused_qkv_(fused_qkv) {
    require(!fused_qkv_ || fused_qkv_->size() == size_t(config_.layers),
            "Qwen21 fused QKV bank must cover every transformer layer");
    require(!fused_qkv_ || config_.epsilon == 1e-6f,
            "Qwen21 fused QKV probe requires the calibrated RMSNorm epsilon");
    metal_qk_rope_ = option_enabled(std::getenv("TURBOCIDER_QWEN21_METAL_QK_ROPE"));
    metal_qk_norm_rope_ = option_enabled(std::getenv("TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE"));
    const char *local_references = std::getenv("TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION");
    reference_local_attention_ = local_references && std::string(local_references) == "1" ? 1 :
                                 local_references && std::string(local_references) == "2" ? 2 :
                                 local_references && std::string(local_references) == "3" ? 3 : 0;
    require(!metal_qk_norm_rope_ || !metal_qk_rope_,
            "Qwen21 Q/K norm-RoPE fusion supersedes the paired RoPE kernel");
    profile_gpu_blocks_ = option_enabled(std::getenv("TURBOCIDER_QWEN21_PROFILE_GPU_BLOCKS"));
    profile_gpu_ops_ = option_enabled(std::getenv("TURBOCIDER_QWEN21_PROFILE_GPU_OPS"));
    profile_prefill_segments_ = option_enabled(std::getenv("TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS"));
    prefill_last_target_only_ = option_enabled(std::getenv(
        "TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC"));
    require(int(profile_gpu_blocks_) + int(profile_gpu_ops_) + int(profile_prefill_segments_) <= 1,
            "select only one Qwen21 GPU profiling mode");
    require(config.layers > 0 && config.heads > 0 && config.head_dim > 0 &&
            config.channels > 0 && config.context_dim > 0, "invalid Qwen21 transformer dimensions");
    int sum = 0;
    for (int axis : config.rope_axes) {
        require(axis > 0 && axis % 2 == 0, "Qwen21 RoPE axes must be positive and even");
        sum += axis;
    }
    require(sum == config.head_dim, "Qwen21 RoPE axes must sum to head dimension");
}

void Transformer::reset() {
    prefix_.clear();
    cached_ffn_.clear();
    db_prev_front_residual_.reset(); db_middle_residual_.reset();
    db_consecutive_steps_ = db_cached_steps_ = 0;
    cached_text_.reset();
    cached_references_.clear();
    prefill_tile_tails_.clear();
}

bool Transformer::prefix_matches(const Tensor &text, int height, int width,
                                 const std::vector<ReferenceLatents> &references) const {
    if (prefix_.size() != size_t(config_.layers) || !cached_text_ ||
        cached_text_->id() != text.id() || text_length_ != text.shape(1) ||
        height_ != height || width_ != width ||
        references.size() != cached_references_.size() ||
        references.size() != reference_geometry_.size()) return false;
    for (size_t i = 0; i < references.size(); ++i)
        if (references[i].latents.id() != cached_references_[i].id() ||
            !(references[i].geometry == reference_geometry_[i])) return false;
    return true;
}

Tensor Transformer::embedding(float timestep, mx::Dtype dtype) const {
    // Reference constructs these constants in NumPy double then casts to MLX
    // float32. Small frequency errors can cross a BF16 rounding boundary.
    static const Tensor frequencies = [] {
        std::vector<float> values(128);
        for (int i = 0; i < 128; ++i)
            values[i] = float(std::exp(-std::log(10000.0) * double(i) / 128.0));
        return Tensor(values.data(), {128}, mx::float32);
    }();
    auto times = Tensor(std::vector<float>{timestep, 0.f}.data(), {2, 1}, mx::float32);
    auto angles = (times * 1000.f) * frequencies;
    auto input = mx::astype(mx::concatenate({mx::cos(angles), mx::sin(angles)}, -1), dtype);
    return linear(silu(linear(input, weights_, "time_text_embed.timestep_embedder.linear_1")),
                  weights_, "time_text_embed.timestep_embedder.linear_2");
}

void Transformer::geometry(int text_length, int height, int width, const std::vector<ReferenceGeometry> &references) {
    if (cosine_ && text_length == text_length_ && height == height_ && width == width_ && references == reference_geometry_) return;
    auto sequence = make_sequence_geometry(text_length, height, width, references);
    reset();
    prefill_blocks_.clear();
    decode_blocks_.clear();
    capture_blocks_.clear();
    reuse_blocks_.clear();
    reuse_last16_blocks_.clear();
    half_reuse_blocks_.clear();
    text_length_ = text_length;
    height_ = height;
    width_ = width;
    reference_geometry_ = references;
    sequence_ = std::move(sequence);
    int tokens = int(sequence_.positions.size());
    std::vector<float> cos_values, sin_values;
    cos_values.reserve(size_t(tokens) * config_.head_dim / 2);
    sin_values.reserve(cos_values.capacity());
    for (int token = 0; token < tokens; ++token) {
        const auto &position = sequence_.positions[token];
        for (int axis = 0; axis < 3; ++axis) {
            int dim = config_.rope_axes[axis];
            for (int i = 0; i < dim; i += 2) {
                float omega = 1.f / std::pow(10000.f, float(i) / dim);
                float angle = float(position[axis]) * omega;
                cos_values.push_back(std::cos(angle));
                sin_values.push_back(std::sin(angle));
            }
        }
    }
    cosine_ = Tensor(cos_values.data(), {tokens, config_.head_dim / 2}, mx::float32);
    sine_ = Tensor(sin_values.data(), {tokens, config_.head_dim / 2}, mx::float32);
}

Tensor Transformer::forward(const Tensor &latents, const Tensor &text, float timestep,
                            int height, int width, bool cache_prefix,
                            std::unordered_map<std::string, Tensor> *trace,
                            const std::vector<ReferenceLatents> &references) {
    require(height > 0 && width > 0 && latents.ndim() == 3 && latents.shape(0) == 1 &&
            latents.shape(1) == height * width && latents.shape(2) == config_.channels,
            "Qwen21 latent shape must be [1,H*W,channels]");
    require(text.ndim() == 3 && text.shape(0) == 1 && text.shape(1) > 0 &&
            text.shape(2) == config_.context_dim && text.dtype() == latents.dtype(),
            "Qwen21 text shape/dtype mismatch");
    require(std::isfinite(timestep) && timestep >= 0.f && timestep <= 1.f,
            "Qwen21 timestep must be a normalized sigma in [0,1]");
    std::vector<ReferenceGeometry> reference_geometry;
    bool same_references = references.size() == cached_references_.size();
    for (size_t i = 0; i < references.size(); ++i) {
        const auto &r = references[i];
        require(r.latents.ndim() == 3 && r.latents.shape(0) == 1 && r.latents.dtype() == latents.dtype() &&
                r.latents.shape(2) == config_.channels && r.latents.shape(1) == int64_t(r.geometry.height) * r.geometry.width,
                "Qwen21 reference latent shape/dtype mismatch");
        reference_geometry.push_back(r.geometry);
        if (i >= cached_references_.size() || r.latents.id() != cached_references_[i].id()) same_references = false;
    }
    geometry(text.shape(1), height, width, reference_geometry);
    // MLX arrays are immutable values: identity binds the cache without a GPU
    // synchronization/hash of a many-megabyte conditioning tensor each step.
    if (!cache_prefix || !cached_text_ || cached_text_->id() != text.id() || !same_references) reset();
    bool reuse = cache_prefix && prefix_.size() == size_t(config_.layers);
    const bool split_mlp = reuse ? bool(decode_mlp_) : bool(prefill_mlp_);
    require(bool(stage_qkv_) == bool(project_qkv_),
            "Qwen21 QKV stage and projection callbacks must be paired");
    require(!plan_qkv_ || (project_qkv_ && observe_qkv_),
            "Qwen21 planned QKV needs paired projection and completed-block observer");
    require(!project_qkv_ || (!split_mlp && !fused_qkv_),
            "Qwen21 QKV tier needs its own GPU FFN and cannot combine with another QKV tier");
    require(!plan_mlp_ || bool(observe_mlp_), "Qwen21 planned MLP requires a completed-block observer");
    const bool half_reuse_ffn = reuse && ffn_cache_mode_ == FFNCacheMode::ReuseEvenAndCapture;
    const bool capture_ffn = reuse && (ffn_cache_mode_ == FFNCacheMode::Capture || half_reuse_ffn);
    const bool reuse_last16_ffn = reuse && ffn_cache_mode_ == FFNCacheMode::ReuseLast16;
    const bool reuse_ffn = reuse && (ffn_cache_mode_ == FFNCacheMode::Reuse ||
                                      reuse_last16_ffn || half_reuse_ffn);
    const bool db_decode = db_cache_enabled_ && reuse && !trace && db_step_ >= 1 &&
                           db_total_steps_ >= 20 && db_step_ < db_total_steps_;
    require(!db_cache_enabled_ || (config_.layers > db_front_blocks + db_back_blocks &&
                db_threshold_ > 0.f && db_threshold_ <= 0.5f &&
                db_max_consecutive_ >= 1 && db_max_consecutive_ <= 8),
            "Qwen21 DBCache geometry, threshold or consecutive skip bound is invalid");
    require(!db_decode || ffn_cache_mode_ == FFNCacheMode::Off,
            "Qwen21 DBCache cannot combine with step-FFN reuse");
    require(!(capture_ffn || reuse_ffn) || !trace,
            "Qwen21 FFN step cache is incompatible with tracing");
    if (reuse_ffn) require(cached_ffn_.size() == size_t(config_.layers),
                           "Qwen21 GPU FFN cache is missing a complete preceding decode step");
    std::vector<Tensor> previous_ffn;
    if (half_reuse_ffn) previous_ffn = std::move(cached_ffn_);
    if (capture_ffn) cached_ffn_.clear();
    require(!split_mlp || !trace, "Qwen21 MLP split trace is not supported");
    auto temb = embedding(timestep, latents.dtype());
    auto modulation = linear(silu(temb), weights_, "modulation.1");
    if (trace) { trace->emplace("temb", slice_axis(temb, 0, 0, 1)); trace->emplace("modulation", slice_axis(modulation, 0, 0, 1)); }
    auto mod_target = mx::split(mx::reshape(slice_axis(modulation, 0, 0, 1), {1, 1, 4 * config_.hidden()}), 4, -1);
    auto mods = mod_target;
    Tensor hidden = linear(latents, weights_, "img_in");
    const Tensor db_input = hidden;
    int prefix_length = sequence_.prefix_length;
    if (!reuse) {
        auto f = mx::astype(text, mx::float32);
        auto scale = mx::astype(weights_.at("txt_in.text_norm.weight"), mx::float32) + 1.f;
        auto normalized = mx::astype(f * mx::rsqrt(mx::mean(f * f, -1, true) + config_.epsilon) * scale, text.dtype());
        auto context = linear(gelu(linear(normalized, weights_, "txt_in.in_layer")), weights_, "txt_in.out_layer");
        std::vector<Tensor> parts;
        for (const auto &segment : sequence_.segments) {
            if (segment.causal)
                parts.push_back(slice_axis(context, 1, segment.text_start, segment.text_start + segment.end - segment.start));
            else if (segment.image_index < int(references.size()))
                parts.push_back(linear(references[segment.image_index].latents, weights_, "img_in"));
            else parts.push_back(hidden);
        }
        hidden = mx::concatenate(parts, 1);
        auto mod_prefix = mx::split(mx::reshape(slice_axis(modulation, 0, 1, 2), {1, 1, 4 * config_.hidden()}), 4, -1);
        for (int i = 0; i < 4; ++i)
            mods[i] = mx::concatenate({mx::broadcast_to(mod_prefix[i], {1, prefix_length, config_.hidden()}),
                                      mx::broadcast_to(mod_target[i], {1, height * width, config_.hidden()})}, 1);
    }
    auto cosine = reuse ? slice_axis(*cosine_, 0, prefix_length, cosine_->shape(0)) : *cosine_;
    auto sine = reuse ? slice_axis(*sine_, 0, prefix_length, sine_->shape(0)) : *sine_;
    if (trace) {
        trace->emplace("input", hidden);
        trace->emplace("cosine", cosine);
        trace->emplace("sine", sine);
        for (int i = 0; i < 4; ++i) trace->emplace("mod" + std::to_string(i), mods[i]);
    }
    std::vector<KV> new_prefix;
    if (trace || release_layer_) { prefill_blocks_.clear(); decode_blocks_.clear(); capture_blocks_.clear();
                 reuse_blocks_.clear(); reuse_last16_blocks_.clear(); half_reuse_blocks_.clear(); }
    // Decode modulation is identical across blocks. Materialize its gate once
    // per forward, and fuse the split-path residual's elementwise operations.
    // The full-GPU block retains its existing compiled arithmetic.
    std::optional<Tensor> split_gate;
    if (split_mlp) split_gate = mx::tanh(mods[3]);
    static auto split_residual = mx::compile([](const std::vector<Tensor> &a) {
        return std::vector<Tensor>{a[0] + a[1] * a[2]};
    });
    std::optional<Tensor> db_middle_input;
    bool db_skip = false;
    for (int i = 0; i < config_.layers; ++i) {
        if (db_decode && i == db_front_blocks) {
            // Match cache-dit's relative L1 change of the front-block
            // residual. Only a completed earlier decode step can supply the
            // middle-block residual; never reuse first-step prefix outputs.
            auto front_residual = mx::astype(hidden - db_input, mx::float32);
            if (db_prev_front_residual_ && db_middle_residual_ &&
                db_step_ >= 8 && db_step_ < db_total_steps_ - 1 &&
                db_consecutive_steps_ < db_max_consecutive_) {
                auto numerator = mx::sum(mx::abs(front_residual - *db_prev_front_residual_));
                auto denominator = mx::sum(mx::abs(*db_prev_front_residual_)) + 1e-6f;
                const float change = (numerator / denominator).item<float>();
                db_skip = std::isfinite(change) && change < db_threshold_;
            }
            db_prev_front_residual_ = mx::copy(front_residual);
            mx::eval(*db_prev_front_residual_);
            if (db_skip) {
                hidden = hidden + *db_middle_residual_;
                mx::eval(hidden);
                ++db_cached_steps_;
                ++db_consecutive_steps_;
                i = config_.layers - db_back_blocks - 1;
                continue;
            }
            db_middle_input = hidden;
            db_consecutive_steps_ = 0;
        }
        const bool target_only_block = prefill_last_target_only_ && !reuse && !trace &&
                                       i == config_.layers - 1;
        // On the final decode step the immediately preceding hybrid FFN
        // output can be reused without touching Core ML's shared backing.
        // The capture path below owns an independent copy per layer.
        const bool reuse_ffn_block = reuse_ffn && (!half_reuse_ffn || i % 2 == 0) &&
            (!reuse_last16_ffn || i >= config_.layers / 2);
        const bool split_candidate = split_mlp &&
            (reuse ? i >= decode_first_block_ : i >= prefill_first_block_) && !reuse_ffn_block;
        const int ffn_rows = target_only_block ? height * width : hidden.shape(1);
        const auto plan = split_candidate ? (plan_mlp_ ? plan_mlp_(i, ffn_rows) : MLPPlan::Split)
                                          : MLPPlan::Gpu;
        const bool split_this = plan == MLPPlan::Split || plan == MLPPlan::SplitUntimed;
        const int qkv_rows = hidden.shape(1);
        const auto qkv_plan = project_qkv_ && !trace && plan_qkv_ ?
            plan_qkv_(i, qkv_rows) : QKVPlan::Hybrid;
        const bool external_qkv = bool(project_qkv_) && !trace &&
            (qkv_plan == QKVPlan::Hybrid || qkv_plan == QKVPlan::HybridTimed);
        const bool measured_qkv = bool(project_qkv_) && !trace &&
            (qkv_plan == QKVPlan::HybridTimed || qkv_plan == QKVPlan::GpuProbe);
        const bool measured_block = plan_mlp_ && (plan == MLPPlan::Split || plan == MLPPlan::GpuProbe);
        auto &functions = half_reuse_ffn ? half_reuse_blocks_ : reuse_last16_ffn ? reuse_last16_blocks_ :
                               reuse_ffn ? reuse_blocks_ : capture_ffn ? capture_blocks_
                               : reuse ? decode_blocks_ : prefill_blocks_;
        if (functions.size() <= size_t(i)) functions.resize(size_t(i) + 1);
        auto &function = functions[i][(split_this ? 1 : 0) + (external_qkv ? 2 : 0)];
        if (!function) {
            const bool profile_ops = profile_gpu_ops_ && !split_this && i == 0;
            const bool profile_segments = profile_prefill_segments_ && !reuse && !split_this && i == 0;
            auto block = [this, i, reuse, prefix_length, split_this, external_qkv, capture_ffn,
                          target_only_block,
                          reuse_ffn_block,
                          profile_ops, profile_segments,
                          metal_rope = metal_qk_rope_, fused_norm_rope = metal_qk_norm_rope_,
                          local_references = reference_local_attention_,
                          tracing = trace != nullptr](const std::vector<Tensor> &args) {
                // The last prefill block exports only target queries. Without
                // a fused QKV/RoPE kernel, avoid projecting and rotating Q for
                // the discarded text/reference rows; K/V still cover every
                // row for target attention and the next-step prefix cache.
                const bool short_q = target_only_block && !fused_qkv_ && !external_qkv &&
                                     !fused_norm_rope && !metal_rope;
                auto mark_start = Clock::now();
                if (profile_ops) {
                    mx::eval(args);
                    mark_start = Clock::now();
                }
                auto mark = [&](const char *phase, std::initializer_list<Tensor> values) {
                    if (!profile_ops) return;
                    mx::eval(std::vector<Tensor>(values));
                    std::cerr << "{\"qwen21_gpu_op\":\"" << phase << "\",\"block\":" << i
                              << ",\"phase\":\"" << (reuse ? "decode" : "prefill")
                              << "\",\"seconds\":" << std::chrono::duration<double>(Clock::now() - mark_start).count()
                              << "}" << std::endl;
                    mark_start = Clock::now();
                };
                auto hidden = args[0];
                std::vector<Tensor> mods(args.begin() + 1, args.begin() + 5);
                const auto &cosine = args[5], &sine = args[6];
                std::string p = "transformer_blocks." + std::to_string(i);
                auto input = layer_norm(hidden, config_.epsilon) * (Tensor(1.f, hidden.dtype()) + mods[0]);
                auto attention_input = input;
                mark("attention_input_norm", {input});
                Tensor q(0.f), k(0.f), v(0.f);
                if (external_qkv) {
                    q = args[args.size() - 3];
                    k = args[args.size() - 2];
                    v = heads(args.back(), config_.heads, config_.head_dim);
                } else if (fused_qkv_) {
                    auto all = metal::project_prepare_qkv(input, fused_qkv_->at(i),
                        weights_.at(p + ".attn.norm_q.weight"),
                        weights_.at(p + ".attn.norm_k.weight"),
                        cosine, sine, 32);
                    q = all[0]; k = all[1]; v = all[2];
                } else {
                    q = linear(short_q ? slice_axis(input, 1, prefix_length, input.shape(1)) : input,
                               weights_, p + ".attn.to_q");
                    k = linear(input, weights_, p + ".attn.to_k");
                    v = heads(linear(input, weights_, p + ".attn.to_v"), config_.heads, config_.head_dim);
                }
                mark("qkv_projection", {q, k, v});
                if (fused_qkv_) {
                    // Q/K normalization and RoPE are part of the projection.
                } else if (fused_norm_rope) {
                    auto pair = metal::prepare_qk(q, k,
                        weights_.at(p + ".attn.norm_q.weight"), weights_.at(p + ".attn.norm_k.weight"),
                        cosine, sine, config_.epsilon);
                    q = pair[0]; k = pair[1];
                } else {
                    q = mx::fast::rms_norm(heads(q, config_.heads, config_.head_dim),
                                           weights_.at(p + ".attn.norm_q.weight"), config_.epsilon);
                    k = mx::fast::rms_norm(heads(k, config_.heads, config_.head_dim),
                                           weights_.at(p + ".attn.norm_k.weight"), config_.epsilon);
                }
                if (!fused_qkv_ && !fused_norm_rope && metal_rope) {
                    auto pair = rope_pairs_pair(q, k, cosine, sine);
                    q = pair[0]; k = pair[1];
                } else if (!fused_qkv_ && !fused_norm_rope) {
                    q = rotate(q,
                        short_q ? slice_axis(cosine, 0, prefix_length, cosine.shape(0)) : cosine,
                        short_q ? slice_axis(sine, 0, prefix_length, sine.shape(0)) : sine);
                    k = rotate(k, cosine, sine);
                }
                mark("qk_norm_rope", {q, k});
                Tensor output = hidden;
                auto pk = k, pv = v;
                if (reuse) {
                    k = mx::concatenate({args[7], k}, 2);
                    v = mx::concatenate({args[8], v}, 2);
                    output = attend(q, k, v);
                } else if (target_only_block) {
                    // This is the last block: no later layer consumes its
                    // reference/text *outputs*. Cache their K/V as usual, but
                    // evaluate attention and FFN only for target queries.
                    pk = slice_axis(k, 2, 0, prefix_length);
                    pv = slice_axis(v, 2, 0, prefix_length);
                    output = attend(short_q ? q : slice_axis(q, 2, prefix_length, q.shape(2)), k, v);
                } else {
                    pk = slice_axis(k, 2, 0, prefix_length);
                    pv = slice_axis(v, 2, 0, prefix_length);
                    std::vector<Tensor> segments;
                    if (profile_segments) mx::eval(q, k, v);
                    for (const auto &segment : sequence_.segments) {
                        auto segment_start = profile_segments ? Clock::now() : Clock::time_point{};
                        auto keys = slice_axis(k, 2, 0, segment.end);
                        auto values = slice_axis(v, 2, 0, segment.end);
                        if (local_references && !segment.causal && segment.image_index > 0 &&
                            (local_references == 1 ||
                             (local_references == 2 &&
                              segment.image_index + 1 == int(reference_geometry_.size())) ||
                             (local_references == 3 && i >= config_.layers / 2)) &&
                            segment.image_index < int(reference_geometry_.size())) {
                            // Keep all earlier text and this reference's own
                            // image tokens, but avoid scanning earlier images
                            // again for each later reference. Target queries
                            // below still attend to the complete prefix.
                            std::vector<Tensor> key_parts, value_parts;
                            for (const auto &prior : sequence_.segments) {
                                if (prior.start >= segment.start) break;
                                if (!prior.causal) continue;
                                key_parts.push_back(slice_axis(k, 2, prior.start, prior.end));
                                value_parts.push_back(slice_axis(v, 2, prior.start, prior.end));
                            }
                            key_parts.push_back(slice_axis(k, 2, segment.start, segment.end));
                            value_parts.push_back(slice_axis(v, 2, segment.start, segment.end));
                            keys = mx::concatenate(key_parts, 2);
                            values = mx::concatenate(value_parts, 2);
                        }
                        auto attended = attend(slice_axis(q, 2, segment.start, segment.end),
                            keys, values, false, {}, false, segment.causal ? "causal" : "");
                        if (profile_segments) {
                            mx::eval(attended);
                            std::cerr << "{\"qwen21_prefill_attention_segment\":" << segments.size()
                                      << ",\"kind\":\"" << (segment.causal ? "text" :
                                         segment.image_index == int(reference_geometry_.size()) ? "target" : "reference")
                                      << "\",\"query_rows\":" << segment.end - segment.start
                                      << ",\"key_rows\":" << keys.shape(2)
                                      << ",\"seconds\":" << std::chrono::duration<double>(Clock::now() - segment_start).count()
                                      << "}" << std::endl;
                        }
                        segments.push_back(std::move(attended));
                    }
                    output = mx::concatenate(segments, 1);
                }
                mark("attention", {output});
                if (target_only_block) {
                    hidden = slice_axis(hidden, 1, prefix_length, hidden.shape(1));
                    for (int gate = 1; gate < 4; ++gate)
                        mods[gate] = slice_axis(mods[gate], 1, prefix_length, mods[gate].shape(1));
                }
                auto projected = linear(output, weights_, p + ".attn.to_out.0");
                mark("attention_output_projection", {projected});
                hidden = hidden + mx::tanh(mods[1]) * projected;
                auto after_attention = hidden;
                input = layer_norm(hidden, config_.epsilon) * (Tensor(1.f, hidden.dtype()) + mods[2]);
                mark("attention_residual_ffn_input", {input});
                if (split_this) return reuse ? std::vector<Tensor>{hidden, input} :
                    std::vector<Tensor>{hidden, input, pk, pv};
                Tensor ff = input, feed = input;
                if (reuse_ffn_block) {
                    feed = args[9];
                } else {
                    if (weights_.has(p + ".img_mlp.gate_up.weight")) {
                        auto gate_up = linear(input, weights_, p + ".img_mlp.gate_up");
                        const int half = gate_up.shape(-1) / 2;
                        ff = silu(slice_axis(gate_up, -1, 0, half)) *
                             slice_axis(gate_up, -1, half, gate_up.shape(-1));
                    } else {
                        ff = silu(linear(input, weights_, p + ".img_mlp.gate_layer")) * linear(input, weights_, p + ".img_mlp.proj");
                    }
                    feed = linear(ff, weights_, p + ".img_mlp.out");
                }
                mark("ffn_gate_up", {ff});
                hidden = hidden + mx::tanh(mods[3]) * feed;
                mark("ffn_down_residual", {hidden});
                if (tracing) return std::vector<Tensor>{hidden, pk, pv, attention_input, q, k, v, output, projected, after_attention, input, ff};
                if (capture_ffn) return std::vector<Tensor>{hidden, feed};
                return reuse ? std::vector<Tensor>{hidden} : std::vector<Tensor>{hidden, pk, pv};
            };
            auto compiled = profile_ops || profile_segments ? BlockFunction(block) : mx::compile(block);
            function = std::move(compiled);
        }
        std::vector<Tensor> args{hidden, mods[0], mods[1], mods[2], mods[3], cosine, sine};
        if (reuse) {
            args.push_back(prefix_[i].key);
            args.push_back(prefix_[i].value);
        }
        if (reuse_ffn_block)
            args.push_back(half_reuse_ffn ? previous_ffn[i] : cached_ffn_[i]);
        // Probe complete blocks from the same materialized upstream inputs,
        // including staging, QKV join, attention and GPU FFN. The GPU probe
        // uses the ordinary fused/lazy block rather than an external QKV ABI.
        if (measured_qkv) mx::eval(args);
        auto qkv_start = measured_qkv ? Clock::now() : Clock::time_point{};
        if (external_qkv) {
            const int rows = qkv_rows;
            stage_qkv_(i, rows); // may overlap the GPU attention-input norm
            auto normed = layer_norm(hidden, config_.epsilon) *
                          (Tensor(1.f, hidden.dtype()) + mods[0]);
            auto projected = project_qkv_(i, normed);
            require(projected.shape() == mx::Shape{1, rows, 3 * config_.hidden()} &&
                        projected.dtype() == hidden.dtype(),
                    "Qwen21 external QKV projection shape/dtype mismatch");
            auto three = mx::split(projected, 3, -1);
            args.insert(args.end(), three.begin(), three.end());
        }
        // Diagnostic only: force each pure-GPU block boundary so the elapsed
        // time can be attributed to that block. This destroys normal lazy
        // scheduling and must never be used as a production speed benchmark.
        const bool profile_block = profile_gpu_blocks_ && !split_mlp;
        if (profile_block) mx::eval(args);
        auto block_start = profile_block ? Clock::now() : Clock::time_point{};
        // Exclude ALL prior work (also request modulation/input preparation)
        // from BOTH measured routes. Ordinary full GPU blocks remain lazy.
        if (measured_block) mx::eval(args);
        auto measured_start = measured_block ? Clock::now() : Clock::time_point{};
        if (split_this && stage_mlp_) stage_mlp_(i, ffn_rows);
        auto outputs = function(args);
        if (profile_block) {
            mx::eval(outputs);
            std::cerr << "{\"qwen21_gpu_block\":" << i
                      << ",\"phase\":\"" << (reuse ? "decode" : "prefill")
                      << "\",\"seconds\":" << std::chrono::duration<double>(Clock::now() - block_start).count()
                      << "}" << std::endl;
        }
        hidden = outputs[0];
        if (capture_ffn && !split_this) cached_ffn_.push_back(outputs[1]);
        if (split_this) {
            Tensor feed = outputs[1];
            if (!reuse && capture_tile_tails_ && cache_prefix &&
                outputs[1].shape(1) > height * width) {
                const int tail_rows = prefix_length % (height * width);
                if (tail_rows) {
                    if (prefill_tile_tails_.empty()) prefill_tile_tails_.resize(config_.layers);
                    auto tail = mx::copy(slice_axis(outputs[1], 1,
                        prefix_length - tail_rows, prefix_length));
                    mx::eval(tail); // own only the unfinished prefix tile, not all prefill rows
                    prefill_tile_tails_[i] = std::move(tail);
                }
            }
            feed = reuse ? decode_mlp_(i, feed) : prefill_mlp_(i, feed);
            if (capture_ffn) {
                // HybridMLP returns a lazy view of its reusable Core ML
                // output buffer. Own and finish this feed *before* the next
                // layer can overwrite the buffer; a plain Tensor assignment
                // would silently poison the final-step FFN cache.
                auto owned = mx::copy(feed);
                mx::eval(owned);
                cached_ffn_.push_back(owned);
            }
            const auto gate = target_only_block
                ? slice_axis(*split_gate, 1, prefix_length, split_gate->shape(1)) : *split_gate;
            hidden = split_residual({hidden, gate, feed})[0];
            // Frozen callbacks can borrow reusable Core ML backing and must
            // finish the residual here. Runtime's untimed callback instead
            // returns owned/evaluated output, so its residual can stay lazy.
            if (plan != MLPPlan::SplitUntimed) mx::eval(hidden);
        }
        if (measured_block) {
            if (!split_this) mx::eval(outputs);
            observe_mlp_(i, ffn_rows, std::chrono::duration<double>(Clock::now() - measured_start).count());
        }
        if (measured_qkv) {
            mx::eval(outputs);
            observe_qkv_(i, qkv_rows, std::chrono::duration<double>(Clock::now() - qkv_start).count());
        }
        if (trace) {
            trace->emplace("block" + std::to_string(i), hidden);
            // Offline calibration uses the *actual* post-attention, modulated
            // FFN input; approximating it from the block output misses the
            // per-step adaptive norm and creates invalid A8 statistics.
            trace->emplace("mlp_input" + std::to_string(i), outputs[10]);
        }
        if (trace && i == 0) {
            const char *names[] = {"attention_input", "q", "k", "v", "attention", "projected", "after_attention", "mlp_input", "ff"};
            for (int j = 0; j < 9; ++j) trace->emplace(names[j], outputs[3 + j]);
        }
        if (!reuse && cache_prefix) new_prefix.push_back(split_this ? KV{outputs[2], outputs[3]} :
                                                        KV{outputs[1], outputs[2]});
        if (release_layer_) {
            // Own only prefix rows, not a view retaining all target K/V.
            // Synchronize every result before releasing compiled constants.
            mx::eval(outputs);
            if (!reuse && cache_prefix) {
                auto &kv = new_prefix.back();
                kv.key = mx::copy(kv.key); kv.value = mx::copy(kv.value);
                mx::eval(kv.key, kv.value);
            }
            functions[i] = {};
            release_layer_("transformer_blocks." + std::to_string(i) + ".");
        }
        if (db_decode && i == config_.layers - db_back_blocks - 1) {
            require(db_middle_input.has_value(), "Qwen21 DBCache middle-block input is missing");
            db_middle_residual_ = mx::copy(hidden - *db_middle_input);
            // HybridMLP's Core ML output backing is reused at the next
            // prediction: own and finish the entire aggregate residual now.
            mx::eval(*db_middle_residual_);
        }
    }
    if (!reuse && !(prefill_last_target_only_ && !trace))
        hidden = slice_axis(hidden, 1, prefix_length, hidden.shape(1));
    auto out_scale = mx::reshape(slice_axis(linear(silu(temb), weights_, "norm_out.linear"), 0, 0, 1), {1, 1, config_.hidden()});
    auto result = linear(layer_norm(hidden, config_.epsilon) * (Tensor(1.f, hidden.dtype()) + out_scale), weights_, "proj_out");
    if (cache_prefix && !reuse) {
        prefix_ = std::move(new_prefix);
        cached_text_ = text;
        for (const auto &r : references) cached_references_.push_back(r.latents);
    }
    if (capture_ffn) mx::eval(cached_ffn_);
    if (trace) { prefill_blocks_.clear(); decode_blocks_.clear(); capture_blocks_.clear();
                 reuse_blocks_.clear(); reuse_last16_blocks_.clear(); half_reuse_blocks_.clear(); }
    return result;
}
} // namespace tc::qwen21

#include "dit.hpp"

#include <algorithm>
#include <cmath>

namespace tc::h3_mlx {
namespace {
Tensor indices(const std::vector<int32_t> &values) {
    return Tensor(values.data(), {int(values.size())}, mx::int32);
}

Tensor silu_local(const Tensor &value) {
    return value * mx::sigmoid(value);
}

Tensor modulate(const Tensor &value, const Tensor &scale, const Tensor &shift) {
    // Match FastVideo's shapeless compiled AdaLN graph.  On BF16 activations,
    // expressing multiply/add outside the compiled boundary changes the
    // intermediate rounding and the error compounds across 50 blocks.
    static auto graph = mx::compile([](const std::vector<Tensor> &args) {
        return std::vector<Tensor>{args[0] * args[1] + args[2]};
    }, true);
    return graph({value, scale, shift})[0];
}
} // namespace

Tensor DiT::cast_for(const Tensor &value, const std::string &prefix) const {
    return mx::astype(value, weights_.projection_dtype(prefix));
}

Tensor DiT::rms(const Tensor &value, const std::string &weight, float epsilon) const {
    return mx::fast::rms_norm(value, weights_.at(weight), epsilon);
}

std::pair<Tensor, Tensor> DiT::rope(const PackedLayout &layout) const {
    const auto &config = weights_.config();
    // Keep float64 only in the CPU geometry builder, matching FastVideo's
    // NumPy contract. MLX/Metal does not support float64 arrays, so cross the
    // backend boundary with an explicitly converted FP32 host buffer.
    std::vector<float> fp32_positions(layout.position_ids.begin(),
                                      layout.position_ids.end());
    auto positions = Tensor(fp32_positions.data(),
        {layout.sequence_length, 3}, mx::float32);
    auto frequency = 1.f / mx::power(
        Tensor(config.rope_theta, mx::float32),
        mx::arange(0, 2 * config.rope_frequency_dim, 2, mx::float32) /
            float(2 * config.rope_frequency_dim));
    std::vector<Tensor> axes;
    for (int axis = 0; axis < 3; ++axis)
        axes.push_back(slice_axis(positions, 1, axis, axis + 1) * frequency);
    auto angles = mx::concatenate(axes, -1);
    angles = mx::concatenate({angles, angles}, -1);
    return {mx::cos(angles), mx::sin(angles)};
}

Tensor DiT::apply_rotary(const Tensor &value, const Tensor &cosine,
                         const Tensor &sine) const {
    const int rotary_dim = cosine.shape(-1);
    auto rotary = mx::astype(slice_axis(value, -1, 0, rotary_dim), mx::float32);
    auto pass = slice_axis(value, -1, rotary_dim, value.shape(-1));
    auto halves = mx::split(rotary, 2, -1);
    auto rotated = mx::concatenate({-halves[1], halves[0]}, -1);
    auto cos = mx::expand_dims(mx::astype(cosine, mx::float32), 1);
    auto sin = mx::expand_dims(mx::astype(sine, mx::float32), 1);
    auto output = mx::astype(rotary * cos + rotated * sin, value.dtype());
    return mx::concatenate({output, pass}, -1);
}

Tensor DiT::attention(
    const Tensor &input, const std::string &prefix,
    const std::optional<std::pair<Tensor, Tensor>> &rotary,
    DiTDebugCapture *debug,
    const VSAGeometry *vsa_geometry,
    double vsa_sparsity,
    VSAPrefixMode vsa_prefix_mode,
    VSAImplementation vsa_implementation,
    const Tensor *gate_compress,
    VSAStats *vsa_stats) const {
    const auto &config = weights_.config();
    const int sequence = input.shape(0);
    auto q = mx::reshape(weights_.linear(cast_for(input, prefix + ".attn.to_q"),
                                         prefix + ".attn.to_q"),
                         {sequence, config.num_heads, config.head_dim});
    auto k = mx::reshape(weights_.linear(cast_for(input, prefix + ".attn.to_k"),
                                         prefix + ".attn.to_k"),
                         {sequence, config.num_heads, config.head_dim});
    auto v = mx::reshape(weights_.linear(cast_for(input, prefix + ".attn.to_v"),
                                         prefix + ".attn.to_v"),
                         {sequence, config.num_heads, config.head_dim});
    q = rms(q, prefix + ".attn.norm_q.weight", config.qk_norm_epsilon);
    k = rms(k, prefix + ".attn.norm_k.weight", config.qk_norm_epsilon);
    if (rotary) {
        q = apply_rotary(q, rotary->first, rotary->second);
        k = apply_rotary(k, rotary->first, rotary->second);
    }
    if (debug) {
        debug->first_query = q;
        debug->first_key = k;
        debug->first_value = v;
        mx::eval(*debug->first_query, *debug->first_key, *debug->first_value);
    }
    std::optional<Tensor> attended_rows;
    if (vsa_geometry && rotary && (vsa_sparsity > 0.f || gate_compress)) {
        auto attended = vsa_attention(q, k, v, *vsa_geometry, vsa_sparsity,
                                      vsa_prefix_mode, vsa_implementation,
                                      gate_compress, vsa_stats);
        if (debug)
            debug->first_attended = attended;
        attended_rows = mx::reshape(attended,
                                    {sequence, config.num_heads * config.head_dim});
    } else {
        q = mx::expand_dims(mx::transpose(q, {1, 0, 2}), 0);
        k = mx::expand_dims(mx::transpose(k, {1, 0, 2}), 0);
        v = mx::expand_dims(mx::transpose(v, {1, 0, 2}), 0);
        auto attended = mx::fast::scaled_dot_product_attention(
            q, k, v, 1.f / std::sqrt(float(config.head_dim)));
        if (debug)
            debug->first_attended = mx::reshape(
                mx::transpose(attended, {0, 2, 1, 3}),
                {sequence, config.num_heads, config.head_dim});
        attended_rows = mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                    {sequence, config.num_heads * config.head_dim});
    }
    require(attended_rows.has_value(), "H3 attention did not produce output");
    return weights_.linear(cast_for(*attended_rows, prefix + ".attn.to_out.0"),
                           prefix + ".attn.to_out.0");
}

Tensor DiT::feed_forward(const Tensor &input, const std::string &prefix) const {
    auto hidden = weights_.linear(cast_for(input, prefix + ".ff.net.0.proj"),
                                  prefix + ".ff.net.0.proj");
    auto halves = mx::split(hidden, 2, -1);
    return weights_.linear(cast_for(halves[0] * silu_local(halves[1]),
                                    prefix + ".ff.net.2"),
                           prefix + ".ff.net.2");
}

Tensor DiT::refine_text(const Tensor &text) const {
    const auto &config = weights_.config();
    auto hidden = weights_.linear(cast_for(text, "context_embedder"), "context_embedder");
    for (int index = 0; index < config.refiner_layers; ++index) {
        const auto prefix = "refiner." + std::to_string(index);
        auto normalized = rms(hidden, prefix + ".norm1.weight", config.norm_epsilon);
        hidden = hidden + attention(normalized, prefix, std::nullopt);
        normalized = rms(hidden, prefix + ".norm2.weight", config.norm_epsilon);
        hidden = hidden + feed_forward(normalized, prefix);
    }
    return rms(hidden, "token_refiner.final_norm.weight", config.final_norm_epsilon);
}

Tensor DiT::block(const Tensor &input, int index, const Tensor &adaln_indices,
                  const Tensor &cosine, const Tensor &sine,
                  int step_index, const VSAConfig &vsa_config,
                  const VSAGeometry *vsa_geometry, VSAStats *vsa_stats,
                  DiTDebugCapture *debug) const {
    const auto &config = weights_.config();
    const auto prefix = "blocks." + std::to_string(index);
    std::array<Tensor, 6> tables = {
        weights_.at("__adaln_cache.block." + std::to_string(index) + ".table.0"),
        weights_.at("__adaln_cache.block." + std::to_string(index) + ".table.1"),
        weights_.at("__adaln_cache.block." + std::to_string(index) + ".table.2"),
        weights_.at("__adaln_cache.block." + std::to_string(index) + ".table.3"),
        weights_.at("__adaln_cache.block." + std::to_string(index) + ".table.4"),
        weights_.at("__adaln_cache.block." + std::to_string(index) + ".table.5"),
    };
    auto gather = [&](int table) { return mx::take(tables[table], adaln_indices, 0); };

    auto normalized = rms(input, prefix + ".norm1.weight", config.norm_epsilon);
    if (debug) debug->first_norm1 = normalized;
    normalized = modulate(normalized, Tensor(1.f, normalized.dtype()) + gather(1), gather(0));
    if (debug) debug->first_modulated1 = normalized;
    std::optional<Tensor> gate;
    const auto gate_prefix = prefix + ".attn.to_gate_compress";
    if (vsa_geometry && vsa_config.gate_compress &&
        weights_.has(gate_prefix + ".weight"))
        gate = mx::reshape(weights_.linear(cast_for(normalized, gate_prefix),
                                           gate_prefix),
                           {normalized.shape(0), config.num_heads, config.head_dim});
    if (debug && gate)
        debug->first_gate = *gate;
    auto attended = attention(
        normalized, prefix, {{cosine, sine}}, debug, vsa_geometry,
        vsa_config.layer_sparsity(index, step_index), vsa_config.prefix_mode,
        vsa_config.implementation, gate ? &*gate : nullptr, vsa_stats);
    if (debug) debug->first_attention = attended;
    auto hidden = input + gather(2) * attended;
    if (debug) debug->first_after_attention = hidden;
    normalized = rms(hidden, prefix + ".norm2.weight", config.norm_epsilon);
    if (debug) debug->first_norm2 = normalized;
    normalized = modulate(normalized, Tensor(1.f, normalized.dtype()) + gather(4), gather(3));
    if (debug) debug->first_modulated2 = normalized;
    auto fed = feed_forward(normalized, prefix);
    if (debug) {
        debug->first_feed_forward = fed;
        mx::eval(*debug->first_norm1, *debug->first_modulated1,
                 *debug->first_attention, *debug->first_after_attention,
                 *debug->first_norm2, *debug->first_modulated2,
                 *debug->first_feed_forward);
    }
    return hidden + gather(5) * fed;
}

DiTOutput DiT::forward(const Tensor &video_rows, const Tensor &audio_rows,
                       const Tensor &text_rows, const PackedLayout &layout,
                       const RowTimesteps &step, const Event &event,
                       std::atomic<bool> &cancelled,
                       int step_index, const VSAConfig &vsa_config,
                       VSAStats *vsa_stats,
                       DiTDebugCapture *debug) const {
    const auto &config = weights_.config();
    require(video_rows.ndim() == 2 && video_rows.shape(0) == int(layout.video_indices.size()) &&
                video_rows.shape(1) == config.patch_dim(),
            "invalid H3 MLX video row geometry");
    require(audio_rows.ndim() == 2 && audio_rows.shape(0) == int(layout.audio_indices.size()) &&
                audio_rows.shape(1) == config.audio_latent_channels,
            "invalid H3 MLX audio row geometry");
    require(text_rows.ndim() == 2 && text_rows.shape(0) == int(layout.text_indices.size()) &&
                text_rows.shape(1) == config.text_dim,
            "invalid H3 MLX text row geometry");
    require(step.inverse.size() == size_t(layout.sequence_length),
            "invalid H3 MLX row timestep map");
    vsa_config.validate(config.num_layers);
    require(!vsa_config.enabled || weights_.identity().vsa_capable,
            "H3 VSA requires a checkpoint with 50 gate-compress matrices");

    std::optional<VSAGeometry> vsa_geometry;
    if (vsa_config.enabled)
        vsa_geometry = build_vsa_geometry(vsa_prefix_segments(layout),
                                          vsa_video_shape(layout, config.patch_size),
                                          vsa_config.tile_size);

    auto rotary = rope(layout);
    auto video = weights_.linear(cast_for(video_rows, "proj_in"), "proj_in");
    auto audio = weights_.linear(cast_for(audio_rows, "audio_proj_in"), "audio_proj_in");
    auto text = refine_text(text_rows);
    if (debug) {
        debug->video_embed = video;
        debug->audio_embed = audio;
        debug->text_embed = text;
        mx::eval(*debug->video_embed, *debug->audio_embed, *debug->text_embed);
    }
    auto packed = mx::concatenate({text, mx::astype(audio, text.dtype()),
                                   mx::astype(video, text.dtype())}, 0);
    require(packed.shape(0) == layout.sequence_length,
            "H3 MLX packed row order differs from the T2V layout");
    if (debug) {
        debug->packed_embed = packed;
        mx::eval(*debug->packed_embed);
    }

    std::vector<int32_t> adaln(layout.sequence_length), row_positions(layout.sequence_length);
    const auto &cache = weights_.identity().adaln_timesteps;
    for (int row = 0; row < layout.sequence_length; ++row) {
        float value = step.unique.at(step.inverse[row]);
        auto found = std::lower_bound(cache.begin(), cache.end(), value);
        require(found != cache.end() && std::abs(*found - value) <= 1e-6f,
                "H3 MLX step timestep is absent from the checkpoint AdaLN cache");
        int position = int(found - cache.begin());
        row_positions[row] = position;
        adaln[row] = position * modality_count + layout.token_tags[row];
    }
    auto adaln_indices = indices(adaln);
    for (int index = 0; index < config.num_layers; ++index) {
        checkpoint(cancelled);
        event("h3_mlx_dit_block", index, config.num_layers);
        packed = block(packed, index, adaln_indices, rotary.first, rotary.second,
                       step_index, vsa_config,
                       vsa_geometry ? &*vsa_geometry : nullptr, vsa_stats,
                       debug && index == 0 ? debug : nullptr);
        if (debug && index == 0) {
            debug->first_block = packed;
            mx::eval(*debug->first_block);
        }
        // Bound the graph and activation lifetime on 36 GiB Macs.
        mx::eval(packed);
    }

    auto rows = indices(row_positions);
    auto normalized = rms(packed, "norm_out.norm.weight", config.final_norm_epsilon);
    auto shift = mx::take(weights_.at("__adaln_cache.norm_out_shift"), rows, 0);
    auto scale = mx::take(weights_.at("__adaln_cache.norm_out_scale"), rows, 0);
    normalized = modulate(normalized, Tensor(1.f, normalized.dtype()) + scale, shift);
    auto video_indices = indices(layout.video_indices);
    auto audio_indices = indices(layout.audio_indices);
    auto video_output = weights_.linear(
        cast_for(mx::take(normalized, video_indices, 0), "proj_out"), "proj_out");
    auto audio_output = weights_.linear(
        cast_for(mx::take(normalized, audio_indices, 0), "audio_proj_out"),
        "audio_proj_out");
    mx::eval({video_output, audio_output});
    checkpoint(cancelled);
    event("h3_mlx_dit_block", config.num_layers, config.num_layers);
    return {std::move(video_output), std::move(audio_output)};
}

Tensor scheduler_step(const Tensor &sample, const Tensor &velocity,
                      const Scheduler &scheduler, int step_index) {
    require(step_index >= 0 && step_index + 1 < int(scheduler.sigmas.size()),
            "invalid H3 scheduler step");
    // FastVideo first promotes the stored FP32 NumPy scalars to Python
    // doubles, computes ratio and its complement independently, and only
    // then lets MLX materialise each scalar as FP32.  Computing `1-ratio`
    // from an already-rounded C++ float changes one ULP at steps 1/2.  That
    // tiny sample difference crosses BF16 projection boundaries in the next
    // 50-block pass and becomes a large final latent divergence.
    const double sigma_from_timestep =
        1.0 - double(scheduler.timesteps[step_index]);
    auto denoised = sample + float(sigma_from_timestep) * velocity;
    const double sigma = double(scheduler.sigmas[step_index]);
    const double next = double(scheduler.sigmas[step_index + 1]);
    const double ratio = next / sigma;
    const float ratio_fp32 = float(ratio);
    const float complement_fp32 = float(1.0 - ratio);
    return mx::astype(ratio_fp32 * mx::astype(sample, mx::float32) +
                          complement_fp32 * mx::astype(denoised, mx::float32),
                      sample.dtype());
}

} // namespace tc::h3_mlx

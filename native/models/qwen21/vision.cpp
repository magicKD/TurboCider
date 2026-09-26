#include "vision.hpp"
#include <algorithm>
#include <cmath>

namespace tc::qwen21 {
namespace {
Tensor gelu(const Tensor &x) {
    // Reference vision GELU has an explicit FP32 sqrt constant, so BF16
    // input promotes here. Preserve that boundary for the mflux oracle.
    return Tensor(.5f, x.dtype()) * x *
        (1.f + mx::tanh(mx::sqrt(Tensor(float(2.0 / M_PI))) *
            (x + Tensor(.044715f, x.dtype()) * mx::power(x, Tensor(3.f, x.dtype())))));
}
Tensor rotate(const Tensor &x, const Tensor &cosine, const Tensor &sine) {
    auto halves = mx::split(x, 2, -1);
    return x * cosine + mx::concatenate({-halves[1], halves[0]}, -1) * sine;
}
Tensor pe_gelu(const Tensor &x, bool approximate) {
    auto f = mx::astype(x, mx::float32);
    auto activated = approximate ? gelu(f) :
        .5f * f * (1.f + mx::erf(f / std::sqrt(2.f)));
    return mx::astype(activated, x.dtype());
}
Tensor pe_projection(const Tensor &x, const Tensor &weight, const Tensor &bias,
                     bool compensated) {
    auto f = mx::astype(x, mx::float32);
    auto w = mx::astype(weight, mx::float32);
    auto b = mx::astype(bias, mx::float32);
    if (!compensated)
        return mx::astype(mx::addmm(b, f, mx::transpose(w)), x.dtype());
    // Short BF16-product reductions followed by compensated FP32 summation.
    // This diagnostic preserves the final BF16 boundary; it is not a change
    // to the reference precision or an accepted production implementation.
    Tensor sum = mx::zeros({x.shape(0), weight.shape(0)}, mx::float32);
    Tensor correction = mx::zeros_like(sum);
    for (int begin = 0; begin < x.shape(-1); begin += 32) {
        const int end = std::min(begin + 32, x.shape(-1));
        auto product = mx::matmul(slice_axis(f, 1, begin, end),
                                 mx::transpose(slice_axis(w, 1, begin, end)));
        auto adjusted = product - correction;
        auto next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
        mx::eval(sum, correction);
    }
    return mx::astype(sum + (b - correction), x.dtype());
}
Tensor pe_attention(const Tensor &q, const Tensor &k, const Tensor &v,
                    std::unordered_map<std::string, Tensor> *trace) {
    // Preserve Transformers eager BF16 QK/scaled-score/probability boundaries;
    // the diffusion tower retains its independently validated SDPA path.
    auto raw_score = mx::astype(mx::matmul(mx::astype(q, mx::float32),
        mx::swapaxes(mx::astype(k, mx::float32), -1, -2)), q.dtype());
    auto score = mx::astype(mx::astype(raw_score, mx::float32) *
        float(1. / std::sqrt(double(q.shape(-1)))), q.dtype());
    auto probability = mx::astype(mx::softmax(mx::astype(score, mx::float32), -1), q.dtype());
    auto output = mx::astype(mx::matmul(mx::astype(probability, mx::float32),
        mx::astype(v, mx::float32)), v.dtype());
    if (trace) {
        trace->insert_or_assign("attn_q", q);
        trace->insert_or_assign("attn_k", k);
        trace->insert_or_assign("attn_v", v);
        trace->insert_or_assign("attn_raw_score", raw_score);
        trace->insert_or_assign("attn_score", score);
        trace->insert_or_assign("attn_probability", probability);
        trace->insert_or_assign("attn_output", output);
    }
    return mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {1, q.shape(2), q.shape(1) * q.shape(3)});
}
}
VisionEncoder::VisionEncoder(const Weights &weights, VisionConfig config)
    : weights_(weights), config_(config) {
    require(config.hidden > 0 && config.heads > 0 && config.hidden % (config.heads * 4) == 0 &&
            config.patch > 0 && config.temporal == 2 && config.channels == 3 && config.layers > 0 &&
            config.merge > 0 && config.position_side > 0, "invalid Qwen21 vision geometry");
    require(std::is_sorted(config.deepstack_layers.begin(), config.deepstack_layers.end()), "vision deepstack layers must be sorted");
    for (int layer : config.deepstack_layers)
        require(layer >= 0 && layer < config.layers, "vision deepstack layer out of range");
    require(!config.pe_qwen35 || config.deepstack_layers.empty(), "Qwen35 PE vision does not use DeepStack");
    require(!config.pe_compensated_projection || config.pe_qwen35,
            "compensated projection is a PE-only diagnostic");
    require(!config.pe_fp32 || (config.pe_qwen35 && !config.pe_compensated_projection),
            "FP32 vision is a PE-only diagnostic, incompatible with compensated projection");
}
Tensor VisionEncoder::norm(const Tensor &x, const std::string &p) const {
    if (config_.pe_qwen35) {
        // Keep normalization and affine accumulation in FP32, then round once
        // at the output boundary, as in the Transformers BF16 LayerNorm.
        auto f = mx::astype(x, mx::float32);
        auto centered = f - mx::mean(f, -1, true);
        auto normalized = centered * mx::rsqrt(mx::mean(centered * centered, -1, true) + 1e-6f);
        return mx::astype(normalized * mx::astype(weights_.at(p + ".weight"), mx::float32) +
            mx::astype(weights_.at(p + ".bias"), mx::float32), x.dtype());
    }
    return mx::fast::layer_norm(x, weights_.at(p + ".weight"), weights_.at(p + ".bias"), 1e-6f);
}
Tensor VisionEncoder::project(const Tensor &x, const std::string &p) const {
    if (!config_.pe_qwen35) return linear(x, weights_, p);
    // Torch linear/conv bias participates in accumulation before BF16 rounding.
    return pe_projection(x, weights_.at(p + ".weight"), weights_.at(p + ".bias"),
                         config_.pe_compensated_projection);
}
Tensor VisionEncoder::merge(const Tensor &x, const std::string &p, bool postshuffle) const {
    int width = config_.hidden * config_.merge * config_.merge;
    auto hidden = postshuffle ? norm(mx::reshape(x, {-1, width}), p + ".norm")
                              : mx::reshape(norm(x, p + ".norm"), {-1, width});
    auto projected = project(hidden, p + ".linear_fc1");
    return project(config_.pe_qwen35 ? pe_gelu(projected, false) : gelu(projected), p + ".linear_fc2");
}
VisionFeatures VisionEncoder::encode(const Tensor &input_patches, int h, int w,
                                     const Event &event, std::atomic<bool> &cancelled,
                                     std::unordered_map<std::string, Tensor> *trace) const {
    const auto &c = config_;
    auto patches = c.pe_fp32 ? mx::astype(input_patches, mx::float32) : input_patches;
    require(h > 0 && w > 0 && h % c.merge == 0 && w % c.merge == 0 &&
            patches.ndim() == 2 && patches.shape(0) == int64_t(h) * w &&
            patches.shape(1) == c.channels * c.temporal * c.patch * c.patch,
            "Qwen21 vision patches/grid mismatch");
    checkpoint(cancelled);
    Tensor hidden(0.f);
    if (c.pe_qwen35) {
        auto flat = mx::reshape(weights_.at("model.visual.patch_embed.proj.weight"), {c.hidden, -1});
        hidden = pe_projection(patches, flat, weights_.at("model.visual.patch_embed.proj.bias"),
                               c.pe_compensated_projection);
    } else {
        auto input = mx::reshape(patches, {h*w,c.channels,c.temporal,c.patch,c.patch});
        input = mx::transpose(input, {0,2,3,4,1});
        auto kernel = mx::transpose(weights_.at("model.visual.patch_embed.proj.weight"), {0,2,3,4,1});
        hidden = mx::conv3d(input, kernel, {c.temporal,c.patch,c.patch}) + weights_.at("model.visual.patch_embed.proj.bias");
        hidden = mx::reshape(hidden, {h*w,c.hidden});
    }
    if (trace) trace->insert_or_assign("patch", hidden);

    std::vector<int> rows, cols;
    for (int block_h=0; block_h<h/c.merge; ++block_h)
        for (int block_w=0; block_w<w/c.merge; ++block_w)
            for (int inner_h=0; inner_h<c.merge; ++inner_h)
                for (int inner_w=0; inner_w<c.merge; ++inner_w) {
                    rows.push_back(block_h*c.merge+inner_h);
                    cols.push_back(block_w*c.merge+inner_w);
                }
    auto row = Tensor(rows.data(), {h*w}, mx::int32), col = Tensor(cols.data(), {h*w}, mx::int32);
    auto ys = mx::take(mx::linspace(0.f,float(c.position_side-1),h),row);
    auto xs = mx::take(mx::linspace(0.f,float(c.position_side-1),w),col);
    auto y0=mx::astype(ys,mx::int32), x0=mx::astype(xs,mx::int32);
    auto y1=mx::minimum(y0+1,Tensor(c.position_side-1)), x1=mx::minimum(x0+1,Tensor(c.position_side-1));
    auto dy=ys-mx::astype(y0,mx::float32), dx=xs-mx::astype(x0,mx::float32);
    // FP32 mode must also promote position lookup/interpolation, not merely
    // projections. BF16 positions would silently retain rounding boundaries.
    const auto &table = weights_.at("model.visual.pos_embed.weight");
    auto positions = c.pe_fp32 ? mx::astype(table, mx::float32) : table;
    auto coefficient = [&](const Tensor &value) {
        // The PE reference pins Transformers 5.4: coefficients are cast to
        // the position-table dtype before multiplication and sequential sums.
        // Newer Transformers changed this boundary; do not mix those oracles.
        return mx::expand_dims(c.pe_qwen35 ? mx::astype(value, positions.dtype()) : value, -1);
    };
    auto pos = mx::take(positions,y0*c.position_side+x0,0)*coefficient((1.f-dy)*(1.f-dx))
             + mx::take(positions,y0*c.position_side+x1,0)*coefficient((1.f-dy)*dx)
             + mx::take(positions,y1*c.position_side+x0,0)*coefficient(dy*(1.f-dx))
             + mx::take(positions,y1*c.position_side+x1,0)*coefficient(dy*dx);
    hidden = hidden + (c.pe_qwen35 ? mx::astype(pos, hidden.dtype()) : pos);
    if (trace) {
        trace->insert_or_assign("pos", pos);
        trace->insert_or_assign("input", hidden);
    }
    int head_dim=c.hidden/c.heads;
    std::vector<float> frequencies;
    for(int i=0;i<head_dim/2;i+=2) frequencies.push_back(1.f/std::pow(10000.f,float(i)/(head_dim/2)));
    auto inv=Tensor(frequencies.data(),{head_dim/4},mx::float32);
    auto angle=mx::concatenate({mx::expand_dims(mx::astype(row,mx::float32),-1)*inv,
                               mx::expand_dims(mx::astype(col,mx::float32),-1)*inv},-1);
    angle=mx::concatenate({angle,angle},-1);
    auto cosine=mx::reshape(mx::cos(angle),{1,1,h*w,head_dim});
    auto sine=mx::reshape(mx::sin(angle),{1,1,h*w,head_dim});
    if (trace && c.pe_qwen35) {
        trace->insert_or_assign("rope_cos", mx::squeeze(cosine, 0));
        trace->insert_or_assign("rope_sin", mx::squeeze(sine, 0));
    }
    std::vector<Tensor> deepstack;
    for(int i=0;i<c.layers;++i) {
        checkpoint(cancelled);
        auto p="model.visual.blocks."+std::to_string(i);
        auto n1 = norm(hidden,p+".norm1");
        auto packed = project(n1,p+".attn.qkv");
        if (trace && i == 0) {
            trace->insert_or_assign("norm1", n1);
            trace->insert_or_assign("qkv", packed);
            if (c.pe_qwen35) {
                const auto &weight = weights_.at(p+".attn.qkv.weight");
                const auto &bias = weights_.at(p+".attn.qkv.bias");
                auto product = mx::matmul(mx::astype(n1,mx::float32),
                    mx::transpose(mx::astype(weight,mx::float32)));
                trace->insert_or_assign("qkv_bias_after_round", mx::astype(product,n1.dtype()) + bias);
                trace->insert_or_assign("qkv_native_addmm", mx::addmm(bias,n1,mx::transpose(weight)));
                trace->insert_or_assign("qkv_native_separate", mx::matmul(n1,mx::transpose(weight)) + bias);
            }
        }
        auto qkv=mx::split(packed,3,-1);
        auto q=heads(mx::expand_dims(qkv[0],0),c.heads,head_dim);
        auto k=heads(mx::expand_dims(qkv[1],0),c.heads,head_dim);
        auto v=heads(mx::expand_dims(qkv[2],0),c.heads,head_dim);
        auto qr = rotate(q,cosine,sine), kr = rotate(k,cosine,sine);
        if (c.pe_qwen35) { qr = mx::astype(qr, q.dtype()); kr = mx::astype(kr, k.dtype()); }
        auto attended=mx::squeeze(c.pe_qwen35 ? pe_attention(qr,kr,v, i == 0 ? trace : nullptr) : attend(qr,kr,v),0);
        auto projected = project(attended,p+".attn.proj");
        hidden=hidden+projected;
        auto n2 = norm(hidden,p+".norm2");
        auto mlp = project(n2,p+".mlp.linear_fc1");
        auto activated = c.pe_qwen35 ? pe_gelu(mlp, true) : gelu(mlp);
        auto mlp_out = project(activated,p+".mlp.linear_fc2");
        if (trace && i == 0) {
            trace->insert_or_assign("attention", projected);
            trace->insert_or_assign("norm2", n2);
            trace->insert_or_assign("fc1", mlp);
            trace->insert_or_assign("activation", activated);
            trace->insert_or_assign("fc2", mlp_out);
        }
        hidden=hidden+mlp_out;
        auto found=std::find(c.deepstack_layers.begin(),c.deepstack_layers.end(),i);
        if(found!=c.deepstack_layers.end())
            deepstack.push_back(merge(hidden,"model.visual.deepstack_merger_list."+std::to_string(found-c.deepstack_layers.begin()),true));
        mx::eval(hidden);
        if (trace) trace->insert_or_assign("block" + std::to_string(i), hidden);
        if(event) event(c.pe_qwen35 ? "qwen35_pe_vision_encode" : "qwen21_vision_encode",i+1,c.layers);
    }
    return {merge(hidden,"model.visual.merger",false), std::move(deepstack)};
}
} // namespace tc::qwen21

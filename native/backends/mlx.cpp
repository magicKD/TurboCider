#include "mlx.hpp"
#include <cmath>
#include <cstring>
#include <map>
namespace tc {
namespace {
struct LoRAPair { std::optional<Tensor> down, up, alpha; };
static bool remove_suffix(std::string &value, const std::string &suffix) {
    if (!value.ends_with(suffix)) return false;
    value.resize(value.size() - suffix.size()); return true;
}
static std::string strip_lora_prefix(std::string value) {
    for (const auto *prefix : {"base_model.model.", "transformer.", "diffusion_model.", "model."})
        if (value.starts_with(prefix)) { value.erase(0, std::strlen(prefix)); break; }
    return value;
}
static std::vector<std::string> lora_targets(std::string stem) {
    stem = strip_lora_prefix(stem);
    std::vector<std::string> result{stem};
    std::smatch match;
    if (std::regex_match(stem, match, std::regex("^double_blocks\\.([0-9]+)\\.(.+)$"))) {
        std::string prefix = "transformer_blocks." + std::string(match[1]) + ".";
        auto tail = std::string(match[2]);
        if (tail == "img_attn.qkv") return {prefix + "attn.to_q", prefix + "attn.to_k", prefix + "attn.to_v"};
        if (tail == "txt_attn.qkv") return {prefix + "attn.add_q_proj", prefix + "attn.add_k_proj", prefix + "attn.add_v_proj"};
    }
    if (std::regex_match(stem, match, std::regex("^single_blocks\\.([0-9]+)\\.(.+)$"))) {
        std::string prefix = "single_transformer_blocks." + std::string(match[1]) + ".attn.";
        auto tail = std::string(match[2]);
        if (tail == "linear1") return {prefix + "to_qkv_mlp_proj"};
        if (tail == "linear2") return {prefix + "to_out"};
    }
    return result;
}
}
void Weights::load(const std::filesystem::path &p, const Event &event,
                   std::atomic<bool> &cancelled) {
    if (!values_.empty())
        return;
    require(std::filesystem::is_directory(p), "missing component: " + p.string());
    std::vector<std::filesystem::path> shards;
    for (auto &f : std::filesystem::directory_iterator(p)) {
        // Diffusers index directories can contain convenience symlinks (for
        // example a link to a ComfyUI single-file checkpoint).  Loading those
        // alongside the real shards would duplicate tensors and can make a
        // partially downloaded model look valid until a late duplicate-key
        // or memory failure.  Only immutable regular shard files belong to a
        // directory-backed component; direct single-file paths remain
        // supported by load_file().
        if (!f.is_symlink() && f.is_regular_file() && f.path().extension() == ".safetensors")
            shards.push_back(f.path());
    }
    std::sort(shards.begin(), shards.end());
    require(!shards.empty(), "no safetensors in " + p.string());
    try {
        int i = 0;
        for (auto &f : shards) {
            checkpoint(cancelled);
            event("load_" + p.filename().string(), i, int(shards.size()));
            auto data = mx::load_safetensors(f.string());
            for (auto &[k, v] : data.first) {
                require(!values_.count(k), "duplicate tensor: " + k);
                values_.emplace(k, v);
            }
            ++i;
        }
    } catch (...) {
        clear();
        throw;
    }
}
void Weights::load_file(const std::filesystem::path &path, const std::string &prefix) {
    auto data = mx::load_safetensors(path.string());
    for (auto &[name, value] : data.first)
        if (name.starts_with(prefix)) {
            auto key = name.substr(prefix.size());
            require(!values_.count(key), "duplicate weight: " + key);
            values_.emplace(key, value);
        }
    require(!values_.empty(), "no tensors match component prefix");
}
void Weights::remap_keys(const std::function<std::string(const std::string &)> &transform) {
    std::unordered_map<std::string, Tensor> remapped;
    remapped.reserve(values_.size());
    for (auto &[key, value] : values_) {
        auto target = transform(key);
        require(!target.empty(), "weight key remap produced an empty key");
        require(!remapped.count(target), "duplicate weight after key remap: " + target);
        remapped.emplace(std::move(target), std::move(value));
    }
    values_ = std::move(remapped);
}
void Weights::fuse_keys(const std::string &target, const std::vector<std::string> &sources,
                        int axis) {
    require(!sources.empty(), "cannot fuse an empty weight list");
    std::vector<Tensor> pieces;
    pieces.reserve(sources.size());
    for (const auto &source : sources) {
        auto found = values_.find(source);
        require(found != values_.end(), "missing weight needed for fusion: " + source);
        pieces.push_back(found->second);
    }
    require(!values_.count(target), "fused weight already exists: " + target);
    values_.emplace(target, mx::concatenate(pieces, axis));
    for (const auto &source : sources)
        values_.erase(source);
}
const Tensor &Weights::at(const std::string &k) const {
    auto i = values_.find(k);
    require(i != values_.end(), "missing weight: " + k);
    return i->second;
}
bool Weights::has(const std::string &k) const {
    return values_.count(k);
}
void Weights::clear() {
    values_.clear();
}
size_t Weights::bytes() const {
    size_t n = 0;
    for (auto &[k, v] : values_)
        n += v.nbytes();
    return n;
}
void Weights::materialize() {
    std::vector<Tensor> arrays;
    arrays.reserve(values_.size());
    for (auto &[key, value] : values_)
        arrays.push_back(value);
    mx::eval(arrays);
}
size_t Weights::apply_loras(const std::vector<LoRAAsset> &adapters, const std::string &role,
                            const Event &event, std::atomic<bool> &cancelled) {
    size_t applied = 0;
    for (size_t index = 0; index < adapters.size(); ++index) {
        const auto &adapter = adapters[index];
        if (adapter.role != role) continue;
        require(std::filesystem::is_regular_file(adapter.path), "LoRA file missing: " + adapter.path);
        checkpoint(cancelled); event("load_lora", int(index), int(adapters.size()));
        auto data = mx::load_safetensors(adapter.path);
        std::map<std::string, LoRAPair> pairs;
        for (auto &[raw, value] : data.first) {
            auto stem = raw;
            if (remove_suffix(stem, ".lora_A.default.weight") || remove_suffix(stem, ".lora_A.weight") ||
                remove_suffix(stem, ".lora_down.weight")) pairs[stem].down = value;
            else if (remove_suffix(stem, ".lora_B.default.weight") || remove_suffix(stem, ".lora_B.weight") ||
                     remove_suffix(stem, ".lora_up.weight")) pairs[stem].up = value;
            else if (remove_suffix(stem, ".alpha") || remove_suffix(stem, ".lora_alpha")) pairs[stem].alpha = value;
        }
        size_t adapter_applied = 0;
        for (auto &[stem, pair] : pairs) {
            if (!pair.down && !pair.up) continue;
            require(pair.down && pair.up, "incomplete LoRA pair: " + stem);
            const auto &down = *pair.down, &up = *pair.up;
            require(down.ndim() == 2 && up.ndim() == 2 && down.shape(0) == up.shape(1),
                    "invalid LoRA rank geometry: " + stem);
            float scale = adapter.strength;
            if (pair.alpha) { require(pair.alpha->size() == 1, "LoRA alpha must be scalar: " + stem); mx::eval(*pair.alpha); scale *= pair.alpha->item<float>() / float(down.shape(0)); }
            int offset = 0;
            for (const auto &target : lora_targets(stem)) {
                auto key = target.ends_with(".weight") ? target : target + ".weight";
                auto found = values_.find(key);
                if (found == values_.end() && target.ends_with(".attention.to_out.0")) {
                    key = target.substr(0, target.size() - std::strlen("to_out.0")) + "out.weight";
                    found = values_.find(key);
                }
                if (found != values_.end()) {
                    auto base = found->second;
                    require(base.ndim() == 2 && base.shape(1) == down.shape(1),
                            "LoRA input does not match " + target);
                    int rows = base.shape(0);
                    require(offset + rows <= up.shape(0),
                            "LoRA output does not match " + target);
                    auto selected = lora_targets(stem).size() == 1
                                        ? up
                                        : slice_axis(up, 0, offset, offset + rows);
                    offset += rows;
                    auto delta = mx::matmul(mx::astype(selected, mx::float32),
                                            mx::astype(down, mx::float32)) *
                                 Tensor(scale, mx::float32);
                    found->second = mx::astype(mx::astype(base, mx::float32) + delta,
                                               base.dtype());
                    mx::eval(found->second);
                    ++adapter_applied;
                    continue;
                }

                // Comfy's Z-Image single-file format stores Q/K/V as one
                // [3*dim, dim] tensor.  Apply ordinary, independently stored
                // LoRA projections to the corresponding in-memory row slice;
                // no merged checkpoint is written to disk.
                std::smatch projection;
                if (std::regex_match(target, projection,
                                     std::regex("^(.+\\.attention)\\.to_([qkv])$"))) {
                    auto fused = values_.find(std::string(projection[1]) + ".qkv.weight");
                    if (fused == values_.end()) continue;
                    auto base = fused->second;
                    require(base.ndim() == 2 && base.shape(0) % 3 == 0 &&
                                base.shape(1) == down.shape(1),
                            "Z-Image fused QKV geometry does not match " + target);
                    int rows = base.shape(0) / 3;
                    require(up.shape(0) == rows,
                            "Z-Image LoRA output does not match " + target);
                    char which = std::string(projection[2])[0];
                    int begin = which == 'q' ? 0 : (which == 'k' ? rows : 2 * rows);
                    auto delta = mx::matmul(mx::astype(up, mx::float32),
                                            mx::astype(down, mx::float32)) *
                                 Tensor(scale, mx::float32);
                    auto merged = mx::astype(
                        mx::astype(slice_axis(base, 0, begin, begin + rows), mx::float32) + delta,
                        base.dtype());
                    std::vector<Tensor> pieces;
                    if (begin) pieces.push_back(slice_axis(base, 0, 0, begin));
                    pieces.push_back(merged);
                    if (begin + rows < base.shape(0))
                        pieces.push_back(slice_axis(base, 0, begin + rows, base.shape(0)));
                    fused->second = mx::concatenate(pieces, 0);
                    mx::eval(fused->second);
                    ++adapter_applied;
                }
            }
        }
        require(adapter_applied > 0, "LoRA did not match any " + role + " weights: " + adapter.path);
        applied += adapter_applied; event("load_lora", int(index + 1), int(adapters.size()));
    }
    return applied;
}
Tensor linear(const Tensor &x, const Weights &w, const std::string &p) {
    auto wt = mx::transpose(w.at(p + ".weight"));
    return w.has(p + ".bias") ? mx::addmm(w.at(p + ".bias"), x, wt) : mx::matmul(x, wt);
}
Tensor silu(const Tensor &x) {
    static auto compiled = mx::compile(
        [](const std::vector<Tensor> &a) { return std::vector<Tensor>{a[0] * mx::sigmoid(a[0])}; },
        true);
    return compiled({x})[0];
}
Tensor rms(const Tensor &x, const Tensor &w, float eps) {
    auto f = mx::astype(x, mx::float32);
    // MLX's fast RMSNorm is a fused Metal primitive.  Keep the former FP32
    // accumulation/weight contract and cast only the final value back to the
    // activation dtype so existing BF16 parity is preserved.
    if (!std::getenv("TURBOCIDER_DISABLE_FUSED_RMSNORM"))
        return mx::astype(mx::fast::rms_norm(f, mx::astype(w, mx::float32), eps),
                          x.dtype());
    return mx::astype(f * mx::rsqrt(mx::mean(mx::square(f), -1, true) + eps) *
                          mx::astype(w, mx::float32),
                      x.dtype());
}
Tensor norm(const Tensor &x) {
    return mx::fast::layer_norm(x, {}, {}, 1e-6f);
}
Tensor slice_axis(const Tensor &x, int axis, int start, int stop) {
    if (axis < 0)
        axis += x.ndim();
    mx::Shape a(x.ndim(), 0), b = x.shape();
    a[axis] = start;
    b[axis] = stop;
    return mx::slice(x, a, b);
}
Tensor heads(const Tensor &x, int n, int d) {
    return mx::transpose(mx::reshape(x, {1, x.shape(1), n, d}), {0, 2, 1, 3});
}
Tensor attend(const Tensor &q, const Tensor &k, const Tensor &v, bool f32,
              const std::optional<Tensor> &mask, bool force_fused) {
    auto dtype = q.dtype();
    auto a = mx::fast::scaled_dot_product_attention(
        f32 ? mx::astype(q, mx::float32) : q, f32 ? mx::astype(k, mx::float32) : k,
        f32 ? mx::astype(v, mx::float32) : v, 1.f / std::sqrt(float(q.shape(-1))), "", mask,
        {}, force_fused);
    if (f32)
        a = mx::astype(a, dtype);
    return mx::reshape(mx::transpose(a, {0, 2, 1, 3}), {1, q.shape(2), q.shape(1) * q.shape(3)});
}
Tensor rope_pairs(const Tensor &x, const Tensor &cos, const Tensor &sin) {
    auto f = mx::reshape(mx::astype(x, mx::float32), {1, x.shape(1), x.shape(2), 64, 2});
    auto parts = mx::split(f, 2, -1);
    auto a = mx::squeeze(parts[0], -1), b = mx::squeeze(parts[1], -1);
    auto c = mx::reshape(cos, {1, 1, x.shape(2), 64}), s = mx::reshape(sin, {1, 1, x.shape(2), 64});
    return mx::astype(mx::reshape(mx::stack({a * c - b * s, b * c + a * s}, -1), x.shape()),
                      x.dtype());
}
std::vector<Tensor> rope_pairs_pair(const Tensor &q, const Tensor &k,
                                    const Tensor &cos, const Tensor &sin) {
    require(q.shape() == k.shape() && q.ndim() == 4 && q.shape(-1) % 2 == 0,
            "paired RoPE geometry mismatch");
    require(cos.shape() == sin.shape() && cos.size() == size_t(q.shape(2) * q.shape(3) / 2),
            "paired RoPE frequency geometry mismatch");
    // Rotate Q and K in one native Metal dispatch.  FLUX repeats one
    // [sequence, head_dim / 2] frequency table across all heads, so the
    // modulo maps a flat Q/K pair back to its shared frequency element.
    static auto kernel = mx::fast::metal_kernel(
        "tc_rope_qk", {"q", "k", "cosine", "sine", "pair_count", "frequency_span"},
        {"q_out", "k_out"},
        "uint pair = thread_position_in_grid.x; "
        "if (pair < uint(pair_count)) { "
        "  uint frequency = pair % uint(frequency_span); uint base = pair * 2; "
        "  float c = float(cosine[frequency]); float s = float(sine[frequency]); "
        "  float qa = float(q[base]); float qb = float(q[base + 1]); "
        "  float ka = float(k[base]); float kb = float(k[base + 1]); "
        "  q_out[base] = T(qa * c - qb * s); "
        "  q_out[base + 1] = T(qb * c + qa * s); "
        "  k_out[base] = T(ka * c - kb * s); "
        "  k_out[base + 1] = T(kb * c + ka * s); "
        "}");
    const int pairs = int(q.size() / 2);
    const int frequency_span = q.shape(2) * q.shape(3) / 2;
    return kernel({q, k, cos, sin, Tensor(pairs), Tensor(frequency_span)},
                  {q.shape(), k.shape()}, {q.dtype(), k.dtype()}, {pairs, 1, 1},
                  {256, 1, 1}, {{"T", q.dtype()}}, {}, false, {});
}
std::vector<float> flux_gpu_sigmas(int tokens, int steps) {
    double m200 = .00016927 * tokens + .45666666, mu = m200;
    if (tokens <= 4300) {
        double m10 = 8.73809524e-5 * tokens + 1.89833333;
        double a = (m200 - m10) / 190.;
        mu = a * steps + (m200 - 200. * a);
    }
    auto t = mx::linspace(1., 1. / steps, steps, mx::float32);
    auto e = mx::exp(Tensor(float(mu)));
    auto values = e / (e + mx::power(1.f / t - 1.f, Tensor(1.f)));
    mx::eval(values);
    std::vector<float> result(values.data<float>(), values.data<float>() + steps);
    result.push_back(0);
    return result;
}
Tensor euler_step(const Tensor &x, const Tensor &noise, float dt) {
    if (x.dtype() == mx::float32) {
        static auto step = mx::compile(
            [](const std::vector<Tensor> &a) { return std::vector<Tensor>{a[0] + a[2] * a[1]}; },
            true);
        return step({x, noise, Tensor(dt)})[0];
    }
    // Custom Metal kernel participates in MLX's dependency/allocator system.
    // Keep the reference BF16 multiply rounding before the residual addition.
    static auto kernel =
        mx::fast::metal_kernel("tc_euler", {"x", "noise", "dt", "count"}, {"out"},
                               "uint i = thread_position_in_grid.x; if(i < uint(count)) { T "
                               "product = T(noise[i] * dt); out[i] = T(x[i] + product); }");
    auto d = mx::astype(Tensor(dt), x.dtype());
    return kernel({x, noise, d, Tensor(int(x.size()))}, {x.shape()}, {x.dtype()},
                  {int(x.size()), 1, 1}, {256, 1, 1}, {{"T", x.dtype()}}, {}, false, {})[0];
}
} // namespace tc

namespace tc {
// DispatchQueue is serial but does not guarantee a stable OS thread. Persistent
// tensors need streams valid across callers; execution_mutex serializes access.
void configure_streams() {
    using namespace tc;
    static auto cpu = mx::new_thread_unsafe_stream(mx::Device(mx::Device::cpu));
    static auto gpu = mx::new_thread_unsafe_stream(mx::Device(mx::Device::gpu));
    mx::set_default_stream(cpu);
    mx::set_default_stream(gpu);
    mx::set_default_device(mx::Device(mx::Device::gpu));
}
} // namespace tc

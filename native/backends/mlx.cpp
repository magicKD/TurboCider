#include "mlx.hpp"
#include <cmath>
#include <cstdlib>
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

static const Tensor &convrot_h256() {
    // comfy-quants uses regular_hadamard(4), recursively Kronecker-squared
    // until 256.  This is not the same row/column ordering as MLX's built-in
    // Sylvester Hadamard transform.  Normalize H4 up front so H256 is
    // orthonormal (0.5^4 == 1/sqrt(256)).
    static const Tensor matrix = [] {
        static const float values[] = {
             .5f,  .5f,  .5f, -.5f,
             .5f,  .5f, -.5f,  .5f,
             .5f, -.5f,  .5f,  .5f,
            -.5f,  .5f,  .5f,  .5f,
        };
        auto h4 = Tensor(values, {4, 4}, mx::float32);
        auto h16 = mx::kron(h4, h4);
        auto h256 = mx::kron(h16, h16);
        mx::eval(h256);
        return h256;
    }();
    return matrix;
}

static Tensor convrot_rotate(const Tensor &x) {
    require(x.shape(-1) % 256 == 0, "ConvRot input must align to 256 values");
    auto grouped_shape = x.shape();
    grouped_shape.back() = x.shape(-1) / 256;
    grouped_shape.push_back(256);
    auto grouped = mx::reshape(x, grouped_shape);
    auto hadamard = mx::astype(convrot_h256(), x.dtype());
    return mx::reshape(mx::matmul(grouped, hadamard), x.shape());
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
void Weights::load_gguf_file(const std::filesystem::path &path) {
    require(values_.empty(), "GGUF weights are already loaded");
    require(std::filesystem::is_regular_file(path) && path.extension() == ".gguf",
            "GGUF checkpoint missing: " + path.string());
    try {
        auto data = mx::load_gguf(path.string());
        for (auto &[name, value] : data.first) {
            require(!values_.count(name), "duplicate GGUF tensor: " + name);
            values_.emplace(std::move(name), std::move(value));
        }
        require(!values_.empty(), "GGUF checkpoint contains no tensors");
    } catch (...) {
        clear();
        throw;
    }
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
bool Weights::quantized(const std::string &prefix) const {
    return has(prefix + ".weight") && has(prefix + ".scales") &&
           !has(prefix + ".comfy_quant") &&
           at(prefix + ".weight").dtype() == mx::uint32;
}

bool Weights::convrot(const std::string &prefix) const {
    if (!has(prefix + ".weight") || !has(prefix + ".comfy_quant"))
        return false;
    const auto dtype = at(prefix + ".weight").dtype();
    const bool raw = dtype == mx::int8 && has(prefix + ".weight_scale");
    const bool packed = dtype == mx::uint32 && has(prefix + ".scales") &&
                        has(prefix + ".biases");
    return raw || packed;
}

void Weights::cast_unquantized_float32(mx::Dtype dtype) {
    std::vector<Tensor> converted;
    for (auto &[key, value] : values_) {
        // ConvRot scale tensors are deliberately FP32.  Comfy's manual-cast
        // path casts the ordinary model parameters to the model dtype while
        // retaining these dequantization scales for FP32 multiplication.
        if (value.dtype() != mx::float32 || key.ends_with(".weight_scale") ||
            key.ends_with(".scales") || key.ends_with(".biases"))
            continue;
        value = mx::astype(value, dtype);
        converted.push_back(value);
    }
    if (!converted.empty())
        mx::eval(converted);
}

size_t Weights::pack_convrot_q8() {
    std::vector<std::string> prefixes;
    for (const auto &[key, _] : values_)
        if (key.ends_with(".comfy_quant"))
            prefixes.push_back(key.substr(0, key.size() - std::strlen(".comfy_quant")));
    std::sort(prefixes.begin(), prefixes.end());

    size_t packed_count = 0;
    for (const auto &prefix : prefixes) {
        auto weight_key = prefix + ".weight";
        auto scale_key = prefix + ".weight_scale";
        auto found = values_.find(weight_key);
        if (found == values_.end() || found->second.dtype() == mx::uint32)
            continue;
        require(found->second.dtype() == mx::int8 && has(scale_key),
                "invalid raw ConvRot tensors: " + prefix);
        const auto &weight = found->second;
        const auto &row_scale = at(scale_key);
        require(weight.ndim() == 2 && weight.shape(1) % 256 == 0 &&
                    row_scale.ndim() == 2 && row_scale.shape(0) == weight.shape(0) &&
                    row_scale.shape(1) == 1,
                "invalid ConvRot Q8 packing geometry: " + prefix);

        // Comfy ConvRot stores signed q in [-128, 127] with one scale for an
        // entire output row. MLX affine Q8 consumes unsigned bytes and
        // reconstructs scale*q+bias per group. The q+128 / -128*scale mapping
        // is exact; repeating the row scale over 32-value groups only changes
        // layout. BF16 scale storage is the fast model-dtype path. FP32 remains
        // available for parity diagnosis because MLX dispatches a materially
        // slower kernel for the full 1024-token workload with FP32 scales.
        auto unsigned_weight = mx::astype(
            mx::astype(weight, mx::int32) + Tensor(128, mx::int32), mx::uint8);
        auto packed = mx::view(unsigned_weight, mx::uint32);
        const auto scale_dtype = std::getenv("TURBOCIDER_Z_CONVROT_FP32_SCALES")
                                     ? mx::float32
                                     : mx::bfloat16;
        auto scales = mx::repeat(mx::astype(row_scale, scale_dtype),
                                 weight.shape(1) / 32, 1);
        auto biases = scales * Tensor(-128.f, scale_dtype);
        mx::eval({packed, scales, biases});

        found->second = std::move(packed);
        values_.emplace(prefix + ".scales", std::move(scales));
        values_.emplace(prefix + ".biases", std::move(biases));
        values_.erase(scale_key);
        ++packed_count;
    }
    return packed_count;
}

namespace {
struct QuantizedGeometry {
    int group_size = 0;
    int bits = 0;
};

QuantizedGeometry quantized_geometry(const Tensor &weight, const Tensor &scales,
                                     int logical_input) {
    require(weight.ndim() == 2 && scales.ndim() == 2 && logical_input > 0 &&
                logical_input % scales.shape(1) == 0,
            "invalid MLX GGUF quantization geometry");
    const int group_size = logical_input / scales.shape(1);
    require((weight.shape(1) * 32) % logical_input == 0,
            "invalid packed MLX GGUF weight width");
    const int bits = weight.shape(1) * 32 / logical_input;
    require(group_size == 32 && (bits == 4 || bits == 8),
            "unsupported native GGUF quantization; expected Q4_0/Q4_1/Q8_0");
    return {group_size, bits};
}
}

Tensor Weights::project(const Tensor &x, const std::string &prefix) const {
    const auto &weight = at(prefix + ".weight");
    Tensor output = x;
    if (convrot(prefix)) {
        const bool packed = weight.dtype() == mx::uint32;
        const int logical_input = packed ? weight.shape(1) * 4 : weight.shape(1);
        require(weight.ndim() == 2 && logical_input % 256 == 0,
                "invalid ConvRot geometry: " + prefix);
        const int input = x.shape(-1);
        require(input == logical_input && input % 256 == 0,
                "ConvRot input does not match " + prefix);
        auto rotated = convrot_rotate(x);
        if (packed) {
            output = mx::astype(
                mx::quantized_matmul(rotated, weight, at(prefix + ".scales"),
                                     at(prefix + ".biases"), true, 32, 8,
                                     "affine"),
                x.dtype());
        } else {
            rotated = mx::astype(rotated, mx::float32);
            auto dense = mx::astype(weight, mx::float32) *
                         mx::astype(at(prefix + ".weight_scale"), mx::float32);
            output = mx::astype(mx::matmul(rotated, mx::transpose(dense)), x.dtype());
        }
    } else if (quantized(prefix)) {
        const auto &scales = at(prefix + ".scales");
        auto geometry = quantized_geometry(weight, scales, x.shape(-1));
        std::optional<Tensor> biases;
        if (has(prefix + ".biases"))
            biases = at(prefix + ".biases");
        output = mx::quantized_matmul(x, weight, scales, biases, true,
                                      geometry.group_size, geometry.bits, "affine");
    } else {
        auto dense = weight;
        if (dense.ndim() != 2)
            dense = mx::reshape(dense, {dense.shape(0), int(dense.size() / dense.shape(0))});
        output = mx::matmul(x, mx::transpose(dense));
    }
    auto runtime = runtime_loras_.find(prefix);
    if (runtime != runtime_loras_.end()) {
        for (const auto &adapter : runtime->second) {
            // Keep the low-rank accumulation in FP32.  Applying A/B as two
            // BF16 matmuls introduced a second lossy boundary that the
            // load-time `up @ down` merge does not have and the error grows
            // across hundreds of adapter projections.  The adapter remains
            // stored in its compact source dtype; only this rank-sized branch
            // is promoted for the projection.
            auto input = mx::astype(x, mx::float32);
            auto down = mx::astype(adapter.down, mx::float32);
            auto up = mx::astype(adapter.up, mx::float32);
            auto low = mx::matmul(input, mx::transpose(down));
            auto delta = mx::matmul(low, mx::transpose(up)) *
                         Tensor(adapter.scale, mx::float32);
            if (adapter.output_start == 0 && adapter.output_end == output.shape(-1))
                output = mx::astype(mx::astype(output, mx::float32) + delta,
                                    output.dtype());
            else {
                require(adapter.output_start >= 0 && adapter.output_end <= output.shape(-1) &&
                            adapter.output_start < adapter.output_end,
                        "invalid runtime LoRA output range: " + prefix);
                auto middle = mx::astype(
                    mx::astype(slice_axis(output, -1, adapter.output_start,
                                          adapter.output_end), mx::float32) + delta,
                    output.dtype());
                std::vector<Tensor> pieces;
                if (adapter.output_start)
                    pieces.push_back(slice_axis(output, -1, 0, adapter.output_start));
                pieces.push_back(std::move(middle));
                if (adapter.output_end < output.shape(-1))
                    pieces.push_back(slice_axis(output, -1, adapter.output_end,
                                                output.shape(-1)));
                output = mx::concatenate(pieces, -1);
            }
        }
    }
    if (has(prefix + ".bias"))
        output = output + mx::astype(at(prefix + ".bias"), output.dtype());
    return output;
}


Tensor Weights::project_range(const Tensor &x, const std::string &prefix,
                              int row_start, int row_end,
                              int col_start, int col_end) const {
    require(convrot(prefix), "project_range requires ConvRot weights: " + prefix);
    const auto &weight = at(prefix + ".weight");
    const int logical_input = weight.dtype() == mx::uint32
                                  ? weight.shape(1) * 4
                                  : weight.shape(1);
    require(weight.ndim() == 2 && row_start >= 0 && row_start < row_end &&
                row_end <= weight.shape(0) && col_start >= 0 && col_start < col_end &&
                col_end <= logical_input && col_start % 256 == 0 &&
                col_end % 256 == 0 && x.shape(-1) == col_end - col_start,
            "invalid ConvRot projection range: " + prefix);
    auto rotated = convrot_rotate(x);
    if (weight.dtype() == mx::uint32) {
        auto q = slice_axis(weight, 0, row_start, row_end);
        q = slice_axis(q, 1, col_start / 4, col_end / 4);
        auto scales = slice_axis(at(prefix + ".scales"), 0, row_start, row_end);
        scales = slice_axis(scales, 1, col_start / 32, col_end / 32);
        auto biases = slice_axis(at(prefix + ".biases"), 0, row_start, row_end);
        biases = slice_axis(biases, 1, col_start / 32, col_end / 32);
        return mx::astype(mx::quantized_matmul(rotated, q, scales, biases, true,
                                               32, 8, "affine"),
                          x.dtype());
    }
    auto q = slice_axis(weight, 0, row_start, row_end);
    q = slice_axis(q, 1, col_start, col_end);
    auto scale = slice_axis(at(prefix + ".weight_scale"), 0, row_start, row_end);
    auto dense = mx::astype(q, mx::float32) * mx::astype(scale, mx::float32);
    return mx::astype(mx::matmul(mx::astype(rotated, mx::float32),
                                 mx::transpose(dense)), x.dtype());
}

void Weights::dequantize(const std::vector<std::string> &prefixes) {
    for (const auto &prefix : prefixes) {
        if (convrot(prefix)) {
            const auto &packed = at(prefix + ".weight");
            const bool affine = packed.dtype() == mx::uint32;
            const int logical_input = affine ? packed.shape(1) * 4 : packed.shape(1);
            require(packed.ndim() == 2 && logical_input % 256 == 0,
                    "invalid ConvRot geometry: " + prefix);
            auto rotated = affine
                ? mx::dequantize(packed, at(prefix + ".scales"),
                                 at(prefix + ".biases"), 32, 8, "affine",
                                 std::nullopt, mx::float32)
                : mx::astype(packed, mx::float32) *
                      mx::astype(at(prefix + ".weight_scale"), mx::float32);
            auto dense = convrot_rotate(rotated);
            dense = mx::astype(dense, mx::bfloat16);
            mx::eval(dense);
            values_.at(prefix + ".weight") = std::move(dense);
            values_.erase(prefix + ".weight_scale");
            values_.erase(prefix + ".scales");
            values_.erase(prefix + ".biases");
            values_.erase(prefix + ".comfy_quant");
            continue;
        }
        if (!quantized(prefix))
            continue;
        auto weight = at(prefix + ".weight");
        auto scales = at(prefix + ".scales");
        const int logical_input = scales.shape(1) * 32;
        auto geometry = quantized_geometry(weight, scales, logical_input);
        std::optional<Tensor> biases;
        if (has(prefix + ".biases"))
            biases = at(prefix + ".biases");
        auto dense = mx::dequantize(weight, scales, biases, geometry.group_size,
                                    geometry.bits, "affine", std::nullopt, mx::float16);
        mx::eval(dense);
        values_.at(prefix + ".weight") = std::move(dense);
        values_.erase(prefix + ".scales");
        values_.erase(prefix + ".biases");
    }
}
const Tensor &Weights::at(const std::string &k) const {
    auto i = values_.find(k);
    require(i != values_.end(), "missing weight: " + k);
    return i->second;
}
bool Weights::has(const std::string &k) const {
    return values_.count(k);
}
void Weights::erase(const std::string &key) {
    values_.erase(key);
    runtime_loras_.erase(key);
}
void Weights::erase_prefix(const std::string &prefix) {
    for (auto it = values_.begin(); it != values_.end();) {
        if (it->first.starts_with(prefix))
            it = values_.erase(it);
        else
            ++it;
    }
    for (auto it = runtime_loras_.begin(); it != runtime_loras_.end();) {
        if (it->first.starts_with(prefix))
            it = runtime_loras_.erase(it);
        else
            ++it;
    }
}
void Weights::clear() {
    values_.clear();
    runtime_loras_.clear();
}
size_t Weights::bytes() const {
    size_t n = 0;
    for (auto &[k, v] : values_)
        n += v.nbytes();
    for (const auto &[_, adapters] : runtime_loras_)
        for (const auto &adapter : adapters)
            n += adapter.down.nbytes() + adapter.up.nbytes();
    return n;
}
void Weights::materialize() {
    std::vector<Tensor> arrays;
    arrays.reserve(values_.size() + runtime_loras_.size() * 2);
    for (auto &[key, value] : values_)
        arrays.push_back(value);
    for (const auto &[_, adapters] : runtime_loras_)
        for (const auto &adapter : adapters) {
            arrays.push_back(adapter.down);
            arrays.push_back(adapter.up);
        }
    mx::eval(arrays);
}
size_t Weights::apply_loras(const std::vector<LoRAAsset> &adapters, const std::string &role,
                            const Event &event, std::atomic<bool> &cancelled,
                            bool inference_time) try {
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
            const auto targets = lora_targets(stem);
            auto resolve_target_key = [&](const std::string &target) {
                auto key = target.ends_with(".weight") ? target : target + ".weight";
                // FLUX owns a real `to_out.0.weight`; Z-Image's Comfy layout
                // instead calls that same logical projection `out.weight`.
                // Preserve the exact key whenever it exists and use the
                // Z-Image spelling only as a fallback.
                if (!values_.count(key) && target.ends_with(".attention.to_out.0"))
                    key = target.substr(0, target.size() - std::strlen("to_out.0")) +
                          "out.weight";
                return key;
            };
            std::map<std::string, int> target_key_counts;
            for (const auto &target : targets)
                ++target_key_counts[resolve_target_key(target)];
            int offset = 0;
            for (const auto &target : targets) {
                auto key = resolve_target_key(target);
                auto found = values_.find(key);
                if (found != values_.end()) {
                    auto base = found->second;
                    auto prefix = key.substr(0, key.size() - std::strlen(".weight"));
                    const bool packed = quantized(prefix) || convrot(prefix);
                    if (packed && !inference_time) {
                        dequantize({prefix});
                        found = values_.find(key);
                        base = found->second;
                    }
                    const int logical_input = convrot(prefix)
                        ? (base.dtype() == mx::uint32 ? base.shape(1) * 4 : base.shape(1))
                        : (quantized(prefix) ? at(prefix + ".scales").shape(1) * 32
                                              : base.shape(1));
                    require(base.ndim() == 2 && logical_input == down.shape(1),
                            "LoRA input does not match " + target);
                    int rows = base.shape(0);
                    require(offset + rows <= up.shape(0),
                            "LoRA output does not match " + target);
                    auto selected = targets.size() == 1
                                        ? up
                                        : slice_axis(up, 0, offset, offset + rows);
                    offset += rows;
                    if (inference_time) {
                        // If multiple logical targets resolve to one fused
                        // base projection, preserve the adapter's row slice.
                        // Distinct q/k/v projections each have their own
                        // output coordinate system and therefore start at 0.
                        const bool shared_projection = target_key_counts[key] > 1;
                        const int output_start = shared_projection ? offset - rows : 0;
                        const int output_end = shared_projection ? offset : rows;
                        runtime_loras_[prefix].push_back(
                            {down, selected, scale, output_start, output_end});
                    } else {
                        auto delta = mx::matmul(mx::astype(selected, mx::float32),
                                                mx::astype(down, mx::float32)) *
                                     Tensor(scale, mx::float32);
                        found->second = mx::astype(mx::astype(base, mx::float32) + delta,
                                                   base.dtype());
                        mx::eval(found->second);
                    }
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
                    auto fused_prefix = std::string(projection[1]) + ".qkv";
                    const bool packed = quantized(fused_prefix) || convrot(fused_prefix);
                    if (packed && !inference_time)
                        dequantize({fused_prefix});
                    auto fused = values_.find(fused_prefix + ".weight");
                    if (fused == values_.end()) continue;
                    auto base = fused->second;
                    const int logical_input = convrot(fused_prefix)
                        ? (base.dtype() == mx::uint32 ? base.shape(1) * 4 : base.shape(1))
                        : (quantized(fused_prefix) ?
                            at(fused_prefix + ".scales").shape(1) * 32 : base.shape(1));
                    require(base.ndim() == 2 && base.shape(0) % 3 == 0 &&
                                logical_input == down.shape(1),
                            "Z-Image fused QKV geometry does not match " + target);
                    int rows = base.shape(0) / 3;
                    require(up.shape(0) == rows,
                            "Z-Image LoRA output does not match " + target);
                    char which = std::string(projection[2])[0];
                    int begin = which == 'q' ? 0 : (which == 'k' ? rows : 2 * rows);
                    if (inference_time) {
                        runtime_loras_[fused_prefix].push_back({down, up, scale, begin,
                                                                 begin + rows});
                        ++adapter_applied;
                        continue;
                    }
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
} catch (...) {
    // LoRA deltas replace weight entries as they are materialized.  If a
    // later pair is malformed, allocation fails, or cancellation arrives,
    // retaining those partially modified entries would let a same-identity
    // retry skip fusion and run a corrupt session.  Drop the component so the
    // owning model must reload the immutable base checkpoint before retrying.
    // This is fail-closed without retaining a second full checkpoint-sized
    // rollback copy in unified memory.
    clear();
    throw;
}
Tensor linear(const Tensor &x, const Weights &w, const std::string &p) {
    return w.project(x, p);
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

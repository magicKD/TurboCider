#include "vae_weights.hpp"

#include <algorithm>

namespace tc::h3_mlx {
namespace {
bool wanted(const std::string &key, const std::vector<std::string> &prefixes) {
    return std::any_of(prefixes.begin(), prefixes.end(),
                       [&](const auto &prefix) { return key.starts_with(prefix); });
}
}

void VAEWeights::load(const std::filesystem::path &root,
                      const std::vector<std::string> &prefixes,
                      mx::Dtype storage_dtype, bool convert_conv3d_layout,
                      const Event &event, std::atomic<bool> &cancelled) {
    require(arrays_.empty(), "H3 MLX VAE weights are already loaded");
    require(std::filesystem::is_directory(root),
            "H3 MLX VAE directory is missing: " + root.string());
    require(!prefixes.empty(), "H3 MLX VAE prefix filter is empty");
    std::vector<std::filesystem::path> shards;
    for (const auto &item : std::filesystem::directory_iterator(root))
        if (!item.is_symlink() && item.is_regular_file() &&
            item.path().extension() == ".safetensors")
            shards.push_back(item.path());
    std::sort(shards.begin(), shards.end());
    require(!shards.empty(), "H3 MLX VAE has no safetensors shards");
    try {
        for (size_t shard = 0; shard < shards.size(); ++shard) {
            checkpoint(cancelled);
            event("h3_mlx_vae_load", static_cast<int>(shard),
                  static_cast<int>(shards.size()));
            auto source = mx::load_safetensors(shards[shard].string()).first;
            for (auto &[key, raw] : source) {
                if (!wanted(key, prefixes)) continue;
                require(!arrays_.count(key), "duplicate H3 MLX VAE tensor: " + key);
                auto value = raw;
                if (value.dtype() == mx::float16 || value.dtype() == mx::bfloat16 ||
                    value.dtype() == mx::float32)
                    value = mx::astype(value, storage_dtype);
                if (convert_conv3d_layout && value.ndim() == 5)
                    value = mx::contiguous(mx::transpose(value, {0, 2, 3, 4, 1}));
                mx::eval(value);
                arrays_.emplace(std::move(key), std::move(value));
            }
            mx::clear_cache();
        }
        checkpoint(cancelled);
        event("h3_mlx_vae_load", static_cast<int>(shards.size()),
              static_cast<int>(shards.size()));
        require(!arrays_.empty(), "H3 MLX VAE prefix filter selected no tensors");
    } catch (...) {
        clear();
        throw;
    }
}

void VAEWeights::clear() {
    arrays_.clear();
    mx::clear_cache();
}

bool VAEWeights::has(const std::string &key) const {
    return arrays_.count(key) != 0;
}

const Tensor &VAEWeights::at(const std::string &key) const {
    auto found = arrays_.find(key);
    require(found != arrays_.end(), "missing H3 MLX VAE tensor: " + key);
    return found->second;
}

size_t VAEWeights::bytes() const {
    size_t total = 0;
    for (const auto &[_, value] : arrays_) total += value.nbytes();
    return total;
}

} // namespace tc::h3_mlx

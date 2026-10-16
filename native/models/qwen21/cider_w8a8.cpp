#include "cider_w8a8.hpp"
#include "../../third_party/cider/w8a8_primitive.h"
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>

namespace tc::qwen21 {
// Adapter for Mininglamp-AI/cider 4d91fcee (MIT).
// Per-channel weights and per-token activations use INT8 TensorOps.
// Unlike optional upstream percentile clipping, packing here is symmetric
// FP32 absmax/127 + round/clip, entirely in MLX, with no calibration.
inline void cider_check(bool ok, const char *message) {
    if (!ok) throw std::invalid_argument(message);
}
inline int cider_round_up(int n, int alignment) {
    cider_check(n > 0 && n <= std::numeric_limits<int>::max() - alignment,
                "Cider dimensions must be positive bounded integers");
    return ((n + alignment - 1) / alignment) * alignment;
}
inline bool cider_float_type(mx::Dtype dtype) {
    return dtype == mx::float16 || dtype == mx::bfloat16 || dtype == mx::float32;
}
inline void cider_check_materialized_bank(const std::vector<Tensor> &bank) {
    for (const auto &a : bank)
        cider_check(a.is_available() && !a.has_primitive() && a.inputs().empty() &&
                    a.siblings().empty() && a.flags().row_contiguous &&
                    a.data_size() * a.itemsize() == a.nbytes(),
                    "Cider bank retained source storage or an unfinished graph");
}
inline std::string cider_source_dtype(mx::Dtype dtype) {
    if (dtype == mx::float16) return "float16";
    if (dtype == mx::bfloat16) return "bfloat16";
    if (dtype == mx::float32) return "float32";
    throw std::invalid_argument("Cider cache requires a floating source dtype");
}
inline std::unordered_map<std::string, std::string> cider_cache_metadata(
        const std::string &key, int n, int k, int group, mx::Dtype dtype, const std::string &sha) {
    return {{"schema", "qwen21-dit-packed-v2"}, {"source_sha256", sha}, {"adapter_recipe", cider_recipe}, {"key", key},
            {"original_shape", std::to_string(n) + "," + std::to_string(k)},
            {"group_size", std::to_string(group)}, {"source_dtype", cider_source_dtype(dtype)},
            {"recipe", "symmetric-fp32-absmax127-mlx-round-clip-v1"},
            {"padding", "zero-nk128"}};
}
inline void cider_validate_cache_key(std::string_view key) {
    if (key.empty() || key.size() > 1024 || key.front() == '/' || key.back() == '/' ||
        key.find("..") != std::string_view::npos)
        throw std::invalid_argument("Cider cache requires a nonempty safe relative key without ..");
    while (!key.empty()) {
        const auto slash = key.find('/');
        const auto component = key.substr(0, slash);
        if (component.empty() || component == "." || component.size() > 255)
            throw std::invalid_argument("Cider cache key contains an invalid path component");
        for (const char c : component)
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.'))
                throw std::invalid_argument("Cider cache key permits only letters, digits, _, -, . and /");
        if (slash == std::string_view::npos) break;
        key.remove_prefix(slash + 1);
    }
}
inline std::filesystem::path cider_cache_path(const std::string &directory,
                                             const std::string &key) {
    cider_validate_cache_key(key);
    std::filesystem::path path(directory);
    for (const auto &component : std::filesystem::path(key)) {
        path /= component;
        cider_check(!std::filesystem::is_symlink(path), "DiT W8A8 cache must not traverse a symlink");
    }
    path /= "packed.safetensors";
    cider_check(!std::filesystem::is_symlink(path), "DiT W8A8 cache must not be a symlink");
    return path;
}
class CacheEntryLock {
    int fd_ = -1;
  public:
    CacheEntryLock(const std::filesystem::path &entry, const std::function<void()> &checkpoint) {
        std::filesystem::create_directories(entry.parent_path());
        auto lock = entry; lock += ".lock";
        fd_ = open(lock.c_str(), O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd_ < 0) throw std::runtime_error("Cannot lock DiT W8A8 cache: " + entry.string());
        try {
            while (flock(fd_, LOCK_EX | LOCK_NB)) {
                if (errno != EWOULDBLOCK && errno != EINTR)
                    throw std::runtime_error("Cannot acquire DiT W8A8 cache lock: " + entry.string());
                checkpoint();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        } catch (...) { close(fd_); fd_ = -1; throw; }
    }
    ~CacheEntryLock() { if (fd_ >= 0) { flock(fd_, LOCK_UN); close(fd_); } }
    CacheEntryLock(const CacheEntryLock &) = delete;
};
CiderCache::CiderCache(const std::filesystem::path &root, const std::string &sha,
                       std::function<void()> checkpoint) : checkpoint_(std::move(checkpoint)) {
    cider_check(sha.size() == 64 && sha.find_first_not_of("0123456789abcdef") == std::string::npos,
                "DiT W8A8 requires the verified checkpoint SHA-256");
    if (!checkpoint_) checkpoint_ = [] {};
    checkpoint_();
    auto directory = root / cider_recipe / sha;
    std::filesystem::path partial;
    for (const auto &component : directory) {
        partial /= component;
        cider_check(!std::filesystem::is_symlink(partial), "DiT W8A8 cache root must not traverse a symlink");
    }
    std::filesystem::create_directories(directory);
    receipt_.directory = directory.string(); receipt_.source_sha256 = sha; receipt_.recipe = cider_recipe;
}
inline std::vector<Tensor> cider_read_packed_cache(
        const std::filesystem::path &path, const std::string &key,
        int n, int k, int group, mx::Dtype source_dtype, const std::string &sha) {
    if (!std::filesystem::is_regular_file(path))
        throw std::invalid_argument("Cider packed cache is missing or not a regular file: " + path.string());
    auto [arrays, metadata] = mx::load_safetensors(path.string());
    for (const auto &[name, expected] : cider_cache_metadata(key, n, k, group, source_dtype, sha)) {
        const auto found = metadata.find(name);
        if (found == metadata.end() || found->second != expected)
            throw std::invalid_argument("Cider packed cache metadata mismatch (" + name + "): " + path.string());
    }
    cider_check(arrays.size() == 3 && arrays.contains("weight") && arrays.contains("scale") &&
                arrays.contains("zero_bias"), "Cider packed cache must contain exactly weight, scale and zero_bias");
    std::vector<Tensor> bank{arrays.at("weight"), arrays.at("scale"), arrays.at("zero_bias")};
    const int pn = cider_round_up(n, 128), pk = cider_round_up(k, 128);
    const mx::Shape scale_shape = group ? mx::Shape{pk / group, pn} : mx::Shape{pn};
    cider_check(bank[0].dtype() == mx::int8 && bank[0].shape() == mx::Shape{pn, pk} &&
                bank[1].dtype() == mx::float32 && bank[1].shape() == scale_shape &&
                bank[2].dtype() == mx::float16 && bank[2].shape() == mx::Shape{n},
                "Cider packed cache shape, scale layout or dtype mismatch");
    // Evaluate only the saved arrays. The original checkpoint tensor is never
    // evaluated or retained by this path; no numerical conversion is applied.
    mx::eval(bank);
    cider_check_materialized_bank(bank);
    const auto *scales = bank[1].data<float>();
    for (size_t i = 0; i < bank[1].size(); ++i)
        cider_check(std::isfinite(scales[i]) && scales[i] > 0.f,
                    "Cider packed cache scales must be finite and positive");
    const auto *bias = bank[2].data<mx::float16_t>();
    for (size_t i = 0; i < bank[2].size(); ++i)
        cider_check(float(bias[i]) == 0.f, "Cider packed cache bias must be zero");
    return bank;
}
inline void cider_write_packed_cache(
        const std::filesystem::path &path, const std::string &key,
        int n, int k, int group, mx::Dtype source_dtype, const std::vector<Tensor> &bank, const std::string &sha) {
    std::filesystem::create_directories(path.parent_path());
    static std::atomic<uint64_t> sequence{0};
    auto temporary = path;
    temporary += ".tmp." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 "." + std::to_string(sequence.fetch_add(1)) + ".safetensors";
    try {
        mx::save_safetensors(temporary.string(),
            {{"weight", bank[0]}, {"scale", bank[1]}, {"zero_bias", bank[2]}},
            cider_cache_metadata(key, n, k, group, source_dtype, sha));
        // Publish a complete file without replacing an existing cache entry.
        // Writers serialize per entry; cancellation never publishes a partial file.
        std::filesystem::create_hard_link(temporary, path);
        std::filesystem::remove(temporary);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

// Three compact arrays, with original N retained by the zero-bias shape.
std::vector<Tensor> CiderCache::pack(const Tensor &weight, const std::string &key) {
    checkpoint_();
    cider_check(weight.ndim() == 2 && cider_float_type(weight.dtype()),
                "Cider packing requires a floating [N,K] weight");
    const int n = weight.shape(0), k = weight.shape(1);
    const int pn = cider_round_up(n, 128), pk = cider_round_up(k, 128);
    constexpr int group = 0;
    auto path = cider_cache_path(receipt_.directory, key);
    CacheEntryLock lock(path, checkpoint_);
    checkpoint_();
    const auto started = std::chrono::steady_clock::now();
    if (std::filesystem::exists(path)) {
        auto bank = cider_read_packed_cache(path, key, n, k, group, weight.dtype(), receipt_.source_sha256);
        ++receipt_.hits; receipt_.bytes_read += std::filesystem::file_size(path);
        receipt_.read_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        checkpoint_();
        return bank;
    }
    const uint64_t needed = uint64_t(pn) * pk + uint64_t(pn) * 4 + uint64_t(n) * 2 + 1024 * 1024;
    cider_check(std::filesystem::space(path.parent_path()).available >= needed,
                "Not enough disk space for DiT W8A8 cache (complete cache needs about 11 GiB)");
    auto fp = mx::astype(weight, mx::float32);
    if (pn != n || pk != k)
        fp = mx::pad(fp, std::vector<std::pair<int,int>>{{0, pn-n}, {0, pk-k}}, Tensor(0.f));
    std::vector<Tensor> result;
    auto scale = mx::max(mx::abs(fp), 1, true) / Tensor(127.f);
    scale = mx::where(scale == Tensor(0.f), Tensor(1.f), scale);
    auto q = mx::astype(mx::clip(mx::round(fp / scale), Tensor(-128.f), Tensor(127.f)), mx::int8);
    result = {mx::contiguous(q), mx::contiguous(mx::reshape(scale, {pn})), mx::zeros({n}, mx::float16)};
    mx::eval(result);
    cider_check_materialized_bank(result);
    checkpoint_();
    cider_write_packed_cache(path, key, n, k, group, weight.dtype(), result, receipt_.source_sha256);
    ++receipt_.created; receipt_.bytes_written += std::filesystem::file_size(path);
    receipt_.prepare_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    checkpoint_();
    return result;
}

Tensor cider_project(const Tensor &input, const std::vector<Tensor> &args, size_t offset) {
    cider_check(offset <= args.size() && args.size() - offset >= 3 && input.ndim() >= 1 &&
                cider_float_type(input.dtype()), "Cider projection arguments are invalid");
    const auto &w = args[offset];
    const auto &scale = args[offset + 1];
    const auto &bias = args[offset + 2];
    cider_check(w.ndim() == 2 && w.dtype() == mx::int8 && scale.dtype() == mx::float32 &&
                bias.ndim() == 1 && bias.dtype() == mx::float16,
                "Cider projection requires INT8 weight, FP32 scales and FP16 zero bias");
    const int pn = w.shape(0), pk = w.shape(1), n = bias.shape(0), k = input.shape(-1);
    cider_check(n > 0 && k > 0 && pn == cider_round_up(n, 128) && pk == cider_round_up(k, 128),
                "Cider bank shape does not match the original projection");
    const size_t rows = input.size() / size_t(k);
    cider_check(rows > 0 && rows <= size_t(std::numeric_limits<int>::max() - 128),
                "Cider projection row count is invalid");
    const int m = int(rows), pm = cider_round_up(m, m <= 64 ? 32 : 128);
    cider_check(scale.ndim() == 1 && scale.shape(0) == pn, "Cider PC scales must be [padded_N]");
    auto x = mx::astype(mx::reshape(input, {m, k}), mx::float16);
    if (pm != m || pk != k)
        x = mx::pad(x, std::vector<std::pair<int,int>>{{0, pm-m}, {0, pk-k}}, Tensor(0, mx::float16));
    x = mx::contiguous(x);
    auto b = pn == n ? bias : mx::pad(bias, std::vector<std::pair<int,int>>{{0, pn-n}}, Tensor(0, mx::float16));
    // M=1 is deliberately padded to 32: every enabled projection uses true
    // INT8 activation GEMM, never upstream's FP-activation MV fallback.
    auto output = cider::perchannel_linear(x, w, scale, b, "builtin:cider-4d91fcee");
    output = mx::slice(output, {0, 0}, {m, n});
    auto shape = input.shape();
    shape.back() = n;
    return mx::astype(mx::reshape(output, shape), input.dtype());
}
} // namespace tc::qwen21

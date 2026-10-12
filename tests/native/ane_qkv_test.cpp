#include "../../native/backends/ane_qkv.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
using namespace tc;
constexpr int rows = 5;
constexpr size_t budget = 512ull << 20;

std::set<fs::path> leases(const fs::path &temporary) {
    std::set<fs::path> result;
    for (const auto &entry : fs::directory_iterator(temporary))
        if (entry.path().filename().string().starts_with("turbocider-runtime-ane-"))
            result.insert(entry.path());
    return result;
}
void no_leases(const fs::path &temporary, const char *phase) {
    require(leases(temporary).empty(), std::string(phase) + ": private lease survived");
}
fs::path live_lease(const fs::path &temporary) {
    const auto live = leases(temporary);
    require(live.size() == 1, "expected exactly one private QKV graph lease");
    require(fs::is_regular_file(*live.begin() / ".runtime-ane-lease-v1"),
            "live graph has no completed lease marker");
    return *live.begin();
}
void same(const Tensor &actual, const Tensor &expected) {
    require(actual.shape() == expected.shape() && actual.dtype() == expected.dtype(),
            "QKV output shape/dtype differs from complete GPU result");
    require(mx::all(mx::isfinite(actual)).item<bool>() &&
                mx::all(actual == expected).item<bool>(),
            "QKV output differs from complete finite GPU result");
}
Tensor gpu(const Tensor &input, float q = 2.f) {
    // Independent GPU reference for the three synthetic diagonal matrices.
    // It also makes the callback row boundaries observable without a model.
    return mx::astype(mx::concatenate({input * q, input * .5f, input * -.25f}, 2), input.dtype());
}
std::vector<Tensor> weights(float q = 2.f) {
    auto diagonal = mx::eye(4096, mx::bfloat16);
    std::vector<Tensor> result{mx::astype(diagonal * q, mx::bfloat16),
                               mx::astype(diagonal * .5f, mx::bfloat16),
                               mx::astype(diagonal * -.25f, mx::bfloat16)};
    mx::eval(result);
    for (const auto &source : result)
        require(source.dtype() == mx::bfloat16, "QKV fixture source must remain BF16");
    return result;
}

void exercise(const fs::path &manifest, const fs::path &temporary) {
    std::atomic<bool> cancelled{false};
    auto sources = weights();
    auto normal = mx::full({1, rows, 4096}, .5f, mx::bfloat16);
    auto overflow = mx::concatenate({mx::full({1, 3, 4096}, .5f, mx::bfloat16),
                                    mx::full({1, 2, 4096}, 60000.f, mx::bfloat16)}, 1);
    mx::eval({normal, overflow});

    {
        ane::HybridQkv runtime(manifest, budget, cancelled);
        require(runtime.available(), "QKV graph did not pass admission/self-test");
        live_lease(temporary);
        auto invalid = weights(100000.f); // BF16 finite; FP16 weight conversion fails.
        runtime.stage(1, rows, invalid);
        std::vector<int> callback_rows;
        auto result = runtime.run(1, normal, [&](int, const Tensor &part) {
            callback_rows.push_back(part.shape(1));
            require(!runtime.retains_resources(), "staging failure retained QKV resources");
            no_leases(temporary, "staging failure before GPU fallback");
            return gpu(part, 100000.f);
        }, cancelled);
        same(result, gpu(normal, 100000.f));
        const auto &m = runtime.metrics();
        require(callback_rows == std::vector<int>{rows} && m.failed && m.failures == 1 &&
                    m.failure_block == 1 && m.calls == 0 && m.fallback_blocks == 1 &&
                    m.failure_reason.find("FP16 staging overflow/nonfinite input") != std::string::npos,
                "staging failure did not retire the graph and recompute the complete input");
        require(!runtime.available() && !runtime.retains_resources(), "failed QKV graph remained owned");
        no_leases(temporary, "staging failure while wrapper alive");
    }
    std::cout << "PASS QKV staging failure: full GPU fallback; resources/lease released while wrapper alive\n";

    {
        ane::HybridQkv runtime(manifest, budget, cancelled);
        require(runtime.available(), "QKV graph did not pass admission/self-test");
        live_lease(temporary);
        runtime.stage(2, rows, sources);
        std::vector<int> callback_rows;
        auto result = runtime.run(2, overflow, [&](int, const Tensor &part) {
            callback_rows.push_back(part.shape(1));
            if (part.shape(1) == 4) {
                require(!runtime.retains_resources(), "prediction failure retained QKV resources");
                no_leases(temporary, "prediction failure before GPU tail fallback");
            }
            return gpu(part);
        }, cancelled);
        same(result, gpu(overflow));
        // First chunk succeeds. The second is finite FP16 input, but q=2
        // overflows the actual Core ML FP16 prediction output. MatMul has no
        // headroom retry, so all four tail rows must be recomputed, not only
        // the failed second chunk. No synthetic production hook is required.
        const auto &m = runtime.metrics();
        require(callback_rows == std::vector<int>{1, 4} && m.failed && m.failures == 1 &&
                    m.calls == 2 && m.fallback_blocks == 1 && m.failure_block == 2 &&
                    m.failure_reason.find("returned NaN/Inf") != std::string::npos,
                "partial prediction failure did not recompute the entire GPU tail");
        require(!runtime.available() && !runtime.retains_resources(), "prediction failure retained graph/scratch");
        no_leases(temporary, "prediction failure while wrapper alive");
        runtime.begin_request();
        runtime.stage(3, rows, sources);
        same(runtime.run(3, normal, [](int, const Tensor &part) { return gpu(part); }, cancelled), gpu(normal));
        require(runtime.metrics().calls == 2, "failed QKV instance retried Core ML");
        no_leases(temporary, "disabled QKV instance while wrapper alive");
    }
    std::cout << "PASS QKV partial prediction failure: complete GPU tail; resources/lease released while wrapper alive\n";

    {
        ane::HybridQkv runtime(manifest, budget, cancelled);
        require(runtime.available(), "QKV graph did not pass admission/self-test");
        const auto lease = live_lease(temporary);
        struct stat initial{};
        require(::lstat(lease.c_str(), &initial) == 0, "cannot inspect live lease");
        runtime.stage(4, rows, sources);
        bool rejected = false;
        try {
            runtime.run(4, normal, [&](int, const Tensor &part) {
                cancelled = true; // After launch, while the worker can still read slots.
                return gpu(part);
            }, cancelled);
        } catch (const Cancelled &) { rejected = true; }
        require(rejected && runtime.available() && runtime.retains_resources(),
                "healthy cancellation retired or poisoned the QKV graph");
        require(live_lease(temporary) == lease, "cancellation replaced/removed the healthy graph lease");
        struct stat after{};
        require(::lstat(lease.c_str(), &after) == 0 &&
                    initial.st_dev == after.st_dev && initial.st_ino == after.st_ino,
                "healthy graph lease inode changed after cancellation");
        cancelled = false;
        runtime.begin_request();
        runtime.stage(5, rows, sources);
        same(runtime.run(5, normal, [](int, const Tensor &part) { return gpu(part); }, cancelled), gpu(normal));
        require(runtime.metrics().calls == 4 && !runtime.metrics().failed,
                "healthy QKV graph did not recover after cancelled two-chunk prediction");
        require(live_lease(temporary) == lease, "recovered QKV graph did not reuse its lease");
    }
    no_leases(temporary, "healthy QKV wrapper destruction");
    std::cout << "PASS QKV cancellation: drained healthy graph reused; lease kept until destruction\n";
}
} // namespace

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3, "usage: ane-qkv-test manifest.json private-tmpdir");
        const auto temporary = fs::canonical(argv[2]);
        struct stat status{};
        tc::require(::lstat(temporary.c_str(), &status) == 0 && S_ISDIR(status.st_mode) &&
                        status.st_uid == ::geteuid() && (status.st_mode & 0777) == 0700,
                    "QKV test requires a fresh private temporary parent");
        tc::require(fs::canonical(fs::temp_directory_path()) == temporary,
                    "TMPDIR must equal the test's isolated temporary parent");
        tc::require(::setenv("TURBOCIDER_RUNTIME_ANE_QKV_CHUNKS", "2", 1) == 0,
                    "cannot set deterministic QKV chunk count");
        no_leases(temporary, "test start");
        exercise(argv[1], temporary);
        std::cout << "PASS runtime QKV wrapper regression (synthetic 2x4096x12288; no checkpoint)\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL QKV wrapper regression: " << error.what() << '\n';
        return 1;
    }
}

#include "models/qwen21/cider_w8a8.hpp"
#include "platform/apple/platform.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace tc;
using namespace tc::qwen21;

template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception &) { rejected = true; }
    assert(rejected);
}
int main() {
    const auto device = device_info();
    if (!device.optimizations().qwen21_layer_streaming) {
        std::cout << "SKIP DiT W8A8 requires M5 Pro / 24 GiB\n"; return 0;
    }
    auto root = std::filesystem::canonical(std::filesystem::temp_directory_path()) /
        ("tc-dit-w8a8-" + std::to_string(getpid()));
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{root};
    const std::string sha(64, 'a');
    CiderCache cache(root, sha, {});
    auto w = mx::reshape(mx::sin(mx::arange(77 * 65, mx::float32) * Tensor(.137f)), {77,65});
    auto bank = cache.pack(w, "dit/test");
    assert(cache.receipt().created == 1 && cache.receipt().hits == 0);
    assert(bank[0].shape() == mx::Shape({128,128}));
    for (auto &value : bank)
        assert(value.is_available() && !value.has_primitive() && value.inputs().empty());
    auto again = cache.pack(w, "dit/test");
    for (size_t i=0; i<bank.size(); ++i) assert(mx::array_equal(bank[i], again[i]).item<bool>());
    assert(cache.receipt().created == 1 && cache.receipt().hits == 1);
    auto compiled = mx::compile([](const std::vector<Tensor> &args) {
        return std::vector<Tensor>{cider_project(args[0], args, 1)};
    });
    for (int m : {1, 37, 129}) {
        auto x = mx::astype(mx::reshape(mx::cos(mx::arange(m * 65, mx::float32) * Tensor(.103f)), {m,65}), mx::float16);
        auto y = cider_project(x, bank, 0);
        auto traced = compiled({x,bank[0],bank[1],bank[2]})[0];
        mx::eval(y,traced);
        assert(mx::array_equal(y,traced).item<bool>());
        // Independent dequantized INT8 reference; M=1 must also quantize A.
        auto xf = mx::astype(x, mx::float32);
        auto sa = mx::max(mx::abs(xf), 1, true) / Tensor(127.f);
        auto aq = mx::clip(mx::round(xf / sa), Tensor(-128.f), Tensor(127.f));
        auto qw = mx::slice(mx::astype(bank[0], mx::float32), {0,0}, {77,65});
        auto sw = mx::slice(bank[1], {0}, {77});
        auto reference = mx::astype(mx::matmul(aq, mx::transpose(qw)) * sa * sw, mx::float16);
        const float error = mx::max(mx::abs(mx::astype(y,mx::float32)-mx::astype(reference,mx::float32))).item<float>();
        assert(error < .004f);
        auto zero = cider_project(mx::zeros({m,65},mx::bfloat16),bank,0);
        assert(zero.dtype() == mx::bfloat16 && mx::max(mx::abs(mx::astype(zero, mx::float32))).item<float>() == 0.f);
    }
    rejects([&] { cache.pack(w, "../escape"); });
    rejects([&] { CiderCache invalid(root, "unverified", {}); });
    CiderCache other(root, std::string(64,'b'), {});
    other.pack(w, "dit/test");
    assert(other.receipt().hits == 0 && other.receipt().created == 1);
    auto path = std::filesystem::path(cache.receipt().directory) / "dit/test/packed.safetensors";
    auto [arrays, metadata] = mx::load_safetensors(path.string());
    for (auto &[name, value] : arrays) mx::eval(value);
    metadata["source_sha256"] = std::string(64,'b');
    mx::save_safetensors(path.string(), arrays, metadata);
    rejects([&] { cache.pack(w, "dit/test"); });
    metadata["source_sha256"] = sha; metadata["adapter_recipe"] = "wrong";
    mx::save_safetensors(path.string(), arrays, metadata);
    rejects([&] { cache.pack(w, "dit/test"); });
    metadata["adapter_recipe"] = cider_recipe;
    arrays.insert_or_assign("scale", mx::zeros({128},mx::float32));
    mx::save_safetensors(path.string(), arrays, metadata);
    rejects([&] { cache.pack(w, "dit/test"); });
    arrays.clear();
    { std::ofstream broken(path, std::ios::trunc); broken << "truncated"; }
    rejects([&] { cache.pack(w, "dit/test"); });
    bool cancel = false;
    CiderCache cancellable(root, sha, [&] { if (cancel) throw std::runtime_error("cancelled"); });
    cancel = true;
    rejects([&] { cancellable.pack(w, "dit/cancelled"); });
    assert(!std::filesystem::exists(std::filesystem::path(cache.receipt().directory) / "dit/cancelled/packed.safetensors"));
    cancel = false; cancellable.pack(w, "dit/cancelled");
    assert(cancellable.receipt().created == 1);
    std::cout << "PASS Qwen21 DiT W8A8: true A8, padding, compiled/eager, cache identity, corruption, cancellation\n";
}

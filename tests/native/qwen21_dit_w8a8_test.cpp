#include "models/qwen21/cider_w8a8.hpp"
#include "platform/apple/platform.hpp"
#include <cassert>
#include <cmath>
#include "models/qwen21/smoothquant_convrot.hpp"
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
Tensor dense_h(const Tensor &x,int g){
 static int h4[4][4]={{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
 std::vector<float> v(g*g);for(int i=0;i<g;++i)for(int j=0;j<g;++j){int a=i,b=j,s=1;for(int stride=1;stride<g;stride*=4){s*=h4[a%4][b%4];a/=4;b/=4;}v[i*g+j]=s/std::sqrt(float(g));}
 auto xf=mx::contiguous(mx::astype(x,mx::float32));mx::eval(xf);std::vector<float> result(xf.size()); const float *data=xf.data<float>(); for(size_t base=0;base<xf.size();base+=g)for(int j=0;j<g;++j){double sum=0;for(int i=0;i<g;++i)sum+=double(data[base+i])*v[i*g+j];result[base+j]=float(sum);} return Tensor(result.data(),x.shape(),mx::float32);
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
    const std::string calibrated_sha = "89f4158d066cc33906a199fca85634f766892dd78f49b6698dabf187ac86c4bc";
    rejects([&] { CiderCache mismatch(root, sha, {}, true); });
    bool combined_cancel = false;
    CiderCache combined(root, calibrated_sha, [&] { if (combined_cancel) throw std::runtime_error("cancelled"); }, true);
    CiderCache unchanged(root, calibrated_sha, {});
    assert(combined.receipt().directory != unchanged.receipt().directory);
    assert(combined.receipt().smoothquant_convrot && !unchanged.receipt().smoothquant_convrot);
    const std::vector<std::pair<int,std::string>> cases{{64,"dit/img_in"},{256,"dit/time_text_embed.timestep_embedder.linear_1"},{4096,"attention/block0/to_k"}};
    for (const auto &[k,key] : cases) {
        const int n = 77, g = k == 64 ? 64 : 256;
        auto weight = mx::astype(mx::reshape(mx::sin(mx::arange(n*k,mx::float32)*Tensor(.137f)), {n,k}),mx::bfloat16);
        combined_cancel = true;
        rejects([&] { combined.pack(weight,key); });
        combined_cancel = false;
        auto packed = combined.pack(weight,key), repeat = combined.pack(weight,key);
        assert(packed[2].shape() == mx::Shape({1,n+k}));
        for (int i=0;i<3;++i) assert(mx::array_equal(packed[i],repeat[i]).item<bool>());
        auto inv = mx::astype(mx::slice(mx::reshape(packed[2],{-1}),{n},{n+k}),mx::float32);
        auto wr = dense_h(mx::astype(weight,mx::float32)/inv,g);
        auto qw = mx::slice(mx::astype(packed[0],mx::float32),{0,0},{n,k});
        auto sw = mx::slice(packed[1],{0},{n});
        assert(mx::max(mx::abs(qw*mx::reshape(sw,{n,1})-wr)/mx::reshape(sw,{n,1})).item<float>() < .51f);
        auto plain = unchanged.pack(weight,key);
        for (int m : {1,37,129}) {
            auto x = mx::astype(mx::reshape(mx::cos(mx::arange(m*k,mx::float32)*Tensor(.103f)),{m,k}),mx::float16);
            auto xf = mx::astype(x,mx::float32)*inv;
            assert(mx::max(mx::abs(sq_convrot_rotate(xf)-dense_h(xf,g))).item<float>() < .00005f);
            xf = mx::astype(mx::astype(sq_convrot_rotate(xf),mx::float16),mx::float32);
            auto sa = mx::max(mx::abs(xf),1,true)/Tensor(127.f);
            // Metal round is half-away-from-zero; MLX round uses ties-to-even.
            // Cider multiplies by the reciprocal scale before rounding.
            auto scaled = xf * (Tensor(1.f) / sa);
            auto aq = mx::clip(mx::sign(scaled) * mx::floor(mx::abs(scaled)+Tensor(.5f)),Tensor(-128.f),Tensor(127.f));
            auto reference = mx::astype(mx::matmul(aq,mx::transpose(qw))*sa*sw,mx::float16);
            auto y = cider_project(x,packed,0);
            auto traced = compiled({x,packed[0],packed[1],packed[2]})[0];
            mx::eval(y,traced);
            assert(mx::array_equal(y,traced).item<bool>());
            const float projection_error = mx::max(mx::abs(mx::astype(y,mx::float32)-mx::astype(reference,mx::float32))).item<float>();
            if (projection_error >= .08f) std::cerr << "SQ error K=" << k << " M=" << m << " error=" << projection_error << " output=" << mx::max(mx::abs(mx::astype(reference,mx::float32))).item<float>() << "\n";
            assert(projection_error < .08f);
            // Same compiled callable switches on/off/on by bank rank, never global state.
            auto off = compiled({x,plain[0],plain[1],plain[2]})[0];
            assert(mx::array_equal(off,cider_project(x,plain,0)).item<bool>());
            assert(mx::array_equal(y,compiled({x,packed[0],packed[1],packed[2]})[0]).item<bool>());
            assert(mx::max(mx::abs(cider_project(mx::zeros({m,k},mx::bfloat16),packed,0))).item<mx::bfloat16_t>() == 0);
        }
        auto filename = std::filesystem::path(combined.receipt().directory)/key/"packed.safetensors";
        auto [saved, meta] = mx::load_safetensors(filename.string());
        for (auto &[_, array] : saved) mx::eval(array);
        saved.insert_or_assign("zero_bias", mx::reshape(mx::concatenate({mx::zeros({n},mx::float16),mx::ones({k},mx::float16)}),{1,n+k}));
        mx::save_safetensors(filename.string(),saved,meta);
        rejects([&] { combined.pack(weight,key); });
    }
    rejects([&] { combined.pack(mx::zeros({77,64},mx::float32), "unknown/projection"); });
    assert(combined.receipt().created == 3 && combined.receipt().hits == 3);
    std::cout << "PASS Qwen21 DiT W8A8: SQ+ConvRot,  true A8, padding, compiled/eager, cache identity, corruption, cancellation\n";
}

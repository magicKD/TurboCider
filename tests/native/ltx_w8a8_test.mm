#include "ltx_gpu.h"
#include "ltx_w8a8.h"
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <stdexcept>
#include <vector>

static char error[2048];
static void check(bool ok) { if (!ok) throw std::runtime_error(error[0] ? error : "check failed"); }
using Buffer = std::unique_ptr<ltx_gpu_buffer, decltype(&ltx_gpu_buffer_free)>;
static Buffer tensor(ltx_gpu *gpu, size_t bytes, const void *data = nullptr) {
    auto *buffer = data ? ltx_gpu_buffer_new_copy(gpu, data, bytes, error, sizeof(error)) :
        ltx_gpu_buffer_new(gpu, bytes, error, sizeof(error));
    check(buffer != nullptr);
    return Buffer(buffer, ltx_gpu_buffer_free);
}
static float f32(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16; float out; std::memcpy(&out, &bits, 4); return out;
}
static uint16_t bf16(float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4);
    if ((bits & 0x7f800000u) == 0x7f800000u) return uint16_t((bits >> 16) | ((bits & 0x7fffffu) ? 64 : 0));
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
static std::vector<uint16_t> values(size_t count, float scale = 1.f) {
    std::vector<uint16_t> data(count);
    for (size_t i = 0; i < count; ++i) data[i] = bf16(scale * std::sin(float(i * 7 + 11) / 19.f));
    return data;
}
static std::vector<int8_t> weights(size_t count) {
    std::vector<int8_t> data(count);
    for (size_t i = 0; i < count; ++i) data[i] = int8_t(int((i * 73 + i / 13) % 256) - 128);
    return data;
}
// Real Gemma gates exceed the safe range of Metal's tanh implementation.
// Exercise the fused product directly so a NaN cannot be hidden by quantization.
static void gated_product_extremes() {
    const float cases[] = {-1e10f, -1024.f, -50.f, -10.f, -3.f, -1.f,
                            0.f, 1.f, 3.f, 10.f, 50.f, 1024.f, 1e10f};
    constexpr uint32_t count = 513;
    std::vector<uint16_t> gate(count), up(count), actual(count);
    for (uint32_t i = 0; i < count; ++i) {
        gate[i] = bf16(cases[i % (sizeof(cases) / sizeof(cases[0]))]);
        up[i] = bf16(i % 2 ? -0.75f : 1.25f);
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    void *context = ltx_w8a8_create((__bridge void *)device, error, sizeof(error));
    check(context != nullptr);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> g = [device newBufferWithBytes:gate.data() length:count * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> u = [device newBufferWithBytes:up.data() length:count * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [device newBufferWithLength:count * 2 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    ltx_w8a8_gated_product(context, (__bridge void *)encoder, (__bridge void *)g,
        (__bridge void *)u, (__bridge void *)out, count);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    check(command.status == MTLCommandBufferStatusCompleted);
    std::memcpy(actual.data(), out.contents, count * 2);
    ltx_w8a8_free(context);
    for (uint32_t i = 0; i < count; ++i) {
        float x = f32(gate[i]);
        float inner = std::clamp((x + ((x * x) * x * 0.044715f)) * 0.7978845608028654f, -10.f, 10.f);
        float activated = f32(bf16((x * 0.5f) * (1.f + std::tanh(inner))));
        float expected = f32(bf16(activated * f32(up[i])));
        check(std::isfinite(f32(actual[i])));
        check(std::abs(f32(actual[i]) - expected) <= std::max(0.008f * std::abs(expected), 1e-5f));
    }
    std::puts("gated GELU/product large finite values and partial grid PASS");
}
// Independent dense H4-kron oracle, not the GPU butterfly implementation.
static std::vector<float> rotate(const std::vector<uint16_t>& x, int m, int k) {
    std::vector<float> out(x.size());
    for (int row = 0; row < m; ++row) for (int block = 0; block < k; block += 256)
        for (int n = 0; n < 256; ++n) {
            double sum = 0;
            for (int j = 0; j < 256; ++j) {
                int sign = 1;
                for (int shift = 0; shift < 8; shift += 2) {
                    int a = (n >> shift) & 3, b = (j >> shift) & 3;
                    if (a + b == 3) sign = -sign;
                }
                sum += sign * double(f32(x[size_t(row) * k + block + j]));
            }
            out[size_t(row) * k + block + n] = f32(bf16(float(sum / 16.)));
        }
    return out;
}
static void linear_oracle(ltx_gpu *gpu, int m, int n, int k, bool bias) {
    auto x = values(size_t(m) * k); auto w = weights(size_t(n) * k);
    for (int i = 0; i < k; ++i) x[i] = 0;
    if (m > 1) for (int i = 0; i < k; ++i) x[k + i] = bf16(f32(x[k + i]) * 100000.f);
    auto b = values(n, 0.1f);
    std::vector<float> scales(n);
    for (int i = 0; i < n; ++i) scales[i] = i % 11 == 0 ? 0.f : (i % 2 ? -1.f : 1.f) / 256.f;
    auto xb = tensor(gpu, x.size() * 2, x.data()), wb = tensor(gpu, w.size(), w.data());
    auto sb = tensor(gpu, scales.size() * 4, scales.data()), bb = tensor(gpu, b.size() * 2, b.data());
    auto out = tensor(gpu, size_t(m) * n * 2);
    check(ltx_gpu_linear_int8_convrot_mps_bf16(gpu, out.get(), xb.get(), wb.get(), sb.get(),
        bias ? bb.get() : nullptr, m, k, n, 256, error, sizeof(error)));
    std::vector<uint16_t> actual(size_t(m) * n);
    check(ltx_gpu_buffer_read(out.get(), actual.data(), actual.size() * 2, error, sizeof(error)));
    auto rotated = rotate(x, m, k);
    double max_error = 0;
    for (int row = 0; row < m; ++row) {
        float peak = 0.f;
        for (int i = 0; i < k; ++i) peak = std::max(peak, std::abs(rotated[size_t(row) * k + i]));
        float a_scale = peak == 0.f ? 1.f : peak / 127.f;
        for (int col = 0; col < n; ++col) {
            int32_t dot = 0;
            for (int i = 0; i < k; ++i) {
                int q = int(std::nearbyint(rotated[size_t(row) * k + i] / a_scale));
                dot += std::clamp(q, -127, 127) * int(w[size_t(col) * k + i]);
            }
            float expected = f32(bf16((float(dot) * a_scale) * scales[col]));
            if (bias) expected = f32(bf16(expected + f32(b[col])));
            double delta = std::abs(double(f32(actual[size_t(row) * n + col])) - expected);
            max_error = std::max(max_error, delta / std::max(std::abs(double(expected)), 1.));
            if (delta > std::max(0.008 * std::abs(double(expected)), 0.004)) {
                std::fprintf(stderr, "oracle mismatch m=%d n=%d k=%d row=%d col=%d expected=%g actual=%g\n",
                    m, n, k, row, col, expected, f32(actual[size_t(row) * n + col]));
                throw std::runtime_error("integer oracle mismatch");
            }
        }
    }
    std::printf("oracle M=%d N=%d K=%d bias=%d max_relative=%g PASS\n", m, n, k, bias, max_error);
}

static void varying_projections(ltx_gpu *gpu) {
    constexpr uint32_t rows = 33, k = 256;
    uint32_t dims[] = {129, 256, 33};
    auto input = values(rows * k);
    auto xb = tensor(gpu, input.size() * 2, input.data());
    std::vector<Buffer> output_storage, weight_storage, scale_storage, bias_storage;
    ltx_gpu_buffer *outputs[3];
    const ltx_gpu_buffer *weights_ptr[3], *scales_ptr[3], *biases_ptr[3];
    for (uint32_t i = 0; i < 3; ++i) {
        auto w = weights(size_t(dims[i]) * k);
        auto b = values(dims[i], 0.1f);
        std::vector<float> scales(dims[i]);
        for (uint32_t j = 0; j < dims[i]; ++j) scales[j] = (j % 2 ? -1.f : 1.f) / 256.f;
        output_storage.push_back(tensor(gpu, size_t(rows) * dims[i] * 2));
        weight_storage.push_back(tensor(gpu, w.size(), w.data()));
        scale_storage.push_back(tensor(gpu, scales.size() * 4, scales.data()));
        bias_storage.push_back(tensor(gpu, b.size() * 2, b.data()));
        outputs[i] = output_storage.back().get(); weights_ptr[i] = weight_storage.back().get();
        scales_ptr[i] = scale_storage.back().get(); biases_ptr[i] = i == 1 ? bias_storage.back().get() : nullptr;
    }
    const bool w8a8 = ltx_gpu_convrot_w8a8_info(gpu, nullptr, nullptr);
    for (uint32_t count : {3u, 2u}) {
        uint64_t p0, a0, p1, a1;
        ltx_gpu_convrot_w8a8_info(gpu, &p0, &a0);
        check(ltx_gpu_linears_int8_convrot_bf16(gpu, outputs, xb.get(), weights_ptr, scales_ptr,
            biases_ptr, dims, count, rows, k, error, sizeof(error)));
        ltx_gpu_convrot_w8a8_info(gpu, &p1, &a1);
        check(p1 - p0 == (w8a8 ? count : 0u) && a1 - a0 == (w8a8 ? 1u : 0u));
        for (uint32_t i = 0; i < count; ++i) {
            auto reference = tensor(gpu, size_t(rows) * dims[i] * 2);
            check(ltx_gpu_linear_int8_convrot_mps_bf16(gpu, reference.get(), xb.get(), weights_ptr[i],
                scales_ptr[i], biases_ptr[i], rows, k, dims[i], 256, error, sizeof(error)));
            std::vector<uint16_t> a(size_t(rows) * dims[i]), b(a.size());
            check(ltx_gpu_buffer_read(outputs[i], a.data(), a.size() * 2, error, sizeof(error)));
            check(ltx_gpu_buffer_read(reference.get(), b.data(), b.size() * 2, error, sizeof(error)));
            check(a == b);
        }
    }
    uint32_t bad_dims[] = {130, 256, 33};
    check(!ltx_gpu_linears_int8_convrot_bf16(gpu, outputs, xb.get(), weights_ptr, scales_ptr,
        biases_ptr, bad_dims, 3, rows, k, error, sizeof(error)));
    std::printf("varying-width GQA shared projections mode=%s count=2/3 byte_exact=PASS padding/bias/bounds=PASS\n", w8a8 ? "w8a8" : "bf16");
}
static void composites(ltx_gpu *gpu) {
    constexpr int m = 7, k = 256, n = 256, h = 512;
    auto x = values(m * k), b = values(n, 0.01f), norm = std::vector<uint16_t>(n, bf16(1.f));
    auto w = weights(n * k), wh = weights(h * k), wd = weights(n * h);
    std::vector<float> scale(h, 1.f / 256.f);
    auto xb = tensor(gpu, x.size()*2, x.data()), wb = tensor(gpu, w.size(), w.data());
    auto sh = tensor(gpu, scale.size()*4, scale.data()), bb = tensor(gpu, b.size()*2, b.data());
    auto nb = tensor(gpu, norm.size()*2, norm.data()), whb = tensor(gpu, wh.size(), wh.data());
    auto wdb = tensor(gpu, wd.size(), wd.data());
    auto q = tensor(gpu, m*n*2), key = tensor(gpu, m*n*2), v = tensor(gpu, m*n*2), out = tensor(gpu, m*n*2);
    uint64_t p0, a0, p1, a1;
    check(ltx_gpu_convrot_w8a8_info(gpu, &p0, &a0));
    check(ltx_gpu_qkv_int8_convrot_mps_bf16(gpu, q.get(), key.get(), v.get(), xb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        nb.get(), nb.get(), m, k, n, 256, 1e-6f, error, sizeof(error)));
    check(ltx_gpu_convrot_w8a8_info(gpu, &p1, &a1));
    check(p1 - p0 == 3 && a1 - a0 == 1);
    p0 = p1; a0 = a1;
    check(ltx_gpu_gated_mlp_int8_convrot_mps_bf16(gpu, out.get(), xb.get(),
        whb.get(), sh.get(), whb.get(), sh.get(), wdb.get(), sh.get(), m, k, h, n, 256, error, sizeof(error)));
    check(ltx_gpu_convrot_w8a8_info(gpu, &p1, &a1));
    check(p1 - p0 == 3 && a1 - a0 == 2);
    check(ltx_gpu_mlp_int8_convrot_mps_bf16(gpu, out.get(), xb.get(),
        whb.get(), sh.get(), nullptr, wdb.get(), sh.get(), bb.get(), m, k, h, n, 256, error, sizeof(error)));
    auto cos = std::vector<uint16_t>(m*n/2, bf16(1.f)), sin = std::vector<uint16_t>(m*n/2, 0);
    auto cb = tensor(gpu, cos.size()*2, cos.data()), sb = tensor(gpu, sin.size()*2, sin.data());
    auto gatew = values(2*k, 0.001f); auto gw = tensor(gpu, gatew.size()*2, gatew.data());
    p0 = p1; a0 = a1;
    check(ltx_gpu_self_attention_int8_mps_bf16(gpu, out.get(), xb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        nb.get(), nb.get(), gw.get(), nullptr, wb.get(), sh.get(), bb.get(), cb.get(), sb.get(),
        m, 2, 128, 256, 1e-6f, error, sizeof(error)));
    check(ltx_gpu_convrot_w8a8_info(gpu, &p1, &a1));
    // Includes the preceding two-projection FFN.
    check(p1 - p0 == 6 && a1 - a0 == 4);
    check(ltx_gpu_cross_attention_int8_mps_bf16(gpu, out.get(), xb.get(), xb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        nb.get(), nb.get(), gw.get(), nullptr, wb.get(), sh.get(), bb.get(), nullptr, nullptr, nullptr, nullptr,
        m, m, k, k, 2, 128, n, 256, 1e-6f, error, sizeof(error)));
    std::vector<uint16_t> result(m*n);
    check(ltx_gpu_buffer_read(out.get(), result.data(), result.size()*2, error, sizeof(error)));
    for (auto value : result) check(std::isfinite(f32(value)));
    std::puts("composites QKV/shared-quantization/gated-MLP/MLP/self-attention/cross-attention PASS");
}

using Captures = std::vector<std::pair<std::string, std::vector<uint16_t>>>;
static Captures capture_operators(ltx_gpu *gpu) {
    constexpr int m = 7, kv = 5, k = 256, h = 512, n = 256;
    auto x = values(m * k), z = values(kv * k, 0.7f);
    auto b = values(h, 0.01f), norm = std::vector<uint16_t>(n, bf16(1.f));
    auto w = weights(n * k), wh = weights(h * k), wd = weights(n * h);
    std::vector<float> scale(h, 1.f/256.f);
    auto xb = tensor(gpu, x.size()*2, x.data()), zb = tensor(gpu, z.size()*2, z.data());
    auto wb = tensor(gpu, w.size(), w.data()), sh = tensor(gpu, scale.size()*4, scale.data());
    auto bb = tensor(gpu, b.size()*2, b.data()), nb = tensor(gpu, norm.size()*2, norm.data());
    auto whb = tensor(gpu, wh.size(), wh.data()), wdb = tensor(gpu, wd.size(), wd.data());
    auto q = tensor(gpu, m*n*2), key = tensor(gpu, m*n*2), v = tensor(gpu, m*n*2), out = tensor(gpu, m*n*2);
    Captures captures;
    auto save = [&](const char* name, ltx_gpu_buffer* buffer) {
        std::vector<uint16_t> result(m*n);
        check(ltx_gpu_buffer_read(buffer, result.data(), result.size()*2, error, sizeof(error)));
        captures.emplace_back(name, std::move(result));
    };
    check(ltx_gpu_qkv_int8_convrot_mps_bf16(gpu, q.get(), key.get(), v.get(), xb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        nb.get(), nb.get(), m, k, n, 256, 1e-6f, error, sizeof(error)));
    save("QKV/query", q.get()); save("QKV/key", key.get()); save("QKV/value", v.get());
    check(ltx_gpu_mlp_int8_convrot_mps_bf16(gpu, out.get(), xb.get(),
        whb.get(), sh.get(), bb.get(), wdb.get(), sh.get(), bb.get(), m, k, h, n, 256, error, sizeof(error)));
    save("MLP", out.get());
    check(ltx_gpu_gated_mlp_int8_convrot_mps_bf16(gpu, out.get(), xb.get(),
        whb.get(), sh.get(), whb.get(), sh.get(), wdb.get(), sh.get(), m, k, h, n, 256, error, sizeof(error)));
    save("gated_MLP", out.get());
    auto cos = std::vector<uint16_t>(m*n/2, bf16(0.8f)), sin = std::vector<uint16_t>(m*n/2, bf16(0.6f));
    auto cb = tensor(gpu, cos.size()*2, cos.data()), sb = tensor(gpu, sin.size()*2, sin.data());
    auto gatew = values(2*k, 0.001f); auto gw = tensor(gpu, gatew.size()*2, gatew.data());
    check(ltx_gpu_self_attention_int8_mps_bf16(gpu, out.get(), xb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        nb.get(), nb.get(), gw.get(), bb.get(), wb.get(), sh.get(), bb.get(), cb.get(), sb.get(),
        m, 2, 128, 256, 1e-6f, error, sizeof(error)));
    save("self_attention_rope", out.get());
    auto mask = std::vector<uint16_t>(m*kv, 0);
    for (int i=0; i<m; ++i) mask[i*kv + i%kv] = bf16(-10000.f);
    auto mb = tensor(gpu, mask.size()*2, mask.data());
    check(ltx_gpu_cross_attention_int8_mps_bf16_masked(gpu, out.get(), xb.get(), zb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        nb.get(), nb.get(), gw.get(), bb.get(), wb.get(), sh.get(), bb.get(), cb.get(), sb.get(), cb.get(), sb.get(),
        mb.get(), m, m, kv, k, k, 2, 128, n, 256, 1e-6f, error, sizeof(error)));
    save("cross_attention_rope_mask", out.get());
    auto kc = tensor(gpu, kv*n*2), vc = tensor(gpu, kv*n*2);
    check(ltx_gpu_cross_attention_kv_int8_mps_bf16(gpu, kc.get(), vc.get(), zb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), nb.get(), kv, k, 2, 128,
        256, 1e-6f, error, sizeof(error)));
    check(ltx_gpu_cross_attention_query_int8_mps_bf16_masked(gpu, out.get(), xb.get(), kc.get(), vc.get(),
        wb.get(), sh.get(), bb.get(), nb.get(), gw.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        mb.get(), m, m, kv, k, 2, 128, n, 256, 1e-6f, error, sizeof(error)));
    save("cross_attention_cached_KV_mask", out.get());
    auto pw = weights(3*n*k); std::vector<float> ps(3*n, 1.f/256.f); auto pb = values(3*n, 0.01f);
    auto pwb = tensor(gpu, pw.size(), pw.data()), psb = tensor(gpu, ps.size()*4, ps.data());
    auto pbb = tensor(gpu, pb.size()*2, pb.data());
    check(ltx_gpu_qkv_packed_int8_convrot_mps_bf16(gpu, q.get(), key.get(), v.get(), xb.get(),
        pwb.get(), psb.get(), pbb.get(), nb.get(), nb.get(), m, k, n, 256, 1e-6f, error, sizeof(error)));
    save("packed_QKV/query", q.get()); save("packed_QKV/key", key.get()); save("packed_QKV/value", v.get());
    return captures;
}
static void compare_operators(const Captures& actual, const Captures& baseline) {
    check(actual.size() == baseline.size());
    for (size_t c=0; c<actual.size(); ++c) {
        double delta=0, norm=0, anorm=0, dot=0;
        for (size_t i=0; i<actual[c].second.size(); ++i) {
            double a = f32(actual[c].second[i]), b = f32(baseline[c].second[i]);
            check(std::isfinite(a) && std::isfinite(b));
            delta += (a-b)*(a-b); norm += b*b; anorm += a*a; dot += a*b;
        }
        double relative = std::sqrt(delta/std::max(norm, 1e-30));
        double cosine = dot/std::sqrt(std::max(anorm*norm, 1e-30));
        std::printf("quality %s relative_l2=%.7f cosine=%.7f\n", actual[c].first.c_str(), relative, cosine);
        check(relative < 0.07 && cosine > 0.998);
    }
}
int main(int argc, char **argv) {
    try {
        check(argc == 2);
        unsetenv("TURBOCIDER_LTX_CONVROT_W8A8");
        unsetenv("TURBOCIDER_LTX_RAW_CONVROT");
        std::unique_ptr<ltx_gpu, decltype(&ltx_gpu_free)> gpu(ltx_gpu_create(argv[1], error, sizeof(error)), ltx_gpu_free);
        check(bool(gpu));
        if (!ltx_gpu_convrot_w8a8_info(gpu.get(), nullptr, nullptr)) {
            std::puts("SKIP native W8A8 unavailable; BF16 context creation PASS");
            return 0;
        }
        gated_product_extremes();
        linear_oracle(gpu.get(), 1, 33, 256, false);
        linear_oracle(gpu.get(), 33, 129, 512, true);
        linear_oracle(gpu.get(), 65, 131, 768, false);
        varying_projections(gpu.get());
        composites(gpu.get());
        auto actual = capture_operators(gpu.get());
        setenv("TURBOCIDER_LTX_CONVROT_W8A8", "0", 1);
        std::unique_ptr<ltx_gpu, decltype(&ltx_gpu_free)> baseline(ltx_gpu_create(argv[1], error, sizeof(error)), ltx_gpu_free);
        check(bool(baseline));
        check(!ltx_gpu_convrot_w8a8_info(baseline.get(), nullptr, nullptr));
        varying_projections(baseline.get());
        compare_operators(actual, capture_operators(baseline.get()));
        std::puts("PASS native LTX ConvRot W8A8");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
